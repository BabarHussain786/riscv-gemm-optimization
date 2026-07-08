#ifndef OPENMP_TILED_GEMM_PARALLEL_H
#define OPENMP_TILED_GEMM_PARALLEL_H

/*
 * Human-readable purpose
 * ----------------------
 * This file is the heart of the OpenMP tiled execution.
 * Think of the full output matrix C as a long notebook page.
 * This code cuts that page into vertical strips, called C-column tiles.
 * OpenMP workers then take different strips and compute them in parallel.
 *
 * Classroom picture:
 *   One teacher has many pages to fill.
 *   Instead of one student filling all pages, the teacher gives one page strip
 *   to each student. Here, each student is one CPU core/thread.
 *   The important rule is simple: two students must not write on the same strip.
 *   This code follows that rule by assigning different C-column tiles.
 *
 * Normal modes:
 *   all selected workers share tiles with OpenMP static scheduling.
 *
 * Mixed K1 mode:
 *   cores 0-3 are treated as IME workers.
 *   cores 4-7 are treated as RVV workers.
 *   IME workers claim more tiles at a time because IME is faster for INT8.
 */

#include "openmp_tiled_gemm_config.h"   /* datatype, kernel kind, and K1 mixed-core definitions */
#include "openmp_tiled_gemm_dispatch.h" /* call_tile_kernel: calls RVV or IME kernel on one tile */
#include "openmp_tiled_gemm_utils.h"    /* aligned allocation and min helper */

/*
 * run_openmp_tile_region
 * ----------------------
 * This function runs only the timed OpenMP region.
 * Inputs A and B are already allocated and filled.
 * Output C already contains the initial C values.
 * The function splits C into column tiles and asks workers to compute them.
 */
static int run_openmp_tile_region(BLASLONG M, BLASLONG N, BLASLONG K,
                                  BLASLONG tile_n, BLASLONG tiles,
                                  size_t worker_b_tile_bytes,
                                  INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                                  int *worker_cpu, BLASLONG *worker_tiles,
                                  INPUT_T **worker_b_tile,
                                  int *actual_threads,
                                  const char **failure_stage)
{
    /* kernel_return stays 0 when all worker kernel calls succeed. */
    /* Shared error code for all workers. It remains 0 when every tile succeeds. */
    int kernel_return = 0;

    /* next_tile is the shared counter used only by mixed dynamic scheduling.
     * Simple meaning: it points to the next C strip that no worker has taken yet.
     */
    BLASLONG next_tile = 0;

    /* OpenMP starts the worker team here. */
/* Start OpenMP. From this point, several threads execute the same block. */
#pragma omp parallel
    {
        /* Each worker has an OpenMP thread id, starting from 0.
         * This is the worker number inside the OpenMP team, not always the CPU id.
         */
        int tid = omp_get_thread_num();

        /* sched_getcpu tells the real Linux CPU/core used by this worker.
         * This matters because K1 has different core domains.
         */
        int cpu = sched_getcpu();

        /* Store the CPU id so the final log can show worker placement. */
        worker_cpu[tid] = cpu;

        /* One thread records how many OpenMP threads were actually launched. */
/* Only one worker writes the total number of launched OpenMP threads. */
#pragma omp single
        *actual_threads = omp_get_num_threads();

#if defined(OMP_KIND_INT8_MIXED)
        /*
         * Mixed mode: cores 0-3 use IME, cores 4-7 use RVV fallback.
         * Each worker gets its own temporary B tile buffer.
         * The allocation happens inside the worker so memory is first-touched
         * by the worker that will use it.
         */
        /* Allocate this worker private B-tile buffer inside the worker thread. */
        worker_b_tile[tid] = (INPUT_T *)aligned_bytes(worker_b_tile_bytes);

        /* If a worker cannot allocate its B tile buffer, mark the run failed. */
        if (worker_b_tile[tid] == NULL) {
#pragma omp critical
            {
                if (kernel_return == 0) {
                    kernel_return = 1;
                    *failure_stage = "WORKER_BUFFER_ALLOCATION";
                }
            }
        } else {
            /* Clear the local B tile buffer before it is reused by this worker. */
            memset(worker_b_tile[tid], 0, worker_b_tile_bytes);

            /*
             * Mixed scheduling loop.
             * Workers repeatedly claim a chunk of C-column tiles.
             * IME workers claim MIXED_IME_TILE_WEIGHT tiles.
             * RVV workers claim MIXED_RVV_TILE_WEIGHT tiles.
             * Example with defaults: IME claims 4 tiles, RVV claims 1 tile.
             * This does not change the math. It only changes how work is shared.
             */
            while (1) {
                /* start_tile is the first C strip claimed by this worker. */
                BLASLONG start_tile;

                /* end_tile is one past the last C strip claimed by this worker. */
                BLASLONG end_tile;

                /* Decide how many tiles this worker should claim this time. */
                /* chunk is how many strips this worker takes in this round. */
                BLASLONG chunk = mixed_tile_chunk_for_cpu(cpu);

                /* Atomically claim tile numbers so two workers never get the same tile. */
#pragma omp atomic capture
                {
                    /* Give this worker the current next free tile. */
                    start_tile = next_tile;

                    /* Move the shared pointer forward so the next worker gets new tiles. */
                    next_tile += chunk;
                }

                /* If all C-column tiles were already claimed, this worker is done. */
                if (start_tile >= tiles) {
                    break;
                }

                /* Clamp the final chunk so it does not go past the last tile. */
                end_tile = min_blaslong(start_tile + chunk, tiles);

                /* Compute each tile inside the claimed chunk. */
                for (BLASLONG tile = start_tile; tile < end_tile; ++tile) {
                    /* n0 is the first output column of this tile.
                     * If tile_N=32, tile 0 starts at column 0,
                     * tile 1 starts at column 32, tile 2 starts at column 64.
                     */
                    BLASLONG n0 = tile * tile_n;

                    /* nb is the real tile width; the final tile may be smaller. */
                    /* nb is normally tile_N, but the last strip may be narrower. */
                    /* nb is the width of this C strip. Last strip may be smaller. */
            BLASLONG nb = min_blaslong(tile_n, N - n0);

                    /* Call the selected micro-kernel on this C-column tile.
                     * The kernel sees only this strip of C, but the math is still GEMM.
                     */
                    int rc = call_tile_kernel(M, nb, K, n0, N, A, B, C,
                                              worker_b_tile[tid]);

                    /* Count this tile for worker-placement reporting. */
                    worker_tiles[tid] += 1;

                    /* If the kernel reports an error, save the first failure. */
                    if (rc != 0) {
#pragma omp critical
                        {
                            if (kernel_return == 0) {
                                kernel_return = rc;
                                *failure_stage = "TIMED_TILED_EXECUTION";
                            }
                        }
                    }
                }
            }
        }
#else
        /*
         * Homogeneous mode.
         * All selected workers run the same type of kernel.
         * OpenMP schedule(static) divides C-column tiles evenly and predictably.
         */
#pragma omp for schedule(static)
        for (BLASLONG tile = 0; tile < tiles; ++tile) {
            /* In static mode, OpenMP decides which worker gets this tile. */
            /* n0 is the first output column handled by this tile. */
            BLASLONG n0 = tile * tile_n;

            /* nb is the number of columns in this tile. */
            /* nb is the width of this C strip. Last strip may be smaller. */
            BLASLONG nb = min_blaslong(tile_n, N - n0);

            /* Call RVV or IME backend for this tile, depending on compile mode.
             * Homogeneous mode means all workers use the same backend.
             */
            int rc = call_tile_kernel(M, nb, K, n0, N, A, B, C,
                                      worker_b_tile[tid]);

            /* Record how many tiles this worker computed. */
            worker_tiles[tid] += 1;

            /* Save first execution failure, if any. */
            if (rc != 0) {
#pragma omp critical
                {
                    if (kernel_return == 0) {
                        kernel_return = rc;
                        *failure_stage = "TIMED_TILED_EXECUTION";
                    }
                }
            }
        }
#endif
    }

    /* Return 0 for success, nonzero if any worker/kernel failed. */
    return kernel_return;
}

#endif /* OPENMP_TILED_GEMM_PARALLEL_H */



