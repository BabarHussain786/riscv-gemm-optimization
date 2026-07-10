#ifndef OPENMP_TILE_SCHEDULER_H
#define OPENMP_TILE_SCHEDULER_H

/*
 * OpenMP tiled execution roadmap
 * -----------------------------
 * This is the main file for understanding the heterogeneous OpenMP idea.
 * It does not implement the RVV or IME arithmetic itself. The arithmetic lives
 * in the micro-kernel folders. This file only decides which part of output C
 * each OpenMP worker computes.
 *
 * Simple picture:
 *   Full GEMM: C = C + A x B
 *   C has M rows and N columns.
 *   This code cuts C by columns into vertical strips.
 *   Each strip is one OpenMP tile.
 *
 * Example:
 *   M = 1024, N = 1024, K = 1024, tile_N = 32
 *   Number of C-column tiles = 1024 / 32 = 32
 *   Tile 0 writes C columns 0..31
 *   Tile 1 writes C columns 32..63
 *   Tile 2 writes C columns 64..95
 *   and so on.
 *
 * Why column tiles are safe:
 *   Every tile writes a different group of C columns.
 *   Therefore two OpenMP workers do not write the same C values.
 *
 * Mixed K1 mode:
 *   cores 0-3 are treated as IME-capable workers.
 *   cores 4-7 are treated as RVV workers.
 *   IME workers can claim more tiles per turn using MIXED_IME_TILE_WEIGHT.
 *   RVV workers claim fewer tiles per turn using MIXED_RVV_TILE_WEIGHT.
 *   This changes only work distribution, not the GEMM equation.
 */

/*
 * run_openmp_tile_region
 * ----------------------
 * This function is the timed OpenMP region.
 *
 * Inputs:
 *   A and B already contain matrix data.
 *   C already contains the starting output values.
 *   tile_n tells how many C columns are placed in one OpenMP tile.
 *
 * Output:
 *   C is updated tile by tile.
 *   worker_cpu tells which real Linux CPU each OpenMP thread used.
 *   worker_tiles tells how many C tiles each thread completed.
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
    /* If every worker succeeds, this value stays zero. */
    int kernel_return = 0;

    /* Mixed mode uses this shared tile number to hand out new C strips. */
    BLASLONG next_tile = 0;

    /*
     * Step 1: start OpenMP.
     * From here, several threads run the same block of code at the same time.
     */
#pragma omp parallel
    {
        /* tid is the OpenMP worker number: 0, 1, 2, ... */
        int tid = omp_get_thread_num();

        /* cpu is the real Linux core where this worker is currently running. */
        int cpu = sched_getcpu();

        /* Save the real CPU id so the benchmark output can show placement. */
        worker_cpu[tid] = cpu;

        /* One worker records the real OpenMP thread count. */
#pragma omp single
        {
            *actual_threads = omp_get_num_threads();
        }

#if defined(OMP_KIND_INT8_MIXED)
        /*
         * Step 2A: mixed IME/RVV mode.
         * Each worker needs a private B-tile buffer because the IME path copies
         * the selected B columns into a compact layout before calling the kernel.
         * Private buffers avoid races between threads.
         */
        worker_b_tile[tid] = (INPUT_T *)aligned_bytes(worker_b_tile_bytes);

        if (worker_b_tile[tid] == NULL) {
            /* Only one worker should record the first failure reason. */
#pragma omp critical
            {
                if (kernel_return == 0) {
                    kernel_return = 1;
                    *failure_stage = "WORKER_BUFFER_ALLOCATION";
                }
            }
        } else {
            /* Start from a clean private B buffer for this worker. */
            memset(worker_b_tile[tid], 0, worker_b_tile_bytes);

            /*
             * Step 3A: dynamic weighted scheduling for K1 mixed mode.
             * Each worker repeatedly takes the next available C-column tile(s).
             * IME cores can take a larger chunk than RVV cores.
             */
            while (1) {
                BLASLONG start_tile;
                BLASLONG end_tile;

                /* Decide chunk size from the real CPU domain. */
                BLASLONG chunk = mixed_tile_chunk_for_cpu(cpu);

                /*
                 * Atomically claim tile numbers.
                 * This is the key safety line: each worker gets a unique tile range.
                 */
#pragma omp atomic capture
                {
                    start_tile = next_tile;
                    next_tile += chunk;
                }

                /* No tiles left: this worker is finished. */
                if (start_tile >= tiles) {
                    break;
                }

                /* The final claimed range may go past the last tile; clamp it. */
                end_tile = min_blaslong(start_tile + chunk, tiles);

                /*
                 * Step 4A: compute the claimed C-column tiles.
                 * Each tile writes C columns [n0, n0 + nb).
                 */
                for (BLASLONG tile = start_tile; tile < end_tile; ++tile) {
                    /* First column of this C tile. */
                    BLASLONG n0 = tile * tile_n;

                    /* Width of this tile; the final tile can be smaller. */
                    BLASLONG nb = min_blaslong(tile_n, N - n0);

                    /* Call RVV or IME backend for this one C-column strip. */
                    int rc = call_tile_kernel(M, nb, K, n0, N, A, B, C,
                                              worker_b_tile[tid]);

                    /* Count completed tiles for the worker-placement report. */
                    worker_tiles[tid] += 1;

                    if (rc != 0) {
                        /* Keep only the first kernel failure. */
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
         * Step 2B: homogeneous mode.
         * All workers run the same backend type, so OpenMP static scheduling is enough.
         */
#pragma omp for schedule(static)
        for (BLASLONG tile = 0; tile < tiles; ++tile) {
            /* First column of this C tile. */
            BLASLONG n0 = tile * tile_n;

            /* Width of this tile; the final tile can be smaller. */
            BLASLONG nb = min_blaslong(tile_n, N - n0);

            /* Compute this C-column strip using the selected micro-kernel. */
            int rc = call_tile_kernel(M, nb, K, n0, N, A, B, C,
                                      worker_b_tile[tid]);

            /* Count completed tiles for the final log. */
            worker_tiles[tid] += 1;

            if (rc != 0) {
                /* Keep only the first kernel failure. */
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

    /* Zero means all OpenMP tile calls finished successfully. */
    return kernel_return;
}

#endif /* OPENMP_TILE_SCHEDULER_H */

