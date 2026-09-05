#ifndef OPENMP_DYNAMIC_EXECUTION_H
#define OPENMP_DYNAMIC_EXECUTION_H

/*
 * DYNAMIC OPENMP ROADMAP
 * ======================
 * This file is the dynamic alternative to openmp_cluster_execution.h.
 * The matrices, micro-kernels, validation, and timing remain unchanged.
 * Only the rule used to give output-column tiles to workers is different.
 *
 * Step 1 -> Start one OpenMP team containing all eight K1 workers.
 * Step 2 -> Pin workers 0-3 to IME cores and workers 4-7 to RVV cores.
 * Step 3 -> OpenMP keeps the unassigned output tiles in its runtime queue.
 * Step 4 -> schedule(dynamic, chunk) gives another chunk to a free worker.
 * Step 5 -> Each worker calls IME or RVV according to its pinned core group.
 * Step 6 -> Check the team size, core placement, and completed tile count.
 *
 * This path intentionally has runtime scheduling overhead. It exists so the
 * experiments can compare dynamic load balancing with the simpler static path.
 */

#if defined(OMP_KIND_INT8_MIXED)

/*
 * ROADMAP BLOCK 1: Run the complete dynamically scheduled tile region
 * --------------------------------------------------------------------
 * OpenMP owns the tile queue. There is no project-specific central scheduler.
 */
static int run_openmp_dynamic_tile_region(
    BLASLONG M, BLASLONG N, BLASLONG K,
    BLASLONG tile_n, BLASLONG tiles, BLASLONG dynamic_chunk,
    INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
    int *worker_cpu, BLASLONG *worker_tiles,
    INPUT_T **worker_a_pack, INPUT_T **worker_b_tile,
    int *actual_threads,
    BLASLONG *ime_completed_tiles,
    BLASLONG *rvv_completed_tiles,
    const char **failure_stage)
{
    int kernel_return = 0;
    int pin_failed[K1_TOTAL_WORKERS] = {0};

    /* A dynamic chunk must contain at least one output-column tile. */
    if (dynamic_chunk <= 0) {
        *failure_stage = "DYNAMIC_CHUNK";
        return 1;
    }

    /* Keep exactly eight workers; this does not disable schedule(dynamic). */
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);

    /*
     * ROADMAP BLOCK 2: Create one worker for every K1 core
     * ----------------------------------------------------
     * Worker IDs also identify the execution path:
     *   0-3 -> native IME
     *   4-7 -> explicit RVV
     */
#pragma omp parallel num_threads(K1_TOTAL_WORKERS) shared(pin_failed, kernel_return)
    {
        int worker_id = omp_get_thread_num();
        int use_rvv_path = worker_id >= K1_WORKERS_PER_CLUSTER;
        int target_cpu = use_rvv_path
            ? K1_RVV_FIRST_CPU + worker_id - K1_WORKERS_PER_CLUSTER
            : K1_IME_FIRST_CPU + worker_id;

        /* Pin first, then record the CPU that actually executes this worker. */
        pin_failed[worker_id] = pin_current_worker(target_cpu) != 0;
        worker_cpu[worker_id] = sched_getcpu();

        /* One worker records the real OpenMP team size. */
#pragma omp single
        {
            *actual_threads = omp_get_num_threads();
        }

        /*
         * ROADMAP BLOCK 3: Let free workers request the next tile chunk
         * ----------------------------------------------------------------
         * Each tile owns different columns of C, so workers never write the
         * same output values. Faster workers naturally complete more tiles.
         */
#pragma omp for schedule(dynamic, dynamic_chunk)
        for (BLASLONG tile = 0; tile < tiles; ++tile) {
            compute_column_tile(tile, M, N, K, tile_n, A, B, C,
                                worker_a_pack[worker_id],
                                worker_b_tile[worker_id], worker_tiles,
                                worker_id, use_rvv_path,
                                &kernel_return, failure_stage);
        }

        /* Verify that no worker moved while its IME/RVV kernels executed. */
        if (sched_getcpu() != target_cpu) {
            pin_failed[worker_id] = 1;
        }
    }

    /*
     * ROADMAP BLOCK 4: Verify that runtime execution matches the plan
     * ----------------------------------------------------------------
     */
    if (*actual_threads != K1_TOTAL_WORKERS && kernel_return == 0) {
        kernel_return = 1;
        *failure_stage = "DYNAMIC_THREAD_COUNT";
    }

    /* Every worker must remain on its requested K1 core. */
    for (int worker = 0; worker < K1_TOTAL_WORKERS; ++worker) {
        if (pin_failed[worker] || worker_cpu[worker] != worker) {
            if (kernel_return == 0) {
                kernel_return = 1;
                *failure_stage = "THREAD_AFFINITY";
            }
            break;
        }
    }

    /* Sum the work completed by each core group. */
    *ime_completed_tiles = 0;
    *rvv_completed_tiles = 0;
    for (int worker = 0; worker < K1_WORKERS_PER_CLUSTER; ++worker) {
        *ime_completed_tiles += worker_tiles[worker];
        *rvv_completed_tiles +=
            worker_tiles[K1_WORKERS_PER_CLUSTER + worker];
    }

    /* Dynamic ownership can vary, but every tile must execute exactly once. */
    if (*ime_completed_tiles + *rvv_completed_tiles != tiles &&
        kernel_return == 0) {
        kernel_return = 1;
        *failure_stage = "DYNAMIC_TILE_COUNT";
    }

    return kernel_return;
}

#endif /* OMP_KIND_INT8_MIXED */

#endif /* OPENMP_DYNAMIC_EXECUTION_H */
