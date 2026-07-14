#ifndef OPENMP_CLUSTER_EXECUTION_H
#define OPENMP_CLUSTER_EXECUTION_H

/*
 * OPENMP ROADMAP
 * ==============
 * Think of output matrix C as a page divided into vertical strips (tiles).
 * This file gives every strip to one core team before the work starts.
 *
 * Step 1 -> Split the strips between the two K1 core clusters.
 * Step 2 -> Start two outer OpenMP threads, one for each cluster.
 * Step 3 -> Each outer thread starts four workers for its four cores.
 * Step 4 -> OpenMP shares each cluster's fixed strips with schedule(static).
 * Step 5 -> Every worker computes its strips using IME or RVV.
 * Step 6 -> Check the core positions and the number of completed strips.
 *
 * Cluster 0: cores 0-3 -> native IME kernel.
 * Cluster 1: cores 4-7 -> explicit RVV kernel.
 *
 * There is no central work queue. No worker asks for one tile at a time.
 */

#if defined(OMP_KIND_INT8_MIXED)
/*
 * ROADMAP BLOCK 1: Describe the K1 hardware teams
 * -------------------------------------------------
 * The outer level has two cluster controllers.
 * Each controller creates four workers, giving eight workers in total.
 */
#define K1_CLUSTER_COUNT 2
#define K1_WORKERS_PER_CLUSTER 4
#define K1_TOTAL_WORKERS 8
#define K1_IME_FIRST_CPU 0
#define K1_RVV_FIRST_CPU 4

/*
 * The default work ratio is 4:1.
 * This means IME receives about four tiles for every one RVV tile.
 * These values create one fixed boundary; they are not a dynamic scheduler.
 */
#ifndef MIXED_IME_TILE_WEIGHT
#define MIXED_IME_TILE_WEIGHT 4
#endif

#ifndef MIXED_RVV_TILE_WEIGHT
#define MIXED_RVV_TILE_WEIGHT 1
#endif

#if MIXED_IME_TILE_WEIGHT <= 0 || MIXED_RVV_TILE_WEIGHT <= 0
#error "Mixed static tile weights must be positive"
#endif

/*
 * ROADMAP BLOCK 2: Calculate the fixed tile split
 * ------------------------------------------------
 * Example: N=1024 and tile_N=32 produce 32 output strips.
 * With the default 4:1 ratio, IME receives 26 and RVV receives 6.
 * This function returns the number belonging to IME. The rest belong to RVV.
 */
static BLASLONG mixed_ime_tile_count(BLASLONG tiles)
{
    /* Add both weights: 4 + 1 = 5 total weight parts. */
    const BLASLONG total_weight =
        (BLASLONG)MIXED_IME_TILE_WEIGHT + (BLASLONG)MIXED_RVV_TILE_WEIGHT;

    /* Give IME its share from every complete group of weight parts. */
    BLASLONG ime_tiles =
        (tiles / total_weight) * (BLASLONG)MIXED_IME_TILE_WEIGHT;

    /* Some tiles may remain after the complete groups. */
    BLASLONG remainder = tiles % total_weight;

    /* Round the remaining IME share to the nearest whole tile. */
    ime_tiles +=
        (remainder * (BLASLONG)MIXED_IME_TILE_WEIGHT + total_weight / 2) /
        total_weight;

    /* If two or more tiles exist, give at least one tile to each cluster. */
    if (tiles > 1 && ime_tiles == 0) {
        ime_tiles = 1;
    }
    if (tiles > 1 && ime_tiles >= tiles) {
        ime_tiles = tiles - 1;
    }
    return ime_tiles;
}

/*
 * ROADMAP BLOCK 3: Put one worker on one exact core
 * -------------------------------------------------
 * CPU_SET selects one core. sched_setaffinity keeps this worker on that core.
 */
static int pin_current_worker(int cpu)
{
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}
#endif

/*
 * ROADMAP BLOCK 4: Save the first worker error
 * -------------------------------------------------
 * Several workers can fail at the same time. The critical section lets only
 * one worker update the shared error information at once.
 */
static void record_parallel_failure(int rc, const char *stage,
                                    int *kernel_return,
                                    const char **failure_stage)
{
#pragma omp critical(openmp_failure)
    {
        if (*kernel_return == 0) {
            *kernel_return = (rc == 0) ? 1 : rc;
            *failure_stage = stage;
        }
    }
}

/*
 * ROADMAP BLOCK 5: Compute one vertical strip of C
 * -------------------------------------------------
 * tile tells us which strip to compute.
 * n0 is its first column, and nb is its real width.
 * The last strip can be smaller than tile_N.
 */
static void compute_column_tile(BLASLONG tile, BLASLONG M, BLASLONG N,
                                BLASLONG K, BLASLONG tile_n,
                                INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                                INPUT_T *A_pack, INPUT_T *B_tile,
                                BLASLONG *worker_tiles, int worker_id,
                                int use_rvv_path, int *kernel_return,
                                const char **failure_stage)
{
    /* Convert the tile number into its first matrix column. */
    BLASLONG n0 = tile * tile_n;

    /* Use tile_N columns, except when the final strip is smaller. */
    BLASLONG nb = min_blaslong(tile_n, N - n0);

    /* Call IME or RVV according to the team that owns this worker. */
    int rc = call_tile_kernel(M, nb, K, n0, N, A, B, C,
                              A_pack, B_tile, use_rvv_path);

    /* Remember that this worker completed one more output strip. */
    worker_tiles[worker_id] += 1;
    if (rc != 0) {
        record_parallel_failure(rc, "TIMED_TILED_EXECUTION",
                                kernel_return, failure_stage);
    }
}

/*
 * ROADMAP BLOCK 6: Start the complete OpenMP computation
 * ------------------------------------------------------
 * This function creates the teams, distributes all strips, and checks that
 * the observed execution matches the requested plan.
 */
static int run_openmp_tile_region(BLASLONG M, BLASLONG N, BLASLONG K,
                                  BLASLONG tile_n, BLASLONG tiles,
                                  INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                                  int *worker_cpu, BLASLONG *worker_tiles,
                                  INPUT_T **worker_a_pack,
                                  INPUT_T **worker_b_tile,
                                  int *actual_threads,
                                  BLASLONG *ime_assigned_tiles,
                                  BLASLONG *rvv_assigned_tiles,
                                  const char **failure_stage)
{
    int kernel_return = 0;

#if defined(OMP_KIND_INT8_MIXED)
    /* Store the number of workers created inside each cluster. */
    int cluster_threads[K1_CLUSTER_COUNT] = {0, 0};

    /* One entry per worker: 1 means core pinning failed. */
    int pin_failed[K1_TOTAL_WORKERS] = {0};

    /* Calculate the fixed IME boundary only once, before OpenMP starts. */
    BLASLONG ime_tiles = mixed_ime_tile_count(tiles);

    /* Return the planned ownership so it can be printed in the result log. */
    *ime_assigned_tiles = ime_tiles;
    *rvv_assigned_tiles = tiles - ime_tiles;

    /* Allow two active OpenMP levels and disable automatic team resizing. */
    omp_set_dynamic(0);
    omp_set_max_active_levels(2);

    /*
     * ROADMAP BLOCK 7: OpenMP level 1 -> two cluster controllers
     * ----------------------------------------------------------
     * Outer thread 0 controls the IME cluster.
     * Outer thread 1 controls the RVV cluster.
     */
#pragma omp parallel num_threads(K1_CLUSTER_COUNT) shared(cluster_threads, pin_failed, kernel_return)
    {
        /* cluster is 0 for IME and 1 for RVV. */
        int cluster = omp_get_thread_num();

        /* IME starts at core 0; RVV starts at core 4. */
        int first_cpu = (cluster == 0) ? K1_IME_FIRST_CPU : K1_RVV_FIRST_CPU;

        /* IME owns [0, ime_tiles); RVV owns [ime_tiles, tiles). */
        BLASLONG begin = (cluster == 0) ? 0 : ime_tiles;
        BLASLONG end = (cluster == 0) ? ime_tiles : tiles;

        /*
         * ROADMAP BLOCK 8: OpenMP level 2 -> four workers per cluster
         * ------------------------------------------------------------
         * Each cluster now creates one worker for each of its four cores.
         */
#pragma omp parallel num_threads(K1_WORKERS_PER_CLUSTER) firstprivate(cluster, first_cpu, begin, end) shared(cluster_threads, pin_failed, kernel_return)
        {
            /* Local worker numbers are 0, 1, 2, and 3 inside each cluster. */
            int local_worker = omp_get_thread_num();

            /* Global IDs are 0-3 for IME and 4-7 for RVV. */
            int worker_id = cluster * K1_WORKERS_PER_CLUSTER + local_worker;

            /* Map the worker ID to its exact K1 core number. */
            int target_cpu = first_cpu + local_worker;

            /* Pin the worker and record the core where it really runs. */
            pin_failed[worker_id] = pin_current_worker(target_cpu) != 0;
            worker_cpu[worker_id] = sched_getcpu();

            /* One worker records the real inner-team size for this cluster. */
#pragma omp single
            {
                cluster_threads[cluster] = omp_get_num_threads();
            }

            /*
             * ROADMAP BLOCK 9: Share the fixed range with schedule(static)
             * -------------------------------------------------------------
             * OpenMP divides this cluster's range once among its four workers.
             * cluster==1 selects RVV; cluster==0 selects native IME.
             */
#pragma omp for schedule(static)
            for (BLASLONG tile = begin; tile < end; ++tile) {
                compute_column_tile(tile, M, N, K, tile_n, A, B, C,
                                    worker_a_pack[worker_id],
                                    worker_b_tile[worker_id], worker_tiles,
                                    worker_id, cluster == 1,
                                    &kernel_return, failure_stage);
            }
        }
    }

    /* Add both inner-team sizes. A correct mixed run gives 4 + 4 = 8. */
    *actual_threads = cluster_threads[0] + cluster_threads[1];

    /*
     * ROADMAP BLOCK 10: Check that execution followed the plan
     * ---------------------------------------------------------
     * First confirm that both clusters really created four workers.
     */
    if (cluster_threads[0] != K1_WORKERS_PER_CLUSTER ||
        cluster_threads[1] != K1_WORKERS_PER_CLUSTER) {
        if (kernel_return == 0) {
            kernel_return = 1;
            *failure_stage = "NESTED_THREAD_COUNT";
        }
    }

    /* Next confirm worker 0 ran on core 0, worker 1 on core 1, and so on. */
    for (int worker = 0; worker < K1_TOTAL_WORKERS; ++worker) {
        if (pin_failed[worker] || worker_cpu[worker] != worker) {
            if (kernel_return == 0) {
                kernel_return = 1;
                *failure_stage = "THREAD_AFFINITY";
            }
            break;
        }
    }

    /* Finally confirm that both teams completed exactly their assigned tiles. */
    {
        BLASLONG ime_done = 0;
        BLASLONG rvv_done = 0;

        for (int worker = 0; worker < K1_WORKERS_PER_CLUSTER; ++worker) {
            ime_done += worker_tiles[worker];
            rvv_done += worker_tiles[K1_WORKERS_PER_CLUSTER + worker];
        }
        if ((ime_done != *ime_assigned_tiles ||
             rvv_done != *rvv_assigned_tiles) && kernel_return == 0) {
            kernel_return = 1;
            *failure_stage = "STATIC_TILE_COUNT";
        }
    }
#else
    /*
     * ROADMAP BLOCK 11: Simpler homogeneous baseline
     * ------------------------------------------------
     * A pure RVV or pure IME baseline needs only one OpenMP team.
     */
    *ime_assigned_tiles = 0;
    *rvv_assigned_tiles = 0;

#pragma omp parallel shared(kernel_return)
    {
        int worker_id = omp_get_thread_num();

        worker_cpu[worker_id] = sched_getcpu();
#pragma omp single
        {
            *actual_threads = omp_get_num_threads();
        }

        /* Share all output strips statically among this one worker team. */
#pragma omp for schedule(static)
        for (BLASLONG tile = 0; tile < tiles; ++tile) {
            compute_column_tile(tile, M, N, K, tile_n, A, B, C,
                                worker_a_pack[worker_id],
                                worker_b_tile[worker_id], worker_tiles,
                                worker_id, 0, &kernel_return, failure_stage);
        }
    }
#endif

    /* Zero means every worker and kernel call completed successfully. */
    return kernel_return;
}

#endif /* OPENMP_CLUSTER_EXECUTION_H */
