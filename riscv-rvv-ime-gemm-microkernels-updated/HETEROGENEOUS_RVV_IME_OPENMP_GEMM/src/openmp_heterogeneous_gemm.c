/*
 * Heterogeneous OpenMP GEMM benchmark.
 *
 * This is the main executable file. It prepares matrices, starts the timed
 * OpenMP tiled execution, validates the result, and prints performance.
 *
 * Clean source map:
 *   openmp_heterogeneous_gemm.c  -> main program, config, helpers, data setup
 *   openmp_tile_scheduler.h      -> OpenMP threads, C-column tiles, core mapping
 *   openmp_kernel_dispatch.h     -> calls the selected RVV or IME micro-kernel
 *   openmp_validation.h          -> serial reference and result comparison
 *
 * Simple flow:
 *   Step 1 -> read M, N, K, and tile_N.
 *   Step 2 -> allocate A, B, C.
 *   Step 3 -> split C into column tiles.
 *   Step 4 -> OpenMP workers compute different C tiles.
 *   Step 5 -> validate C and print time/GFLOPS/GOPS.
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* OpenBLAS-style integer type used by the micro-kernel APIs. */
typedef long BLASLONG;

/*
 * KERNEL_SYMBOL is supplied by the shell script at compile time.
 * Example: -DKERNEL_SYMBOL=dgemm_kernel_8x4_zvl128b_lmul4_unroll1
 */
#ifndef KERNEL_SYMBOL
#error "KERNEL_SYMBOL must be defined by the build script"
#endif

/*
 * Compile-time mode selection.
 * The same benchmark file is compiled many times with different -D flags.
 * Each mode chooses datatype, output type, metric name, and kernel prototype.
 */
#if defined(OMP_KIND_FP32)
typedef float INPUT_T;
typedef float OUTPUT_T;
#define METRIC_NAME "GFLOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_FP_TOLERANCE"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  INPUT_T alpha, INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                  BLASLONG ldc);

#elif defined(OMP_KIND_FP64)
typedef double INPUT_T;
typedef double OUTPUT_T;
#define METRIC_NAME "GFLOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_FP_TOLERANCE"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  INPUT_T alpha, INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                  BLASLONG ldc);

#elif defined(OMP_KIND_INT8_RVV)
typedef int8_t INPUT_T;
typedef int32_t OUTPUT_T;
#define METRIC_NAME "GOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_EXACT_INT32"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  OUTPUT_T alpha, INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                  BLASLONG ldc);

#elif defined(OMP_KIND_INT8_IME) || defined(OMP_KIND_INT8_MIXED)
typedef int8_t INPUT_T;
typedef int32_t OUTPUT_T;
#define METRIC_NAME "GOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_EXACT_INT32"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  OUTPUT_T alpha, const INPUT_T *A, const INPUT_T *B,
                  OUTPUT_T *C, BLASLONG ldc);
#else
#error "Define one OMP_KIND_* macro"
#endif

#if (defined(OMP_KIND_INT8_IME) || defined(OMP_KIND_INT8_MIXED)) && !defined(OMP_IME_INPUT_FULL_MATRIX)
#error "Native IME OpenMP builds require full K-major input"
#endif

/*
 * K1 mixed-core map.
 * Cores 0-3 are treated as IME-capable; cores 4-7 are treated as RVV.
 */
#if defined(OMP_KIND_INT8_MIXED)
#ifndef MIXED_IME_CORE_FIRST
#define MIXED_IME_CORE_FIRST 0
#endif
#ifndef MIXED_IME_CORE_LAST
#define MIXED_IME_CORE_LAST 3
#endif
#ifndef MIXED_IME_TILE_WEIGHT
#define MIXED_IME_TILE_WEIGHT 4
#endif
#ifndef MIXED_RVV_TILE_WEIGHT
#define MIXED_RVV_TILE_WEIGHT 1
#endif

static int mixed_cpu_uses_ime(int cpu)
{
    return cpu >= MIXED_IME_CORE_FIRST && cpu <= MIXED_IME_CORE_LAST;
}

static BLASLONG mixed_tile_chunk_for_cpu(int cpu)
{
    return mixed_cpu_uses_ime(cpu) ? MIXED_IME_TILE_WEIGHT : MIXED_RVV_TILE_WEIGHT;
}
#endif

/* Read current monotonic time in seconds. Used before and after OpenMP work. */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Allocate 64-byte aligned memory for vector/IME-friendly matrix storage. */
static void *aligned_bytes(size_t bytes)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    return ptr;
}

/* Safely multiply sizes; returns 0 if the result would overflow size_t. */
static int checked_size_product(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

/* Small helper used when the last C-column tile is narrower than tile_N. */
static BLASLONG min_blaslong(BLASLONG a, BLASLONG b)
{
    return (a < b) ? a : b;
}

/* Read yes/no runtime flags such as OMP_VALIDATE and OMP_WARMUP. */
static int env_enabled(const char *name, int default_value)
{
    const char *value = getenv(name);
    return (value == NULL) ? default_value : atoi(value) != 0;
}

#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
/* Deterministic floating-point input pattern, repeated around zero. */
static void fill_input(INPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (INPUT_T)((int)(i % 13) - 6) * (INPUT_T)0.1;
    }
}

static void fill_output(OUTPUT_T *x, size_t n)
{
    fill_input(x, n);
}
#else
/* Deterministic INT8 input pattern: -6, -5, ..., 6, then repeat. */
static void fill_input(INPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (INPUT_T)((int)(i % 13) - 6);
    }
}

/* Initial INT32 C values before GEMM applies C = C + A*B. */
static void fill_output(OUTPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (OUTPUT_T)((int)(i % 7) - 3);
    }
}
#endif

#include "openmp_kernel_dispatch.h"
#include "openmp_validation.h"
#include "openmp_tile_scheduler.h"
int main(int argc, char **argv)
{
    /* Step 1: Read the matrix size from the command line.
     * Default case is 1024 x 1024 x 1024 GEMM.
     */
    /* M means how many rows the output C has. */
    BLASLONG M = (argc > 1) ? atol(argv[1]) : 1024;      /* Rows of A and C. */

    /* N means how many columns the output C has. */
    BLASLONG N = (argc > 2) ? atol(argv[2]) : 1024;      /* Columns of B and C. */

    /* K is the dot-product length: each C value sums K multiply-adds. */
    BLASLONG K = (argc > 3) ? atol(argv[3]) : 1024;      /* Shared inner dimension. */

    /* tile_n is the width of one vertical strip of C. */
    BLASLONG tile_n = (argc > 4) ? atol(argv[4]) : 64;   /* Number of C columns per OpenMP tile. */
    BLASLONG tiles;                                      /* Total number of column tiles. */

    /* Step 2: Read optional runtime switches from environment variables.
     * OMP_VALIDATE=1 checks correctness. OMP_WARMUP=1 runs one untimed warmup.
     */
    /* Read OMP_VALIDATE. If the user does not set it, validation stays ON. */
    int validation_enabled = env_enabled("OMP_VALIDATE", 1);

    /* Read OMP_WARMUP. If the user does not set it, one warmup run is used. */
    int warmup_enabled = env_enabled("OMP_WARMUP", 1);

    /* Step 3: Keep status variables for the run.
     * kernel_return reports kernel errors, validation_rc reports reference errors,
     * failure_stage records where a failure happened, if any.
     */
    /* Start with success. If any kernel call fails, this becomes nonzero. */
    int kernel_return = 0;

    /* Return code from the serial reference kernel used for validation. */
    int validation_rc = 0;

    /* Final yes/no flag: 1 means this benchmark run failed. */
    int run_failed;

    /* Filled by OpenMP: how many threads actually entered the parallel region. */
    int actual_threads = 0;

    /* Maximum OpenMP threads allowed by the runtime/environment. */
    int max_threads;

    /* worker_cpu[tid] stores the Linux CPU/core where OpenMP thread tid ran. */
    int *worker_cpu;

    /* worker_tiles[tid] stores how many C-column tiles thread tid computed. */
    BLASLONG *worker_tiles;

    /* worker_b_tile[tid] is a private packed B buffer for IME-style tile calls. */
    INPUT_T **worker_b_tile;

    /* Human-readable place where failure happened; printed in the log. */
    const char *failure_stage = "NONE";

    /* Step 4: Timing and measurement variables. */
    /* Start timestamp of the timed OpenMP section. */
    double t0;

    /* End timestamp of the timed OpenMP section. */
    double t1;

    /* Runtime in seconds: time_sec = t1 - t0. */
    double time_sec;

    /* Final performance number: GFLOPS or GOPS depending on datatype. */
    double metric_value;

    /* Step 5: Validation statistics.
     * mismatch_count is the number of wrong output elements.
     * max_error is the largest absolute output difference.
     */
    double max_error = 0.0;
    size_t mismatch_count = 0;

    /* Step 6: Matrix element counts and byte counts.
     * These are calculated safely before allocation to avoid overflow.
     */
    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;
    size_t worker_b_tile_elements;
    size_t worker_b_tile_bytes;

    /* Step 7: Main matrix pointers.
     * A and B are inputs. C is modified by OpenMP. C_initial stores the original C.
     * C_reference stores the serial reference result when validation is enabled.
     */
    INPUT_T *A;
    INPUT_T *B;
    OUTPUT_T *C;
    OUTPUT_T *C_initial;
    OUTPUT_T *C_reference = NULL;

    /* Step 8: Reject invalid problem sizes early.
     * tile_N must be a multiple of 8 because the micro-kernels are 8-row based
     * and the IME packing path expects clean tile boundaries.
     */
    if (M <= 0 || N <= 0 || K <= 0 || tile_n <= 0 || tile_n % 8 != 0) {
        fprintf(stderr, "Usage: %s M N K [tile_N]\n", argv[0]);
        fprintf(stderr, "tile_N must be a positive multiple of 8.\n");
        return 1;
    }

    /* Step 9: Convert N columns into OpenMP column tiles.
     * Example: N=1024 and tile_N=64 gives 16 tiles.
     */
    tiles = (N + tile_n - 1) / tile_n;

    /* Step 10: Compute memory sizes safely.
     * A has M*K elements, B has N*K elements, and C has M*N elements.
     */
    if (!checked_size_product((size_t)M, (size_t)K, &sizeA) ||
        !checked_size_product((size_t)N, (size_t)K, &sizeB) ||
        !checked_size_product((size_t)M, (size_t)N, &sizeC) ||
        !checked_size_product(sizeA, sizeof(INPUT_T), &bytesA) ||
        !checked_size_product(sizeB, sizeof(INPUT_T), &bytesB) ||
        !checked_size_product(sizeC, sizeof(OUTPUT_T), &bytesC) ||
        !checked_size_product((size_t)tile_n, (size_t)K,
                              &worker_b_tile_elements) ||
        !checked_size_product(worker_b_tile_elements, sizeof(INPUT_T),
                              &worker_b_tile_bytes)) {
        fprintf(stderr, "matrix size exceeds addressable memory\n");
        return 1;
    }

    /* Step 11: Allocate aligned memory.
     * Alignment helps RVV/IME kernels and avoids unnecessary memory penalties.
     */
    /* Allocate A input matrix storage. */
    A = (INPUT_T *)aligned_bytes(bytesA);

    /* Allocate B input matrix storage. */
    B = (INPUT_T *)aligned_bytes(bytesB);

    /* Allocate C output matrix storage for the OpenMP tiled run. */
    C = (OUTPUT_T *)aligned_bytes(bytesC);

    /* Allocate a saved copy of the starting C values. */
    C_initial = (OUTPUT_T *)aligned_bytes(bytesC);

    /* Allocate serial reference C only when validation is enabled. */
    if (validation_enabled) {
        C_reference = (OUTPUT_T *)aligned_bytes(bytesC);
    }

    /* Step 12: Stop if any allocation failed. */
    if (A == NULL || B == NULL || C == NULL || C_initial == NULL ||
        (validation_enabled && C_reference == NULL)) {
        fprintf(stderr, "allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(C_initial);
        free(C_reference);
        return 1;
    }

    /* Step 13: Fill A, B, and starting C with deterministic values.
     * Deterministic data makes repeated runs comparable and debuggable.
     */
    /* Fill matrix A with deterministic test values. */
    fill_input(A, sizeA);

    /* Fill matrix B with deterministic test values. */
    fill_input(B, sizeB);

    /* Fill the original C values before GEMM updates them. */
    fill_output(C_initial, sizeC);

    /* Step 14: Copy the initial C into the working C buffer.
     * GEMM updates C as C = C + alpha * A * B, so C must start from a known value.
     */
    memcpy(C, C_initial, bytesC);

    /* Step 15: Optional warmup.
     * This runs the selected kernel once before timing so cold-start effects are reduced.
     */
    if (warmup_enabled) {
        /* Run one full GEMM before timing; this warms caches/runtime paths. */
        int warmup_rc = call_full_kernel(M, N, K, A, B, C);
        if (warmup_rc != 0) {
            fprintf(stderr, "warmup kernel returned %d\n", warmup_rc);
            kernel_return = warmup_rc;
            failure_stage = "WARMUP";
        }

        /* Restore C after warmup because the timed run must start from C_initial. */
        memcpy(C, C_initial, bytesC);
    }

    /* Step 16: Optional serial reference.
     * The same selected kernel is run once serially over the full matrix.
     * The OpenMP tiled result is compared against this reference later.
     */
    if (validation_enabled && kernel_return == 0) {
        /* Reset reference C to the exact same starting values. */
        memcpy(C_reference, C_initial, bytesC);

        /* Compute serial reference C using the same selected kernel. */
        validation_rc = call_full_kernel(M, N, K, A, B, C_reference);
        if (validation_rc != 0) {
            fprintf(stderr, "serial validation kernel returned %d\n", validation_rc);
            kernel_return = validation_rc;
            failure_stage = "SERIAL_REFERENCE";
        }
    }

    /* Step 17: Allocate per-thread bookkeeping.
     * worker_cpu tells which CPU each thread actually used.
     * worker_tiles tells how much work each thread completed.
     */
    max_threads = omp_get_max_threads();
    /* Allocate CPU-id array: one entry per possible OpenMP thread. */
    worker_cpu = (int *)malloc((size_t)max_threads * sizeof(int));

    /* Allocate tile counters, initialized to zero by calloc. */
    worker_tiles = (BLASLONG *)calloc((size_t)max_threads, sizeof(BLASLONG));

    /* Allocate array of per-worker B-tile pointers, initialized to NULL. */
    worker_b_tile = (INPUT_T **)calloc((size_t)max_threads, sizeof(INPUT_T *));
    if (worker_cpu == NULL || worker_tiles == NULL || worker_b_tile == NULL) {
        fprintf(stderr, "worker metadata allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(C_initial);
        free(C_reference);
        free(worker_cpu);
        free(worker_tiles);
        free(worker_b_tile);
        return 1;
    }

    /* Step 18: Initialize worker metadata and IME private B buffers.
     * IME kernels need B tiles in a compact packed layout, so each worker gets
     * its own buffer to avoid races between OpenMP threads.
     */
    for (int i = 0; i < max_threads; ++i) {
        /* -1 means this worker has not reported a real CPU id yet. */
        worker_cpu[i] = -1;
#if defined(OMP_KIND_INT8_IME)
        worker_b_tile[i] = (INPUT_T *)aligned_bytes(worker_b_tile_bytes);
        if (worker_b_tile[i] == NULL) {
            fprintf(stderr, "IME worker B-tile allocation failed\n");
            for (int j = 0; j < i; ++j) {
                free(worker_b_tile[j]);
            }
            free(A);
            free(B);
            free(C);
            free(C_initial);
            free(C_reference);
            free(worker_cpu);
            free(worker_tiles);
            free(worker_b_tile);
            return 1;
        }
#endif
    }

    /* Step 19: Timed OpenMP tile region.
     * This is the main heterogeneous part.
     * Think of C as a large page and tile_N as the width of one strip.
     * For N=1024 and tile_N=32, the page has 32 strips.
     * In mixed mode, cores 0-3 compute their strips through IME and
     * cores 4-7 compute their strips through RVV fallback.
     * Because each strip is a different set of C columns, two cores do not
     * write the same C values at the same time.
     */
    /* Take timestamp immediately before OpenMP workers start computing C tiles. */
    t0 = now_sec();
    {
        int parallel_return = run_openmp_tile_region(M, N, K, tile_n, tiles,
                                                     worker_b_tile_bytes,
                                                     A, B, C,
                                                     worker_cpu, worker_tiles,
                                                     worker_b_tile,
                                                     &actual_threads,
                                                     &failure_stage);

        /* Keep an earlier warmup/reference error if one already happened. */
        if (kernel_return == 0) {
            kernel_return = parallel_return;
        }
    }
    /* Take timestamp immediately after all OpenMP workers finish. */
    t1 = now_sec();

    /* Step 20: Convert wall-clock timestamps into elapsed seconds. */
    time_sec = t1 - t0;

    /* Step 21: Compare OpenMP output against the serial reference.
     * A mismatch means the tile splitting or kernel dispatch produced a wrong C value.
     */
    if (validation_enabled && validation_rc == 0 && kernel_return == 0) {
        /* Compare every C element from OpenMP result against the serial result. */
        mismatch_count = compare_outputs(C, C_reference, sizeC, &max_error);
        if (mismatch_count != 0) {
            failure_stage = "NUMERICAL_VALIDATION";
        }
    }

    /* Step 22: Convert correctness state into pass/fail. */
    run_failed = kernel_return != 0 || mismatch_count != 0;

    /* Step 23: Compute throughput.
     * Every output value C(i,j) is a dot product.
     * One dot product uses K multiplications and K additions.
     * That is why GEMM is counted as about 2*M*N*K operations.
     * The unit name is selected in config.h:
     * GFLOPS for FP32/FP64 and GOPS for INT8/IME.
     */
    metric_value = (!run_failed && time_sec > 0.0)
        ? (2.0 * (double)M * (double)N * (double)K) / (time_sec * 1e9)
        : 0.0;

    /* Step 24: Print human-readable run information. */
    printf("M=%ld N=%ld K=%ld tile_N=%ld tiles=%ld threads=%d\n",
           M, N, K, tile_n, tiles, actual_threads);
    printf("TIMING_SCOPE=timed_parallel_tile_region\n");
    printf("VALIDATION_METHOD=%s\n", validation_enabled ? VALIDATION_NAME : "DISABLED");
    printf("FAILURE_STAGE=%s\n", failure_stage);

    /* Step 25: Print where each worker ran and how many tiles it completed.
     * Example output item: 0:cpu2:IME:tiles8
     * This means OpenMP thread 0 ran on CPU 2, used IME, and computed 8 tiles.
     */
    printf("WORKER_PLACEMENT=");
    for (int i = 0; i < actual_threads; ++i) {
#if defined(OMP_KIND_INT8_MIXED)
        printf("%s%d:cpu%d:%s:tiles%ld", (i == 0) ? "" : ";",
               i, worker_cpu[i],
               mixed_cpu_uses_ime(worker_cpu[i]) ? "IME" : "RVV",
               worker_tiles[i]);
#else
        printf("%s%d:cpu%d:tiles%ld", (i == 0) ? "" : ";",
               i, worker_cpu[i], worker_tiles[i]);
#endif
    }
    printf("\n");

    /* Step 26: Print the final benchmark values used by the shell scripts. */
    printf("Time: %.9f sec\n", time_sec);
    printf("%s: %.6f\n", METRIC_NAME, metric_value);
    printf("MISMATCH_COUNT=%zu\n", mismatch_count);
    printf("MAX_ERROR=%.17g\n", max_error);
    printf("KERNEL_RETURN=%d\n", kernel_return);

    /* Step 27: Print one compact CSV line so scripts can collect results automatically. */
    printf("CSV_RUN,%.9f,%.6f,%d,%s,%s,%zu,%.17g,%d\n",
           time_sec, metric_value, kernel_return, failure_stage, METRIC_NAME,
           mismatch_count, max_error, actual_threads);

    /* Step 28: Free all memory before leaving. */
    free(A);
    free(B);
    free(C);
    free(C_initial);
    free(C_reference);
    free(worker_cpu);
    free(worker_tiles);
    for (int i = 0; i < max_threads; ++i) {
        free(worker_b_tile[i]);
    }
    free(worker_b_tile);

    /* Step 29: Return 0 only when timing and validation both succeeded. */
    return run_failed ? 1 : 0;
}



