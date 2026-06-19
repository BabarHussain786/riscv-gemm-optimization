/*
 * Shared OpenMP tiled GEMM benchmark.
 *
 * FP64, FP32, and INT8 RVV modes call one RVV micro-kernel family per run.
 * INT8 IME mode calls the native IME wrapper. The mixed K1 mode uses the
 * same IME wrapper plus its local RVV fallback: IME cores execute the native
 * path and RVV-only cores execute the fallback path on their assigned C tiles.
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

typedef long BLASLONG;

#ifndef KERNEL_SYMBOL
#error "KERNEL_SYMBOL must be defined by the build script"
#endif

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

/* K1 mixed scheduling defaults: cores 0-3 are IME-capable, cores 4-7 are RVV-only. */
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

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *aligned_bytes(size_t bytes)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    return ptr;
}

static int checked_size_product(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static BLASLONG min_blaslong(BLASLONG a, BLASLONG b)
{
    return (a < b) ? a : b;
}

static int env_enabled(const char *name, int default_value)
{
    const char *value = getenv(name);
    return (value == NULL) ? default_value : atoi(value) != 0;
}

#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
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
static void fill_input(INPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (INPUT_T)((int)(i % 13) - 6);
    }
}

static void fill_output(OUTPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (OUTPUT_T)((int)(i % 7) - 3);
    }
}
#endif

static int call_full_kernel(BLASLONG M, BLASLONG N, BLASLONG K,
                            INPUT_T *A, INPUT_T *B, OUTPUT_T *C)
{
#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
    return KERNEL_SYMBOL(M, N, K, (INPUT_T)1, A, B, C, M);
#else
    return KERNEL_SYMBOL(M, N, K, 1, A, B, C, M);
#endif
}

/* Dispatch one C-column work tile to the selected datatype-specific backend. */
static int call_tile_kernel(BLASLONG M, BLASLONG N, BLASLONG K,
                            BLASLONG n0, BLASLONG full_n,
                            INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                            INPUT_T *B_tile)
{
#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
    (void)full_n;
    (void)B_tile;
    return KERNEL_SYMBOL(M, N, K, (INPUT_T)1,
                         A, B + n0 * K, C + n0 * M, M);
#elif defined(OMP_KIND_INT8_RVV)
    (void)full_n;
    (void)B_tile;
    return KERNEL_SYMBOL(M, N, K, 1,
                         A, B + n0 * K, C + n0 * M, M);
#elif defined(OMP_KIND_INT8_IME) || defined(OMP_KIND_INT8_MIXED)
    if (B_tile == NULL) {
        return 1;
    }

    for (BLASLONG k = 0; k < K; ++k) {
        for (BLASLONG column = 0; column < N; ++column) {
            B_tile[(size_t)k * (size_t)N + (size_t)column] =
                B[(size_t)k * (size_t)full_n + (size_t)(n0 + column)];
        }
    }

    return KERNEL_SYMBOL(M, N, K, 1, A, B_tile, C + n0 * M, M);
#else
#error "Unsupported OpenMP kernel kind"
#endif
}

static size_t compare_outputs(const OUTPUT_T *actual, const OUTPUT_T *reference,
                              size_t n, double *max_error)
{
    size_t mismatches = 0;
    double largest = 0.0;

    for (size_t i = 0; i < n; ++i) {
#if defined(OMP_KIND_FP32)
        double actual_value = (double)actual[i];
        double reference_value = (double)reference[i];
        double error;
        double tolerance;
        if (!isfinite(actual_value) || !isfinite(reference_value)) {
            ++mismatches;
            largest = INFINITY;
            continue;
        }
        error = fabs(actual_value - reference_value);
        tolerance = 1e-5 + 1e-5 * fabs(reference_value);
#elif defined(OMP_KIND_FP64)
        double actual_value = (double)actual[i];
        double reference_value = (double)reference[i];
        double error;
        double tolerance;
        if (!isfinite(actual_value) || !isfinite(reference_value)) {
            ++mismatches;
            largest = INFINITY;
            continue;
        }
        error = fabs(actual_value - reference_value);
        tolerance = 1e-12 + 1e-12 * fabs(reference_value);
#else
        double error = fabs((double)((int64_t)actual[i] - (int64_t)reference[i]));
        double tolerance = 0.0;
#endif
        if (error > largest) {
            largest = error;
        }
        if (error > tolerance) {
            ++mismatches;
        }
    }

    *max_error = largest;
    return mismatches;
}

int main(int argc, char **argv)
{
    BLASLONG M = (argc > 1) ? atol(argv[1]) : 1024;
    BLASLONG N = (argc > 2) ? atol(argv[2]) : 1024;
    BLASLONG K = (argc > 3) ? atol(argv[3]) : 1024;
    BLASLONG tile_n = (argc > 4) ? atol(argv[4]) : 64;
    BLASLONG tiles;
    BLASLONG next_tile = 0;
    int validation_enabled = env_enabled("OMP_VALIDATE", 1);
    int warmup_enabled = env_enabled("OMP_WARMUP", 1);
    int kernel_return = 0;
    int validation_rc = 0;
    int run_failed;
    int actual_threads = 0;
    int max_threads;
    int *worker_cpu;
    BLASLONG *worker_tiles;
    INPUT_T **worker_b_tile;
    const char *failure_stage = "NONE";
    double t0;
    double t1;
    double time_sec;
    double metric_value;
    double max_error = 0.0;
    size_t mismatch_count = 0;
    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;
    size_t worker_b_tile_elements;
    size_t worker_b_tile_bytes;
    INPUT_T *A;
    INPUT_T *B;
    OUTPUT_T *C;
    OUTPUT_T *C_initial;
    OUTPUT_T *C_reference = NULL;

    if (M <= 0 || N <= 0 || K <= 0 || tile_n <= 0 || tile_n % 8 != 0) {
        fprintf(stderr, "Usage: %s M N K [tile_N]\n", argv[0]);
        fprintf(stderr, "tile_N must be a positive multiple of 8.\n");
        return 1;
    }

    tiles = (N + tile_n - 1) / tile_n;
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

    A = (INPUT_T *)aligned_bytes(bytesA);
    B = (INPUT_T *)aligned_bytes(bytesB);
    C = (OUTPUT_T *)aligned_bytes(bytesC);
    C_initial = (OUTPUT_T *)aligned_bytes(bytesC);

    if (validation_enabled) {
        C_reference = (OUTPUT_T *)aligned_bytes(bytesC);
    }

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

    fill_input(A, sizeA);
    fill_input(B, sizeB);
    fill_output(C_initial, sizeC);
    memcpy(C, C_initial, bytesC);

    if (warmup_enabled) {
        int warmup_rc = call_full_kernel(M, N, K, A, B, C);
        if (warmup_rc != 0) {
            fprintf(stderr, "warmup kernel returned %d\n", warmup_rc);
            kernel_return = warmup_rc;
            failure_stage = "WARMUP";
        }
        memcpy(C, C_initial, bytesC);
    }

    if (validation_enabled && kernel_return == 0) {
        memcpy(C_reference, C_initial, bytesC);
        validation_rc = call_full_kernel(M, N, K, A, B, C_reference);
        if (validation_rc != 0) {
            fprintf(stderr, "serial validation kernel returned %d\n", validation_rc);
            kernel_return = validation_rc;
            failure_stage = "SERIAL_REFERENCE";
        }
    }

    max_threads = omp_get_max_threads();
    worker_cpu = (int *)malloc((size_t)max_threads * sizeof(int));
    worker_tiles = (BLASLONG *)calloc((size_t)max_threads, sizeof(BLASLONG));
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
    for (int i = 0; i < max_threads; ++i) {
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

    t0 = now_sec();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();

        worker_cpu[tid] = cpu;
#pragma omp single
        actual_threads = omp_get_num_threads();

#if defined(OMP_KIND_INT8_MIXED)
        /* Allocate/touch each temporary B tile after CPU placement for local first touch. */
        worker_b_tile[tid] = (INPUT_T *)aligned_bytes(worker_b_tile_bytes);
        if (worker_b_tile[tid] == NULL) {
#pragma omp critical
            {
                if (kernel_return == 0) {
                    kernel_return = 1;
                    failure_stage = "WORKER_BUFFER_ALLOCATION";
                }
            }
        } else {
            memset(worker_b_tile[tid], 0, worker_b_tile_bytes);
            /* Weighted dynamic claiming gives more C-column tiles to faster IME workers. */
            while (1) {
                BLASLONG start_tile;
                BLASLONG end_tile;
                BLASLONG chunk = mixed_tile_chunk_for_cpu(cpu);

#pragma omp atomic capture
                {
                    start_tile = next_tile;
                    next_tile += chunk;
                }

                if (start_tile >= tiles) {
                    break;
                }
                end_tile = min_blaslong(start_tile + chunk, tiles);

                for (BLASLONG tile = start_tile; tile < end_tile; ++tile) {
                    BLASLONG n0 = tile * tile_n;
                    BLASLONG nb = min_blaslong(tile_n, N - n0);
                    int rc = call_tile_kernel(M, nb, K, n0, N, A, B, C,
                                              worker_b_tile[tid]);
                    worker_tiles[tid] += 1;
                    if (rc != 0) {
#pragma omp critical
                        {
                            if (kernel_return == 0) {
                                kernel_return = rc;
                                failure_stage = "TIMED_TILED_EXECUTION";
                            }
                        }
                    }
                }
            }
        }
#else
/* Homogeneous FP/RVV/IME modes use static ownership of C-column tiles. */
#pragma omp for schedule(static)
        for (BLASLONG tile = 0; tile < tiles; ++tile) {
            BLASLONG n0 = tile * tile_n;
            BLASLONG nb = min_blaslong(tile_n, N - n0);
            int rc = call_tile_kernel(M, nb, K, n0, N, A, B, C,
                                      worker_b_tile[tid]);
            worker_tiles[tid] += 1;
            if (rc != 0) {
#pragma omp critical
                {
                    if (kernel_return == 0) {
                        kernel_return = rc;
                        failure_stage = "TIMED_TILED_EXECUTION";
                    }
                }
            }
        }
#endif
    }
    t1 = now_sec();

    time_sec = t1 - t0;
    if (validation_enabled && validation_rc == 0 && kernel_return == 0) {
        mismatch_count = compare_outputs(C, C_reference, sizeC, &max_error);
        if (mismatch_count != 0) {
            failure_stage = "NUMERICAL_VALIDATION";
        }
    }

    run_failed = kernel_return != 0 || mismatch_count != 0;
    metric_value = (!run_failed && time_sec > 0.0)
        ? (2.0 * (double)M * (double)N * (double)K) / (time_sec * 1e9)
        : 0.0;

    printf("M=%ld N=%ld K=%ld tile_N=%ld tiles=%ld threads=%d\n",
           M, N, K, tile_n, tiles, actual_threads);
    printf("TIMING_SCOPE=timed_parallel_tile_region\n");
    printf("VALIDATION_METHOD=%s\n", validation_enabled ? VALIDATION_NAME : "DISABLED");
    printf("FAILURE_STAGE=%s\n", failure_stage);
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
    printf("Time: %.9f sec\n", time_sec);
    printf("%s: %.6f\n", METRIC_NAME, metric_value);
    printf("MISMATCH_COUNT=%zu\n", mismatch_count);
    printf("MAX_ERROR=%.17g\n", max_error);
    printf("KERNEL_RETURN=%d\n", kernel_return);
    printf("CSV_RUN,%.9f,%.6f,%d,%s,%s,%zu,%.17g,%d\n",
           time_sec, metric_value, kernel_return, failure_stage, METRIC_NAME,
           mismatch_count, max_error, actual_threads);

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
    return run_failed ? 1 : 0;
}

