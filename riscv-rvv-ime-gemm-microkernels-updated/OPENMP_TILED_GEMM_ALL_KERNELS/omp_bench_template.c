#define _POSIX_C_SOURCE 200809L

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef long BLASLONG;

#ifndef KERNEL_SYMBOL
#error "KERNEL_SYMBOL must be defined by the build script"
#endif

#if defined(OMP_KIND_FP32)
typedef float GEMM_T;
#define METRIC_NAME "GFLOPS"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  GEMM_T alpha, GEMM_T *A, GEMM_T *B, GEMM_T *C,
                  BLASLONG ldc);
#elif defined(OMP_KIND_FP64)
typedef double GEMM_T;
#define METRIC_NAME "GFLOPS"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  GEMM_T alpha, GEMM_T *A, GEMM_T *B, GEMM_T *C,
                  BLASLONG ldc);
#elif defined(OMP_KIND_INT8_RVV)
typedef int8_t GEMM_I8;
typedef int32_t GEMM_I32;
#define METRIC_NAME "GOPS"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  GEMM_I32 alpha, GEMM_I8 *A, GEMM_I8 *B, GEMM_I32 *C,
                  BLASLONG ldc);
#elif defined(OMP_KIND_INT8_IME)
typedef int8_t GEMM_I8;
typedef int32_t GEMM_I32;
#define METRIC_NAME "GOPS"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  GEMM_I32 alpha, const GEMM_I8 *A, const GEMM_I8 *B,
                  GEMM_I32 *C, BLASLONG ldc);
#else
#error "Define one OMP_KIND_* macro"
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

static BLASLONG min_blaslong(BLASLONG a, BLASLONG b)
{
    return (a < b) ? a : b;
}

#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
static void fill_fp(GEMM_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (GEMM_T)((int)(i % 13) - 6) * (GEMM_T)0.1;
    }
}
#else
static void fill_i8(GEMM_I8 *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (GEMM_I8)((int)(i % 13) - 6);
    }
}

static void fill_i32(GEMM_I32 *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (GEMM_I32)((int)(i % 7) - 3);
    }
}
#endif

int main(int argc, char **argv)
{
    BLASLONG M = (argc > 1) ? atol(argv[1]) : 1024;
    BLASLONG N = (argc > 2) ? atol(argv[2]) : 1024;
    BLASLONG K = (argc > 3) ? atol(argv[3]) : 1024;
    BLASLONG tile_n = (argc > 4) ? atol(argv[4]) : 64;
    BLASLONG tiles;
    int error_flag = 0;
    double t0;
    double t1;
    double time_sec;
    double metric_value;

    if (M <= 0 || N <= 0 || K <= 0 || tile_n <= 0) {
        fprintf(stderr, "Usage: %s M N K [tile_N]\n", argv[0]);
        return 1;
    }

    tiles = (N + tile_n - 1) / tile_n;

#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
    size_t sizeA = (size_t)M * (size_t)K;
    size_t sizeB = (size_t)N * (size_t)K;
    size_t sizeC = (size_t)M * (size_t)N;
    GEMM_T *A = (GEMM_T *)aligned_bytes(sizeA * sizeof(GEMM_T));
    GEMM_T *B = (GEMM_T *)aligned_bytes(sizeB * sizeof(GEMM_T));
    GEMM_T *C = (GEMM_T *)aligned_bytes(sizeC * sizeof(GEMM_T));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    fill_fp(A, sizeA);
    fill_fp(B, sizeB);
    fill_fp(C, sizeC);
#else
    size_t sizeA = (size_t)M * (size_t)K;
    size_t sizeB = (size_t)N * (size_t)K;
    size_t sizeC = (size_t)M * (size_t)N;
    GEMM_I8 *A = (GEMM_I8 *)aligned_bytes(sizeA * sizeof(GEMM_I8));
    GEMM_I8 *B = (GEMM_I8 *)aligned_bytes(sizeB * sizeof(GEMM_I8));
    GEMM_I32 *C = (GEMM_I32 *)aligned_bytes(sizeC * sizeof(GEMM_I32));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    fill_i8(A, sizeA);
    fill_i8(B, sizeB);
    fill_i32(C, sizeC);
#endif

    t0 = now_sec();
#pragma omp parallel for schedule(static) reduction(|:error_flag)
    for (BLASLONG tile = 0; tile < tiles; ++tile) {
        BLASLONG n0 = tile * tile_n;
        BLASLONG nb = min_blaslong(tile_n, N - n0);
        int rc;

#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
        rc = KERNEL_SYMBOL(M, nb, K, (GEMM_T)1,
                           A, B + n0 * K, C + n0 * M, M);
#elif defined(OMP_KIND_INT8_RVV)
        rc = KERNEL_SYMBOL(M, nb, K, 1,
                           A, B + n0 * K, C + n0 * M, M);
#else
        rc = KERNEL_SYMBOL(M, nb, K, 1,
                           A, B + n0 * K, C + n0 * M, M);
#endif
        if (rc != 0) {
            error_flag = 1;
        }
    }
    t1 = now_sec();

    time_sec = t1 - t0;
    metric_value = (error_flag == 0 && time_sec > 0.0)
        ? (2.0 * (double)M * (double)N * (double)K) / (time_sec * 1e9)
        : 0.0;

    printf("M=%ld N=%ld K=%ld tile_N=%ld threads=%d\n",
           M, N, K, tile_n, omp_get_max_threads());
    printf("Time: %.9f sec\n", time_sec);
    printf("%s: %.6f\n", METRIC_NAME, metric_value);
    printf("KERNEL_RETURN=%d\n", error_flag);
    printf("CSV_RUN,%.9f,%.6f,%d,%s\n",
           time_sec, metric_value, error_flag, METRIC_NAME);

    free(A);
    free(B);
    free(C);
    return error_flag;
}
