#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

typedef long BLASLONG;
typedef int8_t IFLOAT;
typedef int32_t OFLOAT;

#ifndef CNAME
#define CNAME igemm_kernel_8x4_zvl256b
#endif

int CNAME(BLASLONG M, BLASLONG N, BLASLONG K,
          OFLOAT alpha, IFLOAT *A, IFLOAT *B, OFLOAT *C, BLASLONG ldc);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_i8(IFLOAT *x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        x[i] = (IFLOAT)((int)(i % 17) - 8);
    }
}

static void fill_i32(OFLOAT *x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        x[i] = (OFLOAT)((int)(i % 23) - 11);
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: %s M N K\n", argv[0]);
        return 0;
    }

    BLASLONG M = atol(argv[1]);
    BLASLONG N = atol(argv[2]);
    BLASLONG K = atol(argv[3]);

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)N * K;
    size_t sizeC = (size_t)M * N;

    IFLOAT *A = aligned_alloc(64, sizeA * sizeof(IFLOAT));
    IFLOAT *B = aligned_alloc(64, sizeB * sizeof(IFLOAT));
    OFLOAT *C = aligned_alloc(64, sizeC * sizeof(OFLOAT));

    if (!A || !B || !C) {
        printf("Allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    fill_i8(A, sizeA);
    fill_i8(B, sizeB);
    fill_i32(C, sizeC);

    double t0 = now_sec();
    int rc = CNAME(M, N, K, 1, A, B, C, M);
    double t1 = now_sec();

    if (rc != 0) {
        printf("KERNEL_RETURN=%d\n", rc);
        free(A); free(B); free(C);
        return rc;
    }

    double time = t1 - t0;
    double gops = (2.0 * (double)M * N * K) / (time * 1e9);

    printf("M=%ld N=%ld K=%ld\n", M, N, K);
    printf("Time: %.6f sec\n", time);
    printf("GOPS: %.4f\n", gops);
    printf("KERNEL_RETURN=%d\n", rc);

    free(A);
    free(B);
    free(C);
    return rc;
}