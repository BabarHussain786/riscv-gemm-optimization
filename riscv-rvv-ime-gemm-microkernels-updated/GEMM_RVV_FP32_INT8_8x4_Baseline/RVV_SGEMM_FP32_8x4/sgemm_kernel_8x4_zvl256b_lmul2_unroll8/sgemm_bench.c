#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

typedef long BLASLONG;
typedef float FLOAT;

#ifndef CNAME
#define CNAME sgemm_kernel_8x4_zvl256b_lmul2_unroll8
#endif

int CNAME(BLASLONG M, BLASLONG N, BLASLONG K,
          FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT *C, BLASLONG ldc);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)((i % 13) - 6) * 0.1f;
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

    FLOAT *A = aligned_alloc(64, sizeA * sizeof(FLOAT));
    FLOAT *B = aligned_alloc(64, sizeB * sizeof(FLOAT));
    FLOAT *C = aligned_alloc(64, sizeC * sizeof(FLOAT));

    if (!A || !B || !C) {
        printf("Allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    fill(A, sizeA);
    fill(B, sizeB);
    fill(C, sizeC);

    double t0 = now_sec();
    int rc = CNAME(M, N, K, 1.0f, A, B, C, M);
    double t1 = now_sec();

    if (rc != 0) {
        printf("KERNEL_RETURN=%d\n", rc);
        free(A);
        free(B);
        free(C);
        return rc;
    }

    double time = t1 - t0;
    double gflops = (2.0 * (double)M * N * K) / (time * 1e9);

    printf("M=%ld N=%ld K=%ld\n", M, N, K);
    printf("Time: %.6f sec\n", time);
    printf("GFLOPS: %.4f\n", gflops);
    printf("KERNEL_RETURN=%d\n", rc);

    free(A);
    free(B);
    free(C);

    return rc;
}
