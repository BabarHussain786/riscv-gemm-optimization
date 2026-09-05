#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef long BLASLONG;
typedef double FLOAT;

#ifndef CNAME
#define CNAME dgemm_kernel_8x8_zvl256b_lmul2_unroll8
#endif

int CNAME(BLASLONG M, BLASLONG N, BLASLONG K,
          FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT *C, BLASLONG ldc);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int parse_positive_long(const char *text, BLASLONG *value)
{
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int checked_product(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return -1;
    }
    *result = a * b;
    return 0;
}

static void *aligned_array(size_t count, size_t element_size)
{
    void *ptr = NULL;
    size_t bytes;

    if (checked_product(count, element_size, &bytes) != 0 || bytes == 0) {
        return NULL;
    }
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    return ptr;
}

static void fill(FLOAT *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (FLOAT)((int)(i % 13) - 6) * 0.1;
    }
}

/* Locate one row/column inside the same 8/4/2/1 packed blocks as the kernel. */
static void packed_block(BLASLONG index, BLASLONG total, BLASLONG main_width,
                         BLASLONG *block_start, BLASLONG *block_width)
{
    BLASLONG start = (index / main_width) * main_width;
    BLASLONG full_end = (total / main_width) * main_width;

    if (index < full_end) {
        *block_start = start;
        *block_width = main_width;
        return;
    }

    start = full_end;
    for (BLASLONG width = main_width / 2; width >= 1; width /= 2) {
        if (total - start >= width) {
            if (index < start + width) {
                *block_start = start;
                *block_width = width;
                return;
            }
            start += width;
        }
    }

    *block_start = index;
    *block_width = 1;
}

static size_t packed_offset(BLASLONG index, BLASLONG total, BLASLONG main_width,
                            BLASLONG k, BLASLONG K)
{
    BLASLONG start;
    BLASLONG width;

    packed_block(index, total, main_width, &start, &width);
    return (size_t)start * (size_t)K + (size_t)k * (size_t)width
           + (size_t)(index - start);
}

static void compute_reference(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                              const FLOAT *A, const FLOAT *B,
                              const FLOAT *C_initial, FLOAT *reference)
{
    for (BLASLONG n = 0; n < N; ++n) {
        for (BLASLONG m = 0; m < M; ++m) {
            double sum = 0.0;
            for (BLASLONG k = 0; k < K; ++k) {
                const size_t ai = packed_offset(m, M, 8, k, K);
                const size_t bi = packed_offset(n, N, 8, k, K);
                sum += (double)A[ai] * (double)B[bi];
            }
            const size_t ci = (size_t)n * (size_t)M + (size_t)m;
            reference[ci] = (FLOAT)((double)C_initial[ci] + (double)alpha * sum);
        }
    }
}

static size_t compare_outputs(const FLOAT *actual, const FLOAT *reference,
                              size_t count, double *max_abs_error)
{
    size_t mismatches = 0;
    double maximum = 0.0;

    for (size_t i = 0; i < count; ++i) {
        const double a = (double)actual[i];
        const double r = (double)reference[i];
        const double error = fabs(a - r);
        const double tolerance = 1e-11 * (1.0 + fabs(r));

        if (!isfinite(a) || !isfinite(r) || error > tolerance) {
            ++mismatches;
        }
        if (error > maximum) {
            maximum = error;
        }
    }

    *max_abs_error = maximum;
    return mismatches;
}

int main(int argc, char **argv)
{
    BLASLONG M;
    BLASLONG N;
    BLASLONG K;
    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    FLOAT *A = NULL;
    FLOAT *B = NULL;
    FLOAT *C = NULL;
    FLOAT *C_initial = NULL;
    FLOAT *reference = NULL;
    const char *validate_env = getenv("GEMM_VALIDATE");
    const int validate = validate_env != NULL && strcmp(validate_env, "0") != 0;

    if (argc != 4 || parse_positive_long(argv[1], &M) != 0
        || parse_positive_long(argv[2], &N) != 0
        || parse_positive_long(argv[3], &K) != 0) {
        fprintf(stderr, "Usage: %s M N K (all dimensions must be positive)\n", argv[0]);
        return 2;
    }

    if (checked_product((size_t)M, (size_t)K, &sizeA) != 0
        || checked_product((size_t)N, (size_t)K, &sizeB) != 0
        || checked_product((size_t)M, (size_t)N, &sizeC) != 0) {
        fprintf(stderr, "Matrix size overflow\n");
        return 2;
    }

    A = aligned_array(sizeA, sizeof(*A));
    B = aligned_array(sizeB, sizeof(*B));
    C = aligned_array(sizeC, sizeof(*C));
    if (validate) {
        C_initial = aligned_array(sizeC, sizeof(*C_initial));
        reference = aligned_array(sizeC, sizeof(*reference));
    }

    if (A == NULL || B == NULL || C == NULL
        || (validate && (C_initial == NULL || reference == NULL))) {
        fprintf(stderr, "Allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(C_initial);
        free(reference);
        return 1;
    }

    fill(A, sizeA);
    fill(B, sizeB);
    fill(C, sizeC);
    if (validate) {
        memcpy(C_initial, C, sizeC * sizeof(*C));
    }

    const double t0 = now_sec();
    const int rc = CNAME(M, N, K, 1.0, A, B, C, M);
    const double t1 = now_sec();

    printf("M=%ld N=%ld K=%ld\n", M, N, K);
    printf("KERNEL_RETURN=%d\n", rc);
    if (rc != 0) {
        free(A);
        free(B);
        free(C);
        free(C_initial);
        free(reference);
        return 255;
    }

    const double elapsed = t1 - t0;
    const double gflops = (2.0 * (double)M * (double)N * (double)K)
                          / (elapsed * 1e9);
    printf("Time: %.6f sec\n", elapsed);
    printf("GFLOPS: %.4f\n", gflops);

    int exit_code = 0;
    if (validate) {
        double max_abs_error = 0.0;
        compute_reference(M, N, K, 1.0, A, B, C_initial, reference);
        const size_t mismatches = compare_outputs(C, reference, sizeC, &max_abs_error);
        printf("VALIDATION=%s mismatches=%zu max_abs_error=%.9g\n",
               mismatches == 0 ? "OK" : "FAIL", mismatches, max_abs_error);
        if (mismatches != 0) {
            exit_code = 3;
        }
    } else {
        printf("VALIDATION=DISABLED\n");
    }

    free(A);
    free(B);
    free(C);
    free(C_initial);
    free(reference);
    return exit_code;
}

