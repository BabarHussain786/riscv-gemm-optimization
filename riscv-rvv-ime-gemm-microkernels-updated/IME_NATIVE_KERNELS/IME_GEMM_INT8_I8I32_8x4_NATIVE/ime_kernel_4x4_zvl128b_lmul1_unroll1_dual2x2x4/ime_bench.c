#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef long BLASLONG;
typedef int8_t GEMM_I8;
typedef int32_t GEMM_I32;

int ime_kernel_4x4_zvl128b_lmul1_unroll1_dual2x2x4(
    BLASLONG M, BLASLONG N, BLASLONG K,
    GEMM_I32 alpha, const GEMM_I8 *A, const GEMM_I8 *B, GEMM_I32 *C, BLASLONG ldc
);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int env_flag_enabled(const char *name)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') return 0;
    if (value[0] == '0' || value[0] == 'n' || value[0] == 'N') return 0;
    return 1;
}

static int size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > (size_t)-1 / a) return 0;
    *out = a * b;
    return 1;
}

static void *aligned_alloc_bytes(size_t bytes)
{
    void *ptr = NULL;
    if (bytes == 0) return NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) return NULL;
    return ptr;
}

static uint32_t u32_from_i32(GEMM_I32 x)
{
    uint32_t y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static GEMM_I32 i32_from_u32(uint32_t x)
{
    GEMM_I32 y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static GEMM_I32 add_scaled_wrap_i32(GEMM_I32 dst, int64_t value, GEMM_I32 alpha)
{
    uint32_t scaled = (uint32_t)((int64_t)alpha * value);
    return i32_from_u32(u32_from_i32(dst) + scaled);
}

static void fill_i8(GEMM_I8 *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (GEMM_I8)((i % 13) - 6);
    }
}

static void fill_i32(GEMM_I32 *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (GEMM_I32)((i % 7) - 3);
    }
}

static void reference_gemm_i8i32(BLASLONG M, BLASLONG N, BLASLONG K,
                                 GEMM_I32 alpha, const GEMM_I8 *A,
                                 const GEMM_I8 *B, GEMM_I32 *C, BLASLONG ldc)
{
    for (BLASLONG col = 0; col < N; ++col) {
        for (BLASLONG row = 0; row < M; ++row) {
            int64_t sum = 0;

#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                sum += (GEMM_I32)A[k * M + row] * (GEMM_I32)B[k * N + col];
            }

            C[col * ldc + row] = add_scaled_wrap_i32(C[col * ldc + row], sum, alpha);
        }
    }
}

static int compare_i32(const GEMM_I32 *got, const GEMM_I32 *ref, size_t n,
                       size_t *mismatches, int64_t *max_abs_error)
{
    *mismatches = 0;
    *max_abs_error = 0;

    for (size_t i = 0; i < n; ++i) {
        int64_t diff = (int64_t)got[i] - (int64_t)ref[i];
        int64_t abs_diff = diff < 0 ? -diff : diff;
        if (abs_diff != 0) {
            ++(*mismatches);
            if (abs_diff > *max_abs_error) *max_abs_error = abs_diff;
        }
    }

    return *mismatches == 0;
}

int main(int argc, char **argv)
{
    BLASLONG M;
    BLASLONG N;
    BLASLONG K;
    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;
    GEMM_I8 *A = NULL;
    GEMM_I8 *B = NULL;
    GEMM_I32 *C = NULL;
    GEMM_I32 *C_ref = NULL;
    int validate;
    double t0;
    double t1;
    double time;
    double gops;
    int rc;

    if (argc < 4) {
        printf("Usage: %s M N K\n", argv[0]);
        printf("Optional: IME_VALIDATE=1 checks output against scalar INT64 reference.\n");
        printf("Optional: SPACEMIT_IME_FORCE_NATIVE=1, SPACEMIT_IME_FORCE_RVV=1, or SPACEMIT_IME_FORCE_SCALAR=1 selects path.\n");
        return 0;
    }

    M = atol(argv[1]);
    N = atol(argv[2]);
    K = atol(argv[3]);

    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "invalid matrix size\n");
        return 1;
    }

    if (!size_mul_ok((size_t)M, (size_t)K, &sizeA) ||
        !size_mul_ok((size_t)N, (size_t)K, &sizeB) ||
        !size_mul_ok((size_t)M, (size_t)N, &sizeC) ||
        !size_mul_ok(sizeA, sizeof(GEMM_I8), &bytesA) ||
        !size_mul_ok(sizeB, sizeof(GEMM_I8), &bytesB) ||
        !size_mul_ok(sizeC, sizeof(GEMM_I32), &bytesC)) {
        fprintf(stderr, "matrix size overflow\n");
        return 1;
    }

    validate = env_flag_enabled("IME_VALIDATE");

    A = (GEMM_I8 *)aligned_alloc_bytes(bytesA);
    B = (GEMM_I8 *)aligned_alloc_bytes(bytesB);
    C = (GEMM_I32 *)aligned_alloc_bytes(bytesC);
    if (validate) {
        C_ref = (GEMM_I32 *)aligned_alloc_bytes(bytesC);
    }

    if (A == NULL || B == NULL || C == NULL || (validate && C_ref == NULL)) {
        fprintf(stderr, "allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(C_ref);
        return 1;
    }

    fill_i8(A, sizeA);
    fill_i8(B, sizeB);
    fill_i32(C, sizeC);
    if (validate) {
        memcpy(C_ref, C, bytesC);
    }

    t0 = now_sec();
    rc = ime_kernel_4x4_zvl128b_lmul1_unroll1_dual2x2x4(M, N, K, 1, A, B, C, M);
    t1 = now_sec();

    time = t1 - t0;
    gops = (2.0 * (double)M * (double)N * (double)K) / (time * 1e9);

    printf("M=%ld N=%ld K=%ld\n", M, N, K);
    printf("Time: %.6f sec\n", time);
    printf("GOPS: %.4f\n", gops);
    printf("KERNEL_RETURN=%d\n", rc);

    if (validate && rc == 0) {
        size_t mismatches;
        int64_t max_abs_error;
        int ok;

        reference_gemm_i8i32(M, N, K, 1, A, B, C_ref, M);
        ok = compare_i32(C, C_ref, sizeC, &mismatches, &max_abs_error);

        printf("VALIDATION=%s mismatches=%zu max_abs_error=%lld\n",
               ok ? "OK" : "FAIL",
               mismatches,
               (long long)max_abs_error);

        if (!ok) rc = 2;
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);

    if (rc != 0) return rc;
    return 0;
}
