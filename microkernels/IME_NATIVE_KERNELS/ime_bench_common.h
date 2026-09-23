#ifndef IME_BENCH_COMMON_H
#define IME_BENCH_COMMON_H

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef IME_BENCH_KERNEL
#error "Define IME_BENCH_KERNEL before including ime_bench_common.h"
#endif

#define IME_STRINGIFY_INNER(x) #x
#define IME_STRINGIFY(x) IME_STRINGIFY_INNER(x)

typedef long BLASLONG;
typedef int8_t GEMM_I8;
typedef int32_t GEMM_I32;

int IME_BENCH_KERNEL(BLASLONG M, BLASLONG N, BLASLONG K,
                     GEMM_I32 alpha, const GEMM_I8 *A, const GEMM_I8 *B,
                     GEMM_I32 *C, BLASLONG ldc);

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

static const char *requested_path_name(void)
{
    if (env_flag_enabled("SPACEMIT_IME_FORCE_SCALAR")) return "SCALAR";
    if (env_flag_enabled("SPACEMIT_IME_FORCE_RVV")) return "RVV";
    if (env_flag_enabled("SPACEMIT_IME_FORCE_NATIVE")) return "IME_NATIVE";
    return "AUTO_DISPATCH";
}

static int size_mul_ok(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return 0;
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

static uint32_t u32_from_i32(GEMM_I32 value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static GEMM_I32 i32_from_u32(uint32_t bits)
{
    GEMM_I32 value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static GEMM_I32 add_scaled_wrap_i32(GEMM_I32 dst, int64_t value,
                                    GEMM_I32 alpha)
{
    uint32_t scaled = (uint32_t)((int64_t)alpha * value);
    return i32_from_u32(u32_from_i32(dst) + scaled);
}

static void fill_i8(GEMM_I8 *values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        values[i] = (GEMM_I8)((i % 13) - 6);
    }
}

static void fill_i32(GEMM_I32 *values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        values[i] = (GEMM_I32)((i % 7) - 3);
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
                sum += (GEMM_I32)A[k * M + row]
                     * (GEMM_I32)B[k * N + col];
            }

            C[col * ldc + row] =
                add_scaled_wrap_i32(C[col * ldc + row], sum, alpha);
        }
    }
}

static size_t compare_i32(const GEMM_I32 *got, const GEMM_I32 *reference,
                          size_t count, int64_t *max_abs_error)
{
    size_t mismatches = 0;
    *max_abs_error = 0;

    for (size_t i = 0; i < count; ++i) {
        int64_t difference = (int64_t)got[i] - (int64_t)reference[i];
        int64_t absolute = difference < 0 ? -difference : difference;
        if (absolute != 0) {
            ++mismatches;
            if (absolute > *max_abs_error) *max_abs_error = absolute;
        }
    }

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
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;
    GEMM_I8 *A = NULL;
    GEMM_I8 *B = NULL;
    GEMM_I32 *C = NULL;
    GEMM_I32 *reference = NULL;
    int validate;
    int return_code;
    int cpu_before;
    int cpu_after;
    double start;
    double finish;
    double elapsed;
    double gops;
    const char *requested_path;

    if (argc < 4) {
        printf("Usage: %s M N K\n", argv[0]);
        printf("Set IME_VALIDATE=1 or GEMM_VALIDATE=1 for correctness checking.\n");
        printf("Select a path with SPACEMIT_IME_FORCE_NATIVE=1, SPACEMIT_IME_FORCE_RVV=1, or SPACEMIT_IME_FORCE_SCALAR=1.\n");
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
        !size_mul_ok(sizeA, sizeof(*A), &bytesA) ||
        !size_mul_ok(sizeB, sizeof(*B), &bytesB) ||
        !size_mul_ok(sizeC, sizeof(*C), &bytesC)) {
        fprintf(stderr, "matrix size overflow\n");
        return 1;
    }

    validate = env_flag_enabled("IME_VALIDATE") ||
               env_flag_enabled("GEMM_VALIDATE");
    A = (GEMM_I8 *)aligned_alloc_bytes(bytesA);
    B = (GEMM_I8 *)aligned_alloc_bytes(bytesB);
    C = (GEMM_I32 *)aligned_alloc_bytes(bytesC);
    if (validate) reference = (GEMM_I32 *)aligned_alloc_bytes(bytesC);

    if (A == NULL || B == NULL || C == NULL ||
        (validate && reference == NULL)) {
        fprintf(stderr, "allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(reference);
        return 1;
    }

    fill_i8(A, sizeA);
    fill_i8(B, sizeB);
    fill_i32(C, sizeC);
    if (validate) memcpy(reference, C, bytesC);

    requested_path = requested_path_name();
    cpu_before = sched_getcpu();
    start = now_sec();
    return_code = IME_BENCH_KERNEL(M, N, K, 1, A, B, C, M);
    finish = now_sec();
    cpu_after = sched_getcpu();

    elapsed = finish - start;
    gops = (2.0 * (double)M * (double)N * (double)K) /
           (elapsed * 1e9);

    printf("KERNEL=%s\n", IME_STRINGIFY(IME_BENCH_KERNEL));
    printf("REQUESTED_PATH=%s\n", requested_path);
    printf("EXECUTED_PATH=%s\n", return_code == 0 ? requested_path : "NONE");
    printf("CPU_BEFORE=%d CPU_AFTER=%d\n", cpu_before, cpu_after);
    printf("M=%ld N=%ld K=%ld\n", M, N, K);
    printf("Time: %.6f sec\n", elapsed);
    printf("GOPS: %.4f\n", gops);
    printf("KERNEL_RETURN=%d\n", return_code);

    if (validate && return_code == 0) {
        int64_t max_abs_error;
        size_t mismatches;

        reference_gemm_i8i32(M, N, K, 1, A, B, reference, M);
        mismatches = compare_i32(C, reference, sizeC, &max_abs_error);
        printf("VALIDATION=%s mismatches=%zu max_abs_error=%lld\n",
               mismatches == 0 ? "OK" : "FAIL", mismatches,
               (long long)max_abs_error);
        if (mismatches != 0) return_code = 2;
    } else if (validate) {
        printf("VALIDATION=SKIPPED kernel_return=%d\n", return_code);
    } else {
        printf("VALIDATION=DISABLED\n");
    }

    free(A);
    free(B);
    free(C);
    free(reference);
    return return_code;
}

#undef IME_STRINGIFY
#undef IME_STRINGIFY_INNER

#endif
