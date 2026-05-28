#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long BLASLONG;

#ifndef KERNEL_SYMBOL
#error "KERNEL_SYMBOL must be defined by the build script"
#endif

#ifndef ACC_MR
#define ACC_MR 8
#endif

#ifndef ACC_NR
#define ACC_NR 4
#endif

#if defined(ACC_KIND_FP32)
typedef float GEMM_T;
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  GEMM_T alpha, GEMM_T *A, GEMM_T *B, GEMM_T *C, BLASLONG ldc);
#elif defined(ACC_KIND_FP64)
typedef double GEMM_T;
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  GEMM_T alpha, GEMM_T *A, GEMM_T *B, GEMM_T *C, BLASLONG ldc);
#elif defined(ACC_KIND_INT8_RVV)
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  int32_t alpha, int8_t *A, int8_t *B, int32_t *C, BLASLONG ldc);
#elif defined(ACC_KIND_INT8_IME)
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  int32_t alpha, const int8_t *A, const int8_t *B, int32_t *C, BLASLONG ldc);
#else
#error "One ACC_KIND_* macro must be defined"
#endif

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static uint32_t next_u32(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

static double uniform01(void)
{
    return (double)next_u32() / (double)UINT32_MAX;
}

static int sign_value(void)
{
    return (next_u32() & 1U) ? 1 : -1;
}

static void *aligned_bytes(size_t bytes)
{
    void *ptr = NULL;
    size_t padded = (bytes + 63U) & ~(size_t)63U;

    if (padded == 0) {
        padded = 64;
    }

    if (posix_memalign(&ptr, 64, padded) != 0) {
        return NULL;
    }

    memset(ptr, 0, padded);
    return ptr;
}

static BLASLONG tile_step(BLASLONG remaining, BLASLONG full_tile)
{
    BLASLONG step = full_tile;

    while (step > remaining && step > 1) {
        step /= 2;
    }

    if (step < 1) {
        step = 1;
    }

    return step;
}

#if defined(ACC_KIND_FP32) || defined(ACC_KIND_FP64)
static GEMM_T fp_value(const char *input_class, size_t idx, int is_a)
{
    double value;

    if (strcmp(input_class, "wide_range") == 0) {
        int exp_min = -20;
        int exp_max = 20;
        int exp_val = exp_min + (int)(next_u32() % (uint32_t)(exp_max - exp_min + 1));
        double mantissa = 1.0 + uniform01();
        value = (double)sign_value() * mantissa * ldexp(1.0, exp_val);
    } else if (strcmp(input_class, "cancellation") == 0) {
        double mag = 0.5 + 0.5 * uniform01();
        int s = ((idx + (size_t)is_a) & 1U) ? -1 : 1;
        value = (double)s * mag;
    } else {
        value = 2.0 * uniform01() - 1.0;
    }

    return (GEMM_T)value;
}

static void fill_fp(GEMM_T *x, size_t n, const char *input_class, int is_a)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = fp_value(input_class, i, is_a);
    }
}

static void reference_fp(BLASLONG M, BLASLONG N, BLASLONG K,
                         GEMM_T alpha, const GEMM_T *A, const GEMM_T *B,
                         long double *Cref, BLASLONG ldc)
{
    BLASLONG n_top = 0;

    while (n_top < N) {
        BLASLONG cols = tile_step(N - n_top, ACC_NR);
        BLASLONG m_top = 0;

        while (m_top < M) {
            BLASLONG rows = tile_step(M - m_top, ACC_MR);
            const GEMM_T *Ablk = &A[(size_t)m_top * (size_t)K];
            const GEMM_T *Bblk = &B[(size_t)n_top * (size_t)K];

            for (BLASLONG c = 0; c < cols; ++c) {
                for (BLASLONG r = 0; r < rows; ++r) {
#if defined(ACC_KIND_FP32)
                    double sum = 0.0;
#else
                    long double sum = 0.0L;
#endif
                    for (BLASLONG k = 0; k < K; ++k) {
#if defined(ACC_KIND_FP32)
                        sum += (double)Ablk[(size_t)k * (size_t)rows + (size_t)r] *
                               (double)Bblk[(size_t)k * (size_t)cols + (size_t)c];
#else
                        sum += (long double)Ablk[(size_t)k * (size_t)rows + (size_t)r] *
                               (long double)Bblk[(size_t)k * (size_t)cols + (size_t)c];
#endif
                    }

                    Cref[(size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r)] +=
                        (long double)alpha * (long double)sum;
                }
            }

            m_top += rows;
        }

        n_top += cols;
    }
}

static int run_fp(const char *input_class, BLASLONG M, BLASLONG N, BLASLONG K)
{
    size_t size_a = (size_t)M * (size_t)K;
    size_t size_b = (size_t)N * (size_t)K;
    size_t size_c = (size_t)M * (size_t)N;

    GEMM_T *A = aligned_bytes(size_a * sizeof(GEMM_T));
    GEMM_T *B = aligned_bytes(size_b * sizeof(GEMM_T));
    GEMM_T *C = aligned_bytes(size_c * sizeof(GEMM_T));
    long double *Cref = aligned_bytes(size_c * sizeof(long double));

    if (!A || !B || !C || !Cref) {
        free(A);
        free(B);
        free(C);
        free(Cref);
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    fill_fp(A, size_a, input_class, 1);
    fill_fp(B, size_b, input_class, 0);

    reference_fp(M, N, K, (GEMM_T)1, A, B, Cref, M);

    int rc = KERNEL_SYMBOL(M, N, K, (GEMM_T)1, A, B, C, M);
    if (rc != 0) {
        printf("KERNEL_RETURN,%d,NA,NA,NA,NA,NA,NA,NA\n", rc);
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 0;
    }

    long double max_abs = 0.0L;
    long double sum_abs = 0.0L;
    long double max_rel = 0.0L;
    long double sum_rel = 0.0L;
    long double eps = 1.0e-30L;

    for (size_t i = 0; i < size_c; ++i) {
        long double diff = fabsl((long double)C[i] - Cref[i]);
        long double denom = fabsl(Cref[i]);
        long double rel;

        if (denom < eps) {
            denom = eps;
        }

        rel = diff / denom;
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (rel > max_rel) {
            max_rel = rel;
        }
        sum_abs += diff;
        sum_rel += rel;
    }

    printf("OK,%d,%.17Le,%.17Le,%.17Le,%.17Le,0,0,0\n",
           rc, max_abs, sum_abs / (long double)size_c,
           max_rel, sum_rel / (long double)size_c);

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}
#endif

#if defined(ACC_KIND_INT8_RVV) || defined(ACC_KIND_INT8_IME)
static int8_t int8_value(const char *input_class, size_t idx, int is_a)
{
    if (strcmp(input_class, "wide_range") == 0) {
        if ((next_u32() % 4U) == 0U) {
            return (int8_t)((int)(next_u32() % 255U) - 127);
        }
        return (int8_t)((int)(next_u32() % 17U) - 8);
    }

    if (strcmp(input_class, "cancellation") == 0) {
        int mag = 1 + (int)(next_u32() % 16U);
        int s = ((idx + (size_t)is_a) & 1U) ? -1 : 1;
        return (int8_t)(s * mag);
    }

    return (int8_t)((int)(next_u32() % 33U) - 16);
}

static void fill_i8(int8_t *x, size_t n, const char *input_class, int is_a)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = int8_value(input_class, i, is_a);
    }
}

static int32_t wrap_i32_add(int32_t dst, int64_t addend)
{
    uint32_t a = (uint32_t)dst;
    uint32_t b = (uint32_t)((int32_t)addend);
    return (int32_t)(a + b);
}

static void reference_i32(BLASLONG M, BLASLONG N, BLASLONG K,
                          int32_t alpha, const int8_t *A, const int8_t *B,
                          int32_t *Cref, int *overflow_count, BLASLONG ldc)
{
    BLASLONG n_top = 0;

    while (n_top < N) {
        BLASLONG cols = tile_step(N - n_top, ACC_NR);
        BLASLONG m_top = 0;

        while (m_top < M) {
            BLASLONG rows = tile_step(M - m_top, ACC_MR);
            const int8_t *Ablk = &A[(size_t)m_top * (size_t)K];
            const int8_t *Bblk = &B[(size_t)n_top * (size_t)K];

            for (BLASLONG c = 0; c < cols; ++c) {
                for (BLASLONG r = 0; r < rows; ++r) {
                    int64_t sum = 0;

                    for (BLASLONG k = 0; k < K; ++k) {
                        sum += (int64_t)Ablk[(size_t)k * (size_t)rows + (size_t)r] *
                               (int64_t)Bblk[(size_t)k * (size_t)cols + (size_t)c];
                    }

                    sum *= (int64_t)alpha;
                    if (sum > INT32_MAX || sum < INT32_MIN) {
                        *overflow_count += 1;
                    }

                    Cref[(size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r)] =
                        wrap_i32_add(Cref[(size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r)], sum);
                }
            }

            m_top += rows;
        }

        n_top += cols;
    }
}

static int run_i32(const char *input_class, BLASLONG M, BLASLONG N, BLASLONG K)
{
    size_t size_a = (size_t)M * (size_t)K;
    size_t size_b = (size_t)N * (size_t)K;
    size_t size_c = (size_t)M * (size_t)N;

    int8_t *A = aligned_bytes(size_a * sizeof(int8_t));
    int8_t *B = aligned_bytes(size_b * sizeof(int8_t));
    int32_t *C = aligned_bytes(size_c * sizeof(int32_t));
    int32_t *Cref = aligned_bytes(size_c * sizeof(int32_t));

    if (!A || !B || !C || !Cref) {
        free(A);
        free(B);
        free(C);
        free(Cref);
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    fill_i8(A, size_a, input_class, 1);
    fill_i8(B, size_b, input_class, 0);

    int overflow_count = 0;
    reference_i32(M, N, K, 1, A, B, Cref, &overflow_count, M);

    int rc = KERNEL_SYMBOL(M, N, K, 1, A, B, C, M);
    if (rc != 0) {
        printf("KERNEL_RETURN,%d,NA,NA,NA,NA,NA,NA,%d\n", rc, overflow_count);
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 0;
    }

    uint64_t mismatch_count = 0;
    int64_t max_diff = 0;

    for (size_t i = 0; i < size_c; ++i) {
        int64_t diff = (int64_t)C[i] - (int64_t)Cref[i];
        int64_t abs_diff = diff < 0 ? -diff : diff;

        if (diff != 0) {
            mismatch_count += 1;
        }
        if (abs_diff > max_diff) {
            max_diff = abs_diff;
        }
    }

    printf("OK,%d,0,0,0,0,%" PRIu64 ",%" PRId64 ",%d\n",
           rc, mismatch_count, max_diff, overflow_count);

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}
#endif

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s input_class M N K\n", argv[0]);
        return 2;
    }

    const char *input_class = argv[1];
    BLASLONG M = atol(argv[2]);
    BLASLONG N = atol(argv[3]);
    BLASLONG K = atol(argv[4]);

    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "M, N, and K must be positive\n");
        return 2;
    }

    rng_state ^= (uint64_t)M * 0x100000001b3ULL;
    rng_state ^= (uint64_t)N * 0xbf58476d1ce4e5b9ULL;
    rng_state ^= (uint64_t)K * 0x94d049bb133111ebULL;

#if defined(ACC_KIND_FP32) || defined(ACC_KIND_FP64)
    return run_fp(input_class, M, N, K);
#else
    return run_i32(input_class, M, N, K);
#endif
}
