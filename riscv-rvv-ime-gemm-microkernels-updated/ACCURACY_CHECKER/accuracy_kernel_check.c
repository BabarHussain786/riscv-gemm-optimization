#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <float.h>
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

#ifndef ACC_FP_BOUND_RATIO_LIMIT
#define ACC_FP_BOUND_RATIO_LIMIT 1.0L
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

static uint64_t rng_state;

static uint32_t next_u32(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

static uint32_t uniform_u32_below(uint32_t bound)
{
    uint32_t threshold;
    uint32_t value;

    if (bound == 0) {
        return next_u32();
    }

    threshold = (uint32_t)(-bound) % bound;
    do {
        value = next_u32();
    } while (value < threshold);

    return value % bound;
}

static int uniform_int_range(int min_value, int max_value)
{
    uint32_t width = (uint32_t)(max_value - min_value + 1);
    return min_value + (int)uniform_u32_below(width);
}

static double uniform01(void)
{
    return (double)next_u32() / 4294967296.0;
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
static int valid_fp_input_class(const char *input_class)
{
    return strcmp(input_class, "bounded_uniform") == 0 ||
           strcmp(input_class, "wide_range") == 0 ||
           strcmp(input_class, "cancellation_stress") == 0;
}

static GEMM_T fp_random_value(const char *input_class)
{
    if (strcmp(input_class, "wide_range") == 0) {
        int exponent = uniform_int_range(-20, 20);
        double mantissa = 1.0 + uniform01();
        return (GEMM_T)((double)sign_value() * mantissa * ldexp(1.0, exponent));
    }

    return (GEMM_T)(2.0 * uniform01() - 1.0);
}

static GEMM_T fp_next_toward_zero(GEMM_T value)
{
#if defined(ACC_KIND_FP32)
    return nextafterf(value, (GEMM_T)0);
#else
    return nextafter(value, (GEMM_T)0);
#endif
}

static void fill_fp_flat(GEMM_T *x, size_t count, const char *input_class)
{
    for (size_t i = 0; i < count; ++i) {
        x[i] = fp_random_value(input_class);
    }
}

/*
 * Each neighboring K pair contributes almost zero to every dot product.
 * The final B value moves one representable step toward zero so the output
 * remains small but nonzero. This exercises cancellation-sensitive paths.
 */
static void fill_fp_cancellation(GEMM_T *x, BLASLONG outer, BLASLONG K,
                                 BLASLONG full_tile, int is_a)
{
    BLASLONG top = 0;

    while (top < outer) {
        BLASLONG width = tile_step(outer - top, full_tile);
        GEMM_T *panel = &x[(size_t)top * (size_t)K];

        for (BLASLONG k = 0; k + 1 < K; k += 2) {
            for (BLASLONG lane = 0; lane < width; ++lane) {
                GEMM_T value = (GEMM_T)((double)sign_value() * (0.5 + 0.5 * uniform01()));
                GEMM_T paired = is_a ? value : (GEMM_T)-value;

                if (!is_a && k + 2 == K) {
                    paired = fp_next_toward_zero(paired);
                }

                panel[(size_t)k * (size_t)width + (size_t)lane] = value;
                panel[(size_t)(k + 1) * (size_t)width + (size_t)lane] = paired;
            }
        }

        if (K & 1) {
            for (BLASLONG lane = 0; lane < width; ++lane) {
                panel[(size_t)(K - 1) * (size_t)width + (size_t)lane] =
                    fp_random_value("bounded_uniform");
            }
        }

        top += width;
    }
}

static void fill_fp_inputs(GEMM_T *A, GEMM_T *B, BLASLONG M, BLASLONG N,
                           BLASLONG K, const char *input_class)
{
    if (strcmp(input_class, "cancellation_stress") == 0) {
        fill_fp_cancellation(A, M, K, ACC_MR, 1);
        fill_fp_cancellation(B, N, K, ACC_NR, 0);
        return;
    }

    fill_fp_flat(A, (size_t)M * (size_t)K, input_class);
    fill_fp_flat(B, (size_t)N * (size_t)K, input_class);
}

static void reference_fp(BLASLONG M, BLASLONG N, BLASLONG K,
                         GEMM_T alpha, const GEMM_T *A, const GEMM_T *B,
                         long double *Cref, long double *Cscale, BLASLONG ldc)
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
                    long double sum = 0.0L;
                    long double scale = 0.0L;

                    for (BLASLONG k = 0; k < K; ++k) {
                        long double a = (long double)Ablk[(size_t)k * (size_t)rows + (size_t)r];
                        long double b = (long double)Bblk[(size_t)k * (size_t)cols + (size_t)c];

                        sum += a * b;
                        scale += fabsl(a * b);
                    }

                    Cref[(size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r)] +=
                        (long double)alpha * sum;
                    Cscale[(size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r)] +=
                        fabsl((long double)alpha) * scale;
                }
            }

            m_top += rows;
        }

        n_top += cols;
    }
}

static long double fp_unit_roundoff(void)
{
#if defined(ACC_KIND_FP32)
    return (long double)FLT_EPSILON / 2.0L;
#else
    return (long double)DBL_EPSILON / 2.0L;
#endif
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
    long double *Cscale = aligned_bytes(size_c * sizeof(long double));

    if (!A || !B || !C || !Cref || !Cscale) {
        free(A);
        free(B);
        free(C);
        free(Cref);
        free(Cscale);
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    fill_fp_inputs(A, B, M, N, K, input_class);
    reference_fp(M, N, K, (GEMM_T)1, A, B, Cref, Cscale, M);

    int rc = KERNEL_SYMBOL(M, N, K, (GEMM_T)1, A, B, C, M);
    if (rc != 0) {
        printf("KERNEL_RETURN,%d,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n", rc);
        free(A);
        free(B);
        free(C);
        free(Cref);
        free(Cscale);
        return 0;
    }

    long double max_abs = 0.0L;
    long double sum_abs = 0.0L;
    long double max_rel = 0.0L;
    long double sum_rel = 0.0L;
    long double sum_diff_sq = 0.0L;
    long double sum_ref_sq = 0.0L;
    long double max_bound_ratio = 0.0L;
    long double unit_roundoff = fp_unit_roundoff();
    uint64_t bound_failure_count = 0;
    uint64_t nonfinite_count = 0;

    for (size_t i = 0; i < size_c; ++i) {
        long double output = (long double)C[i];
        long double diff = fabsl(output - Cref[i]);
        long double ref_abs = fabsl(Cref[i]);
        long double bound = (long double)K * unit_roundoff * Cscale[i];
        long double rel = ref_abs == 0.0L ? (diff == 0.0L ? 0.0L : INFINITY) : diff / ref_abs;
        long double bound_ratio = bound == 0.0L ? (diff == 0.0L ? 0.0L : INFINITY) : diff / bound;

        if (!isfinite(output) || !isfinite(Cref[i]) || !isfinite(diff) ||
            !isfinite(rel) || !isfinite(bound_ratio)) {
            nonfinite_count += 1;
        }
        if (!isfinite(bound_ratio) || bound_ratio > ACC_FP_BOUND_RATIO_LIMIT) {
            bound_failure_count += 1;
        }
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (rel > max_rel) {
            max_rel = rel;
        }
        if (bound_ratio > max_bound_ratio) {
            max_bound_ratio = bound_ratio;
        }

        sum_abs += diff;
        sum_rel += rel;
        sum_diff_sq += diff * diff;
        sum_ref_sq += Cref[i] * Cref[i];
    }

    long double frobenius_rel =
        sum_ref_sq == 0.0L ? (sum_diff_sq == 0.0L ? 0.0L : INFINITY)
                           : sqrtl(sum_diff_sq) / sqrtl(sum_ref_sq);
    const char *status =
        bound_failure_count == 0 && nonfinite_count == 0 ? "OK" : "NUMERICAL_FAILED";

    printf("%s,%d,%.17Le,%.17Le,%.17Le,%.17Le,%.17Le,%.17Le,%" PRIu64 ",%" PRIu64 ",0,0,0\n",
           status, rc, max_abs, sum_abs / (long double)size_c,
           max_rel, sum_rel / (long double)size_c, frobenius_rel,
           max_bound_ratio, bound_failure_count, nonfinite_count);

    free(A);
    free(B);
    free(C);
    free(Cref);
    free(Cscale);
    return 0;
}
#endif

#if defined(ACC_KIND_INT8_RVV) || defined(ACC_KIND_INT8_IME)
static int valid_i8_input_class(const char *input_class)
{
    return strcmp(input_class, "bounded_uniform") == 0 ||
           strcmp(input_class, "full_range_uniform") == 0 ||
           strcmp(input_class, "mixed_magnitude") == 0 ||
           strcmp(input_class, "cancellation_stress") == 0;
}

static int8_t int8_random_value(const char *input_class)
{
    if (strcmp(input_class, "full_range_uniform") == 0) {
        return (int8_t)uniform_int_range(-128, 127);
    }

    if (strcmp(input_class, "mixed_magnitude") == 0) {
        if (uniform_u32_below(4) == 0) {
            return (int8_t)uniform_int_range(-128, 127);
        }
        return (int8_t)uniform_int_range(-8, 8);
    }

    return (int8_t)uniform_int_range(-16, 16);
}

static void fill_i8_flat(int8_t *x, size_t count, const char *input_class)
{
    for (size_t i = 0; i < count; ++i) {
        x[i] = int8_random_value(input_class);
    }
}

/*
 * Each neighboring K pair cancels, except for a one-unit perturbation in
 * the final B pair. The expected result is deliberately small and nonzero.
 */
static void fill_i8_cancellation(int8_t *x, BLASLONG outer, BLASLONG K,
                                 BLASLONG full_tile, int is_a)
{
    BLASLONG top = 0;

    while (top < outer) {
        BLASLONG width = tile_step(outer - top, full_tile);
        int8_t *panel = &x[(size_t)top * (size_t)K];

        for (BLASLONG k = 0; k + 1 < K; k += 2) {
            for (BLASLONG lane = 0; lane < width; ++lane) {
                int value = sign_value() * uniform_int_range(1, 16);
                int paired = is_a ? value : -value;

                if (!is_a && k + 2 == K) {
                    paired += paired < 0 ? 1 : -1;
                }

                panel[(size_t)k * (size_t)width + (size_t)lane] = (int8_t)value;
                panel[(size_t)(k + 1) * (size_t)width + (size_t)lane] = (int8_t)paired;
            }
        }

        if (K & 1) {
            for (BLASLONG lane = 0; lane < width; ++lane) {
                panel[(size_t)(K - 1) * (size_t)width + (size_t)lane] =
                    int8_random_value("bounded_uniform");
            }
        }

        top += width;
    }
}

static void fill_i8_inputs(int8_t *A, int8_t *B, BLASLONG M, BLASLONG N,
                           BLASLONG K, const char *input_class)
{
    if (strcmp(input_class, "cancellation_stress") == 0) {
        fill_i8_cancellation(A, M, K, ACC_MR, 1);
        fill_i8_cancellation(B, N, K, ACC_NR, 0);
        return;
    }

    fill_i8_flat(A, (size_t)M * (size_t)K, input_class);
    fill_i8_flat(B, (size_t)N * (size_t)K, input_class);
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

    fill_i8_inputs(A, B, M, N, K, input_class);

    int overflow_count = 0;
    reference_i32(M, N, K, 1, A, B, Cref, &overflow_count, M);

    int rc = KERNEL_SYMBOL(M, N, K, 1, A, B, C, M);
    if (rc != 0) {
        printf("KERNEL_RETURN,%d,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,%d\n", rc, overflow_count);
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

    const char *status =
        mismatch_count == 0 && max_diff == 0 && overflow_count == 0 ? "OK" : "NUMERICAL_FAILED";

    printf("%s,%d,NA,NA,NA,NA,NA,NA,0,0,%" PRIu64 ",%" PRId64 ",%d\n",
           status, rc, mismatch_count, max_diff, overflow_count);

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}
#endif

int main(int argc, char **argv)
{
    char *seed_end = NULL;

    if (argc != 6) {
        fprintf(stderr, "Usage: %s input_class M N K seed\n", argv[0]);
        return 2;
    }

    const char *input_class = argv[1];
    BLASLONG M = atol(argv[2]);
    BLASLONG N = atol(argv[3]);
    BLASLONG K = atol(argv[4]);
    uint64_t seed = strtoull(argv[5], &seed_end, 10);

    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "M, N, and K must be positive\n");
        return 2;
    }
    if (!seed_end || *seed_end != '\0') {
        fprintf(stderr, "seed must be an unsigned integer\n");
        return 2;
    }

#if defined(ACC_KIND_FP32) || defined(ACC_KIND_FP64)
    if (!valid_fp_input_class(input_class)) {
        fprintf(stderr, "unsupported floating-point input class: %s\n", input_class);
        return 2;
    }
#else
    if (!valid_i8_input_class(input_class)) {
        fprintf(stderr, "unsupported INT8 input class: %s\n", input_class);
        return 2;
    }
#endif

    rng_state = seed ^ 0x9e3779b97f4a7c15ULL;
    rng_state ^= (uint64_t)M * 0x100000001b3ULL;
    rng_state ^= (uint64_t)N * 0xbf58476d1ce4e5b9ULL;
    rng_state ^= (uint64_t)K * 0x94d049bb133111ebULL;

#if defined(ACC_KIND_FP32) || defined(ACC_KIND_FP64)
    return run_fp(input_class, M, N, K);
#else
    return run_i32(input_class, M, N, K);
#endif
}
