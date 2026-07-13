#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long BLASLONG;

#ifndef KERNEL_SYMBOL
#error "KERNEL_SYMBOL must be defined by the build command"
#endif

#ifndef ACC_MR
#define ACC_MR 8
#endif

#ifndef ACC_NR
#define ACC_NR 4
#endif

#if defined(ACC_KIND_INT8_RVV)
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  int32_t alpha, int8_t *A, int8_t *B,
                  int32_t *C, BLASLONG ldc);
#elif defined(ACC_KIND_INT8_IME)
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  int32_t alpha, const int8_t *A, const int8_t *B,
                  int32_t *C, BLASLONG ldc);
#else
#error "Define ACC_KIND_INT8_RVV or ACC_KIND_INT8_IME"
#endif

#if defined(ACC_KIND_INT8_IME)
#if !defined(ACC_IME_INPUT_FULL_MATRIX)
#error "Define ACC_IME_INPUT_FULL_MATRIX for IME"
#endif
#endif

static uint64_t rng_state;

static uint32_t next_u32(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

static uint32_t uniform_u32_below(uint32_t bound)
{
    uint32_t threshold = (uint32_t)(-bound) % bound;
    uint32_t value;

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

static int sign_value(void)
{
    return (next_u32() & 1U) ? 1 : -1;
}

static void *aligned_bytes(size_t bytes)
{
    void *ptr = NULL;
    size_t padded;

    if (bytes > SIZE_MAX - 63U) {
        return NULL;
    }
    padded = (bytes + 63U) & ~(size_t)63U;

    if (padded == 0) {
        padded = 64;
    }
    if (posix_memalign(&ptr, 64, padded) != 0) {
        return NULL;
    }

    memset(ptr, 0, padded);
    return ptr;
}

static int size_mul_ok(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static BLASLONG tile_step(BLASLONG remaining, BLASLONG full_tile)
{
    BLASLONG step = full_tile;

    while (step > remaining && step > 1) {
        step /= 2;
    }

    return step < 1 ? 1 : step;
}

static int valid_input_class(const char *input_class)
{
    return strcmp(input_class, "bounded_uniform") == 0 ||
           strcmp(input_class, "full_range_uniform") == 0 ||
           strcmp(input_class, "mixed_magnitude") == 0 ||
           strcmp(input_class, "cancellation_stress") == 0;
}

static int parse_positive_blaslong(const char *text, BLASLONG *value)
{
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0) {
        return 0;
    }

    *value = (BLASLONG)parsed;
    return 1;
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

#if !defined(ACC_KIND_INT8_IME)
static void fill_i8_cancellation_packed(int8_t *x, BLASLONG outer, BLASLONG K,
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
#elif defined(ACC_IME_INPUT_FULL_MATRIX)
static void fill_i8_cancellation_full_matrix(int8_t *x, BLASLONG outer,
                                             BLASLONG K, int is_a)
{
    for (BLASLONG k = 0; k + 1 < K; k += 2) {
        for (BLASLONG lane = 0; lane < outer; ++lane) {
            int value = sign_value() * uniform_int_range(1, 16);
            int paired = is_a ? value : -value;

            if (!is_a && k + 2 == K) {
                paired += paired < 0 ? 1 : -1;
            }

            x[(size_t)k * (size_t)outer + (size_t)lane] = (int8_t)value;
            x[(size_t)(k + 1) * (size_t)outer + (size_t)lane] = (int8_t)paired;
        }
    }

    if (K & 1) {
        for (BLASLONG lane = 0; lane < outer; ++lane) {
            x[(size_t)(K - 1) * (size_t)outer + (size_t)lane] =
                int8_random_value("bounded_uniform");
        }
    }
}
#endif

static void fill_i8_inputs(int8_t *A, int8_t *B, BLASLONG M, BLASLONG N,
                           BLASLONG K, const char *input_class)
{
    if (strcmp(input_class, "cancellation_stress") == 0) {
#if defined(ACC_IME_INPUT_FULL_MATRIX)
        fill_i8_cancellation_full_matrix(A, M, K, 1);
        fill_i8_cancellation_full_matrix(B, N, K, 0);
#else
        fill_i8_cancellation_packed(A, M, K, ACC_MR, 1);
        fill_i8_cancellation_packed(B, N, K, ACC_NR, 0);
#endif
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

/*
 * Compute the trusted matrix answer independently in INT64.
 * The final value is stored as INT32 only after overflow has been checked.
 */
static void trusted_answer_i32(BLASLONG M, BLASLONG N, BLASLONG K,
                               int32_t alpha, const int8_t *A, const int8_t *B,
                               int32_t *Ctrusted, int *overflow_count, BLASLONG ldc)
{
    BLASLONG n_top = 0;

    while (n_top < N) {
        BLASLONG cols = tile_step(N - n_top, ACC_NR);
        BLASLONG m_top = 0;

        while (m_top < M) {
            BLASLONG rows = tile_step(M - m_top, ACC_MR);
#if !defined(ACC_KIND_INT8_IME)
            const int8_t *Ablk = &A[(size_t)m_top * (size_t)K];
            const int8_t *Bblk = &B[(size_t)n_top * (size_t)K];
#endif

            for (BLASLONG c = 0; c < cols; ++c) {
                for (BLASLONG r = 0; r < rows; ++r) {
                    int64_t sum = 0;

                    for (BLASLONG k = 0; k < K; ++k) {
#if defined(ACC_IME_INPUT_FULL_MATRIX)
                        sum += (int64_t)A[(size_t)k * (size_t)M + (size_t)(m_top + r)] *
                               (int64_t)B[(size_t)k * (size_t)N + (size_t)(n_top + c)];
#else
                        sum += (int64_t)Ablk[(size_t)k * (size_t)rows + (size_t)r] *
                               (int64_t)Bblk[(size_t)k * (size_t)cols + (size_t)c];
#endif
                    }

                    sum *= (int64_t)alpha;
                    if (sum > INT32_MAX || sum < INT32_MIN) {
                        *overflow_count += 1;
                    }

                    Ctrusted[(size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r)] =
                        wrap_i32_add(Ctrusted[(size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r)], sum);
                }
            }

            m_top += rows;
        }

        n_top += cols;
    }
}

static int compare_i64(const void *lhs, const void *rhs)
{
    int64_t a = *(const int64_t *)lhs;
    int64_t b = *(const int64_t *)rhs;

    return (a > b) - (a < b);
}

static int write_diff_histogram(const char *path, const int64_t *diffs, size_t count)
{
    FILE *fp = fopen(path, "w");
    size_t i = 0;

    if (!fp) {
        fprintf(stderr, "failed to open diff histogram '%s': %s\n", path, strerror(errno));
        return 1;
    }

    fprintf(fp, "diff_value,count\n");
    while (i < count) {
        int64_t value = diffs[i];
        size_t run = 1;

        while (i + run < count && diffs[i + run] == value) {
            run += 1;
        }

        fprintf(fp, "%" PRId64 ",%zu\n", value, run);
        i += run;
    }

    fclose(fp);
    return 0;
}

static int write_input_histogram(const char *path, const int8_t *A, size_t count_a,
                                 const int8_t *B, size_t count_b)
{
    uint64_t count_a_values[256] = {0};
    uint64_t count_b_values[256] = {0};
    FILE *fp;

    for (size_t i = 0; i < count_a; ++i) {
        count_a_values[(uint8_t)A[i]] += 1;
    }
    for (size_t i = 0; i < count_b; ++i) {
        count_b_values[(uint8_t)B[i]] += 1;
    }

    fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "failed to open input histogram '%s': %s\n", path, strerror(errno));
        return 1;
    }

    fprintf(fp, "int8_value,count_A,count_B\n");
    for (int value = -128; value <= 127; ++value) {
        uint8_t idx = (uint8_t)((int8_t)value);
        fprintf(fp, "%d,%" PRIu64 ",%" PRIu64 "\n",
                value, count_a_values[idx], count_b_values[idx]);
    }

    fclose(fp);
    return 0;
}

static int run_check(const char *input_class, BLASLONG M, BLASLONG N, BLASLONG K,
                     const char *diff_hist_path, const char *input_hist_path)
{
    size_t size_a;
    size_t size_b;
    size_t size_c;
    size_t bytes_c_i32;
    size_t bytes_c_i64;

    if (!size_mul_ok((size_t)M, (size_t)K, &size_a) ||
        !size_mul_ok((size_t)N, (size_t)K, &size_b) ||
        !size_mul_ok((size_t)M, (size_t)N, &size_c) ||
        !size_mul_ok(size_c, sizeof(int32_t), &bytes_c_i32) ||
        !size_mul_ok(size_c, sizeof(int64_t), &bytes_c_i64)) {
        fprintf(stderr, "matrix size is too large\n");
        return 1;
    }

    int8_t *A = aligned_bytes(size_a * sizeof(int8_t));
    int8_t *B = aligned_bytes(size_b * sizeof(int8_t));
    int32_t *C = aligned_bytes(bytes_c_i32);
    int32_t *Ctrusted = aligned_bytes(bytes_c_i32);
    int64_t *diffs = aligned_bytes(bytes_c_i64);
    uint64_t mismatch_count = 0;
    int64_t max_abs_diff = 0;
    int overflow_count = 0;
    int rc;

    if (!A || !B || !C || !Ctrusted || !diffs) {
        fprintf(stderr, "allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(Ctrusted);
        free(diffs);
        return 1;
    }

    fill_i8_inputs(A, B, M, N, K, input_class);
    if (input_hist_path && write_input_histogram(input_hist_path, A, size_a, B, size_b) != 0) {
        free(A);
        free(B);
        free(C);
        free(Ctrusted);
        free(diffs);
        return 1;
    }

    trusted_answer_i32(M, N, K, 1, A, B, Ctrusted, &overflow_count, M);

    rc = KERNEL_SYMBOL(M, N, K, 1, A, B, C, M);
    if (rc != 0) {
        printf("status,return_code,total_elements,mismatch_count,max_integer_difference,overflow_count,exact_match_rate,diff_histogram,input_histogram\n");
        printf("KERNEL_RETURN,%d,%zu,NA,NA,%d,NA,%s,%s\n",
               rc, size_c, overflow_count, diff_hist_path,
               input_hist_path ? input_hist_path : "NA");
        free(A);
        free(B);
        free(C);
        free(Ctrusted);
        free(diffs);
        return 0;
    }

    for (size_t i = 0; i < size_c; ++i) {
        int64_t diff = (int64_t)C[i] - (int64_t)Ctrusted[i];
        int64_t abs_diff = diff < 0 ? -diff : diff;

        diffs[i] = diff;
        if (diff != 0) {
            mismatch_count += 1;
        }
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
    }

    qsort(diffs, size_c, sizeof(*diffs), compare_i64);
    if (write_diff_histogram(diff_hist_path, diffs, size_c) != 0) {
        free(A);
        free(B);
        free(C);
        free(Ctrusted);
        free(diffs);
        return 1;
    }

    printf("status,return_code,total_elements,mismatch_count,max_integer_difference,overflow_count,exact_match_rate,diff_histogram,input_histogram\n");
    printf("%s,%d,%zu,%" PRIu64 ",%" PRId64 ",%d,%.12f,%s,%s\n",
           mismatch_count == 0 && max_abs_diff == 0 && overflow_count == 0 ? "OK" : "NUMERICAL_FAILED",
           rc, size_c, mismatch_count, max_abs_diff, overflow_count,
           size_c ? (double)(size_c - mismatch_count) / (double)size_c : 0.0,
           diff_hist_path, input_hist_path ? input_hist_path : "NA");

    free(A);
    free(B);
    free(C);
    free(Ctrusted);
    free(diffs);
    return 0;
}

int main(int argc, char **argv)
{
    char *seed_end = NULL;
    BLASLONG M;
    BLASLONG N;
    BLASLONG K;

    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "Usage: %s input_class M N K seed diff_histogram_csv [input_histogram_csv]\n",
                argv[0]);
        return 2;
    }

    const char *input_class = argv[1];
    if (!parse_positive_blaslong(argv[2], &M) ||
        !parse_positive_blaslong(argv[3], &N) ||
        !parse_positive_blaslong(argv[4], &K)) {
        fprintf(stderr, "M, N, and K must be positive integers\n");
        return 2;
    }

    errno = 0;
    uint64_t seed = strtoull(argv[5], &seed_end, 10);
    const char *diff_hist_path = argv[6];
    const char *input_hist_path = argc == 8 ? argv[7] : NULL;

    if (!valid_input_class(input_class)) {
        fprintf(stderr, "unsupported input class: %s\n", input_class);
        return 2;
    }
    if (errno != 0 || argv[5][0] == '-' || !seed_end ||
        seed_end == argv[5] || *seed_end != '\0') {
        fprintf(stderr, "seed must be an unsigned integer\n");
        return 2;
    }

    rng_state = seed ^ 0x9e3779b97f4a7c15ULL;
    rng_state ^= (uint64_t)M * 0x100000001b3ULL;
    rng_state ^= (uint64_t)N * 0xbf58476d1ce4e5b9ULL;
    rng_state ^= (uint64_t)K * 0x94d049bb133111ebULL;

    return run_check(input_class, M, N, K, diff_hist_path, input_hist_path);
}
