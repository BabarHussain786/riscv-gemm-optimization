/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 *
 * SpacemiT IME INT8 GEMM microkernel: ime_kernel_8x8_zvl128b_lmulmf2_unroll2
 *
 * Mathematical operation:
 *   C += alpha * (A x B)
 *   A and B contain signed INT8 values. C contains INT32 values.
 *
 * Caller layout:
 *   A[k * M + row], B[k * N + column], C[column * ldc + row]
 *
 * Private native layouts:
 *   K1 / A60:  C(4x4) += A(4x8)  x B(8x4)
 *   K3 / A100: C(8x8) += A(8x16) x B(16x8)
 *
 * Software tile:
 *   8x8. The driver composes complete native hardware tiles internally.
 *
 * Experimental LMUL note:
 *   Current public SpacemiT documentation reserves non-1 IME LMUL values.
 *   This mf2 variant remains available for explicit research-board testing,
 *   but normal builds use its local RVV fallback unless
 *   SPACEMIT_IME_ENABLE_MF2_NATIVE is defined.
 *
 * Portability:
 *   The matching rvv_fallback.c remains local to this kernel folder.
 *   SPACEMIT_IME_REQUIRE_HARDWARE disables fallback during accuracy testing.
 */
#define _GNU_SOURCE

#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(__riscv) && defined(__riscv_vector)
#define SPACEMIT_IME_HAS_RVV 1
#else
#define SPACEMIT_IME_HAS_RVV 0
#endif

#if defined(SPACEMIT_IME_ENABLE_MF2_NATIVE)
#define SPACEMIT_IME_NATIVE_ENABLED 1
#else
#define SPACEMIT_IME_NATIVE_ENABLED 0
#endif

typedef long BLASLONG;

#define UNROLL_K 2

enum {
    IME_MR = 8,
    IME_NR = 8,
    IME_ALIGNMENT = 64,
    IME_MAX_OUTPUT_VALUES = 64
};

typedef enum ime_profile {
    IME_PROFILE_NONE = 0,
    IME_PROFILE_A60,
    IME_PROFILE_A100
} ime_profile;

int igemm_kernel_8x8_zvl128b_lmulmf2_unroll2_i8i32(BLASLONG M, BLASLONG N, BLASLONG K,
                      int32_t alpha, int8_t *A, int8_t *B,
                      int32_t *C, BLASLONG ldc);

static uint32_t u32_from_i32(int32_t x)
{
    uint32_t y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static int32_t i32_from_u32(uint32_t x)
{
    int32_t y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static int32_t add_scaled_wrap_i32(int32_t dst, int64_t value, int32_t alpha)
{
    uint32_t scaled = (uint32_t)((int64_t)alpha * value);
    return i32_from_u32(u32_from_i32(dst) + scaled);
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
    size_t padded;

    if (bytes == 0 || bytes > SIZE_MAX - (IME_ALIGNMENT - 1)) return NULL;
    padded = (bytes + IME_ALIGNMENT - 1) & ~(size_t)(IME_ALIGNMENT - 1);
    if (posix_memalign(&ptr, IME_ALIGNMENT, padded) != 0) return NULL;
    return ptr;
}

static BLASLONG pick_row_block(BLASLONG left)
{
    if (left >= 8) return 8;
    if (left >= 4) return 4;
    if (left >= 2) return 2;
    return 1;
}

static BLASLONG pick_col_block(BLASLONG left)
{
    if (left >= IME_NR) return IME_NR;
    if (left >= 4) return 4;
    if (left >= 2) return 2;
    return 1;
}

static void scalar_gemm_block(BLASLONG rows, BLASLONG cols, BLASLONG K,
                              int32_t alpha, const int8_t *A, BLASLONG lda,
                              const int8_t *B, BLASLONG ldb,
                              int32_t *C, BLASLONG ldc)
{
    for (BLASLONG c = 0; c < cols; ++c) {
        for (BLASLONG r = 0; r < rows; ++r) {
            int64_t sum = 0;

#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                sum += (int32_t)A[k * lda + r] * (int32_t)B[k * ldb + c];
            }

            C[c * ldc + r] = add_scaled_wrap_i32(C[c * ldc + r], sum, alpha);
        }
    }
}

static int scalar_gemm_full(BLASLONG M, BLASLONG N, BLASLONG K,
                            int32_t alpha, const int8_t *A, const int8_t *B,
                            int32_t *C, BLASLONG ldc)
{
    scalar_gemm_block(M, N, K, alpha, A, M, B, N, C, ldc);
    return 0;
}

static int rvv_fallback_gemm(BLASLONG M, BLASLONG N, BLASLONG K,
                             int32_t alpha, const int8_t *A, const int8_t *B,
                             int32_t *C, BLASLONG ldc)
{
#if defined(SPACEMIT_IME_REQUIRE_HARDWARE)
    (void)M; (void)N; (void)K; (void)alpha;
    (void)A; (void)B; (void)C; (void)ldc;
    return -99;
#else
    size_t a_bytes;
    size_t b_bytes;
    int8_t *A_pack;
    int8_t *B_pack;
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;
    int rc;

    if (!size_mul_ok((size_t)M, (size_t)K, &a_bytes) ||
        !size_mul_ok((size_t)N, (size_t)K, &b_bytes)) {
        return scalar_gemm_full(M, N, K, alpha, A, B, C, ldc);
    }

    A_pack = (int8_t *)aligned_alloc_bytes(a_bytes);
    B_pack = (int8_t *)aligned_alloc_bytes(b_bytes);
    if (A_pack == NULL || B_pack == NULL) {
        free(A_pack);
        free(B_pack);
        return scalar_gemm_full(M, N, K, alpha, A, B, C, ldc);
    }

    while (m_top < M) {
        BLASLONG rows = pick_row_block(M - m_top);
        for (BLASLONG k = 0; k < K; ++k) {
            for (BLASLONG r = 0; r < rows; ++r) {
                A_pack[m_top * K + k * rows + r] = A[k * M + m_top + r];
            }
        }
        m_top += rows;
    }

    while (n_top < N) {
        BLASLONG cols = pick_col_block(N - n_top);
        for (BLASLONG k = 0; k < K; ++k) {
            for (BLASLONG c = 0; c < cols; ++c) {
                B_pack[n_top * K + k * cols + c] = B[k * N + n_top + c];
            }
        }
        n_top += cols;
    }

    rc = igemm_kernel_8x8_zvl128b_lmulmf2_unroll2_i8i32(M, N, K, alpha, A_pack, B_pack, C, ldc);
    free(A_pack);
    free(B_pack);
    if (rc == 0) return 0;
    return scalar_gemm_full(M, N, K, alpha, A, B, C, ldc);
#endif
}

typedef struct ime_affinity_guard {
    cpu_set_t saved_mask;
    int active;
} ime_affinity_guard;

static int cpu_has_ime(int cpu)
{
    char marker[128];
    struct stat st;

    if (cpu < 0) return 0;
    snprintf(marker, sizeof(marker),
             "/sys/firmware/devicetree/base/cpus/cpu@%d/cpu-ai", cpu);
    if (stat(marker, &st) == 0) return 1;
    snprintf(marker, sizeof(marker),
             "/sys/firmware/devicetree/base/cpus/cpu@%x/cpu-ai", cpu);
    return stat(marker, &st) == 0;
}

static int ime_affinity_guard_enter(ime_affinity_guard *guard)
{
#if SPACEMIT_IME_HAS_RVV
    cpu_set_t target_mask;
    int target = -1;
    int cpu;

    memset(guard, 0, sizeof(*guard));
    if (sched_getaffinity(0, sizeof(guard->saved_mask), &guard->saved_mask) != 0) {
        return 0;
    }

    cpu = sched_getcpu();
    if (cpu >= 0 && CPU_ISSET(cpu, &guard->saved_mask) && cpu_has_ime(cpu)) {
        target = cpu;
    } else {
        for (cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &guard->saved_mask) && cpu_has_ime(cpu)) {
                target = cpu;
                break;
            }
        }
    }

    if (target < 0) return 0;
    CPU_ZERO(&target_mask);
    CPU_SET(target, &target_mask);
    if (sched_setaffinity(0, sizeof(target_mask), &target_mask) != 0) return 0;
    if (sched_getcpu() != target) {
        (void)sched_setaffinity(0, sizeof(guard->saved_mask), &guard->saved_mask);
        return 0;
    }

    guard->active = 1;
    return 1;
#else
    (void)guard;
    return 0;
#endif
}

static void ime_affinity_guard_leave(ime_affinity_guard *guard)
{
    if (guard->active) {
        (void)sched_setaffinity(0, sizeof(guard->saved_mask), &guard->saved_mask);
    }
}

static ime_profile detect_ime_profile(void)
{
#if SPACEMIT_IME_HAS_RVV && SPACEMIT_IME_NATIVE_ENABLED
    size_t vlmax;

    __asm__ __volatile__(
        "li t0,-1\n\t"
        "vsetvli %0,t0,e8,m1,ta,ma\n\t"
        : "=r"(vlmax)
        :
        : "t0");

    if (vlmax >= 128) return IME_PROFILE_A100;
    if (vlmax >= 32) return IME_PROFILE_A60;
#endif
    return IME_PROFILE_NONE;
}

static BLASLONG profile_kr(ime_profile profile)
{
    return profile == IME_PROFILE_A100 ? 8 : 4;
}

static BLASLONG profile_native_rows(ime_profile profile)
{
    return profile == IME_PROFILE_A100 ? 8 : 4;
}

static BLASLONG profile_native_cols(ime_profile profile)
{
    return profile_native_rows(profile);
}

static size_t b_panel_bytes(ime_profile profile, BLASLONG K_main)
{
    BLASLONG cols = profile == IME_PROFILE_A100 ? 8 : IME_NR;
    return (size_t)cols * (size_t)K_main;
}

static void pack_a_panel(ime_profile profile, const int8_t *A,
                         BLASLONG m_top, BLASLONG M, BLASLONG K_main,
                         int8_t *dst)
{
    BLASLONG kr = profile_kr(profile);
    BLASLONG native_rows = profile_native_rows(profile);

    for (BLASLONG kb = 0; kb < K_main; kb += kr) {
        for (BLASLONG rb = 0; rb < IME_MR; rb += native_rows) {
            for (BLASLONG r = 0; r < native_rows; ++r) {
                for (BLASLONG kk = 0; kk < kr; ++kk) {
                    *dst++ = A[(kb + kk) * M + m_top + rb + r];
                }
            }
        }
    }
}

static void pack_b_panel(ime_profile profile, const int8_t *B,
                         BLASLONG n_top, BLASLONG N, BLASLONG K_main,
                         int8_t *dst)
{
    BLASLONG kr = profile_kr(profile);
    BLASLONG native_cols = profile_native_cols(profile);
    BLASLONG packed_cols = profile == IME_PROFILE_A100 ? 8 : IME_NR;

    for (BLASLONG kb = 0; kb < K_main; kb += kr) {
        for (BLASLONG cb = 0; cb < packed_cols; cb += native_cols) {
            for (BLASLONG c = 0; c < native_cols; ++c) {
                for (BLASLONG kk = 0; kk < kr; ++kk) {
                    BLASLONG column = cb + c;
                    *dst++ = column < IME_NR
                        ? B[(kb + kk) * N + n_top + column]
                        : 0;
                }
            }
        }
    }
}

#if SPACEMIT_IME_HAS_RVV && SPACEMIT_IME_NATIVE_ENABLED
#define IME_A60_DOT_STEP \
        "vle8.v v16,(t2)\n\t" \
        "addi t2,t2,16\n\t" \
        "vle8.v v17,(t2)\n\t" \
        "addi t2,t2,16\n\t" \
        "vle8.v v20,(t3)\n\t" \
        "addi t3,t3,16\n\t" \
        "vle8.v v21,(t3)\n\t" \
        "addi t3,t3,16\n\t" \
        ".word 0xE348342Bu\n\t" \
        ".word 0xE348B52Bu\n\t" \
        ".word 0xE358362Bu\n\t" \
        ".word 0xE358B72Bu\n\t"
#define IME_A100_DOT_STEP \
        "vle8.v v16,(t2)\n\t" \
        "addi t2,t2,64\n\t" \
        "vle8.v v20,(t3)\n\t" \
        "addi t3,t3,64\n\t" \
        ".word 0xE348342Bu\n\t"

#if UNROLL_K == 1
#define IME_A60_DOT_UNROLL IME_A60_DOT_STEP
#elif UNROLL_K == 2
#define IME_A60_DOT_UNROLL IME_A60_DOT_STEP IME_A60_DOT_STEP
#elif UNROLL_K == 4
#define IME_A60_DOT_UNROLL IME_A60_DOT_STEP IME_A60_DOT_STEP IME_A60_DOT_STEP IME_A60_DOT_STEP
#elif UNROLL_K == 8
#define IME_A60_DOT_UNROLL IME_A60_DOT_STEP IME_A60_DOT_STEP IME_A60_DOT_STEP IME_A60_DOT_STEP \
                      IME_A60_DOT_STEP IME_A60_DOT_STEP IME_A60_DOT_STEP IME_A60_DOT_STEP
#else
#error "Unsupported IME unroll factor"
#endif
#if UNROLL_K == 1
#define IME_A100_DOT_UNROLL IME_A100_DOT_STEP
#elif UNROLL_K == 2
#define IME_A100_DOT_UNROLL IME_A100_DOT_STEP IME_A100_DOT_STEP
#elif UNROLL_K == 4
#define IME_A100_DOT_UNROLL IME_A100_DOT_STEP IME_A100_DOT_STEP IME_A100_DOT_STEP IME_A100_DOT_STEP
#elif UNROLL_K == 8
#define IME_A100_DOT_UNROLL IME_A100_DOT_STEP IME_A100_DOT_STEP IME_A100_DOT_STEP IME_A100_DOT_STEP \
                      IME_A100_DOT_STEP IME_A100_DOT_STEP IME_A100_DOT_STEP IME_A100_DOT_STEP
#else
#error "Unsupported IME unroll factor"
#endif

static void ime_accumulate_a60(const int8_t *A_pack, const int8_t *B_pack,
                              BLASLONG k_groups,
                              int32_t out[IME_MAX_OUTPUT_VALUES])
{
    __asm__ __volatile__(
        "li t0,16\n\t"
        "vsetvli zero,t0,e32,m2,ta,ma\n\t"
        "vmv.v.i v8,0\n\t"
        "vmv.v.i v10,0\n\t"
        "vmv.v.i v12,0\n\t"
        "vmv.v.i v14,0\n\t"
        "li t0,16\n\t"
        "vsetvli zero,t0,e8,mf2,ta,ma\n\t"
        "mv t1,%[kgroups]\n\t"
        "mv t2,%[aptr]\n\t"
        "mv t3,%[bptr]\n\t"
        "beqz t1,2f\n\t"
        ".balign 16\n"
        "1:\n\t"
        IME_A60_DOT_UNROLL
        "addi t1,t1,-1\n\t"
        "bnez t1,1b\n\t"
        "2:\n\t"
        "li t0,16\n\t"
        "vsetvli zero,t0,e32,m2,ta,ma\n\t"
        "mv t4,%[out]\n\t"
        "vse32.v v8,(t4)\n\t"
        "addi t4,t4,64\n\t"
        "vse32.v v10,(t4)\n\t"
        "addi t4,t4,64\n\t"
        "vse32.v v12,(t4)\n\t"
        "addi t4,t4,64\n\t"
        "vse32.v v14,(t4)\n\t"
        :
        : [kgroups] "r"(k_groups),
          [aptr] "r"(A_pack),
          [bptr] "r"(B_pack),
          [out] "r"(out)
        : "t0", "t1", "t2", "t3", "t4", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v20", "v21");
}

static void ime_accumulate_a100(const int8_t *A_pack, const int8_t *B_pack,
                              BLASLONG k_groups,
                              int32_t out[IME_MAX_OUTPUT_VALUES])
{
    __asm__ __volatile__(
        "li t0,64\n\t"
        "vsetvli zero,t0,e32,m2,ta,ma\n\t"
        "vmv.v.i v8,0\n\t"
        "li t0,64\n\t"
        "vsetvli zero,t0,e8,mf2,ta,ma\n\t"
        "mv t1,%[kgroups]\n\t"
        "mv t2,%[aptr]\n\t"
        "mv t3,%[bptr]\n\t"
        "beqz t1,2f\n\t"
        ".balign 16\n"
        "1:\n\t"
        IME_A100_DOT_UNROLL
        "addi t1,t1,-1\n\t"
        "bnez t1,1b\n\t"
        "2:\n\t"
        "li t0,64\n\t"
        "vsetvli zero,t0,e32,m2,ta,ma\n\t"
        "mv t4,%[out]\n\t"
        "vse32.v v8,(t4)\n\t"
        :
        : [kgroups] "r"(k_groups),
          [aptr] "r"(A_pack),
          [bptr] "r"(B_pack),
          [out] "r"(out)
        : "t0", "t1", "t2", "t3", "t4", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v20", "v21");
}
#endif

static size_t output_index(ime_profile profile, int r, int c)
{
    if (profile == IME_PROFILE_A100) {
        return (size_t)r * 8 + (size_t)c;
    }

    return (size_t)(c / 4) * 32 +
           (size_t)(r / 4) * 16 +
           (size_t)(r % 4) * 4 +
           (size_t)(c % 4);
}

static void scatter_output(ime_profile profile, BLASLONG m_top, BLASLONG n_top,
                           int32_t alpha,
                           const int32_t out[IME_MAX_OUTPUT_VALUES],
                           int32_t *C, BLASLONG ldc)
{
    for (int c = 0; c < IME_NR; ++c) {
        for (int r = 0; r < IME_MR; ++r) {
            size_t ci = (size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r);
            C[ci] = add_scaled_wrap_i32(C[ci], out[output_index(profile, r, c)], alpha);
        }
    }
}

static void native_accumulate(ime_profile profile, const int8_t *A_pack,
                              const int8_t *B_pack, BLASLONG k_groups,
                              int32_t out[IME_MAX_OUTPUT_VALUES])
{
#if SPACEMIT_IME_HAS_RVV && SPACEMIT_IME_NATIVE_ENABLED
    if (profile == IME_PROFILE_A100) {
        ime_accumulate_a100(A_pack, B_pack, k_groups, out);
    } else {
        ime_accumulate_a60(A_pack, B_pack, k_groups, out);
    }
#else
    (void)profile; (void)A_pack; (void)B_pack; (void)k_groups; (void)out;
#endif
}

static int ime_gemm_s8s8(BLASLONG M, BLASLONG N, BLASLONG K,
                         int32_t alpha, const int8_t *A, const int8_t *B,
                         int32_t *C, BLASLONG ldc)
{
    ime_affinity_guard guard;
    ime_profile profile;
    BLASLONG kr;
    BLASLONG K_main;
    BLASLONG m_blocks;
    BLASLONG n_blocks;
    size_t a_panel_size;
    size_t b_panel_size;
    size_t a_all_size;
    int8_t *A_all = NULL;
    int8_t *B_pack = NULL;

    memset(&guard, 0, sizeof(guard));
    if (M <= 0 || N <= 0 || K <= 0 || alpha == 0) return 0;
    if (A == NULL || B == NULL || C == NULL || ldc < M) return -1;
    if (!SPACEMIT_IME_NATIVE_ENABLED || !ime_affinity_guard_enter(&guard)) {
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    profile = detect_ime_profile();
    if (profile == IME_PROFILE_NONE) {
        ime_affinity_guard_leave(&guard);
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    kr = profile_kr(profile);
    K_main = (K / (kr * UNROLL_K)) * (kr * UNROLL_K);
    m_blocks = M / IME_MR;
    n_blocks = N / IME_NR;

    if (K_main == 0 || m_blocks == 0 || n_blocks == 0 ||
        !size_mul_ok((size_t)IME_MR, (size_t)K_main, &a_panel_size) ||
        !size_mul_ok((size_t)m_blocks, a_panel_size, &a_all_size)) {
        ime_affinity_guard_leave(&guard);
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    b_panel_size = b_panel_bytes(profile, K_main);
    A_all = (int8_t *)aligned_alloc_bytes(a_all_size);
    B_pack = (int8_t *)aligned_alloc_bytes(b_panel_size);
    if (A_all == NULL || B_pack == NULL) {
        free(A_all);
        free(B_pack);
        ime_affinity_guard_leave(&guard);
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    for (BLASLONG mb = 0; mb < m_blocks; ++mb) {
        pack_a_panel(profile, A, mb * IME_MR, M, K_main,
                     &A_all[(size_t)mb * a_panel_size]);
    }

    for (BLASLONG nb = 0; nb < n_blocks; ++nb) {
        BLASLONG n_top = nb * IME_NR;
        pack_b_panel(profile, B, n_top, N, K_main, B_pack);

        for (BLASLONG mb = 0; mb < m_blocks; ++mb) {
            BLASLONG m_top = mb * IME_MR;
            int32_t out[IME_MAX_OUTPUT_VALUES] __attribute__((aligned(64)));

            native_accumulate(profile, &A_all[(size_t)mb * a_panel_size],
                              B_pack, K_main / (kr * UNROLL_K), out);
            scatter_output(profile, m_top, n_top, alpha, out, C, ldc);

            if (K_main != K) {
                scalar_gemm_block(IME_MR, IME_NR, K - K_main, alpha,
                                  &A[K_main * M + m_top], M,
                                  &B[K_main * N + n_top], N,
                                  &C[n_top * ldc + m_top], ldc);
            }
        }

        if (m_blocks * IME_MR != M) {
            BLASLONG m_top = m_blocks * IME_MR;
            scalar_gemm_block(M - m_top, IME_NR, K, alpha,
                              &A[m_top], M, &B[n_top], N,
                              &C[n_top * ldc + m_top], ldc);
        }
    }

    if (n_blocks * IME_NR != N) {
        BLASLONG n_top = n_blocks * IME_NR;
        scalar_gemm_block(M, N - n_top, K, alpha, A, M, &B[n_top], N,
                          &C[n_top * ldc], ldc);
    }

    free(A_all);
    free(B_pack);
    ime_affinity_guard_leave(&guard);
    return 0;
}

int ime_kernel_8x8_zvl128b_lmulmf2_unroll2(BLASLONG M, BLASLONG N, BLASLONG K,
              int32_t alpha, const int8_t *A, const int8_t *B,
              int32_t *C, BLASLONG ldc)
{
    return ime_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
}

#if SPACEMIT_IME_HAS_RVV && SPACEMIT_IME_NATIVE_ENABLED
#undef IME_A100_DOT_UNROLL
#undef IME_A60_DOT_UNROLL
#undef IME_A100_DOT_STEP
#undef IME_A60_DOT_STEP
#endif
