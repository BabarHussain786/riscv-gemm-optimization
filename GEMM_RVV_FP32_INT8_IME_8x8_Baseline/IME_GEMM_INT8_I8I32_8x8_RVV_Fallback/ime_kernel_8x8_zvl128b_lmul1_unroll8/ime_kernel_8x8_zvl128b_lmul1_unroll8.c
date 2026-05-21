/*
 * Copyright (c) 2026 University of Salerno 
 *
 * Created by Babar Hussain.
 */

/*
RVV IME KERNEL
Variant: ime_kernel_8x8_zvl128b_lmul1_unroll8
Settings:
 LMUL=1 (m1)
 M=8
 N=8
 precision=int8 x int8 -> int32 (IME)
 unroll_factor=8 (K unroll x8)
 Kernel Name: ime_kernel_8x8_zvl128b_lmul1_unroll8
*/
#define _GNU_SOURCE

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#include <sched.h>
#include <sys/stat.h>
#endif

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#define SPACEMIT_IME_HAS_RVV 1
#else
#define SPACEMIT_IME_HAS_RVV 0
#endif

typedef long BLASLONG;

int igemm_kernel_8x8_zvl128b_lmul1_unroll8_i8i32(
    BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha,
    int8_t *A, int8_t *B, int32_t *C, BLASLONG ldc);
#define UNROLL_K 8

enum {
    IME_MR = 8,
    IME_NR = 8,
    IME_KR = 8
};

#define VMADOT_R01 0xE348342Bu
#define VMADOT_R23 0xE348B52Bu
#define VMADOT_R45 0xE349362Bu
#define VMADOT_R67 0xE349B72Bu
#define _S(x) #x
#define S(x) _S(x)

typedef struct ime_affinity_guard {
#if defined(__linux__)
    cpu_set_t saved_mask;
#endif
    int active;
} ime_affinity_guard;

static uint32_t u32_from_i32(int32_t x) {
    uint32_t y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static int32_t i32_from_u32(uint32_t x) {
    int32_t y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static int32_t add_wrap_i32(int32_t dst, int32_t addend) {
    uint32_t sum = u32_from_i32(dst) + u32_from_i32(addend);
    return i32_from_u32(sum);
}

static int32_t add_scaled_wrap_i32(int32_t dst, int64_t acc, int32_t alpha) {
    uint32_t scaled = (uint32_t)((int64_t)alpha * acc);
    uint32_t sum = u32_from_i32(dst) + scaled;
    return i32_from_u32(sum);
}

#if defined(__linux__)
static int cpu_has_ime(int cpu) {
    char path[128];
    struct stat st;

    if (cpu < 0) return 0;

    snprintf(path, sizeof(path),
             "/sys/firmware/devicetree/base/cpus/cpu@%d/cpu-ai", cpu);
    if (stat(path, &st) == 0) return 1;

    snprintf(path, sizeof(path),
             "/sys/firmware/devicetree/base/cpus/cpu@%x/cpu-ai", cpu);
    return stat(path, &st) == 0;
}
#else
static int cpu_has_ime(int cpu) {
    (void)cpu;
    return 0;
}
#endif

static int ime_affinity_guard_enter(ime_affinity_guard *guard) {
#if defined(__linux__) && SPACEMIT_IME_HAS_RVV
    cpu_set_t target_mask;
    int cpu;
    int target = -1;

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
    if (sched_setaffinity(0, sizeof(target_mask), &target_mask) != 0) {
        return 0;
    }

    cpu = sched_getcpu();
    if (cpu != target || !cpu_has_ime(cpu)) {
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

static void ime_affinity_guard_leave(ime_affinity_guard *guard) {
#if defined(__linux__) && SPACEMIT_IME_HAS_RVV
    if (guard->active) {
        (void)sched_setaffinity(0, sizeof(guard->saved_mask), &guard->saved_mask);
    }
#else
    (void)guard;
#endif
}

static inline const int8_t *pack_A_panel(const int8_t *A, BLASLONG m_top, BLASLONG K) {
    return &A[m_top * K];
}

static inline const int8_t *pack_B_panel4(const int8_t *B, BLASLONG n_top, BLASLONG K) {
    return &B[n_top * K];
}

static void scalar_gemm_block_range(BLASLONG rows, BLASLONG cols,
                                    BLASLONG K, BLASLONG k_begin, BLASLONG k_end,
                                    int32_t alpha,
                                    const int8_t *A_blk, const int8_t *B_blk,
                                    int32_t *C_blk, BLASLONG ldc) {
    if (k_begin >= k_end) return;

    for (BLASLONG c = 0; c < cols; ++c) {
        const int8_t *bcol = &B_blk[c * K + k_begin];

        for (BLASLONG r = 0; r < rows; ++r) {
            const int8_t *arow = &A_blk[r * K + k_begin];
            int64_t acc = 0;
            BLASLONG kc = k_end - k_begin;
#pragma GCC unroll 8
            for (BLASLONG k = 0; k < kc; ++k) {
                acc += (int32_t)arow[k] * (int32_t)bcol[k];
            }

            C_blk[c * ldc + r] = add_scaled_wrap_i32(C_blk[c * ldc + r], acc, alpha);
        }
    }
}


static int scalar_gemm_full_prepacked(BLASLONG M, BLASLONG N, BLASLONG K,
                                      int32_t alpha,
                                      const int8_t *A,
                                      const int8_t *B,
                                      int32_t *C,
                                      BLASLONG ldc) {
    BLASLONG n_top = 0;

    while (n_top < N) {
        const BLASLONG cols_left = N - n_top;
        const BLASLONG cols = (cols_left >= 8) ? 8 :
                              (cols_left >= 4) ? 4 :
                              (cols_left >= 2) ? 2 : 1;
        BLASLONG m_top = 0;

        while (m_top < M) {
            const BLASLONG rows_left = M - m_top;
            const BLASLONG rows = (rows_left >= 8) ? 8 :
                                  (rows_left >= 4) ? 4 :
                                  (rows_left >= 2) ? 2 : 1;

            scalar_gemm_block_range(rows, cols, K, 0, K, alpha,
                                    &A[m_top * K],
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += rows;
        }

        n_top += cols;
    }

    return 0;
}

static int rvv_fallback_gemm_s8s8(BLASLONG M, BLASLONG N, BLASLONG K,
                                  int32_t alpha,
                                  const int8_t *A,
                                  const int8_t *B,
                                  int32_t *C,
                                  BLASLONG ldc) {
    int rc = igemm_kernel_8x8_zvl128b_lmul1_unroll8_i8i32(M, N, K, alpha, (int8_t *)A, (int8_t *)B, C, ldc);
    if (rc == 0) return 0;
    return scalar_gemm_full_prepacked(M, N, K, alpha, A, B, C, ldc);
}
static int ime_vector_shape_supported(void) {
#if SPACEMIT_IME_HAS_RVV
    size_t vl32;
    size_t vl8_panel;
    size_t vl8_bpanel;

    __asm__ __volatile__(
        "li t0,8\n\t"
        "vsetvli %0,t0,e32,m1,ta,ma\n\t"
        "li t0,16\n\t"
        "vsetvli %1,t0,e8,m1,ta,ma\n\t"
        "li t0,32\n\t"
        "vsetvli %2,t0,e8,m1,ta,ma\n\t"
        : "=r"(vl32), "=r"(vl8_panel), "=r"(vl8_bpanel)
        :
        : "t0");

    return vl32 >= 8 && vl8_panel >= 16 && vl8_bpanel >= 32;
#else
    return 0;
#endif
}
#if SPACEMIT_IME_HAS_RVV
#define IME_8X8_DOT_STEP \
        "vle8.v v16,(t2)\n\t" \
        "addi t2,t2,16\n\t" \
        "vle8.v v17,(t2)\n\t" \
        "addi t2,t2,16\n\t" \
        "vle8.v v18,(t2)\n\t" \
        "addi t2,t2,16\n\t" \
        "vle8.v v19,(t2)\n\t" \
        "addi t2,t2,16\n\t" \
        "li t0,32\n\t" \
        "vsetvli zero,t0,e8,m1,ta,ma\n\t" \
        "vle8.v v20,(t3)\n\t" \
        "addi t3,t3,32\n\t" \
        ".word " S(VMADOT_R01) "\n\t" \
        ".word " S(VMADOT_R23) "\n\t" \
        ".word " S(VMADOT_R45) "\n\t" \
        ".word " S(VMADOT_R67) "\n\t" \
        "li t0,16\n\t" \
        "vsetvli zero,t0,e8,m1,ta,ma\n\t"

#if UNROLL_K == 1
#define IME_8X8_DOT_UNROLL IME_8X8_DOT_STEP
#elif UNROLL_K == 2
#define IME_8X8_DOT_UNROLL IME_8X8_DOT_STEP IME_8X8_DOT_STEP
#elif UNROLL_K == 4
#define IME_8X8_DOT_UNROLL IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP
#elif UNROLL_K == 8
#define IME_8X8_DOT_UNROLL IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP IME_8X8_DOT_STEP
#else
#error "Unsupported UNROLL_K for IME 8x8 kernel"
#endif

static void ime_kernel_8x8_acc(BLASLONG K_main, int32_t alpha,
                               const int8_t *A_pack, const int8_t *B_pack,
                               int32_t out0[8], int32_t out1[8],
                               int32_t out2[8], int32_t out3[8]) {
    const BLASLONG K_steps = K_main / IME_KR;
    const BLASLONG K_groups = K_steps / UNROLL_K;

    __asm__ __volatile__(
        "li t0,8\n\t"
        "vsetvli zero,t0,e32,m1,ta,ma\n\t"
        "vmv.v.i v8,0\n\t"
        "vmv.v.i v10,0\n\t"
        "vmv.v.i v12,0\n\t"
        "vmv.v.i v14,0\n\t"

        "li t0,16\n\t"
        "vsetvli zero,t0,e8,m1,ta,ma\n\t"
        "mv t1,%[kgroups]\n\t"
        "mv t2,%[aptr]\n\t"
        "mv t3,%[bptr]\n\t"
        "beqz t1,2f\n\t"

        ".balign 16\n"
        "1:\n\t"
        IME_8X8_DOT_UNROLL
        "addi t1,t1,-1\n\t"
        "bnez t1,1b\n\t"

        "2:\n\t"
        "li t0,8\n\t"
        "vsetvli zero,t0,e32,m1,ta,ma\n\t"
        "vmul.vx v8,v8,%[alpha]\n\t"
        "vmul.vx v10,v10,%[alpha]\n\t"
        "vmul.vx v12,v12,%[alpha]\n\t"
        "vmul.vx v14,v14,%[alpha]\n\t"

        "vse32.v v8,(%[o0])\n\t"
        "vse32.v v10,(%[o1])\n\t"
        "vse32.v v12,(%[o2])\n\t"
        "vse32.v v14,(%[o3])\n\t"

        :
        : [kgroups] "r"(K_groups),
          [aptr] "r"(A_pack),
          [bptr] "r"(B_pack),
          [alpha] "r"(alpha),
          [o0] "r"(out0),
          [o1] "r"(out1),
          [o2] "r"(out2),
          [o3] "r"(out3)
        : "t0", "t1", "t2", "t3", "memory",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20");
}
#undef IME_8X8_DOT_UNROLL
#undef IME_8X8_DOT_STEP
#endif


static void commit_8x4_unit_stride(int32_t *C, BLASLONG ldc,
                                   const int32_t out0[8], const int32_t out1[8],
                                   const int32_t out2[8], const int32_t out3[8]) {
    int32_t *c0 = &C[0 * ldc];
    int32_t *c1 = &C[1 * ldc];
    int32_t *c2 = &C[2 * ldc];
    int32_t *c3 = &C[3 * ldc];

    for (BLASLONG r = 0; r < 8; ++r) {
        c0[r] = add_wrap_i32(c0[r], out0[r]);
        c1[r] = add_wrap_i32(c1[r], out1[r]);
        c2[r] = add_wrap_i32(c2[r], out2[r]);
        c3[r] = add_wrap_i32(c3[r], out3[r]);
    }
}

static void run_block_8x4(BLASLONG K, BLASLONG K_main,
                          int32_t alpha,
                          const int8_t *A_blk,
                          const int8_t *B4_blk,
                          int32_t *C_blk,
                          BLASLONG ldc,
                          int use_ime) {
    if (use_ime && K_main > 0) {
#if SPACEMIT_IME_HAS_RVV
        int32_t out0[8], out1[8], out2[8], out3[8];
        ime_kernel_8x8_acc(K_main, alpha, A_blk, B4_blk, out0, out1, out2, out3);
        commit_8x4_unit_stride(C_blk, ldc, out0, out1, out2, out3);
#endif
        if (K_main != K) {
            scalar_gemm_block_range(8, 4, K, K_main, K, alpha, A_blk, B4_blk, C_blk, ldc);
        }
        return;
    }

    scalar_gemm_block_range(8, 4, K, 0, K, alpha, A_blk, B4_blk, C_blk, ldc);
}

static int ime_gemm_s8s8_prepacked(BLASLONG M, BLASLONG N, BLASLONG K,
                                   int32_t alpha,
                                   const int8_t *A,
                                   const int8_t *B,
                                   int32_t *C,
                                   BLASLONG ldc) {
    BLASLONG n_top = 0;
    const BLASLONG K_steps = K / IME_KR;
    const BLASLONG K_main = (K_steps / UNROLL_K) * UNROLL_K * IME_KR;
    ime_affinity_guard guard;
    int use_ime = 0;

    memset(&guard, 0, sizeof(guard));

    if (M <= 0 || N <= 0 || K <= 0 || alpha == 0) return 0;
    if (ldc < M) return -1;
    if (A == NULL || B == NULL || C == NULL) return -1;

    if (!SPACEMIT_IME_HAS_RVV || !ime_vector_shape_supported() || K_main == 0) {
        return rvv_fallback_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
    }

    use_ime = ime_affinity_guard_enter(&guard);
    if (!use_ime) {
        return rvv_fallback_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
    }

    /* -- MAIN PASS: N / 8 */
    for (BLASLONG j = 0; j < N / 8; ++j) {
        BLASLONG m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            const int8_t *A_blk = pack_A_panel(A, m_top, K);
            const int8_t *B4_0 = pack_B_panel4(B, n_top + 0, K);
            const int8_t *B4_1 = pack_B_panel4(B, n_top + 4, K);

            run_block_8x4(K, K_main, alpha, A_blk, B4_0,
                          &C[(n_top + 0) * ldc + m_top], ldc, use_ime);
            run_block_8x4(K, K_main, alpha, A_blk, B4_1,
                          &C[(n_top + 4) * ldc + m_top], ldc, use_ime);

            m_top += 8;
        }

        if (M & 4) {
            scalar_gemm_block_range(4, 8, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 4;
        }

        if (M & 2) {
            scalar_gemm_block_range(2, 8, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 2;
        }

        if (M & 1) {
            scalar_gemm_block_range(1, 8, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 1;
        }

        n_top += 8;
    }

    /* -- tails for N = 4 */
    if (N & 4) {
        BLASLONG m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            const int8_t *A_blk = pack_A_panel(A, m_top, K);
            const int8_t *B4_0 = pack_B_panel4(B, n_top + 0, K);

            run_block_8x4(K, K_main, alpha, A_blk, B4_0,
                          &C[(n_top + 0) * ldc + m_top], ldc, use_ime);

            m_top += 8;
        }

        if (M & 4) {
            scalar_gemm_block_range(4, 4, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 4;
        }

        if (M & 2) {
            scalar_gemm_block_range(2, 4, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 2;
        }

        if (M & 1) {
            scalar_gemm_block_range(1, 4, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 1;
        }

        n_top += 4;
    }

    /* -- tails for N = 2 */
    if (N & 2) {
        BLASLONG m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            scalar_gemm_block_range(8, 2, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 8;
        }

        if (M & 4) {
            scalar_gemm_block_range(4, 2, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 4;
        }

        if (M & 2) {
            scalar_gemm_block_range(2, 2, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 2;
        }

        if (M & 1) {
            scalar_gemm_block_range(1, 2, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 1;
        }

        n_top += 2;
    }

    /* -- tails for N = 1 */
    if (N & 1) {
        BLASLONG m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            scalar_gemm_block_range(8, 1, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 8;
        }

        if (M & 4) {
            scalar_gemm_block_range(4, 1, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 4;
        }

        if (M & 2) {
            scalar_gemm_block_range(2, 1, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 2;
        }

        if (M & 1) {
            scalar_gemm_block_range(1, 1, K, 0, K, alpha,
                                    pack_A_panel(A, m_top, K),
                                    &B[n_top * K],
                                    &C[n_top * ldc + m_top],
                                    ldc);
            m_top += 1;
        }

        n_top += 1;
    }

    ime_affinity_guard_leave(&guard);
    return 0;
}

int ime_kernel_8x8_zvl128b_lmul1_unroll8(
    BLASLONG M, BLASLONG N, BLASLONG K,
    int32_t alpha, const int8_t *A, const int8_t *B, int32_t *C, BLASLONG ldc)
{
    return ime_gemm_s8s8_prepacked(M, N, K, alpha, A, B, C, ldc);
}


