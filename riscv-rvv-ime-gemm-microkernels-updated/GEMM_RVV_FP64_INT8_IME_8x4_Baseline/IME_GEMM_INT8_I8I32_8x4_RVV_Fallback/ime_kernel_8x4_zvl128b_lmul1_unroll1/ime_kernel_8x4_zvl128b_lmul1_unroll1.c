/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 */

/*
RVV IME KERNEL
Variant: ime_kernel_8x4_zvl128b_lmul1_unroll1
Settings:
 LMUL=1 (m1)
 M=8
 N=4
 precision=int8 x int8 -> int32 (IME)
 unroll_factor=1 (No unroll)
 Kernel Name: ime_kernel_8x4_zvl128b_lmul1_unroll1
*/
/*
 * Kernel dataflow summary: ime_kernel_8x4_zvl128b_lmul1_unroll1
 * Input A/B: original INT8 matrices supplied by the benchmark or caller.
 * Packing: IME fast path packs full tiles; RVV fallback repacks into local IGEMM panels.
 * Accumulation: INT32 VMADOT accumulators on IME, or INT32 RVV accumulators in fallback.
 * Output C: column-major INT32 tile updated with alpha-scaled dot products.
 */
/*
 * Architecture overview
 * ---------------------
 * Purpose: IME INT8 GEMM front-end for the 8x4 micro-tile family.
 * Step 1: Receive caller matrices in the benchmark layout and check the active CPU core.
 * Step 2: Use the IME VMADOT path when the process is running on an IME-capable core.
 * Step 3: Pack full tiles into the IME-friendly layout before the hardware dot-product loop.
 * Step 4: Use the local RVV fallback source when IME execution is not available.
 * Step 5: Write alpha-scaled INT32 dot products back into column-major C.
 */
#define _GNU_SOURCE

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

#define UNROLL_K 1

enum {
    IME_MR = 8,
    IME_NR = 4,
    IME_KR = 8,
    IME_ALIGNMENT = 64
};

/* Kernel section: IME tiled GEMM driver
 * Packs full 8x4 or 8x8 tiles, dispatches the IME micro-kernel, and handles fallback paths.
 */
static int ime_gemm_s8s8(BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha,
                      const int8_t *A, const int8_t *B, int32_t *C,
                      BLASLONG ldc);
int igemm_kernel_8x4_zvl128b_lmul1_unroll1_i8i32(
    BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha,
    int8_t *A, int8_t *B, int32_t *C, BLASLONG ldc);

static int boundary_cleanup_gemm_full(BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha,
                            const int8_t *A, const int8_t *B,
                            int32_t *C, BLASLONG ldc);
static int size_mul_ok(size_t a, size_t b, size_t *out);
static void *aligned_alloc_bytes(size_t bytes);

/* Kernel section: IME RVV fallback packer
 * Repackages caller A/B matrices into the local RVV IGEMM layout before calling rvv_fallback.c.
 */
static int rvv_fallback_gemm_s8s8(
    BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha,
    const int8_t *A, const int8_t *B, int32_t *C, BLASLONG ldc)
{
    size_t a_count;
    size_t b_count;
    int8_t *A_rvv = NULL;
    int8_t *B_rvv = NULL;
    BLASLONG n_top = 0;
    int rc;

    if (!size_mul_ok((size_t)M, (size_t)K, &a_count) ||
        !size_mul_ok((size_t)N, (size_t)K, &b_count)) {
        return boundary_cleanup_gemm_full(M, N, K, alpha, A, B, C, ldc);
    }

    A_rvv = (int8_t *)aligned_alloc_bytes(a_count);
    B_rvv = (int8_t *)aligned_alloc_bytes(b_count);
    if (A_rvv == NULL || B_rvv == NULL) {
        free(A_rvv);
        free(B_rvv);
        return boundary_cleanup_gemm_full(M, N, K, alpha, A, B, C, ldc);
    }

    {
        BLASLONG m_top = 0;

        while (m_top < M) {
            const BLASLONG rows_left = M - m_top;
            const BLASLONG rows = (rows_left >= 8) ? 8 :
                                  (rows_left >= 4) ? 4 :
                                  (rows_left >= 2) ? 2 : 1;

            for (BLASLONG k = 0; k < K; ++k) {
                for (BLASLONG r = 0; r < rows; ++r) {
                    A_rvv[m_top * K + k * rows + r] = A[k * M + m_top + r];
                }
            }

            m_top += rows;
        }
    }

    while (n_top + 4 <= N) {
        for (BLASLONG k = 0; k < K; ++k) {
            for (BLASLONG c = 0; c < 4; ++c) {
                B_rvv[n_top * K + k * 4 + c] = B[k * N + n_top + c];
            }
        }
        n_top += 4;
    }

    if (N - n_top >= 2) {
        for (BLASLONG k = 0; k < K; ++k) {
            for (BLASLONG c = 0; c < 2; ++c) {
                B_rvv[n_top * K + k * 2 + c] = B[k * N + n_top + c];
            }
        }
        n_top += 2;
    }

    if (N - n_top >= 1) {
        for (BLASLONG k = 0; k < K; ++k) {
            B_rvv[n_top * K + k] = B[k * N + n_top];
        }
    }

    rc = igemm_kernel_8x4_zvl128b_lmul1_unroll1_i8i32(M, N, K, alpha, A_rvv, B_rvv, C, ldc);
    free(A_rvv);
    free(B_rvv);

    if (rc == 0) return 0;
    return boundary_cleanup_gemm_full(M, N, K, alpha, A, B, C, ldc);
}


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

static int size_mul_ok(size_t a, size_t b, size_t *out) {
#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)
    return !__builtin_mul_overflow(a, b, out);
#endif
#endif
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int round_up_ok(size_t value, size_t align, size_t *out) {
    size_t rem;

    if (align == 0) return 0;
    rem = value % align;
    if (rem == 0) {
        *out = value;
        return 1;
    }
    if (value > SIZE_MAX - (align - rem)) return 0;
    *out = value + (align - rem);
    return 1;
}

static void *aligned_alloc_bytes(size_t bytes) {
    void *ptr = NULL;
    size_t padded_bytes;

    if (bytes == 0) return NULL;
    if (!round_up_ok(bytes, IME_ALIGNMENT, &padded_bytes)) return NULL;
    if (posix_memalign(&ptr, IME_ALIGNMENT, padded_bytes) != 0) return NULL;
    return ptr;
}

static BLASLONG pick_row_block(BLASLONG rows_left) {
    if (rows_left >= IME_MR) return IME_MR;
    if (rows_left >= 4) return 4;
    if (rows_left >= 2) return 2;
    return 1;
}

static BLASLONG pick_col_block(BLASLONG cols_left) {
    if (cols_left >= IME_NR) return IME_NR;
    if (cols_left >= 2) return 2;
    return 1;
}

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

static int32_t add_scaled_wrap_i32(int32_t dst, int64_t acc, int32_t alpha) {
    uint32_t scaled = (uint32_t)((int64_t)alpha * acc);
    uint32_t sum = u32_from_i32(dst) + scaled;
    return i32_from_u32(sum);
}

static void pack_A_panel(const int8_t *A, BLASLONG m_top, BLASLONG M,
                         int8_t *dst, BLASLONG K_main) {
    const BLASLONG K_steps = K_main / IME_KR;

    for (BLASLONG ks = 0; ks < K_steps; ++ks) {
        const BLASLONG kb = ks * IME_KR;
        for (int sub = 0; sub < 4; ++sub) {
            const int r0 = sub * 2;
            for (int r = r0; r < r0 + 2; ++r) {
                for (int kk = 0; kk < IME_KR; ++kk) {
                    *dst++ = A[(kb + kk) * M + m_top + r];
                }
            }
        }
    }
}

static void pack_B_panel(const int8_t *B, BLASLONG n_top, BLASLONG N,
                         int8_t *dst, BLASLONG K_main) {
    const BLASLONG K_steps = K_main / IME_KR;

    for (BLASLONG ks = 0; ks < K_steps; ++ks) {
        const BLASLONG kb = ks * IME_KR;
        for (int c = 0; c < IME_NR; ++c) {
            for (int kk = 0; kk < IME_KR; ++kk) {
                *dst++ = B[(kb + kk) * N + n_top + c];
            }
        }
    }
}

static void boundary_cleanup_gemm_block(BLASLONG rows, BLASLONG cols, BLASLONG K,
                              int32_t alpha, const int8_t *A, BLASLONG lda,
                              const int8_t *B, BLASLONG ldb,
                              int32_t *C, BLASLONG ldc) {
    for (BLASLONG c = 0; c < cols; ++c) {
        for (BLASLONG r = 0; r < rows; ++r) {
            int64_t acc = 0;

#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                acc += (int32_t)A[k * lda + r] * (int32_t)B[k * ldb + c];
            }

            C[c * ldc + r] = add_scaled_wrap_i32(C[c * ldc + r], acc, alpha);
        }
    }
}

static int boundary_cleanup_gemm_full(BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha,
                            const int8_t *A, const int8_t *B,
                            int32_t *C, BLASLONG ldc) {
    BLASLONG n_top = 0;

    while (n_top < N) {
        const BLASLONG cols = pick_col_block(N - n_top);
        BLASLONG m_top = 0;

        while (m_top < M) {
            const BLASLONG rows = pick_row_block(M - m_top);
            boundary_cleanup_gemm_block(rows, cols, K, alpha,
                              &A[m_top], M,
                              &B[n_top], N,
                              &C[n_top * ldc + m_top], ldc);
            m_top += rows;
        }

        n_top += cols;
    }

    return 0;
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
#define IME_8X4_DOT_STEP \
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
#define IME_8X4_DOT_UNROLL IME_8X4_DOT_STEP
#elif UNROLL_K == 2
#define IME_8X4_DOT_UNROLL IME_8X4_DOT_STEP IME_8X4_DOT_STEP
#elif UNROLL_K == 4
#define IME_8X4_DOT_UNROLL IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP
#elif UNROLL_K == 8
#define IME_8X4_DOT_UNROLL IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP IME_8X4_DOT_STEP
#else
#error "Unsupported UNROLL_K for IME 8x4 kernel"
#endif

/* Kernel section: IME fast-path compute
 * Uses the encoded VMADOT instruction sequence for INT8xINT8 to INT32 accumulation.
 */
static void ime_kernel_8x4(BLASLONG K_main, int32_t alpha,
                           const int8_t *A_pack, const int8_t *B_pack,
                           int32_t *C, BLASLONG ldc) {
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
        IME_8X4_DOT_UNROLL
        "addi t1,t1,-1\n\t"
        "bnez t1,1b\n\t"

        "2:\n\t"
        "li t0,8\n\t"
        "vsetvli zero,t0,e32,m1,ta,ma\n\t"
        "vmul.vx v8,v8,%[alpha]\n\t"
        "vmul.vx v10,v10,%[alpha]\n\t"
        "vmul.vx v12,v12,%[alpha]\n\t"
        "vmul.vx v14,v14,%[alpha]\n\t"

        "slli t4,%[ldc],2\n\t"
        "li t0,4\n\t"
        "vsetvli zero,t0,e32,m1,ta,ma\n\t"

        "mv t5,%[cptr]\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v8\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        "addi t5,%[cptr],4\n\t"
        "vslidedown.vi v25,v8,4\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v25\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        "addi t5,%[cptr],8\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v10\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        "addi t5,%[cptr],12\n\t"
        "vslidedown.vi v25,v10,4\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v25\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        "addi t5,%[cptr],16\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v12\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        "addi t5,%[cptr],20\n\t"
        "vslidedown.vi v25,v12,4\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v25\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        "addi t5,%[cptr],24\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v14\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        "addi t5,%[cptr],28\n\t"
        "vslidedown.vi v25,v14,4\n\t"
        "vlse32.v v24,(t5),t4\n\t"
        "vadd.vv v24,v24,v25\n\t"
        "vsse32.v v24,(t5),t4\n\t"

        :
        : [kgroups] "r"(K_groups),
          [aptr] "r"(A_pack),
          [bptr] "r"(B_pack),
          [alpha] "r"(alpha),
          [cptr] "r"(C),
          [ldc] "r"(ldc)
        : "t0", "t1", "t2", "t3", "t4", "t5", "memory",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v24", "v25");
}
#undef IME_8X4_DOT_UNROLL
#undef IME_8X4_DOT_STEP
#endif

/* Kernel section: IME tiled GEMM driver
 * Packs full 8x4 or 8x8 tiles, dispatches the IME micro-kernel, and handles fallback paths.
 */
static int ime_gemm_s8s8(BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha,
                      const int8_t *A, const int8_t *B, int32_t *C,
                      BLASLONG ldc) {
    const BLASLONG K_steps = K / IME_KR;
    const BLASLONG K_main = (K_steps / UNROLL_K) * UNROLL_K * IME_KR;
    const BLASLONG M_blocks = M / IME_MR;
    const BLASLONG N_blocks = N / IME_NR;
    size_t a_panel_bytes;
    size_t b_panel_bytes;
    size_t a_pack_bytes;
    int8_t *Aall = NULL;
    int8_t *Bp = NULL;
    ime_affinity_guard guard;
    int rc = 0;

    if (M <= 0 || N <= 0 || K <= 0 || alpha == 0) return 0;
    if (ldc < M) {
        fprintf(stderr, "ime_gemm_s8s8: ldc<M\n");
        return -1;
    }
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "ime_gemm_s8s8: null\n");
        return -1;
    }

    if (!SPACEMIT_IME_HAS_RVV || !ime_vector_shape_supported() || K_main == 0 || M_blocks == 0 || N_blocks == 0) {
        return rvv_fallback_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
    }

    if (!ime_affinity_guard_enter(&guard)) {
        return rvv_fallback_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
    }

    if (!size_mul_ok((size_t)K_steps, (size_t)(IME_MR * IME_KR), &a_panel_bytes) ||
        !size_mul_ok((size_t)K_steps, (size_t)(IME_NR * IME_KR), &b_panel_bytes) ||
        !size_mul_ok((size_t)M_blocks, a_panel_bytes, &a_pack_bytes)) {
        rc = rvv_fallback_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
        goto out;
    }

    Aall = (int8_t *)aligned_alloc_bytes(a_pack_bytes);
    Bp = (int8_t *)aligned_alloc_bytes(b_panel_bytes);
    if (Aall == NULL || Bp == NULL) {
        free(Aall);
        free(Bp);
        rc = rvv_fallback_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
        goto out;
    }

    for (BLASLONG i = 0; i < M_blocks; ++i) {
        pack_A_panel(A, i * IME_MR, M, &Aall[(size_t)i * a_panel_bytes], K_main);
    }

    {
        BLASLONG n_top = 0;

        for (BLASLONG j = 0; j < N_blocks; ++j) {
            BLASLONG m_top = 0;

            pack_B_panel(B, n_top, N, Bp, K_main);

            for (BLASLONG i = 0; i < M_blocks; ++i) {
#if SPACEMIT_IME_HAS_RVV
                ime_kernel_8x4(K_main, alpha,
                               &Aall[(size_t)i * a_panel_bytes],
                               Bp,
                               &C[n_top * ldc + m_top],
                               ldc);
#endif
                if (K_main != K) {
                    boundary_cleanup_gemm_block(IME_MR, IME_NR, K - K_main, alpha,
                                      &A[m_top + K_main * M], M,
                                      &B[n_top + K_main * N], N,
                                      &C[n_top * ldc + m_top], ldc);
                }

                m_top += IME_MR;
            }

            while (m_top < M) {
                const BLASLONG rows = pick_row_block(M - m_top);
                boundary_cleanup_gemm_block(rows, IME_NR, K, alpha,
                                  &A[m_top], M,
                                  &B[n_top], N,
                                  &C[n_top * ldc + m_top], ldc);
                m_top += rows;
            }

            n_top += IME_NR;
        }

        while (n_top < N) {
            const BLASLONG cols = pick_col_block(N - n_top);
            BLASLONG m_top = 0;

            while (m_top < M) {
                const BLASLONG rows = pick_row_block(M - m_top);
                boundary_cleanup_gemm_block(rows, cols, K, alpha,
                                  &A[m_top], M,
                                  &B[n_top], N,
                                  &C[n_top * ldc + m_top], ldc);
                m_top += rows;
            }

            n_top += cols;
        }
    }

out:
    free(Aall);
    free(Bp);
    ime_affinity_guard_leave(&guard);
    return rc;
}

/* Kernel section: public IME wrapper entry point
 * Chooses IME execution on IME-capable cores and RVV vector fallback otherwise.
 */
int ime_kernel_8x4_zvl128b_lmul1_unroll1(
    BLASLONG M, BLASLONG N, BLASLONG K,
    int32_t alpha, const int8_t *A, const int8_t *B, int32_t *C, BLASLONG ldc)
{
    return ime_gemm_s8s8(M, N, K, alpha, A, B, C, ldc);
}
