/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 *
 * SpacemiT IME INT8 GEMM microkernel: ime_kernel_4x4_zvl128b_lmul1_unroll1_dual2x2x4
 *
 * Child-friendly view:
 *   This kernel multiplies INT8 values from A and B.
 *   It adds the products into INT32 values in C.
 *   Fast path: use native IME hardware on IME-capable cores.
 *   Fallback path: use RVV, then scalar C if needed.
 *   Main software tile: 4 rows by 4 columns of output C.
 *
 * Mathematical operation:
 *   C += alpha * (A x B)
 *   A and B contain signed INT8 values. C contains INT32 values.
 *
 * Caller layout:
 *   A[k * M + row], B[k * N + column], C[column * ldc + row]
 *
 * Private native layouts:
 *   K1 small mode: C(4x4) is built from two dual 2x2x4 native IME operations
 *   Native hardware tile: 2x2x4[x2], selected with vl=16 INT8 lanes
 *
 * Software tile:
 *   4x4. The driver composes two dual 2x2x4 native IME operations internally.
 *
 * Documented LMUL:
 *   The experimental small IME mode uses LMUL=1 and vl*SEW=128.
 *
 * Portability:
 *   The matching rvv_fallback.c remains local to this kernel folder.
 *   SPACEMIT_IME_REQUIRE_HARDWARE disables fallback during accuracy testing.
 */
/*
 * -----------------------------------------------------------------------------
 * ROADMAP 0: Build environment and public contract
 * -----------------------------------------------------------------------------
 * This first part only prepares the file: headers, compile-time feature checks,
 * constants, and the public RVV fallback symbol. Nothing is computed yet.
 */
/* Enable GNU/Linux helpers such as CPU affinity functions. */
#define _GNU_SOURCE

#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Check at compile time whether this build has RISC-V vector support. */
#if defined(__riscv) && defined(__riscv_vector)
#define SPACEMIT_IME_HAS_RVV 1
#else
#define SPACEMIT_IME_HAS_RVV 0
#endif

/* Keep native IME path enabled for this kernel. */
#define SPACEMIT_IME_NATIVE_ENABLED 1

/* OpenBLAS-style integer type used for matrix sizes and strides. */
typedef long BLASLONG;

/* This variant uses unroll factor 1 in the K loop. */
#define UNROLL_K 1

enum {
    /* Number of C rows handled by the kernel code tile. */
    IME_MR = 4,

    /* Number of C columns handled by the kernel code tile. */
    IME_NR = 4,

    /* Packed panels are aligned to 64 bytes for hardware-friendly access. */
    IME_ALIGNMENT = 64,

    /* Maximum temporary output values produced by native IME profiles. */
    IME_MAX_OUTPUT_VALUES = 16
};

/* Which native IME hardware shape is selected by this experimental kernel. */
typedef enum ime_profile {
    IME_PROFILE_NONE = 0,
    IME_PROFILE_A60,
    IME_PROFILE_A100
} ime_profile;

/* External RVV INT8 kernel used when native IME is not selected. */
int igemm_kernel_8x4_zvl128b_lmul1_unroll1_i8i32(BLASLONG M, BLASLONG N, BLASLONG K,
                      int32_t alpha, int8_t *A, int8_t *B,
                      int32_t *C, BLASLONG ldc);

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 1: Small safety helpers
 * -----------------------------------------------------------------------------
 * These helpers keep integer behavior, allocation sizes, and environment flags
 * predictable before any GEMM work starts.
 */
/* Reinterpret signed INT32 bits as unsigned bits without changing the bit pattern. */
static uint32_t u32_from_i32(int32_t x)
{
    /* y receives the same 32 bits as x, only viewed as unsigned. */
    uint32_t y;
    /* memcpy avoids undefined behavior from direct signed/unsigned aliasing. */
    memcpy(&y, &x, sizeof(y));
    /* Return the unchanged bit pattern. */
    return y;
}

/* Reinterpret unsigned bits back as signed INT32. */
static int32_t i32_from_u32(uint32_t x)
{
    /* y receives the same 32 bits as x, only viewed as signed. */
    int32_t y;
    /* memcpy keeps the conversion as a pure bit reinterpretation. */
    memcpy(&y, &x, sizeof(y));
    /* Return the unchanged bit pattern. */
    return y;
}

/* Add alpha*value into C with normal 32-bit wraparound behavior. */
static int32_t add_scaled_wrap_i32(int32_t dst, int64_t value, int32_t alpha)
{
    /* First scale the computed dot product by alpha. */
    uint32_t scaled = (uint32_t)((int64_t)alpha * value);
    /* Then add it to the old C value with exact 32-bit wraparound. */
    return i32_from_u32(u32_from_i32(dst) + scaled);
}

/* Safely multiply two sizes before allocating memory. */
static int size_mul_ok(size_t a, size_t b, size_t *out)
{
    /* Refuse the multiplication if it would overflow size_t. */
    if (a != 0 && b > SIZE_MAX / a) return 0;
    /* Store the safe product for later allocation. */
    *out = a * b;
    /* Return success. */
    return 1;
}

/* Allocate memory aligned to 64 bytes for packed A/B panels. */
static void *aligned_alloc_bytes(size_t bytes)
{
    /* ptr will point to the aligned memory block. */
    void *ptr = NULL;
    /* padded is bytes rounded up to the next alignment boundary. */
    size_t padded;

    /* Reject zero-size and overflow-prone requests. */
    if (bytes == 0 || bytes > SIZE_MAX - (IME_ALIGNMENT - 1)) return NULL;
    /* Round the allocation size up to 64-byte alignment. */
    padded = (bytes + IME_ALIGNMENT - 1) & ~(size_t)(IME_ALIGNMENT - 1);
    /* Ask the OS/runtime for aligned memory. */
    if (posix_memalign(&ptr, IME_ALIGNMENT, padded) != 0) return NULL;
    /* Return the aligned buffer to the caller. */
    return ptr;
}

/*
 * Optional runtime switches for validation:
 *   SPACEMIT_IME_FORCE_NATIVE=1  -> require native IME path, return error if unavailable.
 *   SPACEMIT_IME_FORCE_RVV=1     -> use the RVV fallback path.
 *   SPACEMIT_IME_FORCE_SCALAR=1  -> use the plain scalar reference path.
 */
static int env_flag_enabled(const char *name)
{
    /* Read the environment variable by name. */
    const char *value = getenv(name);
    /* Missing or empty means disabled. */
    if (value == NULL || value[0] == '\0') return 0;
    /* 0/no/No means disabled. Any other value means enabled. */
    if (value[0] == '0' || value[0] == 'n' || value[0] == 'N') return 0;
    /* The flag is enabled. */
    return 1;
}

/* Choose how many leftover rows can be handled by fallback packing. */
static BLASLONG pick_row_block(BLASLONG left)
{
    /* Prefer the largest row block still available. */
    if (left >= 8) return 8;
    /* If 8 rows are not left, try 4 rows. */
    if (left >= 4) return 4;
    /* If 4 rows are not left, try 2 rows. */
    if (left >= 2) return 2;
    /* Last case: only 1 row remains. */
    return 1;
}

/* Choose how many leftover columns can be handled by fallback packing. */
static BLASLONG pick_col_block(BLASLONG left)
{
    /* Prefer the main 4-column block. */
    if (left >= IME_NR) return IME_NR;
    /* Keep this fallback generic if IME_NR changes. */
    if (left >= 4) return 4;
    /* Handle a 2-column tail if present. */
    if (left >= 2) return 2;
    /* Last case: only 1 column remains. */
    return 1;
}

/*
 * Small plain-C GEMM block.
 * Used for fallback and leftover edges.
 * Simple formula for every output cell:
 *   C(row,col) = C(row,col) + alpha * sum_k A(row,k) * B(k,col)
 */
static void scalar_gemm_block(BLASLONG rows, BLASLONG cols, BLASLONG K,
                              int32_t alpha, const int8_t *A, BLASLONG lda,
                              const int8_t *B, BLASLONG ldb,
                              int32_t *C, BLASLONG ldc)
{
    /* Walk over each output column in this small scalar block. */
    for (BLASLONG c = 0; c < cols; ++c) {
        /* Walk over each output row in that column. */
        for (BLASLONG r = 0; r < rows; ++r) {
            /* sum holds the dot product before it is added into INT32 C. */
            int64_t sum = 0;

#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                sum += (int32_t)A[k * lda + r] * (int32_t)B[k * ldb + c];
            }

            /* Store the final contribution back into C(row,column). */
            C[c * ldc + r] = add_scaled_wrap_i32(C[c * ldc + r], sum, alpha);
        }
    }
}

/*
 * Roadmap note:
 * The scalar block above is the simplest version of the math. It is slower, but
 * it is easy to trust because it directly computes each C(row,column) dot product.
 */
/* Full scalar GEMM fallback for the whole matrix. Slow but always valid. */
static int scalar_gemm_full(BLASLONG M, BLASLONG N, BLASLONG K,
                            int32_t alpha, const int8_t *A, const int8_t *B,
                            int32_t *C, BLASLONG ldc)
{
    scalar_gemm_block(M, N, K, alpha, A, M, B, N, C, ldc);
    return 0;
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 2: RVV fallback path
 * -----------------------------------------------------------------------------
 * If this thread is not running on an IME-capable core, this path keeps the same
 * INT8 x INT8 -> INT32 math but uses the RVV INT8 micro-kernel instead.
 *
 * Flow:
 *   1. Allocate packed A and B buffers.
 *   2. Copy column-major BLAS input data into the RVV kernel layout.
 *   3. Call the RVV INT8 kernel.
 *   4. If RVV fails, use the scalar reference path.
 */
static int rvv_fallback_gemm(BLASLONG M, BLASLONG N, BLASLONG K,
                             int32_t alpha, const int8_t *A, const int8_t *B,
                             int32_t *C, BLASLONG ldc)
{
    /* During strict hardware-only testing, fallback can be disabled. */
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

    /* Calculate packed A and B sizes safely. */
    if (!size_mul_ok((size_t)M, (size_t)K, &a_bytes) ||
        !size_mul_ok((size_t)N, (size_t)K, &b_bytes)) {
        return scalar_gemm_full(M, N, K, alpha, A, B, C, ldc);
    }

    /* Allocate packed A buffer for RVV fallback. */
    A_pack = (int8_t *)aligned_alloc_bytes(a_bytes);
    /* Allocate packed B buffer for RVV fallback. */
    B_pack = (int8_t *)aligned_alloc_bytes(b_bytes);
    if (A_pack == NULL || B_pack == NULL) {
        free(A_pack);
        free(B_pack);
        return scalar_gemm_full(M, N, K, alpha, A, B, C, ldc);
    }

    /* Pack A by row blocks so the RVV fallback kernel can read it easily. */
    while (m_top < M) {
        BLASLONG rows = pick_row_block(M - m_top);
        for (BLASLONG k = 0; k < K; ++k) {
            /* Walk over the rows inside this packed A block. */
            for (BLASLONG r = 0; r < rows; ++r) {
                /* Copy A(k,row) from original layout into packed layout. */
                A_pack[m_top * K + k * rows + r] = A[k * M + m_top + r];
            }
        }
        m_top += rows;
    }

    /* Pack B by column blocks so the RVV fallback kernel can read it easily. */
    while (n_top < N) {
        BLASLONG cols = pick_col_block(N - n_top);
        for (BLASLONG k = 0; k < K; ++k) {
            /* Walk over the columns inside this packed B block. */
            for (BLASLONG c = 0; c < cols; ++c) {
                /* Copy B(k,column) from original layout into packed layout. */
                B_pack[n_top * K + k * cols + c] = B[k * N + n_top + c];
            }
        }
        n_top += cols;
    }

    /* Call the RVV INT8 fallback kernel using packed A and B. */
    rc = igemm_kernel_8x4_zvl128b_lmul1_unroll1_i8i32(M, N, K, alpha, A_pack, B_pack, C, ldc);
    free(A_pack);
    free(B_pack);
    if (rc == 0) return 0;
    return scalar_gemm_full(M, N, K, alpha, A, B, C, ldc);
#endif
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 3: Core selection for native IME
 * -----------------------------------------------------------------------------
 * Native IME instructions should run only on IME-capable cores. This block checks
 * the Linux CPU/device-tree information, temporarily pins the thread to a valid
 * core, and restores the original affinity at the end.
 */
/* Saves/restores CPU affinity while trying to run on an IME-capable core. */
typedef struct ime_affinity_guard {
    cpu_set_t saved_mask;
    int active;
} ime_affinity_guard;

/* Check Linux device-tree marker to see whether this CPU has AI/IME support. */
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

/* Try to pin the current thread to an IME-capable CPU before native IME use. */
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

/* Restore the original CPU affinity after native IME work finishes. */
static void ime_affinity_guard_leave(ime_affinity_guard *guard)
{
    if (guard->active) {
        (void)sched_setaffinity(0, sizeof(guard->saved_mask), &guard->saved_mask);
    }
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 4: Detect native IME tile shape
 * -----------------------------------------------------------------------------
 * This experimental source selects the smaller K1 native mode. With SEW=8 and vl=16, IME selects 2x2x4[x2].
 */
/* Detect whether the current core can run the small 2x2x4[x2] IME mode. */
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

    /*
     * Experimental small native mode:
     * VLEN=256 supports vl*SEW=128, which selects INT8 2x2x4[x2].
     * We only need at least 16 INT8 lanes for this path.
     */
    if (vlmax >= 16) return IME_PROFILE_A60;
#endif
    return IME_PROFILE_NONE;
}

/* K chunk size used by the native IME dot primitive. */
static BLASLONG profile_kr(ime_profile profile)
{
    (void)profile;
    return 4;
}

/* Native hardware row count for one IME primitive. */
static BLASLONG profile_native_rows(ime_profile profile)
{
    (void)profile;
    return 2;
}

/* Native hardware column count for one IME primitive. */
static BLASLONG profile_native_cols(ime_profile profile)
{
    (void)profile;
    return 2;
}

/* Number of bytes needed to pack one B panel for native IME. */
static size_t b_panel_bytes(ime_profile profile, BLASLONG K_main)
{
    (void)profile;
    /*
     * Each native 2x2x4[x2] operation needs two B copies.
     * A 4-column software tile uses two column-pair operations per K group.
     */
    return (size_t)2 * (size_t)IME_NR * (size_t)K_main;
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 5: Pack BLAS matrices into IME tile layout
 * -----------------------------------------------------------------------------
 * The caller gives A and B in BLAS/Fortran-style column-major layout. The IME
 * instruction does not read that layout directly, so these functions create small
 * packed panels in the exact order the hardware dot-product instruction expects.
 */
/*
 * Pack one 4-row A panel into the order expected by 2x2x4[x2] IME.
 * Child view: take the needed A values from the big A matrix and place them
 * in a small clean row/column order for the hardware instruction.
 */
static void pack_a_panel(ime_profile profile, const int8_t *A,
                         BLASLONG m_top, BLASLONG M, BLASLONG K_main,
                         int8_t *dst)
{
    /* kr is the K depth consumed by one native IME dot step. */
    BLASLONG kr = profile_kr(profile);
    /* native_rows is 2 for the 2x2x4[x2] native mode. */
    BLASLONG native_rows = profile_native_rows(profile);

    /* Move through K in native IME chunks. */
    for (BLASLONG kb = 0; kb < K_main; kb += kr) {
        /* Move through the 4 software rows in native 2-row groups. */
        for (BLASLONG rb = 0; rb < IME_MR; rb += native_rows) {
            /* Pick one row inside the native IME row group. */
            for (BLASLONG r = 0; r < native_rows; ++r) {
                /* Pick one K value inside the current native K chunk. */
                for (BLASLONG kk = 0; kk < kr; ++kk) {
                    /* Copy one A value into the packed IME A panel. */
                    *dst++ = A[(kb + kk) * M + m_top + rb + r];
                }
            }
        }
    }
}

/* Pack one 4-column B panel into the order expected by IME. */
static void pack_b_panel(ime_profile profile, const int8_t *B,
                         BLASLONG n_top, BLASLONG N, BLASLONG K_main,
                         int8_t *dst)
{
    BLASLONG kr = profile_kr(profile);
    BLASLONG native_cols = profile_native_cols(profile);

    /*
     * For one 4x4 software tile, run two native column-pair operations:
     *   columns 0-1 and columns 2-3.
     * Each native 2x2x4[x2] operation contains two copies, so B is duplicated
     * once for the top 2 rows and once for the bottom 2 rows.
     */
    for (BLASLONG kb = 0; kb < K_main; kb += kr) {
        for (BLASLONG cb = 0; cb < IME_NR; cb += native_cols) {
            for (BLASLONG copy = 0; copy < 2; ++copy) {
                (void)copy;
                for (BLASLONG c = 0; c < native_cols; ++c) {
                    for (BLASLONG kk = 0; kk < kr; ++kk) {
                        BLASLONG column = cb + c;
                        *dst++ = B[(kb + kk) * N + n_top + column];
                    }
                }
            }
        }
    }
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 6: Native IME dot-product instructions
 * -----------------------------------------------------------------------------
 * This is the only hardware-specific part. The inline assembly loads packed A/B
 * panels into vector registers and executes encoded IME vmadot instructions.
 * The surrounding C code prepares data for this block and stores its result.
 */
#if SPACEMIT_IME_HAS_RVV && SPACEMIT_IME_NATIVE_ENABLED
/*
 * One small native A60 IME dot step using vl*SEW=128:
 *   native primitive = 2x2x4[x2].
 * One operation produces two C(2x2) blocks = 8 INT32 values.
 * This 4x4 software kernel uses two such operations:
 *   columns 0-1 -> left half of C(4x4)
 *   columns 2-3 -> right half of C(4x4)
 */
#define IME_A60_DOT_STEP \
        "vle8.v v16,(t2)\n\t" \
        "addi t2,t2,16\n\t" \
        "vle8.v v20,(t3)\n\t" \
        "addi t3,t3,16\n\t" \
        ".word 0xE348342Bu\n\t" \
        "vle8.v v21,(t3)\n\t" \
        "addi t3,t3,16\n\t" \
        ".word 0xE358352Bu\n\t"
/* One native A100 IME dot step: load larger packed A/B and execute encoded vmadot word. */
#define IME_A100_DOT_STEP \
        "vle8.v v16,(t2)\n\t" \
        "addi t2,t2,128\n\t" \
        "vle8.v v20,(t3)\n\t" \
        "addi t3,t3,128\n\t" \
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

/* Run native A60 IME accumulation and write INT32 tile results to out[]. */
static void ime_accumulate_a60(const int8_t *A_pack, const int8_t *B_pack,
                              BLASLONG k_groups,
                              int32_t out[IME_MAX_OUTPUT_VALUES])
{
    /*
     * Assembly flow for the experimental 2x2x4[x2] mode:
     *   1. Clear two INT32 accumulator registers.
     *   2. Use vl=16 INT8 lanes so IME selects 2x2x4[x2].
     *   3. Load one packed A block: two A(2x4) copies.
     *   4. Run vmadot for B columns 0-1 and again for B columns 2-3.
     *   5. Store two 8-value outputs, forming one C(4x4) software tile.
     */
    __asm__ __volatile__(
        "li t0,8\n\t"
        "vsetvli zero,t0,e32,m1,ta,ma\n\t"
        "vmv.v.i v8,0\n\t"
        "vmv.v.i v10,0\n\t"
        "li t0,16\n\t"
        "vsetvli zero,t0,e8,m1,ta,ma\n\t"
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
        "li t0,8\n\t"
        "vsetvli zero,t0,e32,m1,ta,ma\n\t"
        "mv t4,%[out]\n\t"
        "vse32.v v8,(t4)\n\t"
        "addi t4,t4,32\n\t"
        "vse32.v v10,(t4)\n\t"
        :
        : [kgroups] "r"(k_groups),
          [aptr] "r"(A_pack),
          [bptr] "r"(B_pack),
          [out] "r"(out)
        : "t0", "t1", "t2", "t3", "t4", "memory", "v8", "v10", "v16", "v20", "v21");
}

/* Run native A100 IME accumulation and write INT32 tile results to out[]. */
static void ime_accumulate_a100(const int8_t *A_pack, const int8_t *B_pack,
                              BLASLONG k_groups,
                              int32_t out[IME_MAX_OUTPUT_VALUES])
{
    /*
     * Assembly flow for A100-style IME:
     *   1. Clear the larger INT32 accumulator tile.
     *   2. Loop over packed K groups.
     *   3. Load the packed A and B panels.
     *   4. Execute the native vmadot instruction.
     *   5. Store the native tile into out[].
     */
    __asm__ __volatile__(
        "li t0,64\n\t"
        "vsetvli zero,t0,e32,m2,ta,ma\n\t"
        "vmv.v.i v8,0\n\t"
        "li t0,128\n\t"
        "vsetvli zero,t0,e8,m1,ta,ma\n\t"
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

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 7: Convert native IME output back to C
 * -----------------------------------------------------------------------------
 * IME writes the tile in its own native order. These helpers translate that order
 * back into normal C(row,column), then add alpha * tile_result into the real C.
 */
/* Convert logical output position (row r, column c) to native IME out[] index. */
static size_t output_index(ime_profile profile, int r, int c)
{
    (void)profile;
    /*
     * out[0..7]  = C rows 0-1 and 2-3 for columns 0-1.
     * out[8..15] = C rows 0-1 and 2-3 for columns 2-3.
     */
    return (size_t)(c / 2) * 8 +
           (size_t)(r / 2) * 4 +
           (size_t)(r % 2) * 2 +
           (size_t)(c % 2);
}

/* Copy native IME out[] values back into the real C matrix. */
static void scatter_output(ime_profile profile, BLASLONG m_top, BLASLONG n_top,
                           int32_t alpha,
                           const int32_t out[IME_MAX_OUTPUT_VALUES],
                           int32_t *C, BLASLONG ldc)
{
    /* Visit each of the 4 output columns in the 4x4 software tile. */
    for (int c = 0; c < IME_NR; ++c) {
        /* Visit each of the 4 output rows in the 4x4 software tile. */
        for (int r = 0; r < IME_MR; ++r) {
            /* ci is the column-major address of C(m_top+r, n_top+c). */
            size_t ci = (size_t)(n_top + c) * (size_t)ldc + (size_t)(m_top + r);
            /* Add alpha times the IME result into the real C matrix. */
            C[ci] = add_scaled_wrap_i32(C[ci], out[output_index(profile, r, c)], alpha);
        }
    }
}

/* Choose A60 or A100 native accumulation based on detected hardware profile. */
static void native_accumulate(ime_profile profile, const int8_t *A_pack,
                              const int8_t *B_pack, BLASLONG k_groups,
                              int32_t out[IME_MAX_OUTPUT_VALUES])
{
#if SPACEMIT_IME_HAS_RVV && SPACEMIT_IME_NATIVE_ENABLED
    /* Use the larger A100 path only when the detected profile says so. */
    if (profile == IME_PROFILE_A100) {
        ime_accumulate_a100(A_pack, B_pack, k_groups, out);
    } else {
        /* K1 lands here: small 2x2x4[x2] path composing one 4x4 software tile. */
        ime_accumulate_a60(A_pack, B_pack, k_groups, out);
    }
#else
    (void)profile; (void)A_pack; (void)B_pack; (void)k_groups; (void)out;
#endif
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 8: Main native IME GEMM driver
 * -----------------------------------------------------------------------------
 * This function is the main story of the kernel:
 *
 *   Step 1: Validate the matrix sizes and pointers.
 *   Step 2: Honor debug/validation environment flags.
 *   Step 3: Move to an IME-capable core, or fall back to RVV.
 *   Step 4: Detect the native IME profile and its K-block size.
 *   Step 5: Split the output matrix C into full 4x4 software tiles.
 *   Step 6: Pack A once, pack each B panel, and run native IME dot products.
 *   Step 7: Scatter each computed tile back into column-major C.
 *   Step 8: Use scalar cleanup for leftover K values, rows, or columns.
 *   Step 9: Free temporary buffers and restore CPU affinity.
 */
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
    int force_native;

    /* --- Step 1: basic input and environment checks ----------------------- */
    /* Start with an empty affinity guard. */
    memset(&guard, 0, sizeof(guard));
    /* Nothing to compute if any dimension is zero or alpha is zero. */
    if (M <= 0 || N <= 0 || K <= 0 || alpha == 0) return 0;
    /* Reject invalid pointers or invalid C leading dimension. */
    if (A == NULL || B == NULL || C == NULL || ldc < M) return -1;

    force_native = env_flag_enabled("SPACEMIT_IME_FORCE_NATIVE");
    if (env_flag_enabled("SPACEMIT_IME_FORCE_SCALAR")) {
        return scalar_gemm_full(M, N, K, alpha, A, B, C, ldc);
    }
    if (env_flag_enabled("SPACEMIT_IME_FORCE_RVV")) {
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    /* --- Step 2: choose native IME or fallback execution path ------------- */
    /* If native IME is unavailable or CPU pinning fails, use RVV fallback. */
    if (!SPACEMIT_IME_NATIVE_ENABLED || !ime_affinity_guard_enter(&guard)) {
        if (force_native) return -98;
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    /* --- Step 3: detect hardware tile shape and full tile coverage -------- */
    /* Detect whether this IME core supports the small 2x2x4[x2] native tile shape. */
    profile = detect_ime_profile();
    if (profile == IME_PROFILE_NONE) {
        ime_affinity_guard_leave(&guard);
        if (force_native) return -98;
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    /* kr is how many K values one native IME primitive consumes per step. */
    kr = profile_kr(profile);
    /* K_main is the largest K part cleanly handled by native IME. */
    K_main = (K / (kr * UNROLL_K)) * (kr * UNROLL_K);
    /* Number of full 4-row blocks in M. */
    m_blocks = M / IME_MR;
    /* Number of full 4-column blocks in N. */
    n_blocks = N / IME_NR;

    if (K_main == 0 || m_blocks == 0 || n_blocks == 0 ||
        !size_mul_ok((size_t)IME_MR, (size_t)K_main, &a_panel_size) ||
        !size_mul_ok((size_t)m_blocks, a_panel_size, &a_all_size)) {
        ime_affinity_guard_leave(&guard);
        if (force_native) return -98;
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    /* --- Step 4: allocate temporary packed panels ------------------------- */
    /* Allocate packed storage for all full A panels and one B panel. */
    b_panel_size = b_panel_bytes(profile, K_main);
    A_all = (int8_t *)aligned_alloc_bytes(a_all_size);
    B_pack = (int8_t *)aligned_alloc_bytes(b_panel_size);
    if (A_all == NULL || B_pack == NULL) {
        free(A_all);
        free(B_pack);
        ime_affinity_guard_leave(&guard);
        if (force_native) return -98;
        return rvv_fallback_gemm(M, N, K, alpha, A, B, C, ldc);
    }

    /* --- Step 5: pack A panels once --------------------------------------- */
    /* Pre-pack every full 4-row A panel once, because A is reused for B panels. */
    for (BLASLONG mb = 0; mb < m_blocks; ++mb) {
        pack_a_panel(profile, A, mb * IME_MR, M, K_main,
                     &A_all[(size_t)mb * a_panel_size]);
    }

    /* --- Step 6: compute full 4x4 output tiles ---------------------------- */
    /* Walk over every full 4-column B/C block. */
    for (BLASLONG nb = 0; nb < n_blocks; ++nb) {
        /* n_top is the first C/B column of this 4-column block. */
        BLASLONG n_top = nb * IME_NR;
        /* Pack current B panel before using it with all A row blocks. */
        pack_b_panel(profile, B, n_top, N, K_main, B_pack);

        /* For this B panel, compute every full 4-row C block. */
        for (BLASLONG mb = 0; mb < m_blocks; ++mb) {
            /* m_top is the first C/A row of this 4-row block. */
            BLASLONG m_top = mb * IME_MR;
            /* out holds the native IME INT32 tile before writing it into C. */
            int32_t out[IME_MAX_OUTPUT_VALUES] __attribute__((aligned(64)));

            /* Run native IME dot product for this packed A panel and B panel. */
            native_accumulate(profile, &A_all[(size_t)mb * a_panel_size],
                              B_pack, K_main / (kr * UNROLL_K), out);
            /* Add the native IME output tile back into the real C matrix. */
            scatter_output(profile, m_top, n_top, alpha, out, C, ldc);

            /* If K had leftover values not handled by IME, finish them in scalar C. */
            if (K_main != K) {
                scalar_gemm_block(IME_MR, IME_NR, K - K_main, alpha,
                                  &A[K_main * M + m_top], M,
                                  &B[K_main * N + n_top], N,
                                  &C[n_top * ldc + m_top], ldc);
            }
        }

        /* If M has leftover rows after full 4-row blocks, compute them scalar. */
        if (m_blocks * IME_MR != M) {
            BLASLONG m_top = m_blocks * IME_MR;
            scalar_gemm_block(M - m_top, IME_NR, K, alpha,
                              &A[m_top], M, &B[n_top], N,
                              &C[n_top * ldc + m_top], ldc);
        }
    }

    /* --- Step 7: final cleanup for leftover columns ----------------------- */
    /* If N has leftover columns after full 4-column blocks, compute them scalar. */
    if (n_blocks * IME_NR != N) {
        BLASLONG n_top = n_blocks * IME_NR;
        scalar_gemm_block(M, N - n_top, K, alpha, A, M, &B[n_top], N,
                          &C[n_top * ldc], ldc);
    }

    /* --- Step 8: release temporary memory and restore CPU placement ------- */
    free(A_all);
    free(B_pack);
    ime_affinity_guard_leave(&guard);
    return 0;
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 9: Public entry point used by scripts
 * -----------------------------------------------------------------------------
 * Benchmark scripts call this stable function name. It simply forwards the work
 * to the readable driver above.
 */
/* Public wrapper name used by benchmark scripts. */
int ime_kernel_4x4_zvl128b_lmul1_unroll1_dual2x2x4(BLASLONG M, BLASLONG N, BLASLONG K,
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






