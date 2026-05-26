/*
RVV INT8 GEMM Kernel
Tile: 8x4
Variant: igemm_kernel_8x4_zvl256b_lmul8_unroll2
LMUL label: 8
LMUL compute path: i8m8_vector_loads_legal_i32_accumulation
Unroll: 2
Math contract: same blocking shape/load as FP32 8x4 (M/8 x N/4 main + tails)
*/

/* LMUL note: this variant uses i8m8 input vector loads/stores and ISA-legal INT32 accumulation to keep the implementation ISA-legal. */
/*
 * Kernel dataflow summary: igemm_kernel_8x4_zvl256b_lmul8_unroll2
 * Input A: packed INT8 A micro-panel ordered by row block and K.
 * Input B: packed INT8 B micro-panel ordered by column block and K.
 * Accumulation: INT32 RVV widening accumulators across K; boundary cleanup handles leftover tiles.
 * Output C: column-major INT32 tile updated with alpha-scaled dot products.
 */
/*
 * Architecture overview
 * ---------------------
 * Purpose: standalone RVV INT8 x INT8 -> INT32 GEMM micro-kernel for 8x4 tiles.
 * Step 1: Read packed INT8 A/B micro-panels produced by the benchmark driver.
 * Step 2: Traverse K while RVV widening instructions build INT32 dot-product sums.
 * Step 3: Apply the selected LMUL and unroll factor shown in the file name.
 * Step 4: Update full vectorized tiles first for the fast path.
 * Step 5: Finish leftover rows or columns with the boundary cleanup path.
 */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

typedef long BLASLONG;
#ifndef CNAME
#define CNAME igemm_kernel_8x4_zvl256b_lmul8_unroll2
#endif

/* Kernel section: INT32 output update helper
 * Applies alpha to the dot-product accumulator and adds it into the INT32 C output element.
 */
static inline int32_t add_scaled_wrap_i32(int32_t dst, int32_t alpha, int64_t acc)
{
    uint32_t a = (uint32_t)dst;
    uint32_t b = (uint32_t)((int32_t)(acc * (int64_t)alpha));
    return (int32_t)(a + b);
}

/* Kernel section: boundary cleanup path
 * Handles leftover rows/columns outside the full vectorized tile shape.
 * Inputs are packed A/B panels; partial sums are accumulated and written back to C.
 */
static inline void boundary_cleanup_block(BLASLONG rows, BLASLONG cols, BLASLONG K,
                                int32_t alpha, const int8_t *Ablk, const int8_t *Bblk,
                                int32_t *Cblk, BLASLONG ldc)
{
    int64_t acc[32] = {0};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 2
    for (BLASLONG k = 0; k < K; ++k) {
        for (BLASLONG n = 0; n < cols; ++n) {
            const int32_t b = (int32_t)Bblk[bi + n];
            for (BLASLONG m = 0; m < rows; ++m) {
                acc[n * rows + m] += (int32_t)Ablk[ai + m] * b;
            }
        }
        ai += rows;
        bi += cols;
    }

    for (BLASLONG n = 0; n < cols; ++n) {
        int32_t *c_col = &Cblk[n * ldc];
        for (BLASLONG m = 0; m < rows; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, acc[n * rows + m]);
        }
    }
}

/* Kernel section: RVV vector micro-kernel
 * Loads packed A/B panels, accumulates dot products in RVV registers, then updates C.
 */
static inline void vec_block_8xN(BLASLONG cols, BLASLONG K,
                                 int32_t alpha, const int8_t *Ablk, const int8_t *Bblk,
                                 int32_t *Cblk, BLASLONG ldc)
{
    const size_t gvl = __riscv_vsetvl_e8m8(8);
    int8_t a_lane[8];
    int64_t acc[32] = {0};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 2
    for (BLASLONG k = 0; k < K; ++k) {
        vint8m8_t a8 = __riscv_vle8_v_i8m8(&Ablk[ai], gvl);
        __riscv_vse8_v_i8m8(a_lane, a8, gvl);
        ai += 8;

        for (BLASLONG n = 0; n < cols; ++n) {
            const int32_t b = (int32_t)Bblk[bi + n];
            for (BLASLONG m = 0; m < 8; ++m) {
                acc[n * 8 + m] += (int32_t)a_lane[m] * b;
            }
        }
        bi += cols;
    }

    for (BLASLONG n = 0; n < cols; ++n) {
        int32_t *c_col = &Cblk[n * ldc];
        for (BLASLONG m = 0; m < 8; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, acc[n * 8 + m]);
        }
    }
}

/* Kernel section: RVV vector micro-kernel
 * Loads packed A/B panels, accumulates dot products in RVV registers, then updates C.
 */
static inline void vec_block_4xN(BLASLONG cols, BLASLONG K,
                                 int32_t alpha, const int8_t *Ablk, const int8_t *Bblk,
                                 int32_t *Cblk, BLASLONG ldc)
{
    const size_t gvl = __riscv_vsetvl_e8m8(4);
    int8_t a_lane[4];
    int64_t acc[16] = {0};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 2
    for (BLASLONG k = 0; k < K; ++k) {
        vint8m8_t a8 = __riscv_vle8_v_i8m8(&Ablk[ai], gvl);
        __riscv_vse8_v_i8m8(a_lane, a8, gvl);
        ai += 4;

        for (BLASLONG n = 0; n < cols; ++n) {
            const int32_t b = (int32_t)Bblk[bi + n];
            for (BLASLONG m = 0; m < 4; ++m) {
                acc[n * 4 + m] += (int32_t)a_lane[m] * b;
            }
        }
        bi += cols;
    }

    for (BLASLONG n = 0; n < cols; ++n) {
        int32_t *c_col = &Cblk[n * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, acc[n * 4 + m]);
        }
    }
}

/* Kernel section: public micro-kernel entry point
 * Walks M/N tiles, selects full vectorized blocks or boundary cleanup tiles, and preserves the C output contract.
 */
int CNAME(BLASLONG M, BLASLONG N, BLASLONG K,
          int32_t alpha, int8_t *A, int8_t *B, int32_t *C, BLASLONG ldc)
{
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;

    if (K <= 0) {
        return 0;
    }

    if (__riscv_vsetvl_e8m8(8) < 8) {
        return -1;
    }

    for (BLASLONG j = 0; j < N / 4; ++j) {
        m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            vec_block_8xN(4, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 8;
        }

        if (M & 4) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            vec_block_4xN(4, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 4;
        }

        if (M & 2) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            boundary_cleanup_block(2, 4, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 2;
        }

        if (M & 1) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            boundary_cleanup_block(1, 4, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 1;
        }

        n_top += 4;
    }

    if (N & 2) {
        m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            vec_block_8xN(2, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 8;
        }

        if (M & 4) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            vec_block_4xN(2, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 4;
        }

        if (M & 2) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            boundary_cleanup_block(2, 2, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 2;
        }

        if (M & 1) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            boundary_cleanup_block(1, 2, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 1;
        }

        n_top += 2;
    }

    if (N & 1) {
        m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            vec_block_8xN(1, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 8;
        }

        if (M & 4) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            vec_block_4xN(1, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 4;
        }

        if (M & 2) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            boundary_cleanup_block(2, 1, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 2;
        }

        if (M & 1) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            boundary_cleanup_block(1, 1, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 1;
        }

        n_top += 1;
    }

    return 0;
}
