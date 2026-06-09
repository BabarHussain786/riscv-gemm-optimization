/*
RVV INT8 GEMM Kernel
Tile: 8x4
Variant: igemm_kernel_8x4_zvl256b_lmulmf4_unroll1
LMUL label: mf4
RVV LMUL mapping: direct
Unroll: 1
Math contract: 8x4 INT8 tiled path with full-tile processing and boundary cleanup
*/

/*
 * Kernel dataflow summary: igemm_kernel_8x4_zvl256b_lmulmf4_unroll1
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
#define CNAME igemm_kernel_8x4_zvl256b_lmulmf4_unroll1
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
    int64_t acc[8] = {0};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 1
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
    const size_t gvl = __riscv_vsetvl_e8mf4(4);
    vint8mf4_t a8_0;
    vint8mf4_t a8_1;
    vint16mf2_t a16_0;
    vint16mf2_t a16_1;
    vint32m1_t r00 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r01 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r02 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r03 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r10 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r11 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r12 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r13 = __riscv_vmv_v_x_i32m1(0, gvl);

    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 1
    for (BLASLONG k = 0; k < K; ++k) {
        a8_0 = __riscv_vle8_v_i8mf4(&Ablk[ai], gvl);
        a8_1 = __riscv_vle8_v_i8mf4(&Ablk[ai + gvl], gvl);
        a16_0 = __riscv_vsext_vf2_i16mf2(a8_0, gvl);
        a16_1 = __riscv_vsext_vf2_i16mf2(a8_1, gvl);
        ai += 8;

        if (cols > 0) {
            const int16_t b = (int16_t)Bblk[bi + 0];
            r00 = __riscv_vwmacc_vx_i32m1(r00, b, a16_0, gvl);
            r10 = __riscv_vwmacc_vx_i32m1(r10, b, a16_1, gvl);
        }
        if (cols > 1) {
            const int16_t b = (int16_t)Bblk[bi + 1];
            r01 = __riscv_vwmacc_vx_i32m1(r01, b, a16_0, gvl);
            r11 = __riscv_vwmacc_vx_i32m1(r11, b, a16_1, gvl);
        }
        if (cols > 2) {
            const int16_t b = (int16_t)Bblk[bi + 2];
            r02 = __riscv_vwmacc_vx_i32m1(r02, b, a16_0, gvl);
            r12 = __riscv_vwmacc_vx_i32m1(r12, b, a16_1, gvl);
        }
        if (cols > 3) {
            const int16_t b = (int16_t)Bblk[bi + 3];
            r03 = __riscv_vwmacc_vx_i32m1(r03, b, a16_0, gvl);
            r13 = __riscv_vwmacc_vx_i32m1(r13, b, a16_1, gvl);
        }
        bi += cols;
    }

    int32_t tmp0[4], tmp1[4];
    if (cols > 0) {
        __riscv_vse32_v_i32m1(tmp0, r00, gvl);
        __riscv_vse32_v_i32m1(tmp1, r10, gvl);
        int32_t *c_col = &Cblk[0 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
        }
    }
    if (cols > 1) {
        __riscv_vse32_v_i32m1(tmp0, r01, gvl);
        __riscv_vse32_v_i32m1(tmp1, r11, gvl);
        int32_t *c_col = &Cblk[1 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
        }
    }
    if (cols > 2) {
        __riscv_vse32_v_i32m1(tmp0, r02, gvl);
        __riscv_vse32_v_i32m1(tmp1, r12, gvl);
        int32_t *c_col = &Cblk[2 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
        }
    }
    if (cols > 3) {
        __riscv_vse32_v_i32m1(tmp0, r03, gvl);
        __riscv_vse32_v_i32m1(tmp1, r13, gvl);
        int32_t *c_col = &Cblk[3 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
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
    const size_t gvl = __riscv_vsetvl_e8mf4(4);
    vint8mf4_t a8;
    vint16mf2_t a16;
    vint32m1_t r0 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r1 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r2 = __riscv_vmv_v_x_i32m1(0, gvl);
    vint32m1_t r3 = __riscv_vmv_v_x_i32m1(0, gvl);

    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 1
    for (BLASLONG k = 0; k < K; ++k) {
        a8 = __riscv_vle8_v_i8mf4(&Ablk[ai], gvl);
        a16 = __riscv_vsext_vf2_i16mf2(a8, gvl);
        ai += 4;

        if (cols > 0) {
            const int16_t b = (int16_t)Bblk[bi + 0];
            r0 = __riscv_vwmacc_vx_i32m1(r0, b, a16, gvl);
        }
        if (cols > 1) {
            const int16_t b = (int16_t)Bblk[bi + 1];
            r1 = __riscv_vwmacc_vx_i32m1(r1, b, a16, gvl);
        }
        if (cols > 2) {
            const int16_t b = (int16_t)Bblk[bi + 2];
            r2 = __riscv_vwmacc_vx_i32m1(r2, b, a16, gvl);
        }
        if (cols > 3) {
            const int16_t b = (int16_t)Bblk[bi + 3];
            r3 = __riscv_vwmacc_vx_i32m1(r3, b, a16, gvl);
        }
        bi += cols;
    }

    int32_t tmp[4];
    if (cols > 0) {
        __riscv_vse32_v_i32m1(tmp, r0, gvl);
        int32_t *c_col = &Cblk[0 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
        }
    }
    if (cols > 1) {
        __riscv_vse32_v_i32m1(tmp, r1, gvl);
        int32_t *c_col = &Cblk[1 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
        }
    }
    if (cols > 2) {
        __riscv_vse32_v_i32m1(tmp, r2, gvl);
        int32_t *c_col = &Cblk[2 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
        }
    }
    if (cols > 3) {
        __riscv_vse32_v_i32m1(tmp, r3, gvl);
        int32_t *c_col = &Cblk[3 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
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

    if (__riscv_vsetvl_e8mf4(4) < 4) {
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
