/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 */

/*
RVV INT8 GEMM Fallback Kernel
Variant: igemm_kernel_8x4_zvl256b_lmul1_unroll1_i8i32
Settings:
 input_LMUL=1 (m1)
 widening_chain=INT8 LMUL 1 -> INT16 LMUL 2 -> INT32 LMUL 4
 M=8
 N=4
 precision=int8 x int8 -> int32
 unroll_factor=1
 Kernel Name: igemm_kernel_8x4_zvl256b_lmul1_unroll1_i8i32
*/
/*
 * Kernel dataflow summary: ime_kernel_8x4_zvl256b_lmul1_unroll1 RVV fallback
 * Input A: local packed INT8 A panel prepared by the IME wrapper.
 * Input B: local packed INT8 B panel prepared by the IME wrapper.
 * Accumulation: INT32 RVV widening accumulators across K.
 * Output C: column-major INT32 tile updated with alpha-scaled dot products.
 */
/*
 * Architecture overview
 * ---------------------
 * Purpose: self-contained RVV fallback used by the matching IME kernel folder.
 * Step 1: Consume packed INT8 A/B panels prepared by the IME wrapper.
 * Step 2: Run the same 8x4 tile shape as the standalone RVV IGEMM family.
 * Step 3: Keep dot-product partial sums in RVV widening registers for full tiles.
 * Step 4: Route leftover rows or columns through the boundary cleanup path.
 * Step 5: Write alpha-scaled INT32 results back into column-major C.
 */
/*
 * -----------------------------------------------------------------------------
 * ROADMAP 0: File setup and RVV contract
 * -----------------------------------------------------------------------------
 * This RVV file is the fallback path paired with the IME kernel.
 * It receives packed INT8 A/B panels, computes INT8 x INT8 -> INT32 GEMM
 * using RVV widening multiply-accumulate, and writes back into column-major C.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 1: Datatypes and RVV helper macros
 * -----------------------------------------------------------------------------
 * These names make the kernel easier to read: INT8 inputs, INT32 outputs,
 * RVV load, RVV sign-extension, RVV widening MAC, and RVV INT32 store.
 */
typedef long BLASLONG;
typedef int8_t GEMM_I8;
typedef int32_t GEMM_I32;

/* Select active INT8 lanes for the current row chunk. */
#define INPUT_VSETVL(n) __riscv_vsetvl_e8m1(n)
typedef vint8m1_t input_i8_t;
typedef vint16m2_t wide_i16_t;
typedef vint32m4_t acc_i32_t;
/* Load packed INT8 A values from memory into an RVV register. */
#define INPUT_VLE8(p, vl) __riscv_vle8_v_i8m1(p, vl)
/* Sign-extend INT8 A values to INT16 before widening accumulation. */
#define WIDEN_I16(v, vl) __riscv_vsext_vf2_i16m2(v, vl)
/* Create an INT32 accumulator vector initialized to zero. */
#define ACC_ZERO(vl) __riscv_vmv_v_x_i32m4(0, vl)
/* RVV widening MAC: acc += scalar_B * vector_A. */
#define ACC_WMACC(acc, b, a, vl) __riscv_vwmacc_vx_i32m4(acc, b, a, vl)
/* Store INT32 accumulator values into the temporary output array. */
#define ACC_STORE(p, v, vl) __riscv_vse32_v_i32m4(p, v, vl)

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 2: Safe INT32 update helpers
 * -----------------------------------------------------------------------------
 * These helpers preserve normal 32-bit wraparound behavior when alpha-scaled
 * dot-product values are added into the output matrix C.
 */
static inline uint32_t u32_from_i32(GEMM_I32 x)
{
    /* y stores the same bits as x, but viewed as unsigned. */
    uint32_t y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static inline GEMM_I32 i32_from_u32(uint32_t x)
{
    /* y stores the same bits as x, but viewed as signed. */
    GEMM_I32 y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

static inline GEMM_I32 add_scaled_wrap_i32(GEMM_I32 dst, int64_t value, GEMM_I32 alpha)
{
    /* Scale the dot-product sum by alpha before adding it to C. */
    uint32_t scaled = (uint32_t)((int64_t)alpha * value);
    /* Add with explicit 32-bit wraparound semantics. */
    return i32_from_u32(u32_from_i32(dst) + scaled);
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 3: Scalar cleanup for leftover rows/columns
 * -----------------------------------------------------------------------------
 * Full tiles use RVV. Small boundary pieces use scalar dot products but keep
 * the same packed A/B layout so the math remains identical.
 */
static inline BLASLONG packed_row_block(BLASLONG left)
{
    /* Prefer the largest remaining row block. */
    if (left >= 8) return 8;
    /* If 8 rows are not available, try 4 rows. */
    if (left >= 4) return 4;
    /* If 4 rows are not available, try 2 rows. */
    if (left >= 2) return 2;
    return 1;
}

/*
 * Packed scalar cleanup for N tails.
 * Ablk uses the same packed layout as the RVV full-tile path:
 *   Ablk[k * rows + r]
 * Bblk uses the local packed column-tail layout:
 *   Bblk[k * cols + c]
 */
static void scalar_packed_block(BLASLONG rows, BLASLONG cols, BLASLONG K,
                                GEMM_I32 alpha, const GEMM_I8 *Ablk,
                                const GEMM_I8 *Bblk, GEMM_I32 *Cblk,
                                BLASLONG ldc)
{
    /* Visit each output column in this small scalar block. */
    for (BLASLONG c = 0; c < cols; ++c) {
        /* Visit each output row in the current column. */
        for (BLASLONG r = 0; r < rows; ++r) {
            /* sum holds one dot product before it is added into C. */
            int64_t sum = 0;

#pragma GCC unroll 1
                /* Walk across K and accumulate dot products. */
            for (BLASLONG k = 0; k < K; ++k) {
                /* Add A(row,k) * B(k,column) into the dot product. */
                sum += (GEMM_I32)Ablk[k * rows + r] *
                       (GEMM_I32)Bblk[k * cols + c];
            }

            /* Store alpha-scaled scalar result back to C. */
            Cblk[c * ldc + r] = add_scaled_wrap_i32(Cblk[c * ldc + r], sum, alpha);
        }
    }
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 4: Store one full 8x4 RVV tile into C
 * -----------------------------------------------------------------------------
 * The RVV path stores four temporary INT32 columns, then this helper adds them
 * into the real column-major output matrix using C[col * ldc + row].
 */
static inline void scatter_8x4(BLASLONG n_top, BLASLONG m_top, GEMM_I32 alpha,
                               GEMM_I32 *C, BLASLONG ldc,
                               const GEMM_I32 out0[8], const GEMM_I32 out1[8],
                               const GEMM_I32 out2[8], const GEMM_I32 out3[8])
{
    /* ci is the start address of this C tile in column-major layout. */
    BLASLONG ci = n_top * ldc + m_top;
    /* out maps each temporary output column to one pointer. */
    const GEMM_I32 *out[4] = {out0, out1, out2, out3};

    /* Store four output columns. */
    for (BLASLONG c = 0; c < 4; ++c) {
        /* Store eight output rows for this column. */
        for (BLASLONG r = 0; r < 8; ++r) {
            C[ci + c * ldc + r] = add_scaled_wrap_i32(C[ci + c * ldc + r],
                                                       out[c][r],
                                                       alpha);
        }
    }
}

/*
 * -----------------------------------------------------------------------------
 * ROADMAP 5: Main RVV INT8 GEMM fallback driver
 * -----------------------------------------------------------------------------
 * Step 1: Start from the first row and column.
 * Step 2: Process full 8x4 output tiles using RVV widening MAC.
 * Step 3: Handle leftover M rows for each 4-column block.
 * Step 4: Handle leftover N columns using scalar packed cleanup.
 * Step 5: Return success after all C tiles are updated.
 */
int igemm_kernel_8x4_zvl256b_lmul1_unroll1_i8i32(
    BLASLONG M, BLASLONG N, BLASLONG K,
    GEMM_I32 alpha, GEMM_I8 *A, GEMM_I8 *B, GEMM_I32 *C, BLASLONG ldc)
{
    /* m_top is the first output row of the current tile. */
    BLASLONG m_top = 0;
    /* n_top is the first output column of the current tile. */
    BLASLONG n_top = 0;

    /* If K is zero, there is no dot product to compute. */
    if (K <= 0) {
        return 0;
    }

    /* Process full 4-column groups first. */
    for (BLASLONG j = 0; j < N / 4; ++j) {
        /* Restart from row zero for this 4-column group. */
        m_top = 0;

        /* Process full 8-row blocks with RVV. */
        for (BLASLONG i = 0; i < M / 8; ++i) {
            /* Temporary INT32 outputs for the four C columns. */
            GEMM_I32 out0[8] = {0};
            GEMM_I32 out1[8] = {0};
            GEMM_I32 out2[8] = {0};
            GEMM_I32 out3[8] = {0};

            /* base moves through the 8 rows according to active RVV lanes. */
            for (BLASLONG base = 0; base < 8; ) {
                /* gvl tells how many INT8 lanes are active in this row chunk. */
                size_t gvl = INPUT_VSETVL((size_t)(8 - base));
                /* Four INT32 accumulators, one for each output column. */
                acc_i32_t acc0 = ACC_ZERO(gvl);
                acc_i32_t acc1 = ACC_ZERO(gvl);
                acc_i32_t acc2 = ACC_ZERO(gvl);
                acc_i32_t acc3 = ACC_ZERO(gvl);

#pragma GCC unroll 1
                for (BLASLONG k = 0; k < K; ++k) {
                    /* A is packed by row block, then K, then row inside block. */
                    BLASLONG ai = m_top * K + k * 8 + base;
                    /* B is packed by column block, then K, then column inside block. */
                    BLASLONG bi = n_top * K + k * 4;
                    /* Load four B scalars for four output columns. */
                    GEMM_I8 b0 = B[bi + 0];
                    GEMM_I8 b1 = B[bi + 1];
                    GEMM_I8 b2 = B[bi + 2];
                    GEMM_I8 b3 = B[bi + 3];
                    /* Load one vector chunk of A rows. */
                    input_i8_t a8 = INPUT_VLE8(&A[ai], gvl);
                    /* Widen A from INT8 to INT16 before RVV widening MAC. */
                    wide_i16_t a16 = WIDEN_I16(a8, gvl);

                    /* Accumulate four columns: acc += B scalar * A vector. */
                    acc0 = ACC_WMACC(acc0, (int16_t)b0, a16, gvl);
                    acc1 = ACC_WMACC(acc1, (int16_t)b1, a16, gvl);
                    acc2 = ACC_WMACC(acc2, (int16_t)b2, a16, gvl);
                    acc3 = ACC_WMACC(acc3, (int16_t)b3, a16, gvl);
                }

                /* Store this row chunk into temporary output arrays. */
                ACC_STORE(&out0[base], acc0, gvl);
                ACC_STORE(&out1[base], acc1, gvl);
                ACC_STORE(&out2[base], acc2, gvl);
                ACC_STORE(&out3[base], acc3, gvl);
                /* Move to the next active row chunk. */
                base += (BLASLONG)gvl;
            }

            /* Add the complete 8x4 temporary tile back into C. */
            scatter_8x4(n_top, m_top, alpha, C, ldc, out0, out1, out2, out3);
            /* Move to the next 8-row tile. */
            m_top += 8;
        }

        /* Handle leftover 4 rows for this 4-column group. */
        if (M & 4) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            GEMM_I32 r00 = 0, r01 = 0, r02 = 0, r03 = 0;
            GEMM_I32 r10 = 0, r11 = 0, r12 = 0, r13 = 0;
            GEMM_I32 r20 = 0, r21 = 0, r22 = 0, r23 = 0;
            GEMM_I32 r30 = 0, r31 = 0, r32 = 0, r33 = 0;

#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                GEMM_I8 a0 = A[ai + 0], a1 = A[ai + 1], a2 = A[ai + 2], a3 = A[ai + 3];
                GEMM_I8 b0 = B[bi + 0], b1 = B[bi + 1], b2 = B[bi + 2], b3 = B[bi + 3];
                ai += 4;
                bi += 4;
                r00 += (GEMM_I32)a0 * (GEMM_I32)b0; r01 += (GEMM_I32)a1 * (GEMM_I32)b0;
                r02 += (GEMM_I32)a2 * (GEMM_I32)b0; r03 += (GEMM_I32)a3 * (GEMM_I32)b0;
                r10 += (GEMM_I32)a0 * (GEMM_I32)b1; r11 += (GEMM_I32)a1 * (GEMM_I32)b1;
                r12 += (GEMM_I32)a2 * (GEMM_I32)b1; r13 += (GEMM_I32)a3 * (GEMM_I32)b1;
                r20 += (GEMM_I32)a0 * (GEMM_I32)b2; r21 += (GEMM_I32)a1 * (GEMM_I32)b2;
                r22 += (GEMM_I32)a2 * (GEMM_I32)b2; r23 += (GEMM_I32)a3 * (GEMM_I32)b2;
                r30 += (GEMM_I32)a0 * (GEMM_I32)b3; r31 += (GEMM_I32)a1 * (GEMM_I32)b3;
                r32 += (GEMM_I32)a2 * (GEMM_I32)b3; r33 += (GEMM_I32)a3 * (GEMM_I32)b3;
            }
            BLASLONG ci = n_top * ldc + m_top;
            GEMM_I32 out[4][4] = {
                {r00, r01, r02, r03},
                {r10, r11, r12, r13},
                {r20, r21, r22, r23},
                {r30, r31, r32, r33}
            };
            for (BLASLONG c = 0; c < 4; ++c) {
                for (BLASLONG r = 0; r < 4; ++r) {
                    C[ci + c * ldc + r] = add_scaled_wrap_i32(C[ci + c * ldc + r],
                                                               out[c][r],
                                                               alpha);
                }
            }
            m_top += 4;
        }

        /* Handle leftover 2 rows for this 4-column group. */
        if (M & 2) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            GEMM_I32 r00 = 0, r01 = 0, r10 = 0, r11 = 0, r20 = 0, r21 = 0, r30 = 0, r31 = 0;

#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                GEMM_I8 a0 = A[ai + 0], a1 = A[ai + 1];
                GEMM_I8 b0 = B[bi + 0], b1 = B[bi + 1], b2 = B[bi + 2], b3 = B[bi + 3];
                ai += 2;
                bi += 4;
                r00 += (GEMM_I32)a0 * (GEMM_I32)b0; r01 += (GEMM_I32)a1 * (GEMM_I32)b0;
                r10 += (GEMM_I32)a0 * (GEMM_I32)b1; r11 += (GEMM_I32)a1 * (GEMM_I32)b1;
                r20 += (GEMM_I32)a0 * (GEMM_I32)b2; r21 += (GEMM_I32)a1 * (GEMM_I32)b2;
                r30 += (GEMM_I32)a0 * (GEMM_I32)b3; r31 += (GEMM_I32)a1 * (GEMM_I32)b3;
            }
            BLASLONG ci = n_top * ldc + m_top;
            GEMM_I32 out[4][2] = {
                {r00, r01},
                {r10, r11},
                {r20, r21},
                {r30, r31}
            };
            for (BLASLONG c = 0; c < 4; ++c) {
                for (BLASLONG r = 0; r < 2; ++r) {
                    C[ci + c * ldc + r] = add_scaled_wrap_i32(C[ci + c * ldc + r],
                                                               out[c][r],
                                                               alpha);
                }
            }
            m_top += 2;
        }

        /* Handle leftover 1 row for this 4-column group. */
        if (M & 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            GEMM_I32 r00 = 0, r10 = 0, r20 = 0, r30 = 0;
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                GEMM_I8 a0 = A[ai];
                GEMM_I8 b0 = B[bi + 0], b1 = B[bi + 1], b2 = B[bi + 2], b3 = B[bi + 3];
                ai += 1;
                bi += 4;
                r00 += (GEMM_I32)a0 * (GEMM_I32)b0;
                r10 += (GEMM_I32)a0 * (GEMM_I32)b1;
                r20 += (GEMM_I32)a0 * (GEMM_I32)b2;
                r30 += (GEMM_I32)a0 * (GEMM_I32)b3;
            }
            BLASLONG ci = n_top * ldc + m_top;
            GEMM_I32 out[4] = {r00, r10, r20, r30};
            for (BLASLONG c = 0; c < 4; ++c) {
                C[ci + c * ldc] = add_scaled_wrap_i32(C[ci + c * ldc],
                                                       out[c],
                                                       alpha);
            }
            m_top += 1;
        }

        /* Move to the next 4-column output group. */
        n_top += 4;
    }

    /* Handle leftover 2 columns when N is not divisible by 4. */
    if (N & 2) {
        /* Start from row zero for this 2-column tail block. */
        m_top = 0;
        while (m_top < M) {
            BLASLONG rows = packed_row_block(M - m_top);
            scalar_packed_block(rows, 2, K, alpha,
                                &A[m_top * K],
                                &B[n_top * K],
                                &C[n_top * ldc + m_top],
                                ldc);
            m_top += rows;
        }
        n_top += 2;
    }

    /* Handle leftover 1 column at the end. */
    if (N & 1) {
        m_top = 0;
        while (m_top < M) {
            BLASLONG rows = packed_row_block(M - m_top);
            scalar_packed_block(rows, 1, K, alpha,
                                &A[m_top * K],
                                &B[n_top * K],
                                &C[n_top * ldc + m_top],
                                ldc);
            m_top += rows;
        }
    }

    /* All full tiles and tail blocks are finished. */
    return 0;
}
