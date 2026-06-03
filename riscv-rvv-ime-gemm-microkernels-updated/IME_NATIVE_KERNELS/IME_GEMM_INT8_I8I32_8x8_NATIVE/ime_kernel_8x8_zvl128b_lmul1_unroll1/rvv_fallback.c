/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 */

/*
RVV INT8 GEMM Fallback Kernel
Variant: igemm_kernel_8x8_zvl128b_lmul1_unroll1_i8i32
Settings:
 LMUL=1 (m1)
 M=8
 N=8
 precision=int8 x int8 -> int32
 unroll_factor=1 (No unroll)
 Kernel Name: igemm_kernel_8x8_zvl128b_lmul1_unroll1_i8i32
*/
/*
 * Kernel dataflow summary: ime_kernel_8x8_zvl128b_lmul1_unroll1 RVV fallback
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
 * Step 2: Run the same 8x8 tile shape as the standalone RVV IGEMM family.
 * Step 3: Keep dot-product partial sums in RVV widening registers for full tiles.
 * Step 4: Route leftover rows or columns through the boundary cleanup path.
 * Step 5: Write alpha-scaled INT32 results back into column-major C.
 */

#include <stdint.h>
#include <riscv_vector.h>

typedef long BLASLONG;
int igemm_kernel_8x8_zvl128b_lmul1_unroll1_i8i32(BLASLONG M, BLASLONG N, BLASLONG K, int32_t alpha, int8_t *A, int8_t *B, int32_t *C, BLASLONG ldc)
{
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;
    if (K <= 0) {
        return 0;
    }
    if (__riscv_vsetvl_e8m1(8) < 8) {
        return -1;
    }
    /* Kernel section: main vectorized tile pass
     * Iterates over full tiles and keeps the hot K loop in RVV vector registers.
     */
    for (BLASLONG j = 0; j < N / 8; j += 1) {
        m_top = 0;
        for (BLASLONG i = 0; i < M / 8; i += 1) {
            size_t gvl = __riscv_vsetvl_e8m1(8);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result1 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result2 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result3 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result4 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result5 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result6 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result7 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                int8_t B1 = B[bi + 1];
                int8_t B2 = B[bi + 2];
                int8_t B3 = B[bi + 3];
                int8_t B4 = B[bi + 4];
                int8_t B5 = B[bi + 5];
                int8_t B6 = B[bi + 6];
                int8_t B7 = B[bi + 7];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                result1 = __riscv_vwmacc_vx_i32m4(result1, (int16_t)B1, A16, gvl);
                result2 = __riscv_vwmacc_vx_i32m4(result2, (int16_t)B2, A16, gvl);
                result3 = __riscv_vwmacc_vx_i32m4(result3, (int16_t)B3, A16, gvl);
                result4 = __riscv_vwmacc_vx_i32m4(result4, (int16_t)B4, A16, gvl);
                result5 = __riscv_vwmacc_vx_i32m4(result5, (int16_t)B5, A16, gvl);
                result6 = __riscv_vwmacc_vx_i32m4(result6, (int16_t)B6, A16, gvl);
                result7 = __riscv_vwmacc_vx_i32m4(result7, (int16_t)B7, A16, gvl);
                ai += 8;
                bi += 8;
            }
            int32_t tmp0[8] = {0};
            int32_t tmp1[8] = {0};
            int32_t tmp2[8] = {0};
            int32_t tmp3[8] = {0};
            int32_t tmp4[8] = {0};
            int32_t tmp5[8] = {0};
            int32_t tmp6[8] = {0};
            int32_t tmp7[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            __riscv_vse32_v_i32m4(tmp1, result1, gvl);
            __riscv_vse32_v_i32m4(tmp2, result2, gvl);
            __riscv_vse32_v_i32m4(tmp3, result3, gvl);
            __riscv_vse32_v_i32m4(tmp4, result4, gvl);
            __riscv_vse32_v_i32m4(tmp5, result5, gvl);
            __riscv_vse32_v_i32m4(tmp6, result6, gvl);
            __riscv_vse32_v_i32m4(tmp7, result7, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp1[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp2[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp3[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp4[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp5[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp6[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp7[r];
            }
            m_top += 8;
        }
        /* Kernel section: boundary cleanup for remaining rows in the main tile pass. */
        if (M & 4) {
            size_t gvl = __riscv_vsetvl_e8m1(4);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result1 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result2 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result3 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result4 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result5 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result6 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result7 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                int8_t B1 = B[bi + 1];
                int8_t B2 = B[bi + 2];
                int8_t B3 = B[bi + 3];
                int8_t B4 = B[bi + 4];
                int8_t B5 = B[bi + 5];
                int8_t B6 = B[bi + 6];
                int8_t B7 = B[bi + 7];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                result1 = __riscv_vwmacc_vx_i32m4(result1, (int16_t)B1, A16, gvl);
                result2 = __riscv_vwmacc_vx_i32m4(result2, (int16_t)B2, A16, gvl);
                result3 = __riscv_vwmacc_vx_i32m4(result3, (int16_t)B3, A16, gvl);
                result4 = __riscv_vwmacc_vx_i32m4(result4, (int16_t)B4, A16, gvl);
                result5 = __riscv_vwmacc_vx_i32m4(result5, (int16_t)B5, A16, gvl);
                result6 = __riscv_vwmacc_vx_i32m4(result6, (int16_t)B6, A16, gvl);
                result7 = __riscv_vwmacc_vx_i32m4(result7, (int16_t)B7, A16, gvl);
                ai += 4;
                bi += 8;
            }
            int32_t tmp0[8] = {0};
            int32_t tmp1[8] = {0};
            int32_t tmp2[8] = {0};
            int32_t tmp3[8] = {0};
            int32_t tmp4[8] = {0};
            int32_t tmp5[8] = {0};
            int32_t tmp6[8] = {0};
            int32_t tmp7[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            __riscv_vse32_v_i32m4(tmp1, result1, gvl);
            __riscv_vse32_v_i32m4(tmp2, result2, gvl);
            __riscv_vse32_v_i32m4(tmp3, result3, gvl);
            __riscv_vse32_v_i32m4(tmp4, result4, gvl);
            __riscv_vse32_v_i32m4(tmp5, result5, gvl);
            __riscv_vse32_v_i32m4(tmp6, result6, gvl);
            __riscv_vse32_v_i32m4(tmp7, result7, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp1[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp2[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp3[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp4[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp5[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp6[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp7[r];
            }
            m_top += 4;
        }
        if (M & 2) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[16] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                int8_t a1 = A[ai + 1];
                for (BLASLONG n = 0; n < 8; ++n) {
                    int8_t bn = B[bi + n];
                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;
                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;
                }
                ai += 2;
                bi += 8;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 8; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n * 2 + 0];
                C[ci + n * ldc + 1] += alpha * (int32_t)result[n * 2 + 1];
            }
            m_top += 2;
        }
        if (M & 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[8] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                for (BLASLONG n = 0; n < 8; ++n) {
                    int8_t bn = B[bi + n];
                    result[n] += (int32_t)a0 * (int32_t)bn;
                }
                ai += 1;
                bi += 8;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 8; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n];
            }
            m_top += 1;
        }
        n_top += 8;
    }
    /* Kernel section: boundary cleanup for the remaining N=4 column group. */
    if (N & 4) {
        m_top = 0;
        for (BLASLONG i = 0; i < M / 8; i += 1) {
            size_t gvl = __riscv_vsetvl_e8m1(8);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result1 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result2 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result3 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                int8_t B1 = B[bi + 1];
                int8_t B2 = B[bi + 2];
                int8_t B3 = B[bi + 3];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                result1 = __riscv_vwmacc_vx_i32m4(result1, (int16_t)B1, A16, gvl);
                result2 = __riscv_vwmacc_vx_i32m4(result2, (int16_t)B2, A16, gvl);
                result3 = __riscv_vwmacc_vx_i32m4(result3, (int16_t)B3, A16, gvl);
                ai += 8;
                bi += 4;
            }
            int32_t tmp0[8] = {0};
            int32_t tmp1[8] = {0};
            int32_t tmp2[8] = {0};
            int32_t tmp3[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            __riscv_vse32_v_i32m4(tmp1, result1, gvl);
            __riscv_vse32_v_i32m4(tmp2, result2, gvl);
            __riscv_vse32_v_i32m4(tmp3, result3, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp1[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp2[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp3[r];
            }
            m_top += 8;
        }
        if (M & 4) {
            size_t gvl = __riscv_vsetvl_e8m1(4);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result1 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result2 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result3 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                int8_t B1 = B[bi + 1];
                int8_t B2 = B[bi + 2];
                int8_t B3 = B[bi + 3];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                result1 = __riscv_vwmacc_vx_i32m4(result1, (int16_t)B1, A16, gvl);
                result2 = __riscv_vwmacc_vx_i32m4(result2, (int16_t)B2, A16, gvl);
                result3 = __riscv_vwmacc_vx_i32m4(result3, (int16_t)B3, A16, gvl);
                ai += 4;
                bi += 4;
            }
            int32_t tmp0[8] = {0};
            int32_t tmp1[8] = {0};
            int32_t tmp2[8] = {0};
            int32_t tmp3[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            __riscv_vse32_v_i32m4(tmp1, result1, gvl);
            __riscv_vse32_v_i32m4(tmp2, result2, gvl);
            __riscv_vse32_v_i32m4(tmp3, result3, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp1[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp2[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp3[r];
            }
            m_top += 4;
        }
        if (M & 2) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[8] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                int8_t a1 = A[ai + 1];
                for (BLASLONG n = 0; n < 4; ++n) {
                    int8_t bn = B[bi + n];
                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;
                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;
                }
                ai += 2;
                bi += 4;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 4; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n * 2 + 0];
                C[ci + n * ldc + 1] += alpha * (int32_t)result[n * 2 + 1];
            }
            m_top += 2;
        }
        if (M & 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[4] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                for (BLASLONG n = 0; n < 4; ++n) {
                    int8_t bn = B[bi + n];
                    result[n] += (int32_t)a0 * (int32_t)bn;
                }
                ai += 1;
                bi += 4;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 4; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n];
            }
            m_top += 1;
        }
        n_top += 4;
    }
    /* Kernel section: boundary cleanup for the remaining N=2 column group. */
    if (N & 2) {
        m_top = 0;
        for (BLASLONG i = 0; i < M / 8; i += 1) {
            size_t gvl = __riscv_vsetvl_e8m1(8);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result1 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                int8_t B1 = B[bi + 1];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                result1 = __riscv_vwmacc_vx_i32m4(result1, (int16_t)B1, A16, gvl);
                ai += 8;
                bi += 2;
            }
            int32_t tmp0[8] = {0};
            int32_t tmp1[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            __riscv_vse32_v_i32m4(tmp1, result1, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp1[r];
            }
            m_top += 8;
        }
        if (M & 4) {
            size_t gvl = __riscv_vsetvl_e8m1(4);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
            vint32m4_t result1 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                int8_t B1 = B[bi + 1];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                result1 = __riscv_vwmacc_vx_i32m4(result1, (int16_t)B1, A16, gvl);
                ai += 4;
                bi += 2;
            }
            int32_t tmp0[8] = {0};
            int32_t tmp1[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            __riscv_vse32_v_i32m4(tmp1, result1, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            ci += ldc;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp1[r];
            }
            m_top += 4;
        }
        if (M & 2) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[4] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                int8_t a1 = A[ai + 1];
                for (BLASLONG n = 0; n < 2; ++n) {
                    int8_t bn = B[bi + n];
                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;
                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;
                }
                ai += 2;
                bi += 2;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 2; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n * 2 + 0];
                C[ci + n * ldc + 1] += alpha * (int32_t)result[n * 2 + 1];
            }
            m_top += 2;
        }
        if (M & 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[2] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                for (BLASLONG n = 0; n < 2; ++n) {
                    int8_t bn = B[bi + n];
                    result[n] += (int32_t)a0 * (int32_t)bn;
                }
                ai += 1;
                bi += 2;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 2; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n];
            }
            m_top += 1;
        }
        n_top += 2;
    }
    /* Kernel section: boundary cleanup for the remaining N=1 column group. */
    if (N & 1) {
        m_top = 0;
        for (BLASLONG i = 0; i < M / 8; i += 1) {
            size_t gvl = __riscv_vsetvl_e8m1(8);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                ai += 8;
                bi += 1;
            }
            int32_t tmp0[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 8; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            m_top += 8;
        }
        if (M & 4) {
            size_t gvl = __riscv_vsetvl_e8m1(4);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            vint32m4_t result0 = __riscv_vmv_v_x_i32m4(0, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t B0 = B[bi + 0];
                vint8m1_t A0 = __riscv_vle8_v_i8m1(&A[ai], gvl);
                vint16m2_t A16 = __riscv_vsext_vf2_i16m2(A0, gvl);
                result0 = __riscv_vwmacc_vx_i32m4(result0, (int16_t)B0, A16, gvl);
                ai += 4;
                bi += 1;
            }
            int32_t tmp0[8] = {0};
            __riscv_vse32_v_i32m4(tmp0, result0, gvl);
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG r = 0; r < 4; ++r) {
                C[ci + r] += alpha * tmp0[r];
            }
            m_top += 4;
        }
        if (M & 2) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[2] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                int8_t a1 = A[ai + 1];
                for (BLASLONG n = 0; n < 1; ++n) {
                    int8_t bn = B[bi + n];
                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;
                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;
                }
                ai += 2;
                bi += 1;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 1; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n * 2 + 0];
                C[ci + n * ldc + 1] += alpha * (int32_t)result[n * 2 + 1];
            }
            m_top += 2;
        }
        if (M & 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            int64_t result[1] = {0};
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                int8_t a0 = A[ai + 0];
                for (BLASLONG n = 0; n < 1; ++n) {
                    int8_t bn = B[bi + n];
                    result[n] += (int32_t)a0 * (int32_t)bn;
                }
                ai += 1;
                bi += 1;
            }
            BLASLONG ci = n_top * ldc + m_top;
            for (BLASLONG n = 0; n < 1; ++n) {
                C[ci + n * ldc + 0] += alpha * (int32_t)result[n];
            }
            m_top += 1;
        }
        n_top += 1;
    }
    return 0;
}
