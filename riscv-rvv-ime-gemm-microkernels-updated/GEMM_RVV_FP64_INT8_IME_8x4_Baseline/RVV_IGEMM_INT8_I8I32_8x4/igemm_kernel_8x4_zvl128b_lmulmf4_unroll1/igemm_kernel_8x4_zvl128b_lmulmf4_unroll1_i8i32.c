/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 */

/*
RVV INT8 GEMM Kernel
Variant: igemm_kernel_8x4_zvl128b_lmulmf4_unroll1
Settings:
 input_LMUL=1/4 (mf4)
 widening_chain=INT8 LMUL 1/4 -> INT16 LMUL 1/2 -> INT32 LMUL 1
 M=8
 N=4
 precision=int8 x int8 -> int32
 unroll_factor=1
 Kernel Name: igemm_kernel_8x4_zvl128b_lmulmf4_unroll1
*/
/*
 * Kernel dataflow summary: igemm_kernel_8x4_zvl128b_lmulmf4_unroll1_i8i32
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
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

typedef long BLASLONG;
typedef int8_t GEMM_I8;
typedef int32_t GEMM_I32;

#define INPUT_VSETVL(n) __riscv_vsetvl_e8mf4(n)
typedef vint8mf4_t input_i8_t;
typedef vint16mf2_t wide_i16_t;
typedef vint32m1_t acc_i32_t;
#define INPUT_VLE8(p, vl) __riscv_vle8_v_i8mf4(p, vl)
#define WIDEN_I16(v, vl) __riscv_vsext_vf2_i16mf2(v, vl)
#define ACC_ZERO(vl) __riscv_vmv_v_x_i32m1(0, vl)
#define ACC_WMACC(acc, b, a, vl) __riscv_vwmacc_vx_i32m1(acc, b, a, vl)
#define ACC_STORE(p, v, vl) __riscv_vse32_v_i32m1(p, v, vl)

static inline void scatter_8x4(BLASLONG n_top, BLASLONG m_top, GEMM_I32 alpha,
                               GEMM_I32 *C, BLASLONG ldc,
                               const GEMM_I32 out0[8], const GEMM_I32 out1[8],
                               const GEMM_I32 out2[8], const GEMM_I32 out3[8])
{
    BLASLONG ci = n_top * ldc + m_top;
    C[ci + 0 * ldc + 0] += alpha * out0[0]; C[ci + 0 * ldc + 1] += alpha * out0[1];
    C[ci + 0 * ldc + 2] += alpha * out0[2]; C[ci + 0 * ldc + 3] += alpha * out0[3];
    C[ci + 0 * ldc + 4] += alpha * out0[4]; C[ci + 0 * ldc + 5] += alpha * out0[5];
    C[ci + 0 * ldc + 6] += alpha * out0[6]; C[ci + 0 * ldc + 7] += alpha * out0[7];

    C[ci + 1 * ldc + 0] += alpha * out1[0]; C[ci + 1 * ldc + 1] += alpha * out1[1];
    C[ci + 1 * ldc + 2] += alpha * out1[2]; C[ci + 1 * ldc + 3] += alpha * out1[3];
    C[ci + 1 * ldc + 4] += alpha * out1[4]; C[ci + 1 * ldc + 5] += alpha * out1[5];
    C[ci + 1 * ldc + 6] += alpha * out1[6]; C[ci + 1 * ldc + 7] += alpha * out1[7];

    C[ci + 2 * ldc + 0] += alpha * out2[0]; C[ci + 2 * ldc + 1] += alpha * out2[1];
    C[ci + 2 * ldc + 2] += alpha * out2[2]; C[ci + 2 * ldc + 3] += alpha * out2[3];
    C[ci + 2 * ldc + 4] += alpha * out2[4]; C[ci + 2 * ldc + 5] += alpha * out2[5];
    C[ci + 2 * ldc + 6] += alpha * out2[6]; C[ci + 2 * ldc + 7] += alpha * out2[7];

    C[ci + 3 * ldc + 0] += alpha * out3[0]; C[ci + 3 * ldc + 1] += alpha * out3[1];
    C[ci + 3 * ldc + 2] += alpha * out3[2]; C[ci + 3 * ldc + 3] += alpha * out3[3];
    C[ci + 3 * ldc + 4] += alpha * out3[4]; C[ci + 3 * ldc + 5] += alpha * out3[5];
    C[ci + 3 * ldc + 6] += alpha * out3[6]; C[ci + 3 * ldc + 7] += alpha * out3[7];
}

int igemm_kernel_8x4_zvl128b_lmulmf4_unroll1_i8i32(
    BLASLONG M, BLASLONG N, BLASLONG K,
    GEMM_I32 alpha, GEMM_I8 *A, GEMM_I8 *B, GEMM_I32 *C, BLASLONG ldc)
{
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;

    if (K <= 0) {
        return 0;
    }

    for (BLASLONG j = 0; j < N / 4; ++j) {
        m_top = 0;

        for (BLASLONG i = 0; i < M / 8; ++i) {
            GEMM_I32 out0[8] = {0};
            GEMM_I32 out1[8] = {0};
            GEMM_I32 out2[8] = {0};
            GEMM_I32 out3[8] = {0};

            for (BLASLONG base = 0; base < 8; ) {
                size_t gvl = INPUT_VSETVL((size_t)(8 - base));
                acc_i32_t acc0 = ACC_ZERO(gvl);
                acc_i32_t acc1 = ACC_ZERO(gvl);
                acc_i32_t acc2 = ACC_ZERO(gvl);
                acc_i32_t acc3 = ACC_ZERO(gvl);

#pragma GCC unroll 1
                for (BLASLONG k = 0; k < K; ++k) {
                    BLASLONG ai = m_top * K + k * 8 + base;
                    BLASLONG bi = n_top * K + k * 4;
                    GEMM_I8 b0 = B[bi + 0];
                    GEMM_I8 b1 = B[bi + 1];
                    GEMM_I8 b2 = B[bi + 2];
                    GEMM_I8 b3 = B[bi + 3];
                    input_i8_t a8 = INPUT_VLE8(&A[ai], gvl);
                    wide_i16_t a16 = WIDEN_I16(a8, gvl);

                    acc0 = ACC_WMACC(acc0, (int16_t)b0, a16, gvl);
                    acc1 = ACC_WMACC(acc1, (int16_t)b1, a16, gvl);
                    acc2 = ACC_WMACC(acc2, (int16_t)b2, a16, gvl);
                    acc3 = ACC_WMACC(acc3, (int16_t)b3, a16, gvl);
                }

                ACC_STORE(&out0[base], acc0, gvl);
                ACC_STORE(&out1[base], acc1, gvl);
                ACC_STORE(&out2[base], acc2, gvl);
                ACC_STORE(&out3[base], acc3, gvl);
                base += (BLASLONG)gvl;
            }

            scatter_8x4(n_top, m_top, alpha, C, ldc, out0, out1, out2, out3);
            m_top += 8;
        }

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
            C[ci + 0 * ldc + 0] += alpha * r00; C[ci + 0 * ldc + 1] += alpha * r01;
            C[ci + 0 * ldc + 2] += alpha * r02; C[ci + 0 * ldc + 3] += alpha * r03;
            C[ci + 1 * ldc + 0] += alpha * r10; C[ci + 1 * ldc + 1] += alpha * r11;
            C[ci + 1 * ldc + 2] += alpha * r12; C[ci + 1 * ldc + 3] += alpha * r13;
            C[ci + 2 * ldc + 0] += alpha * r20; C[ci + 2 * ldc + 1] += alpha * r21;
            C[ci + 2 * ldc + 2] += alpha * r22; C[ci + 2 * ldc + 3] += alpha * r23;
            C[ci + 3 * ldc + 0] += alpha * r30; C[ci + 3 * ldc + 1] += alpha * r31;
            C[ci + 3 * ldc + 2] += alpha * r32; C[ci + 3 * ldc + 3] += alpha * r33;
            m_top += 4;
        }

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
            C[ci + 0 * ldc + 0] += alpha * r00; C[ci + 0 * ldc + 1] += alpha * r01;
            C[ci + 1 * ldc + 0] += alpha * r10; C[ci + 1 * ldc + 1] += alpha * r11;
            C[ci + 2 * ldc + 0] += alpha * r20; C[ci + 2 * ldc + 1] += alpha * r21;
            C[ci + 3 * ldc + 0] += alpha * r30; C[ci + 3 * ldc + 1] += alpha * r31;
            m_top += 2;
        }

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
            C[ci + 0 * ldc] += alpha * r00;
            C[ci + 1 * ldc] += alpha * r10;
            C[ci + 2 * ldc] += alpha * r20;
            C[ci + 3 * ldc] += alpha * r30;
            m_top += 1;
        }

        n_top += 4;
    }

    if (N & 2) {
        m_top = 0;
        for (BLASLONG i = 0; i < M; ++i) {
            GEMM_I32 s0 = 0, s1 = 0;
            BLASLONG ai = i * K;
            BLASLONG bi = n_top * K;
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                s0 += (GEMM_I32)A[ai + k] * (GEMM_I32)B[bi + 0];
                s1 += (GEMM_I32)A[ai + k] * (GEMM_I32)B[bi + 1];
                bi += 2;
            }
            C[n_top * ldc + i] += alpha * s0;
            C[(n_top + 1) * ldc + i] += alpha * s1;
        }
        n_top += 2;
    }

    if (N & 1) {
        for (BLASLONG i = 0; i < M; ++i) {
            GEMM_I32 s0 = 0;
            BLASLONG ai = i * K;
            BLASLONG bi = n_top * K;
#pragma GCC unroll 1
            for (BLASLONG k = 0; k < K; ++k) {
                s0 += (GEMM_I32)A[ai + k] * (GEMM_I32)B[bi];
                bi += 1;
            }
            C[n_top * ldc + i] += alpha * s0;
        }
    }

    return 0;
}
