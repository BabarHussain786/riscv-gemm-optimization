/*
 * Copyright (c) 2026 University of Salerno 
 *
 * Created by Babar Hussain.
 */

/*
RVV INT KERNEL
Variant: igemm_kernel_8x8_zvl256b_lmulmf8_unroll1_i8i32
Settings:
 LMUL=1/8 (mf8)
 M=8
 N=8
 precision=int8 x int8 -> int32
 unroll_factor=1 (No unroll)
 Kernel Name: igemm_kernel_8x8_zvl256b_lmulmf8_unroll1_i8i32
*/
#include <stdint.h>
#include <riscv_vector.h>

typedef long BLASLONG;

typedef int8_t IFLOAT;

typedef int32_t OFLOAT;

int igemm_kernel_8x8_zvl256b_lmulmf8_unroll1_i8i32(    BLASLONG M, BLASLONG N, BLASLONG K,    OFLOAT alpha, IFLOAT *A, IFLOAT *B, OFLOAT *C, BLASLONG ldc){

    BLASLONG m_top = 0;

    BLASLONG n_top = 0;

    if (K <= 0) {

        return 0;

    }

    if (__riscv_vsetvl_e8mf8(8) < 8) {

        return -1;

    }

    // -- MAIN PASS
    for (BLASLONG j = 0;

 j < N / 8;

 j += 1) {

        m_top = 0;

        for (BLASLONG i = 0;

 i < M / 8;

 i += 1) {

            size_t gvl = __riscv_vsetvl_e8mf8(8);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result1 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result2 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result3 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result4 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result5 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result6 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result7 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                IFLOAT B1 = B[bi + 1];

                IFLOAT B2 = B[bi + 2];

                IFLOAT B3 = B[bi + 3];

                IFLOAT B4 = B[bi + 4];

                IFLOAT B5 = B[bi + 5];

                IFLOAT B6 = B[bi + 6];

                IFLOAT B7 = B[bi + 7];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                result1 = __riscv_vwmacc_vx_i32mf2(result1, (int16_t)B1, A16, gvl);

                result2 = __riscv_vwmacc_vx_i32mf2(result2, (int16_t)B2, A16, gvl);

                result3 = __riscv_vwmacc_vx_i32mf2(result3, (int16_t)B3, A16, gvl);

                result4 = __riscv_vwmacc_vx_i32mf2(result4, (int16_t)B4, A16, gvl);

                result5 = __riscv_vwmacc_vx_i32mf2(result5, (int16_t)B5, A16, gvl);

                result6 = __riscv_vwmacc_vx_i32mf2(result6, (int16_t)B6, A16, gvl);

                result7 = __riscv_vwmacc_vx_i32mf2(result7, (int16_t)B7, A16, gvl);

                ai += 8;

                bi += 8;

            }

            OFLOAT tmp0[8] = {

0}

;

            OFLOAT tmp1[8] = {

0}

;

            OFLOAT tmp2[8] = {

0}

;

            OFLOAT tmp3[8] = {

0}

;

            OFLOAT tmp4[8] = {

0}

;

            OFLOAT tmp5[8] = {

0}

;

            OFLOAT tmp6[8] = {

0}

;

            OFLOAT tmp7[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            __riscv_vse32_v_i32mf2(tmp1, result1, gvl);

            __riscv_vse32_v_i32mf2(tmp2, result2, gvl);

            __riscv_vse32_v_i32mf2(tmp3, result3, gvl);

            __riscv_vse32_v_i32mf2(tmp4, result4, gvl);

            __riscv_vse32_v_i32mf2(tmp5, result5, gvl);

            __riscv_vse32_v_i32mf2(tmp6, result6, gvl);

            __riscv_vse32_v_i32mf2(tmp7, result7, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp1[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp2[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp3[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp4[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp5[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp6[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp7[r];

            }

            m_top += 8;

        }

        // -- tails for main pass        if (M & 4) {

            size_t gvl = __riscv_vsetvl_e8mf8(4);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result1 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result2 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result3 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result4 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result5 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result6 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result7 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                IFLOAT B1 = B[bi + 1];

                IFLOAT B2 = B[bi + 2];

                IFLOAT B3 = B[bi + 3];

                IFLOAT B4 = B[bi + 4];

                IFLOAT B5 = B[bi + 5];

                IFLOAT B6 = B[bi + 6];

                IFLOAT B7 = B[bi + 7];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                result1 = __riscv_vwmacc_vx_i32mf2(result1, (int16_t)B1, A16, gvl);

                result2 = __riscv_vwmacc_vx_i32mf2(result2, (int16_t)B2, A16, gvl);

                result3 = __riscv_vwmacc_vx_i32mf2(result3, (int16_t)B3, A16, gvl);

                result4 = __riscv_vwmacc_vx_i32mf2(result4, (int16_t)B4, A16, gvl);

                result5 = __riscv_vwmacc_vx_i32mf2(result5, (int16_t)B5, A16, gvl);

                result6 = __riscv_vwmacc_vx_i32mf2(result6, (int16_t)B6, A16, gvl);

                result7 = __riscv_vwmacc_vx_i32mf2(result7, (int16_t)B7, A16, gvl);

                ai += 4;

                bi += 8;

            }

            OFLOAT tmp0[8] = {

0}

;

            OFLOAT tmp1[8] = {

0}

;

            OFLOAT tmp2[8] = {

0}

;

            OFLOAT tmp3[8] = {

0}

;

            OFLOAT tmp4[8] = {

0}

;

            OFLOAT tmp5[8] = {

0}

;

            OFLOAT tmp6[8] = {

0}

;

            OFLOAT tmp7[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            __riscv_vse32_v_i32mf2(tmp1, result1, gvl);

            __riscv_vse32_v_i32mf2(tmp2, result2, gvl);

            __riscv_vse32_v_i32mf2(tmp3, result3, gvl);

            __riscv_vse32_v_i32mf2(tmp4, result4, gvl);

            __riscv_vse32_v_i32mf2(tmp5, result5, gvl);

            __riscv_vse32_v_i32mf2(tmp6, result6, gvl);

            __riscv_vse32_v_i32mf2(tmp7, result7, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp1[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp2[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp3[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp4[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp5[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp6[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp7[r];

            }

            m_top += 4;

        }

        if (M & 2) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[16] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                IFLOAT a1 = A[ai + 1];

                for (BLASLONG n = 0;

 n < 8;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;

                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;

                }

                ai += 2;

                bi += 8;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 8;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n * 2 + 0];

                C[ci + n * ldc + 1] += alpha * (OFLOAT)result[n * 2 + 1];

            }

            m_top += 2;

        }

        if (M & 1) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[8] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                for (BLASLONG n = 0;

 n < 8;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n] += (int32_t)a0 * (int32_t)bn;

                }

                ai += 1;

                bi += 8;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 8;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n];

            }

            m_top += 1;

        }

        n_top += 8;

    // -- tails for N=4
    if (N & 4) {

        m_top = 0;

        for (BLASLONG i = 0;

 i < M / 8;

 i += 1) {

            size_t gvl = __riscv_vsetvl_e8mf8(8);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result1 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result2 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result3 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                IFLOAT B1 = B[bi + 1];

                IFLOAT B2 = B[bi + 2];

                IFLOAT B3 = B[bi + 3];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                result1 = __riscv_vwmacc_vx_i32mf2(result1, (int16_t)B1, A16, gvl);

                result2 = __riscv_vwmacc_vx_i32mf2(result2, (int16_t)B2, A16, gvl);

                result3 = __riscv_vwmacc_vx_i32mf2(result3, (int16_t)B3, A16, gvl);

                ai += 8;

                bi += 4;

            }

            OFLOAT tmp0[8] = {

0}

;

            OFLOAT tmp1[8] = {

0}

;

            OFLOAT tmp2[8] = {

0}

;

            OFLOAT tmp3[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            __riscv_vse32_v_i32mf2(tmp1, result1, gvl);

            __riscv_vse32_v_i32mf2(tmp2, result2, gvl);

            __riscv_vse32_v_i32mf2(tmp3, result3, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp1[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp2[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp3[r];

            }

            m_top += 8;

        }

        if (M & 4) {

            size_t gvl = __riscv_vsetvl_e8mf8(4);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result1 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result2 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result3 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                IFLOAT B1 = B[bi + 1];

                IFLOAT B2 = B[bi + 2];

                IFLOAT B3 = B[bi + 3];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                result1 = __riscv_vwmacc_vx_i32mf2(result1, (int16_t)B1, A16, gvl);

                result2 = __riscv_vwmacc_vx_i32mf2(result2, (int16_t)B2, A16, gvl);

                result3 = __riscv_vwmacc_vx_i32mf2(result3, (int16_t)B3, A16, gvl);

                ai += 4;

                bi += 4;

            }

            OFLOAT tmp0[8] = {

0}

;

            OFLOAT tmp1[8] = {

0}

;

            OFLOAT tmp2[8] = {

0}

;

            OFLOAT tmp3[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            __riscv_vse32_v_i32mf2(tmp1, result1, gvl);

            __riscv_vse32_v_i32mf2(tmp2, result2, gvl);

            __riscv_vse32_v_i32mf2(tmp3, result3, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp1[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp2[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp3[r];

            }

            m_top += 4;

        }

        if (M & 2) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[8] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                IFLOAT a1 = A[ai + 1];

                for (BLASLONG n = 0;

 n < 4;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;

                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;

                }

                ai += 2;

                bi += 4;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 4;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n * 2 + 0];

                C[ci + n * ldc + 1] += alpha * (OFLOAT)result[n * 2 + 1];

            }

            m_top += 2;

        }

        if (M & 1) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[4] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                for (BLASLONG n = 0;

 n < 4;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n] += (int32_t)a0 * (int32_t)bn;

                }

                ai += 1;

                bi += 4;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 4;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n];

            }

            m_top += 1;

        }

        n_top += 4;

    }

    // -- tails for N=2
    if (N & 2) {

        m_top = 0;

        for (BLASLONG i = 0;

 i < M / 8;

 i += 1) {

            size_t gvl = __riscv_vsetvl_e8mf8(8);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result1 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                IFLOAT B1 = B[bi + 1];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                result1 = __riscv_vwmacc_vx_i32mf2(result1, (int16_t)B1, A16, gvl);

                ai += 8;

                bi += 2;

            }

            OFLOAT tmp0[8] = {

0}

;

            OFLOAT tmp1[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            __riscv_vse32_v_i32mf2(tmp1, result1, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp1[r];

            }

            m_top += 8;

        }

        if (M & 4) {

            size_t gvl = __riscv_vsetvl_e8mf8(4);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            vint32mf2_t result1 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                IFLOAT B1 = B[bi + 1];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                result1 = __riscv_vwmacc_vx_i32mf2(result1, (int16_t)B1, A16, gvl);

                ai += 4;

                bi += 2;

            }

            OFLOAT tmp0[8] = {

0}

;

            OFLOAT tmp1[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            __riscv_vse32_v_i32mf2(tmp1, result1, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            ci += ldc;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp1[r];

            }

            m_top += 4;

        }

        if (M & 2) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[4] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                IFLOAT a1 = A[ai + 1];

                for (BLASLONG n = 0;

 n < 2;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;

                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;

                }

                ai += 2;

                bi += 2;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 2;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n * 2 + 0];

                C[ci + n * ldc + 1] += alpha * (OFLOAT)result[n * 2 + 1];

            }

            m_top += 2;

        }

        if (M & 1) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[2] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                for (BLASLONG n = 0;

 n < 2;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n] += (int32_t)a0 * (int32_t)bn;

                }

                ai += 1;

                bi += 2;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 2;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n];

            }

            m_top += 1;

        }

        n_top += 2;

    }

    // -- tails for N=1
    if (N & 1) {

        m_top = 0;

        for (BLASLONG i = 0;

 i < M / 8;

 i += 1) {

            size_t gvl = __riscv_vsetvl_e8mf8(8);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                ai += 8;

                bi += 1;

            }

            OFLOAT tmp0[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 8;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            m_top += 8;

        }

        if (M & 4) {

            size_t gvl = __riscv_vsetvl_e8mf8(4);

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            vint32mf2_t result0 = __riscv_vmv_v_x_i32mf2(0, gvl);

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT B0 = B[bi + 0];

                vint8mf8_t A0 = __riscv_vle8_v_i8mf8(&A[ai], gvl);

                vint16mf4_t A16 = __riscv_vsext_vf2_i16mf4(A0, gvl);

                result0 = __riscv_vwmacc_vx_i32mf2(result0, (int16_t)B0, A16, gvl);

                ai += 4;

                bi += 1;

            }

            OFLOAT tmp0[8] = {

0}

;

            __riscv_vse32_v_i32mf2(tmp0, result0, gvl);

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG r = 0;

 r < 4;

 ++r) {

                C[ci + r] += alpha * tmp0[r];

            }

            m_top += 4;

        }

        if (M & 2) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[2] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                IFLOAT a1 = A[ai + 1];

                for (BLASLONG n = 0;

 n < 1;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n * 2 + 0] += (int32_t)a0 * (int32_t)bn;

                    result[n * 2 + 1] += (int32_t)a1 * (int32_t)bn;

                }

                ai += 2;

                bi += 1;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 1;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n * 2 + 0];

                C[ci + n * ldc + 1] += alpha * (OFLOAT)result[n * 2 + 1];

            }

            m_top += 2;

        }

        if (M & 1) {

            BLASLONG ai = m_top * K;

            BLASLONG bi = n_top * K;

            int64_t result[1] = {

0}

;

            
#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                IFLOAT a0 = A[ai + 0];

                for (BLASLONG n = 0;

 n < 1;

 ++n) {

                    IFLOAT bn = B[bi + n];

                    result[n] += (int32_t)a0 * (int32_t)bn;

                }

                ai += 1;

                bi += 1;

            }

            BLASLONG ci = n_top * ldc + m_top;

            for (BLASLONG n = 0;

 n < 1;

 ++n) {

                C[ci + n * ldc + 0] += alpha * (OFLOAT)result[n];

            }

            m_top += 1;

        }

        n_top += 1;

    }

    return 0;

}



