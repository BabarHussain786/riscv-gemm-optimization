/*
 * Copyright (c) 2026 University of Salerno 
 *
 * Created by Babar Hussain.
 */

/*
RVV INT KERNEL
Variant: igemm_kernel_8x8_zvl256b_lmul4_unroll1_i8i32
Settings:
 LMUL=4 (m4)
 M=8
 N=8
 precision=int8 x int8 -> int32
 unroll_factor=1 (No unroll)
 Kernel Name: igemm_kernel_8x8_zvl256b_lmul4_unroll1_i8i32
*/
/* Legality note: this variant uses true i8m4 input vector loads/stores, then scalar int32 accumulation. A fully vector-widened i8m4 path would require illegal accumulator LMULs. */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

typedef long BLASLONG;

typedef int8_t IFLOAT;

typedef int32_t OFLOAT;

static inline OFLOAT add_scaled_wrap_i32(OFLOAT dst, int64_t acc, OFLOAT alpha){

    uint32_t sum = (uint32_t)dst + (uint32_t)((int64_t)alpha * acc);

    return (OFLOAT)sum;

}

static void scalar_block(BLASLONG rows, BLASLONG cols, BLASLONG K,                         OFLOAT alpha, const IFLOAT *A, const IFLOAT *B,                         OFLOAT *C, BLASLONG ldc){

    for (BLASLONG c = 0;

 c < cols;

 ++c) {

        for (BLASLONG r = 0;

 r < rows;

 ++r) {

            int64_t acc = 0;

#pragma GCC unroll 1
            for (BLASLONG k = 0;

 k < K;

 ++k) {

                acc += (int32_t)A[k * rows + r] * (int32_t)B[k * cols + c];

            }

            C[c * ldc + r] = add_scaled_wrap_i32(C[c * ldc + r], acc, alpha);

        }

    }

}

static void vector_block_rows8(BLASLONG cols, BLASLONG K,
                               OFLOAT alpha, const IFLOAT *A, const IFLOAT *B,
                               OFLOAT *C, BLASLONG ldc)
{
    int64_t acc[8][8] = {0};
    size_t gvl = __riscv_vsetvl_e8m4(8);

#pragma GCC unroll 1
    for (BLASLONG k = 0; k < K; ++k) {
        int8_t a_lane[8] = {0};
        vint8m4_t A0 = __riscv_vle8_v_i8m4(&A[k * 8], gvl);
        __riscv_vse8_v_i8m4(a_lane, A0, gvl);

        for (BLASLONG c = 0; c < cols; ++c) {
            int32_t b = (int32_t)B[k * cols + c];
            for (BLASLONG r = 0; r < 8; ++r) {
                acc[c][r] += (int32_t)a_lane[r] * b;
            }
        }
    }

    for (BLASLONG c = 0; c < cols; ++c) {
        OFLOAT tmp0[8] = {0};
        for (BLASLONG r = 0; r < 8; ++r) {
            tmp0[r] = (OFLOAT)acc[c][r];
        }
        BLASLONG ci = c * ldc;
        for (BLASLONG r = 0; r < 8; ++r) {
            C[ci + r] = add_scaled_wrap_i32(C[ci + r], (int64_t)tmp0[r], alpha);
        }
    }
}

int igemm_kernel_8x8_zvl256b_lmul4_unroll1_i8i32(    BLASLONG M, BLASLONG N, BLASLONG K,    OFLOAT alpha, IFLOAT *A, IFLOAT *B, OFLOAT *C, BLASLONG ldc){

    BLASLONG m_top = 0;

    BLASLONG n_top = 0;

    if (K <= 0) {

        return 0;

    }

    if (__riscv_vsetvl_e8m4(8) < 8) {

        return -1;

    }

    // -- MAIN PASS
    for (BLASLONG j = 0;

 j < N / 8;

 ++j) {

        m_top = 0;

        for (BLASLONG i = 0;

 i < M / 8;

 ++i) {

            vector_block_rows8(8, K, alpha,                               &A[m_top * K],                               &B[n_top * K],                               &C[n_top * ldc + m_top],                               ldc);

            m_top += 8;

        }

        // -- tails for main pass        if (M & 4) {

            scalar_block(4, 8, K, alpha,                         &A[m_top * K],                         &B[n_top * K],                         &C[n_top * ldc + m_top],                         ldc);

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

 ++i) {

            vector_block_rows8(4, K, alpha,                               &A[m_top * K],                               &B[n_top * K],                               &C[n_top * ldc + m_top],                               ldc);

            m_top += 8;

        }

        if (M & 4) {

            scalar_block(4, 4, K, alpha,                         &A[m_top * K],                         &B[n_top * K],                         &C[n_top * ldc + m_top],                         ldc);

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

 ++i) {

            vector_block_rows8(2, K, alpha,                               &A[m_top * K],                               &B[n_top * K],                               &C[n_top * ldc + m_top],                               ldc);

            m_top += 8;

        }

        if (M & 4) {

            scalar_block(4, 2, K, alpha,                         &A[m_top * K],                         &B[n_top * K],                         &C[n_top * ldc + m_top],                         ldc);

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

 ++i) {

            vector_block_rows8(1, K, alpha,                               &A[m_top * K],                               &B[n_top * K],                               &C[n_top * ldc + m_top],                               ldc);

            m_top += 8;

        }

        if (M & 4) {

            scalar_block(4, 1, K, alpha,                         &A[m_top * K],                         &B[n_top * K],                         &C[n_top * ldc + m_top],                         ldc);

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





