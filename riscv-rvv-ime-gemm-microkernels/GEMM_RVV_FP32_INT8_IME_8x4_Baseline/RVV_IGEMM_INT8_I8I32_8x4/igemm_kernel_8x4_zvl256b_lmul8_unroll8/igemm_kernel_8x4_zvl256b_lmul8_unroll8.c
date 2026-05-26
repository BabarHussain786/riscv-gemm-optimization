/*
RVV INT8 GEMM Kernel
Tile: 8x4
Variant: igemm_kernel_8x4_zvl256b_lmul8_unroll8
LMUL label: 8
LMUL compute path: true_i8m8_scalar_i32_accumulation
Unroll: 8
Math contract: same blocking shape/load as FP32 8x4 (M/8 x N/4 main + tails)
*/

/* Legality note: this variant uses true i8m8 input vector loads/stores, then scalar int32 accumulation. A fully vector-widened i8m8 path would require illegal accumulator LMULs. */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

typedef long BLASLONG;
typedef int8_t IFLOAT;
typedef int32_t OFLOAT;

#ifndef CNAME
#define CNAME igemm_kernel_8x4_zvl256b
#endif

static inline OFLOAT add_scaled_wrap_i32(OFLOAT dst, OFLOAT alpha, int64_t acc)
{
    uint32_t a = (uint32_t)dst;
    uint32_t b = (uint32_t)((int32_t)(acc * (int64_t)alpha));
    return (OFLOAT)(a + b);
}

static inline void scalar_block(BLASLONG rows, BLASLONG cols, BLASLONG K,
                                OFLOAT alpha, const IFLOAT *Ablk, const IFLOAT *Bblk,
                                OFLOAT *Cblk, BLASLONG ldc)
{
    int64_t acc[32] = {0};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 8
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
        OFLOAT *c_col = &Cblk[n * ldc];
        for (BLASLONG m = 0; m < rows; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, acc[n * rows + m]);
        }
    }
}

static inline void vec_block_8xN(BLASLONG cols, BLASLONG K,
                                 OFLOAT alpha, const IFLOAT *Ablk, const IFLOAT *Bblk,
                                 OFLOAT *Cblk, BLASLONG ldc)
{
    const size_t gvl = __riscv_vsetvl_e8m8(8);
    IFLOAT a_lane[8];
    int64_t acc[32] = {0};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 8
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
        OFLOAT *c_col = &Cblk[n * ldc];
        for (BLASLONG m = 0; m < 8; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, acc[n * 8 + m]);
        }
    }
}

static inline void vec_block_4xN(BLASLONG cols, BLASLONG K,
                                 OFLOAT alpha, const IFLOAT *Ablk, const IFLOAT *Bblk,
                                 OFLOAT *Cblk, BLASLONG ldc)
{
    const size_t gvl = __riscv_vsetvl_e8m8(4);
    IFLOAT a_lane[4];
    int64_t acc[16] = {0};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 8
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
        OFLOAT *c_col = &Cblk[n * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, acc[n * 4 + m]);
        }
    }
}

int CNAME(BLASLONG M, BLASLONG N, BLASLONG K,
          OFLOAT alpha, IFLOAT *A, IFLOAT *B, OFLOAT *C, BLASLONG ldc)
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
            scalar_block(2, 4, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 2;
        }

        if (M & 1) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            scalar_block(1, 4, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
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
            scalar_block(2, 2, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 2;
        }

        if (M & 1) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            scalar_block(1, 2, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
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
            scalar_block(2, 1, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 2;
        }

        if (M & 1) {
            const BLASLONG ai = m_top * K;
            const BLASLONG bi = n_top * K;
            scalar_block(1, 1, K, alpha, &A[ai], &B[bi], &C[n_top * ldc + m_top], ldc);
            m_top += 1;
        }

        n_top += 1;
    }

    return 0;
}