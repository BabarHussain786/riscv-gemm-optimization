/*
RVV INT8 GEMM Kernel
Tile: 8x4
Variant: igemm_kernel_8x4_zvl128b_lmul1_unroll8
LMUL label: 1
LMUL compute path: native
Unroll: 8
Math contract: same blocking shape/load as FP32 8x4 (M/8 x N/4 main + tails)
*/

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

typedef long BLASLONG;
typedef int8_t IFLOAT;
typedef int32_t OFLOAT;

#ifndef CNAME
#define CNAME igemm_kernel_8x4_zvl128b
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
    int64_t acc[8] = {0};
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
    const size_t gvl = __riscv_vsetvl_e8m1(4);
    vint8m1_t a8_0;
    vint8m1_t a8_1;
    vint16m2_t a16_0;
    vint16m2_t a16_1;
    vint32m4_t r00 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r01 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r02 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r03 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r10 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r11 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r12 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r13 = __riscv_vmv_v_x_i32m4(0, gvl);

    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 8
    for (BLASLONG k = 0; k < K; ++k) {
        a8_0 = __riscv_vle8_v_i8m1(&Ablk[ai], gvl);
        a8_1 = __riscv_vle8_v_i8m1(&Ablk[ai + gvl], gvl);
        a16_0 = __riscv_vsext_vf2_i16m2(a8_0, gvl);
        a16_1 = __riscv_vsext_vf2_i16m2(a8_1, gvl);
        ai += 8;

        if (cols > 0) {
            const int16_t b = (int16_t)Bblk[bi + 0];
            r00 = __riscv_vwmacc_vx_i32m4(r00, b, a16_0, gvl);
            r10 = __riscv_vwmacc_vx_i32m4(r10, b, a16_1, gvl);
        }
        if (cols > 1) {
            const int16_t b = (int16_t)Bblk[bi + 1];
            r01 = __riscv_vwmacc_vx_i32m4(r01, b, a16_0, gvl);
            r11 = __riscv_vwmacc_vx_i32m4(r11, b, a16_1, gvl);
        }
        if (cols > 2) {
            const int16_t b = (int16_t)Bblk[bi + 2];
            r02 = __riscv_vwmacc_vx_i32m4(r02, b, a16_0, gvl);
            r12 = __riscv_vwmacc_vx_i32m4(r12, b, a16_1, gvl);
        }
        if (cols > 3) {
            const int16_t b = (int16_t)Bblk[bi + 3];
            r03 = __riscv_vwmacc_vx_i32m4(r03, b, a16_0, gvl);
            r13 = __riscv_vwmacc_vx_i32m4(r13, b, a16_1, gvl);
        }
        bi += cols;
    }

    OFLOAT tmp0[4], tmp1[4];
    if (cols > 0) {
        __riscv_vse32_v_i32m4(tmp0, r00, gvl);
        __riscv_vse32_v_i32m4(tmp1, r10, gvl);
        OFLOAT *c_col = &Cblk[0 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
        }
    }
    if (cols > 1) {
        __riscv_vse32_v_i32m4(tmp0, r01, gvl);
        __riscv_vse32_v_i32m4(tmp1, r11, gvl);
        OFLOAT *c_col = &Cblk[1 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
        }
    }
    if (cols > 2) {
        __riscv_vse32_v_i32m4(tmp0, r02, gvl);
        __riscv_vse32_v_i32m4(tmp1, r12, gvl);
        OFLOAT *c_col = &Cblk[2 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
        }
    }
    if (cols > 3) {
        __riscv_vse32_v_i32m4(tmp0, r03, gvl);
        __riscv_vse32_v_i32m4(tmp1, r13, gvl);
        OFLOAT *c_col = &Cblk[3 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m]     = add_scaled_wrap_i32(c_col[m],     alpha, tmp0[m]);
            c_col[4 + m] = add_scaled_wrap_i32(c_col[4 + m], alpha, tmp1[m]);
        }
    }
}

static inline void vec_block_4xN(BLASLONG cols, BLASLONG K,
                                 OFLOAT alpha, const IFLOAT *Ablk, const IFLOAT *Bblk,
                                 OFLOAT *Cblk, BLASLONG ldc)
{
    const size_t gvl = __riscv_vsetvl_e8m1(4);
    vint8m1_t a8;
    vint16m2_t a16;
    vint32m4_t r0 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r1 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r2 = __riscv_vmv_v_x_i32m4(0, gvl);
    vint32m4_t r3 = __riscv_vmv_v_x_i32m4(0, gvl);

    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 8
    for (BLASLONG k = 0; k < K; ++k) {
        a8 = __riscv_vle8_v_i8m1(&Ablk[ai], gvl);
        a16 = __riscv_vsext_vf2_i16m2(a8, gvl);
        ai += 4;

        if (cols > 0) {
            const int16_t b = (int16_t)Bblk[bi + 0];
            r0 = __riscv_vwmacc_vx_i32m4(r0, b, a16, gvl);
        }
        if (cols > 1) {
            const int16_t b = (int16_t)Bblk[bi + 1];
            r1 = __riscv_vwmacc_vx_i32m4(r1, b, a16, gvl);
        }
        if (cols > 2) {
            const int16_t b = (int16_t)Bblk[bi + 2];
            r2 = __riscv_vwmacc_vx_i32m4(r2, b, a16, gvl);
        }
        if (cols > 3) {
            const int16_t b = (int16_t)Bblk[bi + 3];
            r3 = __riscv_vwmacc_vx_i32m4(r3, b, a16, gvl);
        }
        bi += cols;
    }

    OFLOAT tmp[4];
    if (cols > 0) {
        __riscv_vse32_v_i32m4(tmp, r0, gvl);
        OFLOAT *c_col = &Cblk[0 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
        }
    }
    if (cols > 1) {
        __riscv_vse32_v_i32m4(tmp, r1, gvl);
        OFLOAT *c_col = &Cblk[1 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
        }
    }
    if (cols > 2) {
        __riscv_vse32_v_i32m4(tmp, r2, gvl);
        OFLOAT *c_col = &Cblk[2 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
        }
    }
    if (cols > 3) {
        __riscv_vse32_v_i32m4(tmp, r3, gvl);
        OFLOAT *c_col = &Cblk[3 * ldc];
        for (BLASLONG m = 0; m < 4; ++m) {
            c_col[m] = add_scaled_wrap_i32(c_col[m], alpha, tmp[m]);
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

    if (__riscv_vsetvl_e8m1(4) < 4) {
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
