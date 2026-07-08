/*
 * Copyright (c) 2026 University of Salerno 
 *
 * Created by Babar Hussain.
 */

/*
RVV FP KERNEL
Variant: sgemm_kernel_8x4_zvl128b_lmul4_unroll2
Settings:
 LMUL=4 (m4)
 M=8
 N=4
 precision=float (FP32)
 unroll_factor=2 (K unroll x2)
 Kernel Name: sgemm_kernel_8x4_zvl128b_lmul4_unroll2
*/
#include <stddef.h>
#include <riscv_vector.h>

typedef long BLASLONG;
typedef float FLOAT;

#ifndef CNAME
#define CNAME sgemm_kernel_8x4_zvl128b
#endif

static inline void scalar_block(BLASLONG rows, BLASLONG cols, BLASLONG K,
                                FLOAT alpha, const FLOAT *Ablk, const FLOAT *Bblk,
                                FLOAT *Cblk, BLASLONG ldc)
{
    FLOAT acc[8] = {0.0f};
    BLASLONG ai = 0;
    BLASLONG bi = 0;

#pragma GCC unroll 2
    for (BLASLONG k = 0; k < K; ++k) {
        for (BLASLONG n = 0; n < cols; ++n) {
            const FLOAT b = Bblk[bi + n];
            for (BLASLONG m = 0; m < rows; ++m) {
                acc[n * rows + m] += Ablk[ai + m] * b;
            }
        }
        ai += rows;
        bi += cols;
    }

    for (BLASLONG n = 0; n < cols; ++n) {
        FLOAT *c_col = &Cblk[n * ldc];
        for (BLASLONG m = 0; m < rows; ++m) {
            c_col[m] += alpha * acc[n * rows + m];
        }
    }
}

static inline void vec_block_8xN(BLASLONG cols, BLASLONG K,
                                 FLOAT alpha, const FLOAT *Ablk, const FLOAT *Bblk,
                                 FLOAT *Cblk, BLASLONG ldc)
{
    const size_t gvl = __riscv_vsetvl_e32m4(4);
    vfloat32m4_t a0 = __riscv_vle32_v_f32m4(&Ablk[0], gvl);
    vfloat32m4_t a1 = __riscv_vle32_v_f32m4(&Ablk[gvl], gvl);
    vfloat32m4_t r00, r01, r02, r03;
    vfloat32m4_t r10, r11, r12, r13;

    if (cols > 0) {
        const FLOAT b = Bblk[0];
        r00 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
        r10 = __riscv_vfmul_vf_f32m4(a1, b, gvl);
    }
    if (cols > 1) {
        const FLOAT b = Bblk[1];
        r01 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
        r11 = __riscv_vfmul_vf_f32m4(a1, b, gvl);
    }
    if (cols > 2) {
        const FLOAT b = Bblk[2];
        r02 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
        r12 = __riscv_vfmul_vf_f32m4(a1, b, gvl);
    }
    if (cols > 3) {
        const FLOAT b = Bblk[3];
        r03 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
        r13 = __riscv_vfmul_vf_f32m4(a1, b, gvl);
    }

    BLASLONG ai = 8;
    BLASLONG bi = cols;

#pragma GCC unroll 1
    for (BLASLONG k = 1; k < K; ++k) {
        a0 = __riscv_vle32_v_f32m4(&Ablk[ai], gvl);
        a1 = __riscv_vle32_v_f32m4(&Ablk[ai + gvl], gvl);
        ai += 8;

        if (cols > 0) {
            const FLOAT b = Bblk[bi + 0];
            r00 = __riscv_vfmacc_vf_f32m4(r00, b, a0, gvl);
            r10 = __riscv_vfmacc_vf_f32m4(r10, b, a1, gvl);
        }
        if (cols > 1) {
            const FLOAT b = Bblk[bi + 1];
            r01 = __riscv_vfmacc_vf_f32m4(r01, b, a0, gvl);
            r11 = __riscv_vfmacc_vf_f32m4(r11, b, a1, gvl);
        }
        if (cols > 2) {
            const FLOAT b = Bblk[bi + 2];
            r02 = __riscv_vfmacc_vf_f32m4(r02, b, a0, gvl);
            r12 = __riscv_vfmacc_vf_f32m4(r12, b, a1, gvl);
        }
        if (cols > 3) {
            const FLOAT b = Bblk[bi + 3];
            r03 = __riscv_vfmacc_vf_f32m4(r03, b, a0, gvl);
            r13 = __riscv_vfmacc_vf_f32m4(r13, b, a1, gvl);
        }

        bi += cols;
    }

    if (cols > 0) {
        FLOAT *c_col = &Cblk[0 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        vfloat32m4_t c1 = __riscv_vle32_v_f32m4(&c_col[gvl], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r00, gvl);
        c1 = __riscv_vfmacc_vf_f32m4(c1, alpha, r10, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
        __riscv_vse32_v_f32m4(&c_col[gvl], c1, gvl);
    }
    if (cols > 1) {
        FLOAT *c_col = &Cblk[1 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        vfloat32m4_t c1 = __riscv_vle32_v_f32m4(&c_col[gvl], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r01, gvl);
        c1 = __riscv_vfmacc_vf_f32m4(c1, alpha, r11, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
        __riscv_vse32_v_f32m4(&c_col[gvl], c1, gvl);
    }
    if (cols > 2) {
        FLOAT *c_col = &Cblk[2 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        vfloat32m4_t c1 = __riscv_vle32_v_f32m4(&c_col[gvl], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r02, gvl);
        c1 = __riscv_vfmacc_vf_f32m4(c1, alpha, r12, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
        __riscv_vse32_v_f32m4(&c_col[gvl], c1, gvl);
    }
    if (cols > 3) {
        FLOAT *c_col = &Cblk[3 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        vfloat32m4_t c1 = __riscv_vle32_v_f32m4(&c_col[gvl], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r03, gvl);
        c1 = __riscv_vfmacc_vf_f32m4(c1, alpha, r13, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
        __riscv_vse32_v_f32m4(&c_col[gvl], c1, gvl);
    }
}

static inline void vec_block_4xN(BLASLONG cols, BLASLONG K,
                                 FLOAT alpha, const FLOAT *Ablk, const FLOAT *Bblk,
                                 FLOAT *Cblk, BLASLONG ldc)
{
    const size_t gvl = __riscv_vsetvl_e32m4(4);
    vfloat32m4_t a0 = __riscv_vle32_v_f32m4(&Ablk[0], gvl);
    vfloat32m4_t r0, r1, r2, r3;

    if (cols > 0) {
        const FLOAT b = Bblk[0];
        r0 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
    }
    if (cols > 1) {
        const FLOAT b = Bblk[1];
        r1 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
    }
    if (cols > 2) {
        const FLOAT b = Bblk[2];
        r2 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
    }
    if (cols > 3) {
        const FLOAT b = Bblk[3];
        r3 = __riscv_vfmul_vf_f32m4(a0, b, gvl);
    }

    BLASLONG ai = 4;
    BLASLONG bi = cols;

#pragma GCC unroll 1
    for (BLASLONG k = 1; k < K; ++k) {
        a0 = __riscv_vle32_v_f32m4(&Ablk[ai], gvl);
        ai += 4;

        if (cols > 0) {
            const FLOAT b = Bblk[bi + 0];
            r0 = __riscv_vfmacc_vf_f32m4(r0, b, a0, gvl);
        }
        if (cols > 1) {
            const FLOAT b = Bblk[bi + 1];
            r1 = __riscv_vfmacc_vf_f32m4(r1, b, a0, gvl);
        }
        if (cols > 2) {
            const FLOAT b = Bblk[bi + 2];
            r2 = __riscv_vfmacc_vf_f32m4(r2, b, a0, gvl);
        }
        if (cols > 3) {
            const FLOAT b = Bblk[bi + 3];
            r3 = __riscv_vfmacc_vf_f32m4(r3, b, a0, gvl);
        }

        bi += cols;
    }

    if (cols > 0) {
        FLOAT *c_col = &Cblk[0 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r0, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
    }
    if (cols > 1) {
        FLOAT *c_col = &Cblk[1 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r1, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
    }
    if (cols > 2) {
        FLOAT *c_col = &Cblk[2 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r2, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
    }
    if (cols > 3) {
        FLOAT *c_col = &Cblk[3 * ldc];
        vfloat32m4_t c0 = __riscv_vle32_v_f32m4(&c_col[0], gvl);
        c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, r3, gvl);
        __riscv_vse32_v_f32m4(&c_col[0], c0, gvl);
    }
}

int CNAME(BLASLONG M, BLASLONG N, BLASLONG K,
          FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT *C, BLASLONG ldc)
{
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;

    if (K <= 0) {
        return 0;
    }

    if (__riscv_vsetvl_e32m4(4) < 4) {
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
