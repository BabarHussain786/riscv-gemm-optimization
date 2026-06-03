/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 */

/*
RVV FP KERNEL
Variant: dgemm_kernel_8x4_zvl128b_lmul4_unroll2
Settings:
 LMUL=4 (m4)
 M=8
 N=4
 precision=double (FP64)
 unroll_factor=2 (K unroll x2)
 Kernel Name: dgemm_kernel_8x4_zvl128b_lmul4_unroll2
*/
/*
 * Kernel dataflow summary: dgemm_kernel_8x4_zvl128b_lmul4_unroll2
 * Input A: packed FP64 A micro-panel ordered by row block and K.
 * Input B: packed FP64 B micro-panel ordered by column block and K.
 * Accumulation: FP64 RVV vector accumulators across K; boundary cleanup handles leftover tiles.
 * Output C: column-major FP64 tile updated as C += alpha * (A * B).
 */
/*
 * Architecture overview
 * ---------------------
 * Purpose: standalone RVV FP64 DGEMM micro-kernel for 8x4 tiles.
 * Step 1: Read packed FP64 A/B micro-panels produced by the benchmark driver.
 * Step 2: Keep FP64 partial sums in RVV vector registers while traversing K.
 * Step 3: Apply the selected LMUL and unroll factor shown in the file name.
 * Step 4: Update full vectorized tiles first for the fast path.
 * Step 5: Finish leftover rows or columns with the boundary cleanup path.
 */
#include <riscv_vector.h>

typedef long BLASLONG;
typedef double FLOAT;

int dgemm_kernel_8x4_zvl128b_lmul4_unroll2(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT *C, BLASLONG ldc)
{
    BLASLONG gvl = 0;
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;

    if (K <= 0) {
        return 0;
    }

    /* Kernel section: main vectorized tile pass

     * Iterates over full tiles and keeps the hot K loop in RVV vector registers.

     */
    for (BLASLONG j = 0; j < N / 4; j += 1) {
        m_top = 0;
        BLASLONG gvl = __riscv_vsetvl_e64m4(8);

        for (BLASLONG i = 0; i < M / 8; i += 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            double B1 = B[bi + 1];
            double B2 = B[bi + 2];
            double B3 = B[bi + 3];
            bi += 4;

            vfloat64m4_t A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
            ai += 8;

            vfloat64m4_t result0 = __riscv_vfmul_vf_f64m4(A0, B0, gvl);
            vfloat64m4_t result1 = __riscv_vfmul_vf_f64m4(A0, B1, gvl);
            vfloat64m4_t result2 = __riscv_vfmul_vf_f64m4(A0, B2, gvl);
            vfloat64m4_t result3 = __riscv_vfmul_vf_f64m4(A0, B3, gvl);

            #pragma GCC unroll 2
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                B1 = B[bi + 1];
                B2 = B[bi + 2];
                B3 = B[bi + 3];
                bi += 4;

                A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
                ai += 8;

                result0 = __riscv_vfmacc_vf_f64m4(result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m4(result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f64m4(result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f64m4(result3, B3, A0, gvl);
            }

            BLASLONG ci = n_top * ldc + m_top;

            vfloat64m4_t c0 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c1 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c2 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c3 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m4(c0, alpha, result0, gvl);
            c1 = __riscv_vfmacc_vf_f64m4(c1, alpha, result1, gvl);
            c2 = __riscv_vfmacc_vf_f64m4(c2, alpha, result2, gvl);
            c3 = __riscv_vfmacc_vf_f64m4(c3, alpha, result3, gvl);

            ci = n_top * ldc + m_top;

            __riscv_vse64_v_f64m4(&C[ci], c0, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c1, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c2, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c3, gvl);
            m_top += 8;
        }

        /* Kernel section: boundary cleanup for remaining rows in the main tile pass. */

        if (M & 4) {
            gvl = __riscv_vsetvl_e64m4(4);

            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            double B1 = B[bi + 1];
            double B2 = B[bi + 2];
            double B3 = B[bi + 3];
            bi += 4;

            vfloat64m4_t A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
            ai += 4;

            vfloat64m4_t result0 = __riscv_vfmul_vf_f64m4(A0, B0, gvl);
            vfloat64m4_t result1 = __riscv_vfmul_vf_f64m4(A0, B1, gvl);
            vfloat64m4_t result2 = __riscv_vfmul_vf_f64m4(A0, B2, gvl);
            vfloat64m4_t result3 = __riscv_vfmul_vf_f64m4(A0, B3, gvl);

            #pragma GCC unroll 2
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                B1 = B[bi + 1];
                B2 = B[bi + 2];
                B3 = B[bi + 3];
                bi += 4;

                A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
                ai += 4;

                result0 = __riscv_vfmacc_vf_f64m4(result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m4(result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f64m4(result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f64m4(result3, B3, A0, gvl);
            }

            BLASLONG ci = n_top * ldc + m_top;

            vfloat64m4_t c0 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c1 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c2 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c3 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m4(c0, alpha, result0, gvl);
            c1 = __riscv_vfmacc_vf_f64m4(c1, alpha, result1, gvl);
            c2 = __riscv_vfmacc_vf_f64m4(c2, alpha, result2, gvl);
            c3 = __riscv_vfmacc_vf_f64m4(c3, alpha, result3, gvl);

            ci = n_top * ldc + m_top;

            __riscv_vse64_v_f64m4(&C[ci], c0, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c1, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c2, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c3, gvl);
            m_top += 4;
        }

        if (M & 2) {
            double result0 = 0;
            double result1 = 0;
            double result2 = 0;
            double result3 = 0;
            double result4 = 0;
            double result5 = 0;
            double result6 = 0;
            double result7 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;

            #pragma GCC unroll 2
            for (BLASLONG k = 0; k < K; k++) {
                result0 += A[ai + 0] * B[bi + 0];
                result1 += A[ai + 1] * B[bi + 0];
                result2 += A[ai + 0] * B[bi + 1];
                result3 += A[ai + 1] * B[bi + 1];
                result4 += A[ai + 0] * B[bi + 2];
                result5 += A[ai + 1] * B[bi + 2];
                result6 += A[ai + 0] * B[bi + 3];
                result7 += A[ai + 1] * B[bi + 3];
                ai += 2;
                bi += 4;
            }

            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * result0;
            C[ci + 0 * ldc + 1] += alpha * result1;
            C[ci + 1 * ldc + 0] += alpha * result2;
            C[ci + 1 * ldc + 1] += alpha * result3;
            C[ci + 2 * ldc + 0] += alpha * result4;
            C[ci + 2 * ldc + 1] += alpha * result5;
            C[ci + 3 * ldc + 0] += alpha * result6;
            C[ci + 3 * ldc + 1] += alpha * result7;
            m_top += 2;
        }

        if (M & 1) {
            double result0 = 0;
            double result1 = 0;
            double result2 = 0;
            double result3 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;

            #pragma GCC unroll 2
            for (BLASLONG k = 0; k < K; k++) {
                result0 += A[ai + 0] * B[bi + 0];
                result1 += A[ai + 0] * B[bi + 1];
                result2 += A[ai + 0] * B[bi + 2];
                result3 += A[ai + 0] * B[bi + 3];
                ai += 1;
                bi += 4;
            }

            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * result0;
            C[ci + 1 * ldc + 0] += alpha * result1;
            C[ci + 2 * ldc + 0] += alpha * result2;
            C[ci + 3 * ldc + 0] += alpha * result3;
            m_top += 1;
        }

        n_top += 4;
    }

    /* Kernel section: boundary cleanup for the remaining N=2 column group. */
    if (N & 2) {
        gvl = __riscv_vsetvl_e64m4(8);
        m_top = 0;

        for (BLASLONG i = 0; i < M / 8; i += 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            double B1 = B[bi + 1];
            bi += 2;

            vfloat64m4_t A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
            ai += 8;

            vfloat64m4_t result0 = __riscv_vfmul_vf_f64m4(A0, B0, gvl);
            vfloat64m4_t result1 = __riscv_vfmul_vf_f64m4(A0, B1, gvl);

            #pragma GCC unroll 2
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                B1 = B[bi + 1];
                bi += 2;

                A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
                ai += 8;

                result0 = __riscv_vfmacc_vf_f64m4(result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m4(result1, B1, A0, gvl);
            }

            BLASLONG ci = n_top * ldc + m_top;

            vfloat64m4_t c0 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c1 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m4(c0, alpha, result0, gvl);
            c1 = __riscv_vfmacc_vf_f64m4(c1, alpha, result1, gvl);

            ci = n_top * ldc + m_top;

            __riscv_vse64_v_f64m4(&C[ci], c0, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c1, gvl);
            m_top += 8;
        }

        if (M & 4) {
            gvl = __riscv_vsetvl_e64m4(4);

            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            double B1 = B[bi + 1];
            bi += 2;

            vfloat64m4_t A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
            ai += 4;

            vfloat64m4_t result0 = __riscv_vfmul_vf_f64m4(A0, B0, gvl);
            vfloat64m4_t result1 = __riscv_vfmul_vf_f64m4(A0, B1, gvl);

            #pragma GCC unroll 2
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                B1 = B[bi + 1];
                bi += 2;

                A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
                ai += 4;

                result0 = __riscv_vfmacc_vf_f64m4(result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m4(result1, B1, A0, gvl);
            }

            BLASLONG ci = n_top * ldc + m_top;

            vfloat64m4_t c0 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            ci += ldc - gvl * 0;
            vfloat64m4_t c1 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m4(c0, alpha, result0, gvl);
            c1 = __riscv_vfmacc_vf_f64m4(c1, alpha, result1, gvl);

            ci = n_top * ldc + m_top;

            __riscv_vse64_v_f64m4(&C[ci], c0, gvl);
            ci += ldc - gvl * 0;
            __riscv_vse64_v_f64m4(&C[ci], c1, gvl);
            m_top += 4;
        }

        if (M & 2) {
            double result0 = 0;
            double result1 = 0;
            double result2 = 0;
            double result3 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;

            #pragma GCC unroll 2
            for (BLASLONG k = 0; k < K; k++) {
                result0 += A[ai + 0] * B[bi + 0];
                result1 += A[ai + 1] * B[bi + 0];
                result2 += A[ai + 0] * B[bi + 1];
                result3 += A[ai + 1] * B[bi + 1];
                ai += 2;
                bi += 2;
            }

            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * result0;
            C[ci + 0 * ldc + 1] += alpha * result1;
            C[ci + 1 * ldc + 0] += alpha * result2;
            C[ci + 1 * ldc + 1] += alpha * result3;
            m_top += 2;
        }

        if (M & 1) {
            double result0 = 0;
            double result1 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;

            #pragma GCC unroll 2
            for (BLASLONG k = 0; k < K; k++) {
                result0 += A[ai + 0] * B[bi + 0];
                result1 += A[ai + 0] * B[bi + 1];
                ai += 1;
                bi += 2;
            }

            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * result0;
            C[ci + 1 * ldc + 0] += alpha * result1;
            m_top += 1;
        }

        n_top += 2;
    }

    /* Kernel section: boundary cleanup for the remaining N=1 column group. */
    if (N & 1) {
        gvl = __riscv_vsetvl_e64m4(8);
        m_top = 0;

        for (BLASLONG i = 0; i < M / 8; i += 1) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            bi += 1;

            vfloat64m4_t A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
            ai += 8;

            vfloat64m4_t result0 = __riscv_vfmul_vf_f64m4(A0, B0, gvl);

            #pragma GCC unroll 2
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                bi += 1;

                A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
                ai += 8;

                result0 = __riscv_vfmacc_vf_f64m4(result0, B0, A0, gvl);
            }

            BLASLONG ci = n_top * ldc + m_top;

            vfloat64m4_t c0 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m4(c0, alpha, result0, gvl);

            ci = n_top * ldc + m_top;

            __riscv_vse64_v_f64m4(&C[ci], c0, gvl);
            m_top += 8;
        }

        if (M & 4) {
            gvl = __riscv_vsetvl_e64m4(4);

            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            bi += 1;

            vfloat64m4_t A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
            ai += 4;

            vfloat64m4_t result0 = __riscv_vfmul_vf_f64m4(A0, B0, gvl);

            #pragma GCC unroll 2
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                bi += 1;

                A0 = __riscv_vle64_v_f64m4(&A[ai + 0 * gvl], gvl);
                ai += 4;

                result0 = __riscv_vfmacc_vf_f64m4(result0, B0, A0, gvl);
            }

            BLASLONG ci = n_top * ldc + m_top;

            vfloat64m4_t c0 = __riscv_vle64_v_f64m4(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m4(c0, alpha, result0, gvl);

            ci = n_top * ldc + m_top;

            __riscv_vse64_v_f64m4(&C[ci], c0, gvl);
            m_top += 4;
        }

        if (M & 2) {
            double result0 = 0;
            double result1 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;

            #pragma GCC unroll 2
            for (BLASLONG k = 0; k < K; k++) {
                result0 += A[ai + 0] * B[bi + 0];
                result1 += A[ai + 1] * B[bi + 0];
                ai += 2;
                bi += 1;
            }

            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * result0;
            C[ci + 0 * ldc + 1] += alpha * result1;
            m_top += 2;
        }

        if (M & 1) {
            double result0 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;

            #pragma GCC unroll 2
            for (BLASLONG k = 0; k < K; k++) {
                result0 += A[ai + 0] * B[bi + 0];
                ai += 1;
                bi += 1;
            }

            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * result0;
            m_top += 1;
        }

        n_top += 1;
    }

    return 0;
}
