/*
 * Copyright (c) 2026 University of Salerno
 *
 * Created by Babar Hussain.
 */

/*
RVV FP KERNEL
Variant: dgemm_kernel_8x4_zvl256b_lmul8_unroll1
Settings:
 LMUL=8 (m8)
 M=8
 N=4
 precision=double (FP64)
 unroll_factor=1 (No unroll)
 Kernel Name: dgemm_kernel_8x4_zvl256b_lmul8_unroll1
*/
/*
 * Kernel dataflow summary: dgemm_kernel_8x4_zvl256b_lmul8_unroll1
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
int dgemm_kernel_8x4_zvl256b_lmul8_unroll1(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT *C, BLASLONG ldc) {
    BLASLONG gvl = 0;
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;
    if (K <= 0) {
        return 0;
    }
    /* ================================
       MAIN PASS (N / 4)
       ================================ */
    for (BLASLONG j = 0; j < N / 4; j++) {
        m_top = 0;
        gvl = __riscv_vsetvl_e64m8(8);
        for (BLASLONG i = 0; i < M / 8; i++) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            double B1 = B[bi + 1];
            double B2 = B[bi + 2];
            double B3 = B[bi + 3];
            bi += 4;
            vfloat64m8_t A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
            ai += 8;
            vfloat64m8_t r0 = __riscv_vfmul_vf_f64m8(A0, B0, gvl);
            vfloat64m8_t r1 = __riscv_vfmul_vf_f64m8(A0, B1, gvl);
            vfloat64m8_t r2 = __riscv_vfmul_vf_f64m8(A0, B2, gvl);
            vfloat64m8_t r3 = __riscv_vfmul_vf_f64m8(A0, B3, gvl);
#pragma GCC unroll 1
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                B1 = B[bi + 1];
                B2 = B[bi + 2];
                B3 = B[bi + 3];
                bi += 4;
                A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
                ai += 8;
                r0 = __riscv_vfmacc_vf_f64m8(r0, B0, A0, gvl);
                r1 = __riscv_vfmacc_vf_f64m8(r1, B1, A0, gvl);
                r2 = __riscv_vfmacc_vf_f64m8(r2, B2, A0, gvl);
                r3 = __riscv_vfmacc_vf_f64m8(r3, B3, A0, gvl);
            }
            BLASLONG ci = n_top * ldc + m_top;
            vfloat64m8_t c0 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c1 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c2 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c3 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m8(c0, alpha, r0, gvl);
            c1 = __riscv_vfmacc_vf_f64m8(c1, alpha, r1, gvl);
            c2 = __riscv_vfmacc_vf_f64m8(c2, alpha, r2, gvl);
            c3 = __riscv_vfmacc_vf_f64m8(c3, alpha, r3, gvl);
            ci = n_top * ldc + m_top;
            __riscv_vse64_v_f64m8(&C[ci], c0, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c1, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c2, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c3, gvl);
            m_top += 8;
        }
        if (M & 4) {
            gvl = __riscv_vsetvl_e64m8(4);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            double B1 = B[bi + 1];
            double B2 = B[bi + 2];
            double B3 = B[bi + 3];
            bi += 4;
            vfloat64m8_t A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
            ai += 4;
            vfloat64m8_t r0 = __riscv_vfmul_vf_f64m8(A0, B0, gvl);
            vfloat64m8_t r1 = __riscv_vfmul_vf_f64m8(A0, B1, gvl);
            vfloat64m8_t r2 = __riscv_vfmul_vf_f64m8(A0, B2, gvl);
            vfloat64m8_t r3 = __riscv_vfmul_vf_f64m8(A0, B3, gvl);
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                B1 = B[bi + 1];
                B2 = B[bi + 2];
                B3 = B[bi + 3];
                bi += 4;
                A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
                ai += 4;
                r0 = __riscv_vfmacc_vf_f64m8(r0, B0, A0, gvl);
                r1 = __riscv_vfmacc_vf_f64m8(r1, B1, A0, gvl);
                r2 = __riscv_vfmacc_vf_f64m8(r2, B2, A0, gvl);
                r3 = __riscv_vfmacc_vf_f64m8(r3, B3, A0, gvl);
            }
            BLASLONG ci = n_top * ldc + m_top;
            vfloat64m8_t c0 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c1 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c2 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c3 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m8(c0, alpha, r0, gvl);
            c1 = __riscv_vfmacc_vf_f64m8(c1, alpha, r1, gvl);
            c2 = __riscv_vfmacc_vf_f64m8(c2, alpha, r2, gvl);
            c3 = __riscv_vfmacc_vf_f64m8(c3, alpha, r3, gvl);
            ci = n_top * ldc + m_top;
            __riscv_vse64_v_f64m8(&C[ci], c0, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c1, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c2, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c3, gvl);
            m_top += 4;
        }
        if (M & 2) {
            double r0 = 0;
            double r1 = 0;
            double r2 = 0;
            double r3 = 0;
            double r4 = 0;
            double r5 = 0;
            double r6 = 0;
            double r7 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            for (BLASLONG k = 0; k < K; k++) {
                r0 += A[ai + 0] * B[bi + 0];
                r1 += A[ai + 1] * B[bi + 0];
                r2 += A[ai + 0] * B[bi + 1];
                r3 += A[ai + 1] * B[bi + 1];
                r4 += A[ai + 0] * B[bi + 2];
                r5 += A[ai + 1] * B[bi + 2];
                r6 += A[ai + 0] * B[bi + 3];
                r7 += A[ai + 1] * B[bi + 3];
                ai += 2;
                bi += 4;
            }
            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * r0;
            C[ci + 0 * ldc + 1] += alpha * r1;
            C[ci + 1 * ldc + 0] += alpha * r2;
            C[ci + 1 * ldc + 1] += alpha * r3;
            C[ci + 2 * ldc + 0] += alpha * r4;
            C[ci + 2 * ldc + 1] += alpha * r5;
            C[ci + 3 * ldc + 0] += alpha * r6;
            C[ci + 3 * ldc + 1] += alpha * r7;
            m_top += 2;
        }
        if (M & 1) {
            double r0 = 0;
            double r1 = 0;
            double r2 = 0;
            double r3 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            for (BLASLONG k = 0; k < K; k++) {
                r0 += A[ai + 0] * B[bi + 0];
                r1 += A[ai + 0] * B[bi + 1];
                r2 += A[ai + 0] * B[bi + 2];
                r3 += A[ai + 0] * B[bi + 3];
                ai += 1;
                bi += 4;
            }
            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * r0;
            C[ci + 1 * ldc + 0] += alpha * r1;
            C[ci + 2 * ldc + 0] += alpha * r2;
            C[ci + 3 * ldc + 0] += alpha * r3;
            m_top += 1;
        }
        n_top += 4;
    }
    /* ================================
       N TAIL = 2
       ================================ */
    if (N & 2) {
        gvl = __riscv_vsetvl_e64m8(8);
        m_top = 0;
        for (BLASLONG i = 0; i < M / 8; i++) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi];
            double B1 = B[bi + 1];
            bi += 2;
            vfloat64m8_t A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
            ai += 8;
            vfloat64m8_t r0 = __riscv_vfmul_vf_f64m8(A0, B0, gvl);
            vfloat64m8_t r1 = __riscv_vfmul_vf_f64m8(A0, B1, gvl);
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi];
                B1 = B[bi + 1];
                bi += 2;
                A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
                ai += 8;
                r0 = __riscv_vfmacc_vf_f64m8(r0, B0, A0, gvl);
                r1 = __riscv_vfmacc_vf_f64m8(r1, B1, A0, gvl);
            }
            BLASLONG ci = n_top * ldc + m_top;
            vfloat64m8_t c0 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c1 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m8(c0, alpha, r0, gvl);
            c1 = __riscv_vfmacc_vf_f64m8(c1, alpha, r1, gvl);
            ci = n_top * ldc + m_top;
            __riscv_vse64_v_f64m8(&C[ci], c0, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c1, gvl);
            m_top += 8;
        }
        if (M & 4) {
            gvl = __riscv_vsetvl_e64m8(4);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            double B1 = B[bi + 1];
            bi += 2;
            vfloat64m8_t A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
            ai += 4;
            vfloat64m8_t r0 = __riscv_vfmul_vf_f64m8(A0, B0, gvl);
            vfloat64m8_t r1 = __riscv_vfmul_vf_f64m8(A0, B1, gvl);
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                B1 = B[bi + 1];
                bi += 2;
                A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
                ai += 4;
                r0 = __riscv_vfmacc_vf_f64m8(r0, B0, A0, gvl);
                r1 = __riscv_vfmacc_vf_f64m8(r1, B1, A0, gvl);
            }
            BLASLONG ci = n_top * ldc + m_top;
            vfloat64m8_t c0 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            ci += ldc;
            vfloat64m8_t c1 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m8(c0, alpha, r0, gvl);
            c1 = __riscv_vfmacc_vf_f64m8(c1, alpha, r1, gvl);
            ci = n_top * ldc + m_top;
            __riscv_vse64_v_f64m8(&C[ci], c0, gvl);
            ci += ldc;
            __riscv_vse64_v_f64m8(&C[ci], c1, gvl);
            m_top += 4;
        }
        if (M & 2) {
            double r0 = 0;
            double r1 = 0;
            double r2 = 0;
            double r3 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            for (BLASLONG k = 0; k < K; k++) {
                r0 += A[ai + 0] * B[bi + 0];
                r1 += A[ai + 1] * B[bi + 0];
                r2 += A[ai + 0] * B[bi + 1];
                r3 += A[ai + 1] * B[bi + 1];
                ai += 2;
                bi += 2;
            }
            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * r0;
            C[ci + 0 * ldc + 1] += alpha * r1;
            C[ci + 1 * ldc + 0] += alpha * r2;
            C[ci + 1 * ldc + 1] += alpha * r3;
            m_top += 2;
        }
        if (M & 1) {
            double r0 = 0;
            double r1 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            for (BLASLONG k = 0; k < K; k++) {
                r0 += A[ai + 0] * B[bi + 0];
                r1 += A[ai + 0] * B[bi + 1];
                ai += 1;
                bi += 2;
            }
            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * r0;
            C[ci + 1 * ldc + 0] += alpha * r1;
            m_top += 1;
        }
        n_top += 2;
    }
    /* ================================
       N TAIL = 1
       ================================ */
    if (N & 1) {
        gvl = __riscv_vsetvl_e64m8(8);
        m_top = 0;
        for (BLASLONG i = 0; i < M / 8; i++) {
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi];
            bi += 1;
            vfloat64m8_t A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
            ai += 8;
            vfloat64m8_t r0 = __riscv_vfmul_vf_f64m8(A0, B0, gvl);
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi];
                bi += 1;
                A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
                ai += 8;
                r0 = __riscv_vfmacc_vf_f64m8(r0, B0, A0, gvl);
            }
            BLASLONG ci = n_top * ldc + m_top;
            vfloat64m8_t c0 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m8(c0, alpha, r0, gvl);
            __riscv_vse64_v_f64m8(&C[ci], c0, gvl);
            m_top += 8;
        }
        if (M & 4) {
            gvl = __riscv_vsetvl_e64m8(4);
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            double B0 = B[bi + 0];
            bi += 1;
            vfloat64m8_t A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
            ai += 4;
            vfloat64m8_t r0 = __riscv_vfmul_vf_f64m8(A0, B0, gvl);
            for (BLASLONG k = 1; k < K; k++) {
                B0 = B[bi + 0];
                bi += 1;
                A0 = __riscv_vle64_v_f64m8(&A[ai], gvl);
                ai += 4;
                r0 = __riscv_vfmacc_vf_f64m8(r0, B0, A0, gvl);
            }
            BLASLONG ci = n_top * ldc + m_top;
            vfloat64m8_t c0 = __riscv_vle64_v_f64m8(&C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m8(c0, alpha, r0, gvl);
            __riscv_vse64_v_f64m8(&C[ci], c0, gvl);
            m_top += 4;
        }
        if (M & 2) {
            double r0 = 0;
            double r1 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            for (BLASLONG k = 0; k < K; k++) {
                r0 += A[ai + 0] * B[bi + 0];
                r1 += A[ai + 1] * B[bi + 0];
                ai += 2;
                bi += 1;
            }
            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * r0;
            C[ci + 0 * ldc + 1] += alpha * r1;
            m_top += 2;
        }
        if (M & 1) {
            double r0 = 0;
            BLASLONG ai = m_top * K;
            BLASLONG bi = n_top * K;
            for (BLASLONG k = 0; k < K; k++) {
                r0 += A[ai + 0] * B[bi + 0];
                ai += 1;
                bi += 1;
            }
            BLASLONG ci = n_top * ldc + m_top;
            C[ci + 0 * ldc + 0] += alpha * r0;
            m_top += 1;
        }
        n_top += 1;
    }
    return 0;
}
