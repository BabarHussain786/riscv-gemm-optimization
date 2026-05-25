/*
 * Copyright (c) 2026 University of Salerno 
 *
 * Created by Babar Hussain.
 */

/*
RVV FP KERNEL
Variant: dgemm_kernel_8x8_zvl128b_lmul2_unroll8
Settings:
 LMUL=2 (m2)
 M=8
 N=8
 precision=double (FP64)
 unroll_factor=8 (K unroll x8)
 Kernel Name: dgemm_kernel_8x8_zvl128b_lmul2_unroll8
*/
#include <riscv_vector.h>

typedef long BLASLONG;
typedef double FLOAT;

int dgemm_kernel_8x8_zvl128b_lmul2_unroll8(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
{
    BLASLONG gvl = 0;
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;

    // -- MAIN PASS

    for (BLASLONG j=0; j<N/8; j+=1) {
        m_top = 0;
        BLASLONG gvl = __riscv_vsetvl_e64m2(8);

        for (BLASLONG i=0; i<M/8; i+=1) {
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            double B1 = B[bi+1];
            double B2 = B[bi+2];
            double B3 = B[bi+3];
            double B4 = B[bi+4];
            double B5 = B[bi+5];
            double B6 = B[bi+6];
            double B7 = B[bi+7];
            bi += 8;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 8;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);
            vfloat64m2_t result1 = __riscv_vfmul_vf_f64m2( A0, B1, gvl);
            vfloat64m2_t result2 = __riscv_vfmul_vf_f64m2( A0, B2, gvl);
            vfloat64m2_t result3 = __riscv_vfmul_vf_f64m2( A0, B3, gvl);
            vfloat64m2_t result4 = __riscv_vfmul_vf_f64m2( A0, B4, gvl);
            vfloat64m2_t result5 = __riscv_vfmul_vf_f64m2( A0, B5, gvl);
            vfloat64m2_t result6 = __riscv_vfmul_vf_f64m2( A0, B6, gvl);
            vfloat64m2_t result7 = __riscv_vfmul_vf_f64m2( A0, B7, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                B4 = B[bi+4];
                B5 = B[bi+5];
                B6 = B[bi+6];
                B7 = B[bi+7];
                bi += 8;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m2( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f64m2( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f64m2( result3, B3, A0, gvl);
                result4 = __riscv_vfmacc_vf_f64m2( result4, B4, A0, gvl);
                result5 = __riscv_vfmacc_vf_f64m2( result5, B5, A0, gvl);
                result6 = __riscv_vfmacc_vf_f64m2( result6, B6, A0, gvl);
                result7 = __riscv_vfmacc_vf_f64m2( result7, B7, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c1 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c2 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c3 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c4 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c5 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c6 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c7 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f64m2( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f64m2( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f64m2( c3, alpha, result3, gvl );
            c4 = __riscv_vfmacc_vf_f64m2( c4, alpha, result4, gvl );
            c5 = __riscv_vfmacc_vf_f64m2( c5, alpha, result5, gvl );
            c6 = __riscv_vfmacc_vf_f64m2( c6, alpha, result6, gvl );
            c7 = __riscv_vfmacc_vf_f64m2( c7, alpha, result7, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c1, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c2, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c3, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c4, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c5, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c6, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c7, gvl);
            m_top += 8;
        }

        // -- tails for main pass

        if( M & 4 ) {
            gvl = __riscv_vsetvl_e64m2(4);

            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            double B1 = B[bi+1];
            double B2 = B[bi+2];
            double B3 = B[bi+3];
            double B4 = B[bi+4];
            double B5 = B[bi+5];
            double B6 = B[bi+6];
            double B7 = B[bi+7];
            bi += 8;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 4;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);
            vfloat64m2_t result1 = __riscv_vfmul_vf_f64m2( A0, B1, gvl);
            vfloat64m2_t result2 = __riscv_vfmul_vf_f64m2( A0, B2, gvl);
            vfloat64m2_t result3 = __riscv_vfmul_vf_f64m2( A0, B3, gvl);
            vfloat64m2_t result4 = __riscv_vfmul_vf_f64m2( A0, B4, gvl);
            vfloat64m2_t result5 = __riscv_vfmul_vf_f64m2( A0, B5, gvl);
            vfloat64m2_t result6 = __riscv_vfmul_vf_f64m2( A0, B6, gvl);
            vfloat64m2_t result7 = __riscv_vfmul_vf_f64m2( A0, B7, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                B4 = B[bi+4];
                B5 = B[bi+5];
                B6 = B[bi+6];
                B7 = B[bi+7];
                bi += 8;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m2( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f64m2( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f64m2( result3, B3, A0, gvl);
                result4 = __riscv_vfmacc_vf_f64m2( result4, B4, A0, gvl);
                result5 = __riscv_vfmacc_vf_f64m2( result5, B5, A0, gvl);
                result6 = __riscv_vfmacc_vf_f64m2( result6, B6, A0, gvl);
                result7 = __riscv_vfmacc_vf_f64m2( result7, B7, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c1 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c2 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c3 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c4 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c5 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c6 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c7 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f64m2( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f64m2( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f64m2( c3, alpha, result3, gvl );
            c4 = __riscv_vfmacc_vf_f64m2( c4, alpha, result4, gvl );
            c5 = __riscv_vfmacc_vf_f64m2( c5, alpha, result5, gvl );
            c6 = __riscv_vfmacc_vf_f64m2( c6, alpha, result6, gvl );
            c7 = __riscv_vfmacc_vf_f64m2( c7, alpha, result7, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c1, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c2, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c3, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c4, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c5, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c6, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c7, gvl);
            m_top += 4;
        }

        if( M & 2 ) {
            double result0 = 0;
            double result1 = 0;
            double result2 = 0;
            double result3 = 0;
            double result4 = 0;
            double result5 = 0;
            double result6 = 0;
            double result7 = 0;
            double result8 = 0;
            double result9 = 0;
            double result10 = 0;
            double result11 = 0;
            double result12 = 0;
            double result13 = 0;
            double result14 = 0;
            double result15 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+1]*B[bi+0];
                result2+=A[ai+0]*B[bi+1];
                result3+=A[ai+1]*B[bi+1];
                result4+=A[ai+0]*B[bi+2];
                result5+=A[ai+1]*B[bi+2];
                result6+=A[ai+0]*B[bi+3];
                result7+=A[ai+1]*B[bi+3];
                result8+=A[ai+0]*B[bi+4];
                result9+=A[ai+1]*B[bi+4];
                result10+=A[ai+0]*B[bi+5];
                result11+=A[ai+1]*B[bi+5];
                result12+=A[ai+0]*B[bi+6];
                result13+=A[ai+1]*B[bi+6];
                result14+=A[ai+0]*B[bi+7];
                result15+=A[ai+1]*B[bi+7];
                ai+=2;
                bi+=8;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+0*ldc+1] += alpha * result1;
            C[ci+1*ldc+0] += alpha * result2;
            C[ci+1*ldc+1] += alpha * result3;
            C[ci+2*ldc+0] += alpha * result4;
            C[ci+2*ldc+1] += alpha * result5;
            C[ci+3*ldc+0] += alpha * result6;
            C[ci+3*ldc+1] += alpha * result7;
            C[ci+4*ldc+0] += alpha * result8;
            C[ci+4*ldc+1] += alpha * result9;
            C[ci+5*ldc+0] += alpha * result10;
            C[ci+5*ldc+1] += alpha * result11;
            C[ci+6*ldc+0] += alpha * result12;
            C[ci+6*ldc+1] += alpha * result13;
            C[ci+7*ldc+0] += alpha * result14;
            C[ci+7*ldc+1] += alpha * result15;
            m_top+=2;
        }

        if( M & 1 ) {
            double result0 = 0;
            double result1 = 0;
            double result2 = 0;
            double result3 = 0;
            double result4 = 0;
            double result5 = 0;
            double result6 = 0;
            double result7 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+0]*B[bi+1];
                result2+=A[ai+0]*B[bi+2];
                result3+=A[ai+0]*B[bi+3];
                result4+=A[ai+0]*B[bi+4];
                result5+=A[ai+0]*B[bi+5];
                result6+=A[ai+0]*B[bi+6];
                result7+=A[ai+0]*B[bi+7];
                ai+=1;
                bi+=8;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+1*ldc+0] += alpha * result1;
            C[ci+2*ldc+0] += alpha * result2;
            C[ci+3*ldc+0] += alpha * result3;
            C[ci+4*ldc+0] += alpha * result4;
            C[ci+5*ldc+0] += alpha * result5;
            C[ci+6*ldc+0] += alpha * result6;
            C[ci+7*ldc+0] += alpha * result7;
            m_top+=1;
        }

        n_top += 8;
    }

    // -- tails for N=4

    if( N & 4 ) {
        gvl = __riscv_vsetvl_e64m2(8);
        m_top = 0;

        for (BLASLONG i=0; i<M/8; i+=1) {
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            double B1 = B[bi+1];
            double B2 = B[bi+2];
            double B3 = B[bi+3];
            bi += 4;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 8;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);
            vfloat64m2_t result1 = __riscv_vfmul_vf_f64m2( A0, B1, gvl);
            vfloat64m2_t result2 = __riscv_vfmul_vf_f64m2( A0, B2, gvl);
            vfloat64m2_t result3 = __riscv_vfmul_vf_f64m2( A0, B3, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m2( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f64m2( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f64m2( result3, B3, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c1 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c2 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c3 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f64m2( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f64m2( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f64m2( c3, alpha, result3, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c1, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c2, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c3, gvl);
            m_top += 8;
        }

        if( M & 4 ) {
            gvl = __riscv_vsetvl_e64m2(4);

            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            double B1 = B[bi+1];
            double B2 = B[bi+2];
            double B3 = B[bi+3];
            bi += 4;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 4;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);
            vfloat64m2_t result1 = __riscv_vfmul_vf_f64m2( A0, B1, gvl);
            vfloat64m2_t result2 = __riscv_vfmul_vf_f64m2( A0, B2, gvl);
            vfloat64m2_t result3 = __riscv_vfmul_vf_f64m2( A0, B3, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m2( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f64m2( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f64m2( result3, B3, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c1 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c2 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c3 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f64m2( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f64m2( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f64m2( c3, alpha, result3, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c1, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c2, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c3, gvl);
            m_top += 4;
        }

        if( M & 2 ) {
            double result0 = 0;
            double result1 = 0;
            double result2 = 0;
            double result3 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+1]*B[bi+0];
                result2+=A[ai+0]*B[bi+1];
                result3+=A[ai+1]*B[bi+1];
                ai+=2;
                bi+=4;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+0*ldc+1] += alpha * result1;
            C[ci+1*ldc+0] += alpha * result2;
            C[ci+1*ldc+1] += alpha * result3;
            m_top+=2;
        }

        if( M & 1 ) {
            double result0 = 0;
            double result1 = 0;
            double result2 = 0;
            double result3 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+0]*B[bi+1];
                result2+=A[ai+0]*B[bi+2];
                result3+=A[ai+0]*B[bi+3];
                ai+=1;
                bi+=4;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+1*ldc+0] += alpha * result1;
            C[ci+2*ldc+0] += alpha * result2;
            C[ci+3*ldc+0] += alpha * result3;
            m_top+=1;
        }

        n_top += 4;
    }

    // -- tails for N=2

    if( N & 2 ) {
        gvl = __riscv_vsetvl_e64m2(8);
        m_top = 0;

        for (BLASLONG i=0; i<M/8; i+=1) {
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            double B1 = B[bi+1];
            bi += 2;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 8;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);
            vfloat64m2_t result1 = __riscv_vfmul_vf_f64m2( A0, B1, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m2( result1, B1, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c1 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f64m2( c1, alpha, result1, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c1, gvl);
            m_top += 8;
        }

        if( M & 4 ) {
            gvl = __riscv_vsetvl_e64m2(4);

            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            double B1 = B[bi+1];
            bi += 2;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 4;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);
            vfloat64m2_t result1 = __riscv_vfmul_vf_f64m2( A0, B1, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f64m2( result1, B1, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl); ci += ldc;
            vfloat64m2_t c1 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f64m2( c1, alpha, result1, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl); ci += ldc;
            __riscv_vse64_v_f64m2( &C[ci], c1, gvl);
            m_top += 4;
        }

        if( M & 2 ) {
            double result0 = 0;
            double result1 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+1]*B[bi+0];
                ai+=2;
                bi+=2;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+0*ldc+1] += alpha * result1;
            m_top+=2;
        }

        if( M & 1 ) {
            double result0 = 0;
            double result1 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+0]*B[bi+1];
                ai+=1;
                bi+=2;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+1*ldc+0] += alpha * result1;
            m_top+=1;
        }

        n_top += 2;
    }

    // -- tails for N=1

    if( N & 1 ) {
        gvl = __riscv_vsetvl_e64m2(8);
        m_top = 0;

        for (BLASLONG i=0; i<M/8; i+=1) {
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            bi += 1;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 8;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl);
            m_top += 8;
        }

        if( M & 4 ) {
            gvl = __riscv_vsetvl_e64m2(4);

            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;
            double B0 = B[bi+0];
            bi += 1;

            vfloat64m2_t A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
            ai += 4;

            vfloat64m2_t result0 = __riscv_vfmul_vf_f64m2( A0, B0, gvl);

            #pragma GCC unroll 8
            for(BLASLONG k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle64_v_f64m2( &A[ai], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f64m2( result0, B0, A0, gvl);
            }

            BLASLONG ci=n_top*ldc+m_top;

            vfloat64m2_t c0 = __riscv_vle64_v_f64m2( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f64m2( c0, alpha, result0, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse64_v_f64m2( &C[ci], c0, gvl);
            m_top += 4;
        }

        if( M & 2 ) {
            double result0 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                ai+=2;
                bi+=1;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            m_top+=2;
        }

        if( M & 1 ) {
            double result0 = 0;
            BLASLONG ai=m_top*K;
            BLASLONG bi=n_top*K;

            #pragma GCC unroll 8
            for(BLASLONG k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                ai+=1;
                bi+=1;
            }

            BLASLONG ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            m_top+=1;
        }

        n_top += 1;
    }

    return 0;
}
