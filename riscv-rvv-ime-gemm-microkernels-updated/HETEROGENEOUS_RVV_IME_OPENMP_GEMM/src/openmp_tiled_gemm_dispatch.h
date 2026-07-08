#ifndef OPENMP_TILED_GEMM_DISPATCH_H
#define OPENMP_TILED_GEMM_DISPATCH_H

/*
 * Human-readable purpose
 * ----------------------
 * This file is the bridge from OpenMP tiles to your real micro-kernels.
 * OpenMP says: "compute C columns n0..n0+N-1".
 * This file converts that request into the exact pointer call expected by
 * FP32, FP64, INT8 RVV, or IME kernels.
 *
 * Simple example:
 *   If the full C matrix has 1024 columns and tile_N=64,
 *   tile 3 starts at column n0 = 3*64 = 192.
 *   The code passes B + n0*K and C + n0*M to the RVV kernel.
 *   For IME, B is first copied into B_tile because IME expects packed B data.
 */

#include "openmp_tiled_gemm_config.h"

/* Run the selected kernel on the full matrix. Used for warmup and validation. */
static int call_full_kernel(BLASLONG M, BLASLONG N, BLASLONG K,
                            INPUT_T *A, INPUT_T *B, OUTPUT_T *C)
{
    /* Full-kernel call means: compute the complete C matrix in one call. */
#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
    return KERNEL_SYMBOL(M, N, K, (INPUT_T)1, A, B, C, M);
#else
    return KERNEL_SYMBOL(M, N, K, 1, A, B, C, M);
#endif
}

/* Run the selected kernel on one C-column tile. */
static int call_tile_kernel(BLASLONG M, BLASLONG N, BLASLONG K,
                            BLASLONG n0, BLASLONG full_n,
                            INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                            INPUT_T *B_tile)
{
    /* Tile-kernel call means: compute only C columns n0 to n0+N-1. */
    /* M is still the full row count; N here is only the tile width. */
#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
    /* FP kernels can point directly into B and C at column n0.
     * B + n0*K means: start reading B from the first column of this tile.
     * C + n0*M means: start writing C at the first output column of this tile.
     */
    (void)full_n;
    (void)B_tile;
    /* alpha is 1, so the kernel performs C_tile = C_tile + A * B_tile. */
    return KERNEL_SYMBOL(M, N, K, (INPUT_T)1,
                         A, B + n0 * K, C + n0 * M, M);
#elif defined(OMP_KIND_INT8_RVV)
    /* INT8 RVV kernels also use direct B and C tile pointers.
     * RVV can consume the selected B columns directly from the full B matrix.
     */
    (void)full_n;
    (void)B_tile;
    /* alpha is 1, so the kernel performs C_tile = C_tile + A * B_tile. */
    return KERNEL_SYMBOL(M, N, K, 1,
                         A, B + n0 * K, C + n0 * M, M);
#elif defined(OMP_KIND_INT8_IME) || defined(OMP_KIND_INT8_MIXED)
    /* IME wrapper expects a compact B tile, so copy the selected columns first.
     * This is like taking only the needed B columns out of the big B matrix
     * and placing them in a small local table before calling IME.
     */
    if (B_tile == NULL) {
        return 1;
    }

    /* Pack B tile in K-major order so the IME wrapper receives the layout it expects. */
    for (BLASLONG k = 0; k < K; ++k) {
        for (BLASLONG column = 0; column < N; ++column) {
            /* Copy B(k, n0+column) from the full matrix into compact B_tile(k, column). */
            B_tile[(size_t)k * (size_t)N + (size_t)column] =
                B[(size_t)k * (size_t)full_n + (size_t)(n0 + column)];
        }
    }

    /* Now call IME on the compact B tile and the matching C output strip. */
    return KERNEL_SYMBOL(M, N, K, 1, A, B_tile, C + n0 * M, M);
#else
#error "Unsupported OpenMP kernel kind"
#endif
}

#endif /* OPENMP_TILED_GEMM_DISPATCH_H */


