#ifndef OPENMP_KERNEL_DISPATCH_H
#define OPENMP_KERNEL_DISPATCH_H

/*
 * Tile-to-kernel dispatch roadmap
 * -------------------------------
 * This file is the bridge between OpenMP tiles and the real micro-kernels.
 * OpenMP only knows: "this worker owns C columns n0..n0+N-1".
 * This file turns that tile request into the correct function call.
 *
 * Step 1 -> receive one C-column tile from openmp_tile_scheduler.h.
 * Step 2 -> compute the correct B and C starting addresses for that tile.
 * Step 3 -> call the selected FP32, FP64, INT8 RVV, IME, or mixed kernel.
 *
 * Pointer math for one tile:
 *   n0       = first output column of this tile
 *   N        = number of columns inside this tile
 *   B tile   = B + n0*K       for direct FP/RVV paths
 *   C tile   = C + n0*M       for all paths
 *
 * IME special case:
 *   IME expects the selected B columns in a compact worker-local buffer.
 *   Therefore the code copies B(k, n0+column) into B_tile(k, column),
 *   then calls the IME wrapper on that compact B tile.
 */

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

#endif /* OPENMP_KERNEL_DISPATCH_H */




