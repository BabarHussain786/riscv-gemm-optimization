#ifndef OPENMP_KERNEL_DISPATCH_H
#define OPENMP_KERNEL_DISPATCH_H

/*
 * KERNEL DISPATCH ROADMAP
 * =======================
 * The scheduler gives one output strip to a worker. This file sends that strip
 * to the correct mathematical engine.
 *
 * Step 1 -> Receive the strip position, width, and worker-private buffers.
 * Step 2 -> If this is an IME worker, make a compact B strip.
 * Step 3 -> If this is an RVV worker, pack A and B into the RVV layout.
 * Step 4 -> Call the selected IME or RVV micro-kernel.
 * Step 5 -> Return the kernel status to the OpenMP scheduler.
 *
 * This file does not decide who owns a tile. It only calls the correct kernel.
 */

#if defined(OMP_KIND_INT8_MIXED)
/*
 * ROADMAP BLOCK 1: Required names and tile shapes for mixed INT8 execution
 * -------------------------------------------------------------------------
 * The build script supplies the matching IME and RVV function names.
 */
#ifndef RVV_KERNEL_SYMBOL
#error "RVV_KERNEL_SYMBOL must name the matching RVV widening kernel"
#endif
#ifndef OMP_KERNEL_MR
#error "OMP_KERNEL_MR must describe the mixed micro-kernel row tile"
#endif
#ifndef OMP_KERNEL_NR
#error "OMP_KERNEL_NR must describe the mixed micro-kernel column tile"
#endif

int RVV_KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                      int32_t alpha, int8_t *A, int8_t *B,
                      int32_t *C, BLASLONG ldc);

/*
 * ROADMAP BLOCK 2: Pick the next RVV row or column block
 * ------------------------------------------------------
 * Use the full micro-kernel size when possible, then 4, 2, or 1 for leftovers.
 */
static BLASLONG packed_block_size(BLASLONG left, BLASLONG full_block)
{
    if (left >= full_block) return full_block;
    if (full_block > 4 && left >= 4) return 4;
    if (left >= 2) return 2;
    return 1;
}

/*
 * ROADMAP BLOCK 3: Prepare and run one RVV output strip
 * -----------------------------------------------------
 * The original matrices are easy to understand, but the low-level RVV kernel
 * expects packed blocks. Each worker packs into its own private memory.
 */
static int call_mixed_rvv_tile_kernel(BLASLONG M, BLASLONG N, BLASLONG K,
                                      BLASLONG n0, BLASLONG full_n,
                                      INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                                      INPUT_T *A_pack, INPUT_T *B_pack)
{
    BLASLONG m_top = 0;
    BLASLONG n_top = 0;

    if (A_pack == NULL || B_pack == NULL) {
        return 1;
    }

    /* Pack A: keep the rows of one micro-kernel block together for every k. */
    while (m_top < M) {
        BLASLONG rows = packed_block_size(M - m_top, OMP_KERNEL_MR);
        for (BLASLONG k = 0; k < K; ++k) {
            for (BLASLONG row = 0; row < rows; ++row) {
                A_pack[m_top * K + k * rows + row] =
                    A[k * M + m_top + row];
            }
        }
        m_top += rows;
    }

    /* Pack B: copy only the output columns owned by this worker's strip. */
    while (n_top < N) {
        BLASLONG columns = packed_block_size(N - n_top, OMP_KERNEL_NR);
        for (BLASLONG k = 0; k < K; ++k) {
            for (BLASLONG column = 0; column < columns; ++column) {
                B_pack[n_top * K + k * columns + column] =
                    B[k * full_n + n0 + n_top + column];
            }
        }
        n_top += columns;
    }

    /* Run INT8 x INT8 with INT32 accumulation through the RVV kernel. */
    return RVV_KERNEL_SYMBOL(M, N, K, 1, A_pack, B_pack,
                             C + n0 * M, M);
}
#endif

/*
 * ROADMAP BLOCK 4: Full-matrix call used outside the timed tile loop
 * ------------------------------------------------------------------
 * Warmup and serial validation need one ordinary full-matrix kernel call.
 */
static int call_full_kernel(BLASLONG M, BLASLONG N, BLASLONG K,
                            INPUT_T *A, INPUT_T *B, OUTPUT_T *C)
{
#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
    return KERNEL_SYMBOL(M, N, K, (INPUT_T)1, A, B, C, M);
#else
    return KERNEL_SYMBOL(M, N, K, 1, A, B, C, M);
#endif
}

/*
 * ROADMAP BLOCK 5: Send one output strip to its selected path
 * -----------------------------------------------------------
 * n0 is the first output column. N is the width of this strip.
 */
static int call_tile_kernel(BLASLONG M, BLASLONG N, BLASLONG K,
                            BLASLONG n0, BLASLONG full_n,
                            INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                            INPUT_T *A_pack, INPUT_T *B_tile,
                            int use_rvv_path)
{
#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
    /* FP32/FP64: call the selected RVV floating-point kernel directly. */
    (void)full_n;
    (void)A_pack;
    (void)B_tile;
    (void)use_rvv_path;
    return KERNEL_SYMBOL(M, N, K, (INPUT_T)1,
                         A, B + n0 * K, C + n0 * M, M);

#elif defined(OMP_KIND_INT8_RVV)
    /* Pure INT8 RVV mode: inputs already follow the selected kernel contract. */
    (void)full_n;
    (void)A_pack;
    (void)B_tile;
    (void)use_rvv_path;
    return KERNEL_SYMBOL(M, N, K, 1,
                         A, B + n0 * K, C + n0 * M, M);

#elif defined(OMP_KIND_INT8_IME)
    /* Pure IME mode: first copy this worker's B columns into a compact strip. */
    (void)A_pack;
    (void)use_rvv_path;
    if (B_tile == NULL) return 1;
    for (BLASLONG k = 0; k < K; ++k) {
        for (BLASLONG column = 0; column < N; ++column) {
            B_tile[k * N + column] = B[k * full_n + n0 + column];
        }
    }
    return KERNEL_SYMBOL(M, N, K, 1, A, B_tile, C + n0 * M, M);

#elif defined(OMP_KIND_INT8_MIXED)
    /* Mixed cluster 1 uses the explicit RVV packing and kernel path. */
    if (use_rvv_path) {
        return call_mixed_rvv_tile_kernel(M, N, K, n0, full_n,
                                          A, B, C, A_pack, B_tile);
    }

    /* Mixed cluster 0 copies a compact B strip and calls native IME. */
    if (B_tile == NULL) return 1;
    for (BLASLONG k = 0; k < K; ++k) {
        for (BLASLONG column = 0; column < N; ++column) {
            B_tile[k * N + column] = B[k * full_n + n0 + column];
        }
    }
    return KERNEL_SYMBOL(M, N, K, 1, A, B_tile, C + n0 * M, M);
#else
#error "Unsupported OpenMP kernel kind"
#endif
}

#endif /* OPENMP_KERNEL_DISPATCH_H */
