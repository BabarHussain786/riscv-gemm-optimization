#ifndef OPENMP_TILED_GEMM_DATA_H
#define OPENMP_TILED_GEMM_DATA_H

/*
 * What this file does
 * -------------------
 * This file creates simple repeatable matrix values.
 * It fills A, B, and the starting C before the benchmark runs.
 *
 * Simple reason:
 *   If every run uses random data, debugging becomes confusing.
 *   If every run uses the same pattern, the result can be checked again.
 */

#include "openmp_tiled_gemm_config.h"

#if defined(OMP_KIND_FP32) || defined(OMP_KIND_FP64)
/*
 * Fill FP32/FP64 input values.
 * The pattern repeats small numbers around zero:
 *   -0.6, -0.5, ..., 0.5, 0.6, then repeat.
 * These values are small enough to avoid huge floating-point growth.
 */
static void fill_input(INPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (INPUT_T)((int)(i % 13) - 6) * (INPUT_T)0.1;
    }
}

/* Starting C uses the same floating-point pattern as A and B. */
static void fill_output(OUTPUT_T *x, size_t n)
{
    fill_input(x, n);
}
#else
/*
 * Fill INT8 input values.
 * The pattern is small signed integers:
 *   -6, -5, ..., 5, 6, then repeat.
 */
static void fill_input(INPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (INPUT_T)((int)(i % 13) - 6);
    }
}

/*
 * Fill INT32 output C with a small starting value.
 * GEMM does not overwrite C from zero; it updates C as C = C + A*B.
 */
static void fill_output(OUTPUT_T *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (OUTPUT_T)((int)(i % 7) - 3);
    }
}
#endif

#endif /* OPENMP_TILED_GEMM_DATA_H */
