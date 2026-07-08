#ifndef OPENMP_TILED_GEMM_VALIDATION_H
#define OPENMP_TILED_GEMM_VALIDATION_H

/*
 * What this file does
 * -------------------
 * This file checks whether OpenMP tiling changed the answer.
 *
 * Very simple idea:
 *   1. First compute C once using the selected kernel in serial form.
 *      This gives the reference answer.
 *   2. Then compute C again using OpenMP tiled execution.
 *   3. Compare both C arrays element by element.
 *
 * If both answers match, the tile splitting is correct.
 * If they differ, then some tile may have been missed, overwritten, or called wrongly.
 */

#include "openmp_tiled_gemm_config.h"

static size_t compare_outputs(const OUTPUT_T *actual, const OUTPUT_T *reference,
                              size_t n, double *max_error)
{
    /* mismatches counts how many C elements are wrong. */
    size_t mismatches = 0;

    /* largest stores the biggest absolute difference seen so far. */
    double largest = 0.0;

    /* Check every output element C[i]. */
    for (size_t i = 0; i < n; ++i) {
#if defined(OMP_KIND_FP32)
        /* FP32 uses tolerance because tiny floating-point differences are normal. */
        double actual_value = (double)actual[i];
        double reference_value = (double)reference[i];
        double error;
        double tolerance;
        if (!isfinite(actual_value) || !isfinite(reference_value)) {
            ++mismatches;
            largest = INFINITY;
            continue;
        }
        error = fabs(actual_value - reference_value);
        tolerance = 1e-5 + 1e-5 * fabs(reference_value);
#elif defined(OMP_KIND_FP64)
        /* FP64 also uses tolerance, but the tolerance is much smaller. */
        double actual_value = (double)actual[i];
        double reference_value = (double)reference[i];
        double error;
        double tolerance;
        if (!isfinite(actual_value) || !isfinite(reference_value)) {
            ++mismatches;
            largest = INFINITY;
            continue;
        }
        error = fabs(actual_value - reference_value);
        tolerance = 1e-12 + 1e-12 * fabs(reference_value);
#else
        /* INT8 paths produce INT32 values, so exact equality is expected. */
        double error = fabs((double)((int64_t)actual[i] - (int64_t)reference[i]));
        double tolerance = 0.0;
#endif
        /* Keep the largest absolute error for reporting. */
        if (error > largest) {
            largest = error;
        }

        /* If this element is outside tolerance, count it as a mismatch. */
        if (error > tolerance) {
            ++mismatches;
        }
    }

    /* Return both summary values to the benchmark driver. */
    *max_error = largest;
    return mismatches;
}

#endif /* OPENMP_TILED_GEMM_VALIDATION_H */
