#ifndef OPENMP_VALIDATION_H
#define OPENMP_VALIDATION_H

/*
 * Validation roadmap
 * ------------------
 * This file answers one question: did OpenMP tiling change the result?
 *
 * Step 1 -> serial reference:
 *   Run the same selected kernel once on the full matrix C.
 *   This is the trusted result for this benchmark run.
 *
 * Step 2 -> OpenMP tiled result:
 *   Run the tiled OpenMP version, where workers compute different C columns.
 *
 * Step 3 -> element-by-element comparison:
 *   For every output value, compare C_openmp[i] with C_reference[i].
 *   INT8/IME expects exact INT32 equality.
 *   FP32/FP64 allows a small floating-point tolerance.
 *
 * If mismatch_count is zero, the C-column split and kernel dispatch are correct.
 */

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

#endif /* OPENMP_VALIDATION_H */


