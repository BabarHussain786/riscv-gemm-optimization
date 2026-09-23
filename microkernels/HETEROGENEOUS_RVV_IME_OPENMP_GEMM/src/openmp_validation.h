#ifndef OPENMP_VALIDATION_H
#define OPENMP_VALIDATION_H

/*
 * VALIDATION ROADMAP
 * ==================
 * This file answers one simple question: did OpenMP change the answer?
 *
 * Step 1 -> Compute C_reference directly from the GEMM equation.
 * Step 2 -> OpenMP computes C_actual by sharing output strips among workers.
 * Step 3 -> This file compares every matching position in the two matrices.
 * Step 4 -> It counts wrong values and remembers the largest difference.
 *
 * INT8/IME must match exactly because both paths produce INT32 values.
 * FP32/FP64 use a small tolerance because floating-point rounding can differ.
 * The reference never calls the tested kernel, so a kernel cannot validate
 * itself. A mismatch count of zero means the tested path matches independent
 * matrix multiplication.
 */

static int compute_independent_reference(
    BLASLONG M, BLASLONG N, BLASLONG K,
    const INPUT_T *A, const INPUT_T *B,
    const OUTPUT_T *C_initial, OUTPUT_T *reference,
    size_t *overflow_count)
{
    size_t overflow = 0;

#if defined(OMP_KIND_FP32)
#pragma omp parallel for schedule(static)
    for (BLASLONG column = 0; column < N; ++column) {
        for (BLASLONG row = 0; row < M; ++row) {
            double sum = 0.0;
            for (BLASLONG k = 0; k < K; ++k) {
                sum += (double)A[k * M + row] *
                       (double)B[k * N + column];
            }
            reference[column * M + row] =
                (OUTPUT_T)((double)C_initial[column * M + row] + sum);
        }
    }
#elif defined(OMP_KIND_FP64)
#pragma omp parallel for schedule(static)
    for (BLASLONG column = 0; column < N; ++column) {
        for (BLASLONG row = 0; row < M; ++row) {
            long double sum = 0.0L;
            for (BLASLONG k = 0; k < K; ++k) {
                sum += (long double)A[k * M + row] *
                       (long double)B[k * N + column];
            }
            reference[column * M + row] =
                (OUTPUT_T)((long double)C_initial[column * M + row] + sum);
        }
    }
#else
#pragma omp parallel for schedule(static) reduction(+:overflow)
    for (BLASLONG column = 0; column < N; ++column) {
        for (BLASLONG row = 0; row < M; ++row) {
            int64_t sum = 0;
            int64_t value;

            for (BLASLONG k = 0; k < K; ++k) {
                sum += (int64_t)A[k * M + row] *
                       (int64_t)B[k * N + column];
            }
            value = (int64_t)C_initial[column * M + row] + sum;
            if (value < INT32_MIN || value > INT32_MAX) {
                ++overflow;
                reference[column * M + row] = 0;
            } else {
                reference[column * M + row] = (OUTPUT_T)value;
            }
        }
    }
#endif

    *overflow_count = overflow;
    return overflow == 0 ? 0 : -1;
}

static size_t compare_outputs(const OUTPUT_T *actual, const OUTPUT_T *reference,
                              size_t n, double *max_error)
{
    /* ROADMAP BLOCK 1: Start both reported validation values at zero. */
    /* mismatches counts how many C elements are wrong. */
    size_t mismatches = 0;

    /* largest stores the biggest absolute difference seen so far. */
    double largest = 0.0;

    /* ROADMAP BLOCK 2: Visit every output element C[i] exactly once. */
    for (size_t i = 0; i < n; ++i) {
#if defined(OMP_KIND_FP32)
        /* ROADMAP BLOCK 3A: FP32 allows small absolute and relative error. */
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
        /* ROADMAP BLOCK 3B: FP64 uses the same idea with a smaller tolerance. */
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
        /* ROADMAP BLOCK 3C: INT8/IME requires exact INT32 equality. */
        double error = fabs((double)((int64_t)actual[i] - (int64_t)reference[i]));
        double tolerance = 0.0;
#endif
        /* ROADMAP BLOCK 4: Save the largest difference seen so far. */
        if (error > largest) {
            largest = error;
        }

        /* If this value is too different, count one wrong output position. */
        if (error > tolerance) {
            ++mismatches;
        }
    }

    /* ROADMAP BLOCK 5: Return both validation results to the main program. */
    *max_error = largest;
    return mismatches;
}

#endif /* OPENMP_VALIDATION_H */
