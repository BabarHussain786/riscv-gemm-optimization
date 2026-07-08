#ifndef OPENMP_TILED_GEMM_UTILS_H
#define OPENMP_TILED_GEMM_UTILS_H

/*
 * What this file does
 * -------------------
 * This file contains small helper tools.
 * These helpers are not matrix multiplication.
 * They are the simple tools the benchmark needs around the matrix work:
 *   timing, memory allocation, safe size calculation, min(), and env flags.
 */

#include "openmp_tiled_gemm_config.h"

/*
 * Read the current time.
 * The benchmark calls this before and after the OpenMP tile region.
 * Difference between the two times gives total runtime in seconds.
 */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * Allocate memory aligned to 64 bytes.
 * Simple meaning: start the matrix memory at a clean hardware-friendly address.
 * This helps vector and IME code avoid unnecessary memory penalties.
 */
static void *aligned_bytes(size_t bytes)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    return ptr;
}

/*
 * Safely multiply two sizes.
 * Example: M*K tells how many elements are in A.
 * If the multiplication is too large for size_t, return 0 instead of wrapping.
 */
static int checked_size_product(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

/* Pick the smaller value. Used for the last tile, which may be shorter. */
static BLASLONG min_blaslong(BLASLONG a, BLASLONG b)
{
    return (a < b) ? a : b;
}

/*
 * Read a yes/no environment flag.
 * Example:
 *   OMP_VALIDATE=1 means validate output.
 *   OMP_VALIDATE=0 means skip validation.
 */
static int env_enabled(const char *name, int default_value)
{
    const char *value = getenv(name);
    return (value == NULL) ? default_value : atoi(value) != 0;
}

#endif /* OPENMP_TILED_GEMM_UTILS_H */
