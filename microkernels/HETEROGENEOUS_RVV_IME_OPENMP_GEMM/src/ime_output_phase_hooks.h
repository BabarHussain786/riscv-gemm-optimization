#ifndef IME_OUTPUT_PHASE_HOOKS_H
#define IME_OUTPUT_PHASE_HOOKS_H

/*
 * Optional timing hook for the native IME tile-to-C conversion.
 *
 * The regular kernels are compiled without IME_PHASE_TIMING and therefore
 * have no dependency on this header.  The phase-timing runner enables the
 * hook and supplies the accumulator from its benchmark driver.
 */
#include <time.h>

extern double ime_output_packing_time_sec;

static inline double ime_output_phase_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static inline void ime_output_phase_add(double start)
{
    double elapsed = ime_output_phase_now() - start;
#pragma omp atomic update
    ime_output_packing_time_sec += elapsed;
}

#endif
