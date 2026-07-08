#ifndef OPENMP_TILED_GEMM_CONFIG_H
#define OPENMP_TILED_GEMM_CONFIG_H

/*
 * What this file does
 * -------------------
 * This file is the "mode selector" for the OpenMP benchmark.
 * The C source is the same, but the shell script compiles it many times.
 * Each compile command adds one -D option, for example:
 *   -D OMP_KIND_FP64
 *   -D OMP_KIND_INT8_RVV
 *   -D OMP_KIND_INT8_MIXED
 *
 * Simple idea:
 *   The benchmark driver asks: "What type of matrix math am I testing?"
 *   This file answers that question by choosing:
 *     1. input datatype,
 *     2. output datatype,
 *     3. printed performance unit,
 *     4. exact micro-kernel function signature.
 *
 * Why this is useful:
 *   FP32 and FP64 use floating-point output and print GFLOPS.
 *   INT8 RVV, IME, and mixed modes use INT8 inputs with INT32 output
 *   and print GOPS.
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* BLASLONG is the integer type used by OpenBLAS-style micro-kernel APIs. */
typedef long BLASLONG;

/*
 * KERNEL_SYMBOL is the real kernel function name.
 * The shell script provides it during compilation.
 * Example: KERNEL_SYMBOL may become dgemm_kernel_8x4_zvl128b_lmul4_unroll1.
 */
#ifndef KERNEL_SYMBOL
#error "KERNEL_SYMBOL must be defined by the build script"
#endif

/*
 * FP32 RVV mode
 * -------------
 * A and B are float.
 * C is float.
 * Performance is floating-point work, so the script prints GFLOPS.
 */
#if defined(OMP_KIND_FP32)
typedef float INPUT_T;
typedef float OUTPUT_T;
#define METRIC_NAME "GFLOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_FP_TOLERANCE"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  INPUT_T alpha, INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                  BLASLONG ldc);

/*
 * FP64 RVV mode
 * -------------
 * A and B are double.
 * C is double.
 * This is the high-precision floating-point path.
 */
#elif defined(OMP_KIND_FP64)
typedef double INPUT_T;
typedef double OUTPUT_T;
#define METRIC_NAME "GFLOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_FP_TOLERANCE"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  INPUT_T alpha, INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                  BLASLONG ldc);

/*
 * INT8 RVV mode
 * -------------
 * A and B are int8_t.
 * C is int32_t.
 * Reason: many INT8 products are added together, so INT32 is safer than INT16.
 */
#elif defined(OMP_KIND_INT8_RVV)
typedef int8_t INPUT_T;
typedef int32_t OUTPUT_T;
#define METRIC_NAME "GOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_EXACT_INT32"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  OUTPUT_T alpha, INPUT_T *A, INPUT_T *B, OUTPUT_T *C,
                  BLASLONG ldc);

/*
 * INT8 IME and mixed mode
 * -----------------------
 * These also use INT8 input and INT32 output.
 * Difference from INT8 RVV:
 *   IME mode calls the native IME wrapper.
 *   Mixed mode lets IME-capable cores use IME and other cores use RVV fallback.
 */
#elif defined(OMP_KIND_INT8_IME) || defined(OMP_KIND_INT8_MIXED)
typedef int8_t INPUT_T;
typedef int32_t OUTPUT_T;
#define METRIC_NAME "GOPS"
#define VALIDATION_NAME "SERIAL_SAME_KERNEL_EXACT_INT32"
int KERNEL_SYMBOL(BLASLONG M, BLASLONG N, BLASLONG K,
                  OUTPUT_T alpha, const INPUT_T *A, const INPUT_T *B,
                  OUTPUT_T *C, BLASLONG ldc);
#else
#error "Define one OMP_KIND_* macro"
#endif

/*
 * Safety rule for IME OpenMP builds.
 * The IME wrapper expects a full K-major input contract.
 * If the shell script forgets that flag, stop at compile time.
 */
#if (defined(OMP_KIND_INT8_IME) || defined(OMP_KIND_INT8_MIXED)) && !defined(OMP_IME_INPUT_FULL_MATRIX)
#error "Native IME OpenMP builds require full K-major input"
#endif

/*
 * K1 mixed-core map
 * -----------------
 * For this project, K1 is treated as two execution domains:
 *   cores 0-3: IME-capable domain
 *   cores 4-7: RVV domain
 *
 * The weights control how many tiles a worker claims in one scheduling step.
 * Default: IME workers claim 4 tiles; RVV workers claim 1 tile.
 * This only changes work sharing. It does not change the GEMM math.
 */
#if defined(OMP_KIND_INT8_MIXED)
#ifndef MIXED_IME_CORE_FIRST
#define MIXED_IME_CORE_FIRST 0
#endif
#ifndef MIXED_IME_CORE_LAST
#define MIXED_IME_CORE_LAST 3
#endif
#ifndef MIXED_IME_TILE_WEIGHT
#define MIXED_IME_TILE_WEIGHT 4
#endif
#ifndef MIXED_RVV_TILE_WEIGHT
#define MIXED_RVV_TILE_WEIGHT 1
#endif

/* Return 1 when this CPU id belongs to the IME-capable core group. */
static int mixed_cpu_uses_ime(int cpu)
{
    return cpu >= MIXED_IME_CORE_FIRST && cpu <= MIXED_IME_CORE_LAST;
}

/* Decide how many C-column tiles this CPU should claim at a time. */
static BLASLONG mixed_tile_chunk_for_cpu(int cpu)
{
    return mixed_cpu_uses_ime(cpu) ? MIXED_IME_TILE_WEIGHT : MIXED_RVV_TILE_WEIGHT;
}
#endif

#endif /* OPENMP_TILED_GEMM_CONFIG_H */
