# OpenMP Tiled GEMM Source Map

This folder splits the OpenMP benchmark into small files so each task is easy to understand.

| File | Human meaning |
|---|---|
| `openmp_tiled_gemm_benchmark.c` | Main driver: reads matrix sizes, allocates A/B/C, runs warmup, validation, timed OpenMP region, and prints results. |
| `openmp_tiled_gemm_config.h` | Chooses the benchmark mode: FP32 RVV, FP64 RVV, INT8 RVV, INT8 IME, or mixed IME/RVV. |
| `openmp_tiled_gemm_utils.h` | Utility helpers: timing, aligned allocation, safe size multiplication, min helper, environment flags. |
| `openmp_tiled_gemm_data.h` | Creates repeatable input data for A, B, and initial C. |
| `openmp_tiled_gemm_dispatch.h` | Connects one OpenMP C-column tile to the correct RVV or IME micro-kernel call. |
| `openmp_tiled_gemm_validation.h` | Compares OpenMP tiled output against a serial same-kernel reference. |
| `openmp_tiled_gemm_parallel.h` | Main OpenMP tiling file: splits C into column tiles, assigns work to threads, and records worker/core placement. |

Main file for explaining OpenMP splitting:

```text
openmp_tiled_gemm_parallel.h
```

Main file for explaining full benchmark execution:

```text
openmp_tiled_gemm_benchmark.c
```
