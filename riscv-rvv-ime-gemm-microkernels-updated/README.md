# RISC-V RVV and SpacemiT IME GEMM Microkernels

This repository contains RISC-V GEMM microkernels for reproducible RVV and
SpacemiT IME experiments on K1 and K3 boards. It is part of PhD research at
the Department of Computer Science, University of Salerno.

## Workloads

| Family | Operation | Output | Software tiles | Backend |
| --- | --- | --- | --- | --- |
| FP32 SGEMM | FP32 x FP32 | FP32 | 8x4, 8x8 | RVV |
| FP64 DGEMM | FP64 x FP64 | FP64 | 8x4, 8x8 | RVV |
| INT8 IGEMM | INT8 x INT8 | INT32 | 8x4, 8x8 | RVV |
| Native IME | INT8 x INT8 | INT32 | 8x4, 8x8 | SpacemiT VMADOT |

## Native IME Sources

The current IME implementations remain in two shared families. Each source
detects the active hardware profile internally: K1/A60 uses native `4x8x4`
VMADOT tiles and K3/A100 uses native `8x16x8` VMADOT tiles. The public folder
layout stays compact while each board executes its correct private packing and
write-back path.

Each native family is kept as a zvl128b-only kernel set, with lmul1 and
lmulmf2, and unroll 1, 2, 4, and 8 variants. The public SpacemiT
documentation describes `LMUL=1`; therefore, `lmulmf2` rows are retained only
as explicit experimental target-board measurements. Default benchmarks skip
them. Enable them only after a board-specific capability check:

```bash
ENABLE_EXPERIMENTAL_MF2=1 bash run_k1_01_rvv_ime_0_7_1024.sh
ENABLE_EXPERIMENTAL_MF2=1 bash run_k3_rvv_ime_0_15_1024.sh
```

## Layout

```text
riscv-rvv-ime-gemm-microkernels/
+-- GEMM_RVV_FP32_INT8_8x4_Baseline/
+-- GEMM_RVV_FP32_INT8_8x8_Baseline/
+-- GEMM_RVV_FP64_INT8_8x4_Baseline/
+-- GEMM_RVV_FP64_INT8_8x8_Baseline/
+-- IME_NATIVE_KERNELS/
|   +-- IME_GEMM_INT8_I8I32_8x4_NATIVE/
|   +-- IME_GEMM_INT8_I8I32_8x8_NATIVE/
+-- ACCURACY_CHECKER/
+-- OPENMP_TILED_GEMM_ALL_KERNELS/
+-- run_k1_01_rvv_ime_0_7_1024.sh
+-- run_k3_rvv_ime_0_15_1024.sh
```

Each native IME variant is self-contained: its directory contains the native
IME source, `rvv_fallback.c`, `ime_bench.c`, and a `Makefile`. The RVV
fallback is available for normal execution. Accuracy builds disable fallback
execution so that a successful IME row proves that VMADOT executed.

All IME wrappers receive the same full K-major caller layout:
`A[k*M+row]`, `B[k*N+column]`, and column-major `C[column*ldc+row]`.

## Full Single-Core Benchmarks

K1 uses RVV cores `0-7` and native IME cores `0-3`:

```bash
bash run_k1_01_rvv_ime_0_7_1024.sh
```

K3 uses RVV cores `0-7` and the A100 IME domain on cores `8-15`:

```bash
bash run_k3_rvv_ime_0_15_1024.sh
```

The K3 launcher uses `/proc/set_ai_thread` before pinning A100 runs when the
Bianbu kernel exposes that interface.

## Accuracy

Run each board path explicitly:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-rvv 1
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-ime 1
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-rvv 1
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-ime 1
```

INT8 validation compares every INT32 result element against a wider-integer
reference computed from the same full-range signed INT8 matrices. Read
`ACCURACY_CHECKER/README.md` for the complete method and CSV fields.

### INT8 IME Native and RVV Fallback Campaigns

These focused campaigns validate only the IME-native and corresponding
RVV-fallback INT8-to-INT32 paths. Their paper names identify related
experimental ideas; they are adaptations for this project, not exact
reproductions.

```bash
ENABLE_MF2=1 bash ACCURACY_CHECKER/INT8_IME_RVV_FALLBACK_METHODS/run_01_ozaki_inspired_elementwise_error.sh k1 1
ENABLE_MF2=1 bash ACCURACY_CHECKER/INT8_IME_RVV_FALLBACK_METHODS/run_02_cascading_gemm_inspired_multi_input.sh k1 1
ENABLE_MF2=1 bash ACCURACY_CHECKER/INT8_IME_RVV_FALLBACK_METHODS/run_03_narrow_range_inspired_validation.sh k1 1
```

The campaigns respectively report element-wise exact-error histograms,
comparisons across multiple controlled input classes, and range/overflow plus
normalized-error metrics. Read
`ACCURACY_CHECKER/INT8_IME_RVV_FALLBACK_METHODS/README.md` for details.

## OpenMP Tiled Extension

The optional OpenMP extension evaluates matrix-level parallel execution after
single-core validation:

```bash
OMP_NUM_THREADS=4 bash OPENMP_TILED_GEMM_ALL_KERNELS/run_all_openmp_kernels.sh k1-ime
OMP_NUM_THREADS=8 bash OPENMP_TILED_GEMM_ALL_KERNELS/run_all_openmp_kernels.sh k3-ime
```

## Requirements

- RISC-V Linux with RVV support.
- GCC with RVV intrinsic support.
- GNU Make, Bash, and `taskset`.
- IME-capable SpacemiT cores for native IME execution.

## Research Notice

This repository is maintained for research work at the University of
Salerno, Department of Computer Science. See `LICENSE`.
