# RISC-V RVV and SpacemiT IME GEMM Microkernels

This repository contains ZVL256 RISC-V GEMM micro-kernels and reproducible
single-core, heterogeneous OpenMP, and numerical-accuracy experiments for
SpacemiT K1 and K3 systems.

## Workloads

| Family | Computation | Output | Software tiles | Backend |
|---|---|---|---|---|
| FP32 SGEMM | FP32 x FP32 | FP32 | 8x4, 8x8 | RVV |
| FP64 DGEMM | FP64 x FP64 | FP64 | 8x4, 8x8 | RVV |
| INT8 IGEMM | INT8 x INT8 | INT32 | 8x4, 8x8 | RVV widening MAC |
| Native IME | INT8 x INT8 | INT32 | 8x4, 8x8 | SpacemiT VMADOT |

All public kernel names, source symbols, Makefiles, and campaign scripts use
the `zvl256b` target. Kernel variants cover LMUL and K-loop unroll factors
encoded directly in each directory name. Experimental IME `lmulmf2` variants
remain disabled unless `ENABLE_EXPERIMENTAL_MF2=1` is requested.

## Repository Layout

```text
riscv-rvv-ime-gemm-microkernels/
+-- GEMM_RVV_FP32_INT8_8x4_Baseline/
+-- GEMM_RVV_FP32_INT8_8x8_Baseline/
+-- GEMM_RVV_FP64_INT8_8x4_Baseline/
+-- GEMM_RVV_FP64_INT8_8x8_Baseline/
+-- IME_NATIVE_KERNELS/
+-- HETEROGENEOUS_RVV_IME_OPENMP_GEMM/
+-- RVV_IME_GEMM_ACCURACY_VALIDATION/
+-- run_k1_single_core_0_7_1024.sh
+-- run_k1_01_rvv_ime_0_7_1024.sh
+-- run_k3_rvv_ime_0_15_1024.sh
```

The four baseline folders contain standalone FP32, FP64, and INT8 RVV kernel
families for 8x4 and 8x8 output tiles. `IME_NATIVE_KERNELS` contains native
IME wrappers and their matching RVV widening kernels. Every kernel directory
contains its source, benchmark driver, and Makefile.

## Single-Core Characterization

K1 benchmarks RVV on cores 0-7 and native IME on cores 0-3:

```bash
bash run_k1_single_core_0_7_1024.sh
```

The wrapper calls the corrected full launcher:

```bash
bash run_k1_01_rvv_ime_0_7_1024.sh
```

K3 benchmarks RVV cores 0-7 and the IME domain on cores 8-15:

```bash
bash run_k3_rvv_ime_0_15_1024.sh
```

## Heterogeneous OpenMP GEMM

The OpenMP module implements the two-level structure requested for K1:

```text
outer level: 2 cluster controllers
  IME cluster: 4 workers pinned to cores 0-3
  RVV cluster: 4 workers pinned to cores 4-7
inner level: each cluster uses omp for schedule(static)
```

The output matrix is split into fixed column-tile ranges once. There is no
central tile queue, atomic tile claim, or dynamic work scheduler. The IME team
calls the native IME wrapper directly; the RVV team calls the matching RVV
widening micro-kernel directly. The default static workload ratio is 4:1 and
can be changed through two environment values.

```bash
cd HETEROGENEOUS_RVV_IME_OPENMP_GEMM
M=1024 N=1024 K=1024 TILE_N=32 RUNS=6 \
MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

The benchmark uses cluster-aware placement: workers are pinned to their target
cores and own separate output regions. The available system memory-node list is
recorded in every OpenMP campaign log.

## Accuracy Validation

The kept accuracy method compares native IME and RVV INT8-to-INT32 outputs
against an independent INT64-accumulation reference:

```bash
cd RVV_IME_GEMM_ACCURACY_VALIDATION
bash run_int8_ime_vs_rvv_accuracy.sh k1 1
```

For every output element, the checker reports exact-match status, signed and
absolute error, mismatch count, maximum error, overflow status, and error
histograms. See `RVV_IME_GEMM_ACCURACY_VALIDATION/README.md` for details.

## Requirements

- RISC-V Linux with RVV and a 256-bit vector implementation.
- GCC with RVV intrinsic and OpenMP support.
- GNU Make, Bash, `taskset`, and standard Linux affinity interfaces.
- IME-capable SpacemiT cores for native IME execution.

## Research Notice

This repository is maintained for research at the Department of Computer
Science, University of Salerno. See `LICENSE`.
