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
detects the active hardware profile internally. On K1/A60, one native step
multiplies `A(4x8)` by `B(8x4)` and accumulates `C(4x4)`. The 8x4 software
kernel combines two native 4x4 output blocks, while the 8x8 software kernel
combines four. On K3/A100, one native step multiplies `A(8x16)` by `B(16x8)`
and accumulates `C(8x8)`.

Each native family uses `zvl128b` as the minimum compile-time vector-length
contract. The measured K1 hardware VLEN is 256 bits, so these kernels execute
on its 256-bit vector implementation without changing their ISA compatibility
label. The families include lmul1 and lmulmf2, with unroll 1, 2, 4, and 8
variants. The public SpacemiT
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
+-- RVV_IME_GEMM_ACCURACY_VALIDATION/
+-- HETEROGENEOUS_RVV_IME_OPENMP_GEMM/
+-- run_k1_01_rvv_ime_0_7_1024.sh
+-- run_k1_single_core_0_7_1024.sh
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

Run the focused K1 IME-versus-RVV equivalence campaign:

```bash
bash RVV_IME_GEMM_ACCURACY_VALIDATION/run_int8_ime_vs_rvv_accuracy.sh k1 1
```

INT8 validation compares every INT32 result element against a wider-integer
reference computed from the same full-range signed INT8 matrices. Read
`RVV_IME_GEMM_ACCURACY_VALIDATION/README.md` for the complete method and CSV
fields.

### INT8 IME Native and RVV Fallback Campaigns

The campaign validates only the IME-native and corresponding RVV-fallback
INT8-to-INT32 paths. It adapts reference-comparison ideas from the Ozaki
accuracy analysis; it does not implement the Ozaki slicing algorithm.

```bash
INPUT_CLASSES="bounded_uniform full_range_uniform mixed_magnitude cancellation_stress" \
bash RVV_IME_GEMM_ACCURACY_VALIDATION/run_int8_ime_vs_rvv_accuracy.sh k1 1
```

The generated summaries report exact matches, mismatches, maximum absolute
error, overflow evidence, and IME/RVV equivalence for each configuration.

## OpenMP Tiled Extension

The OpenMP extension evaluates homogeneous RVV, homogeneous IME, and mixed
RVV-IME matrix-level execution after single-core validation:

```bash
cd HETEROGENEOUS_RVV_IME_OPENMP_GEMM
M=1024 N=1024 K=1024 RUNS=6 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

Read `HETEROGENEOUS_RVV_IME_OPENMP_GEMM/README.md` for execution modes, core
placement, tile scheduling, validation, timing scope, and result files.

## Requirements

- RISC-V Linux with RVV support.
- GCC with RVV intrinsic support.
- GNU Make, Bash, and `taskset`.
- IME-capable SpacemiT cores for native IME execution.

## Research Notice

This repository is maintained for research work at the University of
Salerno, Department of Computer Science. See `LICENSE`.
