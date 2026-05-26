# RISC-V RVV and SpacemiT IME GEMM Microkernels

This repository contains a curated set of RISC-V GEMM microkernels for evaluating RVV vector execution and SpacemiT IME acceleration across FP32, FP64, and INT8 workloads. The project is organized for reproducible single-core benchmarking on K1 and K3 boards, with kernel variants separated by datatype, tile shape, vector length setting, LMUL label, and unroll factor.

The work is part of PhD research at the Department of Computer Science, University of Salerno.

## Scope

| Family | Operation | Output | Tile Shapes | Backend |
|---|---:|---:|---:|---|
| FP32 SGEMM | FP32 x FP32 | FP32 | 8x4, 8x8 | RVV |
| FP64 DGEMM | FP64 x FP64 | FP64 | 8x4, 8x8 | RVV |
| INT8 IGEMM | INT8 x INT8 | INT32 | 8x4, 8x8 | RVV |
| IME INT8 GEMM | INT8 x INT8 | INT32 | 8x4, 8x8 | SpacemiT IME with local RVV fallback |

## Repository Layout

```text
riscv-rvv-ime-gemm-microkernels/
+-- GEMM_RVV_FP32_INT8_IME_8x4_Baseline/
+-- GEMM_RVV_FP32_INT8_IME_8x8_Baseline/
+-- GEMM_RVV_FP64_INT8_IME_8x4_Baseline/
+-- GEMM_RVV_FP64_INT8_IME_8x8_Baseline/
+-- run_k1_single_core_0_7_1024.sh
+-- run_k3_rvv_ime_0_15_1024.sh
+-- README.md
+-- LICENSE
```

Each baseline directory contains three kernel families:

```text
RVV_SGEMM_FP32_*              FP32 RVV kernels
RVV_DGEMM_FP64_*              FP64 RVV kernels
RVV_IGEMM_INT8_I8I32_*        INT8 RVV kernels
IME_GEMM_INT8_I8I32_*_RVV_Fallback
                              IME kernels with self-contained RVV fallback
```

Each individual kernel variant folder contains its source file, benchmark driver, and Makefile. IME folders additionally include `rvv_fallback.c`, so the IME folder can be built without depending on the standalone INT8 RVV tree.

## Kernel Variant Naming

Kernel folders follow a fixed naming convention:

```text
<family>_kernel_<tile>_zvl<VLEN>b_lmul<LMUL>_unroll<UNROLL>
```

Examples:

```text
sgemm_kernel_8x4_zvl256b_lmul4_unroll8
igemm_kernel_8x8_zvl128b_lmulmf4_unroll2
ime_kernel_8x4_zvl256b_lmul1_unroll4
```

The name records the micro-tile shape, RVV vector-length target, LMUL label, and loop unroll factor used by the implementation.

## Build One Kernel

Run these commands on the RISC-V target board:

```bash
cd ~/riscv-rvv-ime-gemm-microkernels/GEMM_RVV_FP32_INT8_IME_8x4_Baseline/RVV_SGEMM_FP32_8x4/sgemm_kernel_8x4_zvl256b_lmul1_unroll1
make clean && make
./bench 1024 1024 1024
```

For an IME kernel on K3, run the benchmark from an IME-enabled CPU domain:

```bash
cd ~/riscv-rvv-ime-gemm-microkernels/GEMM_RVV_FP64_INT8_IME_8x4_Baseline/IME_GEMM_INT8_I8I32_8x4_RVV_Fallback/ime_kernel_8x4_zvl128b_lmul1_unroll1
make clean && make
bash -lc 'echo $$ > /proc/set_ai_thread && taskset -c 8 ./bench 1024 1024 1024'
```

## Full Benchmark Campaigns

### K1: RVV single-core campaign

```bash
cd ~/riscv-rvv-ime-gemm-microkernels
bash run_k1_single_core_0_7_1024.sh
```

Default settings:

```text
M=N=K=1024
RUNS=6
CORES=0 1 2 3 4 5 6 7
```

### K3: RVV plus IME campaign

```bash
cd ~/riscv-rvv-ime-gemm-microkernels
bash run_k3_rvv_ime_0_15_1024.sh
```

Default core domains:

```text
RVV cores: 0 1 2 3 4 5 6 7
IME cores: 8 9 10 11 12 13 14 15
```

The K3 script uses `/proc/set_ai_thread` when available before pinning IME runs to cores 8-15. This is required on Bianbu K3 systems where IME-capable A100 cores are exposed through a separate CPU domain.

## Optional Benchmark Parameters

```bash
M=512 N=512 K=512 RUNS=10 bash run_k1_single_core_0_7_1024.sh
```

```bash
M=1024 N=1024 K=1024 RUNS=6 \
RVV_CORES="0 1 2 3 4 5 6 7" \
IME_CORES="8 9 10 11 12 13 14 15" \
bash run_k3_rvv_ime_0_15_1024.sh
```

## Output Files

K1 campaigns write per-baseline results under each baseline directory:

```text
single_core_results_<M>/
+-- single_core_raw_latest.csv
+-- single_core_summary_latest.csv
+-- single_core_live_*.log
+-- raw_logs/
```

K3 campaigns write combined results at the project root:

```text
k3_rvv_ime_results_<M>/
+-- k3_rvv_ime_raw_latest.csv
+-- k3_rvv_ime_summary_latest.csv
+-- k3_rvv_ime_live_latest.log
+-- raw_logs/
```

Use the summary CSV files for plots and tables. Use raw CSV files when per-run statistics or debugging are needed.

## Requirements

- RISC-V Linux target with RVV support.
- GCC or compatible C compiler with RVV intrinsic support.
- GNU Make and Bash.
- `taskset` from util-linux for core pinning.
- SpacemiT K3 with IME-capable cores for IME hardware runs.

## Notes For Reproducibility

- The benchmark driver reports `GFLOPS` for FP32/FP64 and `GOPS` for INT8/IME.
- Each result row records successful runs, failed runs, build failures, and nonzero kernel return counts.
- For publication plots, filter to rows with successful runs and zero failure counters.
- IME kernels include a local RVV fallback implementation inside each IME variant folder.

## License And Research Notice

This project is maintained for research work at the University of Salerno, Department of Computer Science. See `LICENSE` for the repository notice.
