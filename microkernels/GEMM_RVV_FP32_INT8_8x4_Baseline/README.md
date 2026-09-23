# FP32 and INT8 RVV GEMM Baseline: 8x4 Tiles

This baseline groups the RVV microkernels used to evaluate FP32 and INT8 workloads with the 8x4 tile shape. Native IME kernels are stored separately under `../IME_NATIVE_KERNELS/` because their hardware tile shape depends on the board profile.

## Active Kernel Families

| Folder | Purpose | Variants |
|---|---|---:|
| `RVV_SGEMM_FP32_8x4/` | FP32 x FP32 -> FP32 RVV SGEMM kernels | 20 |
| `RVV_IGEMM_INT8_I8I32_8x4/` | INT8 x INT8 -> INT32 RVV IGEMM kernels | 20 |

The baseline therefore contains 40 independently buildable kernel variants.

## Directory Contract

Each kernel variant is stored in its own folder:

```text
<kernel_variant>/
+-- <kernel_variant>.c
+-- *_bench.c
+-- Makefile
```

## Run This Baseline

```bash
cd ~/riscv-rvv-ime-gemm-microkernels/GEMM_RVV_FP32_INT8_8x4_Baseline
bash run_single_core_0_7_all_kernels_1024.sh
```

Default settings:

```text
M=N=K=1024
RUNS=6
CORES=0 1 2 3 4 5 6 7
VALIDATE=1
VALIDATE_M=15, VALIDATE_N=7, VALIDATE_K=13
```

`taskset` is required. Before timing a kernel, the runner pins it to the first
requested core and validates the irregular `15x7x13` shape. This shape exercises
the full 8x4 path and the row, column, and K cleanup paths. A kernel that fails
validation is recorded but is not benchmarked.

Override the defaults when needed:

```bash
M=512 N=512 K=512 RUNS=10 CORES="0 1 2 3" bash run_single_core_0_7_all_kernels_1024.sh
```

Set `VALIDATE=0` only when repeating a campaign whose binaries have already
passed validation.

## Result Files

The per-baseline runner writes results under:

```text
single_core_results_<M>/
+-- single_core_raw_latest.csv
+-- single_core_summary_latest.csv
+-- single_core_live_*.log
+-- raw_logs/
```

Use `single_core_summary_latest.csv` for tables and plots. It reports mean,
median, population standard deviation, minimum, maximum, average time, run
status counts, and validation status. Use `raw_logs/` to inspect each build,
validation, or timed run.

## Notes

- Kernel names encode tile shape, ZVL target, LMUL label, and unroll factor.
- INT8 kernels compute INT8 x INT8 dot products into INT32 outputs.
- Boundary cleanup handles leftover rows or columns outside the full micro-tile path.
- FP32 validation uses an independent double-accumulation reference and a
  floating-point tolerance.
- INT8 validation uses an independent INT64 reference calculation and exact
  comparison after applying the kernel's INT32 output semantics.
