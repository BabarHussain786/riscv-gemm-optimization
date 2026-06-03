# FP64 and INT8 RVV GEMM Baseline: 8x4 Tiles

This baseline groups the RVV microkernels used to evaluate FP64 and INT8 workloads with the 8x4 tile shape. Native IME kernels are stored separately under `../IME_NATIVE_KERNELS/` because their hardware tile shape depends on the board profile.

## Active Kernel Families

| Folder | Purpose | Variants |
|---|---|---:|
| `RVV_DGEMM_FP64_8x4/` | FP64 x FP64 -> FP64 RVV DGEMM kernels | 16 |
| `RVV_IGEMM_INT8_I8I32_8x4/` | INT8 x INT8 -> INT32 RVV IGEMM kernels | 28 |

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
cd ~/riscv-rvv-ime-gemm-microkernels/GEMM_RVV_FP64_INT8_IME_8x4_Baseline
bash run_single_core_0_7_all_kernels_1024.sh
```

Default settings:

```text
M=N=K=1024
RUNS=6
CORES=0 1 2 3 4 5 6 7
```

Override the defaults when needed:

```bash
M=512 N=512 K=512 RUNS=10 CORES="0 1 2 3" bash run_single_core_0_7_all_kernels_1024.sh
```

## Result Files

The per-baseline runner writes results under:

```text
single_core_results_<M>/
+-- single_core_raw_latest.csv
+-- single_core_summary_latest.csv
+-- single_core_live_*.log
+-- raw_logs/
```

Use `single_core_summary_latest.csv` for tables and plots. Use `raw_logs/` to inspect individual build or runtime behavior.

## Notes

- Kernel names encode tile shape, ZVL target, LMUL label, and unroll factor.
- INT8 kernels compute INT8 x INT8 dot products into INT32 outputs.
- Boundary cleanup handles leftover rows or columns outside the full micro-tile path.
