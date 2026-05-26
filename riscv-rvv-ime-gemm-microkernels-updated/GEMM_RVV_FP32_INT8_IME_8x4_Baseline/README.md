# FP32, INT8, and IME GEMM Baseline: 8x4 Tiles

This baseline groups the RVV and IME microkernels used to evaluate FP32 plus INT8 workloads with the 8x4 tile shape. It is one of the four top-level benchmark groups used by the K1 and K3 campaign scripts.

## Active Kernel Families

| Folder | Purpose | Variants |
|---|---|---:|
| `RVV_SGEMM_FP32_8x4/` | FP32 x FP32 -> FP32 RVV SGEMM kernels | 20 |
| `RVV_IGEMM_INT8_I8I32_8x4/` | INT8 x INT8 -> INT32 RVV IGEMM kernels | 28 |
| `IME_GEMM_INT8_I8I32_8x4_RVV_Fallback/` | INT8 x INT8 -> INT32 IME kernels with local RVV fallback | 8 |

## Directory Contract

Each kernel variant is stored in its own folder:

```text
<kernel_variant>/
+-- <kernel_variant>.c
+-- *_bench.c
+-- Makefile
```

Each IME variant is self-contained:

```text
<ime_kernel_variant>/
+-- ime_kernel_*.c
+-- rvv_fallback.c
+-- ime_bench.c
+-- Makefile
```

## Run This Baseline

```bash
cd ~/riscv-rvv-ime-gemm-microkernels/GEMM_RVV_FP32_INT8_IME_8x4_Baseline
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
- INT8 and IME kernels compute INT8 x INT8 dot products into INT32 outputs.
- Boundary cleanup handles leftover rows or columns outside the full micro-tile path.
