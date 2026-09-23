# FP64 and INT8 RVV GEMM Baseline: 8x4 Tiles

This baseline contains the validated RVV microkernels used for FP64 and INT8 experiments with the 8x4 output tile.

| Folder | Computation | Variants |
|---|---|---:|
| `RVV_DGEMM_FP64_8x4/` | FP64 x FP64 -> FP64 | 12 |
| `RVV_IGEMM_INT8_I8I32_8x4/` | INT8 x INT8 -> INT32 | 20 |

Run the complete single-core campaign:

```bash
chmod +x run_single_core_0_7_all_kernels_1024.sh
M=1024 N=1024 K=1024 RUNS=6 bash run_single_core_0_7_all_kernels_1024.sh
```

Before timing a kernel, the runner validates it with the odd shape `15x7x13`. Results are written to `single_core_results_<M>/`; use `single_core_summary_latest.csv` for analysis and `raw_logs/` for detailed failures.

Kernel names encode data type, 8x4 tile shape, ZVL256 target, LMUL, and K-loop unroll factor. Native IME kernels remain under `../IME_NATIVE_KERNELS/`.