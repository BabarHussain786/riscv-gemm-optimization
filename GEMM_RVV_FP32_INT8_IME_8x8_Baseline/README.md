# GEMM RVV FP32/INT8/IME 8x8 Baseline

This folder contains active FP32 SGEMM, INT8 iGEMM, and IME INT8 8x8 kernels.

## Active Folders

- RVV_SGEMM_FP32_8x8/
- RVV_IGEMM_INT8_I8I32_8x8/
- IME_GEMM_INT8_I8I32_8x8_RVV_Fallback/
- extra/ for archived or generated material not needed for normal benchmarking

Each IME kernel subfolder is self-contained and includes:

  ime_kernel_*.c
  rvv_fallback.c
  ime_bench.c
  Makefile

## Per-Baseline Runner

  bash run_single_core_0_7_all_kernels_1024.sh

Default settings:

  M=N=K=1024
  RUNS=6
  CORES=0 1 2 3 4 5 6 7

This script is also called by the root K1 runner.
