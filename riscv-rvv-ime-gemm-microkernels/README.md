# GEMM RVV FP32/FP64 INT8 IME Baselines

This repository contains RISC-V GEMM micro-kernel baselines for FP32 SGEMM, FP64 DGEMM, INT8 iGEMM with int32 accumulation, and SpacemiT IME INT8 GEMM with RVV fallback.

The project is organized by datatype, tile shape, backend, LMUL setting, and unroll factor. Each IME kernel folder is self-contained: it includes its IME source, benchmark source, Makefile, and a local rvv_fallback.c source file.

## Repository Layout

  GEMM_RVV_FP32_INT8_IME_8x4_Baseline
  GEMM_RVV_FP32_INT8_IME_8x8_Baseline
  GEMM_RVV_FP64_INT8_IME_8x4_Baseline
  GEMM_RVV_FP64_INT8_IME_8x8_Baseline
  run_k1_single_core_0_7_1024.sh
  run_k3_rvv_ime_0_15_1024.sh

Each baseline folder contains the active RVV/IME kernel trees plus run_single_core_0_7_all_kernels_1024.sh.

## Run On K1

  cd ~/riscv-rvv-ime-gemm-microkernels
  bash run_k1_single_core_0_7_1024.sh

This runs the four baseline groups through their per-baseline run_single_core_0_7_all_kernels_1024.sh scripts on cores 0-7.

## Run On K3

  cd ~/riscv-rvv-ime-gemm-microkernels
  bash run_k3_rvv_ime_0_15_1024.sh

Default K3 core campaigns:

  RVV cores: 0-7
  IME cores: 8-15

For K3, the IME campaign uses /proc/set_ai_thread when available before pinning runs to cores 8-15. This is needed because the A100 IME cores are exposed through a separate CPU domain on some Bianbu setups.

If /proc/set_ai_thread is missing or not writable, the IME campaign may still fail for cores 8-15 with an affinity error.

## Optional Parameters

  M=1024 N=1024 K=1024 RUNS=6 bash run_k1_single_core_0_7_1024.sh
  M=1024 N=1024 K=1024 RUNS=6 RVV_CORES="0 1 2 3 4 5 6 7" IME_CORES="8 9 10 11 12 13 14 15" bash run_k3_rvv_ime_0_15_1024.sh

## Requirements

- RISC-V Linux target with RVV support.
- SpacemiT IME-capable cores for IME kernels.
- Bash, Make, and a C compiler configured for the target system.
- taskset is optional; scripts use it when available.

## License

University of Salerno research notice. See LICENSE.
