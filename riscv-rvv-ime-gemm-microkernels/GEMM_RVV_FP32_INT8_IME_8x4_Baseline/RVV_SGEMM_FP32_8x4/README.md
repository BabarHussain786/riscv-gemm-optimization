# FP32 SGEMM RVV 8x4 Kernel Pack

This package contains FP32 RVV SGEMM 8x4 kernel variants for benchmarking on RISC-V.

## Variant set
- Tile: `8x4`
- Family: `sgemm_kernel_8x4_zvl256b_*`
- LMUL labels: `lmul1`, `lmul2`, `lmul4`, `lmul8`, `lmulmf2`
- Unroll factors: `unroll1`, `unroll2`, `unroll4`, `unroll8`

Total variants: `20`

## Per-variant folder contents
Each kernel folder includes:
- `<kernel_name>.c`
- `sgemm_bench.c`
- `Makefile`
- `run.sh`

## Top-level helper scripts
- `quick_check.sh`
- `run_all_benchmarks.sh`
- `run_cluster_wise_sgemm.sh`
- `run_single_core_sgemm.sh`

## Quick usage
```bash
chmod +x quick_check.sh run_all_benchmarks.sh run_cluster_wise_sgemm.sh run_single_core_sgemm.sh
./quick_check.sh
./run_all_benchmarks.sh
```

## Notes
- Folder and kernel naming are aligned: each folder builds its matching `<folder_name>.c`.
- This directory is cleaned for source control (no benchmark binaries/log outputs included).