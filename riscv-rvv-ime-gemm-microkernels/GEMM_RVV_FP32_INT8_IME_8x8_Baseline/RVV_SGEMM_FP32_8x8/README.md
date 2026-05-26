# FP32 SGEMM RVV 8x8 Kernel Pack

This package contains FP32 RVV SGEMM 8x8 kernel variants for benchmarking on RISC-V.

## Variant set
- Tile: `8x8`
- Family: `sgemm_kernel_8x8_zvl128b_*`
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
- `run_all.sh`
- `run_square_only.sh`
- `run_cluster_wise_8x8.sh`
- `run_single_core_8x8.sh`

## Quick usage
```bash
chmod +x run_all.sh run_square_only.sh run_cluster_wise_8x8.sh run_single_core_8x8.sh
./run_all.sh
```

## Notes
- Folder and kernel naming are aligned: each folder builds its matching `<folder_name>.c`.
- This directory is cleaned for source control (no benchmark binaries/log outputs included).