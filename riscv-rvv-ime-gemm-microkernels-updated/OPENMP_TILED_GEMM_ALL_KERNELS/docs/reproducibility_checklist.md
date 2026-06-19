# Reproducibility Checklist

Use this checklist before reporting OpenMP heterogeneous results.

1. Confirm the board core map matches the script assumptions:

```text
K1 IME-capable cores: 0-3
K1 RVV-only cores:    4-7
```

2. Confirm CPU affinity is available:

```bash
command -v taskset
```

3. Run the compile-only check:

```bash
bash OPENMP_TILED_GEMM_ALL_KERNELS/scripts/check_openmp_tiled_gemm_builds.sh
```

4. Run the K1 campaign:

```bash
cd OPENMP_TILED_GEMM_ALL_KERNELS
M=1024 N=1024 K=1024 RUNS=6 MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

5. Check completion:

```bash
grep -R "DONE" results/*/openmp_live_*.log results/k1_openmp_heterogeneous_live_latest.log
```

6. Use these files for analysis:

```text
results/k1_openmp_heterogeneous_raw_latest.csv
results/k1_openmp_heterogeneous_summary_latest.csv
```

7. Inspect `worker_placement` in the raw CSV for mixed-mode runs to verify that cores 0-3 executed the IME path and cores 4-7 executed the RVV fallback path.