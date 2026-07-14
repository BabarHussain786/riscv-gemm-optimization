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
bash HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/check_openmp_tiled_gemm_builds.sh
```

4. Confirm that nested OpenMP is enabled:

```bash
export OMP_MAX_ACTIVE_LEVELS=2
```

5. Record the available system memory nodes:

```bash
cat /sys/devices/system/node/online
```

The K1 campaign uses cluster-aware placement through exact core affinity and
separate output ownership.

6. Run the K1 campaign with one common tile width:

```bash
cd HETEROGENEOUS_RVV_IME_OPENMP_GEMM
M=1024 N=1024 K=1024 TILE_N=32 RUNS=6 \
MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

7. Check completion. Accept only `status=OK`:

```bash
grep -R "DONE status=" results/*/openmp_live_*.log results/k1_openmp_heterogeneous_live_latest.log
```

8. Use these files for analysis:

```text
results/k1_openmp_heterogeneous_raw_latest.csv
results/k1_openmp_heterogeneous_summary_latest.csv
```

9. Inspect the mixed log and raw CSV:

```text
STATIC_TILE_SPLIT must equal the sum of completed worker tiles.
Workers 0-3 must run on cores 0-3 through IME.
Workers 4-7 must run on cores 4-7 through the explicit RVV kernel call.
MISMATCH_COUNT must be 0.
```
