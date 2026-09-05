# Reproducibility Checklist

Use these checks before reporting K1 OpenMP results.

## 1. Hardware Map

```text
IME-capable cores: 0-3
RVV-only cores:    4-7
```

Confirm affinity support:

```bash
command -v taskset
```

## 2. Build Check

```bash
bash scripts/check_openmp_tiled_gemm_builds.sh
```

## 3. Complete Campaign

```bash
M=1024 N=1024 K=1024 TILE_N=32 RUNS=6 \
MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
DYNAMIC_CHUNK=1 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

Set `RUN_DYNAMIC=0` only when a static-only campaign is intentionally required.

## 4. Completion

```bash
grep -R "DONE status=" results/*/openmp_live_*.log \
  results/k1_openmp_heterogeneous_live_latest.log
```

Accept only `DONE status=OK`.

The live log must also record `COMPILER`, `BUILD_FLAGS`, `PRIMARY_SOURCE`, and,
for mixed mode, `MIXED_RVV_SOURCE`.

## 5. Placement, Path, and Output Checks

```text
actual_threads = 8
workers 0-3 -> CPUs 0-3 before and after execution -> IME
workers 4-7 -> CPUs 4-7 before and after execution -> RVV
IME completed strips + RVV completed strips = ceil(N / tile_N)
mismatch_count = 0
reference_overflow_count = 0
```

The printed count unit must be `TILE_COUNT_UNIT=output_column_strips`. Confirm that native modes print `IME_NATIVE_REQUIRED` or `IME_NATIVE_PLUS_RVV_EXPLICIT`; a native IME result must never be accepted as an automatic RVV fallback.

Static mode must also satisfy:

```text
T_IME = round(T * W_IME / (W_IME + W_RVV))
T_RVV = T - T_IME
```

Dynamic mode may produce a different IME/RVV split on each repetition. Record `schedule_chunk` and the observed `output_strip_distribution` for every run.

## 6. Analysis Files

```text
results/k1_openmp_heterogeneous_raw_latest.csv
results/k1_openmp_heterogeneous_summary_latest.csv
```
