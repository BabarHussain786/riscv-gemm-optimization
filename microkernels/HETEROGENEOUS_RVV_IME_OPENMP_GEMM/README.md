# Heterogeneous RVV-IME OpenMP GEMM

This module measures one INT8 x INT8 -> INT32 GEMM on the heterogeneous K1 processor. Cores 0-3 execute native IME kernels and cores 4-7 execute matching RVV kernels. The same kernels, matrices, validation, and timing rules are used for both scheduling policies.

## Project Roadmap

```text
Step 1 -> Create A[M x K], B[K x N], and C[M x N].
Step 2 -> Divide C into non-overlapping column tiles of width tile_N.
Step 3 -> Select static or dynamic OpenMP tile scheduling.
Step 4 -> Pin IME workers to cores 0-3 and RVV workers to cores 4-7.
Step 5 -> Call the selected 8x4 or 8x8 IME/RVV micro-kernel.
Step 6 -> Check core placement, completed strips, and numerical output.
Step 7 -> Save time, GOPS/GFLOPS, and statistics in CSV files.
```

The OpenMP output-strip count is:

```text
output_strips = ceil(N / tile_N)
```

For `N=1024` and `tile_N=32`, `C` is divided into 32 independent column strips. This count is not the number of 8x4, 8x8, or native IME micro-tiles. Each strip has one owner, so workers never write the same output values.

## Scheduling Policies

| Policy | OpenMP structure | Tile assignment | Purpose |
|---|---|---|---|
| `static` | two outer cluster threads; four workers per cluster | fixed IME and RVV ranges | low-overhead reference for regular GEMM |
| `dynamic` | one team of eight pinned workers | `schedule(dynamic, chunk)` | measure load balancing and scheduling overhead |

The static IME share is computed once:

```text
T_IME = round(T * W_IME / (W_IME + W_RVV))
T_RVV = T - T_IME
```

The default `4:1` weights give 26 IME tiles and 6 RVV tiles when `T=32`. Dynamic mode has no fixed ratio: a free worker receives the next OpenMP chunk, so the final IME/RVV counts are measured after execution.

Both paths compute the same equation:

```text
C(i,j) = C(i,j) + sum_k A(i,k) * B(k,j)
```

## Main Files

```text
src/openmp_heterogeneous_gemm.c   setup, policy selection, timing, and output
src/openmp_cluster_execution.h    static two-cluster OpenMP execution
src/openmp_dynamic_execution.h    dynamic eight-worker OpenMP execution
src/openmp_kernel_dispatch.h      IME/RVV micro-kernel dispatch
src/openmp_validation.h           independent reference and output comparison

scripts/run_openmp_tiled_gemm_mode.sh
scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
scripts/run_k1_strong_scaling.sh
scripts/run_k1_weak_scaling.sh
scripts/run_k1_partitioning_analysis.sh
scripts/run_k1_kernel_tuning.sh
scripts/check_openmp_tiled_gemm_builds.sh
analysis/plot_k1_paper_experiments.py
```

## K1 Modes

| Mode | Cores | Execution |
|---|---:|---|
| `k1-rvv` | 0-7 | all-core RVV baseline |
| `k1-rvv-only` | 4-7 | RVV-cluster baseline |
| `k1-ime` | 0-3 | native IME-cluster baseline |
| `k1-mixed-rvv-ime` | 0-7 | heterogeneous static or dynamic execution |

## Run the Complete Campaign

The campaign runs homogeneous baselines, mixed static scheduling, and mixed dynamic scheduling:

```bash
M=1024 N=1024 K=1024 TILE_N=32 RUNS=6 \
MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
DYNAMIC_CHUNK=1 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

Detached execution:

```bash
nohup bash -lc 'M=1024 N=1024 K=1024 TILE_N=32 RUNS=6 MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 DYNAMIC_CHUNK=1 bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh' > results/k1_openmp_nohup_latest.log 2>&1 &
```

Run only mixed static scheduling:

```bash
GEMM_TILE_SCHEDULE=static \
bash scripts/run_openmp_tiled_gemm_mode.sh k1-mixed-rvv-ime 1024 1024 1024 32 6
```

Run only mixed dynamic scheduling:

```bash
GEMM_TILE_SCHEDULE=dynamic GEMM_DYNAMIC_CHUNK=1 \
bash scripts/run_openmp_tiled_gemm_mode.sh k1-mixed-rvv-ime 1024 1024 1024 32 6
```

## K1 Paper Experiments

The remaining K1 evaluations are separate so one interrupted study does not
overwrite another study. Each script uses the validated INT8-to-INT32 paths
and publishes timestamped results plus stable `latest` files.

Strong scaling keeps `1024x1024x1024` fixed. It measures the RVV and native IME
8x4 INT8 kernels separately with 1, 2, 4, and 8 RVV workers or 1, 2, and 4
IME workers. The default K1 Plot 1 run repeats this comparison for
`LMUL=1` and unroll factors `1,2,4,8`. Static and dynamic heterogeneous
policies are separate experiments and are excluded by default. Linux `perf`
records cycles, instructions, IPC, cache references, and cache misses:

```bash
bash scripts/run_k1_strong_scaling.sh
```

The terminal prints a compact row-and-column summary. Detailed output for
every kernel and run remains in the timestamped experiment log and CSV files.
Set `INCLUDE_HETEROGENEOUS=1` only when heterogeneous rows are also required.
Set `COLLECT_PERF=0` only when hardware-counter access is unavailable.

Weak scaling starts from `512x512x512` and scales each square dimension by
the cube root of the worker count, rounded to a multiple of eight:

```bash
bash scripts/run_k1_weak_scaling.sh
```

Static/dynamic partitioning compares fixed IME:RVV ratios and dynamic chunk
sizes using one matrix, kernel pair, and output-strip width:

```bash
bash scripts/run_k1_partitioning_analysis.sh
```

Kernel tuning sweeps `tile_N=8,16,32,64,128`. The existing inventory supplies
the 8x4/8x8, LMUL, and unroll dimensions:

```bash
bash scripts/run_k1_kernel_tuning.sh
```

After these campaigns and the accuracy campaign finish, create every
available K1 figure from measured CSV data:

```bash
python3 -m pip install -r analysis/requirements.txt
python3 analysis/plot_k1_paper_experiments.py
```

Figures are saved as PNG and PDF under `analysis/figures/`. Missing inputs are
reported and synthetic values are never substituted.

## Results

Use these combined files for analysis:

```text
results/k1_openmp_heterogeneous_raw_latest.csv
results/k1_openmp_heterogeneous_summary_latest.csv
```

Policy-specific aliases are also created:

```text
results/openmp_raw_latest_k1-mixed-rvv-ime-static.csv
results/openmp_raw_latest_k1-mixed-rvv-ime-dynamic.csv
```

Each row records the scheduling policy, dynamic chunk, observed IME/RVV output-strip distribution, worker placement, validation, time, and throughput. Summary rows report mean, median, minimum, maximum, and sample standard deviation.

Focused paper experiments publish these analysis files:

```text
results/paper_experiments/k1_strong_scaling_raw_latest.csv
results/paper_experiments/k1_weak_scaling_raw_latest.csv
results/paper_experiments/k1_partitioning_raw_latest.csv
results/paper_experiments/k1_kernel_tuning_raw_latest.csv
```

All modes start from the same unpacked contract: `A[k*M+i]`, `B[k*N+j]`, and `C[j*M+i]`. Required RVV/IME packing is part of the timed tile region. In mixed mode, the RVV workers link the exact canonical source used by the pure-RVV campaign; the IME folder's separate fallback implementation is not used. Board runs pin every worker to one exact CPU and reject a placement mismatch. The first run is a mandatory correctness gate: INT8 uses independent INT64 accumulation followed by exact INT32 comparison, while FP32/FP64 use an independent higher-precision accumulation and tolerance. If validation fails, later repetitions for that kernel are not recorded.

The campaign excludes configurations whose names do not match a valid VLEN=256 execution path: FP32 8x8 `LMUL=mf2`, FP64 8x4 `LMUL=1`, INT8 `LMUL=4/8`, and duplicate INT8 sources outside the canonical `GEMM_RVV_FP32_INT8_*` trees.

## Valid Run Conditions

```text
workers:          exactly 8 in mixed mode
IME placement:    workers 0-3 on cores 0-3
RVV placement:    workers 4-7 on cores 4-7
strip total:      IME strips + RVV strips = all output strips
correctness:      mismatch_count = 0
```

See `docs/k1_heterogeneous_openmp_methodology.md` for the experimental method and `docs/openmp_result_schema.md` for every output field.
