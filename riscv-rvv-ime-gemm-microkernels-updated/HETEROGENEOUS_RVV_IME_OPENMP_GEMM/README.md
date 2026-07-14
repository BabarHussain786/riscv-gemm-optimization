# Heterogeneous RVV-IME OpenMP GEMM

This module compares homogeneous and heterogeneous GEMM execution on the K1. It keeps the existing RVV and IME micro-kernels unchanged and applies OpenMP only above the micro-kernel level.

## Execution Model

The output matrix `C` is divided into contiguous column tiles:

```text
tiles = ceil(N / tile_N)
```

For `N=1024` and `tile_N=32`, there are 32 OpenMP tiles. In mixed INT8 mode, the tiles are divided once using the static IME:RVV weights:

```text
IME tiles = round(tiles * IME_WEIGHT / (IME_WEIGHT + RVV_WEIGHT))
RVV tiles = tiles - IME tiles

Default 4:1 split: IME=26 tiles, RVV=6 tiles
```

OpenMP follows the K1 hierarchy:

```text
outer team: 2 cluster controllers
  cluster 0: 4 workers on cores 0-3, native IME path
  cluster 1: 4 workers on cores 4-7, explicit RVV path
```

Each cluster runs `parallel for schedule(static)` over its fixed tile range. There is no global tile queue, atomic tile claim, or dynamic work scheduler. The IME team calls the native IME wrapper with hardware execution required; the RVV team calls the matching low-level RVV widening kernel directly. Because the two ranges contain different columns of `C`, workers never update the same output values.

INT8 arithmetic is identical on both paths:

```text
C(i,j) = C(i,j) + sum_k A(i,k) * B(k,j)
INT8 x INT8 -> INT32 accumulation and output
```

FP32 and FP64 use RVV kernels on the selected cores. Native IME is used only for INT8-to-INT32 GEMM.

## Main Files

```text
src/openmp_heterogeneous_gemm.c   matrix setup, timing, validation, output
src/openmp_cluster_execution.h    nested 2-cluster OpenMP and static split
src/openmp_kernel_dispatch.h      one OpenMP tile -> selected micro-kernel
src/openmp_validation.h           serial same-kernel comparison

scripts/run_openmp_tiled_gemm_mode.sh
scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
scripts/check_openmp_tiled_gemm_builds.sh
```

## K1 Modes

| Mode | Cores | Execution |
|---|---:|---|
| `k1-rvv` | 0-7 | RVV all-core baseline |
| `k1-rvv-only` | 4-7 | RVV-cluster baseline |
| `k1-ime` | 0-3 | native IME-cluster baseline |
| `k1-mixed-rvv-ime` | 0-7 | nested IME and RVV cluster teams |

## Run

From this directory:

```bash
M=1024 N=1024 K=1024 TILE_N=32 RUNS=6 \
MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

Detached execution:

```bash
nohup bash -lc 'M=1024 N=1024 K=1024 TILE_N=32 RUNS=6 MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh' > results/k1_openmp_nohup_latest.log 2>&1 &
```

One mode only:

```bash
bash scripts/run_openmp_tiled_gemm_mode.sh k1-mixed-rvv-ime 1024 1024 1024 32 6
```

The runner rejects mixed execution unless exactly eight workers are available. It also fails when no buildable kernel source is found, so an empty campaign cannot be reported as successful.

## Results

The main analysis files are:

```text
results/k1_openmp_heterogeneous_raw_latest.csv
results/k1_openmp_heterogeneous_summary_latest.csv
```

Raw data records each run, matrix dimensions, tile width, timing, throughput, validation, worker placement, and the fixed IME/RVV tile split. Summary data reports mean, median, minimum, maximum, sample standard deviation, and timing for each kernel configuration.

Mixed logs also print the fixed cluster ownership:

```text
STATIC_TILE_SPLIT=IME:26;RVV:6
WORKER_PLACEMENT=0:cpu0:IME:...;...;7:cpu7:RVV:...
```

## Correctness and Timing

The first measured repetition compares the OpenMP result with an untimed serial execution of the same kernel. INT8 requires exact INT32 equality; FP32 and FP64 use absolute-plus-relative tolerance. Independent high-precision accuracy remains in the separate accuracy-validation module.

The timed interval contains nested OpenMP team creation, static tile loops, required input packing, kernel execution, and output updates. Matrix allocation, OpenMP worker-buffer allocation, warmup, validation, and memory cleanup are outside the timed interval. Any temporary allocation performed internally by a selected micro-kernel remains part of that kernel's measured time.

Thread affinity is checked at runtime for mixed K1 mode: workers 0-3 must execute on cores 0-3 and workers 4-7 on cores 4-7. This cluster-aware placement combines exact core affinity with contiguous output ownership. Each campaign also records the available system memory nodes from `/sys/devices/system/node/online`.

See `docs/k1_heterogeneous_openmp_methodology.md` for the complete experimental method.
