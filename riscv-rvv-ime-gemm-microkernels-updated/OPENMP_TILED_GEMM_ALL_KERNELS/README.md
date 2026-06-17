# OpenMP Tiled GEMM

This folder evaluates tiled GEMM execution on the heterogeneous RISC-V K1 core layout. The main mixed mode, `k1-mixed-rvv-ime`, uses all cores 0-7: cores 0-3 execute the IME-capable INT8 kernel path, while cores 4-7 execute the RVV fallback path. The same folder also keeps homogeneous RVV-only and IME-only modes for baseline comparison.

The micro-kernel tile and the OpenMP work tile are different levels of blocking:

```text
Complete GEMM:        C = A x B, for example M=N=K=1024
OpenMP work tile:     a column block of C, for example 1024x32 or 1024x64
Micro-kernel tile:    one existing 8x4 or 8x8 output block
```

Each OpenMP worker receives one or more column tiles of `C`. Since workers write disjoint output columns, the tiled parallel region avoids write races in `C`.

## Heterogeneous K1 Mode

`k1-mixed-rvv-ime` is the heterogeneous execution mode requested for the K1 platform. It compiles each IME wrapper together with its local `rvv_fallback.c` file. At runtime:

```text
cores 0-3: IME wrapper enters the native IME path
cores 4-7: IME wrapper falls back to the RVV INT8 kernel path
```

Both paths compute the same INT8 GEMM operation:

```text
INT8 x INT8 -> INT32 accumulation
C[column * ldc + row] += sum_k A[k * M + row] * B[k * N + column]
```

The mixed scheduler uses dynamic tile assignment. IME-side workers claim larger chunks, and RVV-side workers claim smaller chunks, so more output tiles are naturally assigned to the faster IME-capable cores. The default weights are:

```text
MIXED_IME_TILE_WEIGHT=4
MIXED_RVV_TILE_WEIGHT=1
```

For a balanced mixed run, use smaller `tile_N` values such as 16 or 32 so the scheduler has enough work tiles to distribute across both core groups.

Private IME/RVV B-tile buffers are allocated inside each OpenMP worker after the worker observes its CPU. This gives thread-local first touch for the temporary tile buffer. Full NUMA-aware placement of the global A, B, and C matrices should still be handled externally when required, for example with platform-specific memory policy tools.

## Input Contracts

RVV kernels receive their existing packed A and B panels.

Native IME `8x4` and `8x8` wrappers receive full K-major matrices:

```text
A[k * M + row]
B[k * N + column]
C[column * ldc + row]
```

For IME and mixed modes, the OpenMP benchmark extracts a contiguous K-major B matrix for each work tile before calling the wrapper. Each worker owns a reusable B-tile buffer.

## Correctness Check

By default, the first measured run for each kernel:

1. Runs one untimed warmup.
2. Produces an untimed serial same-kernel result.
3. Runs the timed OpenMP tiled execution.
4. Compares the complete tiled output with the serial result.

INT8-to-INT32 results require exact equality. FP32 and FP64 results use a small absolute-plus-relative tolerance. This validates OpenMP tiling and assembly of `C`; the separate accuracy checker remains responsible for comparison against independent numerical references.

Later performance repetitions keep the warmup but skip repeated serial validation. Set `VALIDATE_EACH_RUN=1` when every repetition must be validated.

## Timing Meaning

The reported timing scope is:

```text
timed_parallel_tile_region
```

It measures the OpenMP parallel region and the tiled kernel calls. It excludes input allocation, initialization, warmup, serial validation, and pre-run metadata allocation. For IME and mixed modes, it includes B-tile extraction and the wrapper's internal packing, native/fallback dispatch, execution, scattering, and cleanup. Therefore these measurements are end-to-end tiled-kernel timings, not compute-only IME timings.

## Supported Modes

| Mode | Core group | Default threads | Purpose |
|---|---:|---:|---|
| `k1-mixed-rvv-ime` | `0-7` | 8 | Heterogeneous K1 run: IME path on cores 0-3 and RVV fallback on cores 4-7 |
| `k1-rvv` / `k1-rvv-all` | `0-7` | 8 | RVV all-core baseline |
| `k1-rvv-only` | `4-7` | 4 | RVV-only baseline disjoint from IME cores |
| `k1-ime` | `0-3` | 4 | IME-only baseline on the IME-capable cluster |
| `k3-rvv` | `0-7` | 8 | K3 RVV run |
| `k3-ime` | `8-15` | 8 | Both K3 IME/A100 clusters |
| `k3-ime-cluster0` | `8-11` | 4 | First K3 IME/A100 cluster |
| `k3-ime-cluster1` | `12-15` | 4 | Second K3 IME/A100 cluster |
| `local` | unrestricted | 1 | Build and local inspection |

The output records the CPU observed by every OpenMP worker. In mixed mode, `WORKER_PLACEMENT` also labels each worker as `IME` or `RVV` and reports the number of tiles it processed.

## Run

From the project root, the main heterogeneous K1 run is:

```bash
MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
bash OPENMP_TILED_GEMM_ALL_KERNELS/run_all_openmp_kernels.sh \
  k1-mixed-rvv-ime 1024 1024 1024 32 6
```

Argument order:

```text
mode M N K tile_N runs
```

Useful controls:

```bash
OMP_NUM_THREADS=8       # override the mode default
OMP_PROC_BIND=close     # OpenMP binding policy
OMP_PLACES=cores        # OpenMP placement definition
OMP_DYNAMIC=false       # require the requested OpenMP team size
OMP_WARMUP=0            # disable the untimed warmup
OMP_VALIDATE=0          # disable serial same-kernel validation
VALIDATE_EACH_RUN=1     # validate every timed repetition
ENABLE_MF2=1            # explicitly include experimental native IME mf2
```

Native IME `lmulmf2` kernels are skipped by default because their hardware path is experimental. RVV fractional-LMUL kernels are unaffected.

## Output

Each run creates:

```text
openmp_results_<mode>_<M>_<timestamp>/
  openmp_raw_<mode>_<M>_runs<runs>_<timestamp>.csv
  openmp_live_<mode>_<M>_runs<runs>_<timestamp>.log
  raw_logs/
  build/
```

The CSV reports requested and actual thread counts, timing scope, validation method, kernel return code, failure stage, mismatch count, maximum error, worker placement, time, and GFLOPS/GOPS.

A run is not marked `OK` when the OpenMP runtime creates a different number of workers than requested.
## Clean Project Files

```text
OPENMP_TILED_GEMM_ALL_KERNELS/
  README.md
  omp_bench_template.c
  run_all_openmp_kernels.sh
  run_k1_all_datatypes_openmp_1024.sh
  check_openmp_builds.sh
  kernel_manifest.csv
  .gitignore
```

`omp_bench_template.c` is the shared OpenMP benchmark implementation. `run_all_openmp_kernels.sh` is the generic campaign runner. `run_k1_all_datatypes_openmp_1024.sh` runs the complete K1 campaign for FP64 RVV, FP32 RVV, INT8 RVV, IME-only, and mixed IME/RVV execution. `check_openmp_builds.sh` performs compile-only checks using `BUILD_ONLY=1`. `kernel_manifest.csv` lists the kernel sources, datatype path, tile shape, LMUL, unroll, and supported modes.

## Quick Checks

Compile the selected OpenMP kernels without running the timed benchmark:

```bash
bash OPENMP_TILED_GEMM_ALL_KERNELS/check_openmp_builds.sh
```

Run the complete K1 1024 campaign:

```bash
bash OPENMP_TILED_GEMM_ALL_KERNELS/run_k1_all_datatypes_openmp_1024.sh
```

For compile-only runs, `BUILD_ONLY=1` records `BUILD_OK` rows in the CSV and skips timed execution.
