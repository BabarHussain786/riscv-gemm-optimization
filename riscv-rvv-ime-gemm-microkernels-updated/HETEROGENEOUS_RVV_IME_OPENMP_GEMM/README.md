# Heterogeneous OpenMP Tiled GEMM Benchmark

This module evaluates tiled GEMM execution on the K1 heterogeneous RISC-V platform. It compares homogeneous RVV execution, IME-cluster execution, and mixed RVV-IME execution under the same matrix size, tile size, and repetition protocol.

The mixed K1 mode, `k1-mixed-rvv-ime`, uses all cores 0-7. Cores 0-3 execute the native IME path, while cores 4-7 execute the RVV fallback path. OpenMP splits the output matrix `C` into column tiles; each worker computes disjoint output columns, so no two workers write the same region of `C`.

## GEMM Decomposition

```text
Full operation:       C = A x B, for example M=N=K=1024
OpenMP work tile:     one column block of C, for example 1024x32 or 1024x64
Micro-kernel tile:    one existing 8x4 or 8x8 kernel update
INT8 arithmetic:      INT8 x INT8 -> INT32 accumulation/output
```

The OpenMP tile size is controlled by `tile_N`. Smaller values, such as 16 or 32, give the scheduler more tiles to distribute across heterogeneous cores. In mixed mode, the default tile weights are:

```text
MIXED_IME_TILE_WEIGHT=4
MIXED_RVV_TILE_WEIGHT=1
```

This gives IME-side workers larger tile claims while RVV-side workers still participate through the fallback path. Temporary B-tile buffers are allocated inside each OpenMP worker after the worker observes its CPU, giving thread-local first touch for these buffers. Platform-specific NUMA placement for global A, B, and C can be controlled externally when required.

## Directory Layout

```text
OPENMP_TILED_GEMM_ALL_KERNELS/
  README.md
  .gitignore
  docs/
    k1_heterogeneous_openmp_methodology.md
    openmp_result_schema.md
    reproducibility_checklist.md
  src/
    openmp_tiled_gemm_benchmark.c    compact main driver
    openmp_tiled_gemm_config.h       datatype, metric, and kernel selection
    openmp_tiled_gemm_utils.h        timing, allocation, size checks
    openmp_tiled_gemm_data.h         deterministic A/B/C initialization
    openmp_tiled_gemm_dispatch.h     full/tiled micro-kernel calls
    openmp_tiled_gemm_validation.h   output comparison
    openmp_tiled_gemm_parallel.h     OpenMP tile scheduling and worker placement
  scripts/
    run_openmp_tiled_gemm_mode.sh
    run_k1_heterogeneous_openmp_gemm_1024.sh
    check_openmp_tiled_gemm_builds.sh
  metadata/
    openmp_tiled_gemm_kernel_manifest.csv
  results/
    generated logs and CSV files, ignored by git
```

Deeper documentation:

```text
docs/k1_heterogeneous_openmp_methodology.md  method, tile split, core mapping, NUMA notes
docs/openmp_result_schema.md                raw CSV, summary CSV, and live-log meanings
docs/reproducibility_checklist.md            board-side checks before reporting results
```

## Execution Modes

| Mode | Core group | Purpose |
|---|---:|---|
| `k1-rvv` / `k1-rvv-all` | 0-7 | All-core RVV baseline |
| `k1-rvv-only` | 4-7 | RVV-cluster baseline |
| `k1-ime` | 0-3 | IME-cluster baseline |
| `k1-mixed-rvv-ime` | 0-7 | Heterogeneous execution: IME on cores 0-3 and RVV fallback on cores 4-7 |
| `k3-rvv` | 0-7 | K3 RVV run |
| `k3-ime` | 8-15 | K3 IME run |
| `k3-ime-cluster0` | 8-11 | First K3 IME cluster |
| `k3-ime-cluster1` | 12-15 | Second K3 IME cluster |
| `local` | unrestricted | Local build or inspection run |

## Final kernel filter

The final K1 OpenMP campaign uses ZVL_FILTER=128b by default. This keeps the submitted benchmark focused on the cleaned 128b kernel set. Use ZVL_FILTER=all only for exploratory local testing. Generated files go to `results/`; downloaded local result copies should stay outside the source tree.

## Run the K1 Comparison Campaign

From the repository root:

```bash
cd OPENMP_TILED_GEMM_ALL_KERNELS
MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 \
bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh
```

Detached run with `nohup`:

```bash
cd OPENMP_TILED_GEMM_ALL_KERNELS
nohup bash -lc 'M=1024 N=1024 K=1024 RUNS=6 MIXED_IME_TILE_WEIGHT=4 MIXED_RVV_TILE_WEIGHT=1 bash scripts/run_k1_heterogeneous_openmp_gemm_1024.sh' > results/k1_openmp_nohup_latest.log 2>&1 &
```

Run one mode only:

```bash
bash scripts/run_openmp_tiled_gemm_mode.sh k1-mixed-rvv-ime 1024 1024 1024 32 6
```

Argument order:

```text
mode M N K tile_N runs
```

## Output Format

The live log follows the same style as the single-core K1 benchmark:

```text
KERNEL: ime_kernel_8x4_zvl128b_lmul1_unroll4
  MODE: K1 heterogeneous OpenMP RVV-IME execution
  FAMILY: RVV_IME_IGEMM_INT8_I8I32_8x4
  PATH: INT8 heterogeneous IME/RVV-fallback GEMM (INT8 x INT8 -> INT32)
  TILE: 8x4 ZVL=128b LMUL=1 UNROLL=4
  WORK: M=1024 N=1024 K=1024 tile_N=32 runs=6
  CORES: cores 0-3 execute IME path; cores 4-7 execute RVV fallback path
    run 1: OK GOPS=... time=... threads=8 validation=1
```

Each mode writes raw and summary CSV files under `results/`. The K1 campaign also writes combined latest files:

```text
results/k1_openmp_heterogeneous_live_latest.log
results/k1_openmp_heterogeneous_raw_latest.csv
results/k1_openmp_heterogeneous_summary_latest.csv
```

The summary CSV reports mean, median, min, max, standard deviation, and mean time for each kernel configuration. The full raw and summary schema is documented in `docs/openmp_result_schema.md`.

## Correctness and Timing

For each kernel, the first measured repetition performs an untimed serial same-kernel validation before timing the OpenMP tiled execution. INT8 outputs require exact INT32 equality. FP32 and FP64 use a small absolute-plus-relative tolerance. Later repetitions skip repeated validation unless `VALIDATE_EACH_RUN=1` is set.

The reported timing scope is `timed_parallel_tile_region`. It includes the OpenMP tiled region and kernel calls. For IME and mixed modes it also includes B-tile extraction, wrapper packing, native/fallback dispatch, execution, scattering, and cleanup. The complete methodology is documented in `docs/k1_heterogeneous_openmp_methodology.md`.

## Build Check

For a board-side reporting checklist, use `docs/reproducibility_checklist.md`.

```bash
bash OPENMP_TILED_GEMM_ALL_KERNELS/scripts/check_openmp_tiled_gemm_builds.sh
```

Useful controls:

```bash
OMP_NUM_THREADS=8       # override mode default
OMP_PROC_BIND=close     # OpenMP binding policy
OMP_PLACES=cores        # OpenMP placement definition
OMP_DYNAMIC=false       # require requested team size
OMP_VALIDATE=0          # disable serial same-kernel validation
VALIDATE_EACH_RUN=1     # validate every timed repetition
ENABLE_MF2=1            # include experimental native IME mf2 path
```
