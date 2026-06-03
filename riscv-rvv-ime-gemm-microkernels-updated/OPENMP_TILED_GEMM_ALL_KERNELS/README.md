# OpenMP Tiled GEMM Extension

This folder adds the first tiled multi-thread extension for the project. It is
used to validate that the GEMM output matrix can be partitioned into independent
tiles and computed safely by multiple threads on one core cluster.

It does not modify the microkernel folders. The script discovers every existing
kernel directory, builds an OpenMP tiled benchmark for that kernel, runs the
selected campaign, and writes CSV results.

## Role in the Project

The purpose of this folder is the single-cluster validation step before a full
heterogeneous RVV+IME implementation. The GEMM output matrix `C` is split into
column tiles:

```text
C = [ C0 | C1 | C2 | ... ]
```

Each tile is independent because it writes to a different part of `C`. The same
kernel computes each tile using the corresponding packed panel of `B`. This
checks that tiled execution, thread scheduling, and kernel calls work correctly
before distributing tiles across different clusters.

The preserved native IME wrappers keep their original input contracts. The
OpenMP launcher supplies full K-major matrices to `8x4` IME wrappers and
prepacked row-oriented panels to `8x8` IME wrappers.

The next extension is a heterogeneous RVV+IME scheduler. In that version, RVV
cores compute some output tiles and IME cores compute other output tiles. The
tile ratio should be chosen from measured throughput, assigning more work to the
faster cluster.

```text
Single-cluster step:
  C tiles -> RVV cores only, or IME cores only

Future heterogeneous step:
  C tiles -> RVV cores 0-7 and IME cores 8-15
```

## Files

```text
omp_bench_template.c
run_all_openmp_kernels.sh
README.md
```

## Supported Campaigns

```text
k1-rvv   RVV FP32, FP64, and INT8 kernels on cores 0-7
k1-ime   IME INT8 kernels on cores 0-3
k3-rvv   RVV FP32, FP64, and INT8 kernels on cores 0-7
k3-ime   IME INT8 kernels on cores 8-15
local    all discovered kernels without forcing a core group
```

## Run Commands

From the project root:

```bash
cd OPENMP_TILED_GEMM_ALL_KERNELS
```

K1 RVV cluster:

```bash
bash run_all_openmp_kernels.sh k1-rvv 1024 1024 1024 64 6
```

K1 IME cluster:

```bash
bash run_all_openmp_kernels.sh k1-ime 1024 1024 1024 64 6
```

K3 RVV cluster:

```bash
bash run_all_openmp_kernels.sh k3-rvv 1024 1024 1024 64 6
```

K3 IME cluster:

```bash
bash run_all_openmp_kernels.sh k3-ime 1024 1024 1024 64 6
```

Argument order:

```text
mode M N K tile_N runs
```

Default values:

```text
M=1024
N=1024
K=1024
tile_N=64
runs=6
OMP_NUM_THREADS=8
```

## Output

Each run creates a timestamped result folder:

```text
openmp_results_<mode>_<M>_<timestamp>/
  openmp_raw_<mode>_<M>_runs<runs>_<timestamp>.csv
  openmp_live_<mode>_<M>_runs<runs>_<timestamp>.log
  raw_logs/
  build/
```

The script also writes latest convenience files:

```text
openmp_raw_latest_<mode>.csv
openmp_live_latest_<mode>.log
```

## What It Measures

The OpenMP layer splits the output matrix by column tiles. Each thread calls the
same existing microkernel on a different column range of `C`, so output columns
are independent across threads.

This folder measures single-cluster tiled throughput. It does not yet split one
GEMM execution between RVV and IME clusters in the same run.

The CSV includes:

```text
baseline
family
kernel
tile shape
ZVL
LMUL
unroll
kind
core group
thread count
time
GFLOPS or GOPS
status
```

Use this folder for the multi-thread extension of the project. Keep the original
single-core benchmarking scripts for LMUL/unroll baseline analysis.
