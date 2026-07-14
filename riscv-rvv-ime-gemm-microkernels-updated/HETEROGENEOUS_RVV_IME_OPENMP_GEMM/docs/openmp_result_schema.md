# OpenMP Tiled GEMM Output Schema

The benchmark writes both raw per-run data and summary statistics. All generated files are placed under `results/`.

## Raw CSV

Per-mode raw files use this pattern:

```text
results/openmp_results_<mode>_<M>_<timestamp>/openmp_raw_<mode>_<M>_runs<runs>_<timestamp>.csv
results/openmp_raw_latest_<mode>.csv
```

The K1 campaign also creates:

```text
results/k1_openmp_heterogeneous_raw_latest.csv
```

Important raw columns:

| Column | Meaning |
|---|---|
| `mode` | Execution mode, such as `k1-rvv-only` or `k1-mixed-rvv-ime` |
| `baseline` | Kernel baseline folder used as source context |
| `family` | Kernel family folder, such as FP64 RVV, INT8 RVV, or IME wrapper family |
| `kernel` | Exact micro-kernel variant name |
| `tile_shape` | Micro-kernel tile shape, for example `8x4` or `8x8` |
| `zvl` | Vector-length profile encoded in the kernel name |
| `lmul` | LMUL value encoded in the kernel name |
| `unroll` | Unroll factor encoded in the kernel name |
| `kind` | Datatype/execution path: `FP32_RVV`, `FP64_RVV`, `INT8_RVV`, `INT8_IME`, or `INT8_MIXED` |
| `core_group` | CPU set used by the mode |
| `requested_threads` | OpenMP thread count requested by the script |
| `actual_threads` | OpenMP team size observed at runtime |
| `M,N,K` | Matrix dimensions |
| `tile_N` | OpenMP column-tile width |
| `run` | Repetition number |
| `status` | `OK`, `BUILD_FAILED`, `RUN_FAILED`, `KERNEL_RETURN`, `NUMERICAL_FAILED`, or `THREAD_COUNT_MISMATCH` |
| `time_sec` | Timed OpenMP tiled region duration in seconds |
| `metric_name` | `GFLOPS` for FP32/FP64, `GOPS` for INT8/IME |
| `metric_value` | Throughput value for the timed run |
| `validation_method` | Same-kernel validation method used for this run |
| `mismatch_count` | Number of output mismatches against the serial same-kernel check |
| `max_error` | Maximum output error observed during validation |
| `worker_placement` | Per-worker CPU/path/tile information |
| `static_tile_split` | Fixed mixed ownership, for example `IME:26;RVV:6`; `NA` for homogeneous modes |
| `log_file` | Raw log for the compiled executable or timed run |

## Summary CSV

Per-mode summary files use this pattern:

```text
results/openmp_summary_latest_<mode>.csv
```

The combined K1 summary is:

```text
results/k1_openmp_heterogeneous_summary_latest.csv
```

Summary rows are grouped by mode, kernel identity, datatype path, core group, requested thread count, `M`, `N`, `K`, `tile_N`, metric, and `static_tile_split`. The summary reports:

```text
ok_runs
mean_metric
median_metric
min_metric
max_metric
sample_std_metric
mean_time_sec
min_time_sec
max_time_sec
failed_runs
build_failed_runs
```

For plotting, use `mean_metric` with `sample_std_metric` as the primary throughput summary. Use `median_metric` when the run distribution is skewed or when a robust central value is preferred.

## Live Log

The live log is intended for terminal monitoring and reviewer readability. It is not the primary analysis file. Use the raw and summary CSV files for figures and tables.

Example live output:

```text
KERNEL: ime_kernel_8x4_zvl256b_lmul1_unroll4
  MODE: K1 heterogeneous OpenMP RVV-IME execution
  PATH: INT8 heterogeneous native-IME/RVV GEMM (INT8 x INT8 -> INT32)
  STATIC_TILE_SPLIT=IME:26;RVV:6
    run 1: OK GOPS=... time=... threads=8 validation=1
      workers: 0:cpu0:IME:tiles7;...;4:cpu4:RVV:tiles2
```
