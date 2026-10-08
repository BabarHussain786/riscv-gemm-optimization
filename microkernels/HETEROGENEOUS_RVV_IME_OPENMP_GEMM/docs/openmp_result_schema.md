# OpenMP GEMM Result Schema

All generated files are stored under `results/`.

## Main Analysis Files

```text
results/k1_openmp_heterogeneous_raw_latest.csv
results/k1_openmp_heterogeneous_summary_latest.csv
```

Mixed policy-specific files use:

```text
results/openmp_raw_latest_k1-mixed-rvv-ime-static.csv
results/openmp_raw_latest_k1-mixed-rvv-ime-dynamic.csv
```

## Raw CSV

One row describes one kernel repetition.

| Column | Meaning |
|---|---|
| `mode` | Execution mode |
| `baseline`, `family`, `kernel` | Exact source and micro-kernel identity |
| `tile_shape` | Micro-kernel shape, such as `8x4` or `8x8` |
| `zvl`, `lmul`, `unroll` | Vector configuration encoded by the kernel |
| `kind` | `FP32_RVV`, `FP64_RVV`, `INT8_RVV`, `INT8_IME`, or `INT8_MIXED` |
| `core_group` | CPU set used by the mode |
| `requested_threads`, `actual_threads` | Requested and observed OpenMP workers |
| `M`, `N`, `K` | Matrix dimensions |
| `tile_N` | Width of one OpenMP output-column strip |
| `run`, `status` | Repetition number and final state |
| `time_sec` | Timed OpenMP tile-region duration |
| `metric_name`, `metric_value` | `GFLOPS` or `GOPS` and its value |
| `validation_method` | Independent output reference used for this row |
| `mismatch_count`, `max_error` | Numerical validation results |
| `worker_placement` | Worker ID, real CPU, path, and completed output strips |
| `static_tile_split` | Planned fixed IME/RVV split; `NA` outside mixed static mode |
| `schedule_policy` | `static` or `dynamic` |
| `schedule_chunk` | Dynamic chunk size; `0` for static mode |
| `output_strip_distribution` | Observed completed output strips, for example `IME:26;RVV:6` |
| `paired_rvv_kernel` | Canonical RVV kernel paired with the IME kernel in mixed mode; `NA` otherwise |
| `log_file` | Full build or run log |
| `perf_cycles`, `perf_instructions` | Optional Linux `perf` counts for the benchmark process |
| `perf_ipc` | Instructions divided by cycles |
| `perf_cache_references`, `perf_cache_misses` | Optional cache-event counts |
| `perf_cache_miss_rate` | `100 * cache_misses / cache_references` |
| `perf_log_file` | Raw `perf stat` CSV; `NA` when counters are disabled |

Focused paper-experiment CSV files add four leading columns:
`experiment`, `series`, `parameter_name`, and `parameter_value`. These identify
the scaling curve or tuning value while preserving every original runner
column.

## Summary CSV

Rows are grouped by kernel configuration, matrix size, scheduling policy, and chunk size. The observed output-strip distribution remains in the raw CSV because it can vary between dynamic repetitions. Statistics are:

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

Use `mean_metric` with `sample_std_metric` for the main performance figure. Use `median_metric` when reporting a robust central value.

## Live Log

Static example:

```text
SCHEDULING_POLICY=static
SCHEDULE_CHUNK=0
STATIC_TILE_SPLIT=IME:26;RVV:6
TILE_COUNT_UNIT=output_column_strips
TILE_DISTRIBUTION=IME:26;RVV:6
```

Dynamic example:

```text
SCHEDULING_POLICY=dynamic
SCHEDULE_CHUNK=1
DYNAMIC_TILE_RESULT=IME:...;RVV:...
TILE_COUNT_UNIT=output_column_strips
TILE_DISTRIBUTION=IME:...;RVV:...
```

The live log is for monitoring. Use the CSV files for plots and tables.
