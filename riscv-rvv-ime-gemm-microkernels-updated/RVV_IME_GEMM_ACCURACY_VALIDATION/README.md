# INT8 IME vs RVV Accuracy Checker

This folder keeps one accuracy method for the paper: exact INT8-to-INT32 reference comparison between the native IME path and the RVV fallback path.

## What It Checks

For each tested kernel, the checker computes:

```text
C_kernel = A_int8 x B_int8 -> C_int32
C_ref    = same multiplication with independent INT64 accumulation
error    = C_kernel - C_ref for every output element
```

A run is correct when every INT32 output element matches the reference and no reference value overflows INT32.

## Kept Method

```bash
bash ACCURACY_CHECKER/run_int8_ime_vs_rvv_accuracy.sh k1 1
```

Default settings:

```text
M = 1024, N = 1024, K = 1024
INPUT_CLASSES = full_range_uniform
ENABLE_MF2 = 1
```

Useful options:

```bash
INPUT_CLASSES="full_range_uniform" bash ACCURACY_CHECKER/run_int8_ime_vs_rvv_accuracy.sh k1 1
INPUT_CLASSES="bounded_uniform full_range_uniform mixed_magnitude cancellation_stress" bash ACCURACY_CHECKER/run_int8_ime_vs_rvv_accuracy.sh k1 1
```

## Files

| File | Role |
| --- | --- |
| `run_int8_ime_vs_rvv_accuracy.sh` | Main clean campaign script. |
| `run_int8_ime_vs_rvv_accuracy_once.sh` | Internal runner for one input class. |
| `ime_rvv_fallback_diff_histogram_check.c` | C checker that computes the reference and error histograms. |

## Outputs

The main output is:

```text
int8_ime_vs_rvv_accuracy_summary_<mode>_latest.csv
```

Each campaign also stores per-input summaries, raw logs, and diff/input histograms in a timestamped result folder.