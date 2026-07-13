# RVV-IME INT8 GEMM Accuracy Validation

## Project Goal

This checker answers one question:

> Do the IME-native and RVV-fallback kernels produce exactly the same
> `INT8 x INT8 -> INT32` matrix result?

Both paths receive the same input matrices and seed. Each path is checked
against one independently calculated trusted answer.

## Mathematical Check

For every output position `(i,j)`:

```text
trusted(i,j) = sum over k of INT64(A(i,k)) x INT64(B(k,j))
kernel(i,j)  = INT32 answer produced by IME or RVV
difference   = absolute(kernel(i,j) - trusted(i,j))
```

A run passes only when:

```text
largest difference = 0
wrong output values = 0
exact-match rate    = 100%
INT32 overflow      = 0
```

For the default `K=1024` full-range INT8 test, the conservative bound is:

```text
1024 x 128 x 128 = 16,777,216 < 2,147,483,647 (INT32 maximum)
```

Therefore, the trusted result should fit safely in INT32.

## Relation to the Ozaki Paper

The validation method adopts these ideas from the Ozaki accuracy analysis:

- execute low-precision matrix multiplication;
- compare its output with a trusted higher-width calculation;
- report the largest numerical difference and the complete error distribution.

This checker does **not** implement Ozaki matrix slicing, Chinese Remainder
Theorem reconstruction, or floating-point emulation. It adapts the paper's
reference-comparison principle to compare IME and RVV integer kernels.

Reference: K. Ozaki, Y. Uchino, and T. Imamura, *Ozaki Scheme II: A
GEMM-oriented emulation of floating-point matrix multiplication using an
integer modular technique*, arXiv:2504.08009.

## Default Scientific Scope

```text
M = 1024, N = 1024, K = 1024
INPUT_CLASSES = full_range_uniform
ENABLE_MF2 = 0
STRICT = 1
```

`LMUL=mf2` native IME kernels are experimental and are disabled by default.
The experimental `4x4` kernel is not included in this `8x4`/`8x8` campaign.

## Run

From the project root:

```bash
bash RVV_IME_GEMM_ACCURACY_VALIDATION/run_int8_ime_vs_rvv_accuracy.sh k1 1
```

Run four input classes:

```bash
INPUT_CLASSES="bounded_uniform full_range_uniform mixed_magnitude cancellation_stress" \
bash RVV_IME_GEMM_ACCURACY_VALIDATION/run_int8_ime_vs_rvv_accuracy.sh k1 1
```

Run experimental MF2 separately:

```bash
ENABLE_MF2=1 STRICT=0 \
bash RVV_IME_GEMM_ACCURACY_VALIDATION/run_int8_ime_vs_rvv_accuracy.sh k1 1
```

## Main Outputs

```text
int8_ime_vs_rvv_accuracy_summary_<mode>_latest.csv
int8_ime_vs_rvv_equivalence_<mode>_latest.csv
int8_ime_vs_rvv_accuracy_live_<mode>_latest.log
```

The detailed accuracy CSV contains every core and run. The equivalence CSV
combines matching IME and RVV configurations and reports:

```text
EQUIVALENT     all IME and RVV runs matched the trusted answer
NOT_VALIDATED  at least one path failed, was missing, or returned an error
```

All histograms, build logs, run logs, and timestamped summaries remain in the
campaign result directory.

## Files

| File | Purpose |
| --- | --- |
| `run_int8_ime_vs_rvv_accuracy.sh` | Runs the complete campaign and creates the final equivalence summary. |
| `run_int8_ime_vs_rvv_accuracy_once.sh` | Builds and runs each supported kernel for one input class. |
| `ime_rvv_fallback_diff_histogram_check.c` | Computes the trusted INT64 answer and element-wise error evidence. |
| `docs/RVV_IME_GEMM_Accuracy_Validation_Ozaki_Clear_Conference.pptx` | Simple conference presentation of the mathematics, Ozaki adaptation, and measured evidence. |
