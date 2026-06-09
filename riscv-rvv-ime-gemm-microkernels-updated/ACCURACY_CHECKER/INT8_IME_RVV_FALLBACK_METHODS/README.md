# INT8 IME Native and RVV Fallback Validation Campaigns

These scripts validate only the INT8-to-INT32 kernels under
`IME_NATIVE_KERNELS`. Each campaign compares both execution paths:

- `IME_NATIVE`: hardware IME path with the local RVV fallback disabled.
- `RVV_FALLBACK`: fallback path from the same kernel source folder.

All methods reuse `../run_ime_vs_rvv_fallback_histograms.sh`, which computes
an independent INT64-accumulated reference and compares every INT32 output
element against it.

The campaigns use reference-comparison ideas related to the cited papers, but
they validate exact INT8-to-INT32 arithmetic and are not reproductions of the
papers' floating-point experiments.

## Scripts

### 1. Element-wise exact-error campaign

```bash
ENABLE_MF2=1 bash ACCURACY_CHECKER/INT8_IME_RVV_FALLBACK_METHODS/run_01_ozaki_inspired_elementwise_error.sh k1 1
```

Related paper: Ozaki-II, because both examine element-wise output errors.

Uses full-range uniform INT8 inputs. It records the complete signed-error
histogram, mismatch count, exact-match rate, maximum absolute error, overflow
count, and input histogram. This is the primary bit-exact validation and the
source for an error-distribution or absolute-error CCDF plot.

### 2. Multi-input reference-comparison campaign

```bash
ENABLE_MF2=1 bash ACCURACY_CHECKER/INT8_IME_RVV_FALLBACK_METHODS/run_02_cascading_gemm_inspired_multi_input.sh k1 1
```

Related paper: Cascading GEMM, because both compare GEMM results against an
independently computed reference across controlled input experiments.

Repeats the reference comparison across:

- `bounded_uniform`
- `full_range_uniform`
- `mixed_magnitude`
- `cancellation_stress`

It produces one combined CSV so sensitivity to input distributions can be
compared.

### 3. Range and overflow campaign

```bash
ENABLE_MF2=1 bash ACCURACY_CHECKER/INT8_IME_RVV_FALLBACK_METHODS/run_03_narrow_range_inspired_validation.sh k1 1
```

Related paper: Narrow-Range Floating Point, because both examine representable
range and overflow safety. The reported integer metrics are specific to this
project.

Uses full-range uniform INT8 inputs and reports:

- mismatch rate
- exact-match rate
- maximum absolute error
- theoretical maximum accumulation magnitude
- INT32 safety margin
- normalized maximum error
- overflow count

For the default `K=1024`, the theoretical full-range accumulation bound is:

```text
1024 * 128 * 128 = 16,777,216
```

## Board Modes

| Mode | IME native cores | RVV fallback core |
| --- | --- | --- |
| `k1` | 0-3 | 4 |
| `k3` | 8-15 | 0 |

All scripts default to `M=N=K=1024`, `BASE_SEED=0`, and `ENABLE_MF2=1`.
