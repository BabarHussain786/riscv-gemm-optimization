# INT8 GEMM quantization accuracy

This additional experiment measures the error introduced when real-valued
matrices are quantized to INT8. It calls the existing RVV and native IME kernels;
it does not replace them or change the report, paper, or previous measurements.

Two questions are checked separately:

1. **Integer correctness:** does every kernel output match an independent INT64
   calculation of the same quantized operands?
2. **Quantization error:** after dequantization, how far is that output from an
   FP64 GEMM calculation on the original real-valued operands?

The results are matrix-level errors, not neural-network classification accuracy.

## Files

| File | Role |
| --- | --- |
| `quantization_accuracy.py` | Generates inputs, quantizes, calls kernels, checks correctness, and writes measurements |
| `kernel_bridge.c` | Calls the project's existing adapters on explicitly selected K1 cores |
| `build_native.py` | Builds the bridge and unchanged selected kernels; records hashes and compiler commands |
| `plot_quantization_accuracy.py` | Checks complete data coverage and creates PNG/SVG box plots |
| `run_quantization_accuracy.sh` | Starts a pilot or full experiment from any working directory |
| `test_quantization_accuracy.py` | Tests the numerical definitions and result-validation rules locally |
| `tests/host_bridge_check.c` | Checks the bridge on a Linux host using explicitly labeled stub backends |

## Existing kernels used

- RVV: `igemm_kernel_8x4_zvl256b_lmulmf8_unroll2`.
- Native IME: `ime_kernel_8x4_zvl256b_lmul1_unroll1`.
- Both have 8x4 output tiles and INT32 accumulation.
- The bridge reuses `benchmarking/src/rvv_adapter.c` and `ime_adapter.c`.
- A is M-by-K column-major; B is K-by-N row-major; C is M-by-N column-major at
  the C interface. The Python program explicitly prepares these memory layouts.
- The IME adapter calls native A60 accumulation directly. It does not substitute
  an RVV fallback. Hardware/profile checks must succeed on the selected core.
- Default cores are RVV CPU 4 and IME CPU 0. Override `--rvv-cpu` and `--ime-cpu`
  if the actual K1 mapping requires it; the bridge checks affinity and support.

This first experiment is deliberately limited to these two kernels and K1/A60
with VLEN=256. It does not establish results for K3, other tiles, all tuning
settings, or heterogeneous OpenMP schedules.

## Protocol

For each independently seeded input pair, compute the original product in FP64.
Reuse the same pair for both quantization schemes and both backends.

- **Per-tensor:** one absolute-maximum scale for all of A, and one for all of B.
- **Per-row/per-column:** one scale per row of A and per column of B.

Both schemes use signed codes from -127 through 127, zero point zero, dynamic
absolute-maximum calibration, and round-to-nearest-even. A zero-valued group
uses scale 1 and integer codes zero. There is no intentional percentile clipping.

For each scheme, compute an independent INT64 product. Every reference value
must fit INT32, and every kernel element must match it exactly. A mismatch,
hardware failure, missing result, or overflow stops the experiment. Such a run
is not silently filtered into a successful plot.

Dequantization multiplies each output by the corresponding A and B scales.
Compare it with the original FP64 product. The CSV records:

- relative Frobenius error (a fraction; the plot multiplies by 100);
- RMSE, maximum absolute error, and maximum error divided by max(abs(reference));
- SQNR in decibels and reference Frobenius norm;
- seed, dimensions, scheme, backend, and exact integer-validation counts.

FP64 is a high-precision floating-point comparison, not exact real arithmetic.
Relative Frobenius error equals RMSE divided by RMS(reference). It is not a
percentage of correct elements. Lower relative error is better.

Zero-reference convention: an all-zero comparison and result have error zero;
a nonzero result against an all-zero comparison is rejected as undefined for
relative metrics. Exact results have SQNR recorded as positive infinity.

The experiment fixes alpha=1 and initial C=0. M must be divisible by 8, N by 4,
and K by 8, excluding scalar cleanup in the selected IME adapter. K is limited
to 131071, a conservative bound ensuring all INT32 partial sums fit even with
full-range signed INT8 inputs. The actual experiment uses [-127,127].

## Inputs and figure

Three reproducible synthetic input distributions are included:

- Uniform: independent values between -1 and 1 in both matrices.
- Gaussian: independent standard-normal values in both matrices.
- Outliers: standard-normal matrices, with 5% of A rows and 5% of B columns
  (rounded up, at least one each) multiplied by 20.

The outlier case tests scale granularity along output rows/columns. It is not a
claim about a particular application's activation or weight distribution.

The final figure has two panels, one for each quantization scheme. Each panel
groups boxes by input distribution and displays RVV and IME side by side. Each
box contains one error value per independently generated matrix pair. Median,
interquartile range, 1.5-IQR whiskers and outliers are shown explicitly.

Correct RVV and IME kernels given the same quantized matrices should have
identical outputs and quantization errors. The program checks this equality.
The boxes compare quantization schemes and input populations, not intrinsic
accuracy differences between correct backends. Repeated identical inputs are
not used as new independent samples.

The full default run uses M=N=K=512 and 100 pairs per distribution: 300 unique
real-valued pairs and 1200 measurement rows (3 distributions x 100 inputs x
2 schemes x 2 backends). The pilot uses 64x64x64 and 3 pairs per distribution:
9 unique pairs and 36 native measurement rows. This is correctness/accuracy
evaluation, not a timing or throughput benchmark.

## Run on the K1

Place this new folder inside the existing `microkernels` project on the K1.
Keep the original project directories next to it. A source check identifies any
missing required files before compiling. It does not execute hardware.

Required: Linux on K1, the project's RVV-compatible GCC toolchain (including
`riscv_vector.h` and OpenMP), Python 3.10 or later, NumPy, and Matplotlib for
plotting. Install Python dependencies in the Python environment you use:

```bash
python3 -m pip install -r requirements.txt
```

From this new folder, check sources and run the pilot:

```bash
python3 build_native.py --check
bash run_quantization_accuracy.sh --pilot
```

After the pilot passes, collect the full dataset and plot:

```bash
bash run_quantization_accuracy.sh
```

Each run creates a new timestamped directory in `results/`. Existing results
are never overwritten. The directory contains:

```text
manifest.json
quantization_accuracy.csv
build/build.json
build/build.log
build/libquant_accuracy.so
plots/quantization_accuracy_native_k1.png
plots/quantization_accuracy_native_k1.svg
```

The manifest must have status `complete`. It records dimensions, source hashes,
compiler settings, CPU information, seeds, input parameters, package versions,
and the CSV hash. Plotting rejects incomplete or invalid datasets. A process
interrupted by an illegal instruction can leave a manifest marked `running`;
that is incomplete, not evidence of a successful native experiment.

To collect on the K1 without installing Matplotlib there:

```bash
bash run_quantization_accuracy.sh --no-plot
```

Copy that run's CSV and sibling `manifest.json` together to a plotting computer.
Then run:

```bash
python3 plot_quantization_accuracy.py --csv /path/to/run/quantization_accuracy.csv
```

Larger experiments can be expensive because each new input needs an independent
INT64 calculation as well as an FP64 product. Start with the pilot; do not
interpret evaluation wall-clock duration as kernel performance.

## Local software verification

This explicit mode exercises input generation, quantization, exact references,
dequantization, output validation and plotting on a normal computer. It never
calls RVV or IME and is never labeled as a native measurement:

```bash
python3 quantization_accuracy.py --software-verification --pilot
python3 -m unittest discover -s . -p 'test_*.py' -v
```

Software output uses the backend name `scalar`, a distinct execution kind and
a prominent warning on its plots. The plotter refuses these data unless
`--allow-software-verification` is explicitly provided. The main program adds
that flag only for this explicit local mode.

No native K1 result is included with the code. The existing zero-error integer
dataset is not converted into invented quantization-error values.

## Checks performed before delivery

- Required original sources were found in the specified project.
- All 20 Python numerical and pipeline tests passed.
- The C bridge compiled with strict warnings on a Linux host and passed its
  host-only argument, allocation, cleanup, and affinity tests.
- A 64x64x64 software-verification run used 100 independent pairs per distribution,
  producing 600 scalar records and labeled PNG/SVG plots. The rendered plot was
  inspected for clipping, overlap, labels, units, and layout.
- Native RISC-V compilation, IME instructions and RVV execution were not tested
  on this Windows computer. The K1 pilot is the required next verification.
