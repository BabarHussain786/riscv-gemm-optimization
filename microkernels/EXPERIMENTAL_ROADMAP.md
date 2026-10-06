# Experimental Roadmap

This project measures RVV, native IME, and heterogeneous RVV--IME INT8 GEMM
on the SpacemiT K1.  The experiments are organized by what is measured, not
by paper-figure numbering.

## 1. Kernel inventory and build checks

Build the canonical RVV and IME kernels for the 8x4 and 8x8 tiles, all
supported LMUL values, and unroll factors U1, U2, U4, and U8.  Experimental
IME `lmulmf2` variants remain excluded unless explicitly enabled.

## 2. Single-core characterization

Measure FP32, FP64, RVV INT8, and native IME INT8 kernels on pinned K1 cores.
Each run performs correctness validation before timed execution and records
time, throughput, optional IPC, and failure status.

## 3. Fair end-to-end comparison

Compare eight-core RVV with the heterogeneous four-RVV plus four-IME path
using the same 1024^3 INT8 workload, tile width, validation method, and seven
timed repetitions.  Input preparation/packing is included in both paths.

## 4. Scaling behavior

- **Strong scaling:** keep the 1024^3 workload fixed while changing the
  worker/core count.
- **Weak scaling:** increase the workload with the available worker count.

Both experiments use pinned workers and record the schedule, worker
placement, timing scope, and validation result.

## 5. Scheduling and partitioning

Compare static and dynamic tile assignment for the heterogeneous path.  The
logs record the worker split, tile ownership, schedule policy, and chunk size
so that load balance can be checked rather than inferred from a single time.

## 6. Execution-phase timing

Where supported, record separate input-packing, kernel-execution, and
output-unpack/scatter/reshape times.  Phase totals are labelled as worker
time or wall-clock time so they are not confused with end-to-end latency.

## 7. Repeatability and statistics

Every reported configuration uses repeated runs.  Raw run records are kept
alongside summaries containing the mean, median, minimum, maximum, and sample
standard deviation.  Failed builds, failed validations, and excluded
experimental kernels are recorded explicitly.

## 8. Numerical correctness

RVV and IME INT8 results are checked against an independent INT64-accumulation
reference.  The checker reports exact matches, mismatch counts, maximum error,
overflow status, and error histograms.

## Main entry points

- `paper_scripts/strong_scaling_performance/`
- `paper_scripts/weak_scaling_performance/`
- `paper_scripts/static_dynamic_scheduling/`
- `paper_scripts/rvv_int8_tuning/`
- `paper_scripts/heterogeneous_rvv_ime_end_to_end/`
- `paper_scripts/fp32_fp64_comparison/`
- `paper_scripts/multicore_comparison/`
- `paper_scripts/correctness_validation/`

Clean datasets are stored under `datasets/`.  The benchmark implementation is
under `benchmarking/` and `HETEROGENEOUS_RVV_IME_OPENMP_GEMM/`.
