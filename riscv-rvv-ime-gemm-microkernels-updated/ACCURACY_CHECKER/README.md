# GEMM Microkernel Accuracy Checker

This folder validates the numerical correctness of the project's packed RVV
and IME GEMM microkernels. It is intentionally separate from the performance
benchmark scripts.

## Scope

The checker validates each packed `8x4` or `8x8` microkernel against an
independent reference calculation generated from the same input panels.

It covers:

- RVV SGEMM: FP32 input and FP32 output.
- RVV DGEMM: FP64 input and FP64 output.
- RVV IGEMM: signed INT8 input and INT32 output.
- IME VMADOT IGEMM: signed INT8 input and INT32 output.

This is microkernel-level validation. It does not validate an external
row-major packing pipeline.

## Matrix Size and Seeds

Every run uses the project matrix size:

```text
M = 1024, N = 1024, K = 1024
```

The driver records a deterministic seed for every run. Repeating the checker
with the same seed reproduces the same generated matrices. Increasing `RUNS`
tests additional reproducible matrices rather than repeating one matrix.

## Input Classes

Floating-point kernels use:

| Class | Purpose |
| --- | --- |
| `bounded_uniform` | Baseline values sampled uniformly from `[-1, 1)` |
| `wide_range` | Random signs, mantissas, and binary exponents from `-20` to `20` |
| `cancellation_stress` | Neighboring K terms almost cancel, leaving a small nonzero result |

INT8 kernels use:

| Class | Purpose |
| --- | --- |
| `bounded_uniform` | Baseline values sampled uniformly from `[-16, 16]` |
| `full_range_uniform` | Full signed INT8 range `[-128, 127]` |
| `mixed_magnitude` | Mostly small values with occasional full-range values |
| `cancellation_stress` | Neighboring K terms cancel except for a one-unit final perturbation |

The class names describe this project's tests. They are informed by GEMM
accuracy-testing practice but are not an exact reproduction of another paper.

## Pass and Fail Rules

For INT8 RVV and IME kernels, the reference uses wider integer arithmetic:

```text
C_ref[i,j] = sum(A[i,k] * B[k,j]), for k = 0 ... 1023
```

An INT8 run passes only when:

```text
mismatch_count = 0
max_integer_difference = 0
overflow_count = 0
```

For FP32 and FP64, exact bitwise equality is not required because RVV fused
multiply-accumulate operations can use a different summation order. The CSV
reports absolute error, relative error, Frobenius relative error, and a
componentwise bound-normalized ratio:

```text
bound_ratio[i,j] =
    abs(C_kernel[i,j] - C_ref[i,j])
    / (K * unit_roundoff * sum(abs(A[i,k] * B[k,j])))
```

A floating-point run fails when any output is non-finite or exceeds the
documented bound.

## IME Path Verification

Accuracy builds of IME wrappers define:

```text
SPACEMIT_IME_REQUIRE_HARDWARE=1
```

This disables the local RVV fallback only for the accuracy checker. A
successful `k1-ime` or `k3-ime` row therefore means the IME VMADOT path ran.
Normal benchmark builds remain self-contained and retain their RVV fallback.

## Commands

Run from the project root.

K1 RVV kernels on cores `0-7`:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-rvv 1
```

K1 IME VMADOT kernels on cores `0-3`:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-ime 1
```

K3 RVV kernels on cores `0-7`:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-rvv 1
```

K3 IME VMADOT kernels on cores `8-15`:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-ime 1
```

For a final accuracy campaign, use more than one reproducible input set:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-rvv 3
```

## Output

The latest summary is written to:

```text
ACCURACY_CHECKER/accuracy_summary_latest.csv
```

Each row includes the core, execution path, seed, input class, matrix size,
kernel metadata, numerical metrics, status, and raw log path.
