# GEMM Microkernel Accuracy Checker

This folder validates numerical correctness separately from performance
measurement.

## Scope

| Mode | Kernels | Cores |
| --- | --- | --- |
| `k1-rvv` | FP32, FP64, and INT8 RVV | 0-7 |
| `k1-ime` | Native INT8 VMADOT | 0-3 |
| `k3-rvv` | FP32, FP64, and INT8 RVV | 0-7 |
| `k3-ime` | Native INT8 VMADOT | 8-15 |

All native IME wrappers receive full K-major matrices and pack their private
A60 or A100 hardware tiles internally.

## Matrix Size and Inputs

Every run uses:

```text
M = 1024, N = 1024, K = 1024
```

The default INT8 experiment fills both matrices with uniformly selected
signed INT8 values in `[-128, 127]`. The default floating-point experiment
uses uniformly selected values in `[-1, 1)`.

The default base seed is `0`, so the first generated matrix pair uses seed
`1`. Recording the seed makes the generated inputs reproducible.

## References and Pass Rules

For INT8 RVV and IME kernels, the checker calculates:

```text
C_ref[i,j] = sum(A[i,k] * B[k,j]), for k = 0 ... 1023
```

using wider integer accumulation. The kernel passes only when every INT32
output element matches exactly and no reference output overflows INT32.

For FP32 and FP64, the checker reports absolute error, relative error,
Frobenius relative error, and a componentwise bound-normalized ratio.
Floating-point results are validated with a documented error bound because
different summation orders can produce small rounding differences.

## Strict Native IME Validation

Accuracy builds define:

```text
SPACEMIT_IME_REQUIRE_HARDWARE=1
```

This disables the local RVV fallback. A successful `k1-ime` or `k3-ime`
row therefore validates the VMADOT path.

Each native family includes `lmul1` and `lmulmf2` tuning variants. The public
SpacemiT documentation describes `LMUL=1`, so the checker skips `lmulmf2` by
default. Run an experimental board capability test explicitly with:

```bash
ENABLE_EXPERIMENTAL_MF2=1 bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-ime 1
ENABLE_EXPERIMENTAL_MF2=1 bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-ime 1
```

## Commands

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-rvv 1
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-ime 1
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-rvv 1
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-ime 1
```

For a final INT8 campaign with three reproducible matrix pairs:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-ime 3
```

## Status Values

| Status | Meaning |
| --- | --- |
| `OK` | The kernel executed and satisfied the numerical rule. |
| `NUMERICAL_FAILED` | The executable ran but its output failed validation. |
| `KERNEL_RETURN` | The kernel rejected the requested execution path. |
| `PATH_UNAVAILABLE` | The requested IME core marker was unavailable. |
| `BUILD_FAILED` | The checker could not compile the kernel. |
| `RUN_FAILED` | The compiled checker could not execute successfully. |
