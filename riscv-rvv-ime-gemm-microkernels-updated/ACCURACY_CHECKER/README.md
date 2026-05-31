# GEMM Microkernel Accuracy Checker

This folder validates the numerical correctness of the project's RVV
microkernels and IME GEMM wrappers. It is intentionally separate from the
performance benchmark scripts.

## Scope

The checker validates each `8x4` or `8x8` callable against an independent
reference calculation generated from the same input matrices.

It covers:

- RVV SGEMM: FP32 input and FP32 output.
- RVV DGEMM: FP64 input and FP64 output.
- RVV IGEMM: signed INT8 input and INT32 output.
- IME VMADOT IGEMM: signed INT8 input and INT32 output.

RVV microkernels receive packed tile panels. IME `8x4` wrappers receive full
K-major matrices and pack their IME tiles internally. IME `8x8` wrappers
receive prepacked row/column panels and split each `8x8` output tile into two
`8x4` VMADOT blocks. The checker preserves the expected input contract for
each path.

## Matrix Size and Seeds

Every run uses the project matrix size:

```text
M = 1024, N = 1024, K = 1024
```

The driver records the pseudorandom-generator seed for every run. The default
base seed is `0`, and the first dataset uses seed `1`. For a single input
class, increasing `RUNS` produces seeds `1, 2, 3, ...`.

The seed is not a hardware parameter or a mathematical constant. It is
recorded so that every microkernel receives the same generated matrices and
the experiment can be reproduced exactly. A different documented base seed
can be selected with `BASE_SEED`.

## Methodology Boundary

The floating-point campaigns are inspired by the uniformly random experiment
structure described by Parikh et al. in Section 5.2 of
[Cascading GEMM: High Precision from Low Precision](https://arxiv.org/pdf/2303.04353),
but they are not an exact reproduction of that paper. The paper evaluates
floating-point cascading GEMM and uses a higher-precision experimental
reference. This project evaluates RVV microkernels and IME VMADOT wrappers.

The INT8 campaigns use a project-specific exact correctness test. Every INT8
output element is compared against an independent wider-integer reference
calculated from the same signed INT8 matrices.

## Default Accuracy Campaign

| Kernels | Default class | Input values | Reference |
| --- | --- | --- | --- |
| RVV SGEMM FP32 | `bounded_uniform` | Uniform FP32 values in `[-1, 1)` | FP64 accumulation |
| RVV DGEMM FP64 | `bounded_uniform` | Uniform FP64 values in `[-1, 1)` | `long double` accumulation |
| RVV IGEMM INT8 | `full_range_uniform` | Uniform signed INT8 values in `[-128, 127]` | Wider integer accumulation |
| IME VMADOT IGEMM INT8 | `full_range_uniform` | Uniform signed INT8 values in `[-128, 127]` | Wider integer accumulation |

## Optional Diagnostic Classes

Additional classes are available for debugging and supplementary experiments.
They do not run unless explicitly selected.

| Datatype | Optional classes |
| --- | --- |
| FP32 and FP64 | `wide_range cancellation_stress` |
| INT8 | `bounded_uniform mixed_magnitude cancellation_stress` |

For example, run the complete INT8 diagnostic set on K1 IME cores:

```bash
INT8_INPUT_CLASSES="bounded_uniform full_range_uniform mixed_magnitude cancellation_stress" \
    bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-ime 1
```

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

Run from the project root. The runner intentionally requires one explicit
board mode. It does not provide an unpinned `all` mode, because K1 and K3 have
different RVV and IME core mappings.

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

For a final accuracy campaign, use more than one reproducible input set. For
example, this checks three INT8 IME matrix pairs using seeds `1`, `2`, and `3`:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-ime 3
```

## Output

Each board mode preserves its own latest summary. For example:

```text
ACCURACY_CHECKER/accuracy_summary_k1_rvv_latest.csv
ACCURACY_CHECKER/accuracy_summary_k1_ime_latest.csv
ACCURACY_CHECKER/accuracy_summary_k3_rvv_latest.csv
ACCURACY_CHECKER/accuracy_summary_k3_ime_latest.csv
```

Each CSV row includes the core, execution path, input contract, reference
method, generator seed, input class, matrix size, kernel metadata, numerical
metrics, status, and raw log path. A build failure uses `core=NOT_RUN` rather
than an ambiguous `core=NA`.

The `status` column is interpreted as follows:

| Status | Meaning |
| --- | --- |
| `OK` | The kernel ran and satisfied the numerical pass rule. |
| `NUMERICAL_FAILED` | The kernel ran, but its output did not satisfy the numerical pass rule. |
| `KERNEL_RETURN` | The executable returned a non-zero status, for example when the required IME path is unavailable. |
| `PATH_UNAVAILABLE` | The requested RVV or IME core path could not be selected on the board. |
| `BUILD_FAILED` | The checker could not compile the selected kernel. |
| `RUN_FAILED` | The compiled checker could not be executed successfully. |
