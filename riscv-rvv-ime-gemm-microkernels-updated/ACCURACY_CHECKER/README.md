# Accuracy Checker

This folder contains a small correctness checker for the RVV and IME GEMM kernels.
It is separate from the performance benchmark scripts.

## Purpose

The checker compares each optimized kernel result with a trusted reference result
computed from the same generated input matrices.

For floating-point kernels:

- FP32 RVV kernels are compared against a higher precision reference path.
- FP64 RVV kernels are compared against a higher precision reference path.

For integer kernels:

- INT8 RVV and INT8 IME kernels are compared against an exact integer reference path.

## Input Classes

The checker uses three input classes inspired by GEMM accuracy studies:

- `uniform`: bounded random values.
- `wide_range`: values with different magnitudes.
- `cancellation`: positive and negative terms designed to make dot products cancel.

## Run

The matrix size is fixed to the project benchmark size:

```text
M = 1024, N = 1024, K = 1024
```

From the project root:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh all 1
```

K1 RVV-only check:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k1-rvv 1
```

This tests cores `0-7` one by one.

K3 RVV-only check on cores 0-7:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-rvv 1
```

This tests cores `0-7` one by one.

K3 IME check on cores 8-15:

```bash
bash ACCURACY_CHECKER/run_accuracy_tests.sh k3-ime 1
```

This tests cores `8-15` one by one.

## Output

The latest CSV is written to:

```text
ACCURACY_CHECKER/accuracy_summary_latest.csv
```

Each row reports:

```text
family,kernel,datatype,tile_shape,zvl,lmul,unroll,input_class,
core,max_abs_error,mean_abs_error,max_rel_error,mean_rel_error,
mismatch_count,max_integer_difference,overflow_count
```
