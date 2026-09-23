# RVV SGEMM FP32 Microkernels: 8x4 Tiles

## Purpose

FP32 SGEMM microkernel family for benchmarking FP32 x FP32 -> FP32 with the 8x4 tile shape on RISC-V targets.

## Variant Matrix

| Property | Value |
|---|---|
| Backend | RVV |
| Tile shape | 8x4 |
| Variant count | 20 |
| ZVL target | 256b |
| LMUL labels | lmulmf2, lmul1, lmul2, lmul4, lmul8 |
| Unroll factors | unroll1, unroll2, unroll4, unroll8 |
| Benchmark driver | `sgemm_bench.c` |
| Reported metric | GFLOPS |

## Per-Variant Layout

```text
<kernel_variant>/
+-- <kernel_variant>.c
+-- sgemm_bench.c
+-- Makefile
```

Build and run one variant:

```bash
cd <kernel_variant>
make clean && make
GEMM_VALIDATE=1 ./bench 15 7 13
GEMM_VALIDATE=0 ./bench 1024 1024 1024
```

## Dataflow Summary

- Input panels are read using the layout expected by the benchmark driver.
- The main tile path uses the selected backend and the LMUL/unroll setting encoded in the folder name.
- The encoded unroll factor is applied to every K loop in both vector and scalar paths.
- Boundary cleanup handles rows or columns not covered by full micro-tiles.
- The output matrix is updated in column-major layout.
- Validation compares the kernel output against an independent reference that
  accumulates each FP32 dot product in double precision.

## Notes

This family is used in the FP32 8x4 baseline and is included in both K1 and K3 RVV benchmark campaigns.
