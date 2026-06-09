# RVV DGEMM FP64 Microkernels: 8x8 Tiles

## Purpose

FP64 DGEMM microkernel family for benchmarking FP64 x FP64 -> FP64 with the 8x8 tile shape on RISC-V targets.

## Variant Matrix

| Property | Value |
|---|---|
| Backend | RVV |
| Tile shape | 8x8 |
| Variant count | 32 |
| ZVL targets | 128b, 256b |
| LMUL labels | lmul1, lmul2, lmul4, lmul8 |
| Unroll factors | unroll1, unroll2, unroll4, unroll8 |
| Benchmark driver | `dgemm_bench.c` |
| Reported metric | GFLOPS |

## Per-Variant Layout

```text
<kernel_variant>/
+-- <kernel_variant>.c
+-- dgemm_bench.c
+-- Makefile
```

Build and run one variant:

```bash
cd <kernel_variant>
make clean && make
./bench 1024 1024 1024
```

## Dataflow Summary

- Input panels are read using the layout expected by the benchmark driver.
- The main tile path uses the selected backend and the LMUL/unroll setting encoded in the folder name.
- Boundary cleanup handles rows or columns not covered by full micro-tiles.
- The output matrix is updated in column-major layout.

## Notes

This family evaluates FP64 RVV behavior for the 8x8 tile shape across two ZVL targets.
