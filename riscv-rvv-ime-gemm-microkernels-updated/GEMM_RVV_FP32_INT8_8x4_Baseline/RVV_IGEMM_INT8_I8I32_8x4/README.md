# RVV IGEMM INT8 Microkernels: 8x4 Tiles

## Purpose

INT8 IGEMM microkernel family for benchmarking INT8 x INT8 -> INT32 with the 8x4 tile shape on RISC-V targets.

## Variant Matrix

| Property | Value |
|---|---|
| Backend | RVV |
| Tile shape | 8x4 |
| Variant count | 28 |
| ZVL target | 256b |
| LMUL labels | lmulmf8, lmulmf4, lmulmf2, lmul1, lmul2, lmul4, lmul8 |
| Unroll factors | unroll1, unroll2, unroll4, unroll8 |
| Benchmark driver | `igemm_bench.c` |
| Reported metric | GOPS |

## Per-Variant Layout

```text
<kernel_variant>/
+-- <kernel_variant>.c
+-- igemm_bench.c
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

This family is the standalone RVV INT8 reference set for the 8x4 tile shape.
