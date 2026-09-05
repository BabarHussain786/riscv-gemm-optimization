# RVV IGEMM INT8 Microkernels: 8x4 Tiles

## Purpose

INT8 IGEMM microkernel family for benchmarking INT8 x INT8 -> INT32 with the 8x4 tile shape on RISC-V targets.

## Variant Matrix

| Property | Value |
|---|---|
| Backend | RVV |
| Tile shape | 8x4 |
| Variant count | 20 |
| ZVL target | 256b |
| LMUL labels | lmulmf8, lmulmf4, lmulmf2, lmul1, lmul2 |
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
GEMM_VALIDATE=1 ./bench 15 7 13
GEMM_VALIDATE=0 ./bench 1024 1024 1024
```

## Dataflow Summary

- Input panels are read using the layout expected by the benchmark driver.
- The main tile path uses the selected backend and the LMUL/unroll setting encoded in the folder name.
- Boundary cleanup handles rows or columns not covered by full micro-tiles.
- The output matrix is updated in column-major layout.
- Validation uses independent INT64 accumulation and then compares every INT32
  output value exactly.

## Notes

This family is the standalone RVV INT8 reference set for the 8x4 tile shape.
LMUL4 and LMUL8 names are intentionally absent: widening INT8 to INT16 and then
accumulating into INT32 would require illegal destination register groupings
beyond the LMUL2 source configuration. Keeping duplicated LMUL2 code under
LMUL4/LMUL8 names would misrepresent the executed kernel.
