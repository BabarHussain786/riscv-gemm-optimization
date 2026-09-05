# RVV DGEMM FP64 Microkernels: 8x8 Tiles

This folder contains the validated FP64 x FP64 -> FP64 RVV microkernels for the 8x8 output tile on a VLEN=256 target.

| Property | Value |
|---|---|
| Backend | RVV 1.0 |
| Tile shape | 8x8 |
| Retained variants | 16 |
| Compile target | `rv64gcv_zvl256b` |
| LMUL labels | lmul1, lmul2, lmul4, lmul8 |
| Unroll factors | 1, 2, 4, 8 where present |
| Output | FP64 |
| Metric | GFLOPS |

Each variant contains its kernel source, `dgemm_bench.c`, and `Makefile`. The benchmark supports `GEMM_VALIDATE=1`, which compares every output value with an independent packed-panel reference before timing campaigns.

```bash
cd <kernel_variant>
make clean && make
GEMM_VALIDATE=1 ./bench 15 15 13
./bench 1024 1024 1024
```

The odd validation shape exercises the full vector path and every row/column remainder path. LMUL1 is valid here because its source explicitly represents the eight rows as two independent four-lane vectors.