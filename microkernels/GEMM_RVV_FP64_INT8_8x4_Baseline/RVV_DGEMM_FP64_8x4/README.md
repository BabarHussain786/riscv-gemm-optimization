# RVV DGEMM FP64 Microkernels: 8x4 Tiles

This folder contains the validated FP64 x FP64 -> FP64 RVV microkernels for the 8x4 output tile on a VLEN=256 target.

| Property | Value |
|---|---|
| Backend | RVV 1.0 |
| Tile shape | 8x4 |
| Retained variants | 12 |
| Compile target | `rv64gcv_zvl256b` |
| LMUL labels | lmul2, lmul4, lmul8 |
| Unroll factors | 1, 2, 4, 8 |
| Output | FP64 |
| Metric | GFLOPS |

Each variant contains its kernel source, `dgemm_bench.c`, and `Makefile`. The benchmark supports `GEMM_VALIDATE=1`, which compares every output value with an independent packed-panel reference before timing campaigns.

```bash
cd <kernel_variant>
make clean && make
GEMM_VALIDATE=1 ./bench 15 7 13
./bench 1024 1024 1024
```

The odd validation shape exercises the full vector path and every row/column remainder path. LMUL1 is intentionally excluded: on VLEN=256, `VLMAX = 256 / 64 = 4`, which cannot represent the kernel's eight-row vector without an explicit split.