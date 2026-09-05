# RVV INT8 IGEMM Microkernels: 8x8

This family contains 16 INT8 x INT8 -> INT32 RVV kernels for a VLEN=256 target.

| Property | Value |
|---|---|
| Software tile | 8x8 output values |
| Source LMUL variants | 1/4, 1/2, 1, 2 |
| K-loop unroll factors | 1, 2, 4, 8 |
| Metric | GOPS |
| Validation | Independent INT64 dot products, converted to the defined INT32 result |

The widening path follows `INT8(LMUL) -> INT16(2 x LMUL) -> INT32(4 x LMUL)`. LMUL=1/4 is the smallest source group that holds eight INT8 rows at VLEN=256. LMUL=2 is the largest source group whose INT32 destination remains legal at LMUL=8. This is why LMUL=1/8, 4, and 8 are intentionally absent.

Build and validate one variant:

```bash
cd igemm_kernel_8x8_zvl256b_lmulmf4_unroll1
make clean && make
GEMM_VALIDATE=1 ./bench 15 15 13
```

The kernel accumulates into INT32 vectors and updates C with explicit modulo-2^32 arithmetic. The `15x15x13` validation exercises the full tile plus every row and column cleanup path.
