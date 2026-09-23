# RVV FP32 SGEMM Microkernels: 8x8

This family contains 16 FP32 x FP32 -> FP32 RVV kernels for a VLEN=256 target.

| Property | Value |
|---|---|
| Software tile | 8x8 output values |
| LMUL variants | 1, 2, 4, 8 |
| K-loop unroll factors | 1, 2, 4, 8 |
| Metric | GFLOPS |
| Validation | Independent double-precision accumulation with FP32 tolerance |

With SEW=32 and VLEN=256, `VLMAX = LMUL x 256 / 32`; therefore LMUL=1 provides eight FP32 lanes and is the smallest valid group for the eight-row tile. LMUL=1/2 provides only four lanes and is intentionally absent.

Build and validate one variant:

```bash
cd sgemm_kernel_8x8_zvl256b_lmul1_unroll1
make clean && make
GEMM_VALIDATE=1 ./bench 15 15 13
```

The `15x15x13` check covers the 8x8 fast path and all 4/2/1 boundary paths before a regular 1024-cubed timing campaign.
