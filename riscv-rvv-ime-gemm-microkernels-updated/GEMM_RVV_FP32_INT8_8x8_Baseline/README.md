# FP32 and INT8 RVV GEMM Baseline: 8x8 Tiles

This baseline contains the validated RVV 8x8 microkernels used for standalone FP32 and INT8 performance measurements on a VLEN=256 target. Native IME kernels are maintained separately under `../IME_NATIVE_KERNELS/`.

## Active Kernel Families

| Folder | Operation | LMUL variants | Unroll variants | Total |
|---|---|---|---|---:|
| `RVV_SGEMM_FP32_8x8/` | FP32 x FP32 -> FP32 | 1, 2, 4, 8 | 1, 2, 4, 8 | 16 |
| `RVV_IGEMM_INT8_I8I32_8x8/` | INT8 x INT8 -> INT32 | 1/4, 1/2, 1, 2 | 1, 2, 4, 8 | 16 |

Each variant directory contains one kernel source, one validated benchmark driver, and one Makefile.

## Run the Campaign

```bash
cd ~/riscv-rvv-ime-gemm-microkernels/GEMM_RVV_FP32_INT8_8x8_Baseline
bash run_single_core_0_7_all_kernels_1024.sh
```

Defaults are `M=N=K=1024`, six timed runs, and cores 0-7. CPU pinning with `taskset` is mandatory. Before timing, every kernel must pass an independent numerical check at `15x15x13`; this shape exercises the full 8x8 path and the 4/2/1 row and column cleanup paths.

Settings can be overridden explicitly:

```bash
M=512 N=512 K=512 RUNS=10 CORES="0 1 2 3" \
  bash run_single_core_0_7_all_kernels_1024.sh
```

Set `VALIDATE=0` only for a deliberate timing-only campaign.

## Results

Results are written under `single_core_results_<M>/`. Use `single_core_summary_latest.csv` for analysis; it reports the mean, median, population standard deviation, minimum, maximum, average time, status counts, and validation state. The raw CSV and `raw_logs/` preserve every individual run.

## Variant Rules

- FP32 uses LMUL 1-8. With VLEN=256, FP32 LMUL=1 already holds the eight rows required by the tile.
- INT8 widening uses source LMUL 1/4 through 2, producing legal INT32 destination groups through LMUL=8.
- FP32 LMUL=1/2 and INT8 LMUL=1/8 were removed because they cannot hold eight source rows at VLEN=256.
- INT8 LMUL=4/8 were removed because widening to INT32 would require illegal destination groups beyond LMUL=8.
- INT8 output updates use explicit modulo-2^32 arithmetic, avoiding undefined signed overflow.
