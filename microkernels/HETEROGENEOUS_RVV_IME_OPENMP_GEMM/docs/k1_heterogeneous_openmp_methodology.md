# K1 Heterogeneous OpenMP GEMM Methodology

## Objective

The experiment compares RVV-only, IME-only, heterogeneous static, and heterogeneous dynamic execution for the same dense GEMM. Existing micro-kernels perform the arithmetic. OpenMP controls only core placement and ownership of output-column strips.

## Matrix Partition

For `C[M x N]`, one OpenMP output strip contains `tile_N` complete columns:

```text
T = ceil(N / tile_N)
strip t owns columns [t*tile_N, min(N, (t+1)*tile_N))
```

Strips are disjoint. Therefore each `C(i,j)` has one owner and no output lock is needed. `T` counts OpenMP strips, not micro-kernel or native IME tiles.

## Static Policy

```text
OpenMP level 1: two cluster controllers
OpenMP level 2: four workers per cluster
IME cluster:    cores 0-3
RVV cluster:    cores 4-7
```

Before execution, the code calculates one fixed boundary:

```text
T_IME = round(T * W_IME / (W_IME + W_RVV))
T_RVV = T - T_IME
```

Each cluster uses `omp for schedule(static)` on its own fixed range. This is the low-overhead policy for a regular workload because no worker requests new tiles during execution.

## Dynamic Policy

```text
OpenMP team: 8 pinned workers
IME workers: IDs/cores 0-3
RVV workers: IDs/cores 4-7
tile loop:   omp for schedule(dynamic, chunk)
```

OpenMP assigns another chunk when a worker becomes free. Faster workers can therefore complete more tiles. This can improve an unknown IME/RVV imbalance, but it adds runtime scheduling overhead. The project does not add a second custom scheduler above OpenMP.

## Kernel Dispatch

Every loop iteration computes one `C[M x tile_N]` strip. IME workers call the native IME wrapper. RVV workers pack the required input panels and call the matching low-level RVV widening kernel. Both paths use the selected 8x4 or 8x8 family and compute:

```text
C(i,j) = C(i,j) + sum_k A(i,k) * B(k,j)
INT8 x INT8 -> INT32 accumulation and output
```

## Timing

The timed interval starts immediately before the OpenMP tile region and ends after all strips complete. It includes OpenMP team creation, scheduling, required packing, kernel execution, and writes to `C`. Matrix allocation, initialization, warmup, independent validation, and cleanup are outside the timed interval.

All compared runs use the same `M`, `N`, `K`, `tile_N`, kernel variant, and repetition count. This isolates the effect of scheduling policy.

## Validation

The first repetition compares the OpenMP output with an untimed reference that does not call the tested kernel. A, B, and the initial C use different fixed-seed pseudo-random patterns so layout errors are not hidden by one repeated input sequence. INT8 products are accumulated in INT64, checked for INT32 overflow, and compared exactly after conversion to INT32. FP32 uses double accumulation and FP64 uses long-double accumulation before tolerance comparison. A failed first check stops later repetitions for that kernel.

Every mode receives the same unpacked contract: `A[k*M+i]`, `B[k*N+j]`, and `C[j*M+i]`. RVV panel packing and IME input preparation required by the selected implementation are included in the timed region. Mixed RVV workers link the same canonical `GEMM_RVV_FP32_INT8_*` source selected by the pure-RVV campaign, not the different fallback copy stored beside each IME kernel. This keeps the implementation identity, input contract, and timing boundary consistent across pure and mixed modes.

The runner excludes known unsupported or misleading VLEN=256 variants before compilation: duplicate INT8 source copies, FP32 8x8 `LMUL=mf2`, FP64 8x4 `LMUL=1`, and INT8 `LMUL=4/8` widening labels.

## Required Checks

```text
actual mixed workers = 8
worker i executes on CPU i
IME completed strips + RVV completed strips = T
mismatch_count = 0
```

For static mode, the completed IME and RVV counts must also equal their fixed ranges. For dynamic mode, the split may change between runs, but the total must always equal `T`.
