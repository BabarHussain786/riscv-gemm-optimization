# K1 Heterogeneous OpenMP GEMM Methodology

## Objective

The experiment measures whether the K1 IME-capable cluster and RVV-only cluster can compute one dense INT8 GEMM concurrently with low scheduling overhead. Existing micro-kernels perform the arithmetic; OpenMP only divides the output matrix and maps work to cores.

## Matrix Partition

For `C[M x N]`, `tile_N` columns form one OpenMP tile:

```text
number of tiles = ceil(N / tile_N)
tile t owns columns [t*tile_N, min(N, (t+1)*tile_N))
```

Tiles are contiguous and disjoint. Therefore each output element has exactly one owner and no lock is required for `C`.

## Nested Static Execution

Mixed mode creates two nested OpenMP levels:

```text
Level 1: two cluster controllers
Level 2: four workers per cluster with schedule(static)
```

Cluster 0 is pinned to cores 0-3 and calls the native IME wrapper with hardware execution required. Cluster 1 is pinned to cores 4-7 and directly calls the matching low-level RVV widening kernel. The runtime placement and completed tile count of every worker are recorded.

The tile boundary is calculated once:

```text
T_IME = round(T * W_IME / (W_IME + W_RVV))
T_RVV = T - T_IME
```

`W_IME=4` and `W_RVV=1` are the default starting values. They describe a static workload ratio, not dynamic chunk sizes. Changing these two values changes the balance without changing the OpenMP structure.

## Kernel Dispatch

Each OpenMP iteration computes one `C[M x tile_N]` column strip. In mixed mode, cores 0-3 call the native IME wrapper with a compact B strip. Cores 4-7 pack canonical A and the owned B strip into the matching RVV layout, then call the RVV widening micro-kernel directly. Both paths apply the selected 8x4 or 8x8 kernel family.

Both paths compute:

```text
C(i,j) = C(i,j) + sum_k A(i,k) * B(k,j)
```

with INT8 inputs and INT32 accumulation/output.

## Timing

Timed work includes nested OpenMP team creation, static loop execution, required input packing, kernel execution, and writes to `C`. Main-matrix and OpenMP worker-buffer allocation, initialization, warmup, serial validation, and cleanup are excluded. Temporary allocation or packing performed inside a selected kernel remains included. Every compared mode uses the same `M`, `N`, `K`, `tile_N`, and repetition count.

## Validation

The first repetition compares the OpenMP output with an untimed serial run of the same selected kernel. INT8 comparison is exact; FP32 and FP64 use an absolute-plus-relative tolerance. This detects tile-boundary, dispatch, and parallel assembly errors. Independent high-precision numerical validation is performed by the separate accuracy module.

## Cluster-Aware Placement

Contiguous tile ranges preserve stable output ownership and avoid inter-cluster write sharing. Mixed workers are pinned to exact K1 cores, and OpenMP-owned worker buffers are allocated before timing. The available system memory-node list is recorded in each campaign log.

## Required Run Checks

A mixed result is valid only when all conditions hold:

```text
inner team sizes: 4 IME workers and 4 RVV workers
worker CPUs:      0,1,2,3 and 4,5,6,7
IME tile total:   equals the fixed IME range
RVV tile total:   equals the fixed RVV range
validation:       mismatch_count = 0
```
