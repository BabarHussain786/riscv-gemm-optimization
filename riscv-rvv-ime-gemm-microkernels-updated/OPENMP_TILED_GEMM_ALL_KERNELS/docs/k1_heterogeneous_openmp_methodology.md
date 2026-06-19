# K1 Heterogeneous OpenMP GEMM Methodology

This note documents the experimental method implemented by the OpenMP tiled GEMM benchmark. The goal is to compare non-heterogeneous and heterogeneous execution on the same K1 matrix-multiplication workload.

## Execution Model

The benchmark treats the K1 as two execution domains:

```text
cores 0-3: IME-capable domain
cores 4-7: RVV-only domain
```

The heterogeneous mode, `k1-mixed-rvv-ime`, uses both domains at the same time. IME-capable cores execute the native IME INT8 path. RVV-only cores execute the RVV fallback path. Both paths compute the same mathematical operation for INT8 GEMM:

```text
C(i,j) = C(i,j) + sum_k A(i,k) * B(k,j)
input:  INT8 A and INT8 B
output: INT32 C with INT32 accumulation
```

The FP32 and FP64 paths are RVV-only baselines because the IME path in this project targets INT8-to-INT32 matrix multiplication.

## Tile Decomposition

OpenMP parallelism is applied above the micro-kernel level. The output matrix `C` is split by column blocks. If `M=N=K=1024` and `tile_N=32`, then the `1024 x 1024` output matrix is divided into 32 column tiles, each with shape `1024 x 32`.

```text
complete output: C[1024 x 1024]
OpenMP tile:     C[1024 x tile_N]
micro-kernel:    existing 8x4 or 8x8 kernel update
```

Each OpenMP worker claims one or more column tiles and writes only its assigned columns of `C`. This avoids write races because no two workers update the same output columns.

## Heterogeneous Scheduling

The mixed scheduler uses dynamic tile assignment. Each worker first observes the CPU on which it is running. On K1, cores 0-3 are treated as IME workers and cores 4-7 are treated as RVV fallback workers.

The default scheduling weights are:

```text
MIXED_IME_TILE_WEIGHT=4
MIXED_RVV_TILE_WEIGHT=1
```

This means an IME worker claims a larger tile chunk than an RVV worker. The purpose is not to force a fixed static split, but to let the faster IME-capable cores receive more work while keeping RVV cores active on the same global GEMM.

## NUMA and Locality Considerations

The benchmark allocates temporary B-tile buffers inside each OpenMP worker after the worker observes its CPU. This gives local first-touch behavior for the temporary per-worker buffers. These buffers are important in IME and mixed modes because the IME wrapper expects a contiguous K-major B tile.

The global matrices `A`, `B`, and `C` are still allocated by the process before the parallel region. If a platform exposes stronger NUMA behavior, external memory-placement tools or platform-specific allocators can be used to bind global matrix pages to the preferred memory domain. The benchmark records worker placement so that performance can be interpreted together with the actual core-domain assignment.

## Compared Cases

The K1 campaign reports four execution cases:

```text
k1-rvv             RVV kernels on cores 0-7
k1-rvv-only        RVV kernels on cores 4-7
k1-ime             native IME kernels on cores 0-3
k1-mixed-rvv-ime   native IME on cores 0-3 plus RVV fallback on cores 4-7
```

These cases separate portable vector execution, cluster-local IME execution, and true heterogeneous execution.

## Validation Meaning

For each kernel configuration, the first measured repetition validates the OpenMP tiled result against an untimed serial same-kernel execution. INT8 outputs require exact INT32 equality. FP32 and FP64 use a small absolute-plus-relative tolerance. This check validates the tiled OpenMP assembly and dispatch logic. Independent numerical accuracy against a higher-precision reference remains part of the separate accuracy-checking workflow.