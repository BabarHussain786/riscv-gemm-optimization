# Source-grounded implementation notes for Lorenzo

All paths here are relative to `microkernels`. The new adapter source is under
`benchmarking/src`; it does not edit the files described below.

## Canonical numerical interface

The actual indexing, rather than some inconsistent older source comments,
defines the contract: A(i,k) is `A[k*M+i]`; B(k,j) is `B[k*N+j]`; C(i,j) is
`C[j*M+i]`. A and C are column-major views, while this canonical B view is
row-major. The operation is C <- C + A*B with alpha=1. The same initialized
arrays and seed are used for every path. Packed layout is separate from these
canonical arrays. Output work is required to reach this final C layout in both
timing modes.

## RVV flow

Source: `HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h`.
Functions: `packed_block_size()` and `call_packed_rvv_tile_kernel()`.

The full row tile is 8; column tile is the selected 4 or 8. Remaining blocks are
decomposed into 4, 2, or 1 where applicable. Within a row block of R rows at i0,
the packer writes `A_pack[i0*K+k*R+r] = A[k*M+i0+r]`. Within a column block of
Q columns at j0 in strip n0, it writes
`B_pack[j0*K+k*Q+c] = B[k*full_N+n0+j0+c]`.

`benchmarking/src/rvv_adapter.c` includes that exact header with a no-op
`KERNEL_SYMBOL` callback solely to separate packing from execution. The actual
selected RVV source is compiled in an out-of-tree adapter translation unit with
the stable callback name `bench_rvv_entry` (using `CNAME` and, for literal-name
8x8 sources, a preprocessor symbol alias), and
called on the packed buffers. It performs widening INT8/INT32 computation,
tail handling, and the final C update internally. There is no separate output
transpose in this interface.

Canonical selected-source pattern:
`GEMM_RVV_FP32_INT8_<tile>_Baseline/RVV_IGEMM_INT8_I8I32_<tile>/<kernel>/<kernel>_i8i32.c`.
The FP64 baseline trees contain mirrored INT8 sources and are not distinct
primary tuning cases. The adapter is a benchmark of these micro-kernels with
the repository's OpenBLAS-style layout, not a benchmark of an installed full
OpenBLAS GEMM library and its complete cache-blocking strategy.

Diagram sequence: canonical A/B -> existing blocked A/B packer -> selected
RVV 8x4 or 8x8 micro-kernel, with fused tails and C update -> canonical C.

## IME flow

Canonical selected-source pattern:
`IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_<tile>_NATIVE/<kernel>/<kernel>.c`.
For example, the 8x4 U1 source is
`IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_8x4_NATIVE/ime_kernel_8x4_zvl256b_lmul1_unroll1/ime_kernel_8x4_zvl256b_lmul1_unroll1.c`.

Actual reused functions: `pack_a_panel()`, `pack_b_panel()`,
`b_panel_bytes()`, `native_accumulate()`, `scatter_output()`,
`output_index()`, and `scalar_gemm_block()`. Capability checks reuse
`cpu_has_ime()` and `detect_ime_profile()`, with an additional exact VLEN=256
gate in the new wrapper. The source's A60 profile consumes depth-eight input
chunks and native 4x4 subtiles; its software tile is 8x4 or 8x8.

The original OpenMP dispatcher first copies the selected B strip into compact
`B[k*strip_N+j]`. The new wrapper retains that copy as part of input packing.
For A60, `pack_a_panel()` orders elements by K chunk, group of four rows, row,
then eight K values. `pack_b_panel()` orders them by K chunk, group of four
columns, column, then eight K values. These are the existing scalar-C loops;
compiler auto-vectorization must not be described as a verified explicit RVV
packing implementation.

The native-depth prefix is `K_main=floor(K/(8*U))*(8*U)` for selected unroll U.
The adapter prepares all required panels ahead of execution to support the
prepacked mode. It then preserves tile order: native accumulation ->
`scatter_output()` -> K-tail cleanup. `scatter_output()` reads the native tile
using `output_index()` and adds into the final column-major C positions. Row
and column tails use the existing scalar helper. If K_main is zero, only
scalar cleanup is performed, without reading an uninitialized native output
buffer. Native capability validation is still required for an IME case.

There is **no separate final whole-matrix transpose** in this path. Diagram
sequence: canonical A/B -> compact B strip plus depth-eight A/B panel packing
-> native IME subtile accumulation -> per-software-tile output scatter and
tail cleanup -> canonical C. Draw output scatter inside the tile loop, not
as a single global transpose after every tile is finished.

## What changed around the untouched kernels

The original IME wrapper packs all A panels and one B panel at a time, executes
it, then reuses the B buffer. The new adapter packs and retains all B panels
for each strip so either backend can start from packed input. It also moves
allocation outside both timing modes and uses 32-column output strips. These
are benchmark workspace/lifecycle choices; they change cache behavior and
must be reported. They are not modifications to instruction encodings,
arithmetic, native kernel functions, or existing benchmark files.

Both layouts are resident before prepacked mixed execution; only the chosen
layout is packed during end-to-end mixed execution. A is packed per strip,
as in the repository's OpenMP helper; the new layer does not assert that this
is globally optimal cache blocking. Static work weights and dynamic chunk
sizes must be recorded and tuned separately from worker kernels.

## Search for earlier packing/timing code

The canonical RVV/IME kernels, heterogeneous dispatch, standalone programs,
paper launchers, validation sources, archived campaign source snapshots, and
the `extra` experimental sources were inspected. The reusable current packing
is located above. Lorenzo's specifically mentioned explicit vectorized packer
and separate packing timer implementation could not be identified in this
checkout. Ask him for the original repository/branch/commit; do not infer that
it never existed. No speculative vectorized replacement was created.

The standalone RVV driver initializes already-packed inputs before its kernel
timer. The standalone IME timer includes the wrapper's allocation, packing,
execution, scatter, tails and cleanup. In contrast, the current OpenMP
dispatcher already includes packing on **both** paths in its timed execution.
Therefore one blanket claim that all existing RVV measurements exclude packing
would be incorrect. Trace each figure to its exact launcher before rewriting
its methodology. Old shell perf measurements around the full process are not
interchangeable with the new worker-region counters.
