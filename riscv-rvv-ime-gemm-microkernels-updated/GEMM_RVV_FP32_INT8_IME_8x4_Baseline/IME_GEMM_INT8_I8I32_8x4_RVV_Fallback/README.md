# IME GEMM INT8 Microkernels With RVV Fallback: 8x4 Tiles

## Purpose

Self-contained SpacemiT IME INT8 GEMM microkernel family for 8x4 tiles. Each variant includes the IME implementation and a local RVV fallback source file, so the folder can be built independently from the standalone INT8 RVV kernel tree.

## Variant Matrix

| Property | Value |
|---|---|
| Operation | INT8 x INT8 -> INT32 |
| Backend | SpacemiT IME fast path with RVV fallback |
| Tile shape | 8x4 |
| Variant count | 8 |
| ZVL target | 256b |
| LMUL labels | lmul1, lmulmf2 |
| Unroll factors | unroll1, unroll2, unroll4, unroll8 |
| Benchmark driver | `ime_bench.c` |
| Reported metric | GOPS |

## Per-Variant Layout

```text
<ime_kernel_variant>/
+-- ime_kernel_*.c
+-- rvv_fallback.c
+-- ime_bench.c
+-- Makefile
```

## Build One Variant

```bash
cd <ime_kernel_variant>
make clean && make
./bench 1024 1024 1024
```

## Run On A K3 IME Core

```bash
bash -lc 'echo $$ > /proc/set_ai_thread && taskset -c 8 ./bench 1024 1024 1024'
```

The command above moves the launched process into the IME-capable CPU domain before pinning it to core 8. Use cores 8-15 for the K3 IME campaign when `/proc/set_ai_thread` is available.

## Dataflow Summary

- The IME wrapper receives caller matrices from the benchmark driver.
- Full tiles are packed into the layout required by the IME dot-product path.
- IME-capable cores execute the VMADOT path.
- If IME execution is unavailable, the local `rvv_fallback.c` implementation performs the matching INT8 RVV computation.
- Results are written as INT32 values in column-major layout.
