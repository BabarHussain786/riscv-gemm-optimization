# Strong-Scaling Performance

## Purpose

This directory provides the reproducibility drivers for the paper's fixed-workload INT8 GEMM strong-scaling study on the SpaceMiT K1. The drivers use the maintained benchmark and kernel sources in the project tree.

## Experiment

The workload is `M=N=K=1024` with signed INT8 inputs, INT32 accumulation, `C <- C + A*B`, and an `8x4` output tile. RVV uses 1, 2, 4, and 8 cores; IME uses 1, 2, and 4 cores. The archived primary data use static scheduling and seven accepted repetitions per point.

## Drivers and commands

Run from the `microkernels` repository root on the K1 Linux environment:

```bash
# Paper-level campaign
bash paper_scripts/strong_scaling_performance/main_driver/run_fig01_strong_scaling.sh

# K1 RVV and IME strong scaling
bash paper_scripts/strong_scaling_performance/extra_driver/run_k1_strong_scaling.sh

# Phase-aware execution
bash paper_scripts/strong_scaling_performance/extra_driver/run_k1_strong_scaling_phase_timing.sh --check
bash paper_scripts/strong_scaling_performance/extra_driver/run_k1_strong_scaling_phase_timing.sh
```

The drivers build the required native benchmark cases, run validation, and write new results to their configured output directories. The phase driver's `--check` option performs contract checks without compiling or running a benchmark.

## Phase measurements

The phase driver records an unprofiled end-to-end measurement, a separate profiled pass for packing and kernel/output/boundary phases, and an optional counter pass for cycles, instructions, IPC, and cache events. Profile values are worker-summed elapsed times and are not a wall-clock decomposition. RVV output and boundary handling remain fused with `kernel_sec`; IME output and boundary contributions are reported separately when available.

## Data

- `../../datasets/fig01_strong_scaling/main_dataset/` contains the primary CSV data, raw runs, summary, and metadata.
- `../../datasets/fig01_strong_scaling/extra_dataset/phase/` contains phase summaries, raw records, manifests, and logs.

Archived data are preserved as read-only evidence; new executions should use a new result directory.
