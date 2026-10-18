# Weak-Scaling Performance

## Purpose

This directory provides the reproducibility drivers for the paper's INT8 GEMM weak-scaling study on the SpaceMiT K1. The workload grows with the number of active cores so that each core receives approximately the same amount of work.

## Experiment

The archived complete campaign uses rounded cubic workloads `512^3`, `672^3`, `832^3`, and `1024^3` for 1, 2, 4, and 8 cores. RVV uses all four core counts; IME uses 1, 2, and 4 cores. The campaign covers the `8x4` and `8x8` tiles across the recorded LMUL and unroll configurations, with static scheduling and independent validation.

## Drivers and commands

Run from the `microkernels` repository root on the K1 Linux environment:

```bash
# Paper-level weak-scaling campaign
bash paper_scripts/weak_scaling_performance/main_driver/run_fig02_weak_scaling.sh

# K1 weak-scaling driver
bash paper_scripts/weak_scaling_performance/extra_driver/run_k1_weak_scaling.sh

# Phase-aware weak-scaling execution
bash paper_scripts/weak_scaling_performance/extra_driver/run_k1_fig03_weak_scaling_phase_aware.sh --check
bash paper_scripts/weak_scaling_performance/extra_driver/run_k1_fig03_weak_scaling_phase_aware.sh
```

The drivers build the required native cases, run numerical validation, and write new results to their configured output directories. The phase-aware `--check` command validates the source inventory and benchmark contract without compiling or running a case.

## Weak-scaling and phase records

The primary results measure end-to-end execution, including work assignment, packing, computation, output handling, and synchronization. The phase-aware driver also records separate counter diagnostics and phase observations. Its adjusted ideal time uses the actual rounded matrix dimensions, rather than assuming perfectly constant work per core. Phase and counter values are diagnostic records and do not replace the primary timing measurement.

## Data

- `../../datasets/fig02_weak_scaling/main_dataset/` contains the cleaned summary, raw runs, and campaign metadata.
- `../../datasets/fig02_weak_scaling/extra_dataset/source_campaign/` contains the complete source campaign, including case summaries, raw measurements, and logs.

Archived data are preserved as read-only evidence; new executions should use a new result directory.
