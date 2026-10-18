# Static and Dynamic Scheduling

## Purpose

This directory provides the reproducibility drivers for the paper's comparison of static and dynamic work assignment on the SpaceMiT K1.

## Experiment

The primary comparison uses a fixed `M=N=K=1024` INT8 GEMM workload, eight workers, four RVV workers, four IME workers, and `8x4` and `8x8` output tiles. Unroll factors U1, U2, U4, and U8 are compared with the same selected RVV/IME kernel pair under both scheduling policies. Static execution uses the fixed partition; dynamic execution uses the shared work queue. The archived complete source campaign contains six repetitions for each of 16 static/dynamic points.

## Drivers and commands

Run from the `microkernels` repository root on the K1 Linux environment:

```bash
# Paper-level scheduling campaign
bash paper_scripts/static_dynamic_scheduling/main_driver/run_fig03_static_vs_dynamic.sh

# K1 scheduling and partitioning driver
bash paper_scripts/static_dynamic_scheduling/extra_driver/run_k1_static_dynamic_scheduling.sh

# Phase-aware scheduling execution
bash paper_scripts/static_dynamic_scheduling/extra_driver/run_k1_fig04_scheduling_phase_aware.sh --check
bash paper_scripts/static_dynamic_scheduling/extra_driver/run_k1_fig04_scheduling_phase_aware.sh
```

The drivers build the required native cases, validate the results, and write new outputs to their configured campaign directories. The phase-aware `--check` command validates source and benchmark contracts without compiling or running a case.

## Measurements

Primary timing is end-to-end and includes packing, computation, output/boundary handling, work assignment, and synchronization. The phase-aware driver alternates static and dynamic order across repetitions, records strip ownership, and can add separate profile and counter passes. Worker-summed phase values are diagnostic and are not an additive wall-clock decomposition. Scheduling overhead is not inferred by subtracting phase values from total time.

## Data

- `../../datasets/fig03_static_dynamic/main_dataset/` contains the cleaned summary, raw runs, and metadata.
- `../../datasets/fig03_static_dynamic/extra_dataset/source_campaign/` contains the complete source campaign, case summaries, raw measurements, and logs.

Archived data are preserved as read-only evidence; new executions should use a new result directory.
