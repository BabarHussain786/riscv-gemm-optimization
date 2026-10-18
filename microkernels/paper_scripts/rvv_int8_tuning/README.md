# RVV INT8 Tuning

## Purpose

This directory provides the reproducibility drivers for the paper's RVV INT8 GEMM tuning study on the SpaceMiT K1.

## Experiment

The workload is fixed at `M=N=K=1024`. The study covers `8x4` and `8x8` output tiles, supported RVV LMUL values, and unroll factors U1, U2, U4, and U8. It measures 36 RVV and 8 IME configurations using prepacked and end-to-end timing, with independent numerical validation. The archived main summary contains 220 rows covering 44 kernels with six accepted repetitions per row.

## Drivers and commands

Run from the `microkernels` repository root on the K1 Linux environment:

```bash
# Paper-level RVV INT8 tuning campaign
bash paper_scripts/rvv_int8_tuning/main_driver/run_fig04_rvv_int8_tuning.sh

# Phase-aware tuning campaign
bash paper_scripts/rvv_int8_tuning/extra_driver/run_k1_fig05_int8_tuning_phase_aware.sh --check
bash paper_scripts/rvv_int8_tuning/extra_driver/run_k1_fig05_int8_tuning_phase_aware.sh
```

The drivers build the required native kernels, run validation, and write new results to their configured campaign directories. The phase-aware `--check` command validates the 36 RVV and 8 IME source configurations without compiling or running a case.

## Measurements

Prepacked timing excludes input packing but includes the required kernel-side updates and synchronization. End-to-end timing uses RVV eight-worker and IME four-worker execution and includes packing, computation, output handling, and synchronization. The phase-aware campaign also records held-out confirmations and writes kernel-selection files only after complete coverage and successful checks.

## Data

- `../../datasets/fig04_rvv_int8_tuning/main_dataset/` contains the cleaned summary, INT8 statistics, raw runs, and metadata.
- `../../datasets/fig04_rvv_int8_tuning/extra_dataset/source_campaign/` contains the complete source campaign, case summaries, raw measurements, and logs.

Archived data are preserved as read-only evidence; new executions should use a new result directory.
