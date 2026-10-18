# RVV-IME INT8 Comparison

## Purpose

This directory provides the reproducibility drivers for the paper's fair end-to-end comparison of the RVV vector path with heterogeneous RVV-IME execution on the SpaceMiT K1.

## Experiment

The primary workload is fixed at `M=N=K=1024` with an `8x4` or `8x8` output tile, eight workers, and matched U1, U2, U4, and U8 kernel pairs at LMUL 1. The comparison measures RVV8, IME4, heterogeneous static 4-RVV plus 4-IME, and optional heterogeneous dynamic execution. All primary modes use the same input, validation, repetition count, and end-to-end timing boundary.

## Drivers and commands

Run from the `microkernels` repository root on the K1 Linux environment:

```bash
# Paper-level fair RVV–IME comparison
bash paper_scripts/heterogeneous_rvv_ime_end_to_end/main_driver/run_fig05_fair_end_to_end.sh

# Phase-aware fair comparison with optional extra workloads
bash paper_scripts/heterogeneous_rvv_ime_end_to_end/extra_driver/run_k1_fig06_fair_heterogeneous_phase_aware.sh --check
bash paper_scripts/heterogeneous_rvv_ime_end_to_end/extra_driver/run_k1_fig06_fair_heterogeneous_phase_aware.sh

# Fixed phase matrix
bash paper_scripts/heterogeneous_rvv_ime_end_to_end/extra_driver/run_k1_fig06_phase_complete.sh --check
bash paper_scripts/heterogeneous_rvv_ime_end_to_end/extra_driver/run_k1_fig06_phase_complete.sh
```

The drivers build the required native cases, validate every result, and write new outputs to their configured campaign directories. The `--check` commands validate source and benchmark contracts without compiling or running a case.

## Measurements

Primary end-to-end timing includes work assignment, input packing, computation, output or boundary handling, and synchronization. Allocation, reference generation, and validation are outside the timed region. Phase and counter passes are separate diagnostics; worker-summed phase values are not an additive wall-clock decomposition. RVV output stores remain fused with the RVV kernel, while IME output and boundary fields are reported when available.

## Data

- `../../datasets/fig05_rvv_ime_comparison/main_dataset/` contains cleaned comparison tables, matched statistics, raw runs, and metadata.
- `../../datasets/fig05_rvv_ime_comparison/extra_dataset/source_campaign/` contains the complete source campaign, case summaries, raw measurements, and logs.

Archived data are preserved as read-only evidence; new executions should use a new result directory.
