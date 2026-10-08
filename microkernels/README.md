# RVV--IME GEMM Microkernels on the SpacemiT K1

This repository contains the K1 source kernels, benchmark workflows, measured
datasets, and analysis assets used for the RVV--IME GEMM paper. The paper
targets a 256-bit RVV implementation with INT8 inputs, INT32 accumulation, and
native IME execution on the K1.

## Project order

The repository is organized in the order used by the paper:

1. **Kernel sources** — RVV FP32, FP64, and INT8 families, plus native IME
   wrappers.
2. **K1 execution module** — OpenMP worker placement, packing, tile execution,
   scheduling, and output formation.
3. **Paper scripts** — reproducible entry points for the eight paper figures.
4. **Accuracy checker** — independent INT64-reference validation for INT8 RVV
   and IME outputs.
5. **Measured datasets** — clean summaries, raw runs, metadata, and campaign
   provenance organized by figure.
6. **Analysis and plots** — scripts in the OpenMP module consume measured
   exports; no synthetic values are generated.

## K1 experiment sequence

Run the paper experiments in this order when collecting a new K1 campaign:

For one command that dispatches all eight figures through the shared planner,
use [`run_all_paper_figures.sh`](run_all_paper_figures.sh). Individual figure
entry points remain available for selective or resumed campaigns.

| Paper figure | Experiment | Entry point | Dataset folder |
|---|---|---|---|
| Fig. 1 | Strong scaling | [`run_fig01_strong_scaling.sh`](paper_scripts/strong_scaling_performance/run_fig01_strong_scaling.sh) | [`datasets/fig01_strong_scaling/`](datasets/fig01_strong_scaling/) |
| Fig. 2 | Weak scaling | [`run_fig02_weak_scaling.sh`](paper_scripts/weak_scaling_performance/run_fig02_weak_scaling.sh) | [`datasets/fig02_weak_scaling/`](datasets/fig02_weak_scaling/) |
| Fig. 3 | Static versus dynamic scheduling | [`run_fig03_static_vs_dynamic.sh`](paper_scripts/static_dynamic_scheduling/run_fig03_static_vs_dynamic.sh) | [`datasets/fig03_static_dynamic/`](datasets/fig03_static_dynamic/) |
| Fig. 4 | RVV INT8 tuning | [`run_fig04_rvv_int8_tuning.sh`](paper_scripts/rvv_int8_tuning/run_fig04_rvv_int8_tuning.sh) | [`datasets/fig04_rvv_int8_tuning/`](datasets/fig04_rvv_int8_tuning/) |
| Fig. 5 | RVV--IME INT8 comparison | [`run_fig05_fair_end_to_end.sh`](paper_scripts/heterogeneous_rvv_ime_end_to_end/run_fig05_fair_end_to_end.sh) | [`datasets/fig05_rvv_ime_comparison/`](datasets/fig05_rvv_ime_comparison/) |
| Fig. 6 | FP32 and FP64 reference measurements | [`run_fig06_rvv_fp32_fp64.sh`](paper_scripts/fp32_fp64_comparison/run_fig06_rvv_fp32_fp64.sh) | [`datasets/fig06_fp32_fp64/`](datasets/fig06_fp32_fp64/) |
| Fig. 7 | Multicore and heterogeneous execution | [`run_fig07_multicore_comparison.sh`](paper_scripts/multicore_comparison/run_fig07_multicore_comparison.sh) | [`datasets/fig07_multicore_heterogeneous/`](datasets/fig07_multicore_heterogeneous/) |
| Fig. 8 | Independent correctness validation | [`run_fig08_correctness.sh`](paper_scripts/correctness_validation/run_fig08_correctness.sh) | [`datasets/fig08_correctness/`](datasets/fig08_correctness/) |

The figure entry points are intentionally short. Complete campaign logic is in
[`paper_scripts/orchestration/`](paper_scripts/orchestration/) and the K1
OpenMP module; the entry points select the figure-specific experiment without
duplicating the implementation.

## Repository layout

```text
microkernels/
├── GEMM_RVV_FP32_INT8_8x4_Baseline/   RVV INT8 and FP32 tile family
├── GEMM_RVV_FP32_INT8_8x8_Baseline/   RVV INT8 and FP32 tile family
├── GEMM_RVV_FP64_INT8_8x4_Baseline/   RVV FP64/INT8 reference family
├── GEMM_RVV_FP64_INT8_8x8_Baseline/   RVV FP64/INT8 reference family
├── IME_NATIVE_KERNELS/                 Native K1 IME wrappers and kernels
├── HETEROGENEOUS_RVV_IME_OPENMP_GEMM/  K1 OpenMP module and analysis tools
├── RVV_IME_GEMM_ACCURACY_VALIDATION/   Independent INT64 reference checker
├── paper_scripts/                      Paper campaign entry points and runners
├── datasets/                           Figure-organized measured exports
└── benchmarking/                       Verification records and campaign support
```

## K1 execution model

The combined K1 configuration uses four IME workers on cores 0--3 and four RVV
workers on cores 4--7. Output-column strips use `tile_N=32`; each strip has one
owner. The timed end-to-end path includes input packing, tile-kernel execution,
scheduling, and output formation. Allocation, initialization, warm-up,
validation, and cleanup are outside that timing scope.

The source kernel families use 256-bit RVV (`zvl256b`) and the paper tile shapes
`8x4` and `8x8`. Kernel directories encode the LMUL and reduction-loop unroll
factor in their names. Native IME calls use the K1 IME path; they do not replace
the RVV kernels or change worker affinity.

## Preliminary checks

Run these checks from the repository root before a board campaign:

```bash
bash paper_scripts/tools/check_project_completeness.sh
python3 paper_scripts/orchestration/paper_campaign.py --dry-run --no-perf
```

The first command checks all shell scripts, delegated workflow targets, figure
entry points, and dataset exports. The second command creates plans only; it
does not build kernels or run hardware measurements.

To run all eight paper figures through one command on the K1, use:

```bash
bash run_all_paper_figures.sh
```

The same command can create a complete plan without touching the hardware:

```bash
bash run_all_paper_figures.sh --dry-run --no-perf
```

## Accuracy validation

The independent checker is kept at
`RVV_IME_GEMM_ACCURACY_VALIDATION/`. It compares native IME and RVV INT8
outputs against an INT64 accumulation reference and reports exact-match,
overflow, mismatch, and error-histogram information. Its paper entry point is
the Fig. 8 script listed above. The source checker remains separate from
`datasets/fig08_correctness/`, which is reserved for its measured outputs.

## Dataset policy

The figure dataset index is [`datasets/README.md`](datasets/README.md), and the
machine-readable inventory is [`datasets/MANIFEST.csv`](datasets/MANIFEST.csv).
Within an available figure export:

```text
clean/       summaries and statistics used for analysis
raw/         per-run records
metadata/    logs, manifests, failure records, and campaign plans
```

The manifest distinguishes complete, archived, partial, and pending datasets.
Missing K1 measurements are documented rather than inferred from plots or
replaced with synthetic values. `paper_results/` is optional planner output and
is not required to build the project or use the measured datasets.

## Requirements

- SpacemiT K1 board with RVV and IME support.
- RISC-V Linux, GCC with RVV intrinsics and OpenMP, GNU Make, Bash, and
  `taskset`.
- A 256-bit RVV implementation (`zvl256b`).
- IME-capable K1 cores for native IME experiments.

All benchmark commands must be run on the K1 for actual measurements. Local
dry-runs validate campaign planning and file organization only.
