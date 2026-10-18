# Paper scripts

The launchers are grouped by purpose. The experiment commands are unchanged;
only their locations are organized.

```text
paper_scripts/
├── orchestration/       campaign planner and internal runners
├── extra/               legacy, standalone, audit, and aggregate utilities
├── strong_scaling_performance/
├── weak_scaling_performance/
├── static_dynamic_scheduling/
├── rvv_int8_tuning/
├── heterogeneous_rvv_ime_end_to_end/
├── fp32_fp64_comparison/
├── multicore_comparison/
└── correctness_validation/
```

The figure directories contain short, named entry points by design. They
delegate to the complete workflows in `orchestration/` or to the established
K1 module scripts; they are not standalone copies of the implementation. The
orchestration runners contain the campaign planning, build, execution, result
collection, and export logic.

Run the planner without building or measuring:

```bash
python3 paper_scripts/orchestration/paper_campaign.py --dry-run --no-perf
```

Check the organization and shell syntax before a board run:

```bash
bash paper_scripts/extra/tools/check_project_completeness.sh
```

The audit checks every shell script under this directory, verifies delegated
workflow targets, and confirms that all eight paper entry points are present.

The root `paper_campaign.py` remains as a compatibility entry point for older
commands; new scripts should use the path under `orchestration/`.

## Phase-aware Figure 3--9 entry points

The new complete Figure 3--9 workflows are stored in the existing
purpose-specific directories with the following names:

```text
weak_scaling_performance/run_k1_fig03_weak_scaling_phase_aware.sh
static_dynamic_scheduling/run_k1_fig04_scheduling_phase_aware.sh
rvv_int8_tuning/run_k1_fig05_int8_tuning_phase_aware.sh
heterogeneous_rvv_ime_end_to_end/run_k1_fig06_fair_heterogeneous_phase_aware.sh
fp32_fp64_comparison/run_k1_fig07_fp_baselines_phase_aware.sh
multicore_comparison/run_k1_fig08_summary_phase_aware.sh
correctness_validation/run_k1_fig09_correctness_phase_aware.sh
```

They use the shared new-benchmark source inventory, validation, CPU affinity,
end-to-end timing scope, phase diagnostics, hardware counters, provenance,
and failure-record conventions; the older launchers remain unchanged.

Run a read-only contract check on all seven workflows:

```bash
bash paper_scripts/extra/run_k1_figures_03_to_09_phase_aware.sh --check
```

Run the workflows sequentially under one campaign identity:

```bash
bash paper_scripts/extra/run_k1_figures_03_to_09_phase_aware.sh
```

Individual entry points can be run when only one figure is needed. Figure 8
is derived from accepted Figure 6 records and does not rerun a kernel.
