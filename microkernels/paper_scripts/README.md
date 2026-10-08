# Paper scripts

The launchers are grouped by purpose. The experiment commands are unchanged;
only their locations are organized.

```text
paper_scripts/
├── orchestration/       campaign planner and internal runners
├── single_core/         complete K1 single-core campaign
├── standalone_int8/     standalone INT8 8x4 and 8x8 campaigns
├── tools/               static completeness audit
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
bash paper_scripts/tools/check_project_completeness.sh
```

The audit checks every shell script under this directory, verifies delegated
workflow targets, and confirms that all eight paper entry points are present.

The root `paper_campaign.py` remains as a compatibility entry point for older
commands; new scripts should use the path under `orchestration/`.
