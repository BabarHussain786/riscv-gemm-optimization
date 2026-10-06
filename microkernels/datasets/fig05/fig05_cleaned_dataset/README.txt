Figure 5 cleaned dataset
========================
Source: fig05_profile_20261002T134311Z/fig05_accepted_runs.csv
Scope: INT8 GEMM, 1024 x 1024 x 1024, 8-core K1 campaign
Configurations: 3 comparison modes x 2 tile shapes (8x4, 8x8) x 4 unroll factors (U1, U2, U4, U8) = 24
Runs: 7 primary total-time runs + 7 profiled phase-timing runs per configuration = 336 accepted records
Quality: all 336 records have status=OK, validation=PASS, accepted=True, publishable=True.
Files:
- fig05_cleaned_dataset.csv: one merged row per configuration; use this for plots/tables.
- fig05_cleaned_runs.csv: normalized run-level records for audit/reproducibility.
Timing:
- primary_total_*: primary end-to-end measurements.
- profiled_input_packing_* and profiled_kernel_*: profiled phase measurements for RVV, static heterogeneous, and dynamic heterogeneous paths.
- profiled_output_packing_* is blank because the current driver does not report a separate output transpose/reshape timer. Blank means not reported, not zero.
- phase_aggregation=sum_worker_elapsed; profiled phase times are aggregate worker-region times and must not be stacked directly as wall-clock components.
- Output packing/reshape is shown in the workflow description but is not independently timed by the current driver.
