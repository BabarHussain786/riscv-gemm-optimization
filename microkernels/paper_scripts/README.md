# K1 paper data collection

One folder contains the master launcher, eight figure launchers, three adapted
existing measurement runners, the campaign planner, and host-only tests.
Kernel sources remain in their existing directories. Python 3.9+, Bash, Make,
taskset, and an RVV-capable GCC are required. Measurements require K1 Linux.
`perf` is optional; unavailable counters must not be interpreted as zero IPC.

## Start with a plan

From the project root:

```bash
bash paper_scripts/run_all.sh --dry-run --output paper_results/k1_plan
```

This writes the complete case list and the available/missing configuration
inventory without compiling or running anything. The default sweep covers
every available nonexperimental canonical LMUL with U1, U2, U4, U8 and both
8x4/8x8 tiles. Inspect the printed case counts: the full sweep can take a long
time and recompiles individual cases. The current inventory gives 108
nonexperimental kernel variants and 700 default cases. Start with one figure
if needed. A case contains multiple repetitions; 700 is not the run count.

## Run everything on K1

```bash
bash paper_scripts/run_all.sh --output paper_results/k1_run_01
```

Run sequentially on an otherwise idle board. Do not run other benchmark
launchers concurrently. This suite locks its source root because standalone
runners use `make clean` and build in the original kernel folders. It does
not change the board frequency governor or install system packages.

Use the same options to resume a planned or interrupted campaign:

```bash
bash paper_scripts/run_all.sh --output paper_results/k1_run_01 --resume
```

Completed cases with recorded data are skipped. Failed/interrupted cases get
a new attempt directory; old attempts remain. Source or option changes refuse
resume, so different configurations cannot silently share one dataset.
After a power loss, inspect `.paper_campaign.lock/owner.json`; remove that lock
directory manually only after confirming its campaign process is no longer running.

## One entry script for each figure

| Figure/data folder | Launcher | Measurement |
|---|---|---|
| `fig01_strong_scaling` | `run_fig01_strong_scaling.sh` | Fixed size; RVV 1/2/4/8 threads and IME 1/2/4 for each kernel |
| `fig02_weak_scaling` | `run_fig02_weak_scaling.sh` | Paper dimensions 512/672/832/1024 for 1/2/4/8 threads |
| `fig03_static_vs_dynamic` | `run_fig03_static_vs_dynamic.sh` | Both policies, 8 threads, all matched IME/RVV variants |
| `fig04_rvv_int8_tuning` | `run_fig04_rvv_int8_tuning.sh` | Standalone canonical RVV INT8 LMUL/unroll inventory |
| `fig05_rvv_vs_ime_int8` | `run_fig05_rvv_vs_ime_int8.sh` | Standalone RVV and IME inventories; keep backends/configurations separate |
| `fig06_rvv_fp32_fp64` | `run_fig06_rvv_fp32_fp64.sh` | Standalone FP32 and FP64 inventory |
| `fig07_rvv_multicore_vs_heterogeneous` | `run_fig07_multicore_comparison.sh` | RVV 8-thread baseline plus 8-thread mixed static/dynamic cases |
| `fig08_correctness` | `run_fig08_correctness.sh` | Canonical INT8 OpenMP validation plus IME/fallback reference tests |

For example:

```bash
bash paper_scripts/run_fig04_rvv_int8_tuning.sh --output paper_results/int8_tuning
```

The figure wrappers fix their figure selection; all other options are shared.

## Collect more data

```bash
bash paper_scripts/run_all.sh \
  --sizes 256,512,1024,2048 --runs 10 \
  --rvv-cores '0 1 2 3 4 5 6 7' --ime-cores '0 1 2 3' \
  --weights 1:1,2:1,4:1,8:1 --chunks 1,2,4 \
  --accuracy-runs 5 --accuracy-shapes 15x15x69,64x64x64,1024x1024x1024 \
  --output paper_results/k1_extended
```

- Performance defaults: size 1024, six repetitions, standalone RVV core 4 and
  IME core 0, strip width 32, static weights 4:1, dynamic chunk 1.
- Standalone core options do not change the established OpenMP core groups.
- `--weak-dimensions 512,672,832,1024` records the paper's approximate weak
  scaling explicitly. It does not claim exactly constant work per core.
- Accuracy defaults: three repetitions, two shapes, four existing input
  classes (`bounded_uniform`, `full_range_uniform`, `mixed_magnitude`,
  `cancellation_stress`) for the IME/fallback harness. Canonical OpenMP INT8
  checks use their existing deterministic input generator; no new classes are invented.
- `--experimental-ime` includes available IME LMUL mf2 variants. They remain
  flagged experimental and require successful validation. They are not silently
  included in the primary default campaign.
- `--no-perf` disables OpenMP perf collection. Standalone runners retain their
  existing timing/throughput fields and do not collect IPC. `--warmups` applies
  to OpenMP; standalone runners retain their own existing behavior.
- `--cc` selects the compiler. The existing runner/Makefile compilation flags
  are retained; commands, build logs, compiler version and source hashes are saved.

## Results

```text
paper_results/k1_run_01/
  manifest.json                 exact configuration, cases and source hashes
  coverage.json                 available, missing and excluded combinations
  host_*.json                   host/compiler/affinity provenance
  fig01_strong_scaling/
    plan.json
    case_statuses.json
    raw_data.csv                unmodified raw fields plus plan metadata
    summary.csv                 successful time samples, grouped per case/core
    cases/<case-id>/
      status.json
      attempt_<timestamp>/
        command.json            exact command and controlled environment
        console.log
        ... original raw CSV, summary, build and validation logs ...
  fig02_weak_scaling/
  ... fig03 through fig08 ...
```

No raw results are produced by dry-run mode. Aggregate CSVs are produced when
execution finishes or is interrupted. Failures stay in raw data and status
files; the master returns a nonzero status if cases fail. Empty output is not
success. Summary SD is sample standard deviation (`n-1`); a single sample has
no reported SD. Failed cases can contain some successful samples: inspect
`case_status` before using any summary. Correctness-only cases may have no
timing summary. Do not move a completed results directory before resolving
the absolute raw-log paths recorded by the original runners.

## Scientific boundaries

This collects NEW data. It does not claim to reproduce the old figure values.
INT8 LMUL 4/8 are absent/excluded, FP64 8x4 LMUL 1/2 and 1 are absent, and
FP32 8x8 LMUL 1/2 is absent/excluded. These appear in `coverage.json`, never
as fabricated zero measurements. LMUL labels are source-directory input LMUL,
not accumulator EMUL. Requested unroll values do not prove compiler unrolling.

Standalone RVV and IME wrappers have different packing/timing boundaries.
Figure 5 data must retain that qualification. Figure 7 supplies matching-size,
matching-total-thread-count OpenMP cases; select matched tile/LMUL/unroll cases
and report the scheduling policy when computing comparisons. Scaling must
hold the selected kernel fixed across core counts. The suite never selects
the fastest variant silently, pools unlike configurations, or calculates an
unqualified speedup. Plot rendering is intentionally separate from collection;
the existing paper figure files are untouched.

## Reused code and testing

`runner_standalone.sh` is adapted from `run_k1_01_rvv_ime_0_7_1024.sh`;
`runner_openmp.sh` from the OpenMP module's `run_openmp_tiled_gemm_mode.sh`;
`runner_accuracy.sh` from `run_int8_ime_vs_rvv_accuracy_once.sh`. Changes are
limited to project/output paths, exact kernel filtering, explicit compiler
selection for Make, and failure reporting.
The original scripts and C sources are unchanged.

```bash
python3 -m unittest discover -s paper_scripts -p 'test_*.py' -v
```

Host tests verify planning, configuration coverage, failure reporting,
environment isolation, CSV handling and resume protection. They do not prove
hardware correctness, counter availability or performance on K1.
