# Reproducible K1 INT8 benchmarking

This is an **additive benchmark layer**, not a replacement kernel implementation.
It reuses the repository's canonical RVV and IME sources. Existing programs,
results, figures, and paper claims are not changed. New measurements must be run
on a SpaceMiT K1 with RVV VLEN=256, working IME support, and Linux/OpenMP.

**Status:** local host diagnostics are not K1 performance results. Read
`REPORT.md` for the checks actually completed and outstanding hardware checks.
The default kernel choices below are explicit starting configurations, not a
claim that they are optimal. Tune independently before selecting paper results.

## Start here

On the K1, open the repository's `microkernels` directory. The Linux checkout
may be anywhere; all commands below are relative to that directory. The source
checkout on Windows is `C:\Users\Public\New project\finalist\microkernels`.

Prerequisites: Python 3.9+, GCC with the RVV intrinsics used by the existing
kernels, RV64GCV/Zvl256b support, OpenMP development/runtime libraries, and Bash.
Optional: GDB for fault diagnostics and matplotlib for plotting.

```bash
python3 -m pip install -r benchmarking/requirements.txt
bash benchmarking/scripts/collect_system_info.sh
# Validate IME on a known candidate CPU BEFORE a long campaign.
bash benchmarking/scripts/diagnose_ime_runtime.sh --cpus 0 --threads 1 --m 16 --n 16 --k 64 --gdb
# Full baseline campaign (includes validation, both scopes, phase profiles,
# 8-core comparisons, repeated selected configuration, and counters).
bash benchmarking/scripts/run_paper_rvv_ime_campaign.sh --m 1024 --n 1024 --k 1024 --warmups 2 --repetitions 7
```

Do not run concurrent benchmark campaigns or unrelated CPU-intensive work on
the board. Stop other campaigns first; the scripts do not kill processes,
change the governor, grant perf permissions, or change system configuration.
Record the board's cooling, frequency policy, and background workload alongside
the automatic system metadata. The default campaign assumes CPU IDs 0--7 and
IME workers on 0--3; capability checks must succeed. Individual commands allow
another verified CPU list; never assume the numeric IDs prove capability.

## Build without changing the original build

Every build requires a new output directory. Use a compiler command appropriate
to the board (default `gcc`, or `--cc 'gcc-14'`, for example, if installed).

```bash
python3 benchmarking/build.py --output benchmarking/build_selected \
  --rvv-kernel igemm_kernel_8x4_zvl256b_lmulmf8_unroll2 \
  --ime-kernel ime_kernel_8x4_zvl256b_lmul1_unroll1
# Equivalent Make target, from microkernels/benchmarking:
# make BUILD_DIR=build_selected CC=gcc
```

RVV and IME tile, LMUL, and unroll selections are independent. There are 36
canonical RVV INT8 configurations and 8 canonical IME LMUL-1 configurations:

| Backend | Tile | LMUL | Unroll |
|---|---|---|---|
| RVV | 8x4 | mf8, mf4, mf2, 1, 2 | 1, 2, 4, 8 |
| RVV | 8x8 | mf4, mf2, 1, 2 | 1, 2, 4, 8 |
| IME | 8x4, 8x8 | 1 | 1, 2, 4, 8 |

The FP64 tree's mirrored INT8 files and experimental IME LMUL variants are not
silently counted as additional primary configurations. Floating-point kernels
are outside this INT8 measurement repair.

## Individual reproducible experiments

`run.py` builds each case out of tree, validates it, executes it, and writes
machine-readable results. Append the same `--rvv-kernel` and `--ime-kernel`
options to every related comparison after tuning. The default shape is 1024^3.

```bash
# RVV and IME, both scopes, matched CPU and input seed.
python3 benchmarking/run.py run --implementation rvv --timing prepacked --cpus 0
python3 benchmarking/run.py run --implementation ime --timing prepacked --cpus 0
python3 benchmarking/run.py run --implementation rvv --timing end_to_end --cpus 0
python3 benchmarking/run.py run --implementation ime --timing end_to_end --cpus 0

# Separate instrumented component measurements; do not substitute their total
# elapsed times for the uninstrumented primary comparison.
python3 benchmarking/run.py run --implementation rvv --timing end_to_end --profile --cpus 0
python3 benchmarking/run.py run --implementation ime --timing end_to_end --profile --cpus 0

# Validation without publishing performance data.
python3 benchmarking/run.py diagnose --implementation rvv --m 17 --n 67 --k 69 --cpus 0
python3 benchmarking/run.py diagnose --implementation ime --m 17 --n 67 --k 69 --cpus 0 --gdb
# Direct already-built executable, validation only:
benchmarking/build_selected/bench --implementation rvv --validate-only 1 --cpus 0

# Central comparison: same original inputs, selected RVV worker kernel, total
# number of workers, datatype, and end-to-end scope.
python3 benchmarking/run.py run --implementation rvv --threads 8 --cpus 0,1,2,3,4,5,6,7 --timing end_to_end
python3 benchmarking/run.py run --implementation mixed --threads 8 --ime-workers 4 --cpus 0,1,2,3,4,5,6,7 --schedule static --weight 4 --timing end_to_end
python3 benchmarking/run.py run --implementation mixed --threads 8 --ime-workers 4 --cpus 0,1,2,3,4,5,6,7 --schedule dynamic --chunk 1 --timing end_to_end

# Fig. 5 timing launcher: aligned and full-workload validation are retained;
# the 15x15x69 tail check belongs to the separate correctness campaign.
bash benchmarking/scripts/run_fig05_complete.sh --m 1024 --n 1024 --k 1024 \
  --warmups 2 --repetitions 7

# Add this only when intentionally running the auxiliary tail check as part of
# a timing case (it is not needed for the Fig. 5 performance comparison).
# .../run_fig05_complete.sh --include-boundary-validation

# Repeatability of ONE selected configuration, not a pool of tuning variants.
bash benchmarking/scripts/run_rvv_ime_repeatability.sh --implementation rvv --repeat-runs 3 --repetitions 15 --warmups 3
# Independent RVV and IME configuration exploration (default one CPU).
bash benchmarking/scripts/run_rvv_ime_tuning.sh --repetitions 7 --warmups 2
# Include exploration in a full campaign if desired.
bash benchmarking/scripts/run_paper_rvv_ime_campaign.sh --include-tuning

# Region counters: instrumented explanatory data, not primary performance.
python3 benchmarking/run.py run --implementation rvv --threads 8 --cpus 0,1,2,3,4,5,6,7 --counters
python3 benchmarking/run.py run --implementation mixed --threads 8 --ime-workers 4 --cpus 0,1,2,3,4,5,6,7 --schedule dynamic --counters
```

Each worker is pinned with `sched_setaffinity`, and its actual CPU is checked
before and after the measured work. CPU IDs must be distinct and permitted by
the launching process. In mixed mode the **first four listed CPUs** are IME
workers, not necessarily CPU IDs 0--3 if another list is supplied. IME checks
the existing `cpu-ai` device-tree marker and exact VLEN=256/A60 profile. A
missing marker, unsupported ISA, affinity failure, or SIGILL is a failure,
never a silent fallback to RVV. A marker alone is not proof: validation must
also execute successfully. Capability detection remains conservative and
depends on the board's device-tree exposure.

Static scheduling partitions 32-column output strips with a configurable
IME:RVV work ratio (`--weight 4` means 4:1), then distributes each partition
round-robin within its worker group. This is a configurable scheduling choice,
not a measured optimal ratio. Dynamic scheduling uses a shared atomic strip
queue (`--chunk 1` by default). Per-worker strip counts are recorded. These
are new reproducible schedules around the original kernels, not claims of
bit-for-bit reproduction of an old scheduler's timing.

## Timing contract

Both paths compute the same bounded INT8/INT32 update, with alpha=1 and the
same initialized C. Every repetition resets C outside the timer.

| Mode | Before timer | Included in total elapsed time |
|---|---|---|
| prepacked | Allocate and pack inputs | Scheduling, kernel, required output update/scatter, tails, synchronization |
| end_to_end | Allocate workspace; inputs already initialized | Scheduling, input packing, kernel, required output update/scatter, tails, synchronization |

Initialization, memory allocation/free, INT64 reference construction, result
validation, OpenMP team creation, CPU pinning, and capability checks are outside
the reported elapsed region for both paths. End-to-end here means the **data
transformation/execution pipeline**, not process startup or allocation latency.
Wall time starts on the master before the start barrier and stops after the
workers' final barrier. This includes synchronization and completion of all
workers, not a sum of worker times.

RVV's C update and tail handling remain fused in its existing kernel. Their
separate times are **null**, not zero; `kernel_sec` includes them. IME uses the
actual per-tile `scatter_output()` between native accumulation and scalar tail
cleanup; it is not described as a whole-matrix transpose.

`--profile` measures component durations. `phase_aggregation=sum_worker_elapsed`
means the sum of worker durations, **not wall-clock decomposition**. Multicore
phase totals can exceed elapsed time; phase percentages are intentionally
reported only for single-thread profiled runs. They need not sum to 100% because
scheduling, synchronization, and timing overhead remain outside phase buckets.
Mixed output fields report only the IME contribution; complete mixed output
cost cannot be separated from the RVV kernel.

Prepacked mixed execution prepares both layouts before timing because dynamic
strip ownership is unknown. End-to-end packs only the assigned path. The new
adapter retains packed panels across execution instead of the original IME
wrapper's allocate/pack-one-B-panel/execute/free lifecycle. Both paths use
preallocated workspaces, and these cache/workspace policies are part of this
new protocol; they must not be presented as the old wrapper timings.

`--counters` uses per-thread `perf_event_open` groups around each worker's
scheduled work. Reference construction, warm-ups and validation are not part
of the **reported** counter records. Counts include user-space assignment,
packing (end-to-end only), compute and output work. They exclude the final
waiting barrier, while wall time includes it. The counter-enable/disable/read
overhead is inside instrumented total time. Therefore counter and phase-profile
runs are excluded from primary timing comparisons/speedups by `analyze.py`.
Unavailable, denied, or multiplexed event groups yield null counters. IPC is
sum(instructions)/sum(cycles), not instructions per MAC or elapsed-time speedup.

## Correctness and acceptance

The driver validates aligned 16x16x64, tail 15x15x69, and the requested full
problem against an independently calculated INT64 reference. Mixed tests also
exercise both backends separately on aligned and tail shapes, since a tiny
mixed problem can otherwise run on just one backend. Each measured output and
warm-up is checked outside timing. Input ranges avoid INT32 overflow; reference
overflow is rejected. This is exact integer validation, not a floating-point
accuracy study or exhaustive proof over the full INT8 input domain.

The runner requires successful validation logs, exact repetition counts,
consistent metadata, positive finite elapsed time, and GOPS equal to
`2*M*N*K/(total_sec*1e9)` for each run. Any bad row or nonzero process exit rejects
the **whole case**, including earlier printed rows. Raw/failed diagnostics are
retained. No hardware validation means no accepted hardware measurements.

## Output and analysis

Each invocation gets a unique UTC timestamp/UUID folder under
`benchmarking/results` (or `--output`). Old campaigns are never overwritten.
Campaign groups separate validation, prepacked, end_to_end, eight_core_rvv,
eight_core_heterogeneous_static, eight_core_heterogeneous_dynamic, tuning,
repeatability and counters. Each executed case has config, command, system,
build/hash, validation and stdout/stderr records plus:

- `raw_all_runs.csv`: emitted observations with acceptance/exclusion status.
- `accepted_runs.csv`: validated native observations only; instrumentation
  flags still distinguish primary timing from profile/counter observations.
- `failed_runs.csv`: failed cases and deliberately excluded host diagnostics.

The campaign CSVs consolidate its cases. Empty categories are not results.
`diagnose` and `--host-test` never publish timing samples. An interrupted case
does not gain acceptance; preserve its logs and rerun in a new directory.

```bash
# Replace the path with the unique directory printed by the launcher.
python3 benchmarking/analyze.py --input benchmarking/results/campaign_TIMESTAMP_ID \
  --output benchmarking/results/analysis_TIMESTAMP_ID
```

Only accepted raw CSV input is supported. Analysis revalidates the schema,
rejects host/failure records, keeps complete configurations separate, reports
mean/median/sample standard deviation of time and independently derived GOPS,
and matches speedup baselines only across equivalent scopes/configurations.
Tables include component timing, single-thread phase shares, and matched RVV
speedup. Plots cover available end-to-end/prepacked comparisons, components,
8 RVV vs 4+4 static/dynamic, tuning and fixed-configuration repeatability.
Missing or ambiguous matched data produce no invented speedup/figure. Use the
analysis rejection report to resolve omissions rather than manually importing
old unmatched data. Never call host or synthetic test plots paper results.

## Unsupported machines and regression tests

On Windows/x86 the native runner must fail early, with no accepted samples.
The scalar backend exists only to test the infrastructure, not to emulate IME.
For Windows host diagnostics use a short `--output` path: deeply nested campaign
paths can exceed Windows path limits. The native target platform is Linux.

```bash
python3 benchmarking/run.py diagnose --implementation ime --m 16 --n 16 --k 64
python3 benchmarking/build.py --host-test --output benchmarking/build_host --cc cc
BENCH_TEST_BINARY="$PWD/benchmarking/build_host/bench" python3 -m unittest discover -s benchmarking/tests -v
python3 benchmarking/run.py run --host-test --m 17 --n 67 --k 69 --repetitions 3
```

The last command's reference measurements remain diagnostic-only. The test
suite uses synthetic fixtures solely to test rejection/aggregation/rendering;
they are not experimental results. See `TECHNICAL_FLOW.md` for source-level
packing/layout details and `REPORT.md` for the saved IME exit-132 investigation.
