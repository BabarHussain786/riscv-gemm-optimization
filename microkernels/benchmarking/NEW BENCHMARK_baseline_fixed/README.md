# NEW BENCHMARK — additive SpaceMiT K1 paper datasets

This folder is a separate shell-only solution. It reads the existing sources;
it does not rewrite kernels, old runners, datasets, figures, or the paper.
Each figure folder contains exactly one `.sh` entry point. `common.sh` shares
build/validation/statistics helpers; `run_all.sh` runs the complete campaign.
No hardware results are supplied or assumed here. The scripts generate datasets,
not new plot images, after execution on the actual K1 Linux board.

## Run

Copy the **whole microkernels project**, including this folder, to the K1 board.
Requirements: Bash 4+, a RISC-V GCC with RVV intrinsics and OpenMP, `jq` 1.6+,
`awk`, `sort`, `sha256sum`, `taskset`, and standard Linux core utilities.
No Python benchmark orchestrator or new C measurement engine is introduced.

```sh
cd '/path/to/microkernels/NEW BENCHMARK'
bash run_all.sh --check
CAMPAIGN=k1_paper_001 RUNS=7 WARMUPS=2 SEED=42 bash run_all.sh
```

`--check` is read-only and works without K1 hardware. It checks source availability
and shell syntax, **not** native compilation, correctness or measured performance.
Default CPU lists are IME `0,1,2,3` and RVV `4,5,6,7`; override these if the actual
board mapping differs. They must contain four unique IDs each without overlap.
Preflight executes aligned native IME validation on **every** selected IME CPU
and RVV validation on every used CPU. Missing CPU-AI/A60 support, wrong VLEN,
unsupported affinity, illegal instructions, or numerical failures are rejected.
Do not externally pin the whole multicore process to one CPU.

```sh
export CAMPAIGN=k1_paper_002
export IME_CPUS=0,1,2,3 RVV_CPUS=4,5,6,7 FP_CPU=4
bash 00_preflight.sh
bash Figure_05_INT8_Kernel_Tuning/fig05_int8_tuning.sh
bash Figure_06_Fair_RVV_IME_Heterogeneous_Comparison/fig06_fair_comparison.sh
```

Keep the same exported CAMPAIGN/settings for dependent individual runners.
Figure 5 must precede scaling, scheduling and comparison; Figure 6 precedes
Figure 8. A failed/partial tuning stage does not export usable selections.
Every figure creates a **new** output directory; existing output is never
overwritten. Use a new campaign to rerun a figure. There is no destructive cleanup.

## Figure folders and datasets

| Paper figure | Shell entry point | Dataset and scope |
|---|---|---|
| 1: Architecture | `Figure_01_RVV_IME_Architecture/fig01_architecture.sh` | Source anchors/hashes, architecture TSV/JSON; no hardware benchmark |
| 2: Strong scaling | `Figure_02_Strong_Scaling/fig02_strong_scaling.sh` | Fixed 1024³; RVV 1/2/4/8, IME 1/2/4; speedup/efficiency; separate counters |
| 3: Weak scaling | `Figure_03_Weak_Scaling/fig03_weak_scaling.sh` | Actual cubes 512/672/832/1024; work/core and workload-corrected ideal time |
| 4: Scheduling | `Figure_04_Static_Dynamic_Scheduling/fig04_scheduling.sh` | Both tiles, U1/2/4/8, alternating paired static/dynamic measurements; selected-only sensitivity |
| 5: INT8 tuning | `Figure_05_INT8_Kernel_Tuning/fig05_int8_tuning.sh` | All 36 RVV + 8 IME configurations, one-core prepacked and multicore end-to-end, independent selections and held-out checks |
| 6: Fair comparison | `Figure_06_Fair_RVV_IME_Heterogeneous_Comparison/fig06_fair_comparison.sh` | Same-tile controls plus independently best tile pair when distinct; RVV8, IME4, static4+4, dynamic4+4; three workloads and separate diagnostics |
| 7: FP baselines | `Figure_07_FP32_FP64_Baselines/fig07_fp_baselines.sh` | 36 FP32 + 28 FP64 configurations; fixed core; numerical gates, per-configuration quartiles/IQR/outliers |
| 8: Heterogeneous summary | `Figure_08_Heterogeneous_Summary/fig08_summary.sh` | Strict offline derivation from one Figure 6 campaign/pair; no independent benchmark |
| 9: Correctness | `Figure_09_INT8_Correctness/fig09_correctness.sh` | All 44 actual INT8 kernels; four input classes, three seeds, aligned/tail shapes; tiny native classifications and selected full-workload validation |

Outputs are placed under `results/<CAMPAIGN>/<figure-id>/`, with raw logs,
accepted records, summaries, metadata and explicit failures. Builds are cached
under that campaign's `_build/`; existing source directories remain untouched.
INT8 raw JSONL retains CPU/worker strip counts, seed, timing scope, kernel names,
LMUL/U, source/binary hashes, warm-ups, counters and measured phase fields.
Summaries keep timing scopes and primary/profile/counter roles separate.
Statistics use arithmetic mean, sample SD and interpolated (Type-7) quantiles.
FP outliers are identified by 1.5×IQR but never silently deleted.

The orchestrator runs preflight, a separate 44-kernel smoke-validation campaign,
architecture tracing, complete INT8 tuning, fair comparison, scheduling,
scaling, FP baselines, expanded correctness and the derived summary. Smoke
counts are **not** added to final Figure 9 denominators.

## Measurement boundaries and scientific limitations

- The unchanged common INT8 C engine uses a flat team: static backend partition
  and cyclic strips, or a dynamic atomic queue. It is **not** the old nested
  static/OpenMP-dynamic engine. Do not combine their times or describe them as
  one implementation.
- End-to-end includes assignment, strip-level input packing, computation,
  required C/output operations, applicable tails and start/completion barriers.
  Workspace allocation, team creation, affinity, references and validation are
  outside the existing timer. It is not whole-process application latency.
- Packing follows the current adapters: A can be repacked per assigned
  32-column strip. This is not a claim of global pack-once OpenBLAS packing.
- Prepacked excludes input packing but still includes output/tails and
  scheduling/synchronization. It is not instruction-only computation time.
- RVV stores/C updates remain fused. Their independent output time is
  unavailable, not zero. IME performs scatter after each native tile, not one
  global transpose at completion.
- Profiled phase times are **summed worker durations**, not an additive
  wall-clock breakdown. Profiling overhead can be substantial; headline times
  always use profile0/counters0. Independent scheduler-overhead timing is
  unavailable and is never inferred by subtracting worker sums.
- Counters cover worker measured regions. All four requested events must be
  available without multiplexing; otherwise the existing C driver emits null
  counters. This does not invalidate uninstrumented times, but supplies no IPC
  evidence. IPC is instructions/cycles, not MAC utilization.
- FP standalone programs time one kernel call on deterministic modulo-13 inputs.
  They pass aligned, tail and full-shape numerical gates before measurement.
  Subsequent timing processes reference that full-shape gate; validation is not
  repeated inside each measured process. Discarded launches are **fresh
  processes**, not in-process warm-ups. SEED does not control those fixed inputs.
- FP64 8×4's available reference is LMUL2/U1. Unsupported LMUL½/LMUL1 results
  and experimental IME mf2 variants are not fabricated.
- The original standalone correctness checker uses alpha1, initial C0 and
  ldc=M. The common driver separately tests nonzero initial C. General alpha
  and padded ldc remain outside these interfaces. CSV status, zero differences,
  overflow and histograms are checked even when a failed checker exits zero.
- Old 72/72 and 8/8 pass counts are not imported. Counts come from this campaign;
  expected unsupported tiny native cases are separate from passing cases.
- Native results still require execution on K1. No shell can recover missing
  historical packing data or certify processor behavior through host checks.

The common driver computes a full INT64 reference twice per invocation outside
timing. Alternating scheduler/mode repetitions uses separate invocations and can
make a complete campaign take considerable time. There is no hidden validation
bypass. Test preflight/smoke first; optional diagnostics and extra workloads can
be disabled explicitly, without falsely claiming they were measured.

Useful switches: `COLLECT_COUNTERS=0`, `COLLECT_PROFILES=0`, `SENSITIVITY=0`,
`EXTRA_WORKLOADS=0`, `PREPACKED_DIAGNOSTICS=0`. Figure 4 defaults counters off.
Figure 6 uses `STATIC_WEIGHT=4` and `DYNAMIC_CHUNK=1`; Figure 8 accepts the same
overrides. `SUMMARY_TILE=best` selects the global independent pair by default;
use 8x4 or 8x8 for a same-tile control.
An explicit `FIG06_DATASET` must satisfy the new solution's accepted-record
contract; unrelated historical data are not silently accepted.

## Delivery verification — 2026-10-09

All 12 delivered shell files passed Bash syntax checks. Read-only source checks
resolved all 108 supported configurations (44 INT8 and 64 floating-point).
Real jq offline tests exercised tuning/selection, independently selected tile
pairs, matched comparisons, scaling, scheduling, statistics and Figure 8
derivation. Negative tests rejected malformed/partial records, incorrect GOPS,
wrong CPU placement, wrong work counts, duplicate samples and mixed identities.
FP parsers and accuracy CSV/histogram gates passed positive/negative fixtures.
Temporary host fixtures/tools are excluded from this benchmark package.

Native K1 compilation, instruction execution and actual datasets are pending.
The 1,720 existing project files had identical before/after SHA-256 inventories;
no existing source, runner, dataset, figure or paper file was changed.
Do not submit generated plots until each relevant coverage/status file confirms
that the required configuration groups and validation cases completed.
