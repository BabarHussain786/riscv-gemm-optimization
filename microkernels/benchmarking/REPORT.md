# Benchmark-layer implementation and verification report

Date: 2026-10-01. Scope: `finalist/microkernels/benchmarking` only.

## Outcome and evidence limits

The additive measurement infrastructure is implemented. It is **not yet a
hardware-validated replacement dataset**. The available machine is Windows
AMD64, not K1. No paper number, plot, or performance claim was changed; no new
RVV/IME timing is asserted. The exact native commands are in `README.md`.

Completed local checks:

- 52 regression tests passed, including seven compiled-C host tests and a
  synthetic-fixture plotting smoke test. The synthetic charts are removed
  after testing and are not published as experimental evidence.
- The common C driver and scalar diagnostic backend compiled and executed
  with Zig 0.13.0/Clang 18. Aligned, tail and main-shape checks passed in both
  timing modes. Deliberate corruption aborted before timing acceptance.
- 23 RISC-V object compilations passed with zero compiler diagnostics: all
  eight IME wrappers, their eight original fallbacks, four representative RVV
  kernels, both tile-shape RVV wrappers, and the Linux driver/counter code
  without OpenMP. Objects were checked as RISC-V ELF (`e_machine=243`).
- The remaining OpenMP object check could not compile because the local
  portable compiler has no `omp.h`. Its GNU target also lacks
  `gnu/stubs-lp64d.h`. These are local toolchain limitations, not successful
  native build/execution claims. Cross-checks used the bundled musl headers;
  the intended K1 build uses the documented native GCC/OpenMP command.
- Python syntax and Bash syntax checks, actual diagnostic campaigns, CSV/JSON
  processing, unique-output-directory checks, and unsupported-hardware gates
  are recorded in `verification/local_checks.json` and associated test log.

The native object commands, hashes and diagnostics are recorded in
`verification/native_object_checks.json`. Diagnostic runs never enter accepted
hardware datasets. Empty accepted datasets correctly produce empty tables and
no performance plots.

## A. Files created

All created files are inside the new `benchmarking` directory:

| Files | Purpose |
|---|---|
| `build.py`, `Makefile` | Isolated builds; source/compiler/command/executable provenance |
| `src/bench.h`, `src/bench.c` | Shared input, validation, scheduling and timing contract |
| `src/rvv_adapter.c`, `src/ime_adapter.c` | Existing packer/kernel interfaces; phase separation |
| `src/reference_adapter.c` | Explicitly nonpublishable host-only diagnostic backend |
| `src/counters.h` | Per-worker region-scoped perf events; null on unavailable counters |
| `run.py` | Validation-gated execution, unique campaigns, CSVs and diagnostics |
| `analyze.py` | Strict accepted-data tables and plots; matched comparisons only |
| `verify_preservation.py` | Read-only original-file SHA256 snapshot/check tool |
| `scripts/run_paper_rvv_ime_campaign.sh` | Complete selected-configuration campaign |
| `scripts/run_rvv_ime_tuning.sh` | 36 RVV plus 8 IME canonical configurations |
| `scripts/run_rvv_ime_repeatability.sh` | Independent repetitions of one fixed configuration |
| `scripts/diagnose_ime_runtime.sh` | Small, pinned, validation-only IME diagnostic |
| `scripts/collect_system_info.sh` | Read-only machine/git/environment inventory |
| `tests/test_runner.py`, `tests/test_analysis.py`, `tests/test_host_driver.py` | Regression and optional compiled-host checks |
| `requirements.txt`, `.gitignore` | Plot dependency and generated-file exclusions |
| `README.md`, `TECHNICAL_FLOW.md`, `CHANGES.md`, `REPORT.md` | Reproduction commands, source-grounded flow, changes and limitations |
| `verification/*` | Local check records; not paper measurement data |

## B. Existing files modified

None by this implementation. The original kernel, benchmark, launcher, paper,
figure and result files were not edited, overwritten, or deleted. New code is
added alongside them rather than patched into an old build target.

## C. Existing files intentionally left unchanged

The four RVV baseline trees, `IME_NATIVE_KERNELS`,
`HETEROGENEOUS_RVV_IME_OPENMP_GEMM`, the accuracy-validation tree,
`paper_scripts`, original top-level scripts, `extra`, and all pre-existing
results remain outside the write scope of this task.

A concurrent filesystem change was detected during verification: the old
`campaign_20260928_201610_144719` folder disappeared and newer result folders,
archives, analysis files and a fair-end-to-end launcher appeared. This task
did not make those changes. Of the 984 initially snapshotted files, all 573
surviving files were byte-identical; all 330 original C/H files survived
unchanged. The 411 missing files belonged to that old campaign; 394 files were
added externally, with no new C/H files. See
`verification/external_change_audit.json`. The benchmark layer does not restore,
delete, or merge those external changes.

## D. Existing RVV packing

`HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h` contains
`packed_block_size()` and `call_packed_rvv_tile_kernel()`. The new wrapper
includes the exact helper and replaces only its final callback with a no-op
inside the new translation unit. The real selected kernel is then called
separately. Original source bytes are unchanged.

## E. Existing IME packing

Each canonical file under
`IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_<8x4-or-8x8>_NATIVE/<kernel>/<kernel>.c`
contains `pack_a_panel()` and `pack_b_panel()`. The adapter includes that source
to reuse its static helpers. Compact-B strip preparation follows the original
OpenMP dispatcher. Exact layout/function details are in `TECHNICAL_FLOW.md`.

## F. Lorenzo's vectorized packer/timers

The specifically described explicit vectorized packing/timing implementation
was not identified in this checkout, including inspected older/extra sources.
The currently reused packers are scalar C; possible compiler vectorization is
not evidence of a separately authored RVV packing routine. Ask Lorenzo for its
repository/branch/commit before replacing these packers. No substitute
vectorized algorithm was invented.

## G. RVV timing boundaries

Prepacked total timing begins after packing and includes assignment,
computation, fused C update/tails and synchronization. End-to-end adds the
existing input packing. Allocation, input initialization, reference generation,
pinning/team creation and validation are outside both scopes. In profiles,
`kernel_sec` includes RVV's inseparable output update and tails; their separate
fields are null, not zero.

## H. IME timing boundaries

The same outer total-time boundaries apply. Prepacked includes native compute,
per-tile scatter, scalar cleanup and synchronization. End-to-end additionally
includes compact-B preparation and A/B panel packing. IME phase profiles record
packing, native compute, scatter and actual tail work separately. No-tail cases
do not execute boundary phase timers. Native output buffers are not redundantly
zeroed: the original accumulation writes the used elements; scalar-only cases
skip native output/scatter entirely.

## I. Prepacked mode

Both backends start with ready packed panels and end with the correct common C
layout. Required output work is included for both. Dynamic mixed execution
prepares both layouts before timing because ownership is unknown. This memory
policy is explicitly recorded in the methodology, not treated as free work in
end-to-end execution.

## J. End-to-end mode

Only each strip's assigned backend is packed inside timing. All paths start
with identical original matrices/seed/initial C and end with the same output
contract. Workspaces are preallocated for both. This is pipeline end-to-end,
not whole-process or allocator latency. IME keeps packed B panels instead of
the old wrapper's one-panel reuse lifecycle; old timings are not relabeled as
this new protocol.

## K. Output handling

RVV performs the update internally. IME calls the existing `scatter_output()`
after each native software tile, using `output_index()` to reach canonical C.
There is no invented final global transpose. Tails retain original scalar
helpers. Mixed output phase fields identify the IME portion only.

## L. Validation gate

Every selected configuration validates aligned 16x16x64, tail 15x15x69 and the
full requested shape against independent INT64 arithmetic before timing.
Mixed validation separately checks both selected backends as well. Every warm-up
and measured result is checked. Nonzero exits, SIGILL, wrong outputs, truncated
repetitions, bad timing/GOPS, or mismatched metadata reject the entire case.
No emitted prefix of a failed case is accepted. Host-reference/diagnostic rows
are explicitly excluded. Validation coverage is finite and uses bounded signed
values; it is not exhaustive over all INT8 inputs or overflow behavior.

## M--O. Central 8-core comparison

The launcher provides 8 RVV workers, 4 IME + 4 RVV static, and 4 IME + 4 RVV
dynamic. Dimensions, input seed, numeric contract, total workers and end-to-end
scope are shared. All use the same chosen RVV worker kernel, while IME selection
is independent. Static defaults to 4:1 IME:RVV strip allocation; dynamic defaults
to one-strip atomic queue chunks. Defaults are not claimed optimal. Exact
individual and campaign commands are in `README.md`.

## P. CPU affinity and capability

One distinct permitted CPU is assigned per worker. Before/after CPU observations
and strip counts are emitted. First listed mixed CPUs are IME workers. The
existing per-CPU device-tree IME marker and exact RVV VLEN=256/A60 profile are
required. Capability failures are fatal, not silent fallback. Before a long
IME-containing campaign, selected IME validation runs independently on each
intended IME CPU. This is conservative: a board without the expected device-tree
marker is rejected even if it might support the ISA through another mechanism.

## Q. Repeatability versus tuning

Tuning enumerates 36 distinct canonical RVV configurations and eight canonical
IME configurations. Repeatability repeats one fixed selected configuration;
repetition numbers are scoped by case ID. Analysis keys include matrix shape,
kernel, LMUL, tile, U, worker split, affinity, schedule, input seed, timing scope,
instrumentation, compiler, source and system provenance. It does not pool changing
worker strip counts or counter readings as configuration dimensions. Ambiguous
RVV baselines do not produce cherry-picked speedups.

## R. Counters and primary metrics

Per-thread perf groups surround each worker's measured assignment/pack/compute/
output work. Process initialization and reported-run validation are excluded.
Unavailable or multiplexed groups return null, not fabricated zero. Counts are
summed across workers; IPC uses summed instructions/cycles. Worker counter
regions exclude final-barrier waiting, whereas wall timing includes it. Counter
control/read and phase-profiling overhead remain in those instrumented totals.
Consequently those totals are excluded from primary performance/speedup plots.
Execution time is primary; per-run GOPS is derived before aggregation. Multicore
worker-phase sums are never mislabeled as percentages of total wall time.

## S. Reproduction and handoff

`README.md` contains build, both timing modes, correctness-only, 8-core RVV,
static/dynamic mixed, repeatability, all-config tuning, counters and analysis
commands. Each new campaign writes unique folders with commands, build hashes,
system/git/ISA metadata, validation evidence, logs and raw/accepted/failed CSVs.
The five shell launchers call the same Python driver. Run one campaign per board
at a time. Existing result datasets are not automatically imported into the new
strict schema. This task does not push a GitHub repository or message Lorenzo;
the new folder and its report are ready to review/share through your repository.

## T. Remaining blockers and questions

1. Build/link the new GCC/OpenMP executable on K1 and execute all 44 canonical
   configurations, both scopes and the three 8-worker cases. Local object checks
   are not hardware correctness or instruction-support proofs.
2. Run the IME diagnostic with GDB on the actual board. Historical campaign
   inspection recorded eight IME failures with exit 132: 8x4 U1 and all 8x8
   variants failed the 15x15x69 boundary check; 8x4 U2/U4/U8 failed main 1024^3
   execution. The old runner sometimes proceeded after a failed boundary gate.
   Exit 132 is consistent with SIGILL, but no saved fault PC/opcode proved the
   cause. Those old log paths are now absent after the external filesystem
   change. Do not assert that the faulting instruction, encoding, or CPU cause
   has been identified. The new handler records fault addresses, each case
   retains its exact executable/hash/flags/affinity, and optional GDB disassembles
   around `$pc`. Non-PIE builds make recorded addresses easier to inspect.
   The representative historical 8x4/U1 executable is the `bench` target in
   `IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_8x4_NATIVE/ime_kernel_8x4_zvl256b_lmul1_unroll1/`.
   Its existing Makefile uses GCC with `-O3 -march=rv64gcv_zvl256b -mabi=lp64d
   -std=c11`; the original launcher calls `./bench` with validation enabled on
   an assigned CPU. A retained historical executable image/hash and fault PC
   are unavailable, so this identifies the build target, not a recovered
   crash-binary identity. Capture those with the new diagnostic when rerunning.
3. Newer externally supplied legacy results include 48 successful mixed LMUL-1
   runs and eight experimental mf2 numerical failures. These are useful context,
   not validation of this new layer or proof of the old exit-132 root cause.
   Experimental IME mf2 remains excluded from the primary configuration set.
4. Obtain Lorenzo's original vectorized packing/timing source and confirm
   whether he wants the existing scalar packers or that implementation in a
   subsequent, separately measured comparison.
5. Confirm board CPU-to-IME mapping, perf permissions/event availability,
   compiler, clock/governor/thermal conditions and intended static work ratio.
   The scripts record/check available information but cannot establish it from
   this Windows host.
6. Generate new hardware datasets before updating figures or claims. Diagnose
   high variability using fixed-configuration repeated runs rather than pooled
   LMUL/tile samples. Do not reuse old mixed timing scopes as matched baselines.

No fabricated measurements, speculative instruction fixes, or silent changes
to the original numerical kernels were used to bypass these blockers.
