# Additive changes

## 2026-10-01: fair INT8 benchmark infrastructure

- Added isolated builds and source/compiler/executable fingerprints. Reason:
  keep original kernels/builds intact and identify exact measured code.
- Added common input, packing, execution and output timing contracts. Reason:
  standalone RVV and IME timers previously covered different work.
- Reused RVV dispatch packing and IME packing/accumulation/scatter/tail helpers.
  Reason: separate measurements without speculative replacement kernels.
- Added pinned 8-RVV and independent-kernel 4+4 static/dynamic execution.
  Reason: provide the central matched-core-count experiment.
- Added full-output INT64 checks, aligned/tail/main gates, per-backend mixed
  checks and transactional sample acceptance. Reason: failed cases must not
  leak partial performance samples into accepted data.
- Added region counters, separate profile passes and null fused/unavailable
  fields. Reason: avoid mismatched whole-process counters and invented phases.
- Added unique campaign directories, metadata, checkpointing, diagnostics,
  36-RVV/8-IME exploration and separate fixed-configuration repeatability.
- Added strict tables/plots and tests. Reason: preserve timing boundaries,
  derive per-run GOPS, prohibit pooling configurations, and reject host/failure
  records. Counter/profile runs are not primary timing comparisons.
- Corrected new-wrapper-only IME profiling overhead: no redundant native-tile
  zero initialization, and no boundary timers for nonexistent tails.
- Added README, source-flow report and final implementation/verification report.

No pre-existing source, launcher, result, figure, or paper file was edited by
this task. Concurrent externally added/removed result files are documented in
REPORT.md rather than overwritten or restored.
