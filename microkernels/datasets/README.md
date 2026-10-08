# Measured datasets

This directory contains measured exports and their provenance records. No
values are generated or interpolated here. Each figure has its own folder;
available exports are
separated into `clean/`, `raw/`, and `metadata/` subfolders. Folders with no
measured export are left absent rather than represented by empty placeholders.
Older measurements are labelled with their campaign date and are never mixed
silently with current data.

The machine-readable inventory is [`MANIFEST.csv`](MANIFEST.csv). Its status
values distinguish measured exports (`complete` and `archived`) from partial or
not-yet-measured experiments (`partial` and `pending`).

## Available exports

| Experiment | Dataset | Status |
|---|---|---|
| Figure 1: strong scaling | `fig01_strong_scaling/` | Current data plus dated archive |
| Figure 2: weak scaling | `fig02_weak_scaling/` | Archived K1 data |
| Figure 3: static/dynamic scheduling | `fig03_static_dynamic/` | Archived K1 data |
| Figure 4: RVV INT8 tuning | `fig04_rvv_int8_tuning/` | Archived K1 data and clean INT8 statistics |
| Figure 5: RVV--IME comparison | `fig05_rvv_ime_comparison/` | Current cleaned and raw campaign data |
| Figure 6: FP32/FP64 reference | `fig06_fp32_fp64/` | FP32 statistics and campaign plan; FP64 rerun required |
| Figure 7: multicore/heterogeneous | `fig07_multicore_heterogeneous/` | Archived K1 data |
| Figure 8: correctness | `fig08_correctness/` | Campaign plan available; K1 validation run required |
| Supporting measurements | `support/` | Single-core and standalone INT8 data |

The archived files came from `Benchmarked 6-9-2026`. In each figure folder,
`clean/` contains the summary or statistics intended for analysis, `raw/`
contains per-run rows, and `metadata/` contains logs, manifests, and failure
records when those exports exist. Some archived heterogeneous runs contain
explicit build-failure rows; those rows are retained for auditability.

## Data that still requires a K1 run

There is no independent correctness-validation CSV in this checkout. The
correctness launcher is present at
`../paper_scripts/correctness_validation/run_fig08_correctness.sh`, but it
must be run on the IME-capable K1 board. A complete current FP64 summary and
new current weak-scaling, scheduling, tuning, FP64, and multicore exports
should also be generated on that board if the final paper is to use them as
one campaign. The repository does not invent those measurements.

## Reproducing the campaign plan

From the project root, the following command checks all eight paper launchers
without building or running a kernel:

```bash
python3 paper_scripts/orchestration/paper_campaign.py --dry-run --no-perf
```

The launchers then write raw rows, summaries, and logs under their selected
output directory. Keep the raw CSV beside its summary when exporting a new
campaign.
