# Independent Correctness Validation

## Purpose

This campaign validates the canonical RVV kernels and forced-native IME wrappers against an independent correctness checker. The planned campaign covers 36 RVV kernels and 8 IME kernels across five shapes, four input classes, and three seeds.

## Drivers

Main campaign:

```bash
bash paper_scripts/correctness_validation/main_driver/run_fig08_correctness.sh
```

K1 phase-aware validation:

```bash
bash paper_scripts/correctness_validation/extra_driver/run_k1_fig09_correctness_phase_aware.sh --check
bash paper_scripts/correctness_validation/extra_driver/run_k1_fig09_correctness_phase_aware.sh --smoke
bash paper_scripts/correctness_validation/extra_driver/run_k1_fig09_correctness_phase_aware.sh
```

`--check` performs read-only interface checks. `--smoke` validates one aligned shape, one input class, and one seed for all 44 kernels. The full run uses the documented five-shape, four-class, three-seed matrix and also checks selected `1024^3` cases.

## Validation contract

The checker uses INT64 reference accumulation, alpha 1, initial C equal to zero, and `ldc=M` in the standalone path. RVV and IME paths are kept distinct; IME validation requires the native extension path rather than an RVV fallback proxy. The selected full-workload checks use `M=N=K=1024`.

## Archived inputs

- Planned metadata: `../../datasets/fig08_correctness/main_dataset/metadata/`
- Source campaign records: `../../datasets/fig08_correctness/extra_dataset/source_campaign/`

No clean independent K1 correctness summary is currently archived. Run the validation on the IME-capable board, then store accepted raw records and a clean summary in new dataset subdirectories without overwriting the plan or source campaign.
