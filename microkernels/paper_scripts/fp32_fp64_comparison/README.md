# FP32/FP64 Reference Measurements

## Purpose

This campaign supplies floating-point reference measurements. It measures standalone RVV FP32 SGEMM and FP64 DGEMM kernels over the fixed `1024^3` workload and the available tile, LMUL, and unroll configurations.

## Drivers

Main campaign:

```bash
bash paper_scripts/fp32_fp64_comparison/main_driver/run_fig06_rvv_fp32_fp64.sh
```

Phase-aware K1 campaign:

```bash
bash paper_scripts/fp32_fp64_comparison/extra_driver/run_k1_fig07_fp_baselines_phase_aware.sh --check
bash paper_scripts/fp32_fp64_comparison/extra_driver/run_k1_fig07_fp_baselines_phase_aware.sh
```

The phase-aware driver inventories 36 FP32 and 28 FP64 configurations. It uses one fixed RVV CPU, seven measured runs, two warmups, and a fresh process for each timed kernel call. `--check` performs the validation and inventory without running the campaign.

## Measurement and validation

The timer covers one standalone kernel call with packed inputs and the output update. Allocation, reference computation, validation, and cleanup are outside the timed interval. Aligned, tail, and full-workload validation gates run before timing. FP32 uses tolerance `1e-5*(1+abs(reference))`; FP64 uses `1e-11*(1+abs(reference))`. New results are written to a new campaign directory under `benchmarking/NEW BENCHMARK/results/`.

## Archived inputs

- Main clean data: `../../datasets/fig06_fp32_fp64/main_dataset/clean/`
- Campaign metadata: `../../datasets/fig06_fp32_fp64/main_dataset/metadata/`
- Complete source campaign: `../../datasets/fig06_fp32_fp64/extra_dataset/source_campaign/`

The archived data are preserved as reference inputs. Re-runs should create a new results directory rather than overwrite these files.
