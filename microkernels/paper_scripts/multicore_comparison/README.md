# Multicore and Heterogeneous Execution

## Purpose

This campaign supplies multicore RVV and heterogeneous RVV–IME measurements. The main campaign is driven by the multicore orchestration entry point; the archived data contain the cleaned summary, raw records, metadata, and source campaign.

## Drivers

Main campaign:

```bash
bash paper_scripts/multicore_comparison/main_driver/run_fig07_multicore_comparison.sh
```

Provenance and summary derivation:

```bash
bash paper_scripts/multicore_comparison/extra_driver/run_k1_fig08_summary_phase_aware.sh --check
bash paper_scripts/multicore_comparison/extra_driver/run_k1_fig08_summary_phase_aware.sh
```

The extra driver is an offline derivation step. It selects matched accepted comparison records for the heterogeneous summary; it does not build or run a kernel. Its default selection is the `1024^3` workload with eight threads, four RVV workers, four IME workers, static weight 4, and dynamic chunk 1. The script retains its original `fig08` name because that is the identifier used by its internal derivation contract.

## Archived inputs

- Main clean data: `../../datasets/fig07_multicore_heterogeneous/main_dataset/clean/`
- Raw records: `../../datasets/fig07_multicore_heterogeneous/main_dataset/raw/`
- Campaign metadata: `../../datasets/fig07_multicore_heterogeneous/main_dataset/metadata/`
- Complete source campaign: `../../datasets/fig07_multicore_heterogeneous/extra_dataset/source_campaign/`

The archived data are preserved as reference inputs. New campaign results should be written to a new results directory rather than overwriting these files.
