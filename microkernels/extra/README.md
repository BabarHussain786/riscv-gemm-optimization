# Optional experiments and supporting material

The main K1 build and benchmark workflow is described in the [project README](../README.md).
Nothing here is required to build the K1 kernel families. Existing files were
moved here, not deleted or regenerated.

| Folder | Contents | Role |
|---|---|---|
| `experiments/k3/` | K3 campaign launcher | Optional hardware-specific experiment; still uses the kernel families at the project root |
| `documents/` | Three PDF documents and the presentation | Supporting reading, outside the main source tree |
| `notebooks/` | `Heteropar.ipynb` | Exploratory notebook, preserved without execution or content changes |
| `reports/` | FP64 assembly inspection report | Existing inspection output, not a build input |
| `generated/` | PDF page renders and previous local OpenMP CSV/log output | Local archive excluded from Git; these files are not newly validated results |

## Optional K3 campaign

From the project root on suitable hardware:

```bash
bash extra/experiments/k3/run_k3_rvv_ime_0_15_1024.sh
```

The launcher resolves the project root three levels above its new location.
Its benchmark logic and default result directory are otherwise unchanged.

## Where files moved

| Original location | New location |
|---|---|
| `accuracy paper.pdf` | `extra/documents/accuracy paper.pdf` |
| `Heteropar.pdf` | `extra/documents/Heteropar.pdf` |
| `RISC-V_RVV_IME_Comparison.pdf` | `extra/documents/RISC-V_RVV_IME_Comparison.pdf` |
| `Heterogeneous_RVV_IME_OpenMP_GEMM_IEEE_Polished_ZVL256.pptx` | `extra/documents/Heterogeneous_RVV_IME_OpenMP_GEMM_IEEE_Polished_ZVL256.pptx` |
| `Heteropar.ipynb` | `extra/notebooks/Heteropar.ipynb` |
| `run_k3_rvv_ime_0_15_1024.sh` | `extra/experiments/k3/run_k3_rvv_ime_0_15_1024.sh` |
| `GEMM_RVV_FP64_INT8_8x4_Baseline/RVV_DGEMM_FP64_8x4/asm_inspection_report_fp64.txt` | `extra/reports/asm_inspection_report_fp64.txt` |
| `tmp/` | `extra/generated/tmp/` |
| `HETEROGENEOUS_RVV_IME_OPENMP_GEMM/results/openmp_*_latest_local.*` (three files) | `extra/generated/openmp-local-run/` |

The OpenMP `results/` folder and its `.gitkeep` remain in place for new runs.
Duplicate-looking INT8 families and experimental IME variants remain in their
source directories because the family launchers and build configuration refer
to them. Moving them would require changing the supported build layout.
