# RISC-V RVV and SpacemiT IME GEMM Optimization

Research code and reproducibility artifacts for INT8 GEMM optimization on the SpacemiT K1 and RISC-V vector architectures.

## Research scope

This project studies portable, compiler-aware GEMM kernels with:

- RISC-V Vector (RVV 1.0) implementations using 256-bit vectors.
- Native SpacemiT IME execution for INT8 matrix operations.
- INT8 inputs with INT32 accumulation and output.
- RVV-only, IME-only, and heterogeneous RVV--IME execution.
- Tile, LMUL, unroll, packing, scheduling, and worker-affinity effects.
- Independent INT64-reference correctness validation.

## Repository navigation

The main project materials are organized as follows:

- [Microkernel suites and K1 workflows](microkernels/README.md)
- [Paper campaign scripts and measured datasets](microkernels/paper_scripts/)
- [K1 OpenMP heterogeneous execution module](microkernels/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/)
- [Independent INT64 correctness validation](microkernels/RVV_IME_GEMM_ACCURACY_VALIDATION/)
- [Readable project report](doc/Final%20Report.pdf)
- [Earlier scalar, tiled, recursive, and loop-interchange implementations](doc/readme.txt)

## Reproducible K1 workflow

From the `microkernels` directory, run the complete paper campaign:

```bash
cd microkernels
bash ./run_all_paper_figures.sh
```

The figure-specific entry points remain available for selective or resumed runs. The workflow records raw measurements, configuration metadata, summarized datasets, and generated analysis assets under the figure-organized directories described in [microkernels/README.md](microkernels/README.md).

## Execution model

The combined K1 configuration uses four IME workers on cores 0--3 and four RVV workers on cores 4--7. Output-column strips use `tile_N=32`, and each strip has one owner. The reported end-to-end path includes input packing, tile-kernel execution, scheduling, and output formation; setup, validation, and cleanup are outside the timed region.

## Research status

The repository contains kernel sources, benchmark scripts, measured campaigns, correctness records, and the documentation used to analyze the RVV--IME design. Results should be interpreted together with the recorded configuration metadata and timing definitions.

## License

See the licenses included with the kernel suites and supporting components.
