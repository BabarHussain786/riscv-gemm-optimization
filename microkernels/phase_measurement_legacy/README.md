# Isolated Figure 2 phase measurement

This directory is a non-destructive copy of the proven legacy Figure 2
strong-scaling campaign.  It discovers and links the original K1 kernels, but
compiles its own copied driver and writes results only under this directory.

The run records, for every accepted repetition:

- total wall-clock time for the OpenMP tile region;
- worker-summed input-packing time;
- worker-summed kernel-call time, explicitly including output stores;
- validation, worker placement, throughput, and the existing performance counters.

The legacy implementation writes each output tile directly from the kernel;
there is no independent output/reshape routine to time.  Therefore output is
reported honestly as fused with the kernel call, and synchronization is marked
as not separately instrumented.  This avoids inventing a phase that the code
does not have.

Run on the K1 from this directory:

```bash
export PHASE_SOURCE_ROOT="$HOME/MAT/kernels riscv rvv 1.0/FINAL/riscv-gemm-optimization/microkernels"
bash ./run_k1_strong_scaling_phase_copy.sh
```

The generated `k1_strong_scaling_phase_raw.csv` is the phase-aware dataset;
the original `paper_results` and legacy source tree are untouched.

