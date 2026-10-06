# Strong-Scaling Performance

This folder contains the two strong-scaling launchers used by the paper:

- `run_fig01_strong_scaling.sh` runs the paper Figure 1 campaign through the
  common paper runner.
- `run_k1_strong_scaling.sh` runs the fair SpaceMiT K1 strong-scaling campaign
  with the heterogeneous RVV--IME implementation.

The wrappers keep the original project paths intact, so they can be launched
from any working directory. Results are written to the configured benchmark
output directory; this folder contains scripts only.
