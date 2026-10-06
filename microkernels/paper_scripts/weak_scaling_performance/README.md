# Weak-Scaling Performance

This folder contains the two weak-scaling launchers used by the paper:

- `run_fig02_weak_scaling.sh` runs the paper Figure 2 campaign through the
  common paper runner.
- `run_k1_weak_scaling.sh` runs the SpaceMiT K1 weak-scaling campaign.

The wrappers preserve the original project paths and can be launched from any
working directory. Results are written to the configured benchmark output
directory; this folder contains scripts only.
