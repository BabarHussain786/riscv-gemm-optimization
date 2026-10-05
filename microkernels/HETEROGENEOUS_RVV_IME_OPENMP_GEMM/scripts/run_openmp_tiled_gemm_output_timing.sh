#!/usr/bin/env bash
set -euo pipefail

# One-mode entry point for the instrumented output-phase experiment.
# The underlying phase runner compiles the phase-aware driver and the native
# IME kernels with IME_PHASE_TIMING enabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_openmp_tiled_gemm_phase_timing.sh" "$@"
