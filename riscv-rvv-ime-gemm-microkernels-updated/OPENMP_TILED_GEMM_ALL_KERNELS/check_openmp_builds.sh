#!/usr/bin/env bash
set -euo pipefail

# Compile-only smoke test for the OpenMP module. It checks that the selected
# FP64, FP32, INT8 RVV, IME, and mixed wrapper builds still link cleanly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_all_openmp_kernels.sh"

M="${M:-128}"
N="${N:-128}"
K="${K:-128}"
TILE_N="${TILE_N:-32}"
CHECK_MODES="${CHECK_MODES:-k1-rvv-only k1-ime k1-mixed-rvv-ime}"

export BUILD_ONLY=1
export OMP_VALIDATE=0
export OMP_WARMUP=0

for mode in ${CHECK_MODES}; do
    echo "============================================================"
    echo "Compile check: ${mode}"
    echo "============================================================"
    bash "${RUNNER}" "${mode}" "${M}" "${N}" "${K}" "${TILE_N}" 1
done

echo "OpenMP compile checks completed."