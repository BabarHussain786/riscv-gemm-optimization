#!/usr/bin/env bash
set -euo pipefail

# K1 fair strong-scaling benchmark launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec bash "${PROJECT_ROOT}/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/run_k1_strong_scaling.sh" "$@"
