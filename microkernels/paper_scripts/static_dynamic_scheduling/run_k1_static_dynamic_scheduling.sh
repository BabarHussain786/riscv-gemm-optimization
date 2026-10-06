#!/usr/bin/env bash
set -euo pipefail

# K1 scheduling/partitioning analysis launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec bash "${PROJECT_ROOT}/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/run_k1_partitioning_analysis.sh" "$@"
