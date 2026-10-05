#!/usr/bin/env bash
set -euo pipefail

# Run the complete three-path phase experiment.  Arguments are M N K tile_N
# runs; all default to the Figure-5 workload.
M="${1:-1024}"
N="${2:-1024}"
K="${3:-1024}"
TILE_N="${4:-32}"
RUNS="${5:-7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_SCRIPT="${SCRIPT_DIR}/../HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/run_openmp_tiled_gemm_output_timing.sh"
LOG_DIR="${SCRIPT_DIR}/../HETEROGENEOUS_RVV_IME_OPENMP_GEMM/results/output_phase_timing_runs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/fig05_output_phase_timing_${STAMP}.log"

{
    echo "OUTPUT_PHASE_TIMING_START=${STAMP}"
    echo "WORKLOAD=${M}x${N}x${K} TILE_N=${TILE_N} RUNS=${RUNS}"
    echo "PHASES=input_packing,kernel_execution,output_scatter"

    KIND_FILTER=INT8_RVV \
    bash "${MODULE_SCRIPT}" k1-rvv "${M}" "${N}" "${K}" "${TILE_N}" "${RUNS}"

    GEMM_TILE_SCHEDULE=static KIND_FILTER=INT8_MIXED \
    bash "${MODULE_SCRIPT}" k1-mixed-rvv-ime "${M}" "${N}" "${K}" "${TILE_N}" "${RUNS}"

    GEMM_TILE_SCHEDULE=dynamic KIND_FILTER=INT8_MIXED \
    bash "${MODULE_SCRIPT}" k1-mixed-rvv-ime "${M}" "${N}" "${K}" "${TILE_N}" "${RUNS}"

    echo "OUTPUT_PHASE_TIMING_DONE=${STAMP}"
} 2>&1 | tee "${LOG}"

echo "LOG=${LOG}"
