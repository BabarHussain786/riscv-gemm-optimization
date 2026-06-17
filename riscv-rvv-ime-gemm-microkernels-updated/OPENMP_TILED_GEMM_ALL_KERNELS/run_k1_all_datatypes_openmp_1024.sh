#!/usr/bin/env bash
set -euo pipefail

# Runs the cleaned K1 OpenMP campaign for all datatype paths used in the paper.
# FP64/FP32/INT8 RVV are evaluated with RVV kernels; INT8 mixed evaluates
# IME cores and RVV fallback cores together through weighted tile scheduling.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_all_openmp_kernels.sh"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
TILE_N_RVV="${TILE_N_RVV:-64}"
TILE_N_IME="${TILE_N_IME:-64}"
TILE_N_MIXED="${TILE_N_MIXED:-32}"

run_mode()
{
    local mode="$1"
    local tile_n="$2"
    echo "============================================================"
    echo "Running ${mode}: M=${M} N=${N} K=${K} tile_N=${tile_n} runs=${RUNS}"
    echo "============================================================"
    bash "${RUNNER}" "${mode}" "${M}" "${N}" "${K}" "${tile_n}" "${RUNS}"
}

# RVV datatype sweep: FP64, FP32, and INT8 RVV kernels.
run_mode k1-rvv "${TILE_N_RVV}"

# Disjoint RVV-cluster baseline on cores 4-7.
run_mode k1-rvv-only "${TILE_N_RVV}"

# IME-only baseline on cores 0-3.
run_mode k1-ime "${TILE_N_IME}"

# True heterogeneous INT8 run: IME cores 0-3 plus RVV fallback cores 4-7.
export MIXED_IME_TILE_WEIGHT="${MIXED_IME_TILE_WEIGHT:-4}"
export MIXED_RVV_TILE_WEIGHT="${MIXED_RVV_TILE_WEIGHT:-1}"
run_mode k1-mixed-rvv-ime "${TILE_N_MIXED}"