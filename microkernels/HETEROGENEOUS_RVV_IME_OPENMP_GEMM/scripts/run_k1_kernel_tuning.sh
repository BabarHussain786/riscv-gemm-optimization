#!/usr/bin/env bash
set -uo pipefail

# Sweep output-strip width while the existing kernel inventory supplies the
# tile shape, LMUL, and unroll dimensions. Unsupported VLEN=256 combinations
# remain excluded by run_openmp_tiled_gemm_mode.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=k1_experiment_common.sh
source "${SCRIPT_DIR}/k1_experiment_common.sh"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
TILE_SIZES="${TILE_SIZES:-8 16 32 64 128}"
ENABLE_EXPERIMENTAL_MF2="${ENABLE_EXPERIMENTAL_MF2:-0}"

for value in "${M}" "${N}" "${K}" "${RUNS}"; do
    require_positive_integer "experiment value" "${value}"
done

start_experiment "k1_kernel_tuning"

CASE_M="${M}"; CASE_N="${N}"; CASE_K="${K}"
CASE_RUNS="${RUNS}"; CASE_SCHEDULE="static"; CASE_CHUNK="1"
CASE_IME_WEIGHT="4"; CASE_RVV_WEIGHT="1"
CASE_PERF_STAT="0"; CASE_PERF_EVENTS="cycles,instructions,cache-references,cache-misses"
CASE_KERNEL="*"
CASE_ENABLE_MF2="${ENABLE_EXPERIMENTAL_MF2}"

for tile_n in ${TILE_SIZES}; do
    require_positive_integer "tile width" "${tile_n}"
    CASE_TILE_N="${tile_n}"

    CASE_MODE="k1-rvv"; CASE_THREADS="8"; CASE_KIND="INT8_RVV"
    run_experiment_case "RVV" "tile_N" "${tile_n}" || true

    CASE_MODE="k1-ime"; CASE_THREADS="4"; CASE_KIND="INT8_IME"
    run_experiment_case "IME" "tile_N" "${tile_n}" || true
done

finish_experiment
