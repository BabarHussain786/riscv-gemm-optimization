#!/usr/bin/env bash
set -uo pipefail

# Compare fixed IME:RVV work ratios with OpenMP dynamic chunk sizes.
# Both policies use the same matrix, output-strip width, kernel pair, and runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=k1_experiment_common.sh
source "${SCRIPT_DIR}/k1_experiment_common.sh"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
TILE_N="${TILE_N:-32}"
RUNS="${RUNS:-6}"
STATIC_RATIOS="${STATIC_RATIOS:-1:4 1:2 1:1 2:1 4:1 8:1}"
DYNAMIC_CHUNKS="${DYNAMIC_CHUNKS:-1 2 4 8}"
IME_KERNEL="${IME_KERNEL:-ime_kernel_8x4_zvl256b_lmul1_unroll4}"

for value in "${M}" "${N}" "${K}" "${TILE_N}" "${RUNS}"; do
    require_positive_integer "experiment value" "${value}"
done

start_experiment "k1_partitioning"

CASE_M="${M}"; CASE_N="${N}"; CASE_K="${K}"
CASE_TILE_N="${TILE_N}"; CASE_RUNS="${RUNS}"
CASE_MODE="k1-mixed-rvv-ime"; CASE_THREADS="8"
CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_MIXED"
CASE_ENABLE_MF2="0"
CASE_PERF_STAT="0"; CASE_PERF_EVENTS="cycles,instructions,cache-references,cache-misses"
CASE_CHUNK="1"

for ratio in ${STATIC_RATIOS}; do
    case "${ratio}" in
        *:*) ;;
        *)
            experiment_log "INVALID_STATIC_RATIO=${ratio}; expected IME:RVV"
            EXPERIMENT_FAILURES=$((EXPERIMENT_FAILURES + 1))
            continue
            ;;
    esac
    CASE_IME_WEIGHT="${ratio%%:*}"
    CASE_RVV_WEIGHT="${ratio##*:}"
    require_positive_integer "IME ratio weight" "${CASE_IME_WEIGHT}"
    require_positive_integer "RVV ratio weight" "${CASE_RVV_WEIGHT}"
    CASE_SCHEDULE="static"
    run_experiment_case "STATIC" "ime_rvv_ratio" "${ratio}" || true
done

CASE_IME_WEIGHT="4"; CASE_RVV_WEIGHT="1"
for chunk in ${DYNAMIC_CHUNKS}; do
    require_positive_integer "dynamic chunk" "${chunk}"
    CASE_CHUNK="${chunk}"
    CASE_SCHEDULE="dynamic"
    run_experiment_case "DYNAMIC" "chunk" "${chunk}" || true
done

finish_experiment
