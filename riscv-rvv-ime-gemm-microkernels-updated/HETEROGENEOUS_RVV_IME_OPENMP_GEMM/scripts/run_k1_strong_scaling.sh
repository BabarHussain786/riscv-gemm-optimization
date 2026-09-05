#!/usr/bin/env bash
set -uo pipefail

# Strong scaling keeps M, N, and K fixed while the worker count increases.
# Hardware counters are collected with perf so IPC and cache-miss rate can be
# shown beside execution time. Set COLLECT_PERF=0 when perf is unavailable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=k1_experiment_common.sh
source "${SCRIPT_DIR}/k1_experiment_common.sh"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
TILE_N="${TILE_N:-32}"
RUNS="${RUNS:-6}"
RVV_CORE_COUNTS="${RVV_CORE_COUNTS:-1 2 4 8}"
IME_CORE_COUNTS="${IME_CORE_COUNTS:-1 2 4}"
COLLECT_PERF="${COLLECT_PERF:-1}"
PERF_EVENTS_LIST="${PERF_EVENTS_LIST:-cycles,instructions,cache-references,cache-misses}"
RVV_KERNEL="${RVV_KERNEL:-igemm_kernel_8x4_zvl256b_lmul1_unroll4}"
IME_KERNEL="${IME_KERNEL:-ime_kernel_8x4_zvl256b_lmul1_unroll4}"
INCLUDE_HETEROGENEOUS="${INCLUDE_HETEROGENEOUS:-1}"

for value in "${M}" "${N}" "${K}" "${TILE_N}" "${RUNS}"; do
    require_positive_integer "experiment value" "${value}"
done

start_experiment "k1_strong_scaling"

CASE_M="${M}"; CASE_N="${N}"; CASE_K="${K}"
CASE_TILE_N="${TILE_N}"; CASE_RUNS="${RUNS}"
CASE_SCHEDULE="static"; CASE_CHUNK="1"
CASE_IME_WEIGHT="4"; CASE_RVV_WEIGHT="1"
CASE_ENABLE_MF2="0"
CASE_PERF_STAT="${COLLECT_PERF}"; CASE_PERF_EVENTS="${PERF_EVENTS_LIST}"

for cores in ${RVV_CORE_COUNTS}; do
    require_positive_integer "RVV core count" "${cores}"
    CASE_MODE="k1-rvv"; CASE_THREADS="${cores}"
    CASE_KERNEL="${RVV_KERNEL}"; CASE_KIND="INT8_RVV"
    run_experiment_case "RVV" "cores" "${cores}" || true
done

for cores in ${IME_CORE_COUNTS}; do
    require_positive_integer "IME core count" "${cores}"
    CASE_MODE="k1-ime"; CASE_THREADS="${cores}"
    CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_IME"
    run_experiment_case "IME" "cores" "${cores}" || true
done

# The present heterogeneous implementation has its intended complete K1
# topology at eight workers: four IME workers and four RVV workers.
if [ "${INCLUDE_HETEROGENEOUS}" = "1" ]; then
    CASE_MODE="k1-mixed-rvv-ime"; CASE_THREADS="8"
    CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_MIXED"
    CASE_SCHEDULE="static"
    run_experiment_case "HET_STATIC" "cores" "8" || true
    CASE_SCHEDULE="dynamic"
    run_experiment_case "HET_DYNAMIC" "cores" "8" || true
fi

finish_experiment
