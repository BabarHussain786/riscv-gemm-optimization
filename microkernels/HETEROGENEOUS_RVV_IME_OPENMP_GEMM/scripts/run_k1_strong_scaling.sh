#!/usr/bin/env bash
set -uo pipefail

# STRONG-SCALING ROADMAP
# Step 1 -> Keep the GEMM size fixed and select one 8x4 INT8 kernel shape.
# Step 2 -> Run RVV and native IME separately while changing core count.
# Step 3 -> Repeat the same experiment for unroll 1, 2, 4, and 8.
# Step 4 -> Record time, GOPS, validation, IPC, and cache counters.
# Step 5 -> Print one compact table for direct use in Plot 1.
#
# Static and dynamic heterogeneous scheduling are separate experiments. They
# are excluded here unless INCLUDE_HETEROGENEOUS=1 is explicitly requested.

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
LMUL="${LMUL:-1}"
UNROLL_FACTORS="${UNROLL_FACTORS:-1 2 4 8}"
RVV_KERNEL_PREFIX="${RVV_KERNEL_PREFIX:-igemm_kernel_8x4_zvl256b_lmul${LMUL}_unroll}"
IME_KERNEL_PREFIX="${IME_KERNEL_PREFIX:-ime_kernel_8x4_zvl256b_lmul${LMUL}_unroll}"
INCLUDE_HETEROGENEOUS="${INCLUDE_HETEROGENEOUS:-0}"
EXPERIMENT_QUIET_CASES="${EXPERIMENT_QUIET_CASES:-1}"

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

for unroll in ${UNROLL_FACTORS}; do
    require_positive_integer "unroll factor" "${unroll}"

    RVV_KERNEL="${RVV_KERNEL_PREFIX}${unroll}_i8i32"
    IME_KERNEL="${IME_KERNEL_PREFIX}${unroll}"
    experiment_log "============================================================"
    experiment_log "CONFIGURATION LMUL=${LMUL} UNROLL=${unroll} RVV=${RVV_KERNEL} IME=${IME_KERNEL}"

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

    # Mixed scheduling stays optional because it has its own paper figure.
    if [ "${INCLUDE_HETEROGENEOUS}" = "1" ]; then
        CASE_MODE="k1-mixed-rvv-ime"; CASE_THREADS="8"
        CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_MIXED"
        CASE_SCHEDULE="static"
        run_experiment_case "HET_STATIC" "cores" "8" || true
        CASE_SCHEDULE="dynamic"
        run_experiment_case "HET_DYNAMIC" "cores" "8" || true
    fi
done

print_result_table()
{
    experiment_log "============================================================"
    experiment_log "K1 STRONG-SCALING SUMMARY (time in ms, throughput in GOPS)"
    experiment_log "Fixed GEMM=${M}x${N}x${K} tile_N=${TILE_N} LMUL=${LMUL}"
    experiment_log "--------------------------------------------------------------------------------"
    experiment_log "series  unroll cores  mean_ms   mean_GOPS  ok_runs  failed  build_failed"
    experiment_log "--------------------------------------------------------------------------------"

    awk -F, '
    NR == 1 { next }
    {
        mean_ms = ($30 == "NA" ? "NA" : $30 * 1000.0)
        printf "%-7s %-6s %-5s %-9s %-10s %-8s %-7s %-12s\n", \
            $2, $12, $4, mean_ms, $25, $24, $33, $34
    }' "${EXPERIMENT_SUMMARY}" | sort -k1,1 -k2,2n -k3,3n |
        while IFS= read -r line; do
            experiment_log "${line}"
        done

    experiment_log "--------------------------------------------------------------------------------"
    experiment_log "CSV summary: ${EXPERIMENT_SUMMARY}"
    experiment_log "CSV raw:     ${EXPERIMENT_RAW}"
    experiment_log "============================================================"
}

print_result_table

finish_experiment
