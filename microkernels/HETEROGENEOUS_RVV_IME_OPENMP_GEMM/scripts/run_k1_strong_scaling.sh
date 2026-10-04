#!/usr/bin/env bash
set -uo pipefail

# STRONG-SCALING ROADMAP
# Step 1 -> Keep the GEMM size fixed and select one 8x4 INT8 kernel shape.
# Step 2 -> Run RVV and native IME separately while changing core count.
# Step 3 -> Repeat the same experiment for unroll 1, 2, 4, and 8.
# Step 4 -> Record end-to-end time, GOPS, validation, IPC, and cache counters.
#           The timing scope is the same for RVV and IME and includes the
#           required input packing performed inside the timed tile region.
# Step 5 -> Verify the timing scope in every generated CSV row.
# Step 6 -> Print one compact table for direct use in the strong-scaling plot.
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
EXPECTED_TIMING_SCOPE="${EXPECTED_TIMING_SCOPE:-parallel_tiles_including_required_packing}"

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

verify_latest_timing_scope()
{
    local result_tag="$1"
    local raw_csv="${COMMON_MODULE_DIR}/results/openmp_raw_latest_${result_tag}.csv"
    local scope_column
    local status_column

    if [ ! -s "${raw_csv}" ]; then
        experiment_log "TIMING_SCOPE_CHECK=FAILED reason=missing_raw_csv file=${raw_csv}"
        return 1
    fi

    scope_column="$(awk -F, '
        NR == 1 {
            for (i = 1; i <= NF; ++i) {
                if ($i == "timing_scope") { print i; exit }
            }
        }
    ' "${raw_csv}")"

    status_column="$(awk -F, '
        NR == 1 {
            for (i = 1; i <= NF; ++i) {
                if ($i == "status") { print i; exit }
            }
        }
    ' "${raw_csv}")"

    if [ -z "${scope_column}" ] || [ -z "${status_column}" ]; then
        experiment_log "TIMING_SCOPE_CHECK=FAILED reason=missing_scope_or_status_column file=${raw_csv}"
        return 1
    fi

    if awk -F, -v scope="${scope_column}" -v status="${status_column}" \
        -v expected="${EXPECTED_TIMING_SCOPE}" '
        NR > 1 && $status == "OK" && $scope != expected { bad = 1 }
        END { exit bad ? 0 : 1 }
    ' "${raw_csv}"; then
        experiment_log "TIMING_SCOPE_CHECK=FAILED expected=${EXPECTED_TIMING_SCOPE} file=${raw_csv}"
        return 1
    fi

    experiment_log "TIMING_SCOPE_CHECK=OK scope=${EXPECTED_TIMING_SCOPE} file=${raw_csv}"
    return 0
}

run_strong_case()
{
    local series="$1"
    local parameter_name="$2"
    local parameter_value="$3"
    local result_tag="${CASE_MODE}"

    if [ "${CASE_MODE}" = "k1-mixed-rvv-ime" ]; then
        result_tag="${CASE_MODE}-${CASE_SCHEDULE}"
    fi

    if ! run_experiment_case "${series}" "${parameter_name}" "${parameter_value}"; then
        return 1
    fi

    if ! verify_latest_timing_scope "${result_tag}"; then
        EXPERIMENT_FAILURES=$((EXPERIMENT_FAILURES + 1))
        return 1
    fi
    return 0
}

experiment_log "TIMING_SCOPE_EXPECTED=${EXPECTED_TIMING_SCOPE}"
experiment_log "FAIR_COMPARISON=RVV and IME use identical end-to-end timing boundaries; required input packing is included."

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
        run_strong_case "RVV" "cores" "${cores}" || true
    done

    for cores in ${IME_CORE_COUNTS}; do
        require_positive_integer "IME core count" "${cores}"
        CASE_MODE="k1-ime"; CASE_THREADS="${cores}"
        CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_IME"
        run_strong_case "IME" "cores" "${cores}" || true
    done

    # Mixed scheduling stays optional because it has its own paper figure.
    if [ "${INCLUDE_HETEROGENEOUS}" = "1" ]; then
        CASE_MODE="k1-mixed-rvv-ime"; CASE_THREADS="8"
        CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_MIXED"
        CASE_SCHEDULE="static"
        run_strong_case "HET_STATIC" "cores" "8" || true
        CASE_SCHEDULE="dynamic"
        run_strong_case "HET_DYNAMIC" "cores" "8" || true
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
