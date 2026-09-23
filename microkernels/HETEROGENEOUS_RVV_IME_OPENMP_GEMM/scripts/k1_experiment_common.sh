#!/usr/bin/env bash

# Shared result handling for the focused K1 paper experiments.
# This file is sourced by the four public run_k1_* experiment scripts.

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_MODULE_DIR="$(cd "${COMMON_SCRIPT_DIR}/.." && pwd)"
COMMON_RUNNER="${COMMON_SCRIPT_DIR}/run_openmp_tiled_gemm_mode.sh"
COMMON_RESULT_ROOT="${COMMON_MODULE_DIR}/results/paper_experiments"

experiment_log()
{
    printf '%s\n' "$*" | tee -a "${EXPERIMENT_LOG}"
}

require_positive_integer()
{
    local name="$1"
    local value="$2"

    case "${value}" in
        ''|*[!0-9]*)
            printf 'ERROR: %s must be a positive integer, received %s\n' "${name}" "${value}" >&2
            exit 1
            ;;
    esac
    if [ "${value}" -le 0 ]; then
        printf 'ERROR: %s must be a positive integer, received %s\n' "${name}" "${value}" >&2
        exit 1
    fi
}

start_experiment()
{
    local experiment_name="$1"

    EXPERIMENT_NAME="${experiment_name}"
    EXPERIMENT_STAMP="$(date +%Y%m%d_%H%M%S)"
    EXPERIMENT_DIR="${COMMON_RESULT_ROOT}/${EXPERIMENT_NAME}_${EXPERIMENT_STAMP}"
    EXPERIMENT_RAW="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_raw.csv"
    EXPERIMENT_SUMMARY="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_summary.csv"
    EXPERIMENT_LOG="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}.log"
    EXPERIMENT_FAILURES=0

    mkdir -p "${EXPERIMENT_DIR}"
    : > "${EXPERIMENT_RAW}"
    : > "${EXPERIMENT_SUMMARY}"
    : > "${EXPERIMENT_LOG}"

    experiment_log "EXPERIMENT=${EXPERIMENT_NAME}"
    experiment_log "STARTED=$(date)"
    experiment_log "RESULT_DIR=${EXPERIMENT_DIR}"
}

append_csv_with_metadata()
{
    local source_csv="$1"
    local destination_csv="$2"
    local series="$3"
    local parameter_name="$4"
    local parameter_value="$5"

    if [ ! -s "${source_csv}" ]; then
        return 1
    fi

    if [ ! -s "${destination_csv}" ]; then
        printf 'experiment,series,parameter_name,parameter_value,' >> "${destination_csv}"
        head -n 1 "${source_csv}" >> "${destination_csv}"
    fi

    tail -n +2 "${source_csv}" |
        awk -v experiment="${EXPERIMENT_NAME}" \
            -v series="${series}" \
            -v parameter_name="${parameter_name}" \
            -v parameter_value="${parameter_value}" \
            'BEGIN { OFS="," } { print experiment,series,parameter_name,parameter_value,$0 }' \
            >> "${destination_csv}"
}

run_experiment_case()
{
    local series="$1"
    local parameter_name="$2"
    local parameter_value="$3"
    local result_tag="${CASE_MODE}"
    local runner_raw
    local runner_summary
    local rc
    local quiet_case_output="${EXPERIMENT_QUIET_CASES:-0}"

    if [ "${CASE_MODE}" = "k1-mixed-rvv-ime" ]; then
        result_tag="${CASE_MODE}-${CASE_SCHEDULE}"
    fi

    runner_raw="${COMMON_MODULE_DIR}/results/openmp_raw_latest_${result_tag}.csv"
    runner_summary="${COMMON_MODULE_DIR}/results/openmp_summary_latest_${result_tag}.csv"
    # Never collect an alias left by an older run when the new case stops
    # before producing its own files.
    rm -f "${runner_raw}" "${runner_summary}"

    experiment_log "============================================================"
    experiment_log "CASE series=${series} ${parameter_name}=${parameter_value} mode=${CASE_MODE} threads=${CASE_THREADS} M=${CASE_M} N=${CASE_N} K=${CASE_K} tile_N=${CASE_TILE_N} schedule=${CASE_SCHEDULE} chunk=${CASE_CHUNK} kernel=${CASE_KERNEL}"

    # Keep the detailed runner output in the experiment log when the public
    # script requests a compact terminal table. The raw and summary CSV files
    # are still produced exactly as in the verbose mode.
    if [ "${quiet_case_output}" = "1" ]; then
        (
            export OMP_NUM_THREADS="${CASE_THREADS}"
            export KERNEL_FILTER="${CASE_KERNEL}"
            export KIND_FILTER="${CASE_KIND}"
            export GEMM_TILE_SCHEDULE="${CASE_SCHEDULE}"
            export GEMM_DYNAMIC_CHUNK="${CASE_CHUNK}"
            export MIXED_IME_TILE_WEIGHT="${CASE_IME_WEIGHT}"
            export MIXED_RVV_TILE_WEIGHT="${CASE_RVV_WEIGHT}"
            export ENABLE_MF2="${CASE_ENABLE_MF2}"
            export PERF_STAT="${CASE_PERF_STAT}"
            export PERF_EVENTS="${CASE_PERF_EVENTS}"
            export GEMM_VALIDATE=1
            export VALIDATE_EACH_RUN=0
            bash "${COMMON_RUNNER}" "${CASE_MODE}" \
                 "${CASE_M}" "${CASE_N}" "${CASE_K}" \
                 "${CASE_TILE_N}" "${CASE_RUNS}"
        ) >> "${EXPERIMENT_LOG}" 2>&1
        rc=$?
    else
        (
            export OMP_NUM_THREADS="${CASE_THREADS}"
            export KERNEL_FILTER="${CASE_KERNEL}"
            export KIND_FILTER="${CASE_KIND}"
            export GEMM_TILE_SCHEDULE="${CASE_SCHEDULE}"
            export GEMM_DYNAMIC_CHUNK="${CASE_CHUNK}"
            export MIXED_IME_TILE_WEIGHT="${CASE_IME_WEIGHT}"
            export MIXED_RVV_TILE_WEIGHT="${CASE_RVV_WEIGHT}"
            export ENABLE_MF2="${CASE_ENABLE_MF2}"
            export PERF_STAT="${CASE_PERF_STAT}"
            export PERF_EVENTS="${CASE_PERF_EVENTS}"
            export GEMM_VALIDATE=1
            export VALIDATE_EACH_RUN=0
            bash "${COMMON_RUNNER}" "${CASE_MODE}" \
                 "${CASE_M}" "${CASE_N}" "${CASE_K}" \
                 "${CASE_TILE_N}" "${CASE_RUNS}"
        ) 2>&1 | tee -a "${EXPERIMENT_LOG}"
        rc=${PIPESTATUS[0]}
    fi

    if ! append_csv_with_metadata "${runner_raw}" "${EXPERIMENT_RAW}" \
            "${series}" "${parameter_name}" "${parameter_value}"; then
        experiment_log "CASE_RESULT=FAILED reason=missing_raw_csv"
        EXPERIMENT_FAILURES=$((EXPERIMENT_FAILURES + 1))
        return 1
    fi

    if ! append_csv_with_metadata "${runner_summary}" "${EXPERIMENT_SUMMARY}" \
            "${series}" "${parameter_name}" "${parameter_value}"; then
        experiment_log "CASE_RESULT=FAILED reason=missing_summary_csv"
        EXPERIMENT_FAILURES=$((EXPERIMENT_FAILURES + 1))
        return 1
    fi

    if [ "${rc}" -ne 0 ]; then
        experiment_log "CASE_RESULT=FAILED runner_rc=${rc}"
        EXPERIMENT_FAILURES=$((EXPERIMENT_FAILURES + 1))
        return 1
    fi

    experiment_log "CASE_RESULT=OK"
    return 0
}

finish_experiment()
{
    local latest_prefix="${COMMON_RESULT_ROOT}/${EXPERIMENT_NAME}"

    experiment_log "============================================================"
    if [ "${EXPERIMENT_FAILURES}" -eq 0 ]; then
        experiment_log "DONE status=OK"
    else
        experiment_log "DONE status=FAILED failed_cases=${EXPERIMENT_FAILURES}"
    fi
    experiment_log "Raw CSV: ${EXPERIMENT_RAW}"
    experiment_log "Summary CSV: ${EXPERIMENT_SUMMARY}"
    experiment_log "Live log: ${EXPERIMENT_LOG}"

    cp "${EXPERIMENT_RAW}" "${latest_prefix}_raw_latest.csv"
    cp "${EXPERIMENT_SUMMARY}" "${latest_prefix}_summary_latest.csv"
    cp "${EXPERIMENT_LOG}" "${latest_prefix}_latest.log"

    if [ "${EXPERIMENT_FAILURES}" -ne 0 ]; then
        return 1
    fi
    return 0
}
