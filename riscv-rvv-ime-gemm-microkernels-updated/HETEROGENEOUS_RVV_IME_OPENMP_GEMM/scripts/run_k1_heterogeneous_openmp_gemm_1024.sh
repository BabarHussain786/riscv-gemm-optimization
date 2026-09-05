#!/usr/bin/env bash
set -u

# CAMPAIGN ROADMAP
# Step 1 -> Run the all-core and cluster-only baselines.
# Step 2 -> Run heterogeneous static scheduling.
# Step 3 -> Run heterogeneous dynamic scheduling when RUN_DYNAMIC=1.
# Step 4 -> Merge all raw and summary CSV files with one header.
# Step 5 -> Publish one combined log and two combined analysis files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_openmp_tiled_gemm_mode.sh"
RESULT_ROOT="${MODULE_DIR}/results"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
TILE_N="${TILE_N:-32}"
DYNAMIC_CHUNK="${DYNAMIC_CHUNK:-1}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
ZVL_FILTER="256b"
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_DIR="${RESULT_ROOT}/k1_openmp_heterogeneous_campaign_${M}_${STAMP}"
CAMPAIGN_LOG="${CAMPAIGN_DIR}/k1_openmp_heterogeneous_live_${M}_runs${RUNS}_${STAMP}.log"
COMBINED_RAW="${CAMPAIGN_DIR}/k1_openmp_heterogeneous_raw_${M}_runs${RUNS}_${STAMP}.csv"
COMBINED_SUMMARY="${CAMPAIGN_DIR}/k1_openmp_heterogeneous_summary_${M}_runs${RUNS}_${STAMP}.csv"

case "${RUN_DYNAMIC}" in
    0|1) ;;
    *)
        echo "RUN_DYNAMIC must be 0 or 1"
        exit 1
        ;;
esac

mkdir -p "${RESULT_ROOT}" "${CAMPAIGN_DIR}"
: > "${CAMPAIGN_LOG}"
: > "${COMBINED_RAW}"
: > "${COMBINED_SUMMARY}"

log_campaign()
{
    printf '%s\n' "$*" | tee -a "${CAMPAIGN_LOG}"
}

append_csv_with_single_header()
{
    local src="$1"
    local dst="$2"

    if [ ! -f "${src}" ]; then
        return 0
    fi

    if [ ! -s "${dst}" ]; then
        cat "${src}" >> "${dst}"
    else
        tail -n +2 "${src}" >> "${dst}"
    fi
}

run_mode()
{
    local mode="$1"
    local tile_n="$2"
    local label="$3"
    local schedule_policy="${4:-static}"
    local dynamic_chunk="${5:-1}"
    local result_tag="${mode}"
    local rc=0

    if [ "${mode}" = "k1-mixed-rvv-ime" ]; then
        result_tag="${mode}-${schedule_policy}"
    fi

    log_campaign "============================================================"
    log_campaign "${label}"
    log_campaign "MODE=${mode} schedule=${schedule_policy} chunk=${dynamic_chunk} M=${M} N=${N} K=${K} tile_N=${tile_n} runs=${RUNS}"
    log_campaign "============================================================"

    # Remove old aliases so a failed mode can never reuse stale CSV data.
    rm -f "${RESULT_ROOT}/openmp_raw_latest_${result_tag}.csv" \
          "${RESULT_ROOT}/openmp_summary_latest_${result_tag}.csv" \
          "${RESULT_ROOT}/openmp_live_latest_${result_tag}.log"

    GEMM_TILE_SCHEDULE="${schedule_policy}" \
    GEMM_DYNAMIC_CHUNK="${dynamic_chunk}" \
        bash "${RUNNER}" "${mode}" "${M}" "${N}" "${K}" \
             "${tile_n}" "${RUNS}" 2>&1 | tee -a "${CAMPAIGN_LOG}"
    rc=${PIPESTATUS[0]}

    append_csv_with_single_header "${RESULT_ROOT}/openmp_raw_latest_${result_tag}.csv" "${COMBINED_RAW}"
    append_csv_with_single_header "${RESULT_ROOT}/openmp_summary_latest_${result_tag}.csv" "${COMBINED_SUMMARY}"

    if [ "${rc}" -eq 0 ]; then
        log_campaign "MODE_DONE=${mode} schedule=${schedule_policy} status=OK"
    else
        log_campaign "MODE_DONE=${mode} schedule=${schedule_policy} status=FAILED rc=${rc}"
    fi

    return "${rc}"
}

log_campaign "K1 OpenMP heterogeneous-versus-baseline GEMM campaign"
log_campaign "Matrix: M=${M} N=${N} K=${K} runs=${RUNS}"
log_campaign "Comparison modes: RVV all-core, RVV-cluster, IME-cluster, mixed static, mixed dynamic"
log_campaign "Common OpenMP tile width: tile_N=${TILE_N}"
log_campaign "Static mixed split: IME weight=${MIXED_IME_TILE_WEIGHT:-4}, RVV weight=${MIXED_RVV_TILE_WEIGHT:-1}"
log_campaign "Dynamic mixed chunk: ${DYNAMIC_CHUNK}; enabled=${RUN_DYNAMIC}"
log_campaign "Kernel filter: ZVL=${ZVL_FILTER}"
log_campaign "Output folder: ${CAMPAIGN_DIR}"

export ZVL_FILTER

status=0

run_mode k1-rvv "${TILE_N}" "All-core RVV OpenMP baseline: cores 0-7 execute RVV kernels" || status=1
run_mode k1-rvv-only "${TILE_N}" "RVV-cluster OpenMP baseline: cores 4-7 execute RVV kernels" || status=1
run_mode k1-ime "${TILE_N}" "IME-cluster OpenMP baseline: cores 0-3 execute native IME kernels" || status=1

export MIXED_IME_TILE_WEIGHT="${MIXED_IME_TILE_WEIGHT:-4}"
export MIXED_RVV_TILE_WEIGHT="${MIXED_RVV_TILE_WEIGHT:-1}"
run_mode k1-mixed-rvv-ime "${TILE_N}" \
    "Heterogeneous static run: fixed IME and RVV tile ranges" \
    static 1 || status=1

if [ "${RUN_DYNAMIC}" = "1" ]; then
    run_mode k1-mixed-rvv-ime "${TILE_N}" \
        "Heterogeneous dynamic run: OpenMP assigns the next available tile chunk" \
        dynamic "${DYNAMIC_CHUNK}" || status=1
fi

log_campaign "============================================================"
if [ "${status}" -eq 0 ]; then
    log_campaign "DONE status=OK"
else
    log_campaign "DONE status=FAILED"
fi
log_campaign "Combined live log: ${CAMPAIGN_LOG}"
log_campaign "Combined raw CSV: ${COMBINED_RAW}"
log_campaign "Combined summary CSV: ${COMBINED_SUMMARY}"
log_campaign "Latest live log: ${RESULT_ROOT}/k1_openmp_heterogeneous_live_latest.log"
log_campaign "Latest raw CSV: ${RESULT_ROOT}/k1_openmp_heterogeneous_raw_latest.csv"
log_campaign "Latest summary CSV: ${RESULT_ROOT}/k1_openmp_heterogeneous_summary_latest.csv"
log_campaign "============================================================"

# Publish aliases only after the complete campaign footer is in the log.
cp "${CAMPAIGN_LOG}" "${RESULT_ROOT}/k1_openmp_heterogeneous_live_latest.log"
cp "${COMBINED_RAW}" "${RESULT_ROOT}/k1_openmp_heterogeneous_raw_latest.csv"
cp "${COMBINED_SUMMARY}" "${RESULT_ROOT}/k1_openmp_heterogeneous_summary_latest.csv"

exit "${status}"
