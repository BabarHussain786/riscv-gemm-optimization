#!/usr/bin/env bash
set -u

# Professor-facing K1 OpenMP campaign launcher.
# It runs homogeneous RVV, homogeneous IME, and heterogeneous IME/RVV modes
# with one combined live log and combined CSV files for direct comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_openmp_tiled_gemm_mode.sh"
RESULT_ROOT="${MODULE_DIR}/results"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
TILE_N_RVV="${TILE_N_RVV:-64}"
TILE_N_IME="${TILE_N_IME:-64}"
TILE_N_MIXED="${TILE_N_MIXED:-32}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_DIR="${RESULT_ROOT}/k1_openmp_heterogeneous_campaign_${M}_${STAMP}"
CAMPAIGN_LOG="${CAMPAIGN_DIR}/k1_openmp_heterogeneous_live_${M}_runs${RUNS}_${STAMP}.log"
COMBINED_RAW="${CAMPAIGN_DIR}/k1_openmp_heterogeneous_raw_${M}_runs${RUNS}_${STAMP}.csv"
COMBINED_SUMMARY="${CAMPAIGN_DIR}/k1_openmp_heterogeneous_summary_${M}_runs${RUNS}_${STAMP}.csv"

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
    local rc=0

    log_campaign "============================================================"
    log_campaign "${label}"
    log_campaign "MODE=${mode} M=${M} N=${N} K=${K} tile_N=${tile_n} runs=${RUNS}"
    log_campaign "============================================================"

    bash "${RUNNER}" "${mode}" "${M}" "${N}" "${K}" "${tile_n}" "${RUNS}" 2>&1 | tee -a "${CAMPAIGN_LOG}"
    rc=${PIPESTATUS[0]}

    append_csv_with_single_header "${RESULT_ROOT}/openmp_raw_latest_${mode}.csv" "${COMBINED_RAW}"
    append_csv_with_single_header "${RESULT_ROOT}/openmp_summary_latest_${mode}.csv" "${COMBINED_SUMMARY}"

    if [ "${rc}" -eq 0 ]; then
        log_campaign "MODE_DONE=${mode} status=OK"
    else
        log_campaign "MODE_DONE=${mode} status=CHECK_CSV rc=${rc}"
    fi

    return "${rc}"
}

log_campaign "K1 OpenMP heterogeneous-versus-baseline GEMM campaign"
log_campaign "Matrix: M=${M} N=${N} K=${K} runs=${RUNS}"
log_campaign "Comparison modes: RVV all-core, RVV-cluster, IME-cluster, mixed IME/RVV"
log_campaign "Mixed policy: IME tile weight=${MIXED_IME_TILE_WEIGHT:-4}, RVV tile weight=${MIXED_RVV_TILE_WEIGHT:-1}"
log_campaign "Output folder: ${CAMPAIGN_DIR}"

status=0

run_mode k1-rvv "${TILE_N_RVV}" "All-core RVV OpenMP baseline: cores 0-7 execute RVV kernels" || status=1
run_mode k1-rvv-only "${TILE_N_RVV}" "RVV-cluster OpenMP baseline: cores 4-7 execute RVV kernels" || status=1
run_mode k1-ime "${TILE_N_IME}" "IME-cluster OpenMP baseline: cores 0-3 execute native IME kernels" || status=1

export MIXED_IME_TILE_WEIGHT="${MIXED_IME_TILE_WEIGHT:-4}"
export MIXED_RVV_TILE_WEIGHT="${MIXED_RVV_TILE_WEIGHT:-1}"
run_mode k1-mixed-rvv-ime "${TILE_N_MIXED}" "Heterogeneous OpenMP run: cores 0-3 use IME and cores 4-7 use RVV fallback" || status=1

cp "${CAMPAIGN_LOG}" "${RESULT_ROOT}/k1_openmp_heterogeneous_live_latest.log"
cp "${COMBINED_RAW}" "${RESULT_ROOT}/k1_openmp_heterogeneous_raw_latest.csv"
cp "${COMBINED_SUMMARY}" "${RESULT_ROOT}/k1_openmp_heterogeneous_summary_latest.csv"

log_campaign "============================================================"
log_campaign "DONE"
log_campaign "Combined live log: ${CAMPAIGN_LOG}"
log_campaign "Combined raw CSV: ${COMBINED_RAW}"
log_campaign "Combined summary CSV: ${COMBINED_SUMMARY}"
log_campaign "Latest live log: ${RESULT_ROOT}/k1_openmp_heterogeneous_live_latest.log"
log_campaign "Latest raw CSV: ${RESULT_ROOT}/k1_openmp_heterogeneous_raw_latest.csv"
log_campaign "Latest summary CSV: ${RESULT_ROOT}/k1_openmp_heterogeneous_summary_latest.csv"
log_campaign "============================================================"

exit "${status}"