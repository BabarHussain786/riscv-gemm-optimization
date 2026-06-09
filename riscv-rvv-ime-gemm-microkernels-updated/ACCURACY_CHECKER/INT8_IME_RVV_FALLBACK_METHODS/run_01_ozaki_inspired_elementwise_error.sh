#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-k1}"
RUNS="${2:-1}"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
BASE_SEED="${BASE_SEED:-0}"
ENABLE_MF2="${ENABLE_MF2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCURACY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${ACCURACY_DIR}/run_ime_vs_rvv_fallback_histograms.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_DIR="${SCRIPT_DIR}/results_01_elementwise_exact_${MODE}_${M}_${STAMP}"
RUN_LOG="${CAMPAIGN_DIR}/elementwise_exact_live.log"

if [ ! -f "${RUNNER}" ]; then
    printf 'Missing runner: %s\n' "${RUNNER}" >&2
    exit 1
fi

mkdir -p "${CAMPAIGN_DIR}"

cat <<EOF | tee "${RUN_LOG}"
Method: reference-based element-wise exact-error validation
Scope: IME native and RVV fallback INT8-to-INT32 kernels only
MODE=${MODE} M=${M} N=${N} K=${K} RUNS=${RUNS}
INPUT_CLASS=full_range_uniform
BASE_SEED=${BASE_SEED}
ENABLE_MF2=${ENABLE_MF2}
EOF

M="${M}" N="${N}" K="${K}" \
INPUT_CLASS="full_range_uniform" \
BASE_SEED="${BASE_SEED}" \
ENABLE_MF2="${ENABLE_MF2}" \
bash "${RUNNER}" "${MODE}" "${RUNS}" | tee -a "${RUN_LOG}"

LATEST_RESULT="$(ls -dt "${ACCURACY_DIR}"/ime_vs_rvv_fallback_histograms_"${MODE}"_"${M}"_* 2>/dev/null | head -n 1)"
LATEST_SUMMARY="${ACCURACY_DIR}/ime_vs_rvv_fallback_summary_${MODE}_latest.csv"

if [ -z "${LATEST_RESULT}" ] || [ ! -f "${LATEST_SUMMARY}" ]; then
    printf 'Runner completed but expected result files were not found.\n' >&2
    exit 1
fi

cp "${LATEST_SUMMARY}" "${CAMPAIGN_DIR}/elementwise_exact_summary.csv"
printf '%s\n' "${LATEST_RESULT}" > "${CAMPAIGN_DIR}/raw_result_directory.txt"

cat <<EOF | tee -a "${RUN_LOG}"
DONE
Summary CSV: ${CAMPAIGN_DIR}/elementwise_exact_summary.csv
Raw signed-error histograms: ${LATEST_RESULT}/histograms
Raw logs: ${LATEST_RESULT}/logs
EOF
