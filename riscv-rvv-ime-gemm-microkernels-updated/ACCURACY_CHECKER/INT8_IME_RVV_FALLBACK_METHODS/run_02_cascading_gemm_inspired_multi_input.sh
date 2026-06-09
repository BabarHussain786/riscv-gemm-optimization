#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-k1}"
RUNS="${2:-1}"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
BASE_SEED="${BASE_SEED:-0}"
ENABLE_MF2="${ENABLE_MF2:-1}"
INPUT_CLASSES="${INPUT_CLASSES:-bounded_uniform full_range_uniform mixed_magnitude cancellation_stress}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCURACY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${ACCURACY_DIR}/run_ime_vs_rvv_fallback_histograms.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_DIR="${SCRIPT_DIR}/results_02_multi_input_${MODE}_${M}_${STAMP}"
COMBINED_CSV="${CAMPAIGN_DIR}/multi_input_combined_summary.csv"
POINTERS="${CAMPAIGN_DIR}/raw_result_directories.txt"
RUN_LOG="${CAMPAIGN_DIR}/multi_input_live.log"

if [ ! -f "${RUNNER}" ]; then
    printf 'Missing runner: %s\n' "${RUNNER}" >&2
    exit 1
fi

mkdir -p "${CAMPAIGN_DIR}"
: > "${POINTERS}"
: > "${RUN_LOG}"

cat <<EOF | tee -a "${RUN_LOG}"
Method: multi-input reference-based exact comparison
Scope: IME native and RVV fallback INT8-to-INT32 kernels only
MODE=${MODE} M=${M} N=${N} K=${K} RUNS=${RUNS}
INPUT_CLASSES=${INPUT_CLASSES}
BASE_SEED=${BASE_SEED}
ENABLE_MF2=${ENABLE_MF2}
EOF

class_index=0
header_written=0

for input_class in ${INPUT_CLASSES}; do
    class_seed=$((BASE_SEED + class_index * 100000))

    printf '\n===== INPUT_CLASS=%s BASE_SEED=%s =====\n' \
        "${input_class}" "${class_seed}" | tee -a "${RUN_LOG}"

    M="${M}" N="${N}" K="${K}" \
    INPUT_CLASS="${input_class}" \
    BASE_SEED="${class_seed}" \
    ENABLE_MF2="${ENABLE_MF2}" \
    bash "${RUNNER}" "${MODE}" "${RUNS}" | tee -a "${RUN_LOG}"

    latest_result="$(ls -dt "${ACCURACY_DIR}"/ime_vs_rvv_fallback_histograms_"${MODE}"_"${M}"_* 2>/dev/null | head -n 1)"
    latest_summary="${ACCURACY_DIR}/ime_vs_rvv_fallback_summary_${MODE}_latest.csv"

    if [ -z "${latest_result}" ] || [ ! -f "${latest_summary}" ]; then
        printf 'Missing result for input class: %s\n' "${input_class}" >&2
        exit 1
    fi

    cp "${latest_summary}" "${CAMPAIGN_DIR}/${input_class}_summary.csv"
    printf '%s,%s\n' "${input_class}" "${latest_result}" >> "${POINTERS}"

    if [ "${header_written}" -eq 0 ]; then
        head -n 1 "${latest_summary}" > "${COMBINED_CSV}"
        header_written=1
    fi
    tail -n +2 "${latest_summary}" >> "${COMBINED_CSV}"

    class_index=$((class_index + 1))
done

cat <<EOF | tee -a "${RUN_LOG}"
DONE
Combined summary CSV: ${COMBINED_CSV}
Per-input summaries: ${CAMPAIGN_DIR}
Raw result directory list: ${POINTERS}
EOF
