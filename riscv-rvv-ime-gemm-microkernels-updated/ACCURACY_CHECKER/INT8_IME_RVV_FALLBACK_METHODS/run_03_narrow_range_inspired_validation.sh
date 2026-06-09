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
CAMPAIGN_DIR="${SCRIPT_DIR}/results_03_range_normalized_${MODE}_${M}_${STAMP}"
RUN_LOG="${CAMPAIGN_DIR}/range_normalized_live.log"
NORMALIZED_CSV="${CAMPAIGN_DIR}/range_normalized_summary.csv"

if [ ! -f "${RUNNER}" ]; then
    printf 'Missing runner: %s\n' "${RUNNER}" >&2
    exit 1
fi

mkdir -p "${CAMPAIGN_DIR}"

cat <<EOF | tee "${RUN_LOG}"
Method: range, overflow, and normalized-error validation
Scope: IME native and RVV fallback INT8-to-INT32 kernels only
MODE=${MODE} M=${M} N=${N} K=${K} RUNS=${RUNS}
INPUT_CLASS=full_range_uniform
BASE_SEED=${BASE_SEED}
ENABLE_MF2=${ENABLE_MF2}
Theoretical full-range bound: K * 128 * 128
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

awk -F, '
BEGIN {
    OFS=","
    print "timestamp,mode,path,core,seed,kernel_family,kernel,tile_shape,zvl,lmul,unroll,status,total_elements,mismatch_count,mismatch_rate,exact_match_rate,max_absolute_error,theoretical_max_abs_output,int32_positive_limit,int32_safety_margin,normalized_max_error,overflow_count,pass"
}
NR > 1 {
    for (i = 1; i <= NF; ++i) {
        gsub(/^"|"$/, "", $i)
    }

    theoretical_bound = $15 * 128 * 128
    int32_limit = 2147483647
    safety_margin = int32_limit - theoretical_bound

    if ($18 ~ /^[0-9]+$/ && $19 ~ /^[0-9]+$/) {
        mismatch_rate = $18 > 0 ? $19 / $18 : 0
    } else {
        mismatch_rate = "NA"
    }

    if ($20 ~ /^[0-9]+$/ && theoretical_bound > 0) {
        normalized_error = $20 / theoretical_bound
    } else {
        normalized_error = "NA"
    }

    pass = ($16 == "OK" && $21 == 0) ? 1 : 0

    print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$16,$18,$19, \
          mismatch_rate,$22,$20,theoretical_bound,int32_limit, \
          safety_margin,normalized_error,$21,pass
}
' "${LATEST_SUMMARY}" > "${NORMALIZED_CSV}"

cp "${LATEST_SUMMARY}" "${CAMPAIGN_DIR}/raw_reference_summary.csv"
printf '%s\n' "${LATEST_RESULT}" > "${CAMPAIGN_DIR}/raw_result_directory.txt"

cat <<EOF | tee -a "${RUN_LOG}"
DONE
Range-normalized summary CSV: ${NORMALIZED_CSV}
Raw reference summary CSV: ${CAMPAIGN_DIR}/raw_reference_summary.csv
Raw signed-error histograms: ${LATEST_RESULT}/histograms
EOF
