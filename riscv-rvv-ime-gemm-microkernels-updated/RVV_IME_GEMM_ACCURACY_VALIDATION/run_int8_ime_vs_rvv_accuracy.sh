#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-k1}"
RUNS="${2:-1}"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
BASE_SEED="${BASE_SEED:-0}"
ENABLE_MF2="${ENABLE_MF2:-0}"
INPUT_CLASSES="${INPUT_CLASSES:-full_range_uniform}"
STRICT="${STRICT:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_int8_ime_vs_rvv_accuracy_once.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_DIR="${SCRIPT_DIR}/int8_ime_vs_rvv_accuracy_${MODE}_${M}_${STAMP}"
COMBINED_CSV="${CAMPAIGN_DIR}/int8_ime_vs_rvv_accuracy_summary.csv"
EQUIVALENCE_CSV="${CAMPAIGN_DIR}/int8_ime_vs_rvv_equivalence.csv"
POINTERS="${CAMPAIGN_DIR}/raw_result_directories.txt"
RUN_LOG="${CAMPAIGN_DIR}/int8_ime_vs_rvv_accuracy_live.log"
LATEST_CSV="${SCRIPT_DIR}/int8_ime_vs_rvv_accuracy_summary_${MODE}_latest.csv"
LATEST_EQUIVALENCE_CSV="${SCRIPT_DIR}/int8_ime_vs_rvv_equivalence_${MODE}_latest.csv"
LATEST_LOG="${SCRIPT_DIR}/int8_ime_vs_rvv_accuracy_live_${MODE}_latest.log"
ONCE_LATEST_CSV="${SCRIPT_DIR}/int8_ime_vs_rvv_accuracy_once_summary_${MODE}_latest.csv"
ONCE_RESULT_POINTER="${SCRIPT_DIR}/int8_ime_vs_rvv_accuracy_once_result_${MODE}_latest.txt"
ONCE_STATUS_POINTER="${SCRIPT_DIR}/int8_ime_vs_rvv_accuracy_once_status_${MODE}_latest.txt"

if [ ! -f "${RUNNER}" ]; then
    printf 'Missing runner: %s\n' "${RUNNER}" >&2
    exit 1
fi

case "${ENABLE_MF2}" in
    0|1) ;;
    *) printf 'ENABLE_MF2 must be 0 or 1.\n' >&2; exit 2 ;;
esac

case "${STRICT}" in
    0|1) ;;
    *) printf 'STRICT must be 0 or 1.\n' >&2; exit 2 ;;
esac

mkdir -p "${CAMPAIGN_DIR}"
: > "${POINTERS}"
: > "${RUN_LOG}"
: > "${LATEST_LOG}"

cat <<EOF | tee -a "${RUN_LOG}" "${LATEST_LOG}"
Goal: prove whether IME-native and RVV-fallback produce the same INT8-to-INT32 matrix answer
Method: both paths must match the same independently calculated INT64 trusted answer
MODE=${MODE} M=${M} N=${N} K=${K} RUNS=${RUNS}
INPUT_CLASSES=${INPUT_CLASSES}
BASE_SEED=${BASE_SEED}
ENABLE_MF2=${ENABLE_MF2}
STRICT=${STRICT}
EOF

class_index=0
header_written=0
runner_failure_count=0

for input_class in ${INPUT_CLASSES}; do
    class_seed=$((BASE_SEED + class_index * 100000))

    printf '\n===== INPUT_CLASS=%s BASE_SEED=%s =====\n' \
        "${input_class}" "${class_seed}" | tee -a "${RUN_LOG}" "${LATEST_LOG}"

    rm -f "${ONCE_LATEST_CSV}" "${ONCE_RESULT_POINTER}" "${ONCE_STATUS_POINTER}"

    M="${M}" N="${N}" K="${K}" \
    INPUT_CLASS="${input_class}" \
    BASE_SEED="${class_seed}" \
    ENABLE_MF2="${ENABLE_MF2}" \
    bash "${RUNNER}" "${MODE}" "${RUNS}" | tee -a "${RUN_LOG}" "${LATEST_LOG}"

    latest_result="$(cat "${ONCE_RESULT_POINTER}" 2>/dev/null || true)"
    latest_summary="${ONCE_LATEST_CSV}"
    runner_status="$(cat "${ONCE_STATUS_POINTER}" 2>/dev/null || true)"

    if [ -z "${latest_result}" ] || [ ! -f "${latest_summary}" ]; then
        printf 'Missing result for input class: %s\n' "${input_class}" >&2
        exit 1
    fi

    if [ "${runner_status}" != "PASS" ]; then
        runner_failure_count=$((runner_failure_count + 1))
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

cp "${COMBINED_CSV}" "${LATEST_CSV}"

# Combine all cores and repetitions into one result per kernel configuration.
# A configuration is EQUIVALENT only when every IME run and every RVV run
# independently matched the same trusted INT64 answer.
equivalence_tmp="${EQUIVALENCE_CSV}.tmp"
awk -F, '
    NR == 1 { next }
    {
        kernel = $7
        gsub(/^"|"$/, "", kernel)
        key = kernel SUBSEP $8 SUBSEP $9 SUBSEP $10 SUBSEP $11 SUBSEP \
              $12 SUBSEP $13 SUBSEP $14 SUBSEP $15
        info[key] = kernel "," $8 "," $9 "," $10 "," $11 "," \
                    $12 "," $13 "," $14 "," $15
        seen[key] = 1

        if ($3 == "IME_NATIVE") {
            ime_runs[key]++
            if ($16 == "OK") ime_ok[key]++
        } else if ($3 == "RVV_FALLBACK") {
            rvv_runs[key]++
            if ($16 == "OK") rvv_ok[key]++
        }
    }
    END {
        print "kernel,tile_shape,zvl,lmul,unroll,input_class,matrix_m,matrix_n,matrix_k,ime_runs,ime_ok,rvv_runs,rvv_ok,equivalence_status"
        for (key in seen) {
            status = (ime_runs[key] > 0 && rvv_runs[key] > 0 && \
                      ime_ok[key] == ime_runs[key] && \
                      rvv_ok[key] == rvv_runs[key]) \
                     ? "EQUIVALENT" : "NOT_VALIDATED"
            print info[key] "," ime_runs[key] "," ime_ok[key] "," \
                  rvv_runs[key] "," rvv_ok[key] "," status
        }
    }
' "${COMBINED_CSV}" > "${equivalence_tmp}"

{
    head -n 1 "${equivalence_tmp}"
    tail -n +2 "${equivalence_tmp}" | sort
} > "${EQUIVALENCE_CSV}"
rm -f "${equivalence_tmp}"
cp "${EQUIVALENCE_CSV}" "${LATEST_EQUIVALENCE_CSV}"

equivalent_configs="$(awk -F, 'NR > 1 && $14 == "EQUIVALENT" {n++} END {print n+0}' "${EQUIVALENCE_CSV}")"
not_validated_configs="$(awk -F, 'NR > 1 && $14 != "EQUIVALENT" {n++} END {print n+0}' "${EQUIVALENCE_CSV}")"
failed_rows="$(awk -F, 'NR > 1 && $16 != "OK" {n++} END {print n+0}' "${COMBINED_CSV}")"

if [ "${equivalent_configs}" -gt 0 ] && \
   [ "${not_validated_configs}" -eq 0 ] && \
   [ "${failed_rows}" -eq 0 ] && \
   [ "${runner_failure_count}" -eq 0 ]; then
    overall_status="PASS"
else
    overall_status="FAIL"
fi

cat <<EOF | tee -a "${RUN_LOG}" "${LATEST_LOG}"
DONE
OVERALL_STATUS=${overall_status}
equivalent_configurations=${equivalent_configs}
not_validated_configurations=${not_validated_configs}
failed_detailed_rows=${failed_rows}
failed_input_class_runners=${runner_failure_count}
Combined summary CSV: ${COMBINED_CSV}
Latest summary CSV: ${LATEST_CSV}
Equivalence CSV: ${EQUIVALENCE_CSV}
Latest equivalence CSV: ${LATEST_EQUIVALENCE_CSV}
Per-input summaries: ${CAMPAIGN_DIR}
Raw result directory list: ${POINTERS}
EOF

if [ "${STRICT}" = "1" ] && [ "${overall_status}" != "PASS" ]; then
    exit 1
fi
