#!/usr/bin/env bash
set -u

MODE="${1:-k1}"
RUNS="${2:-1}"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
INPUT_CLASS="${INPUT_CLASS:-full_range_uniform}"
BASE_SEED="${BASE_SEED:-0}"
ENABLE_MF2="${ENABLE_MF2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS="${SCRIPT_DIR}/ime_rvv_fallback_diff_histogram_check.c"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${SCRIPT_DIR}/ime_vs_rvv_fallback_histograms_${MODE}_${M}_${STAMP}"
BUILD_DIR="${RESULT_DIR}/build"
HIST_DIR="${RESULT_DIR}/histograms"
LOG_DIR="${RESULT_DIR}/logs"
CSV="${RESULT_DIR}/ime_vs_rvv_fallback_summary_${MODE}_${M}_${STAMP}.csv"
LATEST_CSV="${SCRIPT_DIR}/ime_vs_rvv_fallback_summary_${MODE}_latest.csv"
LATEST_LOG="${SCRIPT_DIR}/ime_vs_rvv_fallback_live_${MODE}_latest.log"

CC="${CC:-gcc}"
ABI="${ABI:-lp64d}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O3 -std=c11 -Wall -Wextra -Wno-unknown-pragmas}"

usage()
{
    cat <<'EOF'
Usage:
  bash ACCURACY_CHECKER/run_ime_vs_rvv_fallback_histograms.sh MODE [RUNS]

MODE:
  k1   IME cores 0-3, RVV fallback core 4
  k3   IME cores 8-15, RVV fallback core 0

Environment options:
  M N K                         default 1024 1024 1024
  INPUT_CLASS                   default full_range_uniform
  BASE_SEED                     default 0
  ENABLE_MF2=1     include LMUL mf2 native IME kernels
EOF
}

case "${MODE}" in
    k1)
        IME_CORES="${IME_CORES:-0 1 2 3}"
        RVV_CORES="${RVV_CORES:-4}"
        ;;
    k3)
        IME_CORES="${IME_CORES:-8 9 10 11 12 13 14 15}"
        RVV_CORES="${RVV_CORES:-0}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

case "${RUNS}" in
    ''|*[!0-9]*|0)
        printf 'RUNS must be a positive integer.\n' >&2
        exit 2
        ;;
esac

if [ ! -f "${HARNESS}" ]; then
    printf 'Missing harness: %s\n' "${HARNESS}" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}" "${HIST_DIR}" "${LOG_DIR}"
: > "${LATEST_LOG}"

log()
{
    printf '%s\n' "$*" | tee -a "${LATEST_LOG}"
}

csv_quote()
{
    printf '%s' "$1" | sed 's/"/""/g; s/^/"/; s/$/"/'
}

kernel_field()
{
    local kernel="$1"
    local regex="$2"
    printf '%s\n' "${kernel}" | sed -n "${regex}"
}

march_for_zvl()
{
    case "$1" in
        128b) printf 'rv64gcv_zvl128b' ;;
        256b) printf 'rv64gcv_zvl256b' ;;
        *) return 1 ;;
    esac
}

input_contract_for_tile()
{
    case "$1" in
        8x4) printf 'FULL_MATRIX' ;;
        8x8) printf 'PREPACKED_ROWS' ;;
        *) return 1 ;;
    esac
}

compile_check()
{
    local mode_kind="$1"
    local march="$2"
    local symbol="$3"
    local mr="$4"
    local nr="$5"
    local input_contract="$6"
    local kernel_c="$7"
    local fallback_c="$8"
    local exe="$9"
    local build_log="${10}"
    local defines=()

    defines+=("-DACC_KIND_INT8_IME")
    defines+=("-DKERNEL_SYMBOL=${symbol}")
    defines+=("-DACC_MR=${mr}")
    defines+=("-DACC_NR=${nr}")

    case "${input_contract}" in
        FULL_MATRIX) defines+=("-DACC_IME_INPUT_FULL_MATRIX=1") ;;
        PREPACKED_ROWS) defines+=("-DACC_IME_INPUT_PREPACKED_ROWS=1") ;;
        *) return 1 ;;
    esac

    if [ "${mode_kind}" = "ime_native" ]; then
        defines+=("-DSPACEMIT_IME_REQUIRE_HARDWARE=1")
        if [ "${ENABLE_MF2}" = "1" ] &&
           printf '%s\n' "${symbol}" | grep -q '_lmulmf2_'; then
            defines+=("-DSPACEMIT_IME_ENABLE_MF2_NATIVE=1")
        fi
    fi

    "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
        "${defines[@]}" "${HARNESS}" "${kernel_c}" "${fallback_c}" \
        -lm -o "${exe}" > "${build_log}" 2>&1
}

run_one()
{
    local exe="$1"
    local core="$2"
    local seed="$3"
    local diff_csv="$4"
    local input_csv="$5"
    local run_log="$6"

    taskset -c "${core}" "${exe}" "${INPUT_CLASS}" "${M}" "${N}" "${K}" \
        "${seed}" "${diff_csv}" "${input_csv}" > "${run_log}" 2>&1
}

printf 'timestamp,mode,path,core,seed,kernel_family,kernel,tile_shape,zvl,lmul,unroll,input_class,matrix_m,matrix_n,matrix_k,status,return_code,total_elements,mismatch_count,max_integer_difference,overflow_count,exact_match_rate,diff_histogram,input_histogram,log_file\n' > "${CSV}"

log "IME native vs RVV fallback histogram accuracy"
log "Method: controlled input, reference C, element-wise C_kernel - C_ref histogram."
log "This follows the paper-style reference comparison principle, adapted to INT8-to-INT32 RVV/IME GEMM."
log "MODE=${MODE} M=${M} N=${N} K=${K} RUNS=${RUNS}"
log "INPUT_CLASS=${INPUT_CLASS}"
log "BASE_SEED=${BASE_SEED}"
log "ENABLE_MF2=${ENABLE_MF2}"
log "IME_CORES=${IME_CORES}"
log "RVV_CORES=${RVV_CORES}"
log "RESULT_DIR=${RESULT_DIR}"
grep Cpus_allowed_list /proc/self/status 2>/dev/null | tee -a "${LATEST_LOG}" || true

kernel_count=0
built_count=0
ok_count=0
build_failed_count=0
run_failed_count=0

while IFS= read -r kernel_dir; do
    kernel_count=$((kernel_count + 1))
    kernel="$(basename "${kernel_dir}")"
    family="$(basename "$(dirname "${kernel_dir}")")"
    tile_shape="$(kernel_field "${kernel}" 's/.*ime_kernel_\([0-9]x[0-9]\)_.*/\1/p')"
    zvl="$(kernel_field "${kernel}" 's/.*_zvl\([0-9]*b\)_.*/\1/p')"
    lmul="$(kernel_field "${kernel}" 's/.*_\(lmul[^_]*\)_.*/\1/p')"
    unroll="$(kernel_field "${kernel}" 's/.*_unroll\([0-9]*\).*/\1/p')"
    mr="${tile_shape%x*}"
    nr="${tile_shape#*x}"
    symbol="${kernel}"
    march="$(march_for_zvl "${zvl}")"
    input_contract="$(input_contract_for_tile "${tile_shape}")"
    kernel_c="${kernel_dir}/${kernel}.c"
    fallback_c="${kernel_dir}/rvv_fallback.c"
    safe_kernel="${family}_${kernel}"

    log "============================================================"
    log "KERNEL=${kernel}"
    log "FAMILY=${family}"
    log "TILE=${tile_shape} ZVL=${zvl} LMUL=${lmul} UNROLL=${unroll}"
    log "INPUT_CONTRACT=${input_contract}"

    if [ ! -f "${kernel_c}" ] || [ ! -f "${fallback_c}" ]; then
        log "SKIP missing source: ${kernel_c} or ${fallback_c}"
        continue
    fi

    ime_exe="${BUILD_DIR}/${safe_kernel}_ime_native"
    rvv_exe="${BUILD_DIR}/${safe_kernel}_rvv_fallback"
    ime_build_log="${LOG_DIR}/${safe_kernel}_ime_native_build.log"
    rvv_build_log="${LOG_DIR}/${safe_kernel}_rvv_fallback_build.log"

    if compile_check "ime_native" "${march}" "${symbol}" "${mr}" "${nr}" \
        "${input_contract}" "${kernel_c}" "${fallback_c}" "${ime_exe}" "${ime_build_log}"; then
        built_count=$((built_count + 1))
    else
        build_failed_count=$((build_failed_count + 1))
        log "IME_NATIVE BUILD_FAILED: ${ime_build_log}"
        {
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                "$(date -Iseconds)" "${MODE}" "IME_NATIVE" "NA" "NA" \
                "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" \
                "${INPUT_CLASS}" "${M}" "${N}" "${K}" "BUILD_FAILED" "NA" "NA" \
                "NA" "NA" "NA" "NA"
            printf '%s,%s,%s\n' "$(csv_quote "NA")" "$(csv_quote "NA")" "$(csv_quote "${ime_build_log}")"
        } >> "${CSV}"
        ime_exe=""
    fi

    if compile_check "rvv_fallback" "${march}" "${symbol}" "${mr}" "${nr}" \
        "${input_contract}" "${kernel_c}" "${fallback_c}" "${rvv_exe}" "${rvv_build_log}"; then
        built_count=$((built_count + 1))
    else
        build_failed_count=$((build_failed_count + 1))
        log "RVV_FALLBACK BUILD_FAILED: ${rvv_build_log}"
        {
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                "$(date -Iseconds)" "${MODE}" "RVV_FALLBACK" "NA" "NA" \
                "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" \
                "${INPUT_CLASS}" "${M}" "${N}" "${K}" "BUILD_FAILED" "NA" "NA" \
                "NA" "NA" "NA" "NA"
            printf '%s,%s,%s\n' "$(csv_quote "NA")" "$(csv_quote "NA")" "$(csv_quote "${rvv_build_log}")"
        } >> "${CSV}"
        rvv_exe=""
    fi

    for run in $(seq 1 "${RUNS}"); do
        seed=$((BASE_SEED + run))

        if [ -n "${ime_exe}" ]; then
            for core in ${IME_CORES}; do
                diff_csv="${HIST_DIR}/${safe_kernel}_ime_native_core${core}_run${run}_diff.csv"
                input_csv="${HIST_DIR}/${safe_kernel}_ime_native_core${core}_run${run}_input.csv"
                run_log="${LOG_DIR}/${safe_kernel}_ime_native_core${core}_run${run}.log"

                if run_one "${ime_exe}" "${core}" "${seed}" "${diff_csv}" "${input_csv}" "${run_log}"; then
                    status="$(awk -F, 'NR==2 {print $1}' "${run_log}")"
                    return_code="$(awk -F, 'NR==2 {print $2}' "${run_log}")"
                    total_elements="$(awk -F, 'NR==2 {print $3}' "${run_log}")"
                    mismatch_count="$(awk -F, 'NR==2 {print $4}' "${run_log}")"
                    max_integer_difference="$(awk -F, 'NR==2 {print $5}' "${run_log}")"
                    overflow_count="$(awk -F, 'NR==2 {print $6}' "${run_log}")"
                    exact_match_rate="$(awk -F, 'NR==2 {print $7}' "${run_log}")"
                    [ "${status}" = "OK" ] && ok_count=$((ok_count + 1)) || run_failed_count=$((run_failed_count + 1))
                else
                    status="RUN_FAILED"
                    return_code="NA"
                    total_elements="NA"
                    mismatch_count="NA"
                    max_integer_difference="NA"
                    overflow_count="NA"
                    exact_match_rate="NA"
                    run_failed_count=$((run_failed_count + 1))
                fi

                log "  IME_NATIVE core ${core} run ${run} seed ${seed}: ${status}"
                {
                    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                        "$(date -Iseconds)" "${MODE}" "IME_NATIVE" "${core}" "${seed}" \
                        "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" \
                        "${INPUT_CLASS}" "${M}" "${N}" "${K}" "${status}" "${return_code}" "${total_elements}" \
                        "${mismatch_count}" "${max_integer_difference}" "${overflow_count}" "${exact_match_rate}"
                    printf '%s,%s,%s\n' "$(csv_quote "${diff_csv}")" "$(csv_quote "${input_csv}")" "$(csv_quote "${run_log}")"
                } >> "${CSV}"
            done
        fi

        if [ -n "${rvv_exe}" ]; then
            for core in ${RVV_CORES}; do
                diff_csv="${HIST_DIR}/${safe_kernel}_rvv_fallback_core${core}_run${run}_diff.csv"
                input_csv="${HIST_DIR}/${safe_kernel}_rvv_fallback_core${core}_run${run}_input.csv"
                run_log="${LOG_DIR}/${safe_kernel}_rvv_fallback_core${core}_run${run}.log"

                if run_one "${rvv_exe}" "${core}" "${seed}" "${diff_csv}" "${input_csv}" "${run_log}"; then
                    status="$(awk -F, 'NR==2 {print $1}' "${run_log}")"
                    return_code="$(awk -F, 'NR==2 {print $2}' "${run_log}")"
                    total_elements="$(awk -F, 'NR==2 {print $3}' "${run_log}")"
                    mismatch_count="$(awk -F, 'NR==2 {print $4}' "${run_log}")"
                    max_integer_difference="$(awk -F, 'NR==2 {print $5}' "${run_log}")"
                    overflow_count="$(awk -F, 'NR==2 {print $6}' "${run_log}")"
                    exact_match_rate="$(awk -F, 'NR==2 {print $7}' "${run_log}")"
                    [ "${status}" = "OK" ] && ok_count=$((ok_count + 1)) || run_failed_count=$((run_failed_count + 1))
                else
                    status="RUN_FAILED"
                    return_code="NA"
                    total_elements="NA"
                    mismatch_count="NA"
                    max_integer_difference="NA"
                    overflow_count="NA"
                    exact_match_rate="NA"
                    run_failed_count=$((run_failed_count + 1))
                fi

                log "  RVV_FALLBACK core ${core} run ${run} seed ${seed}: ${status}"
                {
                    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                        "$(date -Iseconds)" "${MODE}" "RVV_FALLBACK" "${core}" "${seed}" \
                        "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" \
                        "${INPUT_CLASS}" "${M}" "${N}" "${K}" "${status}" "${return_code}" "${total_elements}" \
                        "${mismatch_count}" "${max_integer_difference}" "${overflow_count}" "${exact_match_rate}"
                    printf '%s,%s,%s\n' "$(csv_quote "${diff_csv}")" "$(csv_quote "${input_csv}")" "$(csv_quote "${run_log}")"
                } >> "${CSV}"
            done
        fi
    done
done < <(find "${PROJECT_ROOT}/IME_NATIVE_KERNELS" -type d -name 'ime_kernel_*' | sort)

cp "${CSV}" "${LATEST_CSV}"

log "============================================================"
log "DONE"
log "kernel_folders=${kernel_count}"
log "built_executables=${built_count}"
log "ok_runs=${ok_count}"
log "build_failed=${build_failed_count}"
log "run_failed=${run_failed_count}"
log "CSV=${CSV}"
log "Latest CSV=${LATEST_CSV}"
log "Histograms=${HIST_DIR}"
log "Logs=${LOG_DIR}"
