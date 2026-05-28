#!/usr/bin/env bash
set -u

MODE="${1:-all}"
M=1024
N=1024
K=1024

if [ "$#" -ge 5 ]; then
    RUNS="${5:-1}"
else
    RUNS="${2:-1}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS="${SCRIPT_DIR}/accuracy_kernel_check.c"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${SCRIPT_DIR}/accuracy_results_${MODE}_${M}_${STAMP}"
BUILD_DIR="${RESULT_DIR}/build"
RAW_LOG_DIR="${RESULT_DIR}/raw_logs"
CSV="${RESULT_DIR}/accuracy_summary_${MODE}_${M}_${STAMP}.csv"
LATEST_CSV="${SCRIPT_DIR}/accuracy_summary_latest.csv"
LATEST_LOG="${SCRIPT_DIR}/accuracy_live_latest.log"

CC="${CC:-gcc}"
ABI="${ABI:-lp64d}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O3 -std=c11 -Wall -Wextra -Wno-unknown-pragmas}"
RVV128_MARCH="${RVV128_MARCH:-rv64gcv_zvl128b}"
RVV256_MARCH="${RVV256_MARCH:-rv64gcv_zvl256b}"
INPUT_CLASSES="${INPUT_CLASSES:-uniform wide_range cancellation}"

mkdir -p "${BUILD_DIR}" "${RAW_LOG_DIR}"
: > "${LATEST_LOG}"

log()
{
    printf '%s\n' "$*" | tee -a "${LATEST_LOG}"
}

csv_quote()
{
    printf '%s' "$1" | sed 's/"/""/g; s/^/"/; s/$/"/'
}

first_match()
{
    local dir="$1"
    local pattern="$2"
    local f

    for f in "${dir}"/${pattern}; do
        if [ -f "${f}" ]; then
            printf '%s\n' "${f}"
            return 0
        fi
    done

    return 1
}

field_from_kernel()
{
    local kernel="$1"
    local regex="$2"
    printf '%s\n' "${kernel}" | sed -n "${regex}"
}

include_kind_in_mode()
{
    local kind="$1"

    case "${MODE}:${kind}" in
        all:*) return 0 ;;
        k1-rvv:FP32_RVV|k1-rvv:FP64_RVV|k1-rvv:INT8_RVV) return 0 ;;
        k3-rvv:FP32_RVV|k3-rvv:FP64_RVV|k3-rvv:INT8_RVV) return 0 ;;
        k3-ime:INT8_IME) return 0 ;;
        local:*) return 0 ;;
        *) return 1 ;;
    esac
}

core_list_for_kind()
{
    local kind="$1"

    case "${MODE}:${kind}" in
        k1-rvv:*|k3-rvv:*) printf '0 1 2 3 4 5 6 7' ;;
        k3-ime:*) printf '8 9 10 11 12 13 14 15' ;;
        *) printf 'NA' ;;
    esac
}

prepare_core_access()
{
    local kind="$1"

    case "${MODE}:${kind}" in
        k3-ime:*)
            if [ -w /proc/set_ai_thread ]; then
                echo $$ > /proc/set_ai_thread
            fi
            ;;
    esac
}

compile_kernel()
{
    local kind="$1"
    local march="$2"
    local symbol="$3"
    local mr="$4"
    local nr="$5"
    local exe="$6"
    local build_log="$7"
    shift 7
    local sources=("$@")
    local define_kind=""

    case "${kind}" in
        FP32_RVV) define_kind="-DACC_KIND_FP32" ;;
        FP64_RVV) define_kind="-DACC_KIND_FP64" ;;
        INT8_RVV) define_kind="-DACC_KIND_INT8_RVV" ;;
        INT8_IME) define_kind="-DACC_KIND_INT8_IME" ;;
        *) return 1 ;;
    esac

    "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
        "${define_kind}" -DKERNEL_SYMBOL="${symbol}" -DACC_MR="${mr}" -DACC_NR="${nr}" \
        "${HARNESS}" "${sources[@]}" -lm -o "${exe}" > "${build_log}" 2>&1
}

kind_label()
{
    case "$1" in
        FP32_RVV) printf 'FP32' ;;
        FP64_RVV) printf 'FP64' ;;
        INT8_RVV|INT8_IME) printf 'INT8' ;;
        *) printf 'UNKNOWN' ;;
    esac
}

printf 'timestamp,mode,core,baseline,family,kernel,datatype,tile_shape,zvl,lmul,unroll,input_class,M,N,K,run,status,return_code,max_abs_error,mean_abs_error,max_rel_error,mean_rel_error,mismatch_count,max_integer_difference,overflow_count,log_file\n' > "${CSV}"

log "Accuracy checker"
log "MODE=${MODE} M=${M} N=${N} K=${K} RUNS=${RUNS}"
log "Matrix size is fixed to 1024x1024x1024 for this project."
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "RESULT_DIR=${RESULT_DIR}"
log "INPUT_CLASSES=${INPUT_CLASSES}"
grep Cpus_allowed_list /proc/self/status 2>/dev/null | tee -a "${LATEST_LOG}" || true

kernel_count=0
build_failed_count=0
run_failed_count=0
ok_count=0

while IFS= read -r -d '' makefile; do
    kernel_dir="$(dirname "${makefile}")"

    case "${kernel_dir}" in
        *ACCURACY_CHECKER*|*OPENMP_TILED_GEMM_EXTENSION*|*OPENMP_TILED_GEMM_ALL_KERNELS*) continue ;;
    esac

    src=""
    fallback_src=""
    kind=""

    if src="$(first_match "${kernel_dir}" 'ime_kernel_*.c')"; then
        kind="INT8_IME"
        fallback_src="${kernel_dir}/rvv_fallback.c"
        if [ ! -f "${fallback_src}" ]; then
            continue
        fi
    elif src="$(first_match "${kernel_dir}" 'sgemm_kernel_*.c')"; then
        kind="FP32_RVV"
    elif src="$(first_match "${kernel_dir}" 'dgemm_kernel_*.c')"; then
        kind="FP64_RVV"
    elif src="$(first_match "${kernel_dir}" 'igemm_kernel_*.c')"; then
        kind="INT8_RVV"
    else
        continue
    fi

    if ! include_kind_in_mode "${kind}"; then
        continue
    fi

    family="$(basename "$(dirname "${kernel_dir}")")"
    baseline="$(basename "$(dirname "$(dirname "${kernel_dir}")")")"
    kernel="$(basename "${src}" .c)"
    symbol="${kernel}"
    tile_shape="$(field_from_kernel "${kernel}" 's/.*kernel_\([0-9]x[0-9]\)_zvl.*/\1/p')"
    zvl="$(field_from_kernel "${kernel}" 's/.*_zvl\([0-9]*b\).*/\1/p')"
    lmul="$(field_from_kernel "${kernel}" 's/.*_lmul\([^_]*\)_unroll.*/\1/p')"
    unroll="$(field_from_kernel "${kernel}" 's/.*_unroll\([0-9]*\).*/\1/p')"
    mr="$(printf '%s\n' "${tile_shape}" | cut -dx -f1)"
    nr="$(printf '%s\n' "${tile_shape}" | cut -dx -f2)"
    datatype="$(kind_label "${kind}")"
    march="${RVV128_MARCH}"

    if [ -z "${mr}" ] || [ -z "${nr}" ]; then
        continue
    fi

    if printf '%s\n' "${kernel}" | grep -q 'zvl256b'; then
        march="${RVV256_MARCH}"
    fi

    safe_name="${baseline}_${family}_${kernel}"
    exe="${BUILD_DIR}/${safe_name}"
    build_log="${RAW_LOG_DIR}/${safe_name}_build.log"
    sources=("${src}")

    if [ "${kind}" = "INT8_IME" ]; then
        sources+=("${fallback_src}")
    fi

    log "============================================================"
    log "KERNEL=${kernel}"
    log "FAMILY=${family}"
    log "KIND=${kind}"

    if ! compile_kernel "${kind}" "${march}" "${symbol}" "${mr}" "${nr}" "${exe}" "${build_log}" "${sources[@]}"; then
        build_failed_count=$((build_failed_count + 1))
        log "BUILD_FAILED: ${build_log}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "NA" "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
            "${datatype}" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "NA" "${M}" "${N}" "${K}" "0" \
            "BUILD_FAILED" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "$(csv_quote "${build_log}")" >> "${CSV}"
        continue
    fi

    kernel_count=$((kernel_count + 1))
    prepare_core_access "${kind}"
    cores="$(core_list_for_kind "${kind}")"

    for core in ${cores}; do
        for input_class in ${INPUT_CLASSES}; do
            run=1
            while [ "${run}" -le "${RUNS}" ]; do
                run_log="${RAW_LOG_DIR}/${safe_name}_core${core}_${input_class}_run${run}.log"

                if [ "${core}" != "NA" ] && command -v taskset >/dev/null 2>&1; then
                    taskset -c "${core}" "${exe}" "${input_class}" "${M}" "${N}" "${K}" > "${run_log}" 2>&1
                else
                    "${exe}" "${input_class}" "${M}" "${N}" "${K}" > "${run_log}" 2>&1
                fi

                rc=$?
                if [ "${rc}" -ne 0 ]; then
                    run_failed_count=$((run_failed_count + 1))
                    result="RUN_FAILED,${rc},NA,NA,NA,NA,NA,NA,NA"
                    log "  core ${core} ${input_class} run ${run}: RUN_FAILED"
                else
                    result="$(tail -n 1 "${run_log}")"
                    status="$(printf '%s\n' "${result}" | cut -d, -f1)"
                    if [ "${status}" = "OK" ]; then
                        ok_count=$((ok_count + 1))
                        log "  core ${core} ${input_class} run ${run}: OK"
                    else
                        run_failed_count=$((run_failed_count + 1))
                        log "  core ${core} ${input_class} run ${run}: ${status}"
                    fi
                fi

                printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                    "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "${core}" "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
                    "${datatype}" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${input_class}" "${M}" "${N}" "${K}" "${run}" \
                    "${result}" "$(csv_quote "${run_log}")" >> "${CSV}"

                run=$((run + 1))
            done
        done
    done
done < <(find "${PROJECT_ROOT}" -name Makefile -print0)

cp "${CSV}" "${LATEST_CSV}"

log "============================================================"
log "DONE"
log "kernels_built=${kernel_count}"
log "ok_runs=${ok_count}"
log "build_failed=${build_failed_count}"
log "run_failed=${run_failed_count}"
log "CSV=${CSV}"
log "Latest CSV=${LATEST_CSV}"
log "Raw logs=${RAW_LOG_DIR}"
