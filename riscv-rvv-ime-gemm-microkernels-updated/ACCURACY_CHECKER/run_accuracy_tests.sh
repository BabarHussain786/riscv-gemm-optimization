#!/usr/bin/env bash
set -u

MODE="${1:-}"
M=1024
N=1024
K=1024
RUNS="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS="${SCRIPT_DIR}/accuracy_kernel_check.c"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${SCRIPT_DIR}/accuracy_results_${MODE}_${M}_${STAMP}"
BUILD_DIR="${RESULT_DIR}/build"
RAW_LOG_DIR="${RESULT_DIR}/raw_logs"
CSV="${RESULT_DIR}/accuracy_summary_${MODE}_${M}_${STAMP}.csv"
LATEST_CSV="${SCRIPT_DIR}/accuracy_summary_${MODE}_latest.csv"
LATEST_LOG="${SCRIPT_DIR}/accuracy_live_${MODE}_latest.log"

CC="${CC:-gcc}"
ABI="${ABI:-lp64d}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O3 -std=c11 -Wall -Wextra -Wno-unknown-pragmas}"
RVV128_MARCH="${RVV128_MARCH:-rv64gcv_zvl128b}"
RVV256_MARCH="${RVV256_MARCH:-rv64gcv_zvl256b}"
BASE_SEED="${BASE_SEED:-0}"
# The default campaign is intentionally minimal: one reproducible uniform
# experiment per datatype. Set these variables explicitly to add diagnostics.
FP_INPUT_CLASSES="${FP_INPUT_CLASSES:-bounded_uniform}"
INT8_INPUT_CLASSES="${INT8_INPUT_CLASSES:-full_range_uniform}"
ENABLE_EXPERIMENTAL_MF2="${ENABLE_EXPERIMENTAL_MF2:-0}"

usage()
{
    cat <<'EOF'
Usage: bash ACCURACY_CHECKER/run_accuracy_tests.sh MODE [RUNS]

MODE must be one of:
  k1-rvv  K1 RVV kernels on cores 0-7
  k1-ime  K1 IME VMADOT kernels on cores 0-3
  k3-rvv  K3 RVV kernels on cores 0-7
  k3-ime  K3 IME VMADOT kernels on cores 8-15
EOF
}

case "${MODE}" in
    k1-rvv|k1-ime|k3-rvv|k3-ime) ;;
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
        k1-rvv:FP32_RVV|k1-rvv:FP64_RVV|k1-rvv:INT8_RVV) return 0 ;;
        k1-ime:INT8_IME) return 0 ;;
        k3-rvv:FP32_RVV|k3-rvv:FP64_RVV|k3-rvv:INT8_RVV) return 0 ;;
        k3-ime:INT8_IME) return 0 ;;
        *) return 1 ;;
    esac
}

include_kernel_dir_in_mode()
{
    local kind="$1"
    local kernel_dir="$2"

    if [ "${kind}" != "INT8_IME" ]; then
        return 0
    fi

    case "${MODE}:${kernel_dir}" in
        k1-ime:*IME_NATIVE_KERNELS/*|k3-ime:*IME_NATIVE_KERNELS/*) return 0 ;;
        *) return 1 ;;
    esac
}

core_list_for_kind()
{
    local kind="$1"

    case "${MODE}:${kind}" in
        k1-rvv:*|k3-rvv:*) printf '0 1 2 3 4 5 6 7' ;;
        k1-ime:*) printf '0 1 2 3' ;;
        k3-ime:*) printf '8 9 10 11 12 13 14 15' ;;
        *) return 1 ;;
    esac
}

input_classes_for_kind()
{
    case "$1" in
        FP32_RVV|FP64_RVV) printf '%s' "${FP_INPUT_CLASSES}" ;;
        INT8_RVV|INT8_IME) printf '%s' "${INT8_INPUT_CLASSES}" ;;
        *) return 1 ;;
    esac
}

execution_path_for_kind()
{
    case "$1" in
        INT8_IME) printf 'IME_VMADOT_REQUIRED' ;;
        FP32_RVV|FP64_RVV|INT8_RVV) printf 'RVV_NATIVE' ;;
        *) printf 'UNKNOWN' ;;
    esac
}

reference_method_for_kind()
{
    case "$1" in
        FP32_RVV) printf 'FP64_ACCUMULATION_REFERENCE' ;;
        FP64_RVV) printf 'LONG_DOUBLE_ACCUMULATION_REFERENCE' ;;
        INT8_RVV|INT8_IME) printf 'INT64_REFERENCE_EXACT_INT32_OUTPUT' ;;
        *) printf 'UNKNOWN' ;;
    esac
}

core_supports_required_path()
{
    local kind="$1"
    local core="$2"

    if [ "${kind}" != "INT8_IME" ]; then
        return 0
    fi

    [ -f "/sys/firmware/devicetree/base/cpus/cpu@${core}/cpu-ai" ]
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
    local input_contract="$6"
    local exe="$7"
    local build_log="$8"
    shift 8
    local sources=("$@")
    local define_kind=""
    local extra_defines=()

    case "${kind}" in
        FP32_RVV) define_kind="-DACC_KIND_FP32" ;;
        FP64_RVV) define_kind="-DACC_KIND_FP64" ;;
        INT8_RVV) define_kind="-DACC_KIND_INT8_RVV" ;;
        INT8_IME)
            define_kind="-DACC_KIND_INT8_IME"
            extra_defines+=("-DSPACEMIT_IME_REQUIRE_HARDWARE=1")
            if [ "${ENABLE_EXPERIMENTAL_MF2}" = "1" ] &&
               printf '%s\n' "${symbol}" | grep -q '_lmulmf2_'; then
                extra_defines+=("-DSPACEMIT_IME_ENABLE_EXPERIMENTAL_MF2_NATIVE=1")
            fi
            case "${input_contract}" in
                IME_FULL_K_MAJOR)
                    extra_defines+=("-DACC_IME_INPUT_FULL_MATRIX=1")
                    ;;
                IME_PREPACKED_ROWS)
                    extra_defines+=("-DACC_IME_INPUT_PREPACKED_ROWS=1")
                    ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac

    "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
        "${define_kind}" "${extra_defines[@]}" \
        -DKERNEL_SYMBOL="${symbol}" -DACC_MR="${mr}" -DACC_NR="${nr}" \
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

printf 'timestamp,mode,core,execution_path,input_contract,reference_method,generator_seed,baseline,family,kernel,datatype,tile_shape,zvl,lmul,unroll,input_class,matrix_m,matrix_n,matrix_k,run_index,status,return_code,max_abs_error,mean_abs_error,max_rel_error,mean_rel_error,frobenius_relative_error,max_bound_ratio,bound_failure_count,nonfinite_count,mismatch_count,max_integer_difference,overflow_count,log_file\n' > "${CSV}"

log "Accuracy checker"
log "MODE=${MODE} M=${M} N=${N} K=${K} RUNS=${RUNS}"
log "Matrix size is fixed to 1024x1024x1024 for this project."
log "Input contracts: packed tile panels for RVV; full K-major matrices for all IME wrappers."
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "RESULT_DIR=${RESULT_DIR}"
log "BASE_SEED=${BASE_SEED}"
log "Seed rule: generator_seed = BASE_SEED + input_class_index * RUNS + run_index."
log "Default single-class run uses seed 1. The seed is recorded for reproducibility; it is not a hardware parameter."
log "FP_INPUT_CLASSES=${FP_INPUT_CLASSES}"
log "INT8_INPUT_CLASSES=${INT8_INPUT_CLASSES}"
log "ENABLE_EXPERIMENTAL_MF2=${ENABLE_EXPERIMENTAL_MF2}"
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

    if ! include_kind_in_mode "${kind}" || ! include_kernel_dir_in_mode "${kind}" "${kernel_dir}"; then
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
    execution_path="$(execution_path_for_kind "${kind}")"
    reference_method="$(reference_method_for_kind "${kind}")"
    input_contract="RVV_PACKED_TILE_PANELS"
    march="${RVV128_MARCH}"

    if [ -z "${mr}" ] || [ -z "${nr}" ]; then
        continue
    fi

    if printf '%s\n' "${kernel}" | grep -q 'zvl256b'; then
        march="${RVV256_MARCH}"
    fi

    if [ "${kind}" = "INT8_IME" ]; then
        input_contract="IME_FULL_K_MAJOR"
        if [ "${lmul}" = "mf2" ] && [ "${ENABLE_EXPERIMENTAL_MF2}" != "1" ]; then
            log "SKIP_EXPERIMENTAL_MF2=${kernel}"
            continue
        fi
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
    log "INPUT_CONTRACT=${input_contract}"
    log "REFERENCE_METHOD=${reference_method}"

    if ! compile_kernel "${kind}" "${march}" "${symbol}" "${mr}" "${nr}" "${input_contract}" "${exe}" "${build_log}" "${sources[@]}"; then
        build_failed_count=$((build_failed_count + 1))
        log "BUILD_FAILED: ${build_log}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "NOT_RUN" "${execution_path}" "${input_contract}" "${reference_method}" "NOT_RUN" \
            "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
            "${datatype}" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "NOT_RUN" "${M}" "${N}" "${K}" "0" \
            "BUILD_FAILED" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "$(csv_quote "${build_log}")" >> "${CSV}"
        continue
    fi

    kernel_count=$((kernel_count + 1))
    prepare_core_access "${kind}"
    cores="$(core_list_for_kind "${kind}")"
    input_classes="$(input_classes_for_kind "${kind}")"

    for core in ${cores}; do
        class_index=0
        for input_class in ${input_classes}; do
            run=1
            while [ "${run}" -le "${RUNS}" ]; do
                run_log="${RAW_LOG_DIR}/${safe_name}_core${core}_${input_class}_run${run}.log"
                seed=$((BASE_SEED + class_index * RUNS + run))

                if ! core_supports_required_path "${kind}" "${core}"; then
                    printf 'IME hardware marker is unavailable on core %s\n' "${core}" > "${run_log}"
                    result="PATH_UNAVAILABLE,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA"
                    rc=0
                elif command -v taskset >/dev/null 2>&1; then
                    taskset -c "${core}" "${exe}" "${input_class}" "${M}" "${N}" "${K}" "${seed}" > "${run_log}" 2>&1
                else
                    "${exe}" "${input_class}" "${M}" "${N}" "${K}" "${seed}" > "${run_log}" 2>&1
                fi

                rc=$?
                if [ "${rc}" -ne 0 ]; then
                    run_failed_count=$((run_failed_count + 1))
                    result="RUN_FAILED,${rc},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA"
                    log "  core ${core} ${input_class} run ${run} seed ${seed}: RUN_FAILED"
                else
                    if [ -z "${result:-}" ]; then
                        result="$(tail -n 1 "${run_log}")"
                    fi
                    status="$(printf '%s\n' "${result}" | cut -d, -f1)"
                    if [ "${status}" = "OK" ]; then
                        ok_count=$((ok_count + 1))
                        log "  core ${core} ${input_class} run ${run} seed ${seed}: OK"
                    else
                        run_failed_count=$((run_failed_count + 1))
                        log "  core ${core} ${input_class} run ${run} seed ${seed}: ${status}"
                    fi
                fi

                printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                    "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "${core}" "${execution_path}" "${input_contract}" "${reference_method}" "${seed}" \
                    "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
                    "${datatype}" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${input_class}" "${M}" "${N}" "${K}" "${run}" \
                    "${result}" "$(csv_quote "${run_log}")" >> "${CSV}"

                unset result
                run=$((run + 1))
            done
            class_index=$((class_index + 1))
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
