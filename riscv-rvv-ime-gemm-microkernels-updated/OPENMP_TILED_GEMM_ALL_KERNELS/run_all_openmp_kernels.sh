#!/usr/bin/env bash
set -u

MODE="${1:-k3-rvv}"
M="${2:-1024}"
N="${3:-1024}"
K="${4:-1024}"
TILE_N="${5:-64}"
RUNS="${6:-6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/omp_bench_template.c"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${SCRIPT_DIR}/openmp_results_${MODE}_${M}_${STAMP}"
BUILD_DIR="${RESULT_DIR}/build"
RAW_LOG_DIR="${RESULT_DIR}/raw_logs"
RAW_CSV="${RESULT_DIR}/openmp_raw_${MODE}_${M}_runs${RUNS}_${STAMP}.csv"
LIVE_LOG="${RESULT_DIR}/openmp_live_${MODE}_${M}_runs${RUNS}_${STAMP}.log"

CC="${CC:-gcc}"
ABI="${ABI:-lp64d}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O3 -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fopenmp}"
RVV128_MARCH="${RVV128_MARCH:-rv64gcv_zvl128b}"
RVV256_MARCH="${RVV256_MARCH:-rv64gcv_zvl256b}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES

TASKSET_CORES=""
CORE_GROUP="local"

case "${MODE}" in
    k1-rvv|k3-rvv)
        TASKSET_CORES="0-7"
        CORE_GROUP="0-7"
        ;;
    k3-ime)
        TASKSET_CORES="8-15"
        CORE_GROUP="8-15"
        if [ -w /proc/set_ai_thread ]; then
            echo $$ > /proc/set_ai_thread
        fi
        ;;
    local)
        TASKSET_CORES=""
        CORE_GROUP="local"
        ;;
    *)
        echo "Usage: $0 {k1-rvv|k3-rvv|k3-ime|local} [M] [N] [K] [tile_N] [runs]"
        exit 1
        ;;
esac

mkdir -p "${BUILD_DIR}" "${RAW_LOG_DIR}"

log()
{
    printf '%s\n' "$*" | tee -a "${LIVE_LOG}"
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
        k3-rvv:FP32_RVV|k3-rvv:FP64_RVV|k3-rvv:INT8_RVV) return 0 ;;
        k3-ime:INT8_IME) return 0 ;;
        local:*) return 0 ;;
        *) return 1 ;;
    esac
}

compile_kernel()
{
    local kind="$1"
    local march="$2"
    local symbol="$3"
    local exe="$4"
    local build_log="$5"
    shift 5
    local sources=("$@")
    local define_kind=""

    case "${kind}" in
        FP32_RVV) define_kind="-DOMP_KIND_FP32" ;;
        FP64_RVV) define_kind="-DOMP_KIND_FP64" ;;
        INT8_RVV) define_kind="-DOMP_KIND_INT8_RVV" ;;
        INT8_IME) define_kind="-DOMP_KIND_INT8_IME" ;;
        *) return 1 ;;
    esac

    "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
        "${define_kind}" -DKERNEL_SYMBOL="${symbol}" \
        "${TEMPLATE}" "${sources[@]}" -o "${exe}" > "${build_log}" 2>&1
}

run_binary_once()
{
    local exe="$1"
    local run_log="$2"

    if [ -n "${TASKSET_CORES}" ] && command -v taskset >/dev/null 2>&1; then
        taskset -c "${TASKSET_CORES}" "${exe}" "${M}" "${N}" "${K}" "${TILE_N}" > "${run_log}" 2>&1
    else
        "${exe}" "${M}" "${N}" "${K}" "${TILE_N}" > "${run_log}" 2>&1
    fi
}

printf 'timestamp,mode,baseline,family,kernel,tile_shape,zvl,lmul,unroll,kind,core_group,threads,M,N,K,tile_N,run,status,return_code,time_sec,metric_name,metric_value,log_file\n' > "${RAW_CSV}"

log "OpenMP all-kernel tiled GEMM campaign"
log "MODE=${MODE} M=${M} N=${N} K=${K} tile_N=${TILE_N} RUNS=${RUNS}"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "RESULT_DIR=${RESULT_DIR}"
log "OMP_NUM_THREADS=${OMP_NUM_THREADS} OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES}"
grep Cpus_allowed_list /proc/self/status 2>/dev/null | tee -a "${LIVE_LOG}" || true

kernel_count=0
build_failed_count=0
run_failed_count=0
ok_count=0

while IFS= read -r -d '' makefile; do
    kernel_dir="$(dirname "${makefile}")"

    case "${kernel_dir}" in
        *OPENMP_TILED_GEMM_EXTENSION*|*OPENMP_TILED_GEMM_ALL_KERNELS*) continue ;;
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
    march="${RVV128_MARCH}"

    if printf '%s\n' "${kernel}" | grep -q 'zvl256b'; then
        march="${RVV256_MARCH}"
    fi

    safe_name="${baseline}_${family}_${kernel}"
    exe="${BUILD_DIR}/${safe_name}"
    build_log="${RAW_LOG_DIR}/${safe_name}_build.log"

    log "============================================================"
    log "KERNEL=${kernel}"
    log "BASELINE=${baseline}"
    log "FAMILY=${family}"
    log "KIND=${kind}"

    sources=("${src}")
    if [ "${kind}" = "INT8_IME" ]; then
        sources+=("${fallback_src}")
    fi

    if ! compile_kernel "${kind}" "${march}" "${symbol}" "${exe}" "${build_log}" "${sources[@]}"; then
        build_failed_count=$((build_failed_count + 1))
        log "BUILD_FAILED: ${build_log}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
            "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" \
            "${M}" "${N}" "${K}" "${TILE_N}" "0" "BUILD_FAILED" "NA" "NA" "NA" "NA" "$(csv_quote "${build_log}")" >> "${RAW_CSV}"
        continue
    fi

    kernel_count=$((kernel_count + 1))

    for run in $(seq 1 "${RUNS}"); do
        run_log="${RAW_LOG_DIR}/${safe_name}_run${run}.log"
        log "run ${run}/${RUNS}"

        run_binary_once "${exe}" "${run_log}"
        rc=$?
        csv_line="$(grep '^CSV_RUN,' "${run_log}" | tail -1 || true)"

        if [ -n "${csv_line}" ]; then
            IFS=',' read -r _ time_sec metric_value kernel_return metric_name <<EOF_CSV
${csv_line}
EOF_CSV
            if [ "${kernel_return}" = "0" ] && [ "${rc}" = "0" ]; then
                status="OK"
                ok_count=$((ok_count + 1))
            else
                status="KERNEL_RETURN"
                run_failed_count=$((run_failed_count + 1))
            fi
        else
            status="RUN_FAILED"
            kernel_return="${rc}"
            time_sec="NA"
            metric_value="NA"
            metric_name="NA"
            run_failed_count=$((run_failed_count + 1))
        fi

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
            "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" \
            "${M}" "${N}" "${K}" "${TILE_N}" "${run}" "${status}" "${kernel_return}" "${time_sec}" "${metric_name}" "${metric_value}" \
            "$(csv_quote "${run_log}")" >> "${RAW_CSV}"
    done
done < <(find "${PROJECT_ROOT}" -type f -name Makefile -print0 | sort -z)

cp "${RAW_CSV}" "${SCRIPT_DIR}/openmp_raw_latest_${MODE}.csv"
cp "${LIVE_LOG}" "${SCRIPT_DIR}/openmp_live_latest_${MODE}.log"

log "============================================================"
log "DONE"
log "kernels_built=${kernel_count}"
log "ok_runs=${ok_count}"
log "failed_runs=${run_failed_count}"
log "build_failed=${build_failed_count}"
log "Raw CSV: ${RAW_CSV}"
log "Latest CSV: ${SCRIPT_DIR}/openmp_raw_latest_${MODE}.csv"
log "Raw logs: ${RAW_LOG_DIR}"
log "============================================================"

if [ "${build_failed_count}" -ne 0 ] || [ "${run_failed_count}" -ne 0 ]; then
    exit 1
fi

exit 0
