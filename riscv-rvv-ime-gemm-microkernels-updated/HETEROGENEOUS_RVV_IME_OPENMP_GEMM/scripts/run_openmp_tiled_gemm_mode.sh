#!/usr/bin/env bash
set -u

MODE="${1:-k3-rvv}"
M="${2:-1024}"
N="${3:-1024}"
K="${4:-1024}"
TILE_N="${5:-64}"
RUNS="${6:-6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_DIR}/.." && pwd)"
TEMPLATE="${MODULE_DIR}/src/openmp_heterogeneous_gemm.c"
RESULT_ROOT="${MODULE_DIR}/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_ROOT}/openmp_results_${MODE}_${M}_${STAMP}"
BUILD_DIR="${RESULT_DIR}/build"
RAW_LOG_DIR="${RESULT_DIR}/raw_logs"
RAW_CSV="${RESULT_DIR}/openmp_raw_${MODE}_${M}_runs${RUNS}_${STAMP}.csv"
SUMMARY_CSV="${RESULT_DIR}/openmp_summary_${MODE}_${M}_runs${RUNS}_${STAMP}.csv"
LIVE_LOG="${RESULT_DIR}/openmp_live_${MODE}_${M}_runs${RUNS}_${STAMP}.log"

CC="${CC:-gcc}"
ABI="${ABI:-lp64d}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O3 -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fopenmp}"
RVV128_MARCH="${RVV128_MARCH:-rv64gcv_zvl128b}"
OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
OMP_PLACES="${OMP_PLACES:-cores}"
OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
OMP_VALIDATE="${OMP_VALIDATE:-1}"
OMP_WARMUP="${OMP_WARMUP:-1}"
VALIDATE_EACH_RUN="${VALIDATE_EACH_RUN:-0}"
ENABLE_MF2="${ENABLE_MF2:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
ZVL_FILTER="${ZVL_FILTER:-128b}"
# BUILD_ONLY=1 compiles selected kernels and records BUILD_OK rows without timed execution.
# ZVL_FILTER=128b keeps the final K1 campaign focused on the 128b kernel set; use ZVL_FILTER=all for exploratory runs.

TASKSET_CORES=""
CORE_GROUP="local"
DEFAULT_THREADS="1"

case "${MODE}" in
    k1-rvv|k1-rvv-all|k3-rvv)
        TASKSET_CORES="0-7"
        CORE_GROUP="0-7"
        DEFAULT_THREADS="8"
        ;;
    k1-rvv-only)
        TASKSET_CORES="4-7"
        CORE_GROUP="4-7"
        DEFAULT_THREADS="4"
        ;;
    k1-ime)
        TASKSET_CORES="0-3"
        CORE_GROUP="0-3"
        DEFAULT_THREADS="4"
        ;;
    k1-mixed-rvv-ime)
        TASKSET_CORES="0-7"
        CORE_GROUP="0-7-mixed-rvv-ime"
        DEFAULT_THREADS="8"
        ;;
    k3-ime)
        TASKSET_CORES="8-15"
        CORE_GROUP="8-15"
        DEFAULT_THREADS="8"
        ;;
    k3-ime-cluster0)
        TASKSET_CORES="8-11"
        CORE_GROUP="8-11"
        DEFAULT_THREADS="4"
        ;;
    k3-ime-cluster1)
        TASKSET_CORES="12-15"
        CORE_GROUP="12-15"
        DEFAULT_THREADS="4"
        ;;
    local)
        TASKSET_CORES=""
        CORE_GROUP="local"
        DEFAULT_THREADS="1"
        ;;
    *)
        echo "Usage: $0 {k1-rvv|k1-rvv-all|k1-rvv-only|k1-ime|k1-mixed-rvv-ime|k3-rvv|k3-ime|k3-ime-cluster0|k3-ime-cluster1|local} [M] [N] [K] [tile_N] [runs]"
        exit 1
        ;;
esac

OMP_NUM_THREADS="${OMP_NUM_THREADS:-${DEFAULT_THREADS}}"
export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_DYNAMIC OMP_WARMUP

mode_label()
{
    case "${MODE}" in
        k1-rvv|k1-rvv-all) printf '%s\n' "K1 all-core RVV OpenMP baseline" ;;
        k1-rvv-only) printf '%s\n' "K1 RVV-cluster OpenMP baseline" ;;
        k1-ime) printf '%s\n' "K1 IME-cluster OpenMP baseline" ;;
        k1-mixed-rvv-ime) printf '%s\n' "K1 heterogeneous OpenMP RVV-IME execution" ;;
        k3-rvv) printf '%s\n' "K3 RVV OpenMP baseline" ;;
        k3-ime) printf '%s\n' "K3 IME OpenMP baseline" ;;
        k3-ime-cluster0) printf '%s\n' "K3 IME cluster-0 OpenMP baseline" ;;
        k3-ime-cluster1) printf '%s\n' "K3 IME cluster-1 OpenMP baseline" ;;
        local) printf '%s\n' "Local OpenMP inspection run" ;;
        *) printf '%s\n' "OpenMP GEMM run" ;;
    esac
}

core_description()
{
    case "${MODE}" in
        k1-rvv|k1-rvv-all) printf '%s\n' "cores 0-7 execute RVV kernels" ;;
        k1-rvv-only) printf '%s\n' "cores 4-7 execute RVV kernels" ;;
        k1-ime) printf '%s\n' "cores 0-3 execute native IME kernels" ;;
        k1-mixed-rvv-ime) printf '%s\n' "cores 0-3 execute IME path; cores 4-7 execute RVV fallback path" ;;
        k3-rvv) printf '%s\n' "cores 0-7 execute RVV kernels" ;;
        k3-ime) printf '%s\n' "cores 8-15 execute IME kernels" ;;
        k3-ime-cluster0) printf '%s\n' "cores 8-11 execute IME kernels" ;;
        k3-ime-cluster1) printf '%s\n' "cores 12-15 execute IME kernels" ;;
        local) printf '%s\n' "unrestricted local OpenMP placement" ;;
        *) printf '%s\n' "core group ${CORE_GROUP}" ;;
    esac
}

kind_label()
{
    case "$1" in
        FP32_RVV) printf '%s\n' "FP32 RVV vector GEMM" ;;
        FP64_RVV) printf '%s\n' "FP64 RVV vector GEMM" ;;
        INT8_RVV) printf '%s\n' "INT8 RVV widening GEMM (INT8 x INT8 -> INT32)" ;;
        INT8_IME) printf '%s\n' "INT8 native IME GEMM (INT8 x INT8 -> INT32)" ;;
        INT8_MIXED) printf '%s\n' "INT8 heterogeneous IME/RVV-fallback GEMM (INT8 x INT8 -> INT32)" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

log_kernel_header()
{
    local baseline="$1"
    local family="$2"
    local kernel="$3"
    local kind="$4"
    local tile_shape="$5"
    local zvl="$6"
    local lmul="$7"
    local unroll="$8"
    local input_contract="$9"

    log "============================================================"
    log "KERNEL: ${kernel}"
    log "  MODE: $(mode_label)"
    log "  BASELINE: ${baseline}"
    log "  FAMILY: ${family}"
    log "  PATH: $(kind_label "${kind}")"
    log "  TILE: ${tile_shape} ZVL=${zvl} LMUL=${lmul} UNROLL=${unroll}"
    log "  WORK: M=${M} N=${N} K=${K} tile_N=${TILE_N} runs=${RUNS}"
    log "  CORES: $(core_description)"
    log "  INPUT_CONTRACT: ${input_contract}"
}

log_run_result()
{
    local run="$1"
    local status="$2"
    local metric_name="$3"
    local metric_value="$4"
    local time_sec="$5"
    local actual_threads="$6"
    local validation="$7"
    local return_code="$8"
    local failure_stage="$9"
    local worker_placement="${10}"

    if [ "${metric_name}" = "NA" ] || [ "${metric_value}" = "NA" ]; then
        log "    run ${run}: ${status} metric=NA time=${time_sec} threads=${actual_threads} validation=${validation} return=${return_code} stage=${failure_stage}"
    else
        log "    run ${run}: ${status} ${metric_name}=${metric_value} time=${time_sec} threads=${actual_threads} validation=${validation}"
    fi

    if [ -n "${worker_placement}" ] && [ "${worker_placement}" != "NA" ]; then
        log "      workers: ${worker_placement}"
    fi
}

write_summary_csv()
{
    awk -F, '
    BEGIN {
        OFS=",";
        print "mode,baseline,family,kernel,tile_shape,zvl,lmul,unroll,kind,core_group,requested_threads,metric_name,ok_runs,mean_metric,median_metric,min_metric,max_metric,std_metric,mean_time_sec,min_time_sec,max_time_sec,failed_runs,build_failed_runs";
    }
    NR == 1 { next }
    {
        baseline=$3; family=$4; kernel=$5;
        gsub(/^"|"$/, "", baseline);
        gsub(/^"|"$/, "", family);
        gsub(/^"|"$/, "", kernel);
        key=$2 OFS baseline OFS family OFS kernel OFS $6 OFS $7 OFS $8 OFS $9 OFS $10 OFS $11 OFS $12 OFS $23;
        if (!(key in seen)) {
            seen[key]=1;
            keys[++key_count]=key;
            min_value[key]="";
            max_value[key]="";
            min_time[key]="";
            max_time[key]="";
        }
        if ($19 == "OK" && $22 != "NA" && $24 != "NA") {
            n[key]++;
            value=$24 + 0.0;
            t=$22 + 0.0;
            values[key, n[key]]=value;
            sum_value[key]+=value;
            sumsq_value[key]+=value*value;
            sum_time[key]+=t;
            if (min_value[key] == "" || value < min_value[key]) min_value[key]=value;
            if (max_value[key] == "" || value > max_value[key]) max_value[key]=value;
            if (min_time[key] == "" || t < min_time[key]) min_time[key]=t;
            if (max_time[key] == "" || t > max_time[key]) max_time[key]=t;
        } else if ($19 == "BUILD_FAILED") {
            build_failed[key]++;
        } else {
            failed[key]++;
        }
    }
    END {
        for (i=1; i<=key_count; ++i) {
            key=keys[i];
            if (n[key] > 0) {
                for (a=1; a<=n[key]; ++a) sorted[a]=values[key, a];
                for (a=1; a<=n[key]; ++a) {
                    for (b=a+1; b<=n[key]; ++b) {
                        if (sorted[b] < sorted[a]) {
                            tmp=sorted[a]; sorted[a]=sorted[b]; sorted[b]=tmp;
                        }
                    }
                }
                if (n[key] % 2) median=sorted[(n[key]+1)/2];
                else median=(sorted[n[key]/2] + sorted[n[key]/2 + 1]) / 2.0;
                mean=sum_value[key] / n[key];
                variance=(sumsq_value[key] / n[key]) - mean*mean;
                if (variance < 0) variance=0;
                std=sqrt(variance);
                mean_time=sum_time[key] / n[key];
                printf "%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%d,%d\n", key, n[key], mean, median, min_value[key], max_value[key], std, mean_time, min_time[key], max_time[key], failed[key]+0, build_failed[key]+0;
            } else {
                printf "%s,0,NA,NA,NA,NA,NA,NA,NA,NA,%d,%d\n", key, failed[key]+0, build_failed[key]+0;
            }
        }
    }' "${RAW_CSV}" > "${SUMMARY_CSV}"
}
require_positive_integer()
{
    local name="$1"
    local value="$2"
    case "${value}" in
        ''|*[!0-9]*)
            echo "${name} must be a positive integer"
            exit 1
            ;;
    esac
    if [ "${value}" -le 0 ]; then
        echo "${name} must be a positive integer"
        exit 1
    fi
}

require_positive_integer "M" "${M}"
require_positive_integer "N" "${N}"
require_positive_integer "K" "${K}"
require_positive_integer "tile_N" "${TILE_N}"
require_positive_integer "runs" "${RUNS}"
require_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"

if [ $((TILE_N % 8)) -ne 0 ]; then
    echo "tile_N must be a positive integer multiple of 8"
    exit 1
fi

if [ "${MODE}" != "local" ] && ! command -v taskset >/dev/null 2>&1; then
    echo "taskset is required for board modes so worker cores are controlled"
    exit 1
fi

mkdir -p "${RESULT_ROOT}" "${BUILD_DIR}" "${RAW_LOG_DIR}"

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
        k1-rvv-all:FP32_RVV|k1-rvv-all:FP64_RVV|k1-rvv-all:INT8_RVV) return 0 ;;
        k1-rvv-only:FP32_RVV|k1-rvv-only:FP64_RVV|k1-rvv-only:INT8_RVV) return 0 ;;
        k1-ime:INT8_IME) return 0 ;;
        k1-mixed-rvv-ime:INT8_MIXED) return 0 ;;
        k3-rvv:FP32_RVV|k3-rvv:FP64_RVV|k3-rvv:INT8_RVV) return 0 ;;
        k3-ime:INT8_IME) return 0 ;;
        k3-ime-cluster0:INT8_IME|k3-ime-cluster1:INT8_IME) return 0 ;;
        local:*) return 0 ;;
        *) return 1 ;;
    esac
}

include_kernel_dir_in_mode()
{
    local kind="$1"
    local kernel_dir="$2"

    if [ "${kind}" != "INT8_IME" ] && [ "${kind}" != "INT8_MIXED" ]; then
        return 0
    fi

    case "${MODE}:${kernel_dir}" in
        k1-ime:*IME_NATIVE_KERNELS/*|k1-mixed-rvv-ime:*IME_NATIVE_KERNELS/*|k3-ime:*IME_NATIVE_KERNELS/*) return 0 ;;
        k3-ime-cluster0:*IME_NATIVE_KERNELS/*|k3-ime-cluster1:*IME_NATIVE_KERNELS/*) return 0 ;;
        local:*IME_NATIVE_KERNELS/*) return 0 ;;
        *) return 1 ;;
    esac
}

compile_kernel()
{
    local kind="$1"
    local march="$2"
    local symbol="$3"
    local input_contract="$4"
    local exe="$5"
    local build_log="$6"
    shift 6
    local sources=("$@")
    local define_kind=""
    local extra_defines=()

    case "${kind}" in
        FP32_RVV) define_kind="-DOMP_KIND_FP32" ;;
        FP64_RVV) define_kind="-DOMP_KIND_FP64" ;;
        INT8_RVV) define_kind="-DOMP_KIND_INT8_RVV" ;;
        INT8_IME)
            define_kind="-DOMP_KIND_INT8_IME"
            extra_defines+=("-DSPACEMIT_IME_REQUIRE_HARDWARE=1")
            case "${input_contract}" in
                IME_FULL_K_MAJOR) extra_defines+=("-DOMP_IME_INPUT_FULL_MATRIX=1") ;;
                *) return 1 ;;
            esac
            if [ "${ENABLE_MF2}" = "1" ] && printf '%s\n' "${symbol}" | grep -q '_lmulmf2_'; then
                extra_defines+=("-DSPACEMIT_IME_ENABLE_MF2_NATIVE=1")
            fi
            ;;
        INT8_MIXED)
            # INT8_MIXED deliberately omits SPACEMIT_IME_REQUIRE_HARDWARE so RVV-only cores can use fallback.
            define_kind="-DOMP_KIND_INT8_MIXED"
            case "${input_contract}" in
                IME_FULL_K_MAJOR) extra_defines+=("-DOMP_IME_INPUT_FULL_MATRIX=1") ;;
                *) return 1 ;;
            esac
            extra_defines+=("-DMIXED_IME_CORE_FIRST=0" "-DMIXED_IME_CORE_LAST=3")
            extra_defines+=("-DMIXED_IME_TILE_WEIGHT=${MIXED_IME_TILE_WEIGHT:-4}")
            extra_defines+=("-DMIXED_RVV_TILE_WEIGHT=${MIXED_RVV_TILE_WEIGHT:-1}")
            if [ "${ENABLE_MF2}" = "1" ] && printf '%s\n' "${symbol}" | grep -q '_lmulmf2_'; then
                extra_defines+=("-DSPACEMIT_IME_ENABLE_MF2_NATIVE=1")
            fi
            ;;
        *) return 1 ;;
    esac

    "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
        "${define_kind}" "${extra_defines[@]}" -DKERNEL_SYMBOL="${symbol}" \
        "${TEMPLATE}" "${sources[@]}" -lm -o "${exe}" > "${build_log}" 2>&1
}

run_binary_once()
{
    local exe="$1"
    local run_log="$2"
    local run_validation="$3"

    if [[ "${MODE}" == k3-ime* ]] && [ -w /proc/set_ai_thread ]; then
        (
            echo "${BASHPID}" > /proc/set_ai_thread
            if [ -n "${TASKSET_CORES}" ] && command -v taskset >/dev/null 2>&1; then
                exec env OMP_VALIDATE="${run_validation}" taskset -c "${TASKSET_CORES}" \
                    "${exe}" "${M}" "${N}" "${K}" "${TILE_N}"
            fi
            exec env OMP_VALIDATE="${run_validation}" \
                "${exe}" "${M}" "${N}" "${K}" "${TILE_N}"
        ) > "${run_log}" 2>&1
    elif [ -n "${TASKSET_CORES}" ] && command -v taskset >/dev/null 2>&1; then
        env OMP_VALIDATE="${run_validation}" taskset -c "${TASKSET_CORES}" \
            "${exe}" "${M}" "${N}" "${K}" "${TILE_N}" > "${run_log}" 2>&1
    else
        env OMP_VALIDATE="${run_validation}" \
            "${exe}" "${M}" "${N}" "${K}" "${TILE_N}" > "${run_log}" 2>&1
    fi
}

printf 'timestamp,mode,baseline,family,kernel,tile_shape,zvl,lmul,unroll,kind,core_group,requested_threads,actual_threads,M,N,K,tile_N,run,status,return_code,failure_stage,time_sec,metric_name,metric_value,timing_scope,validation_method,mismatch_count,max_error,worker_placement,log_file\n' > "${RAW_CSV}"

log "OpenMP all-kernel tiled GEMM campaign"
log "MODE=${MODE} ($(mode_label)) M=${M} N=${N} K=${K} tile_N=${TILE_N} RUNS=${RUNS}"
log "CORE_PLAN=$(core_description)"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "RESULT_DIR=${RESULT_DIR}"
log "OMP_NUM_THREADS=${OMP_NUM_THREADS} OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES} OMP_DYNAMIC=${OMP_DYNAMIC}"
log "OMP_VALIDATE=${OMP_VALIDATE} VALIDATE_EACH_RUN=${VALIDATE_EACH_RUN} OMP_WARMUP=${OMP_WARMUP} ENABLE_MF2=${ENABLE_MF2} BUILD_ONLY=${BUILD_ONLY} ZVL_FILTER=${ZVL_FILTER} MIXED_IME_TILE_WEIGHT=${MIXED_IME_TILE_WEIGHT:-4} MIXED_RVV_TILE_WEIGHT=${MIXED_RVV_TILE_WEIGHT:-1}"
grep Cpus_allowed_list /proc/self/status 2>/dev/null | tee -a "${LIVE_LOG}" || true

kernel_count=0
build_failed_count=0
run_failed_count=0
ok_count=0

while IFS= read -r -d '' makefile; do
    kernel_dir="$(dirname "${makefile}")"

    case "${kernel_dir}" in
        *OPENMP_TILED_GEMM_EXTENSION*|*OPENMP_TILED_GEMM_ALL_KERNELS*|*HETEROGENEOUS_RVV_IME_OPENMP_GEMM*) continue ;;
    esac

    src=""
    fallback_src=""
    kind=""

    if src="$(first_match "${kernel_dir}" 'ime_kernel_*.c')"; then
        if [ "${MODE}" = "k1-mixed-rvv-ime" ]; then
            kind="INT8_MIXED"
        else
            kind="INT8_IME"
        fi
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
    march="${RVV128_MARCH}"
    if [ "${ZVL_FILTER}" != "all" ] && [ "${zvl}" != "${ZVL_FILTER}" ]; then
        log "SKIP_ZVL_FILTER=${kernel} requested=${ZVL_FILTER}"
        continue
    fi
    input_contract="NOT_APPLICABLE"
    if [ "${kind}" = "INT8_IME" ] || [ "${kind}" = "INT8_MIXED" ]; then
        case "${tile_shape}" in
            8x4|8x8) input_contract="IME_FULL_K_MAJOR" ;;
            *) continue ;;
        esac
        if [ "${lmul}" = "mf2" ] && [ "${ENABLE_MF2}" != "1" ]; then
            log "SKIP_EXPERIMENTAL_MF2=${kernel}"
            continue
        fi
    fi

    safe_name="${baseline}_${family}_${kernel}"
    exe="${BUILD_DIR}/${safe_name}"
    build_log="${RAW_LOG_DIR}/${safe_name}_build.log"

    log_kernel_header "${baseline}" "${family}" "${kernel}" "${kind}" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${input_contract}"

    sources=("${src}")
    if [ "${kind}" = "INT8_IME" ] || [ "${kind}" = "INT8_MIXED" ]; then
        sources+=("${fallback_src}")
    fi

    if ! compile_kernel "${kind}" "${march}" "${symbol}" "${input_contract}" "${exe}" "${build_log}" "${sources[@]}"; then
        build_failed_count=$((build_failed_count + 1))
        log "BUILD_FAILED: ${build_log}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
            "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" "NA" \
            "${M}" "${N}" "${K}" "${TILE_N}" "0" "BUILD_FAILED" "NA" "BUILD" "NA" "NA" "NA" \
            "NA" "NA" "NA" "NA" "NA" "$(csv_quote "${build_log}")" >> "${RAW_CSV}"
        continue
    fi

    kernel_count=$((kernel_count + 1))

    if [ "${BUILD_ONLY}" = "1" ]; then
        log "BUILD_ONLY_OK: ${exe}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
            "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" "NA" \
            "${M}" "${N}" "${K}" "${TILE_N}" "0" "BUILD_OK" "0" "NONE" "NA" "NA" "NA" \
            "NA" "NA" "NA" "NA" "NA" "$(csv_quote "${build_log}")" >> "${RAW_CSV}"
        continue
    fi

    for run in $(seq 1 "${RUNS}"); do
        run_log="${RAW_LOG_DIR}/${safe_name}_run${run}.log"
        run_validation="${OMP_VALIDATE}"
        if [ "${run}" -gt 1 ] && [ "${VALIDATE_EACH_RUN}" != "1" ]; then
            run_validation="0"
        fi
        run_binary_once "${exe}" "${run_log}" "${run_validation}"
        rc=$?
        csv_line="$(grep '^CSV_RUN,' "${run_log}" | tail -1 || true)"
        timing_scope="$(sed -n 's/^TIMING_SCOPE=//p' "${run_log}" | tail -1)"
        validation_method="$(sed -n 's/^VALIDATION_METHOD=//p' "${run_log}" | tail -1)"
        failure_stage="$(sed -n 's/^FAILURE_STAGE=//p' "${run_log}" | tail -1)"
        worker_placement="$(sed -n 's/^WORKER_PLACEMENT=//p' "${run_log}" | tail -1)"

        if [ -n "${csv_line}" ]; then
            IFS=',' read -r _ time_sec metric_value kernel_return failure_stage metric_name mismatch_count max_error actual_threads <<EOF_CSV
${csv_line}
EOF_CSV
            if [ "${mismatch_count}" != "0" ]; then
                status="NUMERICAL_FAILED"
                run_failed_count=$((run_failed_count + 1))
            elif [ "${kernel_return}" != "0" ] || [ "${rc}" != "0" ]; then
                status="KERNEL_RETURN"
                run_failed_count=$((run_failed_count + 1))
            elif [ "${actual_threads}" != "${OMP_NUM_THREADS}" ]; then
                status="THREAD_COUNT_MISMATCH"
                run_failed_count=$((run_failed_count + 1))
            else
                status="OK"
                ok_count=$((ok_count + 1))
            fi
        else
            status="RUN_FAILED"
            kernel_return="${rc}"
            time_sec="NA"
            metric_value="NA"
            metric_name="NA"
            mismatch_count="NA"
            max_error="NA"
            actual_threads="NA"
            failure_stage="PROCESS"
            timing_scope="NA"
            validation_method="NA"
            worker_placement="NA"
            run_failed_count=$((run_failed_count + 1))
        fi

        log_run_result "${run}" "${status}" "${metric_name}" "${metric_value}" "${time_sec}" "${actual_threads}" "${run_validation}" "${kernel_return}" "${failure_stage}" "${worker_placement}"

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" "$(csv_quote "${kernel}")" \
            "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" "${actual_threads}" \
            "${M}" "${N}" "${K}" "${TILE_N}" "${run}" "${status}" "${kernel_return}" "${failure_stage}" "${time_sec}" "${metric_name}" "${metric_value}" \
            "${timing_scope}" "${validation_method}" "${mismatch_count}" "${max_error}" "$(csv_quote "${worker_placement}")" \
            "$(csv_quote "${run_log}")" >> "${RAW_CSV}"
    done
done < <(find "${PROJECT_ROOT}" -type f -name Makefile -print0 | sort -z)

write_summary_csv

cp "${RAW_CSV}" "${RESULT_ROOT}/openmp_raw_latest_${MODE}.csv"
cp "${SUMMARY_CSV}" "${RESULT_ROOT}/openmp_summary_latest_${MODE}.csv"
cp "${LIVE_LOG}" "${RESULT_ROOT}/openmp_live_latest_${MODE}.log"

log "============================================================"
log "DONE"
log "kernels_built=${kernel_count}"
log "ok_runs=${ok_count}"
log "failed_runs=${run_failed_count}"
log "build_failed=${build_failed_count}"
log "Raw CSV: ${RAW_CSV}"
log "Summary CSV: ${SUMMARY_CSV}"
log "Latest raw CSV: ${RESULT_ROOT}/openmp_raw_latest_${MODE}.csv"
log "Latest summary CSV: ${RESULT_ROOT}/openmp_summary_latest_${MODE}.csv"
log "Latest live log: ${RESULT_ROOT}/openmp_live_latest_${MODE}.log"
log "Raw logs: ${RAW_LOG_DIR}"
log "============================================================"

if [ "${build_failed_count}" -ne 0 ] || [ "${run_failed_count}" -ne 0 ]; then
    exit 1
fi

exit 0


