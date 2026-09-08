#!/usr/bin/env bash
set -u

# RUNNER ROADMAP
# Step 1 -> Read mode, matrix size, tile width, and scheduling policy.
# Step 2 -> Find every compatible kernel in the project.
# Step 3 -> Compile one benchmark executable for each kernel.
# Step 4 -> Run validation once and repeat the timed measurement.
# Step 5 -> Read the benchmark output and append one raw CSV row.
# Step 6 -> Calculate per-kernel mean, median, and deviation.
# Step 7 -> Publish stable latest-file aliases under results/.

MODE="${1:-k3-rvv}"
M="${2:-1024}"
N="${3:-1024}"
K="${4:-1024}"
TILE_N="${5:-64}"
RUNS="${6:-6}"

# Select the scheduling experiment. Dynamic is valid only for mixed K1 mode.
# GEMM_* avoids the OMP_* namespace reserved by the OpenMP runtime. The old
# names are read once for command compatibility and then removed from children.
GEMM_TILE_SCHEDULE="${GEMM_TILE_SCHEDULE:-${OMP_TILE_SCHEDULE:-static}}"
GEMM_DYNAMIC_CHUNK="${GEMM_DYNAMIC_CHUNK:-${OMP_DYNAMIC_CHUNK:-1}}"
unset OMP_TILE_SCHEDULE OMP_DYNAMIC_CHUNK
RESULT_TAG="${MODE}"
if [ "${MODE}" = "k1-mixed-rvv-ime" ]; then
    RESULT_TAG="${MODE}-${GEMM_TILE_SCHEDULE}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_DIR}/.." && pwd)"
TEMPLATE="${MODULE_DIR}/src/openmp_heterogeneous_gemm.c"
RESULT_ROOT="${MODULE_DIR}/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_ROOT}/openmp_results_${RESULT_TAG}_${M}_${STAMP}"
BUILD_DIR="${RESULT_DIR}/build"
RAW_LOG_DIR="${RESULT_DIR}/raw_logs"
RAW_CSV="${RESULT_DIR}/openmp_raw_${RESULT_TAG}_${M}_runs${RUNS}_${STAMP}.csv"
SUMMARY_CSV="${RESULT_DIR}/openmp_summary_${RESULT_TAG}_${M}_runs${RUNS}_${STAMP}.csv"
LIVE_LOG="${RESULT_DIR}/openmp_live_${RESULT_TAG}_${M}_runs${RUNS}_${STAMP}.log"

CC="${CC:-gcc}"
ABI="${ABI:-lp64d}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O3 -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fopenmp}"
# This module intentionally benchmarks only its ZVL256 kernel set.
RVV_MARCH="rv64gcv_zvl256b"
OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
OMP_PLACES="${OMP_PLACES:-cores}"
OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
OMP_MAX_ACTIVE_LEVELS="${OMP_MAX_ACTIVE_LEVELS:-2}"
GEMM_VALIDATE="${GEMM_VALIDATE:-${OMP_VALIDATE:-1}}"
GEMM_WARMUP="${GEMM_WARMUP:-${OMP_WARMUP:-1}}"
unset OMP_VALIDATE OMP_WARMUP
VALIDATE_EACH_RUN="${VALIDATE_EACH_RUN:-0}"
ENABLE_MF2="${ENABLE_MF2:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
KERNEL_FILTER="${KERNEL_FILTER:-*}"
KIND_FILTER="${KIND_FILTER:-*}"
PERF_STAT="${PERF_STAT:-0}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-references,cache-misses}"
ZVL_FILTER="256b"
# BUILD_ONLY=1 compiles selected kernels and records BUILD_OK rows without timed execution.
# The project intentionally contains only the ZVL256 kernel set.

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
export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_DYNAMIC OMP_MAX_ACTIVE_LEVELS
export GEMM_TILE_SCHEDULE GEMM_DYNAMIC_CHUNK GEMM_WARMUP

case "${MODE}" in
    k1-rvv|k1-rvv-all|k3-rvv|k3-ime) MAX_THREADS=8 ;;
    k1-rvv-only|k1-ime|k3-ime-cluster0|k3-ime-cluster1) MAX_THREADS=4 ;;
    k1-mixed-rvv-ime) MAX_THREADS=8 ;;
    local) MAX_THREADS=0 ;;
esac

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
        k1-mixed-rvv-ime) printf '%s\n' "cores 0-3 execute native IME; cores 4-7 call the RVV kernel directly" ;;
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
        INT8_MIXED) printf '%s\n' "INT8 heterogeneous native-IME/RVV GEMM (INT8 x INT8 -> INT32)" ;;
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
    local static_tile_split="${11}"
    local schedule_policy="${12}"
    local schedule_chunk="${13}"
    local tile_distribution="${14}"

    if [ "${metric_name}" = "NA" ] || [ "${metric_value}" = "NA" ]; then
        log "    run ${run}: ${status} metric=NA time=${time_sec} threads=${actual_threads} validation=${validation} return=${return_code} stage=${failure_stage}"
    else
        log "    run ${run}: ${status} ${metric_name}=${metric_value} time=${time_sec} threads=${actual_threads} validation=${validation}"
    fi

    if [ -n "${static_tile_split}" ] && [ "${static_tile_split}" != "NA" ]; then
        log "      static split: ${static_tile_split}"
    fi
    log "      schedule: ${schedule_policy} chunk=${schedule_chunk}"
    if [ -n "${tile_distribution}" ] && [ "${tile_distribution}" != "NA" ]; then
        log "      completed output strips: ${tile_distribution}"
    fi
    if [ -n "${worker_placement}" ] && [ "${worker_placement}" != "NA" ]; then
        log "      workers: ${worker_placement}"
    fi
}

write_summary_csv()
{
    # SUMMARY BLOCK: Group equal experiments and calculate run statistics.
    awk -F, '
    BEGIN {
        OFS=",";
        print "mode,baseline,family,kernel,tile_shape,zvl,lmul,unroll,kind,core_group,requested_threads,M,N,K,tile_N,metric_name,static_tile_split,schedule_policy,schedule_chunk,ok_runs,mean_metric,median_metric,min_metric,max_metric,sample_std_metric,mean_time_sec,min_time_sec,max_time_sec,failed_runs,build_failed_runs";
    }
    NR == 1 { next }
    {
        baseline=$3; family=$4; kernel=$5;
        gsub(/^"|"$/, "", baseline);
        gsub(/^"|"$/, "", family);
        gsub(/^"|"$/, "", kernel);
        static_split=$30;
        gsub(/^"|"$/, "", static_split);
        key=$2 OFS baseline OFS family OFS kernel OFS $6 OFS $7 OFS $8 OFS $9 OFS $10 OFS $11 OFS $12 OFS $14 OFS $15 OFS $16 OFS $17 OFS $23 OFS static_split OFS $31 OFS $32;
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
        } else if ($19 == "BUILD_OK") {
            build_ok[key]++;
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
                if (n[key] > 1) variance=(sumsq_value[key] - n[key]*mean*mean) / (n[key]-1);
                else variance=0;
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

require_binary_flag()
{
    local name="$1"
    local value="$2"
    case "${value}" in
        0|1) ;;
        *)
            echo "${name} must be 0 or 1"
            exit 1
            ;;
    esac
}

require_positive_integer "M" "${M}"
require_positive_integer "N" "${N}"
require_positive_integer "K" "${K}"
require_positive_integer "tile_N" "${TILE_N}"
require_positive_integer "runs" "${RUNS}"
require_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"
require_positive_integer "GEMM_DYNAMIC_CHUNK" "${GEMM_DYNAMIC_CHUNK}"
require_binary_flag "GEMM_VALIDATE" "${GEMM_VALIDATE}"
require_binary_flag "GEMM_WARMUP" "${GEMM_WARMUP}"
require_binary_flag "VALIDATE_EACH_RUN" "${VALIDATE_EACH_RUN}"
require_binary_flag "ENABLE_MF2" "${ENABLE_MF2}"
require_binary_flag "BUILD_ONLY" "${BUILD_ONLY}"
require_binary_flag "PERF_STAT" "${PERF_STAT}"

if [ "${PERF_STAT}" = "1" ] && ! command -v perf >/dev/null 2>&1; then
    echo "PERF_STAT=1 requires the Linux perf command"
    exit 1
fi

if [ "${MODE}" != "local" ] && [ "${OMP_NUM_THREADS}" -gt "${MAX_THREADS}" ]; then
    echo "${MODE} supports at most ${MAX_THREADS} workers, requested ${OMP_NUM_THREADS}"
    exit 1
fi

if [ "${MODE}" = "k1-mixed-rvv-ime" ] && [ "${OMP_NUM_THREADS}" -ne 8 ]; then
    echo "k1-mixed-rvv-ime requires exactly 8 workers: 4 IME and 4 RVV"
    exit 1
fi

if [ "${MODE}" = "k1-mixed-rvv-ime" ]; then
    require_positive_integer "MIXED_IME_TILE_WEIGHT" "${MIXED_IME_TILE_WEIGHT:-4}"
    require_positive_integer "MIXED_RVV_TILE_WEIGHT" "${MIXED_RVV_TILE_WEIGHT:-1}"
fi

if [ "${BUILD_ONLY}" != "1" ] && [ "${GEMM_VALIDATE}" != "1" ]; then
    echo "GEMM_VALIDATE=1 is required for timed campaign results"
    exit 1
fi

case "${GEMM_TILE_SCHEDULE}" in
    static|dynamic) ;;
    *)
        echo "GEMM_TILE_SCHEDULE must be static or dynamic"
        exit 1
        ;;
esac

if [ "${MODE}" != "k1-mixed-rvv-ime" ] &&
   [ "${GEMM_TILE_SCHEDULE}" != "static" ]; then
    echo "Dynamic scheduling is available only for k1-mixed-rvv-ime"
    exit 1
fi

SCHEDULE_CHUNK_VALUE="0"
if [ "${GEMM_TILE_SCHEDULE}" = "dynamic" ]; then
    SCHEDULE_CHUNK_VALUE="${GEMM_DYNAMIC_CHUNK}"
fi

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

# Write one CSV row from separate fields; quoted fields stay untouched.
append_raw_row()
{
    local separator=""
    local field

    for field in "$@"; do
        printf '%s%s' "${separator}" "${field}" >> "${RAW_CSV}"
        separator=","
    done
    printf '\n' >> "${RAW_CSV}"
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

kernel_skip_reason()
{
    local kind="$1"
    local kernel_dir="$2"
    local tile_shape="$3"
    local lmul="$4"

    # Keep one canonical INT8 source tree. The FP64 baseline contains another
    # source file with the same INT8 kernel name but different implementation.
    if [ "${kind}" = "INT8_RVV" ]; then
        case "${kernel_dir}" in
            *GEMM_RVV_FP32_INT8_8x4_Baseline/*|*GEMM_RVV_FP32_INT8_8x8_Baseline/*) ;;
            *)
                printf '%s\n' "duplicate INT8 source; canonical tree is GEMM_RVV_FP32_INT8_*"
                return 0
                ;;
        esac
    fi

    # VLEN=256 gives only four FP32 lanes at LMUL=mf2. These 8x8 sources
    # advance by eight rows after loading one four-lane vector.
    if [ "${kind}" = "FP32_RVV" ] &&
       [ "${tile_shape}" = "8x8" ] && [ "${lmul}" = "mf2" ]; then
        printf '%s\n' "FP32 8x8 LMUL=mf2 has only four active lanes on VLEN=256"
        return 0
    fi

    # The FP32 8x8 N&4 cleanup handles only two columns when rows remain.
    if [ "${kind}" = "FP32_RVV" ] &&
       [ "${tile_shape}" = "8x8" ] &&
       [ $((N % 8)) -ge 4 ] && [ $((M % 8)) -ne 0 ]; then
        printf '%s\n' "FP32 8x8 boundary path is unsupported for this M,N shape"
        return 0
    fi

    # Eight FP64 rows require LMUL>=2 on the measured VLEN=256 machine.
    if [ "${kind}" = "FP64_RVV" ] &&
       [ "${tile_shape}" = "8x4" ] && [ "${lmul}" = "1" ]; then
        printf '%s\n' "FP64 8x4 LMUL=1 provides four lanes, not eight"
        return 0
    fi

    # INT8 widening e8 -> e16 -> e32 doubles EMUL twice. Starting from
    # LMUL4 or LMUL8 would exceed the architectural EMUL=8 limit; these source
    # variants use fallback/scalar arithmetic and are not the advertised path.
    if [ "${kind}" = "INT8_RVV" ] &&
       { [ "${lmul}" = "4" ] || [ "${lmul}" = "8" ]; }; then
        printf '%s\n' "INT8 LMUL=${lmul} is not a native widening-MAC configuration"
        return 0
    fi

    return 1
}

compile_kernel()
{
    # BUILD BLOCK: Select datatype macros and link the requested micro-kernel.
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
    local kernel_mr
    local kernel_nr
    local rvv_symbol
    local driver_obj native_obj rvv_obj

    kernel_mr="$(printf '%s\n' "${symbol}" | sed -n 's/.*kernel_\([0-9]*\)x[0-9]*_.*/\1/p')"
    kernel_nr="$(printf '%s\n' "${symbol}" | sed -n 's/.*kernel_[0-9]*x\([0-9]*\)_.*/\1/p')"
    if [ -z "${kernel_mr}" ] || [ -z "${kernel_nr}" ]; then
        return 1
    fi
    extra_defines+=("-DOMP_KERNEL_MR=${kernel_mr}")
    extra_defines+=("-DOMP_KERNEL_NR=${kernel_nr}")

    # Homogeneous modes also pin worker i to one exact core, rather than only
    # limiting the process to a broad taskset mask.
    case "${MODE}" in
        k1-rvv|k1-rvv-all|k3-rvv)
            extra_defines+=("-DOMP_FIRST_CPU=0")
            extra_defines+=("-DOMP_EXPECTED_THREADS=${OMP_NUM_THREADS}")
            ;;
        k1-rvv-only)
            extra_defines+=("-DOMP_FIRST_CPU=4")
            extra_defines+=("-DOMP_EXPECTED_THREADS=${OMP_NUM_THREADS}")
            ;;
        k1-ime)
            extra_defines+=("-DOMP_FIRST_CPU=0")
            extra_defines+=("-DOMP_EXPECTED_THREADS=${OMP_NUM_THREADS}")
            ;;
        k3-ime)
            extra_defines+=("-DOMP_FIRST_CPU=8")
            extra_defines+=("-DOMP_EXPECTED_THREADS=${OMP_NUM_THREADS}")
            ;;
        k3-ime-cluster0)
            extra_defines+=("-DOMP_FIRST_CPU=8")
            extra_defines+=("-DOMP_EXPECTED_THREADS=${OMP_NUM_THREADS}")
            ;;
        k3-ime-cluster1)
            extra_defines+=("-DOMP_FIRST_CPU=12")
            extra_defines+=("-DOMP_EXPECTED_THREADS=${OMP_NUM_THREADS}")
            ;;
    esac

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
            define_kind="-DOMP_KIND_INT8_MIXED"
            case "${input_contract}" in
                IME_FULL_K_MAJOR) extra_defines+=("-DOMP_IME_INPUT_FULL_MATRIX=1") ;;
                *) return 1 ;;
            esac
            # Mixed mode exposes two separate calls: required native IME and
            # the matching low-level RVV widening kernel.
            rvv_symbol="${symbol/#ime_kernel_/igemm_kernel_}_i8i32"
            extra_defines+=("-DSPACEMIT_IME_REQUIRE_HARDWARE=1")
            extra_defines+=("-DRVV_KERNEL_SYMBOL=${rvv_symbol}")
            # The two values define one static IME/RVV tile boundary.
            extra_defines+=("-DMIXED_IME_TILE_WEIGHT=${MIXED_IME_TILE_WEIGHT:-4}")
            extra_defines+=("-DMIXED_RVV_TILE_WEIGHT=${MIXED_RVV_TILE_WEIGHT:-1}")
            if [ "${ENABLE_MF2}" = "1" ] && printf '%s\n' "${symbol}" | grep -q '_lmulmf2_'; then
                extra_defines+=("-DSPACEMIT_IME_ENABLE_MF2_NATIVE=1")
            fi
            ;;
        *) return 1 ;;
    esac

    # Mixed mode contains two C functions with different names: the native
    # IME source keeps the IME symbol, while the RVV source keeps its own
    # igemm_*_i8i32 symbol.  Compile them separately so one global CNAME
    # cannot rename both functions to the same symbol.
    if [ "${kind}" = "INT8_MIXED" ]; then
        if [ "${#sources[@]}" -ne 2 ]; then
            return 1
        fi
        driver_obj="${exe}.driver.o"
        native_obj="${exe}.ime.o"
        rvv_obj="${exe}.rvv.o"
        {
            "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
                "${define_kind}" "${extra_defines[@]}" \
                -DKERNEL_SYMBOL="${symbol}" -c "${TEMPLATE}" -o "${driver_obj}"
            "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
                "${define_kind}" "${extra_defines[@]}" \
                -DCNAME="${symbol}" -c "${sources[0]}" -o "${native_obj}"
            "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
                "${define_kind}" "${extra_defines[@]}" \
                -DCNAME="${rvv_symbol}" -c "${sources[1]}" -o "${rvv_obj}"
            "${CC}" "${driver_obj}" "${native_obj}" "${rvv_obj}" -lm -o "${exe}"
        } > "${build_log}" 2>&1
        local rc=$?
        rm -f "${driver_obj}" "${native_obj}" "${rvv_obj}"
        return "${rc}"
    fi

    "${CC}" ${CFLAGS_COMMON} -march="${march}" -mabi="${ABI}" \
        "${define_kind}" "${extra_defines[@]}" \
        -DCNAME="${symbol}" -DKERNEL_SYMBOL="${symbol}" \
        "${TEMPLATE}" "${sources[@]}" -lm -o "${exe}" > "${build_log}" 2>&1
}

run_binary_once()
{
    # EXECUTION BLOCK: Apply the board CPU mask and run one repetition.
    local exe="$1"
    local run_log="$2"
    local run_validation="$3"
    local run_kind="$4"
    local perf_log="$5"
    local -a execution_env
    local -a benchmark_command

    # Remove inherited force flags so a previous shell command cannot silently
    # change the measured path. Native IME modes explicitly require IME.
    execution_env=(env -u SPACEMIT_IME_FORCE_SCALAR
                       -u SPACEMIT_IME_FORCE_RVV
                       -u SPACEMIT_IME_FORCE_NATIVE)
    case "${run_kind}" in
        INT8_IME|INT8_MIXED)
            execution_env+=(SPACEMIT_IME_FORCE_NATIVE=1)
            ;;
    esac

    if [ -n "${TASKSET_CORES}" ] && command -v taskset >/dev/null 2>&1; then
        benchmark_command=("${execution_env[@]}" GEMM_VALIDATE="${run_validation}"
                           taskset -c "${TASKSET_CORES}"
                           "${exe}" "${M}" "${N}" "${K}" "${TILE_N}")
    else
        benchmark_command=("${execution_env[@]}" GEMM_VALIDATE="${run_validation}"
                           "${exe}" "${M}" "${N}" "${K}" "${TILE_N}")
    fi

    if [[ "${MODE}" == k3-ime* ]] && [ -w /proc/set_ai_thread ]; then
        (
            echo "${BASHPID}" > /proc/set_ai_thread
            if [ "${PERF_STAT}" = "1" ]; then
                exec env LC_ALL=C perf stat -x, --no-big-num -e "${PERF_EVENTS}" \
                    -o "${perf_log}" -- "${benchmark_command[@]}"
            fi
            exec "${benchmark_command[@]}"
        ) > "${run_log}" 2>&1
    elif [ "${PERF_STAT}" = "1" ]; then
        env LC_ALL=C perf stat -x, --no-big-num -e "${PERF_EVENTS}" \
            -o "${perf_log}" -- "${benchmark_command[@]}" > "${run_log}" 2>&1
    else
        "${benchmark_command[@]}" > "${run_log}" 2>&1
    fi
}

perf_stat_value()
{
    local perf_log="$1"
    local requested_event="$2"

    if [ ! -s "${perf_log}" ]; then
        printf '%s\n' "NA"
        return 0
    fi

    awk -F, -v requested="${requested_event}" '
        {
            event=$3;
            value=$1;
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", event);
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value);
            if (event == requested || index(event, requested ":") == 1) {
                if (value ~ /^[0-9]+([.][0-9]+)?$/) print value;
                else print "NA";
                exit;
            }
        }
    ' "${perf_log}"
}

printf 'timestamp,mode,baseline,family,kernel,tile_shape,zvl,lmul,unroll,kind,core_group,requested_threads,actual_threads,M,N,K,tile_N,run,status,return_code,failure_stage,time_sec,metric_name,metric_value,timing_scope,validation_method,mismatch_count,max_error,worker_placement,static_tile_split,schedule_policy,schedule_chunk,output_strip_distribution,paired_rvv_kernel,log_file,perf_cycles,perf_instructions,perf_ipc,perf_cache_references,perf_cache_misses,perf_cache_miss_rate,perf_log_file\n' > "${RAW_CSV}"

log "OpenMP all-kernel tiled GEMM campaign"
log "MODE=${MODE} ($(mode_label)) M=${M} N=${N} K=${K} tile_N=${TILE_N} RUNS=${RUNS}"
log "CORE_PLAN=$(core_description)"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "RESULT_DIR=${RESULT_DIR}"
log "COMPILER=$(${CC} --version 2>/dev/null | sed -n '1p')"
log "BUILD_FLAGS=${CFLAGS_COMMON} -march=${RVV_MARCH} -mabi=${ABI}"
log "OMP_NUM_THREADS=${OMP_NUM_THREADS} OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES} OMP_DYNAMIC=${OMP_DYNAMIC} OMP_MAX_ACTIVE_LEVELS=${OMP_MAX_ACTIVE_LEVELS}"
log "TILE_SCHEDULE=${GEMM_TILE_SCHEDULE} DYNAMIC_CHUNK=${GEMM_DYNAMIC_CHUNK}"
log "GEMM_VALIDATE=${GEMM_VALIDATE} VALIDATE_EACH_RUN=${VALIDATE_EACH_RUN} GEMM_WARMUP=${GEMM_WARMUP} ENABLE_MF2=${ENABLE_MF2} BUILD_ONLY=${BUILD_ONLY} ZVL_FILTER=${ZVL_FILTER} KERNEL_FILTER=${KERNEL_FILTER} KIND_FILTER=${KIND_FILTER} PERF_STAT=${PERF_STAT} STATIC_IME_WEIGHT=${MIXED_IME_TILE_WEIGHT:-4} STATIC_RVV_WEIGHT=${MIXED_RVV_TILE_WEIGHT:-1}"
grep Cpus_allowed_list /proc/self/status 2>/dev/null | tee -a "${LIVE_LOG}" || true
if [ -r /sys/devices/system/node/online ]; then
    log "MEMORY_NODES_ONLINE=$(cat /sys/devices/system/node/online)"
else
    log "MEMORY_NODES_ONLINE=not_exposed"
fi
log "CLUSTER_PLACEMENT=exact_core_affinity; contiguous_output_ownership"

kernel_count=0
build_failed_count=0
run_failed_count=0
ok_count=0

while IFS= read -r -d '' makefile; do
    kernel_dir="$(dirname "${makefile}")"

    case "${kernel_dir}" in
        *HETEROGENEOUS_RVV_IME_OPENMP_GEMM*) continue ;;
    esac

    src=""
    canonical_rvv_src=""
    rvv_kernel="NA"
    kind=""

    if src="$(first_match "${kernel_dir}" 'ime_kernel_*.c')"; then
        if [ "${MODE}" = "k1-mixed-rvv-ime" ]; then
            kind="INT8_MIXED"
        else
            kind="INT8_IME"
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

    # Focused paper experiments can select one kernel and one datatype path.
    # The default '*' keeps the original all-kernel campaign unchanged.
    case "${kernel}" in
        ${KERNEL_FILTER}) ;;
        *) continue ;;
    esac
    case "${kind}" in
        ${KIND_FILTER}) ;;
        *) continue ;;
    esac

    tile_shape="$(field_from_kernel "${kernel}" 's/.*kernel_\([0-9]x[0-9]\)_zvl.*/\1/p')"
    zvl="$(field_from_kernel "${kernel}" 's/.*_zvl\([0-9]*b\).*/\1/p')"
    lmul="$(field_from_kernel "${kernel}" 's/.*_lmul\([^_]*\)_unroll.*/\1/p')"
    unroll="$(field_from_kernel "${kernel}" 's/.*_unroll\([0-9]*\).*/\1/p')"
    if [ "${zvl}" != "256b" ]; then
        log "SKIP_UNSUPPORTED_ZVL=${kernel} zvl=${zvl}; required=256b"
        continue
    fi
    march="${RVV_MARCH}"
    input_contract="CANONICAL_FULL_MATRIX"
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

    if skip_reason="$(kernel_skip_reason "${kind}" "${kernel_dir}" \
                                      "${tile_shape}" "${lmul}")"; then
        log "SKIP_UNSUPPORTED_KERNEL=${kernel}; reason=${skip_reason}"
        continue
    fi

    # Mixed mode must use the same canonical RVV source as the pure-RVV
    # campaign.  The similarly named rvv_fallback.c beside an IME kernel is a
    # different implementation and would make the comparison ambiguous.
    if [ "${kind}" = "INT8_MIXED" ]; then
        rvv_kernel="${symbol/#ime_kernel_/igemm_kernel_}"
        canonical_rvv_src="${PROJECT_ROOT}/GEMM_RVV_FP32_INT8_${tile_shape}_Baseline/RVV_IGEMM_INT8_I8I32_${tile_shape}/${rvv_kernel}/${rvv_kernel}_i8i32.c"
        if [ ! -f "${canonical_rvv_src}" ]; then
            log "SKIP_MISSING_CANONICAL_RVV=${kernel}; expected=${canonical_rvv_src}"
            continue
        fi
    fi

    safe_name="${baseline}_${family}_${kernel}"
    exe="${BUILD_DIR}/${safe_name}"
    build_log="${RAW_LOG_DIR}/${safe_name}_build.log"

    log_kernel_header "${baseline}" "${family}" "${kernel}" "${kind}" "${tile_shape}" "${zvl}" "${lmul}" "${unroll}" "${input_contract}"
    log "PRIMARY_SOURCE=${src}"
    if [ "${kind}" = "INT8_MIXED" ]; then
        log "MIXED_RVV_SOURCE=${canonical_rvv_src}"
    fi

    sources=("${src}")
    if [ "${kind}" = "INT8_MIXED" ]; then
        sources+=("${canonical_rvv_src}")
    fi

    if ! compile_kernel "${kind}" "${march}" "${symbol}" "${input_contract}" "${exe}" "${build_log}" "${sources[@]}"; then
        build_failed_count=$((build_failed_count + 1))
        log "BUILD_FAILED: ${build_log}"
        append_raw_row \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" \
            "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" \
            "$(csv_quote "${kernel}")" "${tile_shape}" "${zvl}" "${lmul}" \
            "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" "NA" \
            "${M}" "${N}" "${K}" "${TILE_N}" "0" "BUILD_FAILED" "NA" \
            "BUILD" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" \
            "${GEMM_TILE_SCHEDULE}" "${SCHEDULE_CHUNK_VALUE}" "NA" \
            "$(csv_quote "${rvv_kernel}")" \
            "$(csv_quote "${build_log}")" \
            "NA" "NA" "NA" "NA" "NA" "NA" "NA"
        continue
    fi

    kernel_count=$((kernel_count + 1))

    if [ "${BUILD_ONLY}" = "1" ]; then
        log "BUILD_ONLY_OK: ${exe}"
        append_raw_row \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" \
            "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" \
            "$(csv_quote "${kernel}")" "${tile_shape}" "${zvl}" "${lmul}" \
            "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" "NA" \
            "${M}" "${N}" "${K}" "${TILE_N}" "0" "BUILD_OK" "0" \
            "NONE" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" \
            "${GEMM_TILE_SCHEDULE}" "${SCHEDULE_CHUNK_VALUE}" "NA" \
            "$(csv_quote "${rvv_kernel}")" \
            "$(csv_quote "${build_log}")" \
            "NA" "NA" "NA" "NA" "NA" "NA" "NA"
        continue
    fi

    for run in $(seq 1 "${RUNS}"); do
        run_log="${RAW_LOG_DIR}/${safe_name}_run${run}.log"
        perf_log="${RAW_LOG_DIR}/${safe_name}_run${run}_perf.csv"
        run_validation="${GEMM_VALIDATE}"
        if [ "${run}" -gt 1 ] && [ "${VALIDATE_EACH_RUN}" != "1" ]; then
            run_validation="0"
        fi
        run_binary_once "${exe}" "${run_log}" "${run_validation}" "${kind}" "${perf_log}"
        rc=$?
        csv_line="$(grep '^CSV_RUN,' "${run_log}" | tail -1 || true)"
        timing_scope="$(sed -n 's/^TIMING_SCOPE=//p' "${run_log}" | tail -1)"
        validation_method="$(sed -n 's/^VALIDATION_METHOD=//p' "${run_log}" | tail -1)"
        failure_stage="$(sed -n 's/^FAILURE_STAGE=//p' "${run_log}" | tail -1)"
        static_tile_split="$(sed -n 's/^STATIC_TILE_SPLIT=//p' "${run_log}" | tail -1)"
        worker_placement="$(sed -n 's/^WORKER_PLACEMENT=//p' "${run_log}" | tail -1)"
        schedule_policy="$(sed -n 's/^SCHEDULING_POLICY=//p' "${run_log}" | tail -1)"
        schedule_chunk="$(sed -n 's/^SCHEDULE_CHUNK=//p' "${run_log}" | tail -1)"
        tile_distribution="$(sed -n 's/^TILE_DISTRIBUTION=//p' "${run_log}" | tail -1)"
        [ -n "${static_tile_split}" ] || static_tile_split="NA"
        [ -n "${schedule_policy}" ] || schedule_policy="${GEMM_TILE_SCHEDULE}"
        [ -n "${schedule_chunk}" ] || schedule_chunk="${SCHEDULE_CHUNK_VALUE}"
        [ -n "${tile_distribution}" ] || tile_distribution="NA"

        perf_cycles="NA"
        perf_instructions="NA"
        perf_ipc="NA"
        perf_cache_references="NA"
        perf_cache_misses="NA"
        perf_cache_miss_rate="NA"
        perf_log_field="NA"
        if [ "${PERF_STAT}" = "1" ]; then
            perf_cycles="$(perf_stat_value "${perf_log}" cycles)"
            perf_instructions="$(perf_stat_value "${perf_log}" instructions)"
            perf_cache_references="$(perf_stat_value "${perf_log}" cache-references)"
            perf_cache_misses="$(perf_stat_value "${perf_log}" cache-misses)"
            [ -n "${perf_cycles}" ] || perf_cycles="NA"
            [ -n "${perf_instructions}" ] || perf_instructions="NA"
            [ -n "${perf_cache_references}" ] || perf_cache_references="NA"
            [ -n "${perf_cache_misses}" ] || perf_cache_misses="NA"
            if [ "${perf_cycles}" != "NA" ] && [ "${perf_cycles}" != "0" ] &&
               [ "${perf_instructions}" != "NA" ]; then
                perf_ipc="$(awk -v ins="${perf_instructions}" -v cyc="${perf_cycles}" 'BEGIN { printf "%.10g", ins / cyc }')"
            fi
            if [ "${perf_cache_references}" != "NA" ] &&
               [ "${perf_cache_references}" != "0" ] &&
               [ "${perf_cache_misses}" != "NA" ]; then
                perf_cache_miss_rate="$(awk -v miss="${perf_cache_misses}" -v ref="${perf_cache_references}" 'BEGIN { printf "%.10g", 100.0 * miss / ref }')"
            fi
            perf_log_field="$(csv_quote "${perf_log}")"
        fi

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
            static_tile_split="NA"
            worker_placement="NA"
            schedule_policy="${GEMM_TILE_SCHEDULE}"
            schedule_chunk="${SCHEDULE_CHUNK_VALUE}"
            tile_distribution="NA"
            run_failed_count=$((run_failed_count + 1))
        fi

        log_run_result "${run}" "${status}" "${metric_name}" "${metric_value}" \
            "${time_sec}" "${actual_threads}" "${run_validation}" \
            "${kernel_return}" "${failure_stage}" "${worker_placement}" \
            "${static_tile_split}" "${schedule_policy}" "${schedule_chunk}" \
            "${tile_distribution}"
        if [ "${PERF_STAT}" = "1" ]; then
            log "      perf: IPC=${perf_ipc} cache_miss_rate=${perf_cache_miss_rate}% cycles=${perf_cycles} instructions=${perf_instructions}"
        fi

        append_raw_row \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${MODE}" \
            "$(csv_quote "${baseline}")" "$(csv_quote "${family}")" \
            "$(csv_quote "${kernel}")" "${tile_shape}" "${zvl}" "${lmul}" \
            "${unroll}" "${kind}" "${CORE_GROUP}" "${OMP_NUM_THREADS}" \
            "${actual_threads}" "${M}" "${N}" "${K}" "${TILE_N}" "${run}" \
            "${status}" "${kernel_return}" "${failure_stage}" "${time_sec}" \
            "${metric_name}" "${metric_value}" "${timing_scope}" \
            "${validation_method}" "${mismatch_count}" "${max_error}" \
            "$(csv_quote "${worker_placement}")" \
            "$(csv_quote "${static_tile_split}")" "${schedule_policy}" \
            "${schedule_chunk}" "$(csv_quote "${tile_distribution}")" \
            "$(csv_quote "${rvv_kernel}")" \
            "$(csv_quote "${run_log}")" \
            "${perf_cycles}" "${perf_instructions}" "${perf_ipc}" \
            "${perf_cache_references}" "${perf_cache_misses}" \
            "${perf_cache_miss_rate}" "${perf_log_field}"

        # Run 1 is the validation gate. Never collect unvalidated timing rows
        # after the tested implementation has already failed correctness.
        if [ "${run}" -eq 1 ] && [ "${run_validation}" = "1" ] &&
           [ "${status}" != "OK" ]; then
            log "    validation gate stopped remaining runs for ${kernel}"
            break
        fi
    done
done < <(find "${PROJECT_ROOT}" -type f -name Makefile -print0 | sort -z)

if [ "${kernel_count}" -eq 0 ]; then
    log "ERROR: no buildable kernel source files were found for mode ${MODE}"
    run_failed_count=$((run_failed_count + 1))
fi

write_summary_csv

log "============================================================"
if [ "${build_failed_count}" -eq 0 ] && [ "${run_failed_count}" -eq 0 ]; then
    log "DONE status=OK"
else
    log "DONE status=FAILED"
fi
log "kernels_built=${kernel_count}"
log "ok_runs=${ok_count}"
log "failed_runs=${run_failed_count}"
log "build_failed=${build_failed_count}"
log "Raw CSV: ${RAW_CSV}"
log "Summary CSV: ${SUMMARY_CSV}"
log "Latest raw CSV: ${RESULT_ROOT}/openmp_raw_latest_${RESULT_TAG}.csv"
log "Latest summary CSV: ${RESULT_ROOT}/openmp_summary_latest_${RESULT_TAG}.csv"
log "Latest live log: ${RESULT_ROOT}/openmp_live_latest_${RESULT_TAG}.log"
log "Raw logs: ${RAW_LOG_DIR}"
log "============================================================"

# Publish aliases only after the final DONE block has been written.
cp "${RAW_CSV}" "${RESULT_ROOT}/openmp_raw_latest_${RESULT_TAG}.csv"
cp "${SUMMARY_CSV}" "${RESULT_ROOT}/openmp_summary_latest_${RESULT_TAG}.csv"
cp "${LIVE_LOG}" "${RESULT_ROOT}/openmp_live_latest_${RESULT_TAG}.log"

if [ "${build_failed_count}" -ne 0 ] || [ "${run_failed_count}" -ne 0 ]; then
    exit 1
fi

exit 0
