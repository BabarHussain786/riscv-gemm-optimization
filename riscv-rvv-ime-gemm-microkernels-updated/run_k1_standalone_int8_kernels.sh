#!/usr/bin/env bash
set -uo pipefail
export LC_ALL=C

# STANDALONE KERNEL BENCHMARK ROADMAP
# Step 1 -> Build one direct RVV or IME micro-kernel.
# Step 2 -> Validate the complete 1024x1024x1024 result once.
# Step 3 -> Run the same kernel several times without OpenMP.
# Step 4 -> Repeat for unroll factors 1, 2, 4, and 8.
# Step 5 -> Print one compact table and save raw and summary CSV files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
RVV_CORES="${RVV_CORES:-${RVV_CORE:-4 5 6 7}}"
IME_CORES="${IME_CORES:-${IME_CORE:-0 1 2 3}}"
COLLECT_PERF="${COLLECT_PERF:-1}"
PERF_EVENTS="${PERF_EVENTS:-cycles:u,instructions:u}"
PERF_EVENT_GROUP="${PERF_EVENT_GROUP:-1}"
if [ "${PERF_EVENT_GROUP}" = "1" ]; then
    PERF_EVENT_SPEC="{${PERF_EVENTS}}"
else
    PERF_EVENT_SPEC="${PERF_EVENTS}"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/k1_standalone_int8_results_${M}_${STAMP}}"
RAW_CSV="${OUT_DIR}/k1_standalone_int8_raw.csv"
SUMMARY_CSV="${OUT_DIR}/k1_standalone_int8_summary.csv"
LIVE_LOG="${OUT_DIR}/k1_standalone_int8.log"
BUILD_LOG_DIR="${OUT_DIR}/build_logs"
PERF_LOG_DIR="${OUT_DIR}/perf_logs"

RVV_ROOT="${SCRIPT_DIR}/GEMM_RVV_FP32_INT8_8x4_Baseline/RVV_IGEMM_INT8_I8I32_8x4"
IME_ROOT="${SCRIPT_DIR}/IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_8x4_NATIVE"

FAILURES=0

require_positive_integer()
{
    local name="$1"
    local value="$2"
    case "${value}" in
        ''|*[!0-9]*)
            printf 'ERROR: %s must be a positive integer, received %s\n' "${name}" "${value}" >&2
            exit 1
            ;;
    esac
    if [ "${value}" -le 0 ]; then
        printf 'ERROR: %s must be a positive integer, received %s\n' "${name}" "${value}" >&2
        exit 1
    fi
}

log()
{
    printf '%s\n' "$*" | tee -a "${LIVE_LOG}"
}

# CPU numbers start at zero; matrix sizes and run counts must stay positive.
require_core_id()
{
    case "$2" in
        ''|*[!0-9]*)
            printf 'ERROR: %s must be a non-negative integer, received %s\n' "$1" "$2" >&2
            exit 1
            ;;
    esac
}

parse_time()
{
    sed -nE 's/.*Time:[[:space:]]*([-+0-9.eE]+)[[:space:]]*sec.*/\1/p' |
        head -n 1
}

parse_gops()
{
    sed -nE 's/.*GOPS:[[:space:]]*([-+0-9.eE]+).*/\1/p' |
        head -n 1
}

parse_ipc()
{
    awk -F',' '
    {
        # perf stat -x,: count, unit, event, runtime, coverage, ...
        event=tolower($3);
        value=$1;
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", event);
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value);
        if (event ~ /^cycles/ && value ~ /^[0-9]+([.][0-9]+)?$/) cycles=value + 0;
        if (event ~ /^instructions/ && value ~ /^[0-9]+([.][0-9]+)?$/) instructions=value + 0;
    }
    END {
        if (cycles > 0 && instructions > 0) printf "%.10g", instructions / cycles;
    }'
}

run_pinned()
{
    local core="$1"
    local kind="$2"
    local validate="$3"
    shift 3
    local env_args=()

    if [ "${kind}" = "IME" ]; then
        env_args=(SPACEMIT_IME_FORCE_NATIVE=1 IME_VALIDATE="${validate}" GEMM_VALIDATE=0)
    else
        env_args=(SPACEMIT_IME_FORCE_RVV=1 IME_VALIDATE=0 GEMM_VALIDATE="${validate}")
    fi

    taskset -c "${core}" env "${env_args[@]}" ./bench "$@"
}

run_pinned_with_perf()
{
    local core="$1"
    local kind="$2"
    local validate="$3"
    local perf_log="$4"
    shift 4
    local env_args=()

    if [ "${kind}" = "IME" ]; then
        env_args=(SPACEMIT_IME_FORCE_NATIVE=1 IME_VALIDATE="${validate}" GEMM_VALIDATE=0)
    else
        env_args=(SPACEMIT_IME_FORCE_RVV=1 IME_VALIDATE=0 GEMM_VALIDATE="${validate}")
    fi

    perf stat -x, -e "${PERF_EVENT_SPEC}" -o "${perf_log}" -- \
        taskset -c "${core}" env "${env_args[@]}" ./bench "$@"
}

run_kernel()
{
    local kind="$1"
    local kernel_dir="$2"
    local kernel_name="$3"
    local core="$4"
    local build_log="${BUILD_LOG_DIR}/${kind}_${kernel_name}_build.log"
    local validation_output
    local validation_rc
    local run
    local output
    local rc
    local time_sec
    local gops
    local ipc
    local perf_log
    local perf_rc
    local status

    log "KERNEL=${kind} ${kernel_name} core=${core}"

    if ! (cd "${kernel_dir}" && make clean > "${build_log}" 2>&1 && make >> "${build_log}" 2>&1); then
        log "  BUILD_FAILED log=${build_log}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${kind}" "${kernel_name}" "${core}" "0" "BUILD_FAILED" "" "" "" "" "${M}x${N}x${K}" "${OUT_DIR}" "NA" "${build_log}" >> "${RAW_CSV}"
        FAILURES=$((FAILURES + 1))
        return
    fi

    if [ ! -x "${kernel_dir}/bench" ]; then
        log "  BENCH_MISSING path=${kernel_dir}/bench"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${kind}" "${kernel_name}" "${core}" "0" "BENCH_MISSING" "" "" "" "" "${M}x${N}x${K}" "${OUT_DIR}" "NA" "${build_log}" >> "${RAW_CSV}"
        FAILURES=$((FAILURES + 1))
        return
    fi

    if validation_output="$(cd "${kernel_dir}" && run_pinned "${core}" "${kind}" "1" "${M}" "${N}" "${K}" 2>&1)"; then
        validation_rc=0
    else
        validation_rc=$?
    fi

    if [ "${validation_rc}" -ne 0 ] ||
       ! printf '%s\n' "${validation_output}" | grep -q 'VALIDATION=OK'; then
        log "  VALIDATION_FAILED"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${kind}" "${kernel_name}" "${core}" "0" "VALIDATION_FAILED" "" "" "" "" "${M}x${N}x${K}" "${OUT_DIR}" "NA" "${build_log}" >> "${RAW_CSV}"
        FAILURES=$((FAILURES + 1))
        return
    fi

    log "  VALIDATION=OK; timed_runs=${RUNS}"

    for run in $(seq 1 "${RUNS}"); do
        perf_log="${PERF_LOG_DIR}/${kind}_${kernel_name}_core${core}_run${run}.perf.csv"
        if [ "${COLLECT_PERF}" = "1" ] && [ "${PERF_AVAILABLE}" = "1" ]; then
            output="$(cd "${kernel_dir}" && run_pinned_with_perf "${core}" "${kind}" "0" "${perf_log}" "${M}" "${N}" "${K}" 2>&1)"
            perf_rc=$?
            rc="${perf_rc}"
            if [ "${perf_rc}" -ne 0 ] && ! printf '%s\n' "${output}" | grep -q 'Time:'; then
                log "  IPC_UNAVAILABLE kernel=${kernel_name} core=${core} run=${run}; rerunning without perf"
                output="$(cd "${kernel_dir}" && run_pinned "${core}" "${kind}" "0" "${M}" "${N}" "${K}" 2>&1)"
                rc=$?
            fi
        else
            output="$(cd "${kernel_dir}" && run_pinned "${core}" "${kind}" "0" "${M}" "${N}" "${K}" 2>&1)"
            rc=$?
        fi

        time_sec="$(printf '%s\n' "${output}" | parse_time)"
        gops="$(printf '%s\n' "${output}" | parse_gops)"
        # perf can return a warning status when an event is unavailable even
        # though the benchmark itself completed and printed valid timings.
        if [ "${rc}" -ne 0 ] && [ -n "${time_sec}" ] && [ -n "${gops}" ]; then
            rc=0
        fi
        ipc="NA"
        if [ -s "${perf_log}" ]; then
            ipc="$(parse_ipc < "${perf_log}")"
            [ -n "${ipc}" ] || ipc="NA"
        fi

        if [ "${rc}" -eq 0 ] && [ -n "${time_sec}" ] && [ -n "${gops}" ]; then
            status="OK"
        else
            status="RUN_FAILED"
            FAILURES=$((FAILURES + 1))
        fi

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${kind}" "${kernel_name}" "${core}" "${run}" "${status}" \
            "${time_sec}" "${gops}" "OK" "${rc}" "${M}x${N}x${K}" "${OUT_DIR}" "${ipc}" "${perf_log}" >> "${RAW_CSV}"

        log "  run=${run} status=${status} time=${time_sec:-NA}s GOPS=${gops:-NA} IPC=${ipc}"
    done
}

write_summary()
{
    printf 'implementation,kernel,core,validation,ok_runs,failed_runs,mean_time_sec,mean_time_ms,mean_gops,min_gops,max_gops,mean_ipc,min_ipc,max_ipc,ipc_runs\n' > "${SUMMARY_CSV}"

    awk -F',' '
    NR == 1 { next }
    {
        key=$1 FS $2 FS $3;
        implementation[key]=$1;
        kernel[key]=$2;
        core[key]=$3;
        if ($5 == "OK") {
            ok[key]++;
            time_sum[key]+=$6;
            gops_sum[key]+=$7;
            if (!(key in gops_min) || $7 < gops_min[key]) gops_min[key]=$7;
            if (!(key in gops_max) || $7 > gops_max[key]) gops_max[key]=$7;
            if ($12 != "NA" && $12 != "") {
                ipc_count[key]++;
                ipc_sum[key]+=$12;
                if (!(key in ipc_min) || $12 < ipc_min[key]) ipc_min[key]=$12;
                if (!(key in ipc_max) || $12 > ipc_max[key]) ipc_max[key]=$12;
            }
        } else if ($5 != "BUILD_FAILED" && $5 != "BENCH_MISSING" && $5 != "VALIDATION_FAILED") {
            failed[key]++;
        }
        if ($5 == "BUILD_FAILED" || $5 == "BENCH_MISSING" || $5 == "VALIDATION_FAILED") failed[key]++;
        keys[key]=1;
    }
    END {
        for (key in keys) {
            n=ok[key]+0;
            if (n > 0) {
                mean_time=time_sum[key]/n;
                mean_gops=gops_sum[key]/n;
                ipc_value=(ipc_count[key] > 0 ? ipc_sum[key]/ipc_count[key] : "NA");
                ipc_min_value=(ipc_count[key] > 0 ? ipc_min[key] : "NA");
                ipc_max_value=(ipc_count[key] > 0 ? ipc_max[key] : "NA");
                printf "%s,%s,%s,OK,%d,%d,%.10g,%.6f,%.10g,%.10g,%.10g,%s,%s,%s,%d\n", \
                    implementation[key], kernel[key], core[key], n, failed[key]+0, \
                    mean_time, mean_time*1000.0, mean_gops, gops_min[key], gops_max[key], \
                    ipc_value, ipc_min_value, ipc_max_value, ipc_count[key]+0;
            } else {
                printf "%s,%s,%s,FAILED,0,%d,NA,NA,NA,NA,NA,NA,NA,NA,0\n", \
                    implementation[key], kernel[key], core[key], failed[key]+0;
            }
        }
    }' "${RAW_CSV}" | sort -t, -k1,1 -k2,2 >> "${SUMMARY_CSV}"
}

print_summary()
{
    log "============================================================"
    log "K1 STANDALONE INT8 KERNEL SUMMARY"
    log "GEMM=${M}x${N}x${K} | OpenMP=NO | LMUL=1 | tile=8x4 | IPC=${COLLECT_PERF}"
    log "-------------------------------------------------------------------------------------"
    log "path  kernel                                      core  valid  runs  ms       GOPS     IPC"
    log "-------------------------------------------------------------------------------------"

    awk -F',' 'NR > 1 {
        printf "%-5s %-44s %-5s %-6s %-5s %-8s %-8s %-8s\n", \
            $1, $2, $3, $4, $5, $8, $9, $12
    }' "${SUMMARY_CSV}" | while IFS= read -r line; do
        log "${line}"
    done

    log "-------------------------------------------------------------------------------------"
    log "Raw CSV: ${RAW_CSV}"
    log "Summary CSV: ${SUMMARY_CSV}"
    log "Detailed log: ${LIVE_LOG}"
    log "============================================================"
}

for value in "${M}" "${N}" "${K}" "${RUNS}"; do
    require_positive_integer "experiment value" "${value}"
done

# Check all core IDs before starting any benchmark.
for core in ${RVV_CORES} ${IME_CORES}; do
    require_core_id "CPU core" "${core}"
done

mkdir -p "${OUT_DIR}" "${BUILD_LOG_DIR}"
mkdir -p "${PERF_LOG_DIR}"
: > "${LIVE_LOG}"
printf 'implementation,kernel,core,run,status,time_sec,gops,validation,return_code,matrix,output_dir,ipc,perf_log\n' > "${RAW_CSV}"

if [ "${COLLECT_PERF}" = "1" ] && command -v perf >/dev/null 2>&1; then
    PERF_AVAILABLE=1
else
    PERF_AVAILABLE=0
fi

log "K1 standalone INT8 kernel benchmark"
log "No OpenMP, no outer tile_N, one pinned process per run"
log "RVV cores=${RVV_CORES}; IME cores=${IME_CORES}; runs=${RUNS}"
    log "IPC collection requested=${COLLECT_PERF}; perf_available=${PERF_AVAILABLE}; events=${PERF_EVENT_SPEC}"

for unroll in 1 2 4 8; do
    rvv_dir="igemm_kernel_8x4_zvl256b_lmul1_unroll${unroll}"
    rvv_kernel="${rvv_dir}_i8i32"
    ime_kernel="ime_kernel_8x4_zvl256b_lmul1_unroll${unroll}"

    for core in ${RVV_CORES}; do
        require_core_id "RVV core" "${core}"
        run_kernel "RVV" "${RVV_ROOT}/${rvv_dir}" "${rvv_kernel}" "${core}"
    done

    for core in ${IME_CORES}; do
        require_core_id "IME core" "${core}"
        run_kernel "IME" "${IME_ROOT}/${ime_kernel}" "${ime_kernel}" "${core}"
    done
done

write_summary
print_summary

if [ "${FAILURES}" -ne 0 ]; then
    log "DONE status=FAILED failures=${FAILURES}"
    exit 1
fi

log "DONE status=OK"
exit 0
