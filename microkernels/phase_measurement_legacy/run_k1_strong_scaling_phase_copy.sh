#!/usr/bin/env bash
set -uo pipefail

# Isolated phase-aware copy of the proven Figure 2 campaign.
#
# Nothing under the original HETEROGENEOUS_RVV_IME_OPENMP_GEMM module or the
# original paper_results directory is changed.  The copied runner discovers
# and links the original kernels, but compiles the instrumented driver stored
# beside this script and writes all results below this new directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="${SCRIPT_DIR}/HETEROGENEOUS_RVV_IME_OPENMP_GEMM"
SOURCE_ROOT="${PHASE_SOURCE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export PHASE_SOURCE_ROOT="${SOURCE_ROOT}"

if [ "${1:-}" = "--check" ]; then
    for required in \
        "${SOURCE_ROOT}/GEMM_RVV_FP32_INT8_8x4_Baseline" \
        "${SOURCE_ROOT}/IME_NATIVE_KERNELS" \
        "${MODULE_DIR}/src/openmp_heterogeneous_gemm.c" \
        "${MODULE_DIR}/scripts/run_openmp_tiled_gemm_mode.sh"; do
        if [ ! -e "${required}" ]; then
            printf 'CHECK_FAILED missing=%s\n' "${required}" >&2
            exit 1
        fi
    done
    printf '%s\n' "CHECK_OK: isolated phase driver and original kernel roots are available."
    printf '%s\n' "CHECK_SCOPE: total wall time; worker-summed packing; kernel including output stores."
    printf '%s\n' "CHECK_NOTE: output reshaping and synchronization are not separate phases in the legacy implementation."
    exit 0
fi

# shellcheck source=/dev/null
source "${MODULE_DIR}/scripts/k1_experiment_common.sh"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
TILE_N="${TILE_N:-32}"
RUNS="${RUNS:-6}"
RVV_CORE_COUNTS="${RVV_CORE_COUNTS:-1 2 4 8}"
IME_CORE_COUNTS="${IME_CORE_COUNTS:-1 2 4}"
UNROLL_FACTORS="${UNROLL_FACTORS:-1 2 4 8}"
LMUL="${LMUL:-1}"
COLLECT_PERF="${COLLECT_PERF:-1}"
PERF_EVENTS_LIST="${PERF_EVENTS_LIST:-cycles,instructions,cache-references,cache-misses}"
EXPERIMENT_QUIET_CASES="${EXPERIMENT_QUIET_CASES:-1}"

for value in "${M}" "${N}" "${K}" "${TILE_N}" "${RUNS}"; do
    require_positive_integer "experiment value" "${value}"
done

start_experiment "k1_strong_scaling_phase"
PHASE_RAW="${EXPERIMENT_DIR}/k1_strong_scaling_phase_raw.csv"

printf '%s\n' \
    'experiment,series,parameter_name,parameter_value,mode,kernel,kind,tile_shape,lmul,unroll,requested_threads,run,status,total_wall_sec,packing_worker_sum_sec,kernel_worker_sum_including_output_sec,output_scope,synchronization_scope,raw_log' \
    > "${PHASE_RAW}"

csv_unquote()
{
    local value="$1"
    value="${value#\"}"
    value="${value%\"}"
    printf '%s' "${value}"
}

collect_phase_rows()
{
    local series="$1"
    local parameter_name="$2"
    local parameter_value="$3"
    local result_tag="${CASE_MODE}"
    local runner_raw="${MODULE_DIR}/results/openmp_raw_latest_${result_tag}.csv"
    [ "${CASE_MODE}" = "k1-mixed-rvv-ime" ] &&
        runner_raw="${MODULE_DIR}/results/openmp_raw_latest_${result_tag}-${CASE_SCHEDULE}.csv"

    if [ ! -s "${runner_raw}" ]; then
        experiment_log "PHASE_RESULT=FAILED reason=missing_runner_raw path=${runner_raw}"
        return 1
    fi

    while IFS=',' read -r timestamp mode baseline family kernel tile_shape zvl lmul \
        unroll kind core_group requested_threads actual_threads row_m row_n row_k \
        row_tile_n run status return_code failure_stage total_sec metric_name \
        metric_value timing_scope validation_method mismatch_count max_error \
        worker_placement static_tile_split schedule_policy schedule_chunk \
        output_strip_distribution paired_rvv_kernel log_file rest; do
        [ -n "${kernel:-}" ] || continue
        log_file="$(csv_unquote "${log_file:-}")"
        packing_sec="NA"
        kernel_sec="NA"
        output_scope="NA"
        synchronization_scope="NA"
        if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
            packing_sec="$(sed -n 's/^PHASE_PACKING_SEC=//p' "${log_file}" | tail -1)"
            kernel_sec="$(sed -n 's/^PHASE_KERNEL_INCLUDING_OUTPUT_SEC=//p' "${log_file}" | tail -1)"
            output_scope="$(sed -n 's/^PHASE_OUTPUT_SCOPE=//p' "${log_file}" | tail -1)"
            synchronization_scope="$(sed -n 's/^PHASE_SYNCHRONIZATION_SCOPE=//p' "${log_file}" | tail -1)"
        fi
        [ -n "${packing_sec}" ] || packing_sec="NA"
        [ -n "${kernel_sec}" ] || kernel_sec="NA"
        [ -n "${output_scope}" ] || output_scope="NA"
        [ -n "${synchronization_scope}" ] || synchronization_scope="NA"

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${EXPERIMENT_NAME}" "${series}" "${parameter_name}" "${parameter_value}" \
            "${mode}" "${kernel}" "${kind}" "${tile_shape}" "${lmul}" "${unroll}" \
            "${requested_threads}" "${run}" "${status}" "${total_sec}" \
            "${packing_sec}" "${kernel_sec}" "${output_scope}" \
            "${synchronization_scope}" "${log_file}" >> "${PHASE_RAW}"
    done < <(tail -n +2 "${runner_raw}")
}

CASE_M="${M}"; CASE_N="${N}"; CASE_K="${K}"
CASE_TILE_N="${TILE_N}"; CASE_RUNS="${RUNS}"
CASE_SCHEDULE="static"; CASE_CHUNK="1"
CASE_IME_WEIGHT="4"; CASE_RVV_WEIGHT="1"
CASE_ENABLE_MF2="0"
CASE_PERF_STAT="${COLLECT_PERF}"; CASE_PERF_EVENTS="${PERF_EVENTS_LIST}"

for unroll in ${UNROLL_FACTORS}; do
    require_positive_integer "unroll factor" "${unroll}"
    RVV_KERNEL="igemm_kernel_8x4_zvl256b_lmul${LMUL}_unroll${unroll}_i8i32"
    IME_KERNEL="ime_kernel_8x4_zvl256b_lmul${LMUL}_unroll${unroll}"

    for cores in ${RVV_CORE_COUNTS}; do
        CASE_MODE="k1-rvv"; CASE_THREADS="${cores}"
        CASE_KERNEL="${RVV_KERNEL}"; CASE_KIND="INT8_RVV"
        run_experiment_case "RVV" "cores" "${cores}" || true
        collect_phase_rows "RVV" "cores" "${cores}" || true
    done

    for cores in ${IME_CORE_COUNTS}; do
        CASE_MODE="k1-ime"; CASE_THREADS="${cores}"
        CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_IME"
        run_experiment_case "IME" "cores" "${cores}" || true
        collect_phase_rows "IME" "cores" "${cores}" || true
    done
done

experiment_log "Phase CSV: ${PHASE_RAW}"
experiment_log "Phase scope: packing is worker-summed; kernel includes output stores; synchronization is not separately instrumented."
finish_experiment || true

printf '%s\n' "${PHASE_RAW}"

