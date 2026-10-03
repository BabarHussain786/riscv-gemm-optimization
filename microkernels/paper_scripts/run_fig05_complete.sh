#!/usr/bin/env bash
set -euo pipefail

# Fair RVV-versus-RVV+IME end-to-end campaign.
#
# Both modes use the same OpenMP driver, matrix dimensions, tile width, number
# of repetitions, validation gate, and eight-worker K1 placement.  The timed
# OpenMP region includes the required input preparation for each path.
# The campaign intentionally selects the eight matched canonical pairs:
# 8x4/8x8, LMUL=1, and U1/U2/U4/U8.  It does not include experimental LMUL=mf2.
#
# This script deliberately does not modify any microkernel source.  It runs
# the existing OpenMP benchmark and copies the resulting CSV/log files into a
# dedicated campaign directory for Fig. 5 analysis.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODULE_SCRIPT="${PROJECT_ROOT}/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/run_openmp_tiled_gemm_mode.sh"
MODULE_RESULTS="${PROJECT_ROOT}/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/results"

M="1024"
N="1024"
K="1024"
TILE_N="32"
RUNS="7"
RUN_DYNAMIC="0"
INCLUDE_EXPERIMENTAL_IME="0"
OUTPUT_ROOT="${PROJECT_ROOT}/paper_results/fig05_fair_end_to_end_$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<'USAGE'
Usage:
  bash run_fig05_fair_end_to_end.sh [options]

Options:
  --size N       Use an NxNxN workload (default: 1024).
  --tile-n N     OpenMP output-strip width (default: 32).
  --runs N       Timed repetitions per kernel (default: 7).
  --dynamic      Also run the heterogeneous dynamic-scheduling case.
  --all-ime      Rejected: this matched campaign intentionally excludes LMUL=mf2.
  --output DIR   Destination directory for copied results.
  -h, --help     Show this help.

The script always runs:
  k1-rvv              RVV on cores 0-7 (8 workers)
  k1-mixed-rvv-ime   RVV+IME on cores 0-7 (8 workers, static schedule)

With --dynamic it additionally runs k1-mixed-rvv-ime with dynamic scheduling.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)
            [[ $# -ge 2 ]] || { echo "ERROR: --size needs a value" >&2; exit 2; }
            M="$2"; N="$2"; K="$2"; shift 2 ;;
        --tile-n)
            [[ $# -ge 2 ]] || { echo "ERROR: --tile-n needs a value" >&2; exit 2; }
            TILE_N="$2"; shift 2 ;;
        --runs)
            [[ $# -ge 2 ]] || { echo "ERROR: --runs needs a value" >&2; exit 2; }
            RUNS="$2"; shift 2 ;;
        --dynamic)
            RUN_DYNAMIC="1"; shift ;;
        --all-ime)
            echo "ERROR: --all-ime is incompatible with the matched Fig. 5 campaign; LMUL=mf2 is excluded." >&2
            exit 2 ;;
        --output)
            [[ $# -ge 2 ]] || { echo "ERROR: --output needs a value" >&2; exit 2; }
            OUTPUT_ROOT="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi

for value_name in M TILE_N RUNS; do
    value="${!value_name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: ${value_name} must be a positive integer" >&2
        exit 2
    }
done

[[ $((TILE_N % 8)) -eq 0 ]] || {
    echo "ERROR: --tile-n must be a multiple of 8" >&2
    exit 2
}
[[ -f "${MODULE_SCRIPT}" ]] || {
    echo "ERROR: OpenMP runner not found: ${MODULE_SCRIPT}" >&2
    exit 1
}

mkdir -p "${OUTPUT_ROOT}"
LOG="${OUTPUT_ROOT}/campaign.log"

log() {
    printf '%s\n' "$*" | tee -a "${LOG}"
}

copy_result_set() {
    local tag="$1"
    local destination="${OUTPUT_ROOT}/${tag}"
    mkdir -p "${destination}"

    for suffix in raw summary live; do
        local source="${MODULE_RESULTS}/openmp_${suffix}_latest_${tag}.csv"
        if [[ "${suffix}" == "live" ]]; then
            source="${MODULE_RESULTS}/openmp_live_latest_${tag}.log"
        fi
        if [[ -f "${source}" ]]; then
            cp -f "${source}" "${destination}/$(basename "${source}")"
            case "${suffix}" in
                raw) cp -f "${source}" "${destination}/raw.csv" ;;
                summary) cp -f "${source}" "${destination}/summary.csv" ;;
                live) cp -f "${source}" "${destination}/live.log" ;;
            esac
        else
            log "ERROR: expected result file missing: ${source}"
            return 1
        fi
    done

    local summary_file="${destination}/summary.csv"
    local raw_file="${destination}/raw.csv"
    local kernel_rows
    kernel_rows="$(awk -F',' 'NR > 1 {n++} END {print n + 0}' "${summary_file}")"
    if [[ "${kernel_rows}" -ne 8 ]]; then
        log "ERROR: expected 8 matched kernels in ${summary_file}, found ${kernel_rows}"
        return 1
    fi

    if ! awk -F',' '
        NR == 1 {
            for (i = 1; i <= NF; ++i) column[$i] = i
            required = (column["input_packing_time_sec"] &&
                        column["kernel_time_sec"] &&
                        column["output_packing_time_sec"] &&
                        column["phase_timing_scope"] &&
                        column["output_packing_status"])
            next
        }
        $(column["status"]) == "OK" &&
        $(column["timing_scope"]) == "parallel_tiles_including_required_packing" &&
        $(column["input_packing_time_sec"]) ~ /^[0-9]+([.][0-9]+)?$/ &&
        $(column["kernel_time_sec"]) ~ /^[0-9]+([.][0-9]+)?$/ &&
        $(column["output_packing_time_sec"]) ~ /^[0-9]+([.][0-9]+)?$/ &&
        $(column["phase_timing_scope"]) != "NA" &&
        $(column["output_packing_status"]) != "NA" {ok = 1}
        END { exit(required && ok ? 0 : 1) }
    ' "${raw_file}"; then
        log "ERROR: ${raw_file} has no successful end-to-end timing row with the expected timing scope"
        return 1
    fi
}

write_matched_outputs() {
    local rvv_file="${OUTPUT_ROOT}/k1-rvv/summary.csv"
    local mixed_file="${OUTPUT_ROOT}/k1-mixed-rvv-ime-static/summary.csv"
    local rvv_raw="${OUTPUT_ROOT}/k1-rvv/raw.csv"
    local mixed_raw="${OUTPUT_ROOT}/k1-mixed-rvv-ime-static/raw.csv"
    local out_file="${OUTPUT_ROOT}/fig05_matched_static_summary.csv"
    local raw_out_file="${OUTPUT_ROOT}/fig05_matched_static_raw.csv"

    [[ -f "${rvv_file}" && -f "${mixed_file}" && -f "${rvv_raw}" && -f "${mixed_raw}" ]] || {
        log "ERROR: cannot build matched outputs; static RVV or mixed data is missing"
        return 1
    }

    {
        awk 'NR == 1 { print "backend," $0; next } { print "RVV," $0 }' "${rvv_raw}"
        awk 'NR > 1 { print "RVV_IME," $0 }' "${mixed_raw}"
    } > "${raw_out_file}"

    awk -F',' -v OFS=',' -v requested_runs="${RUNS}" '
        function key_from_row() { return $5 "|" $7 "|" $8 }
        BEGIN {
            print "tile_shape","lmul","unroll",
                  "rvv_kernel","ime_kernel",
                  "rvv_ok_runs","ime_ok_runs",
                  "rvv_mean_time_sec","ime_mean_time_sec",
                  "rvv_median_metric","ime_median_metric",
                  "rvv_sample_std_metric","ime_sample_std_metric",
                  "rvv_mean_metric","ime_mean_metric",
                  "rvv_mean_input_packing_sec","ime_mean_input_packing_sec",
                  "rvv_mean_kernel_sec","ime_mean_kernel_sec",
                  "rvv_mean_output_packing_sec","ime_mean_output_packing_sec",
                  "rvv_phase_timing_scope","ime_phase_timing_scope",
                  "rvv_output_packing_status","ime_output_packing_status",
                  "rvv_status","ime_status"
        }
        FNR == 1 { next }
        FILENAME == ARGV[1] {
            k = key_from_row()
            rvv[k] = $4
            rvv_tile[k] = $5
            rvv_lmul[k] = $7
            rvv_unroll[k] = $8
            rvv_ok[k] = $20
            rvv_failed[k] = $29
            rvv_build_failed[k] = $30
            rvv_time[k] = $26
            rvv_med[k] = $22
            rvv_std[k] = $25
            rvv_metric[k] = $21
            rvv_input[k] = $31
            rvv_kernel_time[k] = $32
            rvv_output[k] = $33
            rvv_phase[k] = $34
            rvv_output_status[k] = $35
            next
        }
        FILENAME == ARGV[2] {
            k = key_from_row()
            if (!(k in rvv)) next
            ime_status = ($20 == requested_runs && $29 == 0 && $30 == 0) ? "OK" : "FAILED"
            rvv_status = (rvv_ok[k] == requested_runs && rvv_failed[k] == 0 && rvv_build_failed[k] == 0) ? "OK" : "FAILED"
            print $5, $7, $8, rvv[k], $4,
                  rvv_ok[k], $20, rvv_time[k], $26,
                  rvv_med[k], $22, rvv_std[k], $25,
                  rvv_metric[k], $21,
                  rvv_input[k], $31, rvv_kernel_time[k], $32,
                  rvv_output[k], $33, rvv_phase[k], $34,
                  rvv_output_status[k], $35,
                  rvv_status, ime_status
            seen[k] = 1
        }
        END {
            for (k in rvv) if (!(k in seen)) {
                split(k, p, "|")
                print p[1], p[2], p[3], rvv[k], "MISSING",
                      rvv_ok[k], "0", rvv_time[k], "NA",
                      rvv_med[k], "NA", rvv_std[k], "NA",
                      rvv_metric[k], "NA",
                      rvv_input[k], "NA", rvv_kernel_time[k], "NA",
                      rvv_output[k], "NA", rvv_phase[k], "NA",
                      rvv_output_status[k], "NA",
                      "FAILED", "MISSING"
            }
        }
    ' "${rvv_file}" "${mixed_file}" > "${out_file}"

    local pair_rows
    pair_rows="$(awk -F',' 'NR > 1 {n++} END {print n + 0}' "${out_file}")"
    if [[ "${pair_rows}" -ne 8 ]]; then
        log "ERROR: expected 8 matched RVV/IME rows, found ${pair_rows}"
        return 1
    fi

    local failed_pairs
    failed_pairs="$(awk -F',' 'NR > 1 && ($26 != "OK" || $27 != "OK") {n++} END {print n + 0}' "${out_file}")"
    if [[ "${failed_pairs}" -ne 0 ]]; then
        log "ERROR: ${failed_pairs} matched kernel pairs are not fully successful"
        return 1
    fi
    log "SAVED: ${raw_out_file}"
    log "SAVED: ${out_file}"
}

run_case() {
    local mode="$1"
    local schedule="$2"
    local tag="${mode}"
    local kind_filter="INT8_RVV"
    if [[ "${mode}" == "k1-mixed-rvv-ime" ]]; then
        tag="${mode}-${schedule}"
        kind_filter="INT8_MIXED"
    fi

    rm -f \
        "${MODULE_RESULTS}/openmp_raw_latest_${tag}.csv" \
        "${MODULE_RESULTS}/openmp_summary_latest_${tag}.csv" \
        "${MODULE_RESULTS}/openmp_live_latest_${tag}.log"

    log "============================================================"
    log "MODE=${mode} SCHEDULE=${schedule} M=${M} N=${N} K=${K} TILE_N=${TILE_N} RUNS=${RUNS}"
    log "TIMING_SCOPE=parallel_tiles_including_required_packing"

    if ! (
        cd "${PROJECT_ROOT}/HETEROGENEOUS_RVV_IME_OPENMP_GEMM"
        GEMM_TILE_SCHEDULE="${schedule}" \
        GEMM_DYNAMIC_CHUNK="1" \
        MIXED_IME_TILE_WEIGHT="4" \
        MIXED_RVV_TILE_WEIGHT="1" \
        GEMM_VALIDATE="1" \
        GEMM_WARMUP="1" \
        VALIDATE_EACH_RUN="0" \
        ENABLE_MF2="${INCLUDE_EXPERIMENTAL_IME}" \
        PERF_STAT="0" \
        KERNEL_FILTER="$([[ "${mode}" == "k1-rvv" ]] && printf '%s' 'igemm_kernel_8x[48]_zvl256b_lmul1_unroll[1248]_i8i32' || printf '%s' 'ime_kernel_8x[48]_zvl256b_lmul1_unroll[1248]')" \
        KIND_FILTER="${kind_filter}" \
        bash "${MODULE_SCRIPT}" "${mode}" "${M}" "${N}" "${K}" "${TILE_N}" "${RUNS}"
    ) 2>&1 | tee -a "${LOG}"; then
        log "FAILED: ${mode} ${schedule}"
        return 1
    fi

    copy_result_set "${tag}"
    log "SAVED: ${OUTPUT_ROOT}/${tag}"
}

print_summary_table() {
    printf '\n'
    printf '%-24s %-42s %-6s %-6s %-6s %7s %8s %12s %12s %-6s\n' \
        'MODE/SCHEDULE' 'KERNEL' 'TILE' 'LMUL' 'UNROLL' 'OK_RUNS' 'THREADS' 'MEAN_TIME(s)' 'MEAN_GOPS' 'STATUS'
    printf '%-24s %-42s %-6s %-6s %-6s %7s %8s %12s %12s %-6s\n' \
        '------------------------' '------------------------------------------' '------' '------' '------' '-------' '--------' '------------' '------------' '------'

    local file
    for file in "${OUTPUT_ROOT}"/*/openmp_summary_*.csv; do
        [[ -f "${file}" ]] || continue
        awk -F',' '
            NR > 1 {
                mode = $1 "/" $18
                status = ($20 + 0 > 0 && $29 + 0 == 0 && $30 + 0 == 0) ? "OK" : "FAILED"
                printf "%-24s %-42s %-6s %-6s %-6s %7s %8s %12s %12s %-6s\n",
                    mode, $4, $5, $7, $8, $20, $11, $26, $21, status
            }
        ' "${file}"
    done
}

log "Fair RVV versus RVV+IME end-to-end campaign"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "OUTPUT_ROOT=${OUTPUT_ROOT}"
log "Same workload, tile width, validation, and eight-worker placement are used for both paths."
log "Matched kernel set: 8x4/8x8, LMUL=1, U1/U2/U4/U8; experimental LMUL=mf2 is excluded."
log "Phase timing: input packing, kernel, output-packing status, and total wall time are recorded."

failures=0
if ! run_case "k1-rvv" "static"; then failures=$((failures + 1)); fi
if ! run_case "k1-mixed-rvv-ime" "static"; then failures=$((failures + 1)); fi
if [[ "${RUN_DYNAMIC}" == "1" ]]; then
    if ! run_case "k1-mixed-rvv-ime" "dynamic"; then failures=$((failures + 1)); fi
fi

print_summary_table

if [[ "${failures}" -eq 0 ]]; then
    write_matched_outputs || failures=$((failures + 1))
fi

log "============================================================"
if [[ "${failures}" -eq 0 ]]; then
    log "DONE status=OK"
else
    log "DONE status=FAILED failures=${failures}"
fi
log "Results are in: ${OUTPUT_ROOT}"

if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
