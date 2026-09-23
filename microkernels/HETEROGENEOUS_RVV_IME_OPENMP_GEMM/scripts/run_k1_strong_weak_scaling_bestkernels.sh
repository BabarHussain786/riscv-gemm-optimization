#!/usr/bin/env bash
set -uo pipefail
export LC_ALL=C

# K1 strong + weak scaling using ONE FIXED kernel configuration per backend.
#
# Scientific rule:
#   - Choose the best VALIDATED RVV kernel from the tuning experiment.
#   - Choose the best VALIDATED IME kernel from the tuning experiment.
#   - Keep those two kernel choices fixed while scaling core count.
#
# This script records:
#   - execution time
#   - INT8 throughput (GOPS)
#   - process-level IPC from perf stat
#
# Put this file in:
#   HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/
#
# Default kernel names below are the previously tested LMUL1/unroll4 kernels.
# Replace them, or override RVV_KERNEL / IME_KERNEL at launch, after selecting
# the best validated tuning configuration for each backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="${MODULE_DIR}/results/paper_experiments"
COMMON_SCRIPT="${SCRIPT_DIR}/k1_experiment_common.sh"

if [ ! -f "${COMMON_SCRIPT}" ]; then
    printf 'ERROR: missing %s\n' "${COMMON_SCRIPT}" >&2
    exit 1
fi

# shellcheck source=k1_experiment_common.sh
source "${COMMON_SCRIPT}"

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
RUNS="${RUNS:-6}"
TILE_N="${TILE_N:-32}"

RVV_CORE_COUNTS="${RVV_CORE_COUNTS:-1 2 4 8}"
IME_CORE_COUNTS="${IME_CORE_COUNTS:-1 2 4}"

PERF_EVENTS_LIST="${PERF_EVENTS_LIST:-cycles,instructions,cache-references,cache-misses}"

# Strong scaling: fixed problem size.
STRONG_M="${STRONG_M:-1024}"
STRONG_N="${STRONG_N:-1024}"
STRONG_K="${STRONG_K:-1024}"

# Weak scaling:
# dimension ~= base_size * p^(1/3), rounded upward to WEAK_ALIGNMENT.
WEAK_BASE_SIZE="${WEAK_BASE_SIZE:-512}"
WEAK_ALIGNMENT="${WEAK_ALIGNMENT:-8}"

# ---------------------------------------------------------------------------
# FIXED backend kernels
# ---------------------------------------------------------------------------
# IMPORTANT:
# These are independent. RVV and IME do NOT need to use the same LMUL/unroll.
#
# Example RVV alternatives in the 8x4 INT8 family include source LMUL:
#   mf8, mf4, mf2, 1, 2
# with requested unroll:
#   1, 2, 4, 8
#
# Main IME path uses LMUL1 with unroll:
#   1, 2, 4, 8
#
# Set each to the best VALIDATED kernel determined before scaling.

RVV_KERNEL="${RVV_KERNEL:-igemm_kernel_8x4_zvl256b_lmul1_unroll4_i8i32}"
IME_KERNEL="${IME_KERNEL:-ime_kernel_8x4_zvl256b_lmul1_unroll4}"

RVV_KIND="${RVV_KIND:-INT8_RVV}"
IME_KIND="${IME_KIND:-INT8_IME}"

for value in "${RUNS}" "${TILE_N}" "${STRONG_M}" "${STRONG_N}" \
             "${STRONG_K}" "${WEAK_BASE_SIZE}" "${WEAK_ALIGNMENT}"; do
    require_positive_integer "experiment value" "${value}"
done

# Experimental IME mf2 kernels require the runner opt-in.
IME_ENABLE_MF2=0
case "${IME_KERNEL}" in
    *_lmulmf2_*) IME_ENABLE_MF2=1 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_DIR="${OUT_DIR:-${RESULT_ROOT}/k1_strong_weak_bestkernels_${STAMP}}"
mkdir -p "${MASTER_DIR}/strong" "${MASTER_DIR}/weak"

MASTER_LOG="${MASTER_DIR}/k1_strong_weak_bestkernels.log"
: > "${MASTER_LOG}"

master_log()
{
    printf '%s\n' "$*" | tee -a "${MASTER_LOG}"
}

scaled_dimension()
{
    local workers="$1"
    awk -v base="${WEAK_BASE_SIZE}" -v workers="${workers}" -v align="${WEAK_ALIGNMENT}" '
        BEGIN {
            target = base * exp(log(workers) / 3.0);
            rounded = int((target + align - 1) / align) * align;
            print rounded;
        }
    '
}

# Build plotting-ready mean/sample-SD CSVs.
make_metrics_csv()
{
    local input_csv="$1"
    local output_csv="$2"
    local tmp="${output_csv}.tmp"

    awk -F, '
        NR == 1 {
            for (i = 1; i <= NF; i++) idx[$i] = i;
            next;
        }
        {
            status = $(idx["status"]);
            if (status != "OK") next;

            series = $(idx["series"]);
            cores = $(idx["parameter_value"]);
            kernel = $(idx["kernel"]);
            unroll = $(idx["unroll"]);
            M = $(idx["M"]);
            N = $(idx["N"]);
            K = $(idx["K"]);
            metric_name = $(idx["metric_name"]);
            t = $(idx["time_sec"]);
            g = $(idx["metric_value"]);
            ipc = $(idx["perf_ipc"]);

            key = series SUBSEP cores SUBSEP kernel SUBSEP unroll SUBSEP \
                  M SUBSEP N SUBSEP K SUBSEP metric_name;

            if (t != "NA" && t != "" && t + 0 >= 0) {
                nt[key]++; st[key] += t; st2[key] += t * t;
            }
            if (g != "NA" && g != "" && g + 0 >= 0) {
                ng[key]++; sg[key] += g; sg2[key] += g * g;
            }
            if (ipc != "NA" && ipc != "" && ipc + 0 >= 0) {
                ni[key]++; si[key] += ipc; si2[key] += ipc * ipc;
            }

            s_series[key] = series;
            s_cores[key] = cores;
            s_kernel[key] = kernel;
            s_unroll[key] = unroll;
            s_M[key] = M;
            s_N[key] = N;
            s_K[key] = K;
            s_metric[key] = metric_name;
        }
        END {
            print "series,cores,kernel,unroll,M,N,K,throughput_metric,runs_ok,mean_time_s,sd_time_s,mean_GOPS,sd_GOPS,mean_IPC,sd_IPC";

            for (key in s_series) {
                mean_t = (nt[key] ? st[key] / nt[key] : 0);
                mean_g = (ng[key] ? sg[key] / ng[key] : 0);
                mean_i = (ni[key] ? si[key] / ni[key] : 0);

                sd_t = 0;
                sd_g = 0;
                sd_i = 0;

                if (nt[key] > 1)
                    sd_t = sqrt((st2[key] - st[key] * st[key] / nt[key]) / (nt[key] - 1));
                if (ng[key] > 1)
                    sd_g = sqrt((sg2[key] - sg[key] * sg[key] / ng[key]) / (ng[key] - 1));
                if (ni[key] > 1)
                    sd_i = sqrt((si2[key] - si[key] * si[key] / ni[key]) / (ni[key] - 1));

                t_mean = (nt[key] ? sprintf("%.10g", mean_t) : "NA");
                t_sd   = (nt[key] ? sprintf("%.10g", sd_t) : "NA");
                g_mean = (ng[key] ? sprintf("%.10g", mean_g) : "NA");
                g_sd   = (ng[key] ? sprintf("%.10g", sd_g) : "NA");
                i_mean = (ni[key] ? sprintf("%.10g", mean_i) : "NA");
                i_sd   = (ni[key] ? sprintf("%.10g", sd_i) : "NA");

                printf "%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s\n", \
                    s_series[key], s_cores[key], s_kernel[key], s_unroll[key], \
                    s_M[key], s_N[key], s_K[key], s_metric[key], nt[key], \
                    t_mean, t_sd, g_mean, g_sd, i_mean, i_sd;
            }
        }
    ' "${input_csv}" > "${tmp}"

    {
        head -n 1 "${tmp}"
        tail -n +2 "${tmp}" | sort -t, -k1,1 -k2,2n -k4,4n
    } > "${output_csv}"

    rm -f "${tmp}"
}

copy_experiment_outputs()
{
    local target_dir="$1"
    cp "${EXPERIMENT_RAW}" "${target_dir}/raw.csv"
    cp "${EXPERIMENT_SUMMARY}" "${target_dir}/summary.csv"
    cp "${EXPERIMENT_LOG}" "${target_dir}/experiment.log"
    make_metrics_csv "${EXPERIMENT_RAW}" "${target_dir}/metrics.csv"
}

master_log "============================================================"
master_log "K1 FIXED-KERNEL STRONG + WEAK SCALING"
master_log "RVV kernel: ${RVV_KERNEL}"
master_log "IME kernel: ${IME_KERNEL}"
master_log "RVV cores: ${RVV_CORE_COUNTS}"
master_log "IME cores: ${IME_CORE_COUNTS}"
master_log "runs=${RUNS}; tile_N=${TILE_N}; perf=ON"
master_log "============================================================"

# -----------------------------------------------------------------------------
# 1. STRONG SCALING
# -----------------------------------------------------------------------------
master_log "K1 STRONG SCALING"
master_log "Fixed GEMM=${STRONG_M}x${STRONG_N}x${STRONG_K}"

start_experiment "k1_strong_scaling_bestkernels"

CASE_M="${STRONG_M}"
CASE_N="${STRONG_N}"
CASE_K="${STRONG_K}"
CASE_TILE_N="${TILE_N}"
CASE_RUNS="${RUNS}"
CASE_SCHEDULE="static"
CASE_CHUNK="1"
CASE_IME_WEIGHT="4"
CASE_RVV_WEIGHT="1"
CASE_PERF_STAT="1"
CASE_PERF_EVENTS="${PERF_EVENTS_LIST}"
EXPERIMENT_QUIET_CASES="1"

for cores in ${RVV_CORE_COUNTS}; do
    require_positive_integer "RVV core count" "${cores}"
    CASE_MODE="k1-rvv"
    CASE_THREADS="${cores}"
    CASE_KERNEL="${RVV_KERNEL}"
    CASE_KIND="${RVV_KIND}"
    CASE_ENABLE_MF2="0"
    run_experiment_case "RVV" "cores" "${cores}" || true
done

for cores in ${IME_CORE_COUNTS}; do
    require_positive_integer "IME core count" "${cores}"
    CASE_MODE="k1-ime"
    CASE_THREADS="${cores}"
    CASE_KERNEL="${IME_KERNEL}"
    CASE_KIND="${IME_KIND}"
    CASE_ENABLE_MF2="${IME_ENABLE_MF2}"
    run_experiment_case "IME" "cores" "${cores}" || true
done

STRONG_FAILURES="${EXPERIMENT_FAILURES}"
copy_experiment_outputs "${MASTER_DIR}/strong"
if finish_experiment; then
    STRONG_RC=0
else
    STRONG_RC=1
fi

# -----------------------------------------------------------------------------
# 2. WEAK SCALING
# -----------------------------------------------------------------------------
master_log "============================================================"
master_log "K1 WEAK SCALING"
master_log "Base=${WEAK_BASE_SIZE}; alignment=${WEAK_ALIGNMENT}"
master_log "Rule: ceil_to_multiple_of_${WEAK_ALIGNMENT}(base * p^(1/3))"

start_experiment "k1_weak_scaling_bestkernels"

CASE_TILE_N="${TILE_N}"
CASE_RUNS="${RUNS}"
CASE_SCHEDULE="static"
CASE_CHUNK="1"
CASE_IME_WEIGHT="4"
CASE_RVV_WEIGHT="1"
CASE_PERF_STAT="1"
CASE_PERF_EVENTS="${PERF_EVENTS_LIST}"
EXPERIMENT_QUIET_CASES="1"

for cores in ${RVV_CORE_COUNTS}; do
    require_positive_integer "RVV core count" "${cores}"
    size="$(scaled_dimension "${cores}")"

    CASE_M="${size}"
    CASE_N="${size}"
    CASE_K="${size}"
    CASE_MODE="k1-rvv"
    CASE_THREADS="${cores}"
    CASE_KERNEL="${RVV_KERNEL}"
    CASE_KIND="${RVV_KIND}"
    CASE_ENABLE_MF2="0"

    run_experiment_case "RVV" "cores" "${cores}" || true
done

for cores in ${IME_CORE_COUNTS}; do
    require_positive_integer "IME core count" "${cores}"
    size="$(scaled_dimension "${cores}")"

    CASE_M="${size}"
    CASE_N="${size}"
    CASE_K="${size}"
    CASE_MODE="k1-ime"
    CASE_THREADS="${cores}"
    CASE_KERNEL="${IME_KERNEL}"
    CASE_KIND="${IME_KIND}"
    CASE_ENABLE_MF2="${IME_ENABLE_MF2}"

    run_experiment_case "IME" "cores" "${cores}" || true
done

WEAK_FAILURES="${EXPERIMENT_FAILURES}"
copy_experiment_outputs "${MASTER_DIR}/weak"
if finish_experiment; then
    WEAK_RC=0
else
    WEAK_RC=1
fi

# -----------------------------------------------------------------------------
# 3. MANIFEST
# -----------------------------------------------------------------------------
cat > "${MASTER_DIR}/manifest.txt" <<MANIFEST
K1 fixed-kernel strong + weak scaling campaign
Date: $(date)

Selected kernels:
  RVV_kernel=${RVV_KERNEL}
  RVV_kind=${RVV_KIND}
  IME_kernel=${IME_KERNEL}
  IME_kind=${IME_KIND}
  IME_enable_mf2=${IME_ENABLE_MF2}

Metrics:
  time = benchmark timed OpenMP GEMM region
  throughput = GOPS for INT8 GEMM
  IPC = external perf-stat process-level instructions/cycles
  perf_events = ${PERF_EVENTS_LIST}

Strong scaling:
  M=${STRONG_M}
  N=${STRONG_N}
  K=${STRONG_K}
  tile_N=${TILE_N}
  runs=${RUNS}
  RVV_core_counts=${RVV_CORE_COUNTS}
  IME_core_counts=${IME_CORE_COUNTS}
  failed_cases=${STRONG_FAILURES}

Weak scaling:
  base_size=${WEAK_BASE_SIZE}
  alignment=${WEAK_ALIGNMENT}
  size_rule=ceil_to_multiple_of_${WEAK_ALIGNMENT}(base_size * p^(1/3))
  tile_N=${TILE_N}
  runs=${RUNS}
  RVV_core_counts=${RVV_CORE_COUNTS}
  IME_core_counts=${IME_CORE_COUNTS}
  failed_cases=${WEAK_FAILURES}
MANIFEST

master_log "============================================================"
master_log "DONE"
master_log "Results root: ${MASTER_DIR}"
master_log "Strong plotting CSV: ${MASTER_DIR}/strong/metrics.csv"
master_log "Weak plotting CSV:   ${MASTER_DIR}/weak/metrics.csv"
master_log "Strong raw CSV:      ${MASTER_DIR}/strong/raw.csv"
master_log "Weak raw CSV:        ${MASTER_DIR}/weak/raw.csv"
master_log "Manifest:            ${MASTER_DIR}/manifest.txt"

if [ "${STRONG_RC}" -ne 0 ] || [ "${WEAK_RC}" -ne 0 ]; then
    exit 1
fi

exit 0
