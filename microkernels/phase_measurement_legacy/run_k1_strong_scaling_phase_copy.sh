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
    if ! command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "CHECK_FAILED: python3 is required for quoted CSV parsing and phase aggregation." >&2
        exit 1
    fi
    for marker in PHASE_PACKING_SEC PHASE_KERNEL_INCLUDING_OUTPUT_SEC PHASE_OUTPUT_SCOPE PHASE_SYNCHRONIZATION_SCOPE; do
        if ! grep -q "${marker}" "${MODULE_DIR}/src/openmp_heterogeneous_gemm.c"; then
            printf 'CHECK_FAILED missing_phase_marker=%s\n' "${marker}" >&2
            exit 1
        fi
    done
    printf '%s\n' "CHECK_OK: isolated phase driver and original kernel roots are available."
    printf '%s\n' "CHECK_EXPORT: quoted per-run phase CSV, grouped statistics, completeness report, and archived logs."
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
# Figure 2 reports seven measured repetitions.  Keep the value overrideable
# for smoke tests, but make the paper campaign default explicit and consistent.
RUNS="${RUNS:-7}"
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
PHASE_SUMMARY="${EXPERIMENT_DIR}/k1_strong_scaling_phase_timing_summary.csv"
PHASE_COMPLETENESS="${EXPERIMENT_DIR}/k1_strong_scaling_phase_completeness.csv"
PHASE_LOG_DIR="${EXPERIMENT_DIR}/phase_logs"
PHASE_LOG_MANIFEST="${EXPERIMENT_DIR}/phase_log_manifest.csv"
mkdir -p "${PHASE_LOG_DIR}"
printf '%s\n' 'series,parameter_name,parameter_value,mode,kernel,run,source_log,archived_log' > "${PHASE_LOG_MANIFEST}"

printf '%s\n' \
    'experiment,series,parameter_name,parameter_value,mode,kernel,kind,tile_shape,lmul,unroll,requested_threads,run,status,total_wall_sec,packing_worker_sum_sec,kernel_worker_sum_including_output_sec,output_scope,synchronization_scope,raw_log' \
    > "${PHASE_RAW}"

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

    # The runner CSV contains quoted fields (worker placement and absolute
    # paths).  Bash's IFS parser cannot parse those rows safely and was the
    # reason the earlier phase export had a 19-column header but 46-column
    # data rows.  Python's csv module is used here only as a standards-compliant
    # CSV reader/writer; it does not change the benchmark or its timing scope.
    python3 - "${runner_raw}" "${PHASE_RAW}" "${PHASE_LOG_DIR}" "${PHASE_LOG_MANIFEST}" \
        "${EXPERIMENT_NAME}" "${series}" "${parameter_name}" "${parameter_value}" \
        "${MODULE_DIR}" <<'PY'
import csv
import os
import re
import shutil
import sys

runner_raw, phase_raw, phase_log_dir, phase_log_manifest, experiment, series, parameter_name, parameter_value, module_dir = sys.argv[1:]

def value(row, *names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name]).strip()
    return ""

def resolve_log(raw):
    if not raw:
        return ""
    raw = raw.strip().strip('"')
    candidates = [raw]
    if not os.path.isabs(raw):
        candidates.extend([
            os.path.join(os.path.dirname(runner_raw), raw),
            os.path.join(module_dir, raw),
        ])
    for candidate in candidates:
        candidate = os.path.expanduser(candidate)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return raw

def marker(text, name):
    found = re.findall(r"^" + re.escape(name) + r"=(.*)$", text, re.MULTILINE)
    return found[-1].strip() if found else "NA"

with open(runner_raw, newline="", encoding="utf-8", errors="replace") as src, \
     open(phase_raw, "a", newline="", encoding="utf-8") as dst, \
     open(phase_log_manifest, "a", newline="", encoding="utf-8") as manifest_dst:
    reader = csv.DictReader(src)
    writer = csv.writer(dst, lineterminator="\n")
    manifest_writer = csv.writer(manifest_dst, lineterminator="\n")
    for row in reader:
        kernel = value(row, "kernel")
        if not kernel:
            continue
        raw_log = resolve_log(value(row, "log_file", "raw_log", "run_log"))
        text = ""
        if os.path.isfile(raw_log):
            with open(raw_log, encoding="utf-8", errors="replace") as log_src:
                text = log_src.read()

        archive_log = ""
        if os.path.isfile(raw_log):
            identity = "_".join([
                series,
                parameter_value,
                value(row, "mode"),
                kernel,
                value(row, "run"),
            ])
            safe_name = re.sub(
                r"[^A-Za-z0-9_.-]+", "_",
                identity + "_" + os.path.basename(raw_log),
            )
            archive_log = os.path.join(phase_log_dir, safe_name)
            shutil.copy2(raw_log, archive_log)
            manifest_writer.writerow([
                series,
                parameter_name,
                parameter_value,
                value(row, "mode"),
                kernel,
                value(row, "run"),
                raw_log,
                archive_log,
            ])

        writer.writerow([
            experiment,
            series,
            parameter_name,
            parameter_value,
            value(row, "mode"),
            kernel,
            value(row, "kind"),
            value(row, "tile_shape"),
            value(row, "lmul"),
            value(row, "unroll"),
            value(row, "requested_threads", "threads"),
            value(row, "run"),
            value(row, "status"),
            value(row, "time_sec", "total_sec", "total_wall_sec"),
            marker(text, "PHASE_PACKING_SEC"),
            marker(text, "PHASE_KERNEL_INCLUDING_OUTPUT_SEC"),
            marker(text, "PHASE_OUTPUT_SCOPE"),
            marker(text, "PHASE_SYNCHRONIZATION_SCOPE"),
            raw_log,
        ])
PY
}

build_phase_summary()
{
    python3 - "${PHASE_RAW}" "${PHASE_SUMMARY}" "${PHASE_COMPLETENESS}" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict

phase_raw, phase_summary, completeness = sys.argv[1:]
groups = defaultdict(list)
total_rows = ok_rows = complete_rows = 0

with open(phase_raw, newline="", encoding="utf-8", errors="replace") as src:
    for row in csv.DictReader(src):
        total_rows += 1
        if row.get("status") == "OK":
            ok_rows += 1
        fields = (row.get("total_wall_sec"), row.get("packing_worker_sum_sec"),
                  row.get("kernel_worker_sum_including_output_sec"))
        try:
            values = tuple(float(x) for x in fields)
        except (TypeError, ValueError):
            continue
        if any(not math.isfinite(x) for x in values):
            continue
        complete_rows += 1
        key = (row.get("series", ""), row.get("parameter_name", ""),
               row.get("parameter_value", ""), row.get("mode", ""),
               row.get("kernel", ""), row.get("kind", ""),
               row.get("tile_shape", ""), row.get("lmul", ""),
               row.get("unroll", ""), row.get("requested_threads", ""))
        groups[key].append(values)

def stats(items, index):
    vals = [x[index] for x in items]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return [len(vals), statistics.mean(vals), statistics.median(vals), sd,
            min(vals), max(vals)]

header = [
    "series", "parameter_name", "parameter_value", "mode", "kernel", "kind",
    "tile_shape", "lmul", "unroll", "requested_threads", "ok_phase_runs",
    "mean_total_wall_sec", "median_total_wall_sec", "sd_total_wall_sec",
    "min_total_wall_sec", "max_total_wall_sec", "mean_packing_sec",
    "median_packing_sec", "sd_packing_sec", "min_packing_sec", "max_packing_sec",
    "mean_kernel_including_output_sec", "median_kernel_including_output_sec",
    "sd_kernel_including_output_sec", "min_kernel_including_output_sec",
    "max_kernel_including_output_sec", "output_scope",
    "synchronization_scope",
]
with open(phase_summary, "w", newline="", encoding="utf-8") as dst:
    writer = csv.writer(dst, lineterminator="\n")
    writer.writerow(header)
    for key in sorted(groups):
        items = groups[key]
        wall = stats(items, 0)
        pack = stats(items, 1)
        kern = stats(items, 2)
        writer.writerow(list(key) + [
            wall[0], f"{wall[1]:.9f}", f"{wall[2]:.9f}", f"{wall[3]:.9f}",
            f"{wall[4]:.9f}", f"{wall[5]:.9f}",
            f"{pack[1]:.9f}", f"{pack[2]:.9f}", f"{pack[3]:.9f}",
            f"{pack[4]:.9f}", f"{pack[5]:.9f}",
            f"{kern[1]:.9f}", f"{kern[2]:.9f}", f"{kern[3]:.9f}",
            f"{kern[4]:.9f}", f"{kern[5]:.9f}",
            "FUSED_WITH_KERNEL_CALL", "NOT_SEPARATELY_INSTRUMENTED",
        ])

with open(completeness, "w", newline="", encoding="utf-8") as dst:
    writer = csv.writer(dst, lineterminator="\n")
    writer.writerow(["metric", "value"])
    writer.writerow(["phase_csv_rows", total_rows])
    writer.writerow(["status_ok_rows", ok_rows])
    writer.writerow(["rows_with_wall_packing_and_kernel", complete_rows])
    writer.writerow(["rows_missing_any_phase_value", total_rows - complete_rows])
PY
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
if ! build_phase_summary; then
    experiment_log "PHASE_RESULT=FAILED reason=phase_summary_generation"
    EXPERIMENT_FAILURES=$((EXPERIMENT_FAILURES + 1))
else
    experiment_log "Phase timing summary CSV: ${PHASE_SUMMARY}"
    experiment_log "Phase completeness CSV: ${PHASE_COMPLETENESS}"
    complete_rows="$(awk -F, 'NR > 1 && $15 != "NA" && $16 != "NA" { n++ } END { print n + 0 }' "${PHASE_RAW}")"
    ok_rows="$(awk -F, 'NR > 1 && $13 == "OK" { n++ } END { print n + 0 }' "${PHASE_RAW}")"
    if [ "${ok_rows}" -eq 0 ] || [ "${complete_rows}" -ne "${ok_rows}" ]; then
        experiment_log "PHASE_RESULT=FAILED reason=missing_phase_markers ok_rows=${ok_rows} complete_rows=${complete_rows}"
        EXPERIMENT_FAILURES=$((EXPERIMENT_FAILURES + 1))
    else
        experiment_log "PHASE_RESULT=OK rows=${complete_rows}"
    fi
fi
experiment_log "Phase scope: packing is worker-summed; kernel includes output stores; synchronization is not separately instrumented."
# Publish aliases beside the existing raw/summary aliases without changing
# the original benchmark tree.  The timestamped experiment directory remains
# the authoritative, self-contained archive.
LATEST_PHASE_PREFIX="${COMMON_RESULT_ROOT}/${EXPERIMENT_NAME}"
[ -f "${PHASE_SUMMARY}" ] && cp "${PHASE_SUMMARY}" "${LATEST_PHASE_PREFIX}_phase_summary_latest.csv"
[ -f "${PHASE_COMPLETENESS}" ] && cp "${PHASE_COMPLETENESS}" "${LATEST_PHASE_PREFIX}_phase_completeness_latest.csv"
if ! finish_experiment; then
    printf '%s\n' "${PHASE_RAW}" >&2
    exit 1
fi

printf '%s\n' "${PHASE_RAW}"
