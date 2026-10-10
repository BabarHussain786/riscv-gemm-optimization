#!/usr/bin/env bash
set -Eeuo pipefail

# K1 strong scaling with phase timing.
#
# This is deliberately separate from run_k1_strong_scaling.sh.  It uses the
# repository's phase-aware benchmarking driver and records:
#   * unprofiled end-to-end timing (the primary scaling measurement),
#   * profiled packing/kernel/output/boundary observations, and
#   * optional counter observations.
#
# Profile values are worker-summed elapsed times, not a wall-clock
# decomposition.  In the RVV path, output and boundary handling are fused into
# kernel_sec; the IME path exposes its output and boundary contributions.

set -o errtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${PROJECT_ROOT}/benchmarking"
RUNNER="${BENCH_ROOT}/run.py"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/k1_strong_scaling_phase_results}"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-7}"
WARMUPS="${WARMUPS:-2}"
SEED="${SEED:-42}"
RVV_CORE_COUNTS="${RVV_CORE_COUNTS:-1 2 4 8}"
IME_CORE_COUNTS="${IME_CORE_COUNTS:-1 2 4}"
RVV_CPUS="${RVV_CPUS:-0,1,2,3,4,5,6,7}"
IME_CPUS="${IME_CPUS:-0,1,2,3}"
COLLECT_COUNTERS="${COLLECT_COUNTERS:-1}"
TIMEOUT="${TIMEOUT:-1800}"
CC="${CC:-}"
SELECTION_FILE="${SELECTION_FILE:-}"

usage() {
    cat <<'HELP'
Usage: bash run_k1_strong_scaling_phase_timing.sh [--check|--help]

Runs fixed-1024^3 strong scaling for the selected 8x4 and 8x8 INT8 kernel
pairs.  For every backend/core-count point it records an unprofiled primary
end-to-end pass and a separate profiled pass.  Counter passes are enabled by
default and can be disabled with COLLECT_COUNTERS=0.

Environment overrides include M, N, K, RUNS, WARMUPS, SEED, RESULT_ROOT,
RVV_CPUS, IME_CPUS, RVV_CORE_COUNTS, IME_CORE_COUNTS, SELECTION_FILE, CC,
and COLLECT_COUNTERS.

The old strong-scaling script is not modified.  Use --check for a read-only
contract check; no build or benchmark is performed by that option.
HELP
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || die "$2 must be a positive integer: $1"
}

join_by_comma() {
    local first=1 value
    for value in "$@"; do
        if (( first )); then
            first=0
        else
            printf ','
        fi
        printf '%s' "$value"
    done
}

cpu_prefix() {
    local list=$1 count=$2
    local -a cpus=()
    local cpu
    IFS=',' read -r -a cpus <<< "$list"
    (( ${#cpus[@]} >= count )) || die "CPU list '$list' has fewer than $count entries"
    for cpu in "${cpus[@]:0:count}"; do
        [[ "$cpu" =~ ^[0-9]+$ ]] || die "invalid CPU id '$cpu' in '$list'"
    done
    join_by_comma "${cpus[@]:0:count}"
}

kernel_source_exists() {
    local backend=$1 tile=$2 kernel=$3
    if [[ "$backend" == rvv ]]; then
        [[ -f "${PROJECT_ROOT}/GEMM_RVV_FP32_INT8_${tile}_Baseline/RVV_IGEMM_INT8_I8I32_${tile}/${kernel}/${kernel}_i8i32.c" ]]
    else
        [[ -f "${PROJECT_ROOT}/IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_${tile}_NATIVE/${kernel}/${kernel}.c" ]]
    fi
}

declare -A RVV_KERNELS=()
declare -A IME_KERNELS=()

load_kernel_selection() {
    local selection="$SELECTION_FILE"
    local newest
    local tile rvv ime

    if [[ -z "$selection" ]]; then
        newest=$(find "${PROJECT_ROOT}/NEW BENCHMARK/results" -type f -name selected_kernels.tsv -print 2>/dev/null | sort | tail -n 1 || true)
        selection="$newest"
    fi

    if [[ -n "$selection" && -f "$selection" ]]; then
        printf 'Using kernel selection: %s\n' "$selection"
        while IFS=$'\t' read -r tile rvv ime; do
            [[ -n "$tile" && "$tile" != tile ]] || continue
            ime=${ime%$'\r'}
            [[ "$tile" == 8x4 || "$tile" == 8x8 ]] || die "invalid tile in selection: $tile"
            [[ -n "$rvv" && -n "$ime" ]] || die "incomplete selection row for tile $tile"
            [[ -z "${RVV_KERNELS[$tile]+set}" ]] || die "duplicate selection for tile $tile"
            RVV_KERNELS["$tile"]="$rvv"
            IME_KERNELS["$tile"]="$ime"
        done < "$selection"
    else
        # Explicit documented starting pairs.  A Figure 5 selection file is
        # preferred automatically when one exists; these defaults keep this
        # standalone Figure 2 runner usable without a completed Figure 5 run.
        printf '%s\n' 'No selected_kernels.tsv found; using documented starting pairs.'
        RVV_KERNELS[8x4]="${RVV_KERNEL_8X4:-igemm_kernel_8x4_zvl256b_lmulmf8_unroll2}"
        IME_KERNELS[8x4]="${IME_KERNEL_8X4:-ime_kernel_8x4_zvl256b_lmul1_unroll1}"
        RVV_KERNELS[8x8]="${RVV_KERNEL_8X8:-igemm_kernel_8x8_zvl256b_lmulmf4_unroll4}"
        IME_KERNELS[8x8]="${IME_KERNEL_8X8:-ime_kernel_8x8_zvl256b_lmul1_unroll1}"
    fi

    for tile in 8x4 8x8; do
        [[ -n "${RVV_KERNELS[$tile]+set}" && -n "${IME_KERNELS[$tile]+set}" ]] || die "selection must contain both $tile kernel pairs"
        kernel_source_exists rvv "$tile" "${RVV_KERNELS[$tile]}" || die "RVV source not found for $tile: ${RVV_KERNELS[$tile]}"
        kernel_source_exists ime "$tile" "${IME_KERNELS[$tile]}" || die "IME source not found for $tile: ${IME_KERNELS[$tile]}"
    done
}

check_contract() {
    require_command python3
    require_command find
    [[ -f "$RUNNER" ]] || die "benchmark runner not found: $RUNNER"
    [[ -f "${BENCH_ROOT}/build.py" ]] || die "benchmark build helper not found"
    positive_integer "$M" M
    positive_integer "$N" N
    positive_integer "$K" K
    positive_integer "$RUNS" RUNS
    [[ "$WARMUPS" =~ ^[0-9]+$ ]] || die "WARMUPS must be a nonnegative integer"
    positive_integer "$SEED" SEED
    positive_integer "$TIMEOUT" TIMEOUT
    load_kernel_selection
    local count
    for count in $RVV_CORE_COUNTS; do positive_integer "$count" RVV_CORE_COUNTS; done
    for count in $IME_CORE_COUNTS; do positive_integer "$count" IME_CORE_COUNTS; done
    cpu_prefix "$RVV_CPUS" "$(echo "$RVV_CORE_COUNTS" | awk '{print $NF}')" >/dev/null
    cpu_prefix "$IME_CPUS" "$(echo "$IME_CORE_COUNTS" | awk '{print $NF}')" >/dev/null
    printf 'Contract OK: fixed %sx%sx%s, runs=%s, warmups=%s, RVV cores=[%s], IME cores=[%s]\n' \
        "$M" "$N" "$K" "$RUNS" "$WARMUPS" "$RVV_CORE_COUNTS" "$IME_CORE_COUNTS"
    printf 'Phase scope: total wall time plus profiled worker phases; RVV output/boundary remain fused.\n'
}

write_scope() {
    cat > "${RESULT_ROOT}/measurement_scope.txt" <<'SCOPE'
Primary timing: end_to_end, unprofiled, seven measured repetitions by default.
The primary total includes assignment, input packing, kernel execution, required
output/scatter and boundary work, and timed synchronization. Allocation,
initialization, reference generation, validation, and cleanup are excluded.

Profile timing: a separate end_to_end profiled pass for the same configuration.
packing_sec and kernel_sec are worker-summed elapsed phases. For IME,
ime_output_sec and ime_boundary_sec identify the IME contribution. RVV output
and boundary handling are fused into kernel_sec and are intentionally null as
separate fields. Profile phase values are not a wall-clock decomposition and
must not be added to total_sec.

Counter timing: optional separate diagnostic pass. Counter values cover worker
measured regions and are not used as primary speedup measurements.

Every invocation writes its native raw/accepted/failed records under runs/.
The consolidated phase_timing_records.csv contains accepted rows only, while
phase_timing_summary.csv reports per-point means, medians, and sample SDs.
SCOPE
}

RUNS_DIR=""
MANIFEST=""

run_one() {
    local tile=$1 backend=$2 cores=$3 role=$4
    local rvv="${RVV_KERNELS[$tile]}" ime="${IME_KERNELS[$tile]}"
    local impl threads ime_workers cpus
    if [[ "$backend" == rvv ]]; then
        impl=rvv; threads=$cores; ime_workers=0
        cpus=$(cpu_prefix "$RVV_CPUS" "$cores")
    else
        impl=ime; threads=$cores; ime_workers=$cores
        cpus=$(cpu_prefix "$IME_CPUS" "$cores")
    fi

    local label="${tile}_${backend}_${cores}_${role}"
    local log="${RESULT_ROOT}/logs/${label}.log"
    local -a command=(python3 "$RUNNER" run
        --project-root "$PROJECT_ROOT"
        --output "$RUNS_DIR"
        --rvv-kernel "$rvv"
        --ime-kernel "$ime"
        --implementation "$impl"
        --timing end_to_end
        --m "$M" --n "$N" --k "$K"
        --threads "$threads" --ime-workers "$ime_workers"
        --cpus "$cpus" --schedule static --weight 4 --chunk 1
        --warmups "$WARMUPS" --repetitions "$RUNS" --seed "$SEED"
        --timeout "$TIMEOUT")
    case "$role" in
        primary) ;;
        profile) command+=(--profile) ;;
        counters) command+=(--counters) ;;
        *) die "unknown role: $role" ;;
    esac
    if [[ -n "$CC" ]]; then command+=(--cc "$CC"); fi

    printf 'Running %-28s (%s)\n' "$label" "$role"
    {
        printf 'COMMAND: '
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$log"
    set +e
    "${command[@]}" >> "$log" 2>&1
    local rc=$?
    set -e

    local campaign_dir
    campaign_dir=$(grep -E '^/.+/(run|campaign|tuning|repeatability)_[0-9]{8}T[0-9]{6}' "$log" | tail -n 1 || true)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$tile" "$backend" "$cores" "$role" "$rvv" "$ime" "$cpus" "$rc" "$campaign_dir" "$log" >> "$MANIFEST"
    if (( rc != 0 )); then
        printf 'WARNING: %s failed; its diagnostics remain in %s\n' "$label" "$log" >&2
    fi
}

aggregate_results() {
    python3 - "$MANIFEST" "$RESULT_ROOT" <<'PY'
import csv
import math
import statistics
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
result_root = Path(sys.argv[2])
fields = ["tile", "backend", "cores", "role", "rvv_kernel", "ime_kernel",
          "cpus", "return_code", "campaign_dir", "log_file"]
records = []
with manifest_path.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream, fieldnames=fields, delimiter="\t")
    next(reader, None)  # discard the explicit header written by the shell driver
    for item in reader:
        campaign = item["campaign_dir"]
        if not campaign:
            continue
        accepted = Path(campaign) / "accepted_runs.csv"
        if not accepted.is_file():
            continue
        with accepted.open(newline="", encoding="utf-8") as rows:
            for row in csv.DictReader(rows):
                row.update({"tile": item["tile"], "backend": item["backend"],
                            "cores": item["cores"], "role": item["role"],
                            "rvv_kernel": item["rvv_kernel"], "ime_kernel": item["ime_kernel"],
                            "cpus": item["cpus"], "campaign_dir": campaign})
                records.append(row)

record_fields = ["tile", "backend", "cores", "role", "rvv_kernel", "ime_kernel", "cpus",
                 "campaign_dir", "case_id", "rep", "status", "validation", "M", "N", "K",
                 "threads", "total_sec", "gops", "packing_sec", "kernel_sec", "output_sec",
                 "boundary_sec", "ime_output_sec", "ime_boundary_sec", "cycles", "instructions",
                 "ipc", "cache_references", "cache_misses"]
with (result_root / "phase_timing_records.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=record_fields)
    writer.writeheader()
    for row in records:
        writer.writerow({key: row.get(key, "") for key in record_fields})

def number(row, key):
    value = row.get(key, "")
    if value in (None, "", "null", "None"):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None

def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return ("", "", "", "", "")
    return (len(values), statistics.fmean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
            statistics.median(values), min(values))

groups = {}
for row in records:
    key = (row["tile"], row["backend"], row["cores"])
    groups.setdefault(key, {"primary": [], "profile": [], "counters": [], "meta": row})
    groups[key].setdefault(row["role"], []).append(row)

summary_fields = ["tile", "backend", "cores", "rvv_kernel", "ime_kernel", "cpus",
                  "primary_n", "total_mean_sec", "total_sd_sec", "total_median_sec",
                  "profile_n", "packing_mean_sec", "packing_sd_sec", "packing_median_sec",
                  "kernel_mean_sec", "kernel_sd_sec", "kernel_median_sec",
                  "output_mean_sec", "output_sd_sec", "output_median_sec",
                  "boundary_mean_sec", "boundary_sd_sec", "boundary_median_sec",
                  "ime_output_mean_sec", "ime_output_sd_sec", "ime_output_median_sec",
                  "ime_boundary_mean_sec", "ime_boundary_sd_sec", "ime_boundary_median_sec",
                  "counter_n", "cycles_mean", "instructions_mean", "ipc_mean",
                  "cache_references_mean", "cache_misses_mean"]
with (result_root / "phase_timing_summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary_fields)
    writer.writeheader()
    for key in sorted(groups):
        group = groups[key]
        meta = group["meta"]
        row = {name: "" for name in summary_fields}
        row.update({"tile": key[0], "backend": key[1], "cores": key[2],
                    "rvv_kernel": meta["rvv_kernel"], "ime_kernel": meta["ime_kernel"],
                    "cpus": meta["cpus"]})
        total = stats([number(item, "total_sec") for item in group["primary"]])
        row["primary_n"], row["total_mean_sec"], row["total_sd_sec"], row["total_median_sec"] = total[:4]
        profile = group["profile"]
        row["profile_n"] = len(profile)
        for source, prefix in (("packing_sec", "packing"), ("kernel_sec", "kernel"),
                               ("output_sec", "output"), ("boundary_sec", "boundary"),
                               ("ime_output_sec", "ime_output"), ("ime_boundary_sec", "ime_boundary")):
            values = stats([number(item, source) for item in profile])
            row[f"{prefix}_mean_sec"], row[f"{prefix}_sd_sec"], row[f"{prefix}_median_sec"] = values[1:4]
        counter = group["counters"]
        row["counter_n"] = len(counter)
        for source in ("cycles", "instructions", "ipc", "cache_references", "cache_misses"):
            values = stats([number(item, source) for item in counter])
            row[f"{source}_mean"] = values[1]
        writer.writerow(row)

print(f"Accepted records: {len(records)}")
print(f"Summary: {result_root / 'phase_timing_summary.csv'}")
PY
}

main() {
    case "${1:-}" in
        --help|-h) usage; return 0 ;;
        --check)
            [[ $# == 1 ]] || die '--check does not accept additional arguments'
            check_contract
            return 0
            ;;
        '') ;;
        *) usage >&2; return 2 ;;
    esac

    check_contract
    require_command python3
    mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/runs"
    RUNS_DIR="${RESULT_ROOT}/runs"
    MANIFEST="${RESULT_ROOT}/manifest.tsv"
    : > "$MANIFEST"
    write_scope
    {
        printf 'tile\tbackend\tcores\trole\trvv_kernel\time_kernel\tcpus\treturn_code\tcampaign_dir\tlog_file\n'
    } > "$MANIFEST"
    {
        printf 'campaign_start_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'project_root\t%s\n' "$PROJECT_ROOT"
        printf 'M\t%s\nN\t%s\nK\t%s\nRUNS\t%s\nWARMUPS\t%s\nSEED\t%s\n' "$M" "$N" "$K" "$RUNS" "$WARMUPS" "$SEED"
    } > "${RESULT_ROOT}/configuration.tsv"

    local tile backend cores
    for tile in 8x4 8x8; do
        for cores in $RVV_CORE_COUNTS; do run_one "$tile" rvv "$cores" primary; done
        for cores in $IME_CORE_COUNTS; do run_one "$tile" ime "$cores" primary; done
        for cores in $RVV_CORE_COUNTS; do run_one "$tile" rvv "$cores" profile; done
        for cores in $IME_CORE_COUNTS; do run_one "$tile" ime "$cores" profile; done
        if [[ "$COLLECT_COUNTERS" == 1 ]]; then
            for cores in $RVV_CORE_COUNTS; do run_one "$tile" rvv "$cores" counters; done
            for cores in $IME_CORE_COUNTS; do run_one "$tile" ime "$cores" counters; done
        fi
    done
    aggregate_results
    printf 'Completed phase-timing campaign: %s\n' "$RESULT_ROOT"
}

main "$@"
