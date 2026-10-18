#!/usr/bin/env bash
# Figure 6 phase-aware campaign.
#
# This is additive: it writes a new campaign below NEW BENCHMARK/results and
# never overwrites the existing Figure 6 dataset or paper_results files.
# The primary matrix exactly matches the paper Figure 6 comparison:
#   2 tiles x 4 unroll factors x 3 headline modes x 7 repetitions.
# Profiles and counters are collected separately from headline timings.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/../../../benchmarking/NEW BENCHMARK/common.sh"

usage() {
  cat <<'HELP'
Usage: bash run_k1_fig06_phase_complete.sh [--help|--check]

Runs the paper Figure 6 matrix on a native SpacemiT K1:
  tiles       8x4 and 8x8
  unroll      U1, U2, U4, U8
  primary     RVV8, static 4-RVV+4-IME, dynamic 4-RVV+4-IME
  repetitions seven by default, with two warm-ups

The primary dataset is 2 x 4 x 3 x 7 = 168 accepted run records when all
cases pass. Separate profile and counter passes are written for RVV8, IME4,
static, and dynamic execution. No Figure 6 source files are modified.

The script intentionally skips the small native preflight that previously
raised SIGILL on IME. Every measured run is still required to return status
OK and validation PASS by the common driver checks.

Phase scope: input packing is recorded when exposed by the driver; IME output
scatter and boundary fields are recorded when exposed; RVV output stores remain
fused with the RVV kernel, and synchronization is not separately timed.
HELP
}

case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check)
    check_common_contract
    printf '%s\n' 'FIG06_CHECK: fixed paper matrix, no selection file, no native run.'
    printf '%s\n' 'FIG06_SCOPE: 168 primary records expected at RUNS=7.'
    exit 0
    ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac

RUNS=${RUNS:-7}
WARMUPS=${WARMUPS:-2}
COLLECT_PROFILES=${COLLECT_PROFILES:-1}
COLLECT_COUNTERS=${COLLECT_COUNTERS:-1}
STATIC_WEIGHT=${STATIC_WEIGHT:-4}
DYNAMIC_CHUNK=${DYNAMIC_CHUNK:-1}

positive_integer STATIC_WEIGHT "$STATIC_WEIGHT"
positive_integer DYNAMIC_CHUNK "$DYNAMIC_CHUNK"
[[ $COLLECT_PROFILES =~ ^[01]$ && $COLLECT_COUNTERS =~ ^[01]$ ]] || {
  die 'COLLECT_PROFILES and COLLECT_COUNTERS must be 0 or 1'
}

init_figure fig06_phase_complete

cat > "$OUT/measurement_scope.txt" <<'SCOPE'
Primary Figure 6 scope: fixed 1024x1024x1024 INT8 GEMM, two output tiles
(8x4 and 8x8), four unroll factors (U1/U2/U4/U8), and three headline modes:
RVV8, static 4-RVV+4-IME, and dynamic 4-RVV+4-IME. Each primary group uses
the common end_to_end timing scope, two warm-ups, and seven measured runs by
default. A separate IME4 profile/counter pass is diagnostic only and is not a
headline Figure 6 bar.

The driver records total wall time and, when available, worker phase fields.
Input packing is measured as performed by each assigned strip. IME output
scatter and IME boundary fields are retained when emitted by the native driver.
RVV output stores remain fused with the RVV kernel. Synchronization is not a
separate timer. Worker phase values are summed worker elapsed times and are not
additive wall-clock components.
SCOPE

mode_args() {
  case "$1" in
    rvv8)
      impl=rvv; threads=8; iw=0; cpus="$RVV_CPUS,$IME_CPUS"; sched=static ;;
    ime4)
      impl=ime; threads=4; iw=4; cpus="$IME_CPUS"; sched=static ;;
    static|dynamic)
      impl=mixed; threads=8; iw=4; cpus="$ALL_CPUS"; sched="$1" ;;
    *) die "Unknown Figure 6 mode: $1"; return 1 ;;
  esac
}

run_measure() {
  if ! measure_case "$@"; then
    printf 'WARNING: %s failed; its raw diagnostics remain under %s/raw\n' "$1" "$OUT" >&2
  fi
}

PRIMARY_MODES=(rvv8 static dynamic)
DIAGNOSTIC_MODES=(rvv8 ime4 static dynamic)
TILES=(8x4 8x8)
UNROLLS=(1 2 4 8)

for tile in "${TILES[@]}"; do
  for unroll in "${UNROLLS[@]}"; do
    rvv="igemm_kernel_${tile}_zvl256b_lmul1_unroll${unroll}"
    ime="ime_kernel_${tile}_zvl256b_lmul1_unroll${unroll}"
    resolve_rvv "$rvv"
    resolve_ime "$ime"

    for mode in "${PRIMARY_MODES[@]}"; do
      mode_args "$mode"
      run_measure "main_${tile}_u${unroll}_${mode}" "$rvv" "$ime" \
        "$impl" end_to_end 1024 1024 1024 "$threads" "$iw" "$cpus" \
        "$sched" "$STATIC_WEIGHT" "$DYNAMIC_CHUNK" 0 0
    done

    for mode in "${DIAGNOSTIC_MODES[@]}"; do
      mode_args "$mode"
      if (( COLLECT_PROFILES )); then
        run_measure "profile_${tile}_u${unroll}_${mode}" "$rvv" "$ime" \
          "$impl" end_to_end 1024 1024 1024 "$threads" "$iw" "$cpus" \
          "$sched" "$STATIC_WEIGHT" "$DYNAMIC_CHUNK" 1 0
      fi
      if (( COLLECT_COUNTERS )); then
        run_measure "counter_${tile}_u${unroll}_${mode}" "$rvv" "$ime" \
          "$impl" end_to_end 1024 1024 1024 "$threads" "$iw" "$cpus" \
          "$sched" "$STATIC_WEIGHT" "$DYNAMIC_CHUNK" 0 1
      fi
    done
  done
done

summarize_common "$OUT/accepted.jsonl" "$OUT/summary"

# Produce a paper-facing primary table with matched RVV speedups. The raw
# accepted.jsonl and summary.json remain the authoritative audit records.
jq '
  def base_key: [.rvv_tile,.rvv_unroll,.M,.N,.K];
  map(select(.measurement_role=="primary" and (.case_id|startswith("main_")))) as $rows |
  $rows | map(. as $x |
    ($rows | map(select((base_key)==($x|base_key) and .implementation=="rvv" and .threads==8)) | .[0]) as $r |
    . + {speedup_vs_rvv8:(if $r==null then null else $r.time.mean/.time.mean end)})
' "$OUT/summary.json" > "$OUT/primary_comparison.json"
jq -r '
  (["tile_shape","unroll","mode","threads","n","mean_sec","median_sec","sd_sec","mean_gops","speedup_vs_rvv8"]|@csv),
  (.[] | [ .rvv_tile,.rvv_unroll,
    (if .implementation=="mixed" then .schedule else .implementation end),
    .threads,.time.n,.time.mean,.time.median,.time.sd,.gops.mean,.speedup_vs_rvv8 ] | @csv)
' "$OUT/primary_comparison.json" > "$OUT/primary_comparison.csv"

# Export phase statistics separately so missing fields are visible as blank,
# rather than silently interpreted as zero.
jq -r '
  (["tile_shape","unroll","mode","threads","n","total_mean_sec","packing_mean_sec","packing_median_sec","packing_sd_sec","kernel_mean_sec","kernel_median_sec","kernel_sd_sec","ime_output_mean_sec","ime_boundary_mean_sec","rvv_output_scope","synchronization_scope","phase_aggregation"]|@csv),
  (.[] | select(.measurement_role=="profile") | [
    .rvv_tile,.rvv_unroll,
    (if .implementation=="mixed" then .schedule else .implementation end),
    .threads,.time.n,.time.mean,
    (.packing.mean // ""),(.packing.median // ""),(.packing.sd // ""),
    (.kernel.mean // ""),(.kernel.median // ""),(.kernel.sd // ""),
    (.ime_output.mean // ""),(.ime_boundary.mean // ""),
    (.rvv_output_time // "fused_not_separately_measured"),
    (.scheduling_overhead // "not_separately_measured"),
    (.phase_aggregation // "sum_worker_elapsed_NOT_wall_clock_decomposition")
  ] | @csv)
' "$OUT/summary.json" > "$OUT/phase_profile_summary.csv"

expected_primary=$(( ${#TILES[@]} * ${#UNROLLS[@]} * ${#PRIMARY_MODES[@]} * RUNS ))
expected_groups=$(( ${#TILES[@]} * ${#UNROLLS[@]} * ${#PRIMARY_MODES[@]} ))
jq -s --argjson expected_records "$expected_primary" --argjson expected_groups "$expected_groups" '
  map(select(.measurement_role=="primary" and (.case_id|startswith("main_")))) as $p |
  map(select(.measurement_role=="profile")) as $q |
  {
    expected_primary_records:$expected_records,
    observed_primary_records:($p|length),
    expected_primary_groups:$expected_groups,
    observed_primary_groups:($p|map(.case_id)|unique|length),
    profile_records:($q|length),
    phase_field_nonnull:{
      packing:($q|map(select(.packing_sec!=null))|length),
      kernel:($q|map(select(.kernel_sec!=null))|length),
      ime_output:($q|map(select(.ime_output_sec!=null))|length),
      ime_boundary:($q|map(select(.ime_boundary_sec!=null))|length)
    },
    rvv_output_scope:"fused_not_separately_measured",
    synchronization_scope:"not_separately_measured"
  }
' "$OUT/accepted.jsonl" > "$OUT/coverage.json"

actual_primary=$(jq -r '.observed_primary_records' "$OUT/coverage.json")
actual_groups=$(jq -r '.observed_primary_groups' "$OUT/coverage.json")
if (( actual_primary != expected_primary || actual_groups != expected_groups )); then
  record_failure coverage "Expected ${expected_primary} primary records in ${expected_groups} groups; observed ${actual_primary} records in ${actual_groups} groups" INCOMPLETE_COVERAGE
fi

finish_figure
