#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Set this figure's lighter counter default before the shared defaults load.
COLLECT_COUNTERS=${COLLECT_COUNTERS:-0}
source "$SCRIPT_DIR/../common.sh"
usage() {
  cat <<'HELP'
Usage: bash fig04_scheduling.sh [--help|--check]
Compares fixed-pair static/dynamic heterogeneous schedules for both tiles and
U1/U2/U4/U8, using Figure 5's per-U independent RVV/IME selections.
Primary: M=N=K=1024, 4 IME+4 RVV workers, weight=4, chunk=1, end_to_end.
Odd repetitions run static then dynamic; even repetitions reverse the order.
SENSITIVITY=1 (default) tests static weights 2/4/8 and dynamic chunks 1/2/4
only for the globally selected pair for each tile, not every configuration.
COLLECT_PROFILES=1 adds separate worker-summed phase/work diagnostics.
COLLECT_COUNTERS=1 adds separate hardware-counter diagnostics (default 0).
Independent scheduler-overhead timing is unavailable in existing C code;
the dataset states this explicitly, rather than inferring it from phases.
--check validates source contracts only and creates no files.
HELP
}
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check) check_common_contract; printf 'Figure 4: U1/2/4/8 paired schedules, selected-only sensitivity; contracts OK.\n'; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
SENSITIVITY=${SENSITIVITY:-1}
COLLECT_PROFILES=${COLLECT_PROFILES:-1}
COLLECT_COUNTERS=${COLLECT_COUNTERS:-0}
for setting in SENSITIVITY COLLECT_PROFILES COLLECT_COUNTERS; do
  [[ "${!setting}" =~ ^[01]$ ]] || { printf '%s must be 0 or 1.\n' "$setting" >&2; exit 2; }
done
init_figure fig04_scheduling
load_selection
load_u_selection
preflight_once
cat > "$OUT/measurement_scope.txt" <<'SCOPE'
This campaign uses the unchanged common C measurement engine, not the old
nested OpenMP static implementation. Static assignment uses contiguous
backend ranges with cyclic strips within each backend; dynamic assignment
uses the common atomic work queue. The same selected independent kernel pair
is used for both schedules at every paired U/tile comparison.
End-to-end includes input packing, computation, required output/boundary
handling, work assignment, and synchronization. Preparation is excluded.
One measured repetition per process makes alternating mode order possible;
each process uses the same explicit warm-up count and full validation.
Each raw record preserves worker CPU identities and completed strip counts;
total completed strips must equal ceil(N/32). Static assigned work follows
the fixed partition; dynamic assignments are the successfully claimed strips.
An independent scheduling-overhead field is unavailable: no subtraction of
worker-summed phases from wall time is valid. Phase aggregation is SUM of
worker durations, not an additive elapsed-time decomposition.
Sensitivity campaigns are separately named and are not pooled into the
U1/U2/U4/U8 primary scheduling comparison.
SCOPE

paired_schedule() {
  local id=$1 rvv=$2 ime=$3 weight=$4 chunk=$5 r mode
  for ((r=1; r<=RUNS; r++)); do
    if ((r % 2)); then modes=(static dynamic); else modes=(dynamic static); fi
    for mode in "${modes[@]}"; do
      if ! measure_case "${id}_${mode}" "$rvv" "$ime" mixed end_to_end 1024 1024 1024 8 4 "$ALL_CPUS" "$mode" "$weight" "$chunk" 0 0 "$r"; then :; fi
    done
  done
}

for pair in "${SELECTED_U_PAIRS[@]}"; do
  IFS='|' read -r tile u rvv ime <<< "$pair"
  snapshot_environment "$OUT/${tile}_U${u}_environment.txt"
  paired_schedule "primary_${tile}_U${u}" "$rvv" "$ime" 4 1
done

for pair in "${SELECTED_PAIRS[@]}"; do
  IFS='|' read -r tile rvv ime <<< "$pair"
  if [[ "$SENSITIVITY" == 1 ]]; then
    for weight in 2 4 8; do
      if ! measure_case "sensitivity_${tile}_static_w${weight}" "$rvv" "$ime" mixed end_to_end 1024 1024 1024 8 4 "$ALL_CPUS" static "$weight" 1 0 0; then :; fi
    done
    for chunk in 1 2 4; do
      if ! measure_case "sensitivity_${tile}_dynamic_c${chunk}" "$rvv" "$ime" mixed end_to_end 1024 1024 1024 8 4 "$ALL_CPUS" dynamic 4 "$chunk" 0 0; then :; fi
    done
  fi
  for mode in static dynamic; do
    if [[ "$COLLECT_PROFILES" == 1 ]]; then
      if ! measure_case "diagnostic_${tile}_${mode}_profile" "$rvv" "$ime" mixed end_to_end 1024 1024 1024 8 4 "$ALL_CPUS" "$mode" 4 1 1 0; then :; fi
    fi
    if [[ "$COLLECT_COUNTERS" == 1 ]]; then
      if ! measure_case "diagnostic_${tile}_${mode}_counter" "$rvv" "$ime" mixed end_to_end 1024 1024 1024 8 4 "$ALL_CPUS" "$mode" 4 1 0 1; then :; fi
    fi
  done
done

if [[ -s "$OUT/accepted.jsonl" ]]; then
  jq -s '
    map(select(.measurement_role=="primary" and (.case_id|startswith("primary_")))) |
    group_by([.rvv_kernel,.ime_kernel,.rvv_tile,.ime_tile,.rvv_unroll,.ime_unroll,.sample_index]) |
    map(. as $r | [$r[]|select(.schedule=="static")] as $s |
      [$r[]|select(.schedule=="dynamic")] as $d |
      if ($s|length)!=1 or ($d|length)!=1 then
        {rvv_kernel:$r[0].rvv_kernel,ime_kernel:$r[0].ime_kernel,
         sample_index:$r[0].sample_index,status:"INCOMPLETE_PAIR"}
      else {rvv_kernel:$r[0].rvv_kernel,ime_kernel:$r[0].ime_kernel,
        tile:$r[0].rvv_tile,rvv_unroll:$r[0].rvv_unroll,ime_unroll:$r[0].ime_unroll,
        sample_index:$r[0].sample_index,status:"OK",static_sec:$s[0].total_sec,
        dynamic_sec:$d[0].total_sec,static_over_dynamic:($s[0].total_sec/$d[0].total_sec),
        static_rvv_strips:([$s[0].workers[]|select(.id>=4)|.strips]|add),
        static_ime_strips:([$s[0].workers[]|select(.id<4)|.strips]|add),
        dynamic_rvv_strips:([$d[0].workers[]|select(.id>=4)|.strips]|add),
        dynamic_ime_strips:([$d[0].workers[]|select(.id<4)|.strips]|add),
        scheduling_overhead_sec:null,scheduling_overhead_status:"NOT_SEPARATELY_TIMED"} end)
  ' "$OUT/accepted.jsonl" > "$OUT/paired_scheduling.json"
  jq -r '(["tile","rvv_kernel","ime_kernel","rvv_unroll","ime_unroll","sample_index","status","static_sec","dynamic_sec","static_over_dynamic","static_rvv_strips","static_ime_strips","dynamic_rvv_strips","dynamic_ime_strips","scheduling_overhead_status"]|@csv),(.[]|[.tile,.rvv_kernel,.ime_kernel,.rvv_unroll,.ime_unroll,.sample_index,.status,.static_sec,.dynamic_sec,.static_over_dynamic,.static_rvv_strips,.static_ime_strips,.dynamic_rvv_strips,.dynamic_ime_strips,.scheduling_overhead_status]|@csv)' "$OUT/paired_scheduling.json" > "$OUT/paired_scheduling.csv"
  if jq -e 'any(.[]; .status!="OK")' "$OUT/paired_scheduling.json" >/dev/null; then
    record_failure paired_scheduling 'At least one static/dynamic repetition pair is incomplete; do not report it as a matched comparison.'
  fi
fi
finish_figure
