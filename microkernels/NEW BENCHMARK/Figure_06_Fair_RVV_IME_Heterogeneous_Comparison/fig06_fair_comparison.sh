#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"
usage() {
  cat <<'HELP'
Usage: bash fig06_fair_comparison.sh [--help|--check]
Figure 5 independent selections, same-tile controls plus a distinct global
best-tile pair when needed, four execution modes:
RVV8, IME4, heterogeneous static4+4, heterogeneous dynamic4+4.
Main 1024^3, plus 512^3 and rectangular 512x1024x768 (EXTRA_WORKLOADS=1).
STATIC_WEIGHT=4, DYNAMIC_CHUNK=1; shared kernel pair and inputs for every mode.
Primary modes rotate order each repetition; diagnostics run separately.
COLLECT_PROFILES=1, COLLECT_COUNTERS=1, PREPACKED_DIAGNOSTICS=1 by default.
--check reads existing source interfaces only; no build/run/files.
HELP
}
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check) check_common_contract; printf 'Four modes share one compiled pair per tile and timing boundary.\n'; exit 0 ;;
  '') [[ $# == 0 ]] || { usage >&2; exit 2; } ;;
  *) usage >&2; exit 2 ;;
esac
STATIC_WEIGHT=${STATIC_WEIGHT:-4}; DYNAMIC_CHUNK=${DYNAMIC_CHUNK:-1}
EXTRA_WORKLOADS=${EXTRA_WORKLOADS:-1}; PREPACKED_DIAGNOSTICS=${PREPACKED_DIAGNOSTICS:-1}
positive_integer STATIC_WEIGHT "$STATIC_WEIGHT"; positive_integer DYNAMIC_CHUNK "$DYNAMIC_CHUNK"
[[ $EXTRA_WORKLOADS =~ ^[01]$ && $PREPACKED_DIAGNOSTICS =~ ^[01]$ ]] || exit 2
init_figure fig06_fair_comparison
load_selection; load_global_selection; preflight_once
# Preserve same-tile controls, then add a distinct global pair if its best tiles differ.
global_present=0
for pair in "${SELECTED_PAIRS[@]}"; do
  IFS='|' read -r tile rvv ime <<< "$pair"
  [[ $rvv != "$GLOBAL_RVV_KERNEL" || $ime != "$GLOBAL_IME_KERNEL" ]] || global_present=1
done
[[ $global_present == 1 ]] || SELECTED_PAIRS+=("independent|$GLOBAL_RVV_KERNEL|$GLOBAL_IME_KERNEL")
cat > "$OUT/measurement_scope.txt" <<'SCOPE'
One common native benchmark binary per independently selected RVV/IME pair.
All primary modes use end_to_end, profile0/counters0, identical canonical input
matrices, alpha1, nonzero initial C, repetitions and explicit CPU maps.
Packing is measured as performed per assigned 32-column strip, including
repeated A packing where required by the existing adapters. Workspace allocation,
team creation/affinity, reference and validation are outside the measured region.
Assignment, input packing, computation, required output/tails and barriers are inside.
IME output scatter is separately instrumentable; RVV C updates remain fused.
Profiling and counters are separate explanatory passes, never headline timings.
Worker phases are SUM_WORKER_ELAPSED, NOT additive wall-time components.
Static backend partition/cyclic workers and dynamic atomic queue are the existing
common driver policies, not the older nested-cluster/OpenMP dynamic implementation.
Valid slow samples and losses relative to either baseline are preserved.
SCOPE
mode_args() {
  case "$1" in
    rvv8) impl=rvv; threads=8; iw=0; cpus="$RVV_CPUS,$IME_CPUS"; sched=static ;;
    ime4) impl=ime; threads=4; iw=4; cpus="$IME_CPUS"; sched=static ;;
    static|dynamic) impl=mixed; threads=8; iw=4; cpus="$ALL_CPUS"; sched=$1 ;;
  esac
}
shapes=(1024x1024x1024); [[ $EXTRA_WORKLOADS == 0 ]] || shapes+=(512x512x512 512x1024x768)
for pair in "${SELECTED_PAIRS[@]}"; do
  IFS='|' read -r tile rvv ime <<< "$pair"
  for shape in "${shapes[@]}"; do
    IFS=x read -r m n k <<< "$shape"
    for ((r=1;r<=RUNS;r++)); do
      # Rotate complete four-mode blocks; static/dynamic order alternates.
      if ((r%2)); then modes=(rvv8 ime4 static dynamic); else modes=(dynamic static ime4 rvv8); fi
      for mode in "${modes[@]}"; do
        mode_args "$mode"
        if ! measure_case "main_${tile}_${shape}_${mode}" "$rvv" "$ime" "$impl" end_to_end "$m" "$n" "$k" "$threads" "$iw" "$cpus" "$sched" "$STATIC_WEIGHT" "$DYNAMIC_CHUNK" 0 0 "$r"; then :; fi
      done
    done
  done
  for mode in rvv8 ime4 static dynamic; do
    mode_args "$mode"
    if [[ $PREPACKED_DIAGNOSTICS == 1 ]]; then
      if ! measure_case "diagnostic_${tile}_${mode}_prepacked" "$rvv" "$ime" "$impl" prepacked 1024 1024 1024 "$threads" "$iw" "$cpus" "$sched" "$STATIC_WEIGHT" "$DYNAMIC_CHUNK" 0 0; then :; fi
    fi
    if [[ $COLLECT_PROFILES == 1 ]]; then
      if ! measure_case "diagnostic_${tile}_${mode}_profile" "$rvv" "$ime" "$impl" end_to_end 1024 1024 1024 "$threads" "$iw" "$cpus" "$sched" "$STATIC_WEIGHT" "$DYNAMIC_CHUNK" 1 0; then :; fi
    fi
    if [[ $COLLECT_COUNTERS == 1 ]]; then
      if ! measure_case "diagnostic_${tile}_${mode}_counter" "$rvv" "$ime" "$impl" end_to_end 1024 1024 1024 "$threads" "$iw" "$cpus" "$sched" "$STATIC_WEIGHT" "$DYNAMIC_CHUNK" 0 1; then :; fi
    fi
  done
  # Single-thread phase diagnostics are independently interpretable; no mixed
  # worker-sum is turned into a wall-clock decomposition.
  if [[ $COLLECT_PROFILES == 1 ]]; then
    if ! measure_case "diagnostic_${tile}_rvv1_profile" "$rvv" "$ime" rvv end_to_end 1024 1024 1024 1 0 "$(cpu_first "$RVV_CPUS")" static "$STATIC_WEIGHT" 1 1 0; then :; fi
    if ! measure_case "diagnostic_${tile}_ime1_profile" "$rvv" "$ime" ime end_to_end 1024 1024 1024 1 1 "$(cpu_first "$IME_CPUS")" static "$STATIC_WEIGHT" 1 1 0; then :; fi
  fi
done
jq -c --arg rvv "$GLOBAL_RVV_KERNEL" --arg ime "$GLOBAL_IME_KERNEL" '. + {is_global_selected_pair:(.rvv_kernel==$rvv and .ime_kernel==$ime)}' "$OUT/accepted.jsonl" > "$OUT/annotated.jsonl"
mv -- "$OUT/annotated.jsonl" "$OUT/accepted.jsonl"
summarize_common "$OUT/accepted.jsonl" "$OUT/comparison_all_scopes"
jq '[.[]|select(.measurement_role=="primary" and .timing_mode=="end_to_end" and (.case_id|startswith("main_")))] |
  group_by([.rvv_kernel,.ime_kernel,.M,.N,.K]) | map(. as $a |
    ([$a[]|select(.implementation=="rvv" and .threads==8)][0]) as $r |
    ([$a[]|select(.implementation=="ime" and .threads==4)][0]) as $i |
    $a | map(. + {speedup_vs_rvv8:(if $r==null then null else $r.time.mean/.time.mean end),
                  speedup_vs_ime4:(if $i==null then null else $i.time.mean/.time.mean end)})) | flatten' "$OUT/comparison_all_scopes.json" > "$OUT/fair_comparison.json"
jq -r '(["rvv_tile","ime_tile","global_selected_pair","M","N","K","mode","threads","rvv_kernel","ime_kernel","n","mean_sec","sd_sec","mean_gops","speedup_vs_rvv8","speedup_vs_ime4"]|@csv),(.[]|[.rvv_tile,.ime_tile,.is_global_selected_pair,.M,.N,.K,(if .implementation=="mixed" then .schedule else .implementation end),.threads,.rvv_kernel,.ime_kernel,.time.n,.time.mean,.time.sd,.gops.mean,.speedup_vs_rvv8,.speedup_vs_ime4]|@csv)' "$OUT/fair_comparison.json" > "$OUT/fair_comparison.csv"
if ! jq -e --argjson runs "$RUNS" --argjson expected "$((${#SELECTED_PAIRS[@]}*4*${#shapes[@]}))" 'length==$expected and all(.[];.time.n==$runs and .speedup_vs_rvv8!=null and .speedup_vs_ime4!=null)' "$OUT/fair_comparison.json" >/dev/null; then
  record_failure fair_comparison_coverage 'Incomplete matched groups/baselines; do not claim a complete four-mode comparison.' INCOMPLETE_COVERAGE
fi
finish_figure
