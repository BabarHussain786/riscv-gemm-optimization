#!/usr/bin/env bash
# New campaign only: existing sources and historical datasets are never edited.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"

usage() {
  cat <<'HELP'
Usage: bash fig02_strong_scaling.sh [--help|--check]
Fixed 1024^3 end-to-end INT8 scaling. RVV: 1/2/4/8 workers;
IME: 1/2/4 workers. Each tile uses the independently selected validated
kernels from this campaign's Figure 5; configurations do not change with p.
Primary measurements exclude profiling/counter instrumentation. Separate
counter passes are enabled by COLLECT_COUNTERS=1 (default). Unavailable
counter values remain null, never zero. RUNS=7, WARMUPS=2, SEED=42 defaults.
The scaling summary requires all 14 primary points with RUNS unique samples
each. IPC mean/SD and availability counts are joined from matched separate
counter records; counter-instrumented times never enter scaling statistics.
Set CAMPAIGN to reuse the Figure 5 selection. --check is read-only and does
not require selection files, a compiler, or a K1 board.
HELP
}
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check) check_common_contract; printf 'Figure 2: fixed 1024^3, RVV 1/2/4/8, IME 1/2/4; contracts OK.\n'; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
COLLECT_COUNTERS=${COLLECT_COUNTERS:-1}
[[ "$COLLECT_COUNTERS" =~ ^[01]$ ]] || { printf 'COLLECT_COUNTERS must be 0 or 1.\n' >&2; exit 2; }
init_figure fig02_strong_scaling
load_selection
preflight_once
cat > "$OUT/measurement_scope.txt" <<'SCOPE'
Primary: fixed M=N=K=1024, common-driver end_to_end, profile=0, counters=0.
Input packing, computation, applicable output/boundary handling, assignment,
and timed synchronization are included; preparation/validation are excluded.
Kernel pair is fixed at every core count within each tile/backend series.
Counter runs are separate diagnostics: IPC=sum instructions/sum cycles over
worker measured regions, not process-wide IPC or MAC efficiency. The worker
counter interval excludes final waiting; wall time includes final barrier.
Scaling-summary IPC statistics use only matched counter records, not primary
timing records; unavailable values remain null with explicit sample counts.
Keep all valid repetitions, including slow ones. Investigate unusual spread
using per-case environment snapshots and raw worker/counter records.
SCOPE

for pair in "${SELECTED_PAIRS[@]}"; do
  IFS='|' read -r tile rvv ime <<< "$pair"
  for impl in rvv ime; do
    if [[ "$impl" == rvv ]]; then counts=(1 2 4 8); available="$RVV_CPUS,$IME_CPUS"
    else counts=(1 2 4); available=$IME_CPUS; fi
    for p in "${counts[@]}"; do
      cpus=$(cpu_prefix "$available" "$p")
      iw=0; [[ "$impl" == ime ]] && iw=$p
      id="${tile}_${impl}_p${p}"
      snapshot_environment "$OUT/${id}_environment.txt"
      if ! measure_case "$id" "$rvv" "$ime" "$impl" end_to_end 1024 1024 1024 "$p" "$iw" "$cpus" static 4 1 0 0; then :; fi
      if [[ "$COLLECT_COUNTERS" == 1 ]]; then
        if ! measure_case "${id}_counters" "$rvv" "$ime" "$impl" end_to_end 1024 1024 1024 "$p" "$iw" "$cpus" static 4 1 0 1; then :; fi
      fi
    done
  done
done

if [[ -s "$OUT/accepted.jsonl" ]]; then
  if jq -s --argjson expected "$RUNS" --argjson counters "$COLLECT_COUNTERS" '
    def mean: add / length;
    def sd: if length > 1 then . as $x | ($x|mean) as $m |
      (map((.-$m)*(.-$m))|add)/(length-1)|sqrt else null end;
    def pointkey: [.implementation,.rvv_kernel,.ime_kernel,.rvv_tile,.ime_tile,.threads,.M,.N,.K];
    def contractkey: [.cpu_ids,.seed,.source_fingerprint,.binary_sha256,.warmups,.primary_scope];
    def ipc_stats($all;$point):
      [$all[]|select(.measurement_role=="counter" and .timing_mode=="end_to_end" and
        pointkey==($point|pointkey) and contractkey==($point|contractkey))] as $c |
      [$c[]|select(.counters_status=="OK" and (.ipc|type)=="number")|.ipc] as $v |
      {mean_ipc:(if ($v|length)>0 then ($v|mean) else null end),
       sd_ipc:(if ($v|length)>0 then ($v|sd) else null end),
       valid_counter_samples:($v|length),counter_samples:($c|length),
       counters_status:(if $counters==0 then "DISABLED"
         elif ($v|length)==$expected and ($c|length)==$expected then "OK"
         else "UNAVAILABLE_OR_MULTIPLEXED" end),
       ipc_scope:"separate_counter_pass_sum_worker_instructions_over_sum_worker_cycles"};
    . as $all |
    map(select(.measurement_role=="primary" and .timing_mode=="end_to_end")) as $primary |
    ([$primary[]]|group_by(pointkey)) as $points |
    ([ ["8x4","8x8"][] as $tile | ["rvv","ime"][] as $impl |
      (if $impl=="rvv" then [1,2,4,8] else [1,2,4] end)[] as $p |
      [$tile,$impl,$p] ]|sort) as $required |
    if ($points|length)!=14 or
      ($points|map([.[0].rvv_tile,.[0].implementation,.[0].threads])|sort)!=$required or
      any($points[]; length!=$expected or (map(.sample_index)|sort)!=[range(1;$expected+1)] or
        (map(contractkey)|unique|length)!=1) or
      any($primary[]; .M!=1024 or .N!=1024 or .K!=1024 or .profiled!=false or .counters_status!="DISABLED") or
      any($primary|group_by(.rvv_tile)[]; (map([.rvv_kernel,.ime_kernel,.ime_tile])|unique|length)!=1)
    then error("Need all 14 fixed-kernel primary scaling points with RUNS matched unique repetitions") else
    $primary |
    group_by([.implementation,.rvv_kernel,.ime_kernel,.rvv_tile,.ime_tile,.M,.N,.K]) |
    map(. as $series | [$series[]|select(.threads==1)|.total_sec] as $b |
      if ($b|length)==0 then error("Missing single-core primary baseline") else
      ($b|mean) as $t1 |
      ($series|group_by(.threads)|map(. as $r | ($r|map(.total_sec)|mean) as $t |
        {implementation:$r[0].implementation,tile:$r[0].rvv_tile,
         rvv_kernel:$r[0].rvv_kernel,ime_kernel:$r[0].ime_kernel,
         cores:$r[0].threads,M:$r[0].M,N:$r[0].N,K:$r[0].K,
         samples:($r|length),mean_sec:$t,sd_sec:($r|map(.total_sec)|sd),
         baseline_mean_sec:$t1,speedup:($t1/$t),
         parallel_efficiency:($t1/($t*$r[0].threads)),
         operations:(2*$r[0].M*$r[0].N*$r[0].K)} + ipc_stats($all;$r[0]))) end) | flatten end
  ' "$OUT/accepted.jsonl" > "$OUT/scaling_summary.json"; then
    jq -r '(["implementation","tile","rvv_kernel","ime_kernel","cores","M","N","K","samples","mean_sec","sd_sec","baseline_mean_sec","speedup","parallel_efficiency","operations","mean_ipc","sd_ipc","valid_counter_samples","counter_samples","counters_status","ipc_scope"]|@csv),(.[]|[.implementation,.tile,.rvv_kernel,.ime_kernel,.cores,.M,.N,.K,.samples,.mean_sec,.sd_sec,.baseline_mean_sec,.speedup,.parallel_efficiency,.operations,.mean_ipc,.sd_ipc,.valid_counter_samples,.counter_samples,.counters_status,.ipc_scope]|@csv)' "$OUT/scaling_summary.json" > "$OUT/scaling_summary.csv"
  else
    record_failure scaling_summary 'Incomplete/invalid primary coverage: require all 14 scaling points, RUNS unique matched samples, fixed kernels, and single-core baselines.'
  fi
fi
finish_figure
