#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"
usage() {
  cat <<'HELP'
Usage: bash fig03_weak_scaling.sh [--help|--check]
Preserved cube-growth experiment: (p,size)=(1,512),(2,672),(4,832),(8,1024).
RVV uses all four points; IME stops at four cores. Each backend retains one
selected tile/LMUL/unroll configuration. Primary end_to_end times are not
profiled. COLLECT_COUNTERS=1 adds separate counter diagnostics.
Writes actual operations/core and rounding-adjusted ideal time:
Tideal,p=T1*(Mp*Np*Kp)/(p*M1*N1*K1), rather than assuming constant work/core.
Requires all 14 primary points with RUNS unique samples; joins mean/SD IPC
only from separately measured, metadata-matched counter records. Counter
availability is explicit and unavailable values stay null, not zero.
--check is read-only and requires neither a board nor a previous campaign.
HELP
}
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check) check_common_contract; printf 'Figure 3: explicit 512/672/832/1024 cubes, fixed selected kernels; contracts OK.\n'; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
COLLECT_COUNTERS=${COLLECT_COUNTERS:-1}
[[ "$COLLECT_COUNTERS" =~ ^[01]$ ]] || { printf 'COLLECT_COUNTERS must be 0 or 1.\n' >&2; exit 2; }
init_figure fig03_weak_scaling
load_selection
preflight_once
cat > "$OUT/measurement_scope.txt" <<'SCOPE'
Preserved rounded cubes with intended 32-element dimension alignment.
p=1/2/4/8 corresponds to dimension=512/672/832/1024; IME stops at p=4.
Kernel tile/LMUL/U remain fixed within each backend series. End-to-end timing
includes input packing, computation, applicable output/boundary handling,
work assignment, and synchronization. Primary times are uninstrumented.
Work/core is not exactly constant because of rounded cube dimensions.
Ideal times are adjusted using actual dimensions and the p=1 mean time.
Separate counter diagnostics must not replace primary timing observations.
IPC mean/SD and availability counts are joined only for identical kernels,
dimensions, workers, CPUs, seed, source/binary hashes, and timing contract.
SCOPE
points=('1|512' '2|672' '4|832' '8|1024')
for pair in "${SELECTED_PAIRS[@]}"; do
  IFS='|' read -r tile rvv ime <<< "$pair"
  for impl in rvv ime; do
    for point in "${points[@]}"; do
      IFS='|' read -r p size <<< "$point"
      if [[ "$impl" == ime ]]; then
        [[ "$p" -le 4 ]] || continue
        iw=$p; cpus=$(cpu_prefix "$IME_CPUS" "$p")
      else iw=0; cpus=$(cpu_prefix "$RVV_CPUS,$IME_CPUS" "$p"); fi
      id="${tile}_${impl}_p${p}_n${size}"
      snapshot_environment "$OUT/${id}_environment.txt"
      if ! measure_case "$id" "$rvv" "$ime" "$impl" end_to_end "$size" "$size" "$size" "$p" "$iw" "$cpus" static 4 1 0 0; then :; fi
      if [[ "$COLLECT_COUNTERS" == 1 ]]; then
        if ! measure_case "${id}_counters" "$rvv" "$ime" "$impl" end_to_end "$size" "$size" "$size" "$p" "$iw" "$cpus" static 4 1 0 1; then :; fi
      fi
    done
  done
done
if [[ -s "$OUT/accepted.jsonl" ]]; then
  if jq -s --argjson expected "$RUNS" --argjson counters "$COLLECT_COUNTERS" '
    def mean: add/length;
    def sd: if length>1 then . as $x | ($x|mean) as $m |
      (map((.-$m)*(.-$m))|add)/(length-1)|sqrt else null end;
    def pointkey: [.implementation,.rvv_kernel,.ime_kernel,.rvv_tile,.ime_tile,.threads,.M,.N,.K];
    def contractkey: [.cpu_ids,.seed,.source_fingerprint,.binary_sha256,.warmups,.primary_scope];
    def size_for_p: if .==1 then 512 elif .==2 then 672 elif .==4 then 832 elif .==8 then 1024 else 0 end;
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
    ($primary|group_by(pointkey)) as $points |
    ([ ["8x4","8x8"][] as $tile | ["rvv","ime"][] as $impl |
      (if $impl=="rvv" then [1,2,4,8] else [1,2,4] end)[] as $p |
      [$tile,$impl,$p] ]|sort) as $required |
    if ($points|length)!=14 or
      ($points|map([.[0].rvv_tile,.[0].implementation,.[0].threads])|sort)!=$required or
      any($points[]; length!=$expected or (map(.sample_index)|sort)!=[range(1;$expected+1)] or
        (map(contractkey)|unique|length)!=1) or
      any($primary[]; .M!=(.threads|size_for_p) or .N!=.M or .K!=.M or .profiled!=false or .counters_status!="DISABLED") or
      any($primary|group_by(.rvv_tile)[]; (map([.rvv_kernel,.ime_kernel,.ime_tile])|unique|length)!=1)
    then error("Need all 14 fixed-kernel primary weak-scaling points with RUNS matched unique repetitions") else
    $primary |
    group_by([.implementation,.rvv_kernel,.ime_kernel,.rvv_tile,.ime_tile]) |
    map(. as $series | [$series[]|select(.threads==1)] as $base |
      if ($base|length)==0 then error("Missing weak-scaling p=1 baseline") else
      ($base|map(.total_sec)|mean) as $t1 |
      ($base[0].M*$base[0].N*$base[0].K) as $w1 |
      ($series|group_by([.threads,.M,.N,.K])|map(. as $r |
        ($r|map(.total_sec)|mean) as $t |
        ($r[0].M*$r[0].N*$r[0].K) as $w |
        ($t1*$w/($r[0].threads*$w1)) as $ideal |
        {implementation:$r[0].implementation,tile:$r[0].rvv_tile,
         rvv_kernel:$r[0].rvv_kernel,ime_kernel:$r[0].ime_kernel,
         cores:$r[0].threads,M:$r[0].M,N:$r[0].N,K:$r[0].K,
         samples:($r|length),mean_sec:$t,sd_sec:($r|map(.total_sec)|sd),
         operations:(2*$w),operations_per_core:(2*$w/$r[0].threads),
         work_per_core_ratio:($w/($r[0].threads*$w1)),
         baseline_mean_sec:$t1,ideal_sec:$ideal,
         adjusted_weak_efficiency:($ideal/$t),
         deviation_from_ideal_percent:(100*($t/$ideal-1))} + ipc_stats($all;$r[0]))) end) | flatten end
  ' "$OUT/accepted.jsonl" > "$OUT/scaling_summary.json"; then
    jq -r '(["implementation","tile","rvv_kernel","ime_kernel","cores","M","N","K","samples","mean_sec","sd_sec","operations","operations_per_core","work_per_core_ratio","baseline_mean_sec","ideal_sec","adjusted_weak_efficiency","deviation_from_ideal_percent","mean_ipc","sd_ipc","valid_counter_samples","counter_samples","counters_status","ipc_scope"]|@csv),(.[]|[.implementation,.tile,.rvv_kernel,.ime_kernel,.cores,.M,.N,.K,.samples,.mean_sec,.sd_sec,.operations,.operations_per_core,.work_per_core_ratio,.baseline_mean_sec,.ideal_sec,.adjusted_weak_efficiency,.deviation_from_ideal_percent,.mean_ipc,.sd_ipc,.valid_counter_samples,.counter_samples,.counters_status,.ipc_scope]|@csv)' "$OUT/scaling_summary.json" > "$OUT/scaling_summary.csv"
  else
    record_failure scaling_summary 'Incomplete/invalid primary coverage: require all 14 correct-size scaling points, RUNS unique matched samples, fixed kernels, and p=1 baselines.'
  fi
fi
finish_figure
