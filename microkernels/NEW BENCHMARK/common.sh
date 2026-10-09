#!/usr/bin/env bash
# Additive K1 benchmark helpers. Existing sources and data are read-only.
set -Eeuo pipefail
NB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$NB_ROOT/.." && pwd -P)
CAMPAIGN=${CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
RUNS=${RUNS:-7}; WARMUPS=${WARMUPS:-2}; SEED=${SEED:-42}; CC=${CC:-gcc}
IME_CPUS=${IME_CPUS:-0,1,2,3}; RVV_CPUS=${RVV_CPUS:-4,5,6,7}
ALL_CPUS="$IME_CPUS,$RVV_CPUS"; FP_CPU=${FP_CPU:-${RVV_CPUS%%,*}}
COLLECT_COUNTERS=${COLLECT_COUNTERS:-1}; COLLECT_PROFILES=${COLLECT_PROFILES:-1}
CFLAGS_ARRAY=(-O3 -std=c11 -Wall -Wextra -Wno-unused-function -Wno-unknown-pragmas
  -march=rv64gcv_zvl256b -mabi=lp64d -fopenmp -fno-pie)
OUT=; FIGURE_ID=; FAILURES=0; BENCH=; SOURCE_FINGERPRINT=; BINARY_SHA256=

die() { printf 'ERROR: %s\n' "$*" >&2; return 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }
positive_integer() { [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer without leading zeros"; }
cpu_first() { printf '%s\n' "${1%%,*}"; }
cpu_prefix() {
  local -a cpus; IFS=, read -r -a cpus <<< "$1"
  (( ${#cpus[@]} >= $2 )) || { die "Insufficient CPUs in $1"; return 1; }
  local i; for ((i=0;i<$2;i++)); do ((i==0)) || printf ','; printf '%s' "${cpus[i]}"; done
  printf '\n'
}
validate_settings() {
  [[ $CAMPAIGN =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 'Unsafe CAMPAIGN name'
  positive_integer RUNS "$RUNS"; positive_integer SEED "$SEED"
  (( ${#RUNS}<=10 && ${#SEED}<=10 && 10#$RUNS<=2147483647 )) || die 'RUNS/SEED out of supported range'
  ((10#$SEED <= 4294967295)) || die 'SEED exceeds uint32 range'
  [[ $WARMUPS =~ ^(0|[1-9][0-9]*)$ ]] || die 'WARMUPS must be nonnegative without leading zeros'
  (( ${#WARMUPS}<=10 && WARMUPS<=2147483647 )) || die 'WARMUPS exceeds C int range'
  [[ $COLLECT_COUNTERS =~ ^[01]$ && $COLLECT_PROFILES =~ ^[01]$ ]] || die 'Diagnostic switches must be 0 or 1'
  [[ $IME_CPUS =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ && $RVV_CPUS =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || die 'Use four IME_CPUS and four RVV_CPUS'
  [[ $FP_CPU =~ ^[0-9]+$ ]] || die 'FP_CPU must be one CPU number'
  local -a cpus; local cpu; local -A seen=(); IFS=, read -r -a cpus <<< "$ALL_CPUS"
  for cpu in "${cpus[@]}"; do
    [[ $cpu =~ ^(0|[1-9][0-9]*)$ && ${#cpu} -le 4 ]] && ((cpu<1024)) || die 'CPU IDs must be canonical integers below 1024'
    [[ ! ${seen[$cpu]+yes} ]] || die 'CPU lists overlap or contain duplicates'
    seen[$cpu]=1
  done
}
inventory_rvv() {
  local tile lm u; local -a lms
  for tile in 8x4 8x8; do
    if [[ $tile == 8x4 ]]; then lms=(mf8 mf4 mf2 1 2); else lms=(mf4 mf2 1 2); fi
    for lm in "${lms[@]}"; do for u in 1 2 4 8; do printf 'igemm_kernel_%s_zvl256b_lmul%s_unroll%s\n' "$tile" "$lm" "$u"; done; done
  done
}
inventory_ime() {
  local tile u; for tile in 8x4 8x8; do for u in 1 2 4 8; do
    printf 'ime_kernel_%s_zvl256b_lmul1_unroll%s\n' "$tile" "$u"
  done; done
}
resolve_rvv() {
  [[ $1 =~ ^igemm_kernel_(8x[48])_zvl256b_lmul(mf8|mf4|mf2|1|2)_unroll(1|2|4|8)$ ]] || { die "Invalid RVV selector $1"; return 1; }
  local tile=${BASH_REMATCH[1]} lm=${BASH_REMATCH[2]}
  [[ $tile != 8x8 || $lm != mf8 ]] || { die 'Unsupported RVV 8x8 mf8'; return 1; }
  RESOLVED_SOURCE="$PROJECT_ROOT/GEMM_RVV_FP32_INT8_${tile}_Baseline/RVV_IGEMM_INT8_I8I32_${tile}/$1/${1}_i8i32.c"
  [[ -f $RESOLVED_SOURCE ]] || { die "Missing $RESOLVED_SOURCE"; return 1; }
}
resolve_ime() {
  [[ $1 =~ ^ime_kernel_(8x[48])_zvl256b_lmul1_unroll(1|2|4|8)$ ]] || { die "Invalid primary IME selector $1"; return 1; }
  RESOLVED_SOURCE="$PROJECT_ROOT/IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_${BASH_REMATCH[1]}_NATIVE/$1/$1.c"
  [[ -f $RESOLVED_SOURCE && -f ${RESOLVED_SOURCE%/*}/rvv_fallback.c ]] || { die "Missing IME source/fallback $1"; return 1; }
}
check_common_contract() {
  local name; while IFS= read -r name; do resolve_rvv "$name"; done < <(inventory_rvv)
  while IFS= read -r name; do resolve_ime "$name"; done < <(inventory_ime)
  local file; for file in bench.c bench.h counters.h rvv_adapter.c ime_adapter.c; do
    [[ -f $PROJECT_ROOT/benchmarking/src/$file ]] || die "Missing common driver $file"
  done
  [[ -f $PROJECT_ROOT/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h ]] || die 'Missing RVV packer'
  printf 'READ-ONLY CHECK: 36 RVV + 8 IME source configurations found; runtime validation is still required.\n'
}
ensure_hardware() {
  [[ $(uname -s) == Linux && $(uname -m) == riscv64 ]] || die 'Native runs require Linux/riscv64 on SpaceMiT K1; use --check on this host'
  local tool; for tool in "$CC" jq awk sort sha256sum taskset; do need "$tool"; done
}
snapshot_environment() {
  local file=$1
  {
    printf 'timestamp_utc=%s\ncampaign=%s\nproject_root=%s\n' "$(date -u +%FT%TZ)" "$CAMPAIGN" "$PROJECT_ROOT"
    printf 'runs=%s\nwarmups=%s\nseed=%s\nime_cpus=%s\nrvv_cpus=%s\nfp_cpu=%s\n' "$RUNS" "$WARMUPS" "$SEED" "$IME_CPUS" "$RVV_CPUS" "$FP_CPU"
    uname -a
    if command -v lscpu >/dev/null 2>&1; then lscpu; fi
    if [[ -r /proc/self/status ]]; then awk '/Cpus_allowed_list/{print}' /proc/self/status; fi
    if [[ -r /proc/cpuinfo ]]; then sed -n '1,100p' /proc/cpuinfo; fi
    if command -v "$CC" >/dev/null 2>&1; then "$CC" --version; fi
    printf 'compile_flags='; printf '%q ' "${CFLAGS_ARRAY[@]}"; printf '\n'
    env | sort | awk '/^(OMP_|GOMP_|GEMM_|SPACEMIT_IME_)/ {print}'
    if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then printf 'perf_event_paranoid='; tr '\n' ' ' < /proc/sys/kernel/perf_event_paranoid; printf '\n'; fi
    local f; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq /sys/class/thermal/thermal_zone*/temp; do
      if [[ -r $f ]]; then printf '%s=' "$f"; tr '\n' ' ' < "$f"; printf '\n'; fi
    done
    if command -v git >/dev/null 2>&1; then git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true; git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no 2>/dev/null || true; fi
  } > "$file"
}
init_figure() {
  FIGURE_ID=$1; local mode=${2:-native}
  validate_settings; need jq; need awk; need sort; need sha256sum
  [[ $mode == offline ]] || ensure_hardware
  OUT="$NB_ROOT/results/$CAMPAIGN/$FIGURE_ID"
  mkdir -p -- "$NB_ROOT/results/$CAMPAIGN"
  mkdir -- "$OUT" || { die "Output already exists: $OUT. Use a new CAMPAIGN; no overwrite is permitted."; return 1; }
  mkdir -- "$OUT/raw" "$OUT/builds"
  : > "$OUT/accepted.jsonl"; : > "$OUT/failures.tsv"; : > "$OUT/validation.jsonl"
  printf 'case_id\tstatus\tmessage\n' > "$OUT/failures.tsv"
  snapshot_environment "$OUT/environment_before.txt"
  printf 'Running %s; output: %s\n' "$FIGURE_ID" "$OUT"
}
record_failure() {
  FAILURES=$((FAILURES+1)); local msg=${2//$'\n'/ }; msg=${msg//$'\t'/ }
  printf '%s\t%s\t%s\n' "$1" "${3:-FAILED}" "$msg" >> "$OUT/failures.tsv"
  printf 'FAILED %s: %s\n' "$1" "$msg" >&2
}
log_command() {
  local log=$1; shift
  { printf '\nCOMMAND: '; printf '%q ' "$@"; printf '\n'; } >> "$log"
  "$@" >> "$log" 2>&1
}
build_pair() {
  local rvv=$1 ime=$2 rs is key dir dispatch config obj file i
  resolve_rvv "$rvv" || return 1; rs=$RESOLVED_SOURCE
  resolve_ime "$ime" || return 1; is=$RESOLVED_SOURCE
  dispatch="$PROJECT_ROOT/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h"
  local -a inputs=("$rs" "$is" "${is%/*}/rvv_fallback.c" "$PROJECT_ROOT/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/"*.h "$PROJECT_ROOT/benchmarking/src/"*.[ch])
  RVV_SOURCE_SHA256=$(sha256sum "$rs" | awk '{print $1}')
  IME_SOURCE_SHA256=$(sha256sum "$is" | awk '{print $1}')
  SOURCE_FINGERPRINT=$(sha256sum -- "${inputs[@]}" | sha256sum | awk '{print $1}')
  key=$({ printf '%s\n' "$rvv" "$ime" "$SOURCE_FINGERPRINT" "$CC" "${CFLAGS_ARRAY[*]}"; "$CC" --version; } | sha256sum | awk '{print $1}')
  dir="$NB_ROOT/results/$CAMPAIGN/_build/$key"; BENCH="$dir/bench"
  if [[ -f $dir/complete && -x $BENCH ]]; then BINARY_SHA256=$(sha256sum "$BENCH" | awk '{print $1}'); return 0; fi
  mkdir -p -- "$dir"; config="$dir/selected_sources.h"
  local rnr; [[ $rvv =~ kernel_8x([48])_ ]] || return 1; rnr=${BASH_REMATCH[1]}
  printf '#define BENCH_RVV_NR %s\n#define BENCH_DISPATCH_SOURCE %s\n#define BENCH_IME_SOURCE %s\n' "$rnr" "$(jq -Rn --arg x "$dispatch" '$x')" "$(jq -Rn --arg x "$is" '$x')" > "$config"
  local -a flags=("${CFLAGS_ARRAY[@]}" -I "$PROJECT_ROOT/benchmarking/src" -include "$config")
  local -a sources=("$PROJECT_ROOT/benchmarking/src/bench.c" "$PROJECT_ROOT/benchmarking/src/rvv_adapter.c" "$PROJECT_ROOT/benchmarking/src/ime_adapter.c" "$rs" "${is%/*}/rvv_fallback.c") objects=() extra=()
  for i in "${!sources[@]}"; do
    obj="$dir/unit_$i.o"; objects+=("$obj"); extra=(); [[ $i != 3 ]] || extra=(-DCNAME=bench_rvv_entry)
    if ! log_command "$dir/build.log" "$CC" "${flags[@]}" "${extra[@]}" -c "${sources[i]}" -o "$obj"; then record_failure "build_$rvv+$ime" "See $dir/build.log" BUILD_FAILED; return 1; fi
  done
  if ! log_command "$dir/build.log" "$CC" "${objects[@]}" -fopenmp -no-pie -o "$BENCH"; then record_failure "link_$rvv+$ime" "See $dir/build.log" BUILD_FAILED; return 1; fi
  local after_fingerprint; after_fingerprint=$(sha256sum -- "${inputs[@]}" | sha256sum | awk '{print $1}')
  [[ $after_fingerprint == "$SOURCE_FINGERPRINT" ]] || { record_failure "build_$rvv+$ime" 'Source files changed during compilation; binary rejected' SOURCE_CHANGED; return 1; }
  sha256sum -- "${inputs[@]}" > "$dir/source_hashes.sha256"
  BINARY_SHA256=$(sha256sum "$BENCH" | awk '{print $1}')
  "$CC" --version > "$dir/compiler.txt"
  if command -v ldd >/dev/null 2>&1; then ldd "$BENCH" > "$dir/runtime_libraries.txt" 2>&1 || true; fi
  jq -n --arg rvv "$rvv" --arg ime "$ime" --arg hash "$SOURCE_FINGERPRINT" --arg binary "$BINARY_SHA256" --arg compiler "$CC" --arg flags "${flags[*]}" '{rvv_kernel:$rvv,ime_kernel:$ime,source_fingerprint:$hash,binary_sha256:$binary,compiler:$compiler,flags:$flags,status:"BUILT_NOT_YET_VALIDATED",engine:"common_flat_team",strip_columns:32}' > "$dir/build.json"
  printf 'built\n' > "$dir/complete"
}
preflight_once() {
  local marker="$NB_ROOT/results/$CAMPAIGN/_preflight.ok" cpu signature
  build_pair igemm_kernel_8x4_zvl256b_lmulmf8_unroll2 ime_kernel_8x4_zvl256b_lmul1_unroll1 || return 1
  signature="$ALL_CPUS|$SOURCE_FINGERPRINT|$BINARY_SHA256"
  if [[ -f $marker ]]; then
    [[ $(< "$marker") == "$signature" ]] || { die 'Preflight CPU/source/compiler identity changed; start a new CAMPAIGN'; return 1; }
    return 0
  fi
  local -a cpus; IFS=, read -r -a cpus <<< "$ALL_CPUS"
  for cpu in "${cpus[@]}"; do
    validate_case "preflight_rvv_cpu$cpu" igemm_kernel_8x4_zvl256b_lmulmf8_unroll2 ime_kernel_8x4_zvl256b_lmul1_unroll1 rvv 16 16 64 1 0 "$cpu" static 4 1 || return 1
  done
  IFS=, read -r -a cpus <<< "$IME_CPUS"
  for cpu in "${cpus[@]}"; do
    validate_case "preflight_ime_cpu$cpu" igemm_kernel_8x4_zvl256b_lmulmf8_unroll2 ime_kernel_8x4_zvl256b_lmul1_unroll1 ime 16 16 64 1 1 "$cpu" static 4 1 || return 1
  done
  printf '%s\n' "$signature" > "$marker"
}
validate_case() {
  local id=$1 rvv=$2 ime=$3 impl=$4 m=$5 n=$6 k=$7 threads=$8 iw=$9 cpus=${10} sched=${11} weight=${12} chunk=${13}
  [[ $id =~ ^[A-Za-z0-9_.+-]+$ ]] || { die 'Unsafe case id'; return 1; }
  if ! build_pair "$rvv" "$ime"; then record_failure "$id" 'Selected pair could not be built/resolved' BUILD_FAILED; return 1; fi
  local raw="$OUT/raw/$id.validation.jsonl" log="$OUT/raw/$id.validation.log"
  [[ ! -e $raw && ! -e $log ]] || { die "Duplicate validation case $id"; return 1; }
  local -a args=(--implementation "$impl" --timing end_to_end --m "$m" --n "$n" --k "$k" --threads "$threads" --ime-workers "$iw" --cpus "$cpus" --schedule "$sched" --weight "$weight" --chunk "$chunk" --warmups 0 --repetitions 1 --seed "$SEED" --profile 0 --counters 0 --validate-only 1)
  { printf 'COMMAND: '; printf '%q ' "$BENCH" "${args[@]}"; printf '\n'; } > "$log"
  if ! "$BENCH" "${args[@]}" > "$raw" 2>> "$log"; then record_failure "$id" "Native validation failed; see $log" VALIDATION_FAILED; return 1; fi
  if ! jq -e -s 'length==1 and .[0].record_type=="validation" and .[0].status=="OK" and .[0].validation=="PASS"' "$raw" >/dev/null; then record_failure "$id" 'Invalid validation record' VALIDATION_FAILED; return 1; fi
  jq -c --arg id "$id" --arg campaign "$CAMPAIGN" --arg rvv "$rvv" --arg ime "$ime" --arg impl "$impl" --arg cpus "$cpus" --arg sched "$sched" --arg hash "$SOURCE_FINGERPRINT" --arg binary "$BINARY_SHA256" --argjson m "$m" --argjson n "$n" --argjson k "$k" --argjson t "$threads" --argjson iw "$iw" '. + {case_id:$id,campaign:$campaign,rvv_kernel:$rvv,ime_kernel:$ime,implementation:$impl,M:$m,N:$n,K:$k,threads:$t,ime_workers:$iw,cpu_list:$cpus,schedule:$sched,source_fingerprint:$hash,binary_sha256:$binary,reference:"independent_INT64_nonzero_initial_C",native_ime_required:($iw>0)}' "$raw" >> "$OUT/validation.jsonl"
}
measure_case() {
  local id=$1 rvv=$2 ime=$3 impl=$4 timing=$5 m=$6 n=$7 k=$8 threads=$9 iw=${10} cpus=${11} sched=${12} weight=${13} chunk=${14} profile=${15} counters=${16} index=${17:-0}
  [[ $id =~ ^[A-Za-z0-9_.+-]+$ ]] || { die 'Unsafe case id'; return 1; }
  if ! build_pair "$rvv" "$ime"; then record_failure "$id" 'Selected pair could not be built/resolved' BUILD_FAILED; return 1; fi
  local reps=$RUNS suffix= role=primary
  if ((index>0)); then reps=1; suffix=".sample$index"; fi
  if ((profile)); then role=profile; elif ((counters)); then role=counter; fi
  local raw="$OUT/raw/$id$suffix.jsonl" log="$OUT/raw/$id$suffix.log" tmp="$OUT/raw/$id$suffix.accepted.jsonl"
  [[ ! -e $raw && ! -e $log ]] || { die "Duplicate case $id$suffix"; return 1; }
  printf 'Measuring %s: %sx%sx%s, %s workers, %s, %s samples (%s).\n' "$id$suffix" "$m" "$n" "$k" "$threads" "$timing" "$reps" "$role"
  local -a args=(--implementation "$impl" --timing "$timing" --m "$m" --n "$n" --k "$k" --threads "$threads" --ime-workers "$iw" --cpus "$cpus" --schedule "$sched" --weight "$weight" --chunk "$chunk" --warmups "$WARMUPS" --repetitions "$reps" --seed "$SEED" --profile "$profile" --counters "$counters" --validate-only 0)
  { printf 'COMMAND: '; printf '%q ' "$BENCH" "${args[@]}"; printf '\n'; } > "$log"
  snapshot_environment "$OUT/raw/$id$suffix.environment_before.txt"
  if ! "$BENCH" "${args[@]}" > "$raw" 2>> "$log"; then record_failure "$id$suffix" "Runtime or correctness failure; raw rows rejected, see $log"; return 1; fi
  if ! jq -e -s --arg impl "$impl" --arg timing "$timing" --arg sched "$sched" --arg cpus "$cpus" --argjson m "$m" --argjson n "$n" --argjson k "$k" --argjson t "$threads" --argjson iw "$iw" --argjson reps "$reps" --argjson seed "$SEED" --argjson prof "$profile" --argjson ctr "$counters" '
    def abs: if .<0 then -. else . end;
    length==$reps and ([.[].rep]|sort)==[range(1;$reps+1)] and all(.[];
      .status=="OK" and .validation=="PASS" and .datatype=="INT8_INT32" and .implementation==$impl and .timing_mode==$timing and .schedule==$sched and
      .M==$m and .N==$n and .K==$k and .threads==$t and .ime_workers==$iw and .rvv_workers==($t-$iw) and .seed==$seed and
      .profiled==($prof==1) and .total_sec>0 and (.total_sec|isfinite) and
      (.gops-(2*$m*$n*$k/(1e9*.total_sec))|abs)<(1e-8*(1+.gops)) and
      .cpu_ids==($cpus|split(",")|map(tonumber)) and (.workers|length)==$t and
      all(.workers[]; .cpu_before==.cpu_after and .cpu_before==($cpus|split(",")|map(tonumber))[.id]) and
      ([.workers[].strips]|add)==(($n+31)/32|floor) and
      (if $ctr==0 then .counters_status=="DISABLED" else .counters_status=="OK" or .counters_status=="UNAVAILABLE_OR_MULTIPLEXED" end)
    )' "$raw" >/dev/null; then record_failure "$id$suffix" 'Malformed/incomplete records, affinity, work, scope, or GOPS check failed' RECORD_REJECTED; return 1; fi
  if ! jq -c --arg id "$id" --arg fig "$FIGURE_ID" --arg campaign "$CAMPAIGN" --arg rvv "$rvv" --arg ime "$ime" --arg role "$role" --arg hash "$SOURCE_FINGERPRINT" --arg binary "$BINARY_SHA256" --arg rvv_hash "$RVV_SOURCE_SHA256" --arg ime_hash "$IME_SOURCE_SHA256" --argjson weight "$weight" --argjson chunk "$chunk" --argjson index "$index" --argjson warm "$WARMUPS" '
    . + {case_id:$id,figure_id:$fig,campaign:$campaign,rvv_kernel:$rvv,ime_kernel:$ime,
      rvv_tile:($rvv|capture("kernel_(?<tile>8x[48])_").tile),ime_tile:($ime|capture("kernel_(?<tile>8x[48])_").tile),
      rvv_lmul:($rvv|capture("lmul(?<lmul>mf8|mf4|mf2|1|2)_").lmul),ime_lmul:"1",
      rvv_unroll:($rvv|capture("unroll(?<u>[1248])$").u|tonumber),ime_unroll:($ime|capture("unroll(?<u>[1248])$").u|tonumber),
      sample_index:(if $index>0 then $index else .rep end),measurement_role:$role,static_weight:$weight,dynamic_chunk:$chunk,
      binary_sha256:$binary,source_fingerprint:$hash,rvv_source_sha256:$rvv_hash,ime_source_sha256:$ime_hash,warmups:$warm,
      primary_scope:"common_flat_team_excludes_allocation_team_creation_affinity_and_reference_includes_assignment_packing_if_e2e_compute_output_tails_and_barriers",
      engine:"common_flat_team_static_partition_cyclic_workers_dynamic_atomic_queue",strip_columns:32,
      scheduling_overhead:"not_separately_measured",rvv_output_time:"fused_not_separately_measured"}' "$raw" > "$tmp"; then record_failure "$id$suffix" 'Metadata enrichment failed' RECORD_REJECTED; return 1; fi
  [[ $(wc -l < "$tmp") -eq $reps ]] || { record_failure "$id$suffix" 'Enrichment lost measurement records' RECORD_REJECTED; return 1; }
  cat -- "$tmp" >> "$OUT/accepted.jsonl"
  snapshot_environment "$OUT/raw/$id$suffix.environment_after.txt"
}
summarize_common() {
  local input=$1 prefix=$2
  if ! jq -s '
    def mean: add/length;
    def quantile($p): sort as $a | (($a|length)-1)*$p as $h | ($h|floor) as $i | $a[$i]+($h-$i)*($a[([$i+1,($a|length)-1]|min)]-$a[$i]);
    def stats: . as $a | ($a|mean) as $m | {n:length,mean:$m,median:quantile(0.5),q1:quantile(0.25),q3:quantile(0.75),min:min,max:max,sd:(if length>1 then (map((.-$m)*(.-$m))|add/(($a|length)-1)|sqrt) else 0 end)};
    def fieldstats($key): [ .[] | .[$key] | select(.!=null)] | if length>0 then stats else null end;
    group_by([.case_id,.implementation,.timing_mode,.M,.N,.K,.threads,.schedule,.rvv_kernel,.ime_kernel,.measurement_role,.static_weight,.dynamic_chunk]) |
    map(if (map([.campaign,.source_fingerprint,.binary_sha256,.cpu_ids,.seed,.warmups,.primary_scope])|unique|length)!=1 or (map(.sample_index)|unique|length)!=length then error("Mixed identities or duplicate repetitions in summary group") else . end |
      .[0] as $r | {case_id:$r.case_id,campaign:$r.campaign,source_fingerprint:$r.source_fingerprint,binary_sha256:$r.binary_sha256,cpu_ids:$r.cpu_ids,seed:$r.seed,warmups:$r.warmups,
      implementation:$r.implementation,timing_mode:$r.timing_mode,M:$r.M,N:$r.N,K:$r.K,threads:$r.threads,rvv_workers:$r.rvv_workers,ime_workers:$r.ime_workers,schedule:$r.schedule,rvv_kernel:$r.rvv_kernel,ime_kernel:$r.ime_kernel,rvv_tile:$r.rvv_tile,ime_tile:$r.ime_tile,rvv_lmul:$r.rvv_lmul,ime_lmul:$r.ime_lmul,rvv_unroll:$r.rvv_unroll,ime_unroll:$r.ime_unroll,is_global_selected_pair:($r.is_global_selected_pair//false),measurement_role:$r.measurement_role,static_weight:$r.static_weight,dynamic_chunk:$r.dynamic_chunk,
      time:([.[].total_sec]|stats),gops:([.[].gops]|stats),ipc:fieldstats("ipc"),
      packing:fieldstats("packing_sec"),kernel:fieldstats("kernel_sec"),ime_output:fieldstats("ime_output_sec"),ime_boundary:fieldstats("ime_boundary_sec"),
      cycles:fieldstats("cycles"),instructions:fieldstats("instructions"),cache_references:fieldstats("cache_references"),cache_misses:fieldstats("cache_misses"),
      phase_aggregation:"sum_worker_elapsed_NOT_wall_clock_decomposition",counter_scope:"sum_worker_measured_regions",
      rvv_output_time:"fused_not_separately_measured",scheduling_overhead:"not_separately_measured"})' "$input" > "$prefix.json"; then return 1; fi
  jq -r '(["case_id","campaign","implementation","timing_mode","M","N","K","threads","schedule","rvv_kernel","ime_kernel","role","weight","chunk","n","mean_sec","sd_sec","median_sec","q1_sec","q3_sec","min_sec","max_sec","mean_gops","mean_ipc"]|@csv),(.[]|[.case_id,.campaign,.implementation,.timing_mode,.M,.N,.K,.threads,.schedule,.rvv_kernel,.ime_kernel,.measurement_role,.static_weight,.dynamic_chunk,.time.n,.time.mean,.time.sd,.time.median,.time.q1,.time.q3,.time.min,.time.max,.gops.mean,.ipc.mean]|@csv)' "$prefix.json" > "$prefix.csv"
}
load_selection() {
  local file=${SELECTION_FILE:-$NB_ROOT/results/$CAMPAIGN/selected_kernels.tsv} tile rvv ime
  [[ -f $file ]] || { die "Run Figure 5 first, or set SELECTION_FILE to its selected_kernels.tsv: $file"; return 1; }
  SELECTED_PAIRS=(); local -A seen=()
  while IFS=$'\t' read -r tile rvv ime; do
    ime=${ime%$'\r'}
    [[ $tile != tile ]] || continue; [[ $tile == 8x4 || $tile == 8x8 ]] || { die 'Invalid selected tile'; return 1; }
    [[ ! ${seen[$tile]+yes} && $rvv == igemm_kernel_${tile}_* && $ime == ime_kernel_${tile}_* ]] || { die 'Duplicate/mismatched selected tile'; return 1; }
    seen[$tile]=1; resolve_rvv "$rvv" || return 1; resolve_ime "$ime" || return 1; SELECTED_PAIRS+=("$tile|$rvv|$ime")
  done < "$file"
  (( ${#SELECTED_PAIRS[@]}==2 )) || die 'Selection needs both supported tiles'
}
load_u_selection() {
  local file=${U_SELECTION_FILE:-$NB_ROOT/results/$CAMPAIGN/selected_u_kernels.tsv} tile u rvv ime
  [[ -f $file ]] || { die "Missing per-U selection: $file"; return 1; }; SELECTED_U_PAIRS=(); local -A seen=()
  while IFS=$'\t' read -r tile u rvv ime; do
    ime=${ime%$'\r'}
    [[ $tile != tile ]] || continue
    [[ ($tile == 8x4 || $tile == 8x8) && $u =~ ^[1248]$ && ! ${seen[$tile:$u]+yes} && $rvv == igemm_kernel_${tile}_*unroll${u} && $ime == ime_kernel_${tile}_*unroll${u} ]] || { die 'Invalid/duplicate per-U selection'; return 1; }
    seen[$tile:$u]=1; resolve_rvv "$rvv" || return 1; resolve_ime "$ime" || return 1; SELECTED_U_PAIRS+=("$tile|$u|$rvv|$ime")
  done < "$file"
  (( ${#SELECTED_U_PAIRS[@]}==8 )) || die 'Per-U selection needs eight tile/U pairs'
}
get_figure_data() {
  SOURCE_DATA=${FIG06_DATASET:-$NB_ROOT/results/$CAMPAIGN/$1/accepted.jsonl}
  [[ -f $SOURCE_DATA ]] || { die "Missing source dataset: $SOURCE_DATA"; return 1; }
}
load_global_selection() {
  local file=${GLOBAL_SELECTION_FILE:-$NB_ROOT/results/$CAMPAIGN/global_selected_pair.tsv} line=0 rvv ime
  [[ -f $file ]] || { die "Missing global independent selection: $file"; return 1; }
  while IFS=$'\t' read -r rvv ime; do
    ime=${ime%$'\r'}
    [[ $rvv != rvv_kernel ]] || continue; line=$((line+1))
    resolve_rvv "$rvv" || return 1; resolve_ime "$ime" || return 1
    GLOBAL_RVV_KERNEL=$rvv; GLOBAL_IME_KERNEL=$ime
  done < "$file"
  ((line==1)) || die 'Global selection must have exactly one kernel pair'
}
finish_figure() {
  if [[ -s $OUT/accepted.jsonl ]]; then summarize_common "$OUT/accepted.jsonl" "$OUT/summary"; fi
  snapshot_environment "$OUT/environment_after.txt"
  jq -n --arg figure "$FIGURE_ID" --arg campaign "$CAMPAIGN" --argjson failures "$FAILURES" '{figure:$figure,campaign:$campaign,failures:$failures,status:(if $failures==0 then "COMPLETED_CHECK_COVERAGE_BEFORE_PUBLICATION" else "INCOMPLETE_DO_NOT_PUBLISH_AS_COMPLETE" end)}' > "$OUT/status.json"
  printf 'Finished %s; failures=%s; data=%s\n' "$FIGURE_ID" "$FAILURES" "$OUT"
  ((FAILURES==0))
}
