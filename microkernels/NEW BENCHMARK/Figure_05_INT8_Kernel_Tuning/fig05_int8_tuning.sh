#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"
usage() {
  cat <<'HELP'
Usage: bash fig05_int8_tuning.sh [--help|--check]
All 36 canonical RVV INT8 + 8 supported native IME configurations.
Each is measured on one fixed backend CPU in PREPACKED mode, then under
end-to-end RVV8 / IME4 execution. All 44 receive multicore confirmation;
this is not a shortlist restricted to LMUL1 or a presumed best U.
Fixed M=N=K=1024; RUNS=7, WARMUPS=2 and SEED=42 can be overridden.
Independent per-tile and per-U selections are written only with complete
coverage. Best configurations receive a separately named held-out run.
--check is read-only; no dataset or build is created.
HELP
}
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check) check_common_contract; printf 'Tuning: 44 configurations x two timing/allocation scopes; K1 runtime still required.\n'; exit 0 ;;
  '') [[ $# == 0 ]] || { usage >&2; exit 2; } ;;
  *) usage >&2; exit 2 ;;
esac
init_figure fig05_int8_tuning
preflight_once
cat > "$OUT/measurement_scope.txt" <<'SCOPE'
Common native C driver, fixed canonical INT8 inputs, alpha1 and nonzero C.
One-core prepacked includes required C updates/scatter/tails and synchronization,
but excludes input packing. It is not claimed to be instruction-only timing.
Every supported configuration then receives RVV8 or IME4 end-to-end measurement.
Tuning selections use only complete uninstrumented end-to-end candidate groups.
Selection is independently performed for each backend and software tile.
Per-U selection chooses the best measured RVV LMUL with U fixed; IME LMUL1.
The subsequent held-out confirmation uses a fresh process but does not alter
the predeclared winner or silently replace it after looking at results.
Allocation, team creation, affinity, reference and validation are outside time.
All emitted measured repetitions are correctness-checked by the existing C code.
SCOPE
for backend in rvv ime; do
  if [[ $backend == rvv ]]; then mapfile -t kernels < <(inventory_rvv); else mapfile -t kernels < <(inventory_ime); fi
  for kernel in "${kernels[@]}"; do
    [[ $kernel =~ kernel_(8x[48])_ ]] || exit 2; tile=${BASH_REMATCH[1]}
    if [[ $backend == rvv ]]; then
      rvv=$kernel; ime="ime_kernel_${tile}_zvl256b_lmul1_unroll1"
      cpu=$(cpu_first "$RVV_CPUS"); workers=0; threads=8; cpus="$RVV_CPUS,$IME_CPUS"
    else
      ime=$kernel
      if [[ $tile == 8x4 ]]; then rvv=igemm_kernel_8x4_zvl256b_lmulmf8_unroll2; else rvv=igemm_kernel_8x8_zvl256b_lmulmf4_unroll4; fi
      cpu=$(cpu_first "$IME_CPUS"); workers=1; threads=4; cpus="$IME_CPUS"
    fi
    if ! measure_case "prepacked_${backend}_${kernel}" "$rvv" "$ime" "$backend" prepacked 1024 1024 1024 1 "$workers" "$cpu" static 4 1 0 0; then :; fi
    [[ $backend == rvv ]] || workers=4
    if ! measure_case "e2e_${backend}_${kernel}" "$rvv" "$ime" "$backend" end_to_end 1024 1024 1024 "$threads" "$workers" "$cpus" static 4 1 0 0; then :; fi
  done
done
summarize_common "$OUT/accepted.jsonl" "$OUT/tuning"
if ! jq -e --argjson runs "$RUNS" '
  [.[]|select(.measurement_role=="primary" and (.case_id|startswith("prepacked_")))] as $p |
  [.[]|select(.measurement_role=="primary" and (.case_id|startswith("e2e_")))] as $e |
  ($p|length)==44 and ($e|length)==44 and all($p[];$runs==.time.n) and all($e[];$runs==.time.n) and
  ([$e[]|select(.implementation=="rvv")|.rvv_kernel]|unique|length)==36 and
  ([$e[]|select(.implementation=="ime")|.ime_kernel]|unique|length)==8
' "$OUT/tuning.json" >/dev/null; then
  record_failure tuning_coverage 'Missing or incomplete configuration groups: selections are NOT published.' INCOMPLETE_COVERAGE
  finish_figure; exit 1
fi
selection="$NB_ROOT/results/$CAMPAIGN/selected_kernels.tsv"
u_selection="$NB_ROOT/results/$CAMPAIGN/selected_u_kernels.tsv"
global_selection="$NB_ROOT/results/$CAMPAIGN/global_selected_pair.tsv"
[[ ! -e $selection && ! -e $u_selection && ! -e $global_selection ]] || { die 'Selection already exists; use a new campaign'; exit 1; }
jq '[.[]|select(.measurement_role=="primary" and (.case_id|startswith("e2e_")))] as $a |
  ["8x4","8x8"]|map(. as $tile |
    ([$a[]|select(.implementation=="rvv" and .rvv_tile==$tile)]|min_by(.time.mean)) as $r |
    ([$a[]|select(.implementation=="ime" and .ime_tile==$tile)]|min_by(.time.mean)) as $i |
    {tile:$tile,rvv_kernel:$r.rvv_kernel,ime_kernel:$i.ime_kernel,rvv_mean_sec:$r.time.mean,ime_mean_sec:$i.time.mean,
      selection_metric:"minimum mean unprofiled multicore end_to_end time",rvv_candidate:$r,ime_candidate:$i})' "$OUT/tuning.json" > "$OUT/selected_pairs.json"
jq -r '(["tile","rvv_kernel","ime_kernel"]|@tsv),(.[]|[.tile,.rvv_kernel,.ime_kernel]|@tsv)' "$OUT/selected_pairs.json" > "$OUT/selected_kernels.tsv"
jq '[.[]|select(.measurement_role=="primary" and (.case_id|startswith("e2e_")))] as $a |
  ([$a[]|select(.implementation=="rvv")]|min_by(.time.mean)) as $r |
  ([$a[]|select(.implementation=="ime")]|min_by(.time.mean)) as $i |
  {rvv_kernel:$r.rvv_kernel,ime_kernel:$i.ime_kernel,rvv_tile:$r.rvv_tile,ime_tile:$i.ime_tile,
   selection_metric:"independent global minimum multicore end_to_end mean",rvv_candidate:$r,ime_candidate:$i}' "$OUT/tuning.json" > "$OUT/global_selected_pair.json"
jq -r '(["rvv_kernel","ime_kernel"]|@tsv),([.rvv_kernel,.ime_kernel]|@tsv)' "$OUT/global_selected_pair.json" > "$OUT/global_selected_pair.tsv"
jq -r '
  [.[]|select(.measurement_role=="primary" and (.case_id|startswith("e2e_")))] as $a |
  (["tile","u","rvv_kernel","ime_kernel"]|@tsv),
  (["8x4","8x8"][] as $tile | [1,2,4,8][] as $u |
    ([$a[]|select(.implementation=="rvv" and .rvv_tile==$tile and .rvv_unroll==$u)]|min_by(.time.mean)) as $r |
    ([$a[]|select(.implementation=="ime" and .ime_tile==$tile and .ime_unroll==$u)]|min_by(.time.mean)) as $i |
    [$tile,$u,$r.rvv_kernel,$i.ime_kernel]|@tsv)' "$OUT/tuning.json" > "$OUT/selected_u_kernels.tsv"
mapfile -t SELECTED_PAIRS < <(jq -r '.[]|[.tile,.rvv_kernel,.ime_kernel]|join("|")' "$OUT/selected_pairs.json" | tr -d '\r')
for pair in "${SELECTED_PAIRS[@]}"; do
  IFS='|' read -r tile rvv ime <<< "$pair"
  if ! measure_case "heldout_${tile}_rvv8" "$rvv" "$ime" rvv end_to_end 1024 1024 1024 8 0 "$RVV_CPUS,$IME_CPUS" static 4 1 0 0; then :; fi
  if ! measure_case "heldout_${tile}_ime4" "$rvv" "$ime" ime end_to_end 1024 1024 1024 4 4 "$IME_CPUS" static 4 1 0 0; then :; fi
done
if ((FAILURES==0)); then
  cp -- "$OUT/selected_kernels.tsv" "$selection"
  cp -- "$OUT/selected_u_kernels.tsv" "$u_selection"
  cp -- "$OUT/global_selected_pair.tsv" "$global_selection"
else
  printf 'Selection not exported: held-out/native checks failed.\n' >&2
fi
finish_figure
