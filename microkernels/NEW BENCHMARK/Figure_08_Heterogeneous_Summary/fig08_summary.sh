#!/usr/bin/env bash
# Dataset derivation only. This script never builds or runs a GEMM kernel.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"
usage() {
  cat <<'HELP'
Usage: bash fig08_summary.sh [--help|--check]
Derives Figure 8 ONLY from accepted Figure 6 raw records. No board, compiler,
or new benchmark is needed. Use the same CAMPAIGN, or set FIG06_DATASET to an
explicit new-solution Figure 6 accepted.jsonl. Historical datasets are not
silently imported. SUMMARY_TILE=best chooses the global independent pair.
Use SUMMARY_TILE=8x8 or 8x4 to select a same-tile control instead.
Optional SUMMARY_RVV_KERNEL and SUMMARY_IME_KERNEL explicitly select a pair
if the source contains more than one. Main shape is 1024^3, mixed 4+4,
end_to_end, unprofiled, counters disabled; STATIC_WEIGHT=4/DYNAMIC_CHUNK=1
by default (use the same overrides as the source Figure 6 campaign).
The script verifies per-record GOPS=2*M*N*K/(1e9*total_sec), campaign identity,
unique kernel pair, and sufficient repetitions for both modes.
--check verifies source contracts only, without reading result files or
creating output. This derivation preserves original record provenance.
HELP
}
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check) check_common_contract; printf 'Figure 8: derivation only from matched Figure 6 records; contracts OK.\n'; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
SUMMARY_TILE=${SUMMARY_TILE:-best}
STATIC_WEIGHT=${STATIC_WEIGHT:-4}; DYNAMIC_CHUNK=${DYNAMIC_CHUNK:-1}
positive_integer STATIC_WEIGHT "$STATIC_WEIGHT"; positive_integer DYNAMIC_CHUNK "$DYNAMIC_CHUNK"
[[ "$SUMMARY_TILE" == best || "$SUMMARY_TILE" == 8x4 || "$SUMMARY_TILE" == 8x8 ]] || { printf 'SUMMARY_TILE must be best, 8x4 or 8x8.\n' >&2; exit 2; }
init_figure fig08_heterogeneous_summary offline
if [[ -n "${FIG06_DATASET:-}" ]]; then SOURCE_DATA=$FIG06_DATASET
elif ! get_figure_data fig06_fair_comparison; then
  record_failure source 'Figure 6 accepted dataset was not found in this campaign.'
  finish_figure; exit 1
fi
[[ -s "$SOURCE_DATA" ]] || { record_failure source "Missing Figure 6 accepted dataset: $SOURCE_DATA"; finish_figure; exit 1; }
command -v jq >/dev/null || { record_failure dependency 'jq is required for dataset derivation.'; finish_figure; exit 1; }
if ! jq -s -e --arg tile "$SUMMARY_TILE" '
  length>0 and (map(.campaign)|unique|length)==1 and
  all(.[]; .figure_id=="fig06_fair_comparison" and .status=="OK" and
    .validation=="PASS" and (.campaign|type)=="string" and
    (.rvv_kernel|type)=="string" and (.ime_kernel|type)=="string")
' "$SOURCE_DATA" >/dev/null; then
  record_failure provenance 'Figure 6 source is malformed, mixes campaigns, or contains nonaccepted/non-Figure-6 records.'
  finish_figure; exit 1
fi
if ! jq -c --arg tile "$SUMMARY_TILE" --arg rvv "${SUMMARY_RVV_KERNEL:-}" --arg ime "${SUMMARY_IME_KERNEL:-}" --argjson weight "$STATIC_WEIGHT" --argjson chunk "$DYNAMIC_CHUNK" '
  select(.measurement_role=="primary" and .implementation=="mixed" and
    .timing_mode=="end_to_end" and .profiled==false and .counters_status=="DISABLED" and
    .M==1024 and .N==1024 and .K==1024 and .threads==8 and .rvv_workers==4 and .ime_workers==4 and
    (if $tile=="best" then .is_global_selected_pair==true else .rvv_tile==$tile and .ime_tile==$tile end) and
    (.schedule=="static" or .schedule=="dynamic") and
    .static_weight==$weight and .dynamic_chunk==$chunk and
    ($rvv=="" or .rvv_kernel==$rvv) and ($ime=="" or .ime_kernel==$ime))
' "$SOURCE_DATA" > "$OUT/source_records.jsonl"; then
  record_failure parsing 'Could not parse Figure 6 input.'; finish_figure; exit 1
fi
if ! jq -s -e --argjson expected "$RUNS" '
  def abs: if .<0 then -. else . end;
  length>0 and (map([.rvv_kernel,.ime_kernel])|unique|length)==1 and
  (map([.cpu_ids,.seed,.warmups,.source_fingerprint,.binary_sha256,.primary_scope,.static_weight,.dynamic_chunk])|unique|length)==1 and
  all(.[]; (.cpu_ids|length)==8 and (.source_fingerprint|type)=="string" and
    (.binary_sha256|type)=="string" and (.seed|type)=="number" and
    (.warmups|type)=="number" and (.primary_scope|type)=="string") and
  all(.[]; .total_sec>0 and .gops>0 and
    (((.gops-(2*.M*.N*.K/(1e9*.total_sec)))/.gops)|abs)<1e-10) and
  ([.[]|select(.schedule=="static")]|length)==$expected and
  ([.[]|select(.schedule=="dynamic")]|length)==$expected and
  (group_by(.schedule)|all(.[]; (map(.sample_index)|unique|length)==length))
' "$OUT/source_records.jsonl" >/dev/null; then
  record_failure identity 'Need one exact kernel pair and matched CPU/seed/source/binary settings, RUNS unique repetitions per mode, and correct per-record GOPS/time identity.'
  finish_figure; exit 1
fi
jq -c --arg source "$SOURCE_DATA" '
  . + {source_figure_id:.figure_id,source_dataset:$source,
       figure_id:"fig08_heterogeneous_summary",derivation_only:true}
' "$OUT/source_records.jsonl" > "$OUT/accepted.jsonl"
jq -s --arg source "$SOURCE_DATA" '
  {source_dataset:$source,source_campaign:.[0].campaign,
   selected_rvv_kernel:.[0].rvv_kernel,selected_ime_kernel:.[0].ime_kernel,
   selected_rvv_tile:.[0].rvv_tile,selected_ime_tile:.[0].ime_tile,source_fingerprints:(map(.source_fingerprint)|unique),
   binary_sha256:(map(.binary_sha256)|unique),record_count:length,
   no_new_benchmark:true,throughput_definition:"2*M*N*K/(1e9*total_sec), verified for every record"}
' "$OUT/source_records.jsonl" > "$OUT/provenance.json"
cat > "$OUT/measurement_scope.txt" <<'SCOPE'
Derived exclusively from one Figure 6 campaign and one explicitly identified
RVV/IME kernel pair. Original times, GOPS, CPU mappings, kernel identities,
source hashes, and campaign are preserved. There is no new benchmark.
Only unprofiled, counter-disabled, main-shape 4+4 end-to-end measurements
are accepted. Packing and required output handling remain inside wall time.
GOPS is checked per repetition. Mean GOPS need not equal operations divided
by mean time; report the two independently calculated sample summaries.
No historical phase measurements or unrelated campaign statistics are used.
SCOPE
finish_figure
