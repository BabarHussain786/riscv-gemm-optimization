#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
RVV_CORES="${RVV_CORES:-0 1 2 3 4 5 6 7}"
IME_CORES="${IME_CORES:-8 9 10 11 12 13 14 15}"
ENABLE_EXPERIMENTAL_MF2="${ENABLE_EXPERIMENTAL_MF2:-0}"
VALIDATE_M="${VALIDATE_M:-15}"
VALIDATE_N="${VALIDATE_N:-15}"
VALIDATE_K="${VALIDATE_K:-69}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/k3_rvv_ime_results_${M}}"
RAW_DIR="${OUT_DIR}/raw_logs"
LIVE_LOG="${OUT_DIR}/k3_rvv_ime_live_${M}_runs${RUNS}_${STAMP}.log"
RAW_CSV="${OUT_DIR}/k3_rvv_ime_raw_${M}_runs${RUNS}_${STAMP}.csv"
SUMMARY_CSV="${OUT_DIR}/k3_rvv_ime_summary_${M}_runs${RUNS}_${STAMP}.csv"
LATEST_LOG="${OUT_DIR}/k3_rvv_ime_live_latest.log"
LATEST_RAW="${OUT_DIR}/k3_rvv_ime_raw_latest.csv"
LATEST_SUMMARY="${OUT_DIR}/k3_rvv_ime_summary_latest.csv"
BUILD_FAILED_COUNT=0
NO_BENCH_COUNT=0
VALIDATION_FAILED_COUNT=0
RUN_FAILED_COUNT=0
OK_RUN_COUNT=0

mkdir -p "${RAW_DIR}"

if command -v taskset >/dev/null 2>&1; then
  HAVE_TASKSET=1
else
  HAVE_TASKSET=0
fi

if [ -w /proc/set_ai_thread ]; then
  HAVE_SET_AI_THREAD=1
else
  HAVE_SET_AI_THREAD=0
fi

board_model() {
  if [ -r /proc/device-tree/model ]; then
    tr -d '\0' < /proc/device-tree/model
  else
    printf 'unknown'
  fi
}

board_compatible() {
  if [ -r /proc/device-tree/compatible ]; then
    tr '\0' ' ' < /proc/device-tree/compatible
  else
    printf 'unknown'
  fi
}

core_domain() {
  local core="$1"
  if [ -f "/sys/firmware/devicetree/base/cpus/cpu@${core}/cpu-ai" ]; then
    printf 'RVV_IME'
  else
    printf 'RVV'
  fi
}

csv_escape() {
  local value="${1-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

csv_row() {
  local first=1
  for value in "$@"; do
    if [ "${first}" -eq 0 ]; then
      printf ',' >> "${RAW_CSV}"
    fi
    csv_escape "${value}" >> "${RAW_CSV}"
    first=0
  done
  printf '\n' >> "${RAW_CSV}"
}

log() {
  printf '%s\n' "$*" | tee -a "${LIVE_LOG}"
}

run_on_core() {
  local core="$1"
  local campaign="$2"
  shift 2
  local path_env=()

  case "${campaign}" in
    IME) path_env=("SPACEMIT_IME_FORCE_NATIVE=1") ;;
    RVV) path_env=("SPACEMIT_IME_FORCE_RVV=1") ;;
  esac

  if [ "${campaign}" = "IME" ] && [ "${HAVE_SET_AI_THREAD}" -eq 1 ]; then
    env "${path_env[@]}" bash -c '
      core="$1"
      shift
      echo $$ > /proc/set_ai_thread || exit 126
      if command -v taskset >/dev/null 2>&1; then
        exec taskset -c "${core}" "$@"
      fi
      exec "$@"
    ' run_on_ai_core "${core}" "$@"
    return
  fi

  if [ "${HAVE_TASKSET}" -eq 1 ]; then
    taskset -c "${core}" env "${path_env[@]}" "$@"
  else
    env "${path_env[@]}" "$@"
  fi
}

parse_metric_name() {
  awk '/GFLOPS[[:space:]]*[:=]/ {print "GFLOPS"; exit} /GOPS[[:space:]]*[:=]/ {print "GOPS"; exit}'
}

parse_metric_value() {
  sed -nE 's/.*(GFLOPS|GOPS)[[:space:]]*[:=][[:space:]]*([-+0-9.eE]+).*/\2/p' | head -n 1
}

parse_time_sec() {
  sed -nE 's/.*([Tt]ime|time_sec)[[:space:]]*[:=][[:space:]]*([-+0-9.eE]+).*/\2/p' | head -n 1
}

run_core_list() {
  local baseline="$1"
  local family="$2"
  local kernel="$3"
  local dir="$4"
  local campaign="$5"
  local cores="$6"

  for core in ${cores}; do
    local domain
    domain="$(core_domain "${core}")"
    log "  ${campaign}: core ${core} (${domain})"

    for run in $(seq 1 "${RUNS}"); do
      local run_log output rc metric_name metric_value time_sec status
      run_log="${RAW_DIR}/${baseline}_${family}_${kernel}_${campaign}_core${core}_run${run}_${STAMP}.log"

      if output="$(cd "${dir}" && run_on_core "${core}" "${campaign}" \
          env GEMM_VALIDATE=0 IME_VALIDATE=0 ./bench "${M}" "${N}" "${K}" 2>&1)"; then
        rc=0
      else
        rc=$?
      fi

      {
        printf 'Baseline: %s\n' "${baseline}"
        printf 'Family: %s\n' "${family}"
        printf 'Kernel: %s\n' "${kernel}"
        printf 'Campaign: %s\n' "${campaign}"
        printf 'Core: %s\n' "${core}"
        printf 'Domain: %s\n' "${domain}"
        printf 'Run: %s\n' "${run}"
        printf 'ReturnCode: %s\n' "${rc}"
        printf '%s\n' '--------------------------------------------------'
        printf '%s\n' "${output}"
      } > "${run_log}"

      metric_name="$(printf '%s\n' "${output}" | parse_metric_name)"
      metric_value="$(printf '%s\n' "${output}" | parse_metric_value)"
      time_sec="$(printf '%s\n' "${output}" | parse_time_sec)"

      if [ "${rc}" -eq 0 ] && [ -n "${metric_value}" ]; then
        status="OK"
        OK_RUN_COUNT=$((OK_RUN_COUNT + 1))
      elif printf '%s\n' "${output}" | grep -q 'KERNEL_RETURN='; then
        status="KERNEL_RETURN"
        RUN_FAILED_COUNT=$((RUN_FAILED_COUNT + 1))
      else
        status="FAILED"
        RUN_FAILED_COUNT=$((RUN_FAILED_COUNT + 1))
      fi

      csv_row "${baseline}" "${family}" "${kernel}" "${campaign}" "${core}" "${domain}" "${run}" "${M}" "${N}" "${K}" "${metric_name:-NA}" "${metric_value}" "${time_sec}" "${status}" "${rc}" "${run_log}"
      log "    run ${run}: ${status} ${metric_name:-metric}=${metric_value:-NA} time=${time_sec:-NA}"
    done
  done
}

validate_kernel() {
  local baseline="$1"
  local family="$2"
  local kernel="$3"
  local dir="$4"
  local campaign="$5"
  local core="$6"
  local validation_log output rc

  validation_log="${RAW_DIR}/${baseline}_${family}_${kernel}_${campaign}_validation_${STAMP}.log"

  if output="$(cd "${dir}" && run_on_core "${core}" "${campaign}" \
      env GEMM_VALIDATE=1 IME_VALIDATE=1 ./bench \
      "${VALIDATE_M}" "${VALIDATE_N}" "${VALIDATE_K}" 2>&1)"; then
    rc=0
  else
    rc=$?
  fi

  {
    printf 'Baseline: %s\n' "${baseline}"
    printf 'Family: %s\n' "${family}"
    printf 'Kernel: %s\n' "${kernel}"
    printf 'Campaign: %s\n' "${campaign}"
    printf 'Core: %s\n' "${core}"
    printf 'ValidationMatrix: %sx%sx%s\n' "${VALIDATE_M}" "${VALIDATE_N}" "${VALIDATE_K}"
    printf 'ReturnCode: %s\n' "${rc}"
    printf '%s\n' '--------------------------------------------------'
    printf '%s\n' "${output}"
  } > "${validation_log}"

  if [ "${rc}" -eq 0 ] && printf '%s\n' "${output}" | grep -q '^VALIDATION=OK'; then
    log "  VALIDATION_OK: ${VALIDATE_M}x${VALIDATE_N}x${VALIDATE_K} on ${campaign} core ${core}"
    return 0
  fi

  log "  VALIDATION_FAILED: ${validation_log}"
  VALIDATION_FAILED_COUNT=$((VALIDATION_FAILED_COUNT + 1))
  csv_row "${baseline}" "${family}" "${kernel}" "${campaign}" "${core}" \
      "$(core_domain "${core}")" "0" "${VALIDATE_M}" "${VALIDATE_N}" \
      "${VALIDATE_K}" "NA" "" "" "VALIDATION_FAILED" "${rc}" \
      "${validation_log}"
  return 1
}

run_family() {
  local baseline="$1"
  local family="$2"
  local root_rel="$3"
  local pattern="$4"
  local campaign_scope="${5:-BOTH}"
  local root="${SCRIPT_DIR}/${baseline}/${root_rel}"

  log ""
  log "############################################################"
  log "BASELINE: ${baseline}"
  log "FAMILY:   ${family}"
  log "ROOT:     ${root_rel}"
  log "PATTERN:  ${pattern}"
  log "############################################################"

  if [ ! -d "${root}" ]; then
    log "[WARN] missing family root: ${root}"
    return 0
  fi

  mapfile -t dirs < <(find "${root}" -maxdepth 1 -mindepth 1 -type d -name "${pattern}" | sort)
  if [ "${#dirs[@]}" -eq 0 ]; then
    log "[WARN] no kernels matched: ${pattern}"
    return 0
  fi

  for dir in "${dirs[@]}"; do
    local kernel build_log validation_core
    local make_args=()
    kernel="$(basename "${dir}")"
    build_log="${RAW_DIR}/${baseline}_${family}_${kernel}_build_${STAMP}.log"

    log ""
    log "KERNEL: ${kernel}"
    if [ "${family}" = "IME_INT8" ] && [[ "${kernel}" == *_lmulmf2_* ]]; then
      if [ "${ENABLE_EXPERIMENTAL_MF2}" != "1" ]; then
        log "  SKIPPED: experimental IME LMUL=mf2 requires ENABLE_EXPERIMENTAL_MF2=1"
        continue
      fi
      make_args+=("ENABLE_MF2=1")
    fi
    if ! (cd "${dir}" && make clean >> "${build_log}" 2>&1 && make "${make_args[@]}" >> "${build_log}" 2>&1); then
      log "  BUILD_FAILED: ${build_log}"
      BUILD_FAILED_COUNT=$((BUILD_FAILED_COUNT + 1))
      for campaign in RVV IME; do
        if [ "${campaign_scope}" = "RVV_ONLY" ] && [ "${campaign}" != "RVV" ]; then
          continue
        fi
        if [ "${campaign_scope}" = "IME_ONLY" ] && [ "${campaign}" != "IME" ]; then
          continue
        fi
        if [ "${campaign}" = "RVV" ]; then
          cores="${RVV_CORES}"
        else
          cores="${IME_CORES}"
        fi
        for core in ${cores}; do
          csv_row "${baseline}" "${family}" "${kernel}" "${campaign}" "${core}" "$(core_domain "${core}")" "0" "${M}" "${N}" "${K}" "NA" "" "" "BUILD_FAILED" "1" "${build_log}"
        done
      done
      continue
    fi

    if [ ! -x "${dir}/bench" ]; then
      log "  NO_BENCH: ${dir}/bench"
      NO_BENCH_COUNT=$((NO_BENCH_COUNT + 1))
      for campaign in RVV IME; do
        if [ "${campaign_scope}" = "RVV_ONLY" ] && [ "${campaign}" != "RVV" ]; then
          continue
        fi
        if [ "${campaign_scope}" = "IME_ONLY" ] && [ "${campaign}" != "IME" ]; then
          continue
        fi
        if [ "${campaign}" = "RVV" ]; then
          cores="${RVV_CORES}"
        else
          cores="${IME_CORES}"
        fi
        for core in ${cores}; do
          csv_row "${baseline}" "${family}" "${kernel}" "${campaign}" "${core}" "$(core_domain "${core}")" "0" "${M}" "${N}" "${K}" "NA" "" "" "NO_BENCH" "1" "${build_log}"
        done
      done
      continue
    fi

    if [ "${campaign_scope}" = "RVV_ONLY" ]; then
      validation_core="${RVV_CORES%% *}"
      if ! validate_kernel "${baseline}" "${family}" "${kernel}" "${dir}" \
          "RVV" "${validation_core}"; then
        continue
      fi
      run_core_list "${baseline}" "${family}" "${kernel}" "${dir}" "RVV" "${RVV_CORES}"
    elif [ "${campaign_scope}" = "IME_ONLY" ]; then
      validation_core="${IME_CORES%% *}"
      if ! validate_kernel "${baseline}" "${family}" "${kernel}" "${dir}" \
          "IME" "${validation_core}"; then
        continue
      fi
      run_core_list "${baseline}" "${family}" "${kernel}" "${dir}" "IME" "${IME_CORES}"
    else
      validation_core="${RVV_CORES%% *}"
      if ! validate_kernel "${baseline}" "${family}" "${kernel}" "${dir}" \
          "RVV" "${validation_core}"; then
        continue
      fi
      run_core_list "${baseline}" "${family}" "${kernel}" "${dir}" "RVV" "${RVV_CORES}"
      validation_core="${IME_CORES%% *}"
      if ! validate_kernel "${baseline}" "${family}" "${kernel}" "${dir}" \
          "IME" "${validation_core}"; then
        continue
      fi
      run_core_list "${baseline}" "${family}" "${kernel}" "${dir}" "IME" "${IME_CORES}"
    fi
  done
}

write_summary() {
  printf 'baseline,family,kernel,campaign,core,domain,metric,ok_runs,avg_value,min_value,max_value,avg_time_sec,ok_count,failed_count,kernel_return_count,build_failed_count,no_bench_count,validation_failed_count\n' > "${SUMMARY_CSV}"
  awk -F',' '
  BEGIN { OFS="," }
  NR == 1 { next }
  function clean(x) { gsub(/^"|"$/, "", x); gsub(/""/, "\"", x); return x }
  {
    baseline=clean($1); family=clean($2); kernel=clean($3); campaign=clean($4); core=clean($5); domain=clean($6);
    metric=clean($11); value=clean($12); time_sec=clean($13); status=clean($14);
    if (metric == "") metric="NA";
    key=baseline SUBSEP family SUBSEP kernel SUBSEP campaign SUBSEP core SUBSEP domain SUBSEP metric;
    labels[key]=baseline OFS family OFS kernel OFS campaign OFS core OFS domain OFS metric;
    status_count[key,status]++;
    if (status == "OK" && value != "") {
      v=value+0; t=time_sec+0;
      sum[key]+=v; timesum[key]+=t; count[key]++;
      if (!(key in min) || v < min[key]) min[key]=v;
      if (!(key in max) || v > max[key]) max[key]=v;
    }
  }
  END {
    for (key in labels) {
      ok=count[key]+0;
      avg=(ok>0 ? sum[key]/ok : "");
      mn=(ok>0 ? min[key] : "");
      mx=(ok>0 ? max[key] : "");
      avgt=(ok>0 ? timesum[key]/ok : "");
      failed=(status_count[key,"FAILED"]+0);
      print labels[key], ok, avg, mn, mx, avgt, status_count[key,"OK"]+0, failed, status_count[key,"KERNEL_RETURN"]+0, status_count[key,"BUILD_FAILED"]+0, status_count[key,"NO_BENCH"]+0, status_count[key,"VALIDATION_FAILED"]+0;
    }
  }
  ' "${RAW_CSV}" | sort >> "${SUMMARY_CSV}"
}

printf 'baseline,family,kernel,campaign,core,domain,run,m,n,k,metric,value,time_sec,status,return_code,log_file\n' > "${RAW_CSV}"

log "============================================================"
log "K3 RVV/IME all-core GEMM benchmark"
log "Root: ${SCRIPT_DIR}"
log "Board: $(board_model)"
log "Compatible: $(board_compatible)"
log "Matrix: M=${M} N=${N} K=${K}"
log "Runs per core: ${RUNS}"
log "RVV campaign cores: ${RVV_CORES}"
log "IME campaign cores: ${IME_CORES}"
log "Core map is detected from /sys/.../cpu-ai"
log "taskset: $([ "${HAVE_TASKSET}" -eq 1 ] && printf enabled || printf missing)"
log "set_ai_thread: $([ "${HAVE_SET_AI_THREAD}" -eq 1 ] && printf enabled || printf missing)"
log "IME launcher: uses /proc/set_ai_thread before taskset when available"
log "Path selection: IME campaign forces native IME; RVV campaign forces RVV"
log "Campaign order: FP32, FP64, canonical INT8 RVV, native IME"
log "Validation gate: ${VALIDATE_M}x${VALIDATE_N}x${VALIDATE_K} before timed runs"
log "Output: ${OUT_DIR}"
log "Started: $(date)"
log "============================================================"

run_family "GEMM_RVV_FP32_INT8_8x4_Baseline" "FP32_SGEMM" "RVV_SGEMM_FP32_8x4" "sgemm_kernel_8x4_zvl256b_lmul*_unroll*"
run_family "GEMM_RVV_FP32_INT8_8x8_Baseline" "FP32_SGEMM" "RVV_SGEMM_FP32_8x8" "sgemm_kernel_8x8_zvl256b_lmul*_unroll*"

run_family "GEMM_RVV_FP64_INT8_8x4_Baseline" "FP64_DGEMM" "RVV_DGEMM_FP64_8x4" "dgemm_kernel_8x4_zvl256b_lmul*_unroll*"
run_family "GEMM_RVV_FP64_INT8_8x8_Baseline" "FP64_DGEMM" "RVV_DGEMM_FP64_8x8" "dgemm_kernel_8x8_zvl256b_lmul*_unroll*"

run_family "GEMM_RVV_FP32_INT8_8x4_Baseline" "INT8_RVV" "RVV_IGEMM_INT8_I8I32_8x4" "igemm_kernel_8x4_zvl256b_lmul*_unroll*"
run_family "GEMM_RVV_FP32_INT8_8x8_Baseline" "INT8_RVV" "RVV_IGEMM_INT8_I8I32_8x8" "igemm_kernel_8x8_zvl256b_lmul*_unroll*"

run_family "IME_NATIVE_KERNELS" "IME_INT8" "IME_GEMM_INT8_I8I32_8x4_NATIVE" "ime_kernel_8x4_zvl*b_lmul*_unroll*" "IME_ONLY"
run_family "IME_NATIVE_KERNELS" "IME_INT8" "IME_GEMM_INT8_I8I32_8x8_NATIVE" "ime_kernel_8x8_zvl*b_lmul*_unroll*" "IME_ONLY"

write_summary

log ""
log "============================================================"
if [ "${BUILD_FAILED_COUNT}" -eq 0 ] && [ "${NO_BENCH_COUNT}" -eq 0 ] &&
   [ "${VALIDATION_FAILED_COUNT}" -eq 0 ] && [ "${RUN_FAILED_COUNT}" -eq 0 ]; then
  CAMPAIGN_STATUS="OK"
else
  CAMPAIGN_STATUS="FAILED"
fi
log "DONE status=${CAMPAIGN_STATUS}: $(date)"
log "ok_runs=${OK_RUN_COUNT}"
log "failed_runs=${RUN_FAILED_COUNT}"
log "validation_failed=${VALIDATION_FAILED_COUNT}"
log "build_failed=${BUILD_FAILED_COUNT}"
log "no_bench=${NO_BENCH_COUNT}"
log "Live log: ${LIVE_LOG}"
log "Raw CSV: ${RAW_CSV}"
log "Summary CSV: ${SUMMARY_CSV}"
log "Latest log: ${LATEST_LOG}"
log "Latest raw CSV: ${LATEST_RAW}"
log "Latest summary CSV: ${LATEST_SUMMARY}"
log "Raw logs: ${RAW_DIR}"
log "============================================================"

cp "${LIVE_LOG}" "${LATEST_LOG}"
cp "${RAW_CSV}" "${LATEST_RAW}"
cp "${SUMMARY_CSV}" "${LATEST_SUMMARY}"

if [ "${CAMPAIGN_STATUS}" != "OK" ]; then
  exit 1
fi
