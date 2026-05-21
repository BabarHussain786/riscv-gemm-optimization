#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_OUT="${SCRIPT_DIR}/all_single_core_results_1024"
MASTER_LOG="${MASTER_OUT}/run_all_four_1024_${STAMP}.log"
LATEST_LOG="${MASTER_OUT}/run_all_four_latest.log"
MASTER_STATUS="${MASTER_OUT}/run_all_four_status_${STAMP}.csv"
LATEST_STATUS="${MASTER_OUT}/run_all_four_status_latest.csv"
RUN_SCRIPT="run_single_core_0_7_all_kernels_1024.sh"

mkdir -p "${MASTER_OUT}"
: > "${MASTER_LOG}"
printf 'baseline,script,start_epoch,end_epoch,duration_sec,status\n' > "${MASTER_STATUS}"

BASELINES=(
  "FP64 8x4|GEMM_RVV_FP64_INT8_IME_8x4_Baseline"
  "FP64 8x8|GEMM_RVV_FP64_INT8_IME_8x8_Baseline"
  "FP32 8x4|GEMM_RVV_FP32_INT8_IME_8x4_Baseline"
  "FP32 8x8|GEMM_RVV_FP32_INT8_IME_8x8_Baseline"
)

log() {
  printf '%s\n' "$*" | tee -a "${MASTER_LOG}"
}

log "============================================================"
log "All single-core GEMM benchmarks"
log "Root: ${SCRIPT_DIR}"
log "Matrix size: M=${M:-1024}, N=${N:-1024}, K=${K:-1024}"
log "Runs per core: ${RUNS:-6}"
log "Cores: ${CORES:-0 1 2 3 4 5 6 7}"
log "Started: $(date)"
log "============================================================"

overall_status=0

for entry in "${BASELINES[@]}"; do
  baseline_name="${entry%%|*}"
  baseline_dir_name="${entry#*|}"
  baseline_dir="${SCRIPT_DIR}/${baseline_dir_name}"
  run_script_path="${baseline_dir}/${RUN_SCRIPT}"

  start_epoch="$(date +%s)"
  log ""
  log "------------------------------------------------------------"
  log "START ${baseline_name}"
  log "Folder: ${baseline_dir}"
  log "Time: $(date)"
  log "------------------------------------------------------------"

  if [[ ! -d "${baseline_dir}" ]]; then
    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))
    log "ERROR: folder not found: ${baseline_dir}"
    printf '%s,%s,%s,%s,%s,%s\n' "${baseline_name}" "${RUN_SCRIPT}" "${start_epoch}" "${end_epoch}" "${duration}" "MISSING_FOLDER" >> "${MASTER_STATUS}"
    overall_status=1
    continue
  fi

  if [[ ! -f "${run_script_path}" ]]; then
    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))
    log "ERROR: script not found: ${run_script_path}"
    printf '%s,%s,%s,%s,%s,%s\n' "${baseline_name}" "${RUN_SCRIPT}" "${start_epoch}" "${end_epoch}" "${duration}" "MISSING_SCRIPT" >> "${MASTER_STATUS}"
    overall_status=1
    continue
  fi

  (
    cd "${baseline_dir}" && bash "./${RUN_SCRIPT}"
  ) 2>&1 | tee -a "${MASTER_LOG}"
  child_status=${PIPESTATUS[0]}

  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))

  if [[ ${child_status} -eq 0 ]]; then
    log "FINISH ${baseline_name}: OK (${duration}s)"
    printf '%s,%s,%s,%s,%s,%s\n' "${baseline_name}" "${RUN_SCRIPT}" "${start_epoch}" "${end_epoch}" "${duration}" "OK" >> "${MASTER_STATUS}"
  else
    log "FINISH ${baseline_name}: FAILED status=${child_status} (${duration}s)"
    printf '%s,%s,%s,%s,%s,%s\n' "${baseline_name}" "${RUN_SCRIPT}" "${start_epoch}" "${end_epoch}" "${duration}" "FAILED_${child_status}" >> "${MASTER_STATUS}"
    overall_status=1
  fi
done

log ""
log "============================================================"
log "Completed: $(date)"
log "Master log: ${MASTER_LOG}"
log "Latest log: ${LATEST_LOG}"
log "Status CSV: ${MASTER_STATUS}"
log "Latest status CSV: ${LATEST_STATUS}"
log "Overall status: ${overall_status}"
log "============================================================"

cp "${MASTER_LOG}" "${LATEST_LOG}"
cp "${MASTER_STATUS}" "${LATEST_STATUS}"

exit "${overall_status}"