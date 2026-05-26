#!/usr/bin/env bash
set -u

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
CORES="${CORES:-0 1 2 3 4 5 6 7}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/single_core_results_${M}}"
RAW_DIR="${OUT_DIR}/raw_logs"
mkdir -p "${RAW_DIR}"

TS="$(date +"%Y%m%d_%H%M%S")"
LIVE_LOG="${OUT_DIR}/single_core_live_${TS}.log"
RAW_CSV="${OUT_DIR}/single_core_raw_${M}_runs${RUNS}_${TS}.csv"
SUMMARY_CSV="${OUT_DIR}/single_core_summary_${M}_runs${RUNS}_${TS}.csv"
LATEST_RAW="${OUT_DIR}/single_core_raw_latest.csv"
LATEST_SUMMARY="${OUT_DIR}/single_core_summary_latest.csv"

FP32_ROOT="${BASE_DIR}/RVV_SGEMM_FP32_8x4"
INT8_ROOT="${BASE_DIR}/RVV_IGEMM_INT8_I8I32_8x4"
IME_ROOT="${BASE_DIR}/IME_GEMM_INT8_I8I32_8x4_RVV_Fallback"

if command -v taskset >/dev/null 2>&1; then
  HAVE_TASKSET=1
else
  HAVE_TASKSET=0
fi

core_domain() {
  case "$1" in
    0|1|2|3) printf "RVV_IME" ;;
    4|5|6|7) printf "RVV" ;;
    *) printf "UNKNOWN" ;;
  esac
}

csv_escape() {
  local s="${1:-}"
  s="${s//\"/\"\"}"
  printf '"%s"' "${s}"
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

run_on_core() {
  local core="$1"
  shift
  if [ "${HAVE_TASKSET}" -eq 1 ]; then
    taskset -c "${core}" "$@"
  else
    "$@"
  fi
}

parse_time() {
  awk '/Time:/ {print $2; exit}'
}

parse_metric_name() {
  awk '/GFLOPS:/ {print "GFLOPS"; exit} /GOPS:/ {print "GOPS"; exit}'
}

parse_metric_value() {
  awk '/GFLOPS:/ {print $2; exit} /GOPS:/ {print $2; exit}'
}

run_family() {
  local family="$1"
  local root="$2"
  local pattern="$3"

  if [ ! -d "${root}" ]; then
    echo "[WARN] Missing family root: ${root}" | tee -a "${LIVE_LOG}"
    return 0
  fi

  mapfile -t dirs < <(find "${root}" -maxdepth 1 -type d -name "${pattern}" | sort)
  if [ "${#dirs[@]}" -eq 0 ]; then
    echo "[WARN] ${family}: no kernels for pattern ${pattern}" | tee -a "${LIVE_LOG}"
    return 0
  fi

  echo "" | tee -a "${LIVE_LOG}"
  echo "##################################################" | tee -a "${LIVE_LOG}"
  echo "FAMILY: ${family}" | tee -a "${LIVE_LOG}"
  echo "##################################################" | tee -a "${LIVE_LOG}"

  for dir in "${dirs[@]}"; do
    local kernel
    kernel="$(basename "${dir}")"
    echo "" | tee -a "${LIVE_LOG}"
    echo "KERNEL: ${kernel}" | tee -a "${LIVE_LOG}"

    (
      cd "${dir}" || exit 1
      make clean >/dev/null 2>&1 || true
      if ! make >/tmp/make_${kernel}_${TS}.log 2>&1; then
        build_log="${RAW_DIR}/${kernel}_build_failed_${TS}.log"
        cp "/tmp/make_${kernel}_${TS}.log" "${build_log}" 2>/dev/null || true
        rm -f "/tmp/make_${kernel}_${TS}.log"
        echo "[ERROR] build failed: ${kernel}" | tee -a "${LIVE_LOG}"
        for core in ${CORES}; do
          csv_row "${family}" "${kernel}" "${core}" "$(core_domain "${core}")" "0" "${M}" "${N}" "${K}" "" "" "" "BUILD_FAILED" "" "${build_log}"
        done
        exit 0
      fi
      rm -f "/tmp/make_${kernel}_${TS}.log"

      if [ ! -f ./bench ]; then
        echo "[ERROR] bench binary missing: ${kernel}" | tee -a "${LIVE_LOG}"
        for core in ${CORES}; do
          csv_row "${family}" "${kernel}" "${core}" "$(core_domain "${core}")" "0" "${M}" "${N}" "${K}" "" "" "" "NO_BENCH" "" ""
        done
        exit 0
      fi

      for core in ${CORES}; do
        domain="$(core_domain "${core}")"
        core_log="${RAW_DIR}/${kernel}_core${core}_${domain}_${TS}.log"
        : > "${core_log}"
        echo "  Core ${core} (${domain})" | tee -a "${LIVE_LOG}"

        for run in $(seq 1 "${RUNS}"); do
          if output="$(run_on_core "${core}" ./bench "${M}" "${N}" "${K}" 2>&1)"; then
            rc=0
          else
            rc=$?
          fi

          {
            echo "=================================================="
            echo "Family: ${family}"
            echo "Kernel: ${kernel}"
            echo "Core: ${core}"
            echo "Domain: ${domain}"
            echo "Run: ${run}"
            echo "ReturnCode: ${rc}"
            echo "--------------------------------------------------"
            echo "${output}"
          } >> "${core_log}"

          time_sec="$(printf '%s\n' "${output}" | parse_time)"
          metric_name="$(printf '%s\n' "${output}" | parse_metric_name)"
          metric_value="$(printf '%s\n' "${output}" | parse_metric_value)"

          if [ "${rc}" -eq 0 ] && [ -n "${metric_value}" ]; then
            status="OK"
          elif printf '%s\n' "${output}" | grep -q 'KERNEL_RETURN='; then
            status="KERNEL_RETURN"
          else
            status="FAILED"
          fi

          csv_row "${family}" "${kernel}" "${core}" "${domain}" "${run}" "${M}" "${N}" "${K}" "${metric_name}" "${metric_value}" "${time_sec}" "${status}" "${rc}" "${core_log}"
          echo "    Run ${run}: ${status} ${metric_name:-metric}=${metric_value:-NA} time=${time_sec:-NA}" | tee -a "${LIVE_LOG}"
        done
      done
    )
  done
}

write_summary() {
  awk -F',' '
  BEGIN {
    OFS=",";
    print "family","kernel","core","domain","metric","ok_runs","avg_value","min_value","max_value","avg_time_sec","ok_count","failed_count","kernel_return_count","build_failed_count","no_bench_count";
  }
  NR==1 { next }
  function clean(x) { gsub(/^"|"$/, "", x); gsub(/""/, "\"", x); return x }
  {
    family=clean($1); kernel=clean($2); core=clean($3); domain=clean($4);
    metric=clean($9); value=clean($10); time=clean($11); status=clean($12);
    status_key=family SUBSEP kernel SUBSEP core SUBSEP domain;
    if (metric == "") metric="NA";
    key=status_key SUBSEP metric;
    labels[key]=family OFS kernel OFS core OFS domain OFS metric;
    status_keys[key]=status_key;
    status_count[key,status]++;
    if (status == "OK" && value != "") {
      v=value+0; t=time+0;
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
      print labels[key], ok, avg, mn, mx, avgt, status_count[key,"OK"]+0, status_count[key,"FAILED"]+0, status_count[key,"KERNEL_RETURN"]+0, status_count[key,"BUILD_FAILED"]+0, status_count[key,"NO_BENCH"]+0;
    }
  }
  ' "${RAW_CSV}" > "${SUMMARY_CSV}"
}

echo "Single-core FP32/INT8/IME 8x4 benchmark sweep" | tee "${LIVE_LOG}"
echo "Generated: $(date)" | tee -a "${LIVE_LOG}"
echo "M=${M} N=${N} K=${K} RUNS=${RUNS}" | tee -a "${LIVE_LOG}"
echo "CORES=${CORES}" | tee -a "${LIVE_LOG}"
echo "Core map: 0-3=RVV_IME, 4-7=RVV" | tee -a "${LIVE_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${LIVE_LOG}"
if [ "${HAVE_TASKSET}" -eq 1 ]; then
  echo "taskset: enabled" | tee -a "${LIVE_LOG}"
else
  echo "taskset: NOT FOUND; runs will not be pinned" | tee -a "${LIVE_LOG}"
fi

echo 'family,kernel,core,domain,run,m,n,k,metric,value,time_sec,status,return_code,log_file' > "${RAW_CSV}"

run_family "FP32_SGEMM" "${FP32_ROOT}" "sgemm_kernel_8x4_zvl256b_lmul*_unroll*"
run_family "INT8_RVV" "${INT8_ROOT}" "igemm_kernel_8x4_zvl256b_lmul*_unroll*"
run_family "IME_INT8_RVV_FALLBACK" "${IME_ROOT}" "ime_kernel_8x4_zvl256b_*_unroll*"

write_summary
cp "${RAW_CSV}" "${LATEST_RAW}"
cp "${SUMMARY_CSV}" "${LATEST_SUMMARY}"

echo "" | tee -a "${LIVE_LOG}"
echo "DONE" | tee -a "${LIVE_LOG}"
echo "Raw CSV: ${RAW_CSV}" | tee -a "${LIVE_LOG}"
echo "Summary CSV: ${SUMMARY_CSV}" | tee -a "${LIVE_LOG}"
echo "Latest raw CSV: ${LATEST_RAW}" | tee -a "${LIVE_LOG}"
echo "Latest summary CSV: ${LATEST_SUMMARY}" | tee -a "${LIVE_LOG}"
echo "Raw logs: ${RAW_DIR}" | tee -a "${LIVE_LOG}"