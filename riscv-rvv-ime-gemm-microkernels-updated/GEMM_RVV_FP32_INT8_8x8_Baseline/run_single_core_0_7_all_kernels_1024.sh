#!/usr/bin/env bash
set -u -o pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
CORES="${CORES:-0 1 2 3 4 5 6 7}"
VALIDATE="${VALIDATE:-1}"
VALIDATE_M="${VALIDATE_M:-15}"
VALIDATE_N="${VALIDATE_N:-15}"
VALIDATE_K="${VALIDATE_K:-13}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/single_core_results_${M}}"
RAW_DIR="${OUT_DIR}/raw_logs"
mkdir -p "${RAW_DIR}"

TS="$(date +"%Y%m%d_%H%M%S")"
LIVE_LOG="${OUT_DIR}/single_core_live_${TS}.log"
RAW_CSV="${OUT_DIR}/single_core_raw_${M}_runs${RUNS}_${TS}.csv"
SUMMARY_CSV="${OUT_DIR}/single_core_summary_${M}_runs${RUNS}_${TS}.csv"
LATEST_RAW="${OUT_DIR}/single_core_raw_latest.csv"
LATEST_SUMMARY="${OUT_DIR}/single_core_summary_latest.csv"

FP32_ROOT="${BASE_DIR}/RVV_SGEMM_FP32_8x8"
INT8_ROOT="${BASE_DIR}/RVV_IGEMM_INT8_I8I32_8x8"

for required_tool in awk find gcc grep make seq sort taskset tee; do
  if ! command -v "${required_tool}" >/dev/null 2>&1; then
    printf 'ERROR: required tool is not available: %s\n' "${required_tool}" >&2
    exit 1
  fi
done

require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: %s must be a positive integer, got: %s\n' "${name}" "${value}" >&2
    exit 1
  fi
}

require_positive_integer M "${M}"
require_positive_integer N "${N}"
require_positive_integer K "${K}"
require_positive_integer RUNS "${RUNS}"
require_positive_integer VALIDATE_M "${VALIDATE_M}"
require_positive_integer VALIDATE_N "${VALIDATE_N}"
require_positive_integer VALIDATE_K "${VALIDATE_K}"
if [ "${VALIDATE}" != "0" ] && [ "${VALIDATE}" != "1" ]; then
  printf 'ERROR: VALIDATE must be 0 or 1, got: %s\n' "${VALIDATE}" >&2
  exit 1
fi

read -r -a CORE_LIST <<< "${CORES}"
if [ "${#CORE_LIST[@]}" -eq 0 ]; then
  printf 'ERROR: CORES must contain at least one CPU number.\n' >&2
  exit 1
fi
for core in "${CORE_LIST[@]}"; do
  if ! [[ "${core}" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: invalid CPU number in CORES: %s\n' "${core}" >&2
    exit 1
  fi
done

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
  taskset -c "${core}" "$@"
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
          csv_row "${family}" "${kernel}" "${core}" "$(core_domain "${core}")" "0" "${M}" "${N}" "${K}" "" "" "" "BUILD_FAILED" "" "${build_log}" "NOT_RUN"
        done
        exit 0
      fi
      rm -f "/tmp/make_${kernel}_${TS}.log"

      if [ ! -f ./bench ]; then
        echo "[ERROR] bench binary missing: ${kernel}" | tee -a "${LIVE_LOG}"
        for core in ${CORES}; do
          csv_row "${family}" "${kernel}" "${core}" "$(core_domain "${core}")" "0" "${M}" "${N}" "${K}" "" "" "" "NO_BENCH" "" "" "NOT_RUN"
        done
        exit 0
      fi

      kernel_validation="DISABLED"
      if [ "${VALIDATE}" -eq 1 ]; then
        validation_core="${CORE_LIST[0]}"
        validation_domain="$(core_domain "${validation_core}")"
        validation_log="${RAW_DIR}/${kernel}_validation_core${validation_core}_${TS}.log"

        if validation_output="$(run_on_core "${validation_core}" env GEMM_VALIDATE=1 ./bench "${VALIDATE_M}" "${VALIDATE_N}" "${VALIDATE_K}" 2>&1)"; then
          validation_rc=0
        else
          validation_rc=$?
        fi

        {
          echo "=================================================="
          echo "Stage: pre-benchmark numerical validation"
          echo "Family: ${family}"
          echo "Kernel: ${kernel}"
          echo "Core: ${validation_core}"
          echo "Domain: ${validation_domain}"
          echo "Shape: ${VALIDATE_M}x${VALIDATE_N}x${VALIDATE_K}"
          echo "ReturnCode: ${validation_rc}"
          echo "--------------------------------------------------"
          echo "${validation_output}"
        } > "${validation_log}"

        if [ "${validation_rc}" -eq 0 ] && printf '%s\n' "${validation_output}" | grep -q '^VALIDATION=OK '; then
          kernel_validation="OK"
          echo "  Validation: OK (${VALIDATE_M}x${VALIDATE_N}x${VALIDATE_K} on core ${validation_core})" | tee -a "${LIVE_LOG}"
        else
          kernel_validation="FAILED"
          echo "[ERROR] validation failed: ${kernel}" | tee -a "${LIVE_LOG}"
          csv_row "${family}" "${kernel}" "${validation_core}" "${validation_domain}" "0" "${VALIDATE_M}" "${VALIDATE_N}" "${VALIDATE_K}" "" "" "" "VALIDATION_FAILED" "${validation_rc}" "${validation_log}" "${kernel_validation}"
          exit 0
        fi
      fi

      for core in ${CORES}; do
        domain="$(core_domain "${core}")"
        core_log="${RAW_DIR}/${kernel}_core${core}_${domain}_${TS}.log"
        : > "${core_log}"
        echo "  Core ${core} (${domain})" | tee -a "${LIVE_LOG}"

        for run in $(seq 1 "${RUNS}"); do
          if output="$(run_on_core "${core}" env GEMM_VALIDATE=0 ./bench "${M}" "${N}" "${K}" 2>&1)"; then
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

          csv_row "${family}" "${kernel}" "${core}" "${domain}" "${run}" "${M}" "${N}" "${K}" "${metric_name}" "${metric_value}" "${time_sec}" "${status}" "${rc}" "${core_log}" "${kernel_validation}"
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
    print "family","kernel","core","domain","metric","ok_runs","avg_value","min_value","max_value","avg_time_sec","ok_count","failed_count","kernel_return_count","build_failed_count","no_bench_count","median_value","stddev_value","validation_failed_count","validation";
  }
  NR==1 { next }
  function clean(x) { gsub(/^"|"$/, "", x); gsub(/""/, "\"", x); return x }
  {
    family=clean($1); kernel=clean($2); core=clean($3); domain=clean($4);
    metric=clean($9); value=clean($10); time=clean($11); status=clean($12);
    validation=clean($15);
    status_key=family SUBSEP kernel SUBSEP core SUBSEP domain;
    if (metric == "") metric="NA";
    key=status_key SUBSEP metric;
    labels[key]=family OFS kernel OFS core OFS domain OFS metric;
    validation_by_key[key]=validation;
    status_count[key,status]++;
    if (status == "OK" && value != "") {
      v=value+0; t=time+0;
      sum[key]+=v; sumsq[key]+=v*v; timesum[key]+=t; count[key]++;
      values[key,count[key]]=v;
      if (!(key in min) || v < min[key]) min[key]=v;
      if (!(key in max) || v > max[key]) max[key]=v;
    }
  }
  END {
    key_count=0;
    for (key in labels) keys[++key_count]=key;
    for (i=2; i<=key_count; i++) {
      current_key=keys[i]; j=i-1;
      while (j>=1 && keys[j] > current_key) {
        keys[j+1]=keys[j]; j--;
      }
      keys[j+1]=current_key;
    }
    for (key_index=1; key_index<=key_count; key_index++) {
      key=keys[key_index];
      ok=count[key]+0;
      avg=(ok>0 ? sum[key]/ok : "");
      mn=(ok>0 ? min[key] : "");
      mx=(ok>0 ? max[key] : "");
      avgt=(ok>0 ? timesum[key]/ok : "");
      median=""; stddev="";
      for (i in ordered) delete ordered[i];
      if (ok > 0) {
        for (i=1; i<=ok; i++) ordered[i]=values[key,i];
        for (i=2; i<=ok; i++) {
          current=ordered[i]; j=i-1;
          while (j>=1 && ordered[j] > current) {
            ordered[j+1]=ordered[j]; j--;
          }
          ordered[j+1]=current;
        }
        if (ok % 2) median=ordered[(ok+1)/2];
        else median=(ordered[ok/2]+ordered[ok/2+1])/2;
        variance=sumsq[key]/ok-avg*avg;
        if (variance < 0 && variance > -1e-12) variance=0;
        stddev=(variance >= 0 ? sqrt(variance) : "");
      }
      print labels[key], ok, avg, mn, mx, avgt, status_count[key,"OK"]+0, status_count[key,"FAILED"]+0, status_count[key,"KERNEL_RETURN"]+0, status_count[key,"BUILD_FAILED"]+0, status_count[key,"NO_BENCH"]+0, median, stddev, status_count[key,"VALIDATION_FAILED"]+0, validation_by_key[key];
    }
  }
  ' "${RAW_CSV}" > "${SUMMARY_CSV}"
}

echo "Single-core FP32/INT8 RVV 8x8 benchmark sweep" | tee "${LIVE_LOG}"
echo "Generated: $(date)" | tee -a "${LIVE_LOG}"
echo "M=${M} N=${N} K=${K} RUNS=${RUNS}" | tee -a "${LIVE_LOG}"
echo "CORES=${CORES}" | tee -a "${LIVE_LOG}"
echo "Core map: 0-3=RVV_IME, 4-7=RVV" | tee -a "${LIVE_LOG}"
echo "VALIDATE=${VALIDATE} validation_shape=${VALIDATE_M}x${VALIDATE_N}x${VALIDATE_K}" | tee -a "${LIVE_LOG}"
echo "Compiler: $(gcc --version | sed -n '1p')" | tee -a "${LIVE_LOG}"
echo "OUT_DIR=${OUT_DIR}" | tee -a "${LIVE_LOG}"
echo "taskset: required and enabled" | tee -a "${LIVE_LOG}"

echo 'family,kernel,core,domain,run,m,n,k,metric,value,time_sec,status,return_code,log_file,validation' > "${RAW_CSV}"

run_family "FP32_SGEMM" "${FP32_ROOT}" "sgemm_kernel_8x8_zvl256b_lmul*_unroll*"
run_family "INT8_RVV" "${INT8_ROOT}" "igemm_kernel_8x8_zvl256b_lmul*_unroll*"

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
