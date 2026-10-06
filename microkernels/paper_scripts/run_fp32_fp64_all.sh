#!/usr/bin/env bash
set -u -o pipefail

# Sweep all available FP32 SGEMM and FP64 DGEMM RVV kernels.
# Raw rows contain separate time_sec and gflops fields.
# Summary contains mean, median, and sample standard deviation for both.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
M="${M:-1024}"; N="${N:-1024}"; K="${K:-1024}"
RUNS="${RUNS:-7}"; PIN_CORE="${PIN_CORE:-0}"; VALIDATE="${VALIDATE:-1}"
VALIDATE_M="${VALIDATE_M:-15}"; VALIDATE_N="${VALIDATE_N:-7}"; VALIDATE_K="${VALIDATE_K:-13}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/benchmarking/results/fp32_fp64_all_${TS}}"
RAW_DIR="${OUT_DIR}/raw_logs"; RAW_CSV="${OUT_DIR}/fp32_fp64_raw.csv"
SUMMARY_CSV="${OUT_DIR}/fp32_fp64_summary.csv"; COVERAGE_CSV="${OUT_DIR}/fp32_fp64_coverage.csv"
LIVE_LOG="${OUT_DIR}/fp32_fp64_live.log"
LMULS=(mf8 mf4 mf2 1 2 4 8); UNROLLS=(1 2 4 8); TILES=(8x4 8x8)

mkdir -p "${RAW_DIR}"
for tool in awk date make mkdir sort; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "ERROR: missing command: ${tool}" >&2; exit 1; }
done
[[ "${M}" =~ ^[1-9][0-9]*$ && "${N}" =~ ^[1-9][0-9]*$ && "${K}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: dimensions must be positive integers" >&2; exit 1; }
[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: RUNS must be a positive integer" >&2; exit 1; }
[[ "${VALIDATE}" = 0 || "${VALIDATE}" = 1 ]] || { echo "ERROR: VALIDATE must be 0 or 1" >&2; exit 1; }
if [[ "${PIN_CORE}" != none ]]; then command -v taskset >/dev/null 2>&1 || { echo "ERROR: taskset is required" >&2; exit 1; }; fi

printf 'precision,tile,lmul,unroll,kernel,source_status,source_dir\n' > "${COVERAGE_CSV}"
printf 'precision,tile,lmul,unroll,kernel,run,status,time_sec,gflops,validation,log_file\n' > "${RAW_CSV}"
printf 'FP32/FP64 RVV sweep\nworkload=${M}x${N}x${K} runs=${RUNS} pin_core=${PIN_CORE} validation=${VALIDATE}\ntime_sec and gflops are separate; summary uses sample SD.\n' | tee "${LIVE_LOG}"

csv_row() { printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$@" >> "${RAW_CSV}"; }
run_bench() {
  local v="$1" m="$2" n="$3" k="$4"
  if [[ "${PIN_CORE}" = none ]]; then env GEMM_VALIDATE="${v}" ./bench "${m}" "${n}" "${k}"
  else taskset -c "${PIN_CORE}" env GEMM_VALIDATE="${v}" ./bench "${m}" "${n}" "${k}"; fi
}
parse_time() { awk '/^Time:/ {print $2; exit}'; }
parse_gflops() { awk -F': ' '/^GFLOPS:/ {print $2; exit}'; }

run_case() {
  local precision="$1" tile="$2" lmul="$3" unroll="$4" kernel="$5" dir="$6"
  local stem="${precision}_${tile}_lmul${lmul}_U${unroll}" log="${RAW_DIR}/${stem}.log"
  local build="${RAW_DIR}/${stem}.build.log" out rc t g v run status
  printf '[%s %s LMUL=%s U%s] build\n' "${precision}" "${tile}" "${lmul}" "${unroll}" | tee -a "${LIVE_LOG}"
  if ! (cd "${dir}" && { make clean >/dev/null 2>&1 || true; make; }) > "${build}" 2>&1; then
    printf 'BUILD_FAILED\n' > "${log}"
    csv_row "${precision}" "${tile}" "${lmul}" "${unroll}" "${kernel}" 0 BUILD_FAILED "" "" NOT_RUN "${build}"
    printf '  BUILD_FAILED\n' | tee -a "${LIVE_LOG}"; return
  fi
  if [[ ! -f "${dir}/bench" ]]; then
    printf 'NO_BENCH\n' > "${log}"
    csv_row "${precision}" "${tile}" "${lmul}" "${unroll}" "${kernel}" 0 NO_BENCH "" "" NOT_RUN "${log}"
    printf '  NO_BENCH\n' | tee -a "${LIVE_LOG}"; return
  fi
  if [[ "${VALIDATE}" = 1 ]]; then
    if out="$(cd "${dir}" && run_bench 1 "${VALIDATE_M}" "${VALIDATE_N}" "${VALIDATE_K}" 2>&1)"; then rc=0; else rc=$?; fi
    printf 'validation rc=%s shape=%sx%sx%s\n%s\n' "${rc}" "${VALIDATE_M}" "${VALIDATE_N}" "${VALIDATE_K}" "${out}" > "${log}"
    if [[ "${rc}" -ne 0 || "${out}" != *VALIDATION=OK* ]]; then
      csv_row "${precision}" "${tile}" "${lmul}" "${unroll}" "${kernel}" 0 VALIDATION_FAILED "" "" NOT_OK "${log}"
      printf '  VALIDATION_FAILED\n' | tee -a "${LIVE_LOG}"; return
    fi
  fi
  : > "${log}"
  for ((run=1; run<=RUNS; run++)); do
    if out="$(cd "${dir}" && run_bench 0 "${M}" "${N}" "${K}" 2>&1)"; then rc=0; else rc=$?; fi
    printf 'run=%s rc=%s\n%s\n' "${run}" "${rc}" "${out}" >> "${log}"
    t="$(printf '%s\n' "${out}" | parse_time)"; g="$(printf '%s\n' "${out}" | parse_gflops)"
    v="$(printf '%s\n' "${out}" | awk -F' ' '/^VALIDATION=/ {print $1; exit}')"
    if [[ "${rc}" -eq 0 && -n "${t}" && -n "${g}" && "${out}" == *KERNEL_RETURN=0* ]]; then status=OK; else status=FAILED; fi
    csv_row "${precision}" "${tile}" "${lmul}" "${unroll}" "${kernel}" "${run}" "${status}" "${t}" "${g}" "${v:-NOT_RUN}" "${log}"
    printf '  run=%s status=%s time_sec=%s gflops=%s\n' "${run}" "${status}" "${t:-NA}" "${g:-NA}" | tee -a "${LIVE_LOG}"
  done
}

available=0; missing=0
for precision in FP32 FP64; do
  if [[ "${precision}" = FP32 ]]; then family=RVV_SGEMM_FP32; prefix=sgemm; else family=RVV_DGEMM_FP64; prefix=dgemm; fi
  for tile in "${TILES[@]}"; do
    root="${PROJECT_ROOT}/GEMM_RVV_${precision}_INT8_${tile}_Baseline/${family}_${tile}"
    for lmul in "${LMULS[@]}"; do
      for unroll in "${UNROLLS[@]}"; do
        kernel="${prefix}_kernel_${tile}_zvl256b_lmul${lmul}_unroll${unroll}"
        dir="${root}/${kernel}"
        if [[ -d "${dir}" && -f "${dir}/Makefile" ]]; then
          printf '%s,%s,%s,%s,%s,AVAILABLE,%s\n' "${precision}" "${tile}" "${lmul}" "${unroll}" "${kernel}" "${dir}" >> "${COVERAGE_CSV}"
          ((available+=1))
          run_case "${precision}" "${tile}" "${lmul}" "${unroll}" "${kernel}" "${dir}"
        else
          printf '%s,%s,%s,%s,%s,MISSING,%s\n' "${precision}" "${tile}" "${lmul}" "${unroll}" "${kernel}" "${dir}" >> "${COVERAGE_CSV}"
          ((missing+=1)); printf '[%s %s LMUL=%s U%s] MISSING\n' "${precision}" "${tile}" "${lmul}" "${unroll}" | tee -a "${LIVE_LOG}"
        fi
      done
    done
  done
done

TMP="${OUT_DIR}/.summary.tmp"
awk -F',' -v runs="${RUNS}" '
BEGIN { OFS="," }
NR==1 { next }
{
  k=$1 SUBSEP $2 SUBSEP $3 SUBSEP $4 SUBSEP $5; label[k]=$1 OFS $2 OFS $3 OFS $4 OFS $5
  total[k]++
  if ($7=="OK" && $8!="" && $9!="") { n[k]++; t[k,n[k]]=$8+0; g[k,n[k]]=$9+0; st[k]+=$8; ss[k]+=$8*$8; sg[k]+=$9; ssg[k]+=$9*$9 }
}
END {
  for (k in label) {
    c=n[k]+0; f=total[k]-c
    if (c>0) {
      mt=st[k]/c; mg=sg[k]/c
      for(i=1;i<=c;i++){ot[i]=t[k,i];og[i]=g[k,i]}
      for(i=2;i<=c;i++){vt=ot[i];j=i-1;while(j>=1&&ot[j]>vt){ot[j+1]=ot[j];j--}ot[j+1]=vt;vg=og[i];j=i-1;while(j>=1&&og[j]>vg){og[j+1]=og[j];j--}og[j+1]=vg}
      if(c%2){medt=ot[(c+1)/2];medg=og[(c+1)/2]}else{medt=(ot[c/2]+ot[c/2+1])/2;medg=(og[c/2]+og[c/2+1])/2}
      if(c>1){sdt=sqrt((ss[k]-st[k]*st[k]/c)/(c-1));sdg=sqrt((ssg[k]-sg[k]*sg[k]/c)/(c-1))}else{sdt="NA";sdg="NA"}
      state=(c==runs?"OK":"INCOMPLETE")
      printf "%s,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%s\n",label[k],c,f,mt,medt,sdt,mg,medg,sdg,state
    } else printf "%s,0,%d,NA,NA,NA,NA,NA,NA,FAILED\n",label[k],f
    delete ot; delete og
  }
}' "${RAW_CSV}" > "${TMP}"
{
  printf 'precision,tile,lmul,unroll,kernel,ok_runs,failed_records,mean_time_sec,median_time_sec,sample_sd_time_sec,mean_gflops,median_gflops,sample_sd_gflops,status\n'
  sort -t, -k1,1 -k2,2 -k3,3 -k4,4n "${TMP}"
} > "${SUMMARY_CSV}"
rm -f "${TMP}"

printf '\nDONE\nExpected combinations: 112\nAvailable source combinations: %s\nMissing source combinations: %s\nRaw CSV: %s\nSummary CSV: %s\nCoverage CSV: %s\nLive log: %s\n' "${available}" "${missing}" "${RAW_CSV}" "${SUMMARY_CSV}" "${COVERAGE_CSV}" "${LIVE_LOG}" | tee -a "${LIVE_LOG}"
