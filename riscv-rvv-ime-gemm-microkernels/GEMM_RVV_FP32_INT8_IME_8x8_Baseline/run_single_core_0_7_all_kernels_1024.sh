#!/usr/bin/env bash
set -u

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
CORES="${CORES:-0 1 2 3 4 5 6 7}"

FP32_ROOT="${BASE_DIR}/RVV_SGEMM_FP32_8x8"
INT8_ROOT="${BASE_DIR}/RVV_IGEMM_INT8_I8I32_8x8"
IME_ROOT="${BASE_DIR}/IME_GEMM_INT8_I8I32_8x8_RVV_Fallback"

OUT_DIR="${OUT_DIR:-${BASE_DIR}/single_core_results_${M}}"
RAW_LOG_DIR="${OUT_DIR}/raw_logs"
mkdir -p "${RAW_LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
RAW_CSV="${OUT_DIR}/single_core_raw_${M}_runs${RUNS}_${TS}.csv"
SUMMARY_CSV="${OUT_DIR}/single_core_summary_${M}_runs${RUNS}_${TS}.csv"
LIVE_LOG="${OUT_DIR}/single_core_live_${M}_runs${RUNS}_${TS}.log"

printf 'family,kernel,core,domain,run,m,n,k,metric,value,time_sec,status,return_code,log_file\n' > "${RAW_CSV}"

domain_for_core() {
    case "$1" in
        0|1|2|3) printf 'RVV_IME' ;;
        *) printf 'RVV' ;;
    esac
}

run_on_core() {
    local core="$1"
    shift
    if command -v taskset >/dev/null 2>&1; then
        taskset -c "${core}" "$@"
    else
        "$@"
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

parse_metric() {
    sed -nE 's/.*(GFLOPS|GOPS)[[:space:]]*[:=][[:space:]]*([-+0-9.eE]+).*/\1 \2/p' | head -n 1
}

parse_time_sec() {
    sed -nE 's/.*([Tt]ime|time_sec)[[:space:]]*[:=][[:space:]]*([-+0-9.eE]+).*/\2/p' | head -n 1
}

run_family() {
    local family="$1"
    local root="$2"
    local pattern="$3"

    if [ ! -d "${root}" ]; then
        printf '\n[%s] SKIP missing directory: %s\n' "${family}" "${root}" | tee -a "${LIVE_LOG}"
        return 0
    fi

    mapfile -t kernels < <(find "${root}" -maxdepth 1 -mindepth 1 -type d -name "${pattern}" | sort)
    if [ "${#kernels[@]}" -eq 0 ]; then
        printf '\n[%s] SKIP no kernels matching: %s\n' "${family}" "${pattern}" | tee -a "${LIVE_LOG}"
        return 0
    fi

    for dir in "${kernels[@]}"; do
        local kernel
        kernel="$(basename "${dir}")"
        local build_log="${RAW_LOG_DIR}/${family}_${kernel}_build.log"

        printf '\n[%s] %s\n' "${family}" "${kernel}" | tee -a "${LIVE_LOG}"

        if ! (cd "${dir}" && make clean >> "${build_log}" 2>&1 && make >> "${build_log}" 2>&1); then
            printf '  BUILD FAILED: %s\n' "${build_log}" | tee -a "${LIVE_LOG}"
            for core in ${CORES}; do
                local domain
                domain="$(domain_for_core "${core}")"
                csv_row "${family}" "${kernel}" "${core}" "${domain}" "0" "${M}" "${N}" "${K}" "NA" "" "" "BUILD_FAILED" "1" "${build_log}"
            done
            continue
        fi

        if [ ! -x "${dir}/bench" ]; then
            printf '  NO BENCH EXECUTABLE\n' | tee -a "${LIVE_LOG}"
            for core in ${CORES}; do
                local domain
                domain="$(domain_for_core "${core}")"
                csv_row "${family}" "${kernel}" "${core}" "${domain}" "0" "${M}" "${N}" "${K}" "NA" "" "" "NO_BENCH" "1" "${build_log}"
            done
            continue
        fi

        for core in ${CORES}; do
            local domain
            domain="$(domain_for_core "${core}")"
            printf '  Core %s (%s)\n' "${core}" "${domain}" | tee -a "${LIVE_LOG}"

            for run in $(seq 1 "${RUNS}"); do
                local run_log="${RAW_LOG_DIR}/${family}_${kernel}_core${core}_run${run}.log"
                local output rc parsed metric value time_sec status

                output="$(cd "${dir}" && run_on_core "${core}" ./bench "${M}" "${N}" "${K}" 2>&1)"
                rc=$?
                printf '%s\n' "${output}" > "${run_log}"

                parsed="$(printf '%s\n' "${output}" | parse_metric)"
                metric="$(printf '%s' "${parsed}" | awk '{print $1}')"
                value="$(printf '%s' "${parsed}" | awk '{print $2}')"
                time_sec="$(printf '%s\n' "${output}" | parse_time_sec)"

                if [ "${rc}" -eq 0 ] && [ -n "${value}" ]; then
                    status="OK"
                    printf '    Run %s: OK %s=%s time=%s\n' "${run}" "${metric}" "${value}" "${time_sec:-NA}" | tee -a "${LIVE_LOG}"
                else
                    status="KERNEL_RETURN"
                    metric="${metric:-NA}"
                    printf '    Run %s: FAIL rc=%s log=%s\n' "${run}" "${rc}" "${run_log}" | tee -a "${LIVE_LOG}"
                fi

                csv_row "${family}" "${kernel}" "${core}" "${domain}" "${run}" "${M}" "${N}" "${K}" "${metric}" "${value}" "${time_sec}" "${status}" "${rc}" "${run_log}"
            done
        done
    done
}

write_summary() {
    printf 'family,kernel,core,domain,metric,ok_runs,avg_value,min_value,max_value,avg_time_sec,ok_count,failed_count,kernel_return_count,build_failed_count,no_bench_count\n' > "${SUMMARY_CSV}"
    awk -F, '
    function clean(x) { gsub(/^"/, "", x); gsub(/"$/, "", x); gsub(/""/, "\"", x); return x }
    NR == 1 { next }
    {
        family = clean($1); kernel = clean($2); core = clean($3); domain = clean($4);
        metric = clean($9); value = clean($10); time_sec = clean($11); status = clean($12);
        key = family SUBSEP kernel SUBSEP core SUBSEP domain SUBSEP metric;
        seen[key] = 1;
        fam[key] = family; ker[key] = kernel; cor[key] = core; dom[key] = domain; met[key] = metric;
        if (status == "OK") {
            ok[key]++;
            sum[key] += value + 0;
            tsum[key] += time_sec + 0;
            if (!(key in minv) || value + 0 < minv[key]) minv[key] = value + 0;
            if (!(key in maxv) || value + 0 > maxv[key]) maxv[key] = value + 0;
        } else if (status == "BUILD_FAILED") {
            build_failed[key]++;
        } else if (status == "NO_BENCH") {
            no_bench[key]++;
        } else if (status == "KERNEL_RETURN") {
            kernel_return[key]++;
        } else {
            failed[key]++;
        }
    }
    END {
        for (key in seen) {
            avg = (ok[key] > 0) ? sum[key] / ok[key] : "";
            tavg = (ok[key] > 0) ? tsum[key] / ok[key] : "";
            minout = (ok[key] > 0) ? minv[key] : "";
            maxout = (ok[key] > 0) ? maxv[key] : "";
            total_failed = failed[key] + kernel_return[key] + build_failed[key] + no_bench[key];
            printf "%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%d,%d,%d,%d,%d\n", fam[key], ker[key], cor[key], dom[key], met[key], ok[key]+0, avg, minout, maxout, tavg, ok[key]+0, total_failed+0, kernel_return[key]+0, build_failed[key]+0, no_bench[key]+0;
        }
    }' "${RAW_CSV}" | sort >> "${SUMMARY_CSV}"
}
printf 'FP32/INT8/IME 8x8 single-core benchmark\n' | tee "${LIVE_LOG}"
printf 'Matrix size: M=%s N=%s K=%s, runs=%s, cores=%s\n' "${M}" "${N}" "${K}" "${RUNS}" "${CORES}" | tee -a "${LIVE_LOG}"
printf 'Output: %s\n' "${OUT_DIR}" | tee -a "${LIVE_LOG}"

run_family "FP32_SGEMM" "${FP32_ROOT}" "sgemm_kernel_8x8_zvl128b_lmul*_unroll*"
run_family "INT8_RVV" "${INT8_ROOT}" "igemm_kernel_8x8_zvl128b_lmul*_unroll*"
run_family "IME_INT8_RVV_FALLBACK" "${IME_ROOT}" "ime_kernel_8x8_zvl128b_lmul*_unroll*"

write_summary

ln -sf "$(basename "${RAW_CSV}")" "${OUT_DIR}/single_core_raw_latest.csv"
ln -sf "$(basename "${SUMMARY_CSV}")" "${OUT_DIR}/single_core_summary_latest.csv"

printf '\nDONE\n' | tee -a "${LIVE_LOG}"
printf 'Raw CSV: %s\n' "${RAW_CSV}" | tee -a "${LIVE_LOG}"
printf 'Summary CSV: %s\n' "${SUMMARY_CSV}" | tee -a "${LIVE_LOG}"
printf 'Latest raw CSV: %s\n' "${OUT_DIR}/single_core_raw_latest.csv" | tee -a "${LIVE_LOG}"
printf 'Latest summary CSV: %s\n' "${OUT_DIR}/single_core_summary_latest.csv" | tee -a "${LIVE_LOG}"
printf 'Raw logs: %s\n' "${RAW_LOG_DIR}" | tee -a "${LIVE_LOG}"