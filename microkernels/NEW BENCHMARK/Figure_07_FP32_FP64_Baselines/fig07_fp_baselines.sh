#!/usr/bin/env bash
# Existing, unmodified standalone kernels: one timed call per fresh process.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd -- "$SCRIPT_DIR/.." && pwd)/common.sh"

usage() {
    cat <<'EOF'
Usage: bash fig07_fp_baselines.sh [--check] [--campaign NAME]
Collect all 36 FP32 and 28 FP64 configurations on one fixed RVV CPU.
Environment: M/N/K=1024, RUNS=7, WARMUPS=2, FP_CPU=first RVV_CPU.
--check reads source files only; it does not build, run, or create output.
The discarded WARMUPS are separate process launches, NOT in-process warm-ups.
Each configuration passes aligned, tail and full-workload validation gates.
Measured fresh processes reuse identical deterministic inputs without repeating
the scalar reference. Input generation and allocation are outside the timer.
EOF
}
CHECK=0
CAMPAIGN_ARG=""
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --check) CHECK=1; shift ;;
        --campaign) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; CAMPAIGN_ARG="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

inventory_fp() {
    local precision tile lmul u prefix family
    for precision in FP32 FP64; do
        [[ "$precision" == FP32 ]] && prefix=sgemm || prefix=dgemm
        for tile in 8x4 8x8; do
            if [[ "$precision" == FP32 && "$tile" == 8x4 ]]; then
                local lmuls=(mf2 1 2 4 8)
            elif [[ "$precision" == FP64 && "$tile" == 8x4 ]]; then
                local lmuls=(2 4 8)
            else
                local lmuls=(1 2 4 8)
            fi
            for lmul in "${lmuls[@]}"; do
                for u in 1 2 4 8; do
                    printf '%s|%s|%s|%s|%s_kernel_%s_zvl256b_lmul%s_unroll%s\n' \
                        "$precision" "$tile" "$lmul" "$u" "$prefix" "$tile" "$lmul" "$u"
                done
            done
        done
    done
}

resolve_fp() {
    local precision="$1" tile="$2" kernel="$3" prefix family
    if [[ "$precision" == FP32 ]]; then prefix=sgemm; family=SGEMM; else prefix=dgemm; family=DGEMM; fi
    FP_SOURCE_DIR="$PROJECT_ROOT/GEMM_RVV_${precision}_INT8_${tile}_Baseline/RVV_${family}_${precision}_${tile}/$kernel"
    FP_SOURCE="$FP_SOURCE_DIR/$kernel.c"
    FP_HARNESS="$FP_SOURCE_DIR/${prefix}_bench.c"
    [[ -f "$FP_SOURCE" && -f "$FP_HARNESS" ]] || return 1
    # The real standalone harness must contain validation and its own timer.
    grep -q 'GEMM_VALIDATE' "$FP_HARNESS" && grep -q 'CLOCK_MONOTONIC' "$FP_HARNESS" && \
        grep -Fq 'fill(A, sizeA);' "$FP_HARNESS" && grep -Fq 'fill(B, sizeB);' "$FP_HARNESS" && \
        grep -Fq 'fill(C, sizeC);' "$FP_HARNESS" && grep -Fq 'i % 13' "$FP_HARNESS"
}

if ((CHECK)); then
    missing=0; count=0
    while IFS='|' read -r precision tile lmul u kernel; do
        count=$((count+1))
        if resolve_fp "$precision" "$tile" "$kernel"; then
            printf 'OK %s %s\n' "$precision" "$FP_SOURCE"
        else
            printf 'MISSING_OR_UNSUPPORTED %s %s\n' "$precision" "$kernel" >&2
            missing=$((missing+1))
        fi
    done < <(inventory_fp)
    printf 'Source configurations: %d (expected 64); unresolved: %d\n' "$count" "$missing"
    [[ "$count" -eq 64 && "$missing" -eq 0 ]]
    exit
fi

[[ -z "$CAMPAIGN_ARG" ]] || CAMPAIGN="$CAMPAIGN_ARG"
init_figure fig07_fp_baselines
preflight_once
M="${M:-1024}"; N="${N:-1024}"; K="${K:-1024}"
for value in "$M" "$N" "$K"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'M/N/K must be positive integers.\n' >&2; exit 2; }
done
FP_CPU="${FP_CPU:-$(cpu_first "$RVV_CPUS")}" 
[[ "$FP_CPU" =~ ^[0-9]+$ ]] || { printf 'FP_CPU must be one integer CPU ID.\n' >&2; exit 2; }
[[ ",$RVV_CPUS," == *",$FP_CPU,"* ]] || { printf 'FP_CPU must belong to the preflight-validated RVV_CPUS list.\n' >&2; exit 2; }
taskset -c "$FP_CPU" true
mkdir -p "$OUT/build" "$OUT/raw" "$OUT/metadata" "$OUT/summary"
snapshot_environment "$OUT/metadata/environment.txt"
cat > "$OUT/metadata/protocol.txt" <<EOF
timing_scope=standalone_single_kernel_call
input_packing_included=no (arrays are generated directly in the kernel packed layout)
output_update_included=yes (fused into kernel)
allocation_reference_validation_included=no
cpu=$FP_CPU
dimensions=$M,$N,$K
input_pattern=existing deterministic modulo-13 floating-point harness pattern; user SEED is not used by this harness
alpha=1; initial_C=existing nonzero deterministic pattern; ldc=M
discarded_process_launches=$WARMUPS
in_process_warmups=none (the existing standalone program has no repetition loop)
measured_process_launches=$RUNS
validation=GEMM_VALIDATE=1 in aligned, combined-tail and full-workload gates before timing
measured_processes=GEMM_VALIDATE=0; validation_not_repeated_each_process; inputs and kernel call are unchanged
validation_proof=per-configuration full-gate log hash attached to every measured row
FP32_tolerance=1e-5*(1+abs(reference)); FP64_tolerance=1e-11*(1+abs(reference))
time_precision=6 decimal places (existing harness output; do not invent extra precision)
quartiles=type-7 linear interpolation; outlier_rule=outside Q1-1.5*IQR or Q3+1.5*IQR
reference_FP64_8x4=LMUL2_U1; other panels=lowest supported LMUL_U1
coverage=36 FP32 + 28 FP64 configurations; no sample pooling across configurations/CPUs
EOF
printf 'precision,tile,lmul,u,kernel,cpu,m,n,k,rep,time_sec,reported_gflops,validation_gate_max_abs_error,full_validation_log_sha256\n' > "$OUT/raw/measurements.csv"
cp "$OUT/raw/measurements.csv" "$OUT/summary/accepted_measurements.csv"
printf 'precision,tile,lmul,u,kernel,status,accepted_repetitions,expected_repetitions\n' > "$OUT/summary/coverage.csv"
printf 'precision,kernel,kernel_sha256,harness_sha256,binary_sha256\n' > "$OUT/metadata/source_hashes.csv"
printf 'precision,kernel,validation_shape,validation_log_sha256,max_abs_error\n' > "$OUT/metadata/validation_gates.csv"
bad=0

# Parse exactly one timed call; gated numerical validation and timing-only logs
# have distinct expected validation states and are never presented as equivalent.
parse_fp_log() {
    awk -v gate="${2:-0}" '
        /^KERNEL_RETURN=/ { rc=substr($0,index($0,"=")+1); nr++ }
        /^Time: / { time=$2; nt++ }
        /^GFLOPS: / { rate=$2; ng++ }
        /^VALIDATION=OK / {
            nv++; for(i=1;i<=NF;i++) {
                if($i ~ /^mismatches=/) { split($i,a,"="); mismatch=a[2] }
                if($i ~ /^max_abs_error=/) { split($i,a,"="); error=a[2] }
            }
        }
        /^VALIDATION=DISABLED$/ {nd++}
        END {
            numeric="^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$"
            if(nr!=1 || rc!="0" || nt!=1 || ng!=1 || time !~ numeric || rate !~ numeric || rate+0<=0) exit 1
            if(gate) {
                if(nv!=1 || nd || mismatch!="0" || error !~ numeric || time+0<0)exit 1
            } else {
                if(nd!=1 || nv || time+0<=0)exit 1
                error="NA"
            }
            printf "%s,%s,%s\n",time,rate,error
        }' "$1"
}

while IFS='|' read -r precision tile lmul u kernel; do
    case_id="${precision}_${kernel}"
    case_dir="$OUT/raw/$case_id"
    mkdir -p "$case_dir"
    valid=1; accepted=0
    if ! resolve_fp "$precision" "$tile" "$kernel"; then
        record_failure "$case_id" 'Missing kernel/harness or unsupported validation interface' SOURCE_FAILED
        valid=0
    fi
    exe="$OUT/build/$case_id"
    if ((valid)) && ! log_command "$case_dir/build.log" "$CC" "${CFLAGS_ARRAY[@]}" \
            "-DCNAME=$kernel" "$FP_SOURCE" "$FP_HARNESS" -no-pie -lm -o "$exe"; then
        record_failure "$case_id" 'Compilation failed; see build.log' BUILD_FAILED
        valid=0
    fi
    if ((valid)); then
        printf '%s,%s,%s,%s,%s\n' "$precision" "$kernel" \
            "$(sha256sum "$FP_SOURCE" | awk '{print $1}')" \
            "$(sha256sum "$FP_HARNESS" | awk '{print $1}')" \
            "$(sha256sum "$exe" | awk '{print $1}')" >> "$OUT/metadata/source_hashes.csv"
        : > "$case_dir/accepted_rows.csv"
        full_validation_hash=NA; full_validation_error=NA
        for gate_shape in 16x16x64 15x15x69 "${M}x${N}x${K}"; do
            IFS=x read -r gate_m gate_n gate_k <<< "$gate_shape"
            logfile="$case_dir/validation_${gate_shape}.log"
            # If the requested full shape equals a small gate, validate it once.
            [[ ! -e "$logfile" ]] || continue
            if log_command "$logfile" taskset -c "$FP_CPU" env GEMM_VALIDATE=1 \
                    "$exe" "$gate_m" "$gate_n" "$gate_k" && parsed="$(parse_fp_log "$logfile" 1)"; then
                gate_hash="$(sha256sum "$logfile" | awk '{print $1}')"
                gate_error="${parsed##*,}"
                printf '%s,%s,%s,%s,%s\n' "$precision" "$kernel" "$gate_shape" "$gate_hash" "$gate_error" \
                    >> "$OUT/metadata/validation_gates.csv"
            else
                record_failure "${case_id}_gate_${gate_shape}" 'Pre-timing numerical validation gate failed' VALIDATION_FAILED
                valid=0; break
            fi
        done
        if ((valid)); then
            full_log="$case_dir/validation_${M}x${N}x${K}.log"
            full_validation_hash="$(sha256sum "$full_log" | awk '{print $1}')"
            parsed="$(parse_fp_log "$full_log" 1)"; full_validation_error="${parsed##*,}"
        fi
        for ((rep=1; rep<=WARMUPS; rep++)); do
            ((valid)) || break
            logfile="$case_dir/discarded_process_${rep}.log"
            if ! log_command "$logfile" taskset -c "$FP_CPU" env GEMM_VALIDATE=0 \
                    "$exe" "$M" "$N" "$K" || ! parse_fp_log "$logfile" > /dev/null; then
                record_failure "${case_id}_discarded_${rep}" 'Discarded launch failed validation or timing parsing' VALIDATION_FAILED
                valid=0
                break
            fi
        done
        if ((valid)); then
            snapshot_environment "$case_dir/environment_before_measurement.txt"
            for ((rep=1; rep<=RUNS; rep++)); do
                logfile="$case_dir/repetition_${rep}.log"
                if log_command "$logfile" taskset -c "$FP_CPU" env GEMM_VALIDATE=0 \
                        "$exe" "$M" "$N" "$K" && parsed="$(parse_fp_log "$logfile")"; then
                    parsed="${parsed%,*},$full_validation_error,$full_validation_hash"
                    row="$precision,$tile,$lmul,$u,$kernel,$FP_CPU,$M,$N,$K,$rep,$parsed"
                    printf '%s\n' "$row" >> "$OUT/raw/measurements.csv"
                    printf '%s\n' "$row" >> "$case_dir/accepted_rows.csv"
                    accepted=$((accepted+1))
                else
                    record_failure "${case_id}_rep_${rep}" 'Measured launch failed numerical validation or parsing' VALIDATION_FAILED
                    valid=0
                fi
            done
            snapshot_environment "$case_dir/environment_after_measurement.txt"
        fi
    fi
    if ((valid)) && [[ "$accepted" -eq "$RUNS" ]]; then
        cat "$case_dir/accepted_rows.csv" >> "$OUT/summary/accepted_measurements.csv"
        status=COMPLETE
    else
        status=REJECTED; bad=$((bad+1))
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$precision" "$tile" "$lmul" "$u" "$kernel" \
        "$status" "$accepted" "$RUNS" >> "$OUT/summary/coverage.csv"
done < <(inventory_fp)

# Sort by exact kernel then time; retain every valid slow sample, including outliers.
{ head -n 1 "$OUT/summary/accepted_measurements.csv"; \
  tail -n +2 "$OUT/summary/accepted_measurements.csv" | sort -t, -k5,5 -k11,11g; } > "$OUT/summary/sorted_measurements.csv"
printf 'precision,tile,lmul,u,kernel,cpu,m,n,k,n,mean_sec,sd_sec,min_sec,q1_sec,median_sec,q3_sec,max_sec,iqr_sec,mean_reported_gflops,outliers\n' > "$OUT/summary/statistics.csv"
printf 'precision,kernel,rep,time_sec,classification\n' > "$OUT/summary/outliers.csv"
awk -F, -v stats="$OUT/summary/statistics.csv" -v outliers="$OUT/summary/outliers.csv" '
    function q(p, h,j) { h=1+(n-1)*p; j=int(h); return j>=n?t[n]:t[j]+(h-j)*(t[j+1]-t[j]) }
    function flush( i,mean,sd,a,b,iqr,out,diff) {
        if(!n) return
        mean=sum/n; sd=0; for(i=1;i<=n;i++){diff=t[i]-mean;sd+=diff*diff}
        sd=n>1?sqrt(sd/(n-1)):0; a=q(.25); b=q(.75); iqr=b-a; out=0
        for(i=1;i<=n;i++) if(t[i]<a-1.5*iqr || t[i]>b+1.5*iqr) {
            out++; printf "%s,%s,%s,%.9g,%s\n",precision,kernel,rep[i],t[i],t[i]<a?"low":"high" >> outliers
        }
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n", \
            precision,tile,lmul,u,kernel,cpu,m,nn,k,n,mean,sd,t[1],a,q(.5),b,t[n],iqr,rate/n,out >> stats
        n=0;sum=0;rate=0
    }
    NR==1 {next}
    $5!=kernel { flush(); precision=$1;tile=$2;lmul=$3;u=$4;kernel=$5;cpu=$6;m=$7;nn=$8;k=$9 }
    {n++;t[n]=$11+0;rep[n]=$10;sum+=t[n];rate+=$12+0}
    END{flush()}' "$OUT/summary/sorted_measurements.csv"

printf 'precision,tile,kernel,reference_kernel,mean_time_speedup_vs_reference\n' > "$OUT/summary/relative_to_reference.csv"
awk -F, '
    NR==FNR {
        if(FNR>1 && $4==1 && (($1=="FP32" && $2=="8x4" && $3=="mf2") ||
            ($1=="FP64" && $2=="8x4" && $3==2) || ($2=="8x8" && $3==1))) {
            ref[$1 SUBSEP $2]=$11; name[$1 SUBSEP $2]=$5
        } next
    }
    FNR>1 {id=$1 SUBSEP $2;if(id in ref && $11>0)printf "%s,%s,%s,%s,%.9g\n",$1,$2,$5,name[id],ref[id]/$11}
' "$OUT/summary/statistics.csv" "$OUT/summary/statistics.csv" >> "$OUT/summary/relative_to_reference.csv"
finish_figure
[[ "$bad" -eq 0 ]]
