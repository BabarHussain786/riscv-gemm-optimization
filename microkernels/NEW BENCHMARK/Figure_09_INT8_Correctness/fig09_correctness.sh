#!/usr/bin/env bash
# Validate canonical RVV kernels and forced-native IME wrappers, not RVV fallback proxies.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd -- "$SCRIPT_DIR/.." && pwd)/common.sh"

usage() {
    cat <<'EOF'
Usage: bash fig09_correctness.sh [--check] [--smoke] [--campaign NAME]
Default: all 36 RVV + 8 IME kernels, 5 shapes, 4 input classes, 3 seeds.
Also classify tiny native-IME shapes separately and validate selected Figure 6
kernels on M=N=K=1024 using the common driver's four execution modes.
--smoke: one aligned shape, bounded_uniform, one seed, all 44 kernels;
         full selected-mode validation is intentionally deferred.
--check: read-only source/interface checks; no build, run or output creation.
Environment: ACC_SEEDS="42 43 44", ACC_SHAPES="16x16x64 15x15x69
17x33x65 24x40x128 32x32x256", M/N/K=1024 for selected full validation.
Alpha=1, initial C=0, ldc=M in standalone checker. It is NOT general-alpha or
padded-ldc validation. The selected common driver adds nonzero initial-C tests.
EOF
}
CHECK=0; SMOKE=0; CAMPAIGN_ARG=""
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --check) CHECK=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --campaign) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; CAMPAIGN_ARG="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
HARNESS="$PROJECT_ROOT/RVV_IME_GEMM_ACCURACY_VALIDATION/ime_rvv_fallback_diff_histogram_check.c"

# All current public IME entry points consume full A column-major / B row-major matrices.
# The old 8x8 accuracy launcher assumed PREPACKED_ROWS; do NOT reuse that assumption.
check_ime_contract() {
    local src="$1"
    grep -Fq 'A[k * M + row], B[k * N + column]' "$src" &&
        grep -Eq 'SPACEMIT_IME_REQUIRE_HARDWARE' "$src"
}

check_inventory() {
    local count=0 missing=0 kernel
    [[ -f "$HARNESS" ]] || { printf 'Missing checker: %s\n' "$HARNESS" >&2; return 1; }
    grep -q 'ACC_KIND_INT8_RVV' "$HARNESS" && grep -q 'ACC_IME_INPUT_FULL_MATRIX' "$HARNESS" || return 1
    while IFS= read -r kernel; do
        count=$((count+1))
        if resolve_rvv "$kernel"; then printf 'OK RVV %s\n' "$RESOLVED_SOURCE";
        else printf 'MISSING RVV %s\n' "$kernel" >&2; missing=$((missing+1)); fi
    done < <(inventory_rvv)
    while IFS= read -r kernel; do
        count=$((count+1))
        if resolve_ime "$kernel" && check_ime_contract "$RESOLVED_SOURCE" && \
                [[ -f "$(dirname -- "$RESOLVED_SOURCE")/rvv_fallback.c" ]]; then
            printf 'OK IME FULL_MATRIX %s\n' "$RESOLVED_SOURCE"
        else printf 'MISSING_OR_CHANGED_CONTRACT IME %s\n' "$kernel" >&2; missing=$((missing+1)); fi
    done < <(inventory_ime)
    printf 'Source configurations: %d (expected 44); unresolved: %d\n' "$count" "$missing"
    [[ "$count" -eq 44 && "$missing" -eq 0 ]]
}
if ((CHECK)); then check_inventory; exit; fi

[[ -z "$CAMPAIGN_ARG" ]] || CAMPAIGN="$CAMPAIGN_ARG"
init_figure fig09_correctness
preflight_once
check_inventory > "$OUT/source_inventory.txt"
mkdir -p "$OUT/build" "$OUT/raw" "$OUT/metadata" "$OUT/summary"
snapshot_environment "$OUT/metadata/environment.txt"
read -r -a CLASSES <<< 'bounded_uniform full_range_uniform mixed_magnitude cancellation_stress'
read -r -a SHAPES <<< "${ACC_SHAPES:-16x16x64 15x15x69 17x33x65 24x40x128 32x32x256}"
read -r -a SEEDS <<< "${ACC_SEEDS:-$SEED $((SEED+1)) $((SEED+2))}"
if ((SMOKE)); then CLASSES=(bounded_uniform); SHAPES=(16x16x64); SEEDS=("$SEED"); fi
declare -A seen_shapes=() seen_seeds=()
for shape in "${SHAPES[@]}"; do
    [[ "$shape" =~ ^[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*$ ]] || { printf 'Invalid shape: %s\n' "$shape" >&2; exit 2; }
    [[ ! "${seen_shapes[$shape]+present}" ]] || { printf 'Duplicate shape: %s\n' "$shape" >&2; exit 2; }
    seen_shapes[$shape]=1
done
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^(0|[1-9][0-9]*)$ && "${#seed}" -le 10 ]] && ((10#$seed<=4294967295)) || \
        { printf 'Accuracy seeds must be canonical decimal uint32 values: %s\n' "$seed" >&2; exit 2; }
    [[ ! "${seen_seeds[$seed]+present}" ]] || { printf 'Duplicate seed: %s\n' "$seed" >&2; exit 2; }
    seen_seeds[$seed]=1
done
RVV_CPU="$(cpu_first "$RVV_CPUS")"; IME_CPU="$(cpu_first "$IME_CPUS")"
taskset -c "$RVV_CPU" true; taskset -c "$IME_CPU" true
cat > "$OUT/metadata/protocol.txt" <<EOF
purpose=correctness_only; execution time is not a performance result
kernel_coverage=36 canonical RVV + 8 supported IME LMUL1 kernels
input_classes=${CLASSES[*]}
seeds=${SEEDS[*]}
shapes=${SHAPES[*]}
smoke_mode=$SMOKE
RVV_CPU=$RVV_CPU; IME_CPU=$IME_CPU
standalone_checker=INT64 dot products with INT32 reference storage; require overflow_count=0
standalone_alpha=1; initial_C=0; ldc=M
RVV_inputs=kernel packed panels; IME_inputs=full A column-major MxK and B row-major KxN matrices for both tile shapes
IME_execution=SPACEMIT_IME_REQUIRE_HARDWARE=1 and SPACEMIT_IME_FORCE_NATIVE=1
IME_tails=scalar boundary handling within a successful native wrapper call
tiny_case_policy=expected unsupported iff M<MR or N<NR or K<8*U on K1 A60
acceptance=exit0 AND status OK AND return0 AND mismatch0 AND maxdifference0 AND overflow0 AND exactmatch1
histogram_acceptance=all M*N differences are zero; A/B histograms total M*K and N*K
full_selected_validation=common driver; alpha1; nonzero initial C; four modes; independent INT64 reference
historical_72_RVV_and_8_IME_counts=not assumed or imported; denominators derived from this campaign
EOF
printf 'backend,tile,lmul,u,kernel,cpu,m,n,k,input_class,seed,case_kind,status,return_code,total_elements,mismatch_count,max_integer_difference,overflow_count,exact_match_rate,process_exit,case_id\n' > "$OUT/raw/cases.csv"
printf 'backend,kernel,kernel_sha256,checker_sha256,fallback_sha256,binary_sha256\n' > "$OUT/metadata/source_hashes.csv"
bad=0

parse_checker_record() {
    awk -F, '
        /^status,return_code,total_elements,mismatch_count,max_integer_difference,overflow_count,exact_match_rate,/ {header=1;next}
        header && /^[A-Z_]+,/ {n++;row=$1","$2","$3","$4","$5","$6","$7}
        END{if(n!=1)exit 1;print row}' "$1"
}

histograms_exact() {
    local diff="$1" input="$2" m="$3" n="$4" k="$5"
    [[ -f "$diff" && -f "$input" ]] || return 1
    awk -F, -v expected="$((m*n))" '
        NR==1 {if($0!="diff_value,count")exit 1;next}
        {if($1!=0 || $2 !~ /^[0-9]+$/)bad=1;sum+=$2;rows++}
        END{if(bad || rows!=1 || sum!=expected)exit 1}' "$diff" &&
    awk -F, -v expected_a="$((m*k))" -v expected_b="$((n*k))" '
        NR==1 {if($0!="int8_value,count_A,count_B")exit 1;next}
        {if($1!=NR-130 || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/)bad=1; a+=$2;b+=$3;rows++}
        END{if(bad || rows!=256 || a!=expected_a || b!=expected_b)exit 1}' "$input"
}

run_accuracy_case() {
    local backend="$1" kernel="$2" tile="$3" lmul="$4" u="$5" exe="$6" cpu="$7"
    local m="$8" n="$9" k="${10}" class="${11}" seed="${12}" kind="${13}"
    local id="${backend}_${kernel}_${m}x${n}x${k}_${class}_seed${seed}_${kind}"
    local dir="$OUT/raw/$id" rc=0 parsed status krc total mismatch maxdiff overflow exact expected=0
    local nr="${tile#*x}" env_args=()
    mkdir -p "$dir"
    if [[ "$backend" == ime ]]; then
        env_args=(SPACEMIT_IME_FORCE_NATIVE=1 SPACEMIT_IME_FORCE_RVV=0 SPACEMIT_IME_FORCE_SCALAR=0)
        if ((m<8 || n<nr || k<8*u)); then expected=1; fi
    fi
    if log_command "$dir/run.log" taskset -c "$cpu" env "${env_args[@]}" "$exe" \
            "$class" "$m" "$n" "$k" "$seed" "$dir/differences.csv" "$dir/inputs.csv"; then rc=0; else rc=$?; fi
    if parsed="$(parse_checker_record "$dir/run.log")"; then
        IFS=, read -r status krc total mismatch maxdiff overflow exact <<< "$parsed"
        if [[ "$rc" -eq 0 && "$status" == OK && "$krc" == 0 && "$total" -eq $((m*n)) && \
              "$mismatch" == 0 && "$maxdiff" == 0 && "$overflow" == 0 ]] && \
                awk -v x="$exact" 'BEGIN{exit !(x ~ /^[0-9]+([.][0-9]+)?$/ && x+0==1)}' && \
                histograms_exact "$dir/differences.csv" "$dir/inputs.csv" "$m" "$n" "$k"; then
            if ((expected)); then
                status=UNEXPECTED_NATIVE_SUCCESS
                record_failure "$id" 'Forced native wrapper unexpectedly accepted a shape with no complete native tile/K block' CONTRACT_CHANGED
                bad=$((bad+1))
            else status=PASS; fi
        elif [[ "$backend" == ime && "$kind" == tiny && "$expected" -eq 1 && \
                "$rc" -eq 0 && "$status" == KERNEL_RETURN && "$krc" == -98 && "$overflow" == 0 ]]; then
            status=EXPECTED_UNSUPPORTED
        else
            status=FAIL
            record_failure "$id" 'Exit status, numerical status, overflow or histogram check failed; see raw case' VALIDATION_FAILED
            bad=$((bad+1))
        fi
    else
        status=PROCESS_OR_PARSE_FAILED; krc=NA; total=NA; mismatch=NA; maxdiff=NA; overflow=NA; exact=NA
        record_failure "$id" 'Missing/ambiguous checker record or native process failure' VALIDATION_FAILED
        bad=$((bad+1))
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$backend" "$tile" "$lmul" "$u" "$kernel" "$cpu" "$m" "$n" "$k" "$class" \
        "$seed" "$kind" "$status" "$krc" "$total" "$mismatch" "$maxdiff" "$overflow" "$exact" "$rc" "$id" \
        >> "$OUT/raw/cases.csv"
}

for backend in rvv ime; do
    if [[ "$backend" == rvv ]]; then mapfile -t KERNELS < <(inventory_rvv); cpu="$RVV_CPU";
    else mapfile -t KERNELS < <(inventory_ime); cpu="$IME_CPU"; fi
    for kernel in "${KERNELS[@]}"; do
        [[ "$kernel" =~ kernel_(8x[48])_zvl256b_lmul([^_]+)_unroll([1248])$ ]] || { printf 'Unsupported selector: %s\n' "$kernel" >&2; exit 2; }
        tile="${BASH_REMATCH[1]}"; lmul="${BASH_REMATCH[2]}"; u="${BASH_REMATCH[3]}"; nr="${tile#*x}"
        case_id="${backend}_${kernel}"
        exe="$OUT/build/$case_id"
        build_dir="$OUT/build/${case_id}_objects"
        mkdir -p "$build_dir"
        if [[ "$backend" == rvv ]]; then
            resolve_rvv "$kernel"; source_file="$RESOLVED_SOURCE"; symbol="${kernel}_i8i32"
            defines=(-DACC_KIND_INT8_RVV "-DKERNEL_SYMBOL=$symbol" -DACC_MR=8 "-DACC_NR=$nr" "-DCNAME=$symbol")
            sources=("$HARNESS" "$source_file"); fallback_hash=NA
        else
            resolve_ime "$kernel"; source_file="$RESOLVED_SOURCE"; symbol="$kernel"
            if ! check_ime_contract "$source_file"; then
                record_failure "$case_id" 'IME input contract no longer matches verified full matrix layout' CONTRACT_CHANGED
                printf '%s,%s,%s,%s,%s,%s,NA,NA,NA,NA,NA,build,CONTRACT_CHANGED,NA,NA,NA,NA,NA,NA,NA,%s\n' \
                    "$backend" "$tile" "$lmul" "$u" "$kernel" "$cpu" "$case_id" >> "$OUT/raw/cases.csv"
                bad=$((bad+1)); continue
            fi
            fallback="$(dirname -- "$source_file")/rvv_fallback.c"
            defines=(-DACC_KIND_INT8_IME -DACC_IME_INPUT_FULL_MATRIX=1 -DSPACEMIT_IME_REQUIRE_HARDWARE=1 \
                "-DKERNEL_SYMBOL=$symbol" -DACC_MR=8 "-DACC_NR=$nr")
            sources=("$HARNESS" "$source_file" "$fallback")
            fallback_hash="$(sha256sum "$fallback" | awk '{print $1}')"
        fi
        if ! log_command "$build_dir/build.log" "$CC" "${CFLAGS_ARRAY[@]}" "${defines[@]}" \
                "${sources[@]}" -no-pie -lm -o "$exe"; then
            record_failure "$case_id" 'Checker/kernel compilation failed' BUILD_FAILED
            printf '%s,%s,%s,%s,%s,%s,NA,NA,NA,NA,NA,build,BUILD_FAILED,NA,NA,NA,NA,NA,NA,NA,%s\n' \
                "$backend" "$tile" "$lmul" "$u" "$kernel" "$cpu" "$case_id" >> "$OUT/raw/cases.csv"
            bad=$((bad+1)); continue
        fi
        printf '%s,%s,%s,%s,%s,%s\n' "$backend" "$kernel" \
            "$(sha256sum "$source_file" | awk '{print $1}')" "$(sha256sum "$HARNESS" | awk '{print $1}')" \
            "$fallback_hash" "$(sha256sum "$exe" | awk '{print $1}')" >> "$OUT/metadata/source_hashes.csv"
        for shape in "${SHAPES[@]}"; do
            IFS=x read -r m n k <<< "$shape"
            for class in "${CLASSES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    run_accuracy_case "$backend" "$kernel" "$tile" "$lmul" "$u" "$exe" "$cpu" \
                        "$m" "$n" "$k" "$class" "$seed" supported
                done
            done
        done
        if [[ "$backend" == ime && "$SMOKE" -eq 0 ]]; then
            for shape in 1x1x1 7x3x7 9x9x9; do
                IFS=x read -r m n k <<< "$shape"
                run_accuracy_case "$backend" "$kernel" "$tile" "$lmul" "$u" "$exe" "$cpu" \
                    "$m" "$n" "$k" bounded_uniform "${SEEDS[0]}" tiny
            done
        fi
    done
done

printf 'backend,tile,lmul,u,kernel,planned_supported_cases,attempted_supported_cases,planned_tiny_cases,attempted_tiny_cases,passed_cases,expected_unsupported_cases,failed_executions,build_failures\n' > "$OUT/summary/correctness_by_kernel.csv"
planned_supported=$((${#SHAPES[@]}*${#CLASSES[@]}*${#SEEDS[@]}))
awk -F, -v planned="$planned_supported" -v smoke="$SMOKE" 'NR>1 {
    id=$1 FS $2 FS $3 FS $4 FS $5;ids[id]=1;backend[id]=$1
    if($12=="build"){build[id]++;next}
    if($12=="supported")supported_attempt[id]++;else if($12=="tiny")tiny_attempt[id]++
    if($13=="PASS")pass[id]++;else if($13=="EXPECTED_UNSUPPORTED")unsupported[id]++;else failed[id]++
} END{for(id in ids)printf "%s,%d,%d,%d,%d,%d,%d,%d,%d\n",id,planned,supported_attempt[id]+0,backend[id]=="ime"&&!smoke?3:0,tiny_attempt[id]+0,pass[id]+0,unsupported[id]+0,failed[id]+0,build[id]+0}' \
    "$OUT/raw/cases.csv" | sort >> "$OUT/summary/correctness_by_kernel.csv"
printf 'backend,executed_supported_or_native_boundary_cases,passing_cases,expected_unsupported_cases,failed_executions,build_failures\n' > "$OUT/summary/correctness_totals.csv"
awk -F, 'NR>1 {
    backends[$1]=1
    if($12=="build"){build[$1]++;next}
    if($13=="PASS"){tested[$1]++;pass[$1]++}
    else if($13=="EXPECTED_UNSUPPORTED")unsupported[$1]++
    else {tested[$1]++;failed[$1]++}
} END{for(b in backends)printf "%s,%d,%d,%d,%d,%d\n",b,tested[b]+0,pass[b]+0,unsupported[b]+0,failed[b]+0,build[b]+0}' \
    "$OUT/raw/cases.csv" | sort >> "$OUT/summary/correctness_totals.csv"

# Full selected-pair tests reuse the common implementation and its real INT64 gate.
# These are separate records, not added to standalone-checker pass denominators.
if ((SMOKE)); then
    printf 'DEFERRED: smoke stage intentionally omits selected full-workload validation.\n' > "$OUT/summary/full_selected_status.txt"
elif load_selection; then
    if load_global_selection; then
        global_present=0
        for pair in "${SELECTED_PAIRS[@]}"; do
            IFS='|' read -r tile rvv ime <<< "$pair"
            [[ "$rvv" != "$GLOBAL_RVV_KERNEL" || "$ime" != "$GLOBAL_IME_KERNEL" ]] || global_present=1
        done
        [[ "$global_present" == 1 ]] || SELECTED_PAIRS+=("independent|$GLOBAL_RVV_KERNEL|$GLOBAL_IME_KERNEL")
    else
        record_failure fig09_global_selected 'Global independent pair unavailable; its full validation is pending' PENDING_SELECTION
        bad=$((bad+1))
    fi
    M="${M:-1024}"; N="${N:-1024}"; K="${K:-1024}"
    for value in "$M" "$N" "$K"; do
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Full-workload M/N/K must be positive decimal integers.\n' >&2; exit 2; }
    done
    for pair in "${SELECTED_PAIRS[@]}"; do
        IFS='|' read -r tile rvv ime <<< "$pair"
        for mode in rvv ime static dynamic; do
            case "$mode" in
                rvv) impl=rvv; threads=8; workers=0; cpus="$ALL_CPUS"; schedule=static ;;
                ime) impl=ime; threads=4; workers=4; cpus="$IME_CPUS"; schedule=static ;;
                static|dynamic) impl=mixed; threads=8; workers=4; cpus="$ALL_CPUS"; schedule="$mode" ;;
            esac
            if ! validate_case "fig09_full_${tile}_${mode}" "$rvv" "$ime" "$impl" "$M" "$N" "$K" \
                    "$threads" "$workers" "$cpus" "$schedule" "${STATIC_WEIGHT:-4}" "${DYNAMIC_CHUNK:-1}"; then
                bad=$((bad+1))
            fi
        done
    done
    printf 'See validation.jsonl and per-case logs for selected full-workload results.\n' > "$OUT/summary/full_selected_status.txt"
else
    record_failure fig09_full_selected 'No validated selected_kernels.tsv: run Fig5/6 selection first; full selected validation pending' PENDING_SELECTION
    printf 'PENDING: selected full-workload validation requires this campaign selected_kernels.tsv.\n' > "$OUT/summary/full_selected_status.txt"
    bad=$((bad+1))
fi
finish_figure
[[ "$bad" -eq 0 ]]
