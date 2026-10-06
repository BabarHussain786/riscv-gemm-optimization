#!/usr/bin/env bash
set -uo pipefail
export LC_ALL=C

# Run the standalone 8x4 and 8x8 INT8 kernel campaigns with one command.
# The campaigns are sequential so they do not compete for the same K1 cores.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

M="${M:-1024}"
N="${N:-1024}"
K="${K:-1024}"
RUNS="${RUNS:-6}"
STAMP="$(date +%Y%m%d_%H%M%S)"
COMBINED_OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/k1_standalone_int8_8x4_8x8_results_${M}_${STAMP}}"
COMBINED_LOG="${COMBINED_OUT_DIR}/combined_campaign.log"

if ! mkdir -p "${COMBINED_OUT_DIR}"; then
    printf 'ERROR: cannot create output directory: %s\n' "${COMBINED_OUT_DIR}" >&2
    exit 1
fi
: > "${COMBINED_LOG}"

log()
{
    printf '%s\n' "$*" | tee -a "${COMBINED_LOG}"
}

run_campaign()
{
    local tile="$1"
    local script="$2"
    local campaign_out="${COMBINED_OUT_DIR}/${tile}"
    local rc

    log "Starting ${tile} standalone INT8 campaign"
    if M="${M}" N="${N}" K="${K}" RUNS="${RUNS}" OUT_DIR="${campaign_out}" \
        bash "${SCRIPT_DIR}/${script}"; then
        rc=0
        log "Finished ${tile} campaign: OK"
    else
        rc=$?
        log "Finished ${tile} campaign: FAILED (exit=${rc})"
    fi
    return "${rc}"
}

log "K1 combined standalone INT8 kernel benchmark"
log "GEMM=${M}x${N}x${K}; runs=${RUNS}; order=8x4,8x8"
log "Results root: ${COMBINED_OUT_DIR}"

failures=0
run_campaign "8x4" "run_k1_standalone_int8_kernels.sh" || failures=$((failures + 1))
run_campaign "8x8" "run_k1_standalone_int8_8x8_kernels.sh" || failures=$((failures + 1))

if [ "${failures}" -ne 0 ]; then
    log "DONE status=FAILED failed_campaigns=${failures}"
    exit 1
fi

log "DONE status=OK"
exit 0
