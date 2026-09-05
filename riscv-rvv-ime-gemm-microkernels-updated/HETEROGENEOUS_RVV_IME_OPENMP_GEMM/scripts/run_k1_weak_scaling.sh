#!/usr/bin/env bash
set -uo pipefail

# Weak scaling keeps approximately the same cubic GEMM work per worker.
# For p workers, each square dimension is base_size * cube_root(p), rounded
# upward to a multiple of eight so the selected kernels keep valid boundaries.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=k1_experiment_common.sh
source "${SCRIPT_DIR}/k1_experiment_common.sh"

BASE_SIZE="${BASE_SIZE:-512}"
ALIGNMENT="${ALIGNMENT:-8}"
TILE_N="${TILE_N:-32}"
RUNS="${RUNS:-6}"
RVV_CORE_COUNTS="${RVV_CORE_COUNTS:-1 2 4 8}"
IME_CORE_COUNTS="${IME_CORE_COUNTS:-1 2 4}"
RVV_KERNEL="${RVV_KERNEL:-igemm_kernel_8x4_zvl256b_lmul1_unroll4}"
IME_KERNEL="${IME_KERNEL:-ime_kernel_8x4_zvl256b_lmul1_unroll4}"

require_positive_integer "BASE_SIZE" "${BASE_SIZE}"
require_positive_integer "ALIGNMENT" "${ALIGNMENT}"
require_positive_integer "TILE_N" "${TILE_N}"
require_positive_integer "RUNS" "${RUNS}"

scaled_dimension()
{
    local workers="$1"
    awk -v base="${BASE_SIZE}" -v workers="${workers}" -v align="${ALIGNMENT}" '
        BEGIN {
            target = base * exp(log(workers) / 3.0);
            rounded = int((target + align - 1) / align) * align;
            print rounded;
        }
    '
}

start_experiment "k1_weak_scaling"

CASE_TILE_N="${TILE_N}"; CASE_RUNS="${RUNS}"
CASE_SCHEDULE="static"; CASE_CHUNK="1"
CASE_IME_WEIGHT="4"; CASE_RVV_WEIGHT="1"
CASE_ENABLE_MF2="0"
CASE_PERF_STAT="0"; CASE_PERF_EVENTS="cycles,instructions,cache-references,cache-misses"

for cores in ${RVV_CORE_COUNTS}; do
    require_positive_integer "RVV core count" "${cores}"
    size="$(scaled_dimension "${cores}")"
    CASE_M="${size}"; CASE_N="${size}"; CASE_K="${size}"
    CASE_MODE="k1-rvv"; CASE_THREADS="${cores}"
    CASE_KERNEL="${RVV_KERNEL}"; CASE_KIND="INT8_RVV"
    run_experiment_case "RVV" "cores" "${cores}" || true
done

for cores in ${IME_CORE_COUNTS}; do
    require_positive_integer "IME core count" "${cores}"
    size="$(scaled_dimension "${cores}")"
    CASE_M="${size}"; CASE_N="${size}"; CASE_K="${size}"
    CASE_MODE="k1-ime"; CASE_THREADS="${cores}"
    CASE_KERNEL="${IME_KERNEL}"; CASE_KIND="INT8_IME"
    run_experiment_case "IME" "cores" "${cores}" || true
done

finish_experiment
