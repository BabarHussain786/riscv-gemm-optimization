#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K1_01_SCRIPT="${SCRIPT_DIR}/run_k1_01_rvv_ime_0_7_1024.sh"

if [ ! -f "${K1_01_SCRIPT}" ]; then
  printf 'ERROR: corrected K1 launcher not found: %s\n' "${K1_01_SCRIPT}" >&2
  exit 1
fi

printf '%s\n' "K1 single-core wrapper"
printf '%s\n' "Using corrected K1_01 RVV/IME launcher."
printf '%s\n' "RVV campaign cores: ${RVV_CORES:-0 1 2 3 4 5 6 7}"
printf '%s\n' "IME campaign cores: ${IME_CORES:-0 1 2 3}"
printf '%s\n' "set_ai_thread: not used on K1"
printf '%s\n' "Output folder: k1_01_rvv_ime_results_${M:-1024}"
printf '\n'

exec bash "${K1_01_SCRIPT}" "$@"
