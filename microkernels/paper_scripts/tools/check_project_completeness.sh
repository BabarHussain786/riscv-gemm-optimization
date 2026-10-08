#!/usr/bin/env bash
set -euo pipefail

# Static audit for the source tree. This does not build kernels or claim that
# a missing measurement has been reproduced; it checks launchers and data
# exports before a board campaign is started.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PAPER_SCRIPTS_ROOT}/.." && pwd)"

expected_launchers=(
  "strong_scaling_performance/run_fig01_strong_scaling.sh"
  "weak_scaling_performance/run_fig02_weak_scaling.sh"
  "static_dynamic_scheduling/run_fig03_static_vs_dynamic.sh"
  "rvv_int8_tuning/run_fig04_rvv_int8_tuning.sh"
  "heterogeneous_rvv_ime_end_to_end/run_fig05_fair_end_to_end.sh"
  "fp32_fp64_comparison/run_fig06_rvv_fp32_fp64.sh"
  "multicore_comparison/run_fig07_multicore_comparison.sh"
  "correctness_validation/run_fig08_correctness.sh"
)

failures=0
valid=0
for rel in "${expected_launchers[@]}"; do
  path="${PAPER_SCRIPTS_ROOT}/${rel}"
  if [[ ! -f "${path}" ]]; then
    printf 'MISSING launcher: %s\n' "${path}" >&2
    failures=$((failures + 1))
  elif ! bash -n "${path}"; then
    printf 'INVALID shell syntax: %s\n' "${path}" >&2
    failures=$((failures + 1))
  else
    valid=$((valid + 1))
  fi
done

# Check every shell script under paper_scripts, including internal runners and
# compatibility wrappers. A figure entry point may delegate to the shared
# orchestration layer, but its target must exist.
shell_count=0
while IFS= read -r -d '' script; do
  shell_count=$((shell_count + 1))
  if ! bash -n "${script}"; then
    printf 'INVALID shell syntax: %s\n' "${script}" >&2
    failures=$((failures + 1))
  fi
done < <(find "${PAPER_SCRIPTS_ROOT}" -type f -name '*.sh' -print0)

required_targets=(
  "run_all_paper_figures.sh"
  "paper_scripts/orchestration/paper_campaign.py"
  "paper_scripts/orchestration/runner_accuracy.sh"
  "paper_scripts/orchestration/runner_openmp.sh"
  "paper_scripts/orchestration/runner_standalone.sh"
  "HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/run_k1_partitioning_analysis.sh"
  "HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/run_k1_strong_scaling.sh"
  "HETEROGENEOUS_RVV_IME_OPENMP_GEMM/scripts/run_k1_weak_scaling.sh"
  "GEMM_RVV_FP32_INT8_8x4_Baseline/RVV_IGEMM_INT8_I8I32_8x4"
  "GEMM_RVV_FP32_INT8_8x8_Baseline/RVV_IGEMM_INT8_I8I32_8x8"
  "IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_8x4_NATIVE"
  "IME_NATIVE_KERNELS/IME_GEMM_INT8_I8I32_8x8_NATIVE"
)
for rel in "${required_targets[@]}"; do
  if [[ ! -e "${PROJECT_ROOT}/${rel}" ]]; then
    printf 'MISSING workflow target: %s\n' "${PROJECT_ROOT}/${rel}" >&2
    failures=$((failures + 1))
  fi
done

root_runner="${PROJECT_ROOT}/run_all_paper_figures.sh"
if [[ ! -f "${root_runner}" ]]; then
  printf 'MISSING root paper runner: %s\n' "${root_runner}" >&2
  failures=$((failures + 1))
elif ! bash -n "${root_runner}"; then
  printf 'INVALID shell syntax: %s\n' "${root_runner}" >&2
  failures=$((failures + 1))
fi

k3="${PROJECT_ROOT}/datasets/extra/experiments/k3/run_k3_rvv_ime_0_15_1024.sh"
if [[ ! -f "${k3}" ]]; then
  printf 'MISSING optional K3 launcher: %s\n' "${k3}" >&2
  failures=$((failures + 1))
elif ! bash -n "${k3}"; then
  printf 'INVALID shell syntax: %s\n' "${k3}" >&2
  failures=$((failures + 1))
fi

dataset_count="$(find "${PROJECT_ROOT}/datasets" -type f \( -name '*.csv' -o -name '*.json' \) | wc -l)"
printf 'VALID paper launchers: %d/%d\n' "${valid}" "${#expected_launchers[@]}"
printf 'Shell scripts checked: %d\n' "${shell_count}"
printf 'Measured data files: %s\n' "${dataset_count}"

if (( failures > 0 )); then
  printf 'AUDIT: FAILED (%d issue(s))\n' "${failures}" >&2
  exit 1
fi
printf 'AUDIT: OK (launchers are present and syntactically valid)\n'
