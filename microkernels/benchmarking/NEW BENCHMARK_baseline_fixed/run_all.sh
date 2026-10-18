#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CAMPAIGN=${CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
scripts=(00_preflight.sh Figure_01_RVV_IME_Architecture/fig01_architecture.sh
  Figure_05_INT8_Kernel_Tuning/fig05_int8_tuning.sh
  Figure_06_Fair_RVV_IME_Heterogeneous_Comparison/fig06_fair_comparison.sh
  Figure_04_Static_Dynamic_Scheduling/fig04_scheduling.sh
  Figure_02_Strong_Scaling/fig02_strong_scaling.sh
  Figure_03_Weak_Scaling/fig03_weak_scaling.sh
  Figure_07_FP32_FP64_Baselines/fig07_fp_baselines.sh
  Figure_09_INT8_Correctness/fig09_correctness.sh
  Figure_08_Heterogeneous_Summary/fig08_summary.sh)
case "${1:-}" in
  --help|-h) printf 'Usage: bash run_all.sh [--check]\nCreates one new campaign inside NEW BENCHMARK/results. --check is read-only. Set CAMPAIGN to a unique name for individual runners to share selections/data.\n'; exit 0 ;;
  --check)
    for script in "${scripts[@]}"; do bash -n "$ROOT/$script"; bash "$ROOT/$script" --check; done
    printf 'All shell/source checks passed. This does NOT establish K1 compilation, accuracy or performance.\n'; exit 0 ;;
  '') [[ $# == 0 ]] || exit 2 ;;
  *) exit 2 ;;
esac
printf 'New campaign: %s\n' "$CAMPAIGN"
# A small actual-kernel gate precedes the expensive tuning campaign. Its counts
# stay separate from the expanded final accuracy suite and are never double-counted.
bash "$ROOT/00_preflight.sh"
CAMPAIGN="${CAMPAIGN}_smoke" bash "$ROOT/Figure_09_INT8_Correctness/fig09_correctness.sh" --smoke
failed=0
for script in "${scripts[@]:1}"; do
  if ! bash "$ROOT/$script"; then printf 'INCOMPLETE: %s (see its logs).\n' "$script" >&2; failed=$((failed+1)); fi
done
printf 'Campaign %s finished; failed figure stages=%s. No old files or measurements were overwritten.\n' "$CAMPAIGN" "$failed"
((failed==0))
