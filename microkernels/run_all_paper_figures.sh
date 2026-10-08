#!/usr/bin/env bash
set -euo pipefail

# Run the complete K1 paper campaign (Figures 1--8) through the shared
# planner. Pass --dry-run --no-perf to create plans without building or
# executing kernels.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER="${SCRIPT_DIR}/paper_scripts/orchestration/paper_campaign.py"

if [[ ! -f "${PLANNER}" ]]; then
  printf 'ERROR: paper campaign planner not found: %s\n' "${PLANNER}" >&2
  exit 1
fi

exec python3 "${PLANNER}" "$@"
