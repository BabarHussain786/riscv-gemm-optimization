#!/usr/bin/env bash
set -euo pipefail

# Shell launcher for the complete Fig. 5 Python benchmark driver.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../.." && pwd)"

exec python3 "${HERE}/run_fig05_complete.py" \
  --project-root "${PROJECT_ROOT}" \
  "$@"
