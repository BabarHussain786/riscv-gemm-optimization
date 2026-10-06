#!/usr/bin/env bash
set -euo pipefail

# Paper Figure 3 static-versus-dynamic scheduling launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec python3 "${PROJECT_ROOT}/paper_scripts/paper_campaign.py" "$@" --figures 3
