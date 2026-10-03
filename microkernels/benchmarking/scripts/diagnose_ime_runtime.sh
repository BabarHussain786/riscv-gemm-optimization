#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${HERE}/run.py" diagnose --implementation ime --m 16 --n 16 --k 64 --repetitions 1 "$@"
