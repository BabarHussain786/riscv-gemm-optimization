#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
case "${1:-}" in
  --help|-h) printf 'Usage: bash 00_preflight.sh [--check]\n--check: read-only source checks on any Bash host. Default: K1-native RVV checks on all CPUs and IME aligned checks on every IME CPU.\n'; exit 0 ;;
  --check) check_common_contract; exit 0 ;;
  '') [[ $# == 0 ]] || exit 2 ;;
  *) exit 2 ;;
esac
init_figure preflight
preflight_once
finish_figure
