#!/usr/bin/env bash
# Source-trace dataset only; no architecture image or measurement is fabricated.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"
case "${1:-}" in
  --help|-h) printf 'Usage: bash fig01_architecture.sh [--check]\nProduces source anchors, hashes and a timing-boundary architecture dataset. No hardware benchmark or figure-image edits.\n'; exit 0 ;;
  --check) check_common_contract; exit 0 ;;
  '') [[ $# == 0 ]] || exit 2 ;;
  *) exit 2 ;;
esac
init_figure fig01_architecture offline
check_common_contract > "$OUT/source_inventory.txt"
files=("$PROJECT_ROOT/benchmarking/src/bench.c" "$PROJECT_ROOT/benchmarking/src/rvv_adapter.c" "$PROJECT_ROOT/benchmarking/src/ime_adapter.c" "$PROJECT_ROOT/HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h")
for tile in 8x4 8x8; do resolve_ime "ime_kernel_${tile}_zvl256b_lmul1_unroll1"; files+=("$RESOLVED_SOURCE"); done
sha256sum "${files[@]}" > "$OUT/source_hashes.sha256"
for file in "${files[@]}"; do
  { printf '\nSOURCE: %s\n' "$file"; grep -nE 'prepare\(|allocate\(|pack\(|execute\(|do_strip\(|call_packed_rvv_tile_kernel|pack_a_panel|pack_b_panel|native_accumulate|scatter_output|scalar_gemm_block|start=bench_now|finish=bench_now|omp barrier|queue\+=' "$file" || true; } >> "$OUT/source_anchors.txt"
done
printf 'backend\tstage\toperation\ttiming\nRVV\tinput\tCanonical A/B to kernel panels per assigned strip\tinside end_to_end; outside prepacked\nRVV\tcompute_output\tVector kernel directly accumulates and stores C; tails fused\tinside both scopes; output not separately timed\nIME\tinput\tCompact B strip then backend-specific A/B panels\tinside end_to_end; outside prepacked\nIME\tcompute\tNative matrix accumulation for complete tiles and K blocks\tinside both scopes\nIME\toutput\tScatter each native output tile into C\tinside both scopes\nIME\tboundary\tApplicable scalar K/M/N tails\tinside both scopes\nCOMMON\tscheduling\tStatic partition/cyclic workers or dynamic atomic queue\tinside wall time; not independently timed\nCOMMON\tsynchronization\tStart and completion barriers\tinside wall time\nCOMMON\tpreparation\tAllocation; reference; team creation; CPU pinning; validation\toutside reported wall time\n' > "$OUT/architecture.tsv"
jq -n --arg campaign "$CAMPAIGN" '{figure:1,campaign:$campaign,record_type:"source_trace_not_benchmark",engine:"common_flat_team",strip_columns:32,
  rvv_flow:["strip-level input packing","vector computation with fused C updates"],ime_flow:["strip preparation and native A/B packing","native tile computation","per-tile output scatter","applicable scalar boundaries"],
  end_to_end_includes:["work assignment","required strip-level packing","computation","output handling","tails","start and completion synchronization"],
  excludes:["workspace allocation","team creation","affinity setup","reference computation","output validation"],
  prepacked_is_not_compute_only:true,global_ime_transpose_at_end:false,plot_generated:false,verification:"source trace; board validation is separate"}' > "$OUT/architecture.json"
finish_figure
