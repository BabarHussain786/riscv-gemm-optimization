#!/usr/bin/env python3
"""Audit FP32/FP64 RVV source and measured-result coverage.

The audit is read-only.  It checks every requested tile/LMUL/unroll
combination and reports source availability separately from benchmark-result
evidence.  A kernel source file is not counted as a measurement.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TILES = ("8x4", "8x8")
LMULS = ("mf8", "mf4", "mf2", "1", "2", "4", "8")
UNROLLS = (1, 2, 4, 8)
RESULT_SUFFIXES = {".csv", ".json", ".log", ".txt"}
MAX_SCAN_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class KernelCase:
    precision: str
    tile: str
    lmul: str
    unroll: int
    directory: Path
    source: Path
    makefile: Path

    @property
    def prefix(self) -> str:
        return "sgemm" if self.precision == "FP32" else "dgemm"

    @property
    def kernel_name(self) -> str:
        return (
            f"{self.prefix}_kernel_{self.tile}_zvl256b_lmul"
            f"{self.lmul}_unroll{self.unroll}"
        )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_root = here.parent.parent
    parser = argparse.ArgumentParser(
        description="Check FP32/FP64 source and measured-result coverage."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="microkernels project directory (default: script parent/parent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional CSV path for the complete audit table",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="return exit code 1 if any source or result combination is missing",
    )
    return parser.parse_args()


def build_cases(root: Path) -> list[KernelCase]:
    cases: list[KernelCase] = []
    for precision in ("FP32", "FP64"):
        prefix = "sgemm" if precision == "FP32" else "dgemm"
        subdir_prefix = "RVV_SGEMM_FP32" if precision == "FP32" else "RVV_DGEMM_FP64"
        for tile in TILES:
            family = root / f"GEMM_RVV_{precision}_INT8_{tile}_Baseline" / (
                f"{subdir_prefix}_{tile}"
            )
            for lmul in LMULS:
                for unroll in UNROLLS:
                    name = (
                        f"{prefix}_kernel_{tile}_zvl256b_lmul{lmul}"
                        f"_unroll{unroll}"
                    )
                    directory = family / name
                    source = directory / f"{name}.c"
                    cases.append(
                        KernelCase(
                            precision=precision,
                            tile=tile,
                            lmul=lmul,
                            unroll=unroll,
                            directory=directory,
                            source=source,
                            makefile=directory / "Makefile",
                        )
                    )
    return cases


def result_file_has_measurement_signal(text: str) -> bool:
    """Reject source/build metadata that merely mentions a kernel path."""
    has_run_or_status = re.search(r"\b(?:run|status|metric|gops|gflops)\b", text, re.I)
    has_time = re.search(
        r"\b(?:time|time_sec|mean_time|duration|elapsed|seconds?)\b", text, re.I
    )
    return bool(has_run_or_status and has_time)


def scan_result_files(root: Path, kernel_names: set[str]) -> dict[str, set[str]]:
    """Return measured-result files mentioning each exact kernel name."""
    found = {name: set() for name in kernel_names}
    names = sorted(kernel_names, key=len, reverse=True)
    combined = re.compile("|".join(re.escape(name) for name in names))

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in RESULT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_SCAN_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not result_file_has_measurement_signal(text):
            continue
        matches = set(combined.findall(text))
        for name in matches:
            found[name].add(str(path))
    return found


def make_rows(cases: list[KernelCase], result_files: dict[str, set[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        source_ok = case.source.is_file() and case.makefile.is_file()
        files = sorted(result_files.get(case.kernel_name, set()))
        rows.append(
            {
                "precision": case.precision,
                "tile": case.tile,
                "lmul": case.lmul,
                "unroll": str(case.unroll),
                "source_status": "AVAILABLE" if source_ok else "MISSING",
                "source_file": str(case.source),
                "result_status": "FOUND" if files else "MISSING",
                "result_file_count": str(len(files)),
                "result_files": "|".join(files),
            }
        )
    return rows


def print_summary(root: Path, rows: list[dict[str, str]]) -> None:
    print(f"Project: {root}")
    print("Audit: FP32 and FP64; tiles 8x4/8x8; LMUL mf8,mf4,mf2,1,2,4,8; U1,U2,U4,U8")
    print("Result files are counted only when they contain kernel, run/status/metric, and timing evidence.")
    print()
    print("PRECISION TILE  SOURCE  RESULT  LMUL COVERAGE                 UNROLL COVERAGE")
    print("--------- -----  ------  ------  ---------------------------  ----------------")
    for precision in ("FP32", "FP64"):
        for tile in TILES:
            subset = [r for r in rows if r["precision"] == precision and r["tile"] == tile]
            source_count = sum(r["source_status"] == "AVAILABLE" for r in subset)
            result_count = sum(r["result_status"] == "FOUND" for r in subset)
            source_lmul = sorted({r["lmul"] for r in subset if r["source_status"] == "AVAILABLE"}, key=LMULS.index)
            source_u = sorted({r["unroll"] for r in subset if r["source_status"] == "AVAILABLE"}, key=lambda x: int(x))
            lmul_text = ",".join(source_lmul) if source_lmul else "none"
            u_text = ",".join(f"U{x}" for x in source_u) if source_u else "none"
            print(
                f"{precision:<9} {tile:<5}  {source_count:>2}/28    "
                f"{result_count:>2}/28    {lmul_text:<27}  {u_text}"
            )
    print()
    missing_sources = [r for r in rows if r["source_status"] == "MISSING"]
    missing_results = [r for r in rows if r["result_status"] == "MISSING"]
    print(f"Source combinations present: {len(rows) - len(missing_sources)}/{len(rows)}")
    print(f"Measured combinations found: {len(rows) - len(missing_results)}/{len(rows)}")
    if missing_sources:
        print("Missing source combinations:")
        for r in missing_sources:
            print(f"  {r['precision']} {r['tile']} LMUL={r['lmul']} U{r['unroll']}")
    if missing_results:
        print("Combinations without measured result evidence:")
        for r in missing_results:
            print(f"  {r['precision']} {r['tile']} LMUL={r['lmul']} U{r['unroll']}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Project directory not found: {root}", file=sys.stderr)
        return 2

    cases = build_cases(root)
    result_files = scan_result_files(root, {case.kernel_name for case in cases})
    rows = make_rows(cases, result_files)
    print_summary(root, rows)
    if args.output:
        write_csv(args.output.resolve(), rows)
        print(f"\nAudit CSV: {args.output.resolve()}")

    if args.fail_on_missing and any(
        r["source_status"] == "MISSING" or r["result_status"] == "MISSING" for r in rows
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
