#!/usr/bin/env python3
"""Run the complete matched eight-core RVV/IME experiment for Fig. 5.

This driver deliberately reuses ``benchmarking/run.py``.  It does not replace
or modify any RVV or IME kernel.  For each of the eight matched LMUL=1 pairs
(8x4/8x8 and U1/U2/U4/U8), it runs:

* eight-core RVV;
* four-RVV plus four-IME static heterogeneous execution; and
* optionally, four-RVV plus four-IME dynamic heterogeneous execution.

The primary measurements use unprofiled ``end_to_end`` timing.  With
``--profile``, a second profiled pass is collected for phase information.  A
profiled phase sum is never substituted for the primary wall-clock time.

Run this on the SpaceMiT K1, not on a Windows host.  Results are written to a
timestamped directory containing raw rows, rejected rows, summary statistics,
the exact commands, and a manifest suitable for plotting Fig. 5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARKING = Path(__file__).resolve().parents[1]
RUNNER = BENCHMARKING / "run.py"
TILES = ("8x4", "8x8")
UNROLLS = (1, 2, 4, 8)
DEFAULT_CPUS = "0,1,2,3,4,5,6,7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=BENCHMARKING.parent)
    p.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKING / "results" / "fig05_complete",
        help="Parent directory for the timestamped Fig. 5 campaign.",
    )
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpus", default=DEFAULT_CPUS)
    p.add_argument("--weight", type=int, default=4)
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--cc", default=None)
    p.add_argument(
        "--static-only",
        action="store_true",
        help="Run only the primary static comparison; dynamic is included by default.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="After primary runs, repeat every case with --profile for phase fields.",
    )
    p.add_argument(
        "--include-boundary-validation",
        action="store_true",
        help="Also run the 15x15x69 tail check; disabled by default for Fig. 5 timing.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    return p.parse_args()


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_output(parent: Path) -> Path:
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / f"fig05_{now_tag()}"
    out.mkdir()
    return out


def kernel_names(tile: str, unroll: int) -> tuple[str, str]:
    rvv = f"igemm_kernel_{tile}_zvl256b_lmul1_unroll{unroll}"
    ime = f"ime_kernel_{tile}_zvl256b_lmul1_unroll{unroll}"
    return rvv, ime


def cases(static_only: bool) -> list[dict[str, Any]]:
    modes = [
        ("RVV_8CORE", "rvv", "static", 0),
        ("HETERO_STATIC_4RVV_4IME", "mixed", "static", 4),
    ]
    if not static_only:
        modes.append(("HETERO_DYNAMIC_4RVV_4IME", "mixed", "dynamic", 4))

    planned: list[dict[str, Any]] = []
    for tile in TILES:
        for unroll in UNROLLS:
            rvv, ime = kernel_names(tile, unroll)
            for comparison, implementation, schedule, ime_workers in modes:
                planned.append(
                    {
                        "comparison": comparison,
                        "implementation": implementation,
                        "schedule": schedule,
                        "ime_workers": ime_workers,
                        "tile_shape": tile,
                        "unroll": unroll,
                        "rvv_kernel": rvv,
                        "ime_kernel": ime,
                    }
                )
    return planned


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def as_float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def accepted_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok", "pass"}


def locate_run(parent: Path) -> Path | None:
    candidates = [p for p in parent.glob("run_*") if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def command_for(args: argparse.Namespace, case: dict[str, Any], profile: bool, case_output: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "run",
        "--project-root",
        str(args.project_root.resolve()),
        "--output",
        str(case_output.resolve()),
        "--rvv-kernel",
        case["rvv_kernel"],
        "--ime-kernel",
        case["ime_kernel"],
        "--implementation",
        case["implementation"],
        "--timing",
        "end_to_end",
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--threads",
        "8",
        "--ime-workers",
        str(case["ime_workers"]),
        "--cpus",
        args.cpus,
        "--schedule",
        case["schedule"],
        "--weight",
        str(args.weight),
        "--chunk",
        str(args.chunk),
        "--warmups",
        str(args.warmups),
        "--repetitions",
        str(args.repetitions),
        "--seed",
        str(args.seed),
        "--timeout",
        str(args.timeout),
    ]
    if profile:
        command.append("--profile")
    if not args.include_boundary_validation:
        # This is a Python boolean flag; run.py converts it to the C driver's
        # explicit ``--skip-boundary-validation 1`` argument.
        command.append("--skip-boundary-validation")
    if args.cc:
        command.extend(["--cc", args.cc])
    return command


def run_case(args: argparse.Namespace, campaign: Path, case: dict[str, Any], profile: bool) -> dict[str, Any]:
    profile_tag = "profiled" if profile else "primary"
    label = f"{case['comparison']}_{case['tile_shape']}_U{case['unroll']}_{profile_tag}"
    case_output = campaign / "runs" / label
    case_output.mkdir(parents=True, exist_ok=True)
    command = command_for(args, case, profile, case_output)
    command_file = case_output / "command.txt"
    command_file.write_text(" ".join(command) + "\n", encoding="utf-8")

    print(f"[{case['comparison']}] {case['tile_shape']} U{case['unroll']} ({profile_tag})", flush=True)
    if args.dry_run:
        print("  DRY-RUN: " + " ".join(command), flush=True)
        return {**case, "profiled": profile, "status": "DRY_RUN", "command": command}

    log_path = case_output / "console.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(args.project_root.resolve()),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    run_dir = locate_run(case_output)
    if run_dir is None:
        return {
            **case,
            "profiled": profile,
            "status": "FAILED",
            "error": f"run.py produced no run directory; returncode={completed.returncode}",
            "run_path": "",
        }

    raw = read_csv(run_dir / "raw_all_runs.csv")
    accepted = read_csv(run_dir / "accepted_runs.csv")
    failed = read_csv(run_dir / "failed_runs.csv")
    ok = completed.returncode == 0 and bool(accepted)
    status = "OK" if ok else "FAILED"
    return {
        **case,
        "profiled": profile,
        "status": status,
        "returncode": completed.returncode,
        "run_path": str(run_dir),
        "raw_rows": raw,
        "accepted_rows": accepted,
        "failed_rows": failed,
        "command": command,
    }


def enrich_rows(results: list[dict[str, Any]], accepted: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for result in results:
        rows = result.get("accepted_rows" if accepted else "raw_rows", [])
        if not rows and not accepted:
            rows = result.get("failed_rows", [])
        if not rows and not accepted and result.get("status") not in ("OK", "DRY_RUN"):
            rows = [
                {
                    "status": "REJECTED",
                    "accepted": "False",
                    "rejection_reason": result.get("error", "case failed before producing CSV rows"),
                    "total_sec": "",
                    "gops": "",
                }
            ]
        for row in rows:
            enriched = {
                **row,
                "fig05_comparison": result["comparison"],
                "tile_shape": result["tile_shape"],
                "unroll": result["unroll"],
                "rvv_kernel_selected": result["rvv_kernel"],
                "ime_kernel_selected": result["ime_kernel"],
                "profile_pass": "yes" if result["profiled"] else "no",
                "fig05_run_path": result.get("run_path", ""),
            }
            output.append(enriched)
    return output


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else None


def statistic_fields(values: list[float], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_median": statistics.median(values) if values else None,
        f"{prefix}_sd": sample_sd(values),
        f"{prefix}_min": min(values) if values else None,
        f"{prefix}_max": max(values) if values else None,
    }


def make_summary(rows: list[dict[str, Any]], repetitions: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not accepted_bool(row.get("accepted")):
            continue
        if str(row.get("status", "")).upper() not in {"OK", "PASS", "ACCEPTED"}:
            continue
        key = (
            str(row.get("fig05_comparison")),
            str(row.get("tile_shape")),
            int(row.get("unroll", 0)),
            str(row.get("profile_pass", "no")),
        )
        if as_float(row.get("total_sec")) is not None:
            groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for (comparison, tile, unroll, profile_pass), group in sorted(groups.items()):
        times = [as_float(r.get("total_sec")) for r in group]
        gops = [as_float(r.get("gops")) for r in group]
        times = [v for v in times if v is not None]
        gops = [v for v in gops if v is not None]
        record: dict[str, Any] = {
            "fig05_comparison": comparison,
            "tile_shape": tile,
            "unroll": unroll,
            "profile_pass": profile_pass,
            "timing_mode": "end_to_end",
            "threads": 8,
            "n_accepted": len(group),
            "n_requested": repetitions,
            "validation": "PASS",
            **statistic_fields(times, "total_sec"),
            **statistic_fields(gops, "gops"),
        }
        for phase in ("packing_sec", "kernel_sec", "output_sec", "boundary_sec"):
            values = [as_float(r.get(phase)) for r in group]
            values = [v for v in values if v is not None]
            record.update(statistic_fields(values, phase))
        summary.append(record)

    # Add matched speedups using the same tile, unroll, and profile pass.
    by_key = {(r["fig05_comparison"], r["tile_shape"], r["unroll"], r["profile_pass"]): r for r in summary}
    for record in summary:
        base = by_key.get(("RVV_8CORE", record["tile_shape"], record["unroll"], record["profile_pass"]))
        if base and record["fig05_comparison"] != "RVV_8CORE":
            base_time = as_float(base.get("total_sec_mean"))
            target_time = as_float(record.get("total_sec_mean"))
            record["speedup_vs_rvv_mean"] = base_time / target_time if base_time and target_time else None
            base_med = as_float(base.get("total_sec_median"))
            target_med = as_float(record.get("total_sec_median"))
            record["speedup_vs_rvv_median"] = base_med / target_med if base_med and target_med else None
        else:
            record["speedup_vs_rvv_mean"] = None
            record["speedup_vs_rvv_median"] = None
    return summary


def write_manifest(path: Path, args: argparse.Namespace, campaign: Path, planned: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    manifest = {
        "purpose": "Fig. 5 matched eight-core RVV versus heterogeneous RVV-IME benchmark",
        "campaign_directory": str(campaign),
        "project_root": str(args.project_root.resolve()),
        "workload": {"M": args.m, "N": args.n, "K": args.k},
        "timing": "end_to_end primary; packing is included in total_sec",
        "rvv_workers": 8,
        "heterogeneous_workers": {"rvv": 4, "ime": 4},
        "tiles": list(TILES),
        "unrolls": list(UNROLLS),
        "primary_cases_planned": len(planned),
        "primary_cases_completed": sum(r.get("status") == "OK" and not r.get("profiled") for r in results),
        "profile_pass_requested": bool(args.profile),
        "profile_note": "Profile phase values are worker-duration sums; they are explanatory and not substituted for total wall time.",
        "validation": ("aligned 16x16x64 and requested full shape; boundary tail check is separate"
                       if not args.include_boundary_validation else
                       "aligned 16x16x64, 15x15x69 boundary, and requested full shape"),
        "commands": [" ".join(r.get("command", [])) for r in results],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.m <= 0 or args.n <= 0 or args.k <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise SystemExit("dimensions/repetitions must be positive; warmups must be nonnegative")
    cpus = [part.strip() for part in args.cpus.split(",") if part.strip()]
    if len(cpus) != 8 or len(set(cpus)) != 8:
        raise SystemExit("--cpus must provide eight distinct CPU IDs for Fig. 5")

    args.project_root = args.project_root.resolve()
    if not RUNNER.is_file():
        raise SystemExit(f"benchmark runner not found: {RUNNER}")
    campaign = make_output(args.output)
    planned = cases(args.static_only)
    results: list[dict[str, Any]] = []

    print(f"Fig. 5 campaign: {campaign}", flush=True)
    print(f"Planned primary cases: {len(planned)}", flush=True)
    print("Primary timing: end_to_end; packing included; runs are sequential.", flush=True)
    if args.include_boundary_validation:
        print("Validation scope: aligned, boundary, and full workload.", flush=True)
    else:
        print("Validation scope: aligned and full workload; boundary check is separate.", flush=True)

    for case in planned:
        result = run_case(args, campaign, case, profile=False)
        results.append(result)
        if result.get("status") not in ("OK", "DRY_RUN"):
            print(f"  FAILED: inspect {campaign / 'runs' / (case['comparison'] + '_' + case['tile_shape'] + '_U' + str(case['unroll']) + '_primary')}", flush=True)
            if args.stop_on_failure:
                break

    if args.profile and not args.dry_run:
        print("Starting optional profiled pass.", flush=True)
        for case in planned:
            result = run_case(args, campaign, case, profile=True)
            results.append(result)
            if result.get("status") not in ("OK", "DRY_RUN") and args.stop_on_failure:
                break

    raw_rows = enrich_rows(results, accepted=False)
    accepted_rows = enrich_rows(results, accepted=True)
    failures = [row for row in raw_rows if not accepted_bool(row.get("accepted"))]
    summary = make_summary(accepted_rows, args.repetitions)

    write_csv(campaign / "fig05_raw_runs.csv", raw_rows)
    write_csv(campaign / "fig05_accepted_runs.csv", accepted_rows)
    write_csv(campaign / "fig05_failures.csv", failures)
    write_csv(campaign / "fig05_summary.csv", summary)
    write_manifest(campaign / "fig05_manifest.json", args, campaign, planned, results)

    print(f"Accepted rows: {len(accepted_rows)}", flush=True)
    print(f"Rejected rows: {len(failures)}", flush=True)
    print(f"Summary: {campaign / 'fig05_summary.csv'}", flush=True)
    print(f"Raw data: {campaign / 'fig05_accepted_runs.csv'}", flush=True)
    if args.dry_run:
        return 0
    all_ok = all(result.get("status") in ("OK", "DRY_RUN") for result in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
