#!/usr/bin/env python3
"""Run a validated Fig. 5 campaign with the shared benchmark layer.

Examples (run from fig05_complete):
  python3 benchmarking/run_fig05.py --dry-run
  python3 benchmarking/run_fig05.py --size 1024 --runs 7 --prepacked --tune
  python3 benchmarking/run_fig05.py --rvv-kernel igemm_kernel_8x4_zvl256b_lmulmf8_unroll2 \
      --ime-kernel ime_kernel_8x8_zvl256b_lmul1_unroll4
  python3 benchmarking/run_fig05.py --host-test --cc gcc

The eight default pairs use RVV LMUL1 and IME LMUL1 at tiles 8x4/8x8,
unroll 1/2/4/8. These are controlled defaults, not a claim of optimal tuning.
Every native campaign includes static and dynamic 4-IME + 4-RVV execution,
an eight-RVV baseline using the identical RVV kernel, and separate single-CPU
phase profiles. Host smoke tests use one 16x16x64 reference case and can never
produce accepted hardware data. Output directories are unique and never reused.
"""
import argparse
import importlib
import sys
from pathlib import Path

import build
import run as runner

HERE = Path(__file__).resolve().parent
GROUPS = ("validation", "01_eight_core_rvv", "02_heterogeneous_static",
          "03_heterogeneous_dynamic", "04_phase_profiles", "05_prepacked", "06_tuning")


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--project-root", type=Path, default=HERE.parent)
    cli.add_argument("--output", type=Path, default=HERE / "results")
    cli.add_argument("--size", type=int, default=1024, help="Square M=N=K workload (default: 1024)")
    cli.add_argument("--runs", "--repetitions", dest="repetitions", type=int, default=7)
    cli.add_argument("--warmups", type=int, default=2)
    cli.add_argument("--tile-n", type=int, default=32, help="Compatibility flag; only 32 is supported")
    cli.add_argument("--cpus", default="0,1,2,3,4,5,6,7",
                     help="Eight distinct CPU IDs; first four run IME in mixed cases")
    cli.add_argument("--rvv-kernel", help="Explicit RVV selection; supply --ime-kernel independently too")
    cli.add_argument("--ime-kernel", help="Explicit IME selection; supply --rvv-kernel independently too")
    cli.add_argument("--rvv-lmul", choices=("mf8", "mf4", "mf2", "1", "2"),
                     help="RVV LMUL for the eight-pair sweep; every source must exist (default: 1)")
    cli.add_argument("--prepacked", action="store_true", help="Also run matched prepacked cases; output stays in scope")
    cli.add_argument("--tune", action="store_true", help="Add the separate 36-RVV + 8-IME single-CPU sweep")
    cli.add_argument("--dynamic", action="store_true", help="Compatibility flag: dynamic is already included")
    cli.add_argument("--all-ime", action="store_true", help="Rejected: experimental IME LMULs are excluded")
    cli.add_argument("--dry-run", action="store_true", help="Write the full plan and metadata without building or running")
    cli.add_argument("--host-test", action="store_true", help="One nonpublishable 16x16x64 reference smoke case")
    cli.add_argument("--cc", help="Compiler command forwarded to the shared build layer")
    cli.add_argument("--gdb", action="store_true", help="Capture a backtrace if a native executable raises SIGILL")
    for flag, default in (("seed", 42), ("weight", 4), ("chunk", 1), ("timeout", 1800)):
        cli.add_argument("--" + flag, type=int, default=default)
    return cli


def normalize_args(args):
    if args.all_ime:
        raise ValueError("--all-ime is excluded: the primary Fig. 5 campaign accepts only IME LMUL1")
    if args.tile_n != 32:
        raise ValueError("--tile-n must be 32: the shared adapter uses 32-column output strips")
    if any(getattr(args, key) <= 0 for key in ("size", "repetitions", "weight", "chunk", "timeout")):
        raise ValueError("size, runs, weight, chunk, and timeout must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be nonnegative")
    try:
        args.cpus = [int(cpu) for cpu in args.cpus.split(",")]
    except ValueError as exc:
        raise ValueError("--cpus requires eight comma-separated integer CPU IDs") from exc
    if len(args.cpus) != 8 or len(set(args.cpus)) != 8 or min(args.cpus) < 0:
        raise ValueError("--cpus requires exactly eight distinct nonnegative CPU IDs")
    if bool(args.rvv_kernel) != bool(args.ime_kernel):
        raise ValueError("an explicit pair requires both --rvv-kernel and --ime-kernel")
    if args.rvv_kernel and args.rvv_lmul is not None:
        raise ValueError("--rvv-lmul applies only to the default sweep; omit it for an explicit pair")
    if args.host_test and (args.prepacked or args.tune):
        raise ValueError("--host-test is one reference smoke case; --prepacked and --tune require native execution")
    args.project_root, args.output = args.project_root.resolve(), args.output.resolve()
    args.m = args.n = args.k = args.size
    if args.host_test:
        args.m, args.n, args.k = 16, 16, 64
        args.warmups, args.repetitions = 0, 1
    # The shared runner accepts this common argument namespace.
    args.command = "campaign"
    args.implementation, args.timing = "rvv", "end_to_end"
    args.threads, args.ime_workers = 8, 4
    args.schedule, args.profile, args.counters = "static", False, False
    args.repeat_runs, args.include_tuning = 1, False
    return args


def configurations(args):
    if args.rvv_kernel:
        pairs = [{"name": "selected_pair", "rvv_kernel": args.rvv_kernel, "ime_kernel": args.ime_kernel}]
    else:
        pairs = [{"name": f"{tile}_U{unroll}",
                  "rvv_kernel": f"igemm_kernel_{tile}_zvl256b_lmul{args.rvv_lmul or '1'}_unroll{unroll}",
                  "ime_kernel": f"ime_kernel_{tile}_zvl256b_lmul1_unroll{unroll}"}
                 for tile in ("8x4", "8x8") for unroll in (1, 2, 4, 8)]
    if args.host_test:
        pairs = pairs[:1]
    for pair in pairs:
        for backend in ("rvv", "ime"):
            build.resolve(args.project_root, pair[backend + "_kernel"], backend)
    return pairs


def case_plan(args, pairs):
    plan = []
    for pair in pairs:
        base = {"rvv_kernel": pair["rvv_kernel"], "ime_kernel": pair["ime_kernel"],
                "implementation": "rvv", "timing_mode": "end_to_end", "threads": 8,
                "cpu_ids": list(args.cpus), "ime_workers": 0, "schedule": "static",
                "profiled": False, "counters": False}
        if args.host_test:
            return [("01_eight_core_rvv", {**base, "implementation": "reference", "threads": 1,
                                            "cpu_ids": [args.cpus[0]]})]
        for timing in (("end_to_end", "prepacked") if args.prepacked else ("end_to_end",)):
            for schedule, group in (("rvv", "01_eight_core_rvv"), ("static", "02_heterogeneous_static"),
                                    ("dynamic", "03_heterogeneous_dynamic")):
                plan.append(("05_prepacked" if timing == "prepacked" else group,
                             {**base, "timing_mode": timing,
                              "implementation": "rvv" if schedule == "rvv" else "mixed",
                              "ime_workers": 0 if schedule == "rvv" else 4,
                              "schedule": "static" if schedule == "rvv" else schedule}))
            for implementation in ("rvv", "ime"):
                plan.append(("05_prepacked" if timing == "prepacked" else "04_phase_profiles",
                             {**base, "timing_mode": timing, "implementation": implementation,
                              "threads": 1, "cpu_ids": [args.cpus[0]], "profiled": True,
                              "ime_workers": int(implementation == "ime")}))
    if args.tune:
        tuning_args = argparse.Namespace(**{**vars(args), "command": "tuning",
                                          "rvv_kernel": pairs[0]["rvv_kernel"],
                                          "ime_kernel": pairs[0]["ime_kernel"]})
        tuning = runner.case_plan(tuning_args)
        counts = {backend: sum(case["implementation"] == backend for _, case in tuning)
                  for backend in ("rvv", "ime")}
        if counts != {"rvv": 36, "ime": 8}:
            raise ValueError(f"incomplete tuning inventory: expected 36 RVV + 8 IME, found {counts}")
        for _, case in tuning:
            case["cpu_ids"] = [args.cpus[0]]
            for backend in ("rvv", "ime"):
                build.resolve(args.project_root, case[backend + "_kernel"], backend)
            plan.append(("06_tuning", case))
    return plan


def preflight_plan(args, plan):
    """Use the common IME planner for each independent selection, before timing."""
    if args.host_test:
        return []
    preflight, seen = [], set()
    for group, case in plan:
        selected = argparse.Namespace(**{**vars(args), "rvv_kernel": case["rvv_kernel"],
                                         "ime_kernel": case["ime_kernel"]})
        for check in runner.ime_preflight_plan(selected, [(group, case)]):
            key = (check["ime_kernel"], tuple(check["cpu_ids"]))
            if key not in seen:
                seen.add(key)
                preflight.append(check)
    return preflight


def analyze_campaign(campaign):
    for module, directory in (("analyze", "analysis"), ("plot_fig05", "figure05")):
        status = importlib.import_module(module).main(["--input", str(campaign),
                                                       "--output", str(campaign / directory)])
        if status:
            raise RuntimeError(f"{module} failed with status {status}; completed measurements retained")


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    requested = vars(args).copy()
    try:
        normalize_args(args)
        pairs = configurations(args)
        plan = case_plan(args, pairs)
        preflight = preflight_plan(args, plan)
    except (OSError, ValueError) as exc:
        cli.error(str(exc))
    campaign = runner.unique_dir(args.output, "fig05")
    for group in GROUPS:
        (campaign / group).mkdir()
    info = runner.system_info(args.project_root)
    metadata = {"schema_version": 1, "campaign": "fig05", "status": "planned",
                "mode": "dry_run" if args.dry_run else "host_test" if args.host_test else "native",
                "shape": [args.m, args.n, args.k], "runs": args.repetitions, "warmups": args.warmups,
                "configuration_policy": "explicit independent pair" if args.rvv_kernel else
                    "controlled tile/unroll pairs; no optimality claim",
                "host_test_scope": "one 16x16x64 reference case; never publishable" if args.host_test else None,
                "cpu_roles": {"eight_rvv": args.cpus, "mixed_ime": args.cpus[:4],
                              "mixed_rvv": args.cpus[4:], "phase_profile": args.cpus[0]},
                "timing_scope": "total_elapsed", "output_in_scope": True,
                "packing_in_scope": {"end_to_end": True, "prepacked": False},
                "phase_aggregation": "sum_worker_elapsed",
                "phase_note": "single-CPU profiles only; RVV output/boundary are fused and remain null",
                "acceptance_policy": "validated native measurements only; host and dry-run never accepted",
                "configurations": pairs, "planned_cases": len(plan), "ime_preflights": len(preflight)}
    runner.write_json(campaign / "system.json", info)
    runner.write_json(campaign / "command.json", {"argv": sys.argv[1:] if argv is None else argv,
                                                 "requested_arguments": requested, "arguments": vars(args)})
    runner.write_json(campaign / "campaign.json", metadata)
    runner.write_json(campaign / "plan.json", {"configurations": pairs, "ime_preflight": preflight,
                                              "cases": [{"group": group, **case} for group, case in plan]})
    all_rows, accepted, failed, had_error = [], [], [], False
    runner.checkpoint(campaign, all_rows, accepted, failed, had_error)
    try:
        if not args.dry_run:
            if not args.host_test and not info["native_platform"]:
                raise ValueError("native Fig. 5 benchmarking requires RISC-V Linux; use --dry-run or --host-test here")
            available = info.get("available_cpus")
            if not args.host_test and available is not None and not set(args.cpus).issubset(available):
                raise ValueError(f"requested CPU IDs unavailable in process affinity: {args.cpus}; available: {available}")
            check_args = argparse.Namespace(**{**vars(args), "m": 16, "n": 16, "k": 64,
                                               "warmups": 0, "repetitions": 1})
            for case in preflight:
                raw, kept, rejected, error = runner.run_case(check_args, campaign, "validation", case, info, True)
                failed.extend(rejected)
                had_error |= error
                runner.checkpoint(campaign, all_rows, accepted, failed, had_error)
                if error:
                    raise RuntimeError("IME preflight failed; stopped before timing any case; see validation diagnostics")
            for group, case in plan:
                raw, kept, rejected, error = runner.run_case(args, campaign, group, case, info)
                all_rows.extend(raw)
                accepted.extend(kept)
                failed.extend(rejected)
                had_error |= error
                runner.checkpoint(campaign, all_rows, accepted, failed, had_error)
                if error:
                    raise RuntimeError(f"{group} failed; campaign stopped; completed cases and process diagnostics retained")
            if accepted and not args.host_test:
                analyze_campaign(campaign)
    except (OSError, ValueError, RuntimeError, ImportError, KeyboardInterrupt) as exc:
        had_error = True
        reason = "interrupted by user; completed cases retained" if isinstance(exc, KeyboardInterrupt) else str(exc)
        failed.append({"status": "REJECTED", "accepted": False, "publishable": False,
                       "diagnostic_only": True, "rejection_reason": reason})
        runner.write_json(campaign / "error.json", {"error": reason})
        print(reason, file=sys.stderr)
    runner.checkpoint(campaign, all_rows, accepted, failed, had_error)
    metadata.update(status="failed" if had_error else "dry_run" if args.dry_run else "complete",
                    accepted_rows=len(accepted), raw_rows=len(all_rows), failed_rows=len(failed))
    runner.write_json(campaign / "campaign.json", metadata)
    print(str(campaign))
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
