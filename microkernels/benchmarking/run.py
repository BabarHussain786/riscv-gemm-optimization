#!/usr/bin/env python3
"""Build, validate, and record native INT8 benchmarks without fabricating results."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RVV_DEFAULT = "igemm_kernel_8x4_zvl256b_lmulmf8_unroll2"
IME_DEFAULT = "ime_kernel_8x4_zvl256b_lmul1_unroll1"
GROUPS = ("validation", "prepacked", "end_to_end", "eight_core_rvv",
          "eight_core_heterogeneous_static", "eight_core_heterogeneous_dynamic",
          "tuning", "repeatability", "counters")
BASE_FIELDS = ["campaign_id", "case_id", "group", "status", "validation", "accepted",
               "publishable", "rejection_reason", "implementation", "timing_mode",
               "M", "N", "K", "threads", "rep", "total_sec", "gops"]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def unique_dir(parent, label):
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (label + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                     + "_" + uuid.uuid4().hex[:10])
    path.mkdir(exist_ok=False)
    return path


def write_csv(path, rows):
    fields = BASE_FIELDS + sorted({key for row in rows for key in row} - set(BASE_FIELDS))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})
    temporary.replace(path)


def inspect_command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace", check=False,
                                timeout=20, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        return {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def system_info(project_root=None):
    info = {"captured_utc": datetime.now(timezone.utc).isoformat(), "system": platform.system(),
            "machine": platform.machine(), "release": platform.release(), "node": platform.node(),
            "python": sys.version, "cpu_count": os.cpu_count(), "environment": {
                key: value for key, value in os.environ.items() if key.startswith(("OMP_", "GOMP_"))}}
    info["uname"] = platform.uname()._asdict()
    if project_root is not None:
        for name, command in (("git_commit", ["rev-parse", "HEAD"]),
                              ("git_status", ["status", "--short", "--untracked-files=normal"])):
            info[name] = inspect_command(["git", "-c", "core.fsmonitor=false", "-C", str(project_root)] + command)
    try:
        info["available_cpus"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        info["available_cpus"] = None
    if platform.system() == "Linux":
        info["lscpu"] = inspect_command(["lscpu", "--json"])
        info["cpu_topology"] = inspect_command(["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ"])
        for filename in ("/proc/cpuinfo", "/proc/version", "/proc/sys/kernel/perf_event_paranoid"):
            try:
                info[filename] = Path(filename).read_text(errors="replace")
            except OSError as exc:
                info[filename] = str(exc)
        info["cpufreq"] = {}
        for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/*")):
            if path.name in ("scaling_governor", "scaling_cur_freq", "scaling_min_freq", "scaling_max_freq"):
                try:
                    info["cpufreq"][str(path)] = path.read_text().strip()
                except OSError:
                    pass
    identity = {key: info.get(key) for key in ("system", "machine", "release", "node", "cpu_count")}
    identity["cpuinfo"] = "\n".join(line for line in info.get("/proc/cpuinfo", "").splitlines()
                                    if not line.lower().startswith(("cpu mhz", "bogomips")))
    info["system_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    info["native_platform"] = info["system"] == "Linux" and info["machine"].lower().startswith("riscv")
    return info


def launch(command, directory, label, timeout):
    """Capture failures exactly; a nonzero process never becomes a measurement."""
    write_json(directory / (label + ".command.json"), {"argv": [str(x) for x in command], "cwd": str(directory)})
    try:
        result = subprocess.run([str(x) for x in command], cwd=directory, capture_output=True,
                                text=True, errors="replace", timeout=timeout, check=False)
        record = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        record = {"returncode": None, "stdout": text(getattr(exc, "stdout", "")),
                  "stderr": text(getattr(exc, "stderr", "")), "error": str(exc)}
    (directory / (label + ".stdout.txt")).write_text(record["stdout"], encoding="utf-8")
    (directory / (label + ".stderr.txt")).write_text(record["stderr"], encoding="utf-8")
    write_json(directory / (label + ".process.json"), {k: v for k, v in record.items() if k not in ("stdout", "stderr")})
    return record


def parse_rows(stdout):
    rows, errors = [], []
    for number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSON row is not an object")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            errors.append(f"stdout line {number}: {exc}")
    return rows, errors


def gate_rows(rows, expected, validation_only=False):
    """Reject the entire invocation if even one record disagrees or is invalid."""
    errors = []
    repetitions = 1 if validation_only else expected["repetitions"]
    if len(rows) != repetitions:
        errors.append(f"expected {repetitions} rows, observed {len(rows)}")
    for number, row in enumerate(rows, 1):
        tag = f"row {number}"
        if row.get("status") != "OK" or row.get("validation") != "PASS":
            errors.append(f"{tag}: status/validation not OK/PASS")
        if validation_only:
            if row.get("record_type") != "validation":
                errors.append(f"{tag}: expected validation record")
            continue
        for key in ("M", "N", "K", "implementation", "threads", "timing_mode", "cpu_ids", "profiled"):
            if key == "cpu_ids" and expected["implementation"] == "reference":
                continue  # Host reference is deliberately unpinned; retain its actual CPU IDs.
            if row.get(key) != expected[key]:
                errors.append(f"{tag}: {key} mismatch")
        if row.get("rep") != number:
            errors.append(f"{tag}: missing, duplicated, or out-of-order repetition")
        sec, gops = row.get("total_sec"), row.get("gops")
        if not finite_number(sec, positive=True) or not finite_number(gops, positive=True):
            errors.append(f"{tag}: nonpositive or nonfinite elapsed time/GOPS")
        elif not math.isclose(gops, 2 * expected["M"] * expected["N"] * expected["K"] / sec / 1e9,
                              rel_tol=1e-5, abs_tol=1e-9):
            errors.append(f"{tag}: GOPS does not equal 2*M*N*K / elapsed / 1e9")
        for key in ("packing_sec", "kernel_sec", "output_sec", "boundary_sec"):
            if key not in row or (row[key] is not None and not finite_number(row[key])):
                errors.append(f"{tag}: invalid or missing {key}")
        if row.get("phase_aggregation") != "sum_worker_elapsed":
            errors.append(f"{tag}: missing phase aggregation semantics")
    return errors


def gate_validation(stderr, shape):
    observed = re.findall(r"^VALIDATION shape=(\d+x\d+x\d+) status=(\w+)$", stderr, re.MULTILINE)
    expected = [("16x16x64", "PASS"), ("15x15x69", "PASS"), ("x".join(map(str, shape)), "PASS")]
    return [] if observed == expected else ["mandatory aligned, boundary, and main shape validation evidence incomplete"]


def finite_number(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def kernel_fields(name, prefix):
    match = re.search(r"_(8x[48])_zvl256b_lmul(mf[248]|[1248])_unroll([1248])$", name)
    return {prefix + "_" + key: value for key, value in zip(("tile", "lmul", "unroll"), match.groups())} if match else {}


def bench_command(binary, case, args, shape=None, validation_only=False):
    m, n, k = shape or (args.m, args.n, args.k)
    values = {"m": m, "n": n, "k": k, "implementation": case["implementation"],
              "timing": case["timing_mode"], "threads": case["threads"],
              "cpus": ",".join(map(str, case["cpu_ids"])), "ime-workers": case["ime_workers"],
              "schedule": case["schedule"], "weight": args.weight, "chunk": args.chunk,
              "warmups": 0 if validation_only else args.warmups,
              "repetitions": 1 if validation_only else args.repetitions, "seed": args.seed,
              "profile": int(case["profiled"]), "counters": int(case["counters"]),
              "validate-only": int(validation_only)}
    return [str(binary)] + [str(item) for key, value in values.items() for item in ("--" + key, value)]


def run_case(args, campaign, group, case, info, validation_only=False):
    validation_only = validation_only or args.command == "diagnose"
    directory = unique_dir(campaign / group, case["implementation"])
    write_json(directory / "system.json", info)
    write_json(directory / "case.json", {**case, "M": args.m, "N": args.n, "K": args.k, "arguments": vars(args)})
    metadata = {"campaign_id": campaign.name, "case_id": directory.name, "group": group,
                "rvv_kernel": case["rvv_kernel"], "ime_kernel": case["ime_kernel"],
                "host_test": args.host_test, "execution_backend": "reference" if args.host_test else "native",
                "datatype": "int8_i32", "system_id": info["system_id"], "timing_scope": "total_elapsed",
                "hostname": info.get("node"),
                "git_commit": info.get("git_commit", {}).get("stdout", "").strip() or None,
                "system_path": str(directory / "system.json"),
                "hardware_platform": info["system"] + "/" + info["machine"],
                "counters_requested": case["counters"],
                "requested_cpu_ids": case["cpu_ids"], "packing_in_scope": case["timing_mode"] == "end_to_end",
                "validation_scope": "full", "ime_workers": case["ime_workers"], "schedule": case["schedule"],
                "weight": args.weight, "chunk": args.chunk, "seed": args.seed, "warmups": args.warmups,
                **kernel_fields(case["rvv_kernel"], "rvv"), **kernel_fields(case["ime_kernel"], "ime")}
    errors, rows, build_meta, last_command = [], [], {}, None
    if not info["native_platform"] and not args.host_test:
        errors.append("native benchmarking requires RISC-V Linux; no executable was run")
    else:
        command = [sys.executable, str(HERE / "build.py"), "--project-root", str(args.project_root),
                   "--output", str(directory / "build"), "--rvv-kernel", case["rvv_kernel"],
                   "--ime-kernel", case["ime_kernel"]]
        if args.host_test:
            command.append("--host-test")
        if args.cc:
            command += ["--cc", args.cc]
        result = launch(command, directory, "build", args.timeout)
        try:
            build_meta = json.loads((directory / "build" / "build.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append("build metadata unavailable: " + str(exc))
        if result["returncode"] != 0:
            errors.append(f"build failed: {result.get('error', result['returncode'])}")
        binary = directory / "build" / ("bench.exe" if os.name == "nt" else "bench")
        if not binary.is_file():
            errors.append("benchmark executable missing")
        metadata.update(build_metadata=build_meta, source_digest=build_meta.get("source_digest", ""))
        if not args.host_test and (not metadata["source_digest"] or build_meta.get("host_test")):
            errors.append("native source provenance absent or build is host-test")
        validations = []
        # The C driver runs all three required shapes in this invocation and again before timing.
        for shape in ((args.m, args.n, args.k),):
            if errors:
                break
            label = "validation_" + "x".join(map(str, shape)) + "_" + str(len(validations))
            last_command = bench_command(binary, case, args, shape, True)
            result = launch(last_command, directory, label, args.timeout)
            parsed, issues = parse_rows(result["stdout"])
            expected = {**case, "M": shape[0], "N": shape[1], "K": shape[2], "repetitions": 1}
            issues += gate_rows(parsed, expected, True)
            issues += gate_validation(result["stderr"], shape)
            if result["returncode"] != 0:
                issues.append(f"process failed: {result.get('error', result['returncode'])}")
            validations.append({"shape": shape, "rows": parsed, "errors": issues})
            errors += [label + ": " + issue for issue in issues]
        write_json(directory / "validation.json", validations)
        if not errors and not validation_only:
            last_command = bench_command(binary, case, args)
            result = launch(last_command, directory, "timing", args.timeout)
            rows, issues = parse_rows(result["stdout"])
            errors += issues + gate_rows(rows, {**case, "M": args.m, "N": args.n, "K": args.k,
                                              "repetitions": args.repetitions})
            errors += gate_validation(result["stderr"], (args.m, args.n, args.k))
            if result["returncode"] != 0:
                errors.append(f"timing process failed: {result.get('error', result['returncode'])}")
        if args.gdb and last_command and result["returncode"] in (-signal.SIGILL, 128 + signal.SIGILL):
            if shutil.which("gdb"):
                launch(["gdb", "--batch", "-ex", "run", "-ex", "bt", "-ex", "x/8i $pc-16",
                        "--args"] + last_command, directory, "gdb", args.timeout)
            else:
                errors.append("SIGILL: gdb requested but not installed")
    reasons = list(errors)
    if args.host_test or case["implementation"] == "reference":
        reasons.append("reference/host-test diagnostics are never paper measurements")
    if args.command == "diagnose":
        reasons.append("diagnostic command does not publish measurements")
    if validation_only:
        reasons.append("validation-only preflight, no timing measurements")
    accepted = not reasons
    enriched = [{**row, **metadata, "accepted": accepted, "publishable": accepted,
                 "rejection_reason": "; ".join(reasons), "command": last_command} for row in rows]
    failed = [] if accepted or (validation_only and not errors) else enriched or [{**metadata, "status": "REJECTED", "accepted": False,
                                               "publishable": False, "diagnostic_only": True,
                                               "rejection_reason": "; ".join(reasons)}]
    write_json(directory / "diagnostics.json", {"errors": errors, "exclusions": reasons,
                                                "accepted_rows": len(enriched) if accepted else 0})
    write_csv(directory / "raw_all_runs.csv", enriched)
    write_csv(directory / "accepted_runs.csv", enriched if accepted else [])
    write_csv(directory / "failed_runs.csv", failed)
    validation_dir = campaign / "validation"
    validation_dir.mkdir(exist_ok=True)
    write_json(validation_dir / (directory.name + ".json"), {
        "case_id": directory.name, "group": group, "status": "FAILED" if errors else "PASS",
        "evidence": str((directory / "validation.json").relative_to(campaign)) if (directory / "validation.json").is_file() else None,
        "diagnostics": str((directory / "diagnostics.json").relative_to(campaign)),
        "requested_cpus": case["cpu_ids"], "implementation": case["implementation"],
        "rvv_kernel": case["rvv_kernel"], "ime_kernel": case["ime_kernel"], "preflight": validation_only})
    print(f"{group}/{directory.name}: {'PASS' if not errors else 'FAIL'}; accepted={len(enriched) if accepted else 0}", flush=True)
    return enriched, enriched if accepted else [], failed, bool(errors)


def tuning_kernels(root):
    """The FP32 tree is canonical; do not duplicate the mirrored FP64 INT8 sources."""
    inventory = []
    for tile, lmuls in (("8x4", ("mf8", "mf4", "mf2", "1", "2")), ("8x8", ("mf4", "mf2", "1", "2"))):
        base = root / f"GEMM_RVV_FP32_INT8_{tile}_Baseline" / f"RVV_IGEMM_INT8_I8I32_{tile}"
        for lmul in lmuls:
            for unroll in (1, 2, 4, 8):
                name = f"igemm_kernel_{tile}_zvl256b_lmul{lmul}_unroll{unroll}"
                if (base / name / (name + "_i8i32.c")).is_file():
                    inventory.append(name)
    return inventory


def ime_tuning_kernels(root):
    inventory = []
    for tile in ("8x4", "8x8"):
        base = root / "IME_NATIVE_KERNELS" / f"IME_GEMM_INT8_I8I32_{tile}_NATIVE"
        for unroll in (1, 2, 4, 8):
            name = f"ime_kernel_{tile}_zvl256b_lmul1_unroll{unroll}"
            if (base / name / (name + ".c")).is_file():
                inventory.append(name)
    return inventory


def case_plan(args):
    base = {"implementation": "reference" if args.host_test else args.implementation,
            "timing_mode": args.timing, "threads": args.threads, "cpu_ids": args.cpus,
            "ime_workers": 0 if args.host_test else args.threads if args.implementation == "ime" else args.ime_workers,
            "schedule": args.schedule, "profiled": args.profile,
            "counters": args.counters, "rvv_kernel": args.rvv_kernel, "ime_kernel": args.ime_kernel}
    if args.command in ("run", "diagnose"):
        if args.host_test:
            base.update(threads=1, cpu_ids=[0])
        return [(args.timing, base)]
    if args.command == "repeatability":
        if args.host_test:
            base.update(threads=1, cpu_ids=[0])
        return [("repeatability", dict(base)) for _ in range(args.repeat_runs)]
    plan = []
    if args.command == "campaign":
        for timing in ("prepacked", "end_to_end"):
            for impl in ("rvv", "ime"):
                plan.append((timing, {**base, "implementation": impl, "timing_mode": timing,
                                     "threads": 1, "cpu_ids": [0], "ime_workers": int(impl == "ime"),
                                     "profiled": False, "counters": False}))
        for schedule in ("rvv", "static", "dynamic"):
            group = "eight_core_rvv" if schedule == "rvv" else "eight_core_heterogeneous_" + schedule
            plan.append((group, {**base, "implementation": "rvv" if schedule == "rvv" else "mixed",
                                "threads": 8, "cpu_ids": list(range(8)), "ime_workers": 0 if schedule == "rvv" else 4,
                                "schedule": "static" if schedule == "rvv" else schedule,
                                "timing_mode": "end_to_end", "profiled": False, "counters": False}))
        plan += [("repeatability", {**base, "profiled": False, "counters": False}) for _ in range(args.repeat_runs)]
        plan.append(("counters", {**base, "counters": True, "profiled": False}))
        for impl in ("rvv", "ime"):
            plan.append(("end_to_end", {**base, "implementation": impl, "timing_mode": "end_to_end",
                                        "threads": 1, "cpu_ids": [0], "ime_workers": int(impl == "ime"),
                                        "profiled": True, "counters": False}))
    if args.command == "tuning" or args.include_tuning:
        inventory = tuning_kernels(args.project_root)
        if not inventory:
            raise ValueError("no canonical RVV tuning sources found in project root")
        for kernel in inventory:
            plan.append(("tuning", {**base, "rvv_kernel": kernel, "implementation": "rvv",
                                   "threads": 1, "cpu_ids": [0], "ime_workers": 0,
                                   "profiled": False, "counters": False}))
        ime_inventory = ime_tuning_kernels(args.project_root)
        if not ime_inventory:
            raise ValueError("no canonical LMUL1 IME tuning sources found in project root")
        for kernel in ime_inventory:
            plan.append(("tuning", {**base, "ime_kernel": kernel, "implementation": "ime",
                                   "threads": 1, "cpu_ids": [0], "ime_workers": 1,
                                   "profiled": False, "counters": False}))
    if args.host_test:
        for _, case in plan:
            case.update(implementation="reference", ime_workers=0, threads=1, cpu_ids=[0])
    return plan


def ime_preflight_plan(args, plan):
    """Execute the selected IME independently on every CPU that will receive IME work."""
    cpus = set()
    for _, case in plan:
        if case["implementation"] == "ime":
            cpus.update(case["cpu_ids"])
        elif case["implementation"] == "mixed":
            cpus.update(case["cpu_ids"][:case["ime_workers"]])
    return [{"implementation": "ime", "timing_mode": "end_to_end", "threads": 1,
             "cpu_ids": [cpu], "ime_workers": 1, "schedule": "static", "profiled": False,
             "counters": False, "rvv_kernel": args.rvv_kernel, "ime_kernel": args.ime_kernel}
            for cpu in sorted(cpus)]


def checkpoint(campaign, all_rows, accepted, failed, had_error):
    write_csv(campaign / "raw_all_runs.csv", all_rows)
    write_csv(campaign / "accepted_runs.csv", accepted)
    write_csv(campaign / "failed_runs.csv", failed)
    write_json(campaign / "summary.json", {"raw_rows": len(all_rows), "accepted_rows": len(accepted),
                                          "failed_rows": len(failed), "had_error": had_error})


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("command", choices=("run", "campaign", "tuning", "repeatability", "diagnose", "system-info"))
    command.add_argument("--project-root", type=Path, default=HERE.parent)
    command.add_argument("--output", type=Path, default=HERE / "results")
    command.add_argument("--rvv-kernel", default=RVV_DEFAULT)
    command.add_argument("--ime-kernel", default=IME_DEFAULT)
    command.add_argument("--implementation", choices=("rvv", "ime", "mixed", "reference"), default="rvv")
    command.add_argument("--timing", choices=("prepacked", "end_to_end"), default="end_to_end")
    for flag, default in (("m", 1024), ("n", 1024), ("k", 1024), ("threads", 1),
                          ("ime-workers", 0), ("warmups", 2), ("repetitions", 7), ("seed", 42),
                          ("weight", 4), ("chunk", 1), ("repeat-runs", 3), ("timeout", 1800)):
        command.add_argument("--" + flag, type=int, default=default)
    command.add_argument("--cpus", default="0", help="Distinct CPU IDs, comma separated; one per worker")
    command.add_argument("--schedule", choices=("static", "dynamic"), default="static")
    command.add_argument("--cc", help="Compiler command, passed to build.py")
    for flag in ("host-test", "profile", "counters", "include-tuning", "gdb"):
        command.add_argument("--" + flag, action="store_true")
    return command


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        args.cpus = [int(cpu) for cpu in args.cpus.split(",")]
    except ValueError:
        cli.error("--cpus must contain comma-separated integer IDs")
    if len(set(args.cpus)) != len(args.cpus) or min(args.cpus) < 0 or len(args.cpus) != args.threads:
        cli.error("provide one distinct nonnegative CPU ID per thread")
    if any(getattr(args, key) <= 0 for key in ("m", "n", "k", "threads", "repetitions", "weight", "chunk", "repeat_runs", "timeout")):
        cli.error("dimensions, repetitions, thread count, weight, chunk, repeat-runs, timeout must be positive")
    if args.warmups < 0 or not 0 <= args.ime_workers <= args.threads:
        cli.error("warmups must be nonnegative; IME workers must be within the thread count")
    if args.implementation == "mixed" and not 0 < args.ime_workers < args.threads and not args.host_test:
        cli.error("mixed execution requires both RVV and IME workers")
    args.project_root, args.output = args.project_root.resolve(), args.output.resolve()
    campaign = unique_dir(args.output, args.command)
    info = system_info(args.project_root)
    write_json(campaign / "system.json", info)
    write_json(campaign / "command.json", {"argv": sys.argv if argv is None else argv, "arguments": vars(args)})
    for group in GROUPS:
        (campaign / group).mkdir()
    all_rows, accepted, failed, had_error = [], [], [], False
    checkpoint(campaign, all_rows, accepted, failed, had_error)
    if args.command != "system-info":
        try:
            plan = case_plan(args)
            write_json(campaign / "plan.json", plan)
            if info["native_platform"] and not args.host_test and args.command in ("campaign", "tuning", "repeatability"):
                preflight_args = argparse.Namespace(**{**vars(args), "m": 16, "n": 16, "k": 64,
                                                      "warmups": 0, "repetitions": 1})
                for case in ime_preflight_plan(args, plan):
                    raw, kept, rejected, error = run_case(preflight_args, campaign, "validation", case, info, True)
                    failed.extend(rejected)
                    had_error |= error
                    checkpoint(campaign, all_rows, accepted, failed, had_error)
                    if error:
                        raise ValueError("IME preflight failed; campaign stopped before timing any case")
            for group, case in plan:
                raw, kept, rejected, error = run_case(args, campaign, group, case, info)
                all_rows.extend(raw)
                accepted.extend(kept)
                failed.extend(rejected)
                had_error |= error
                checkpoint(campaign, all_rows, accepted, failed, had_error)
                # Unsupported hosts fail once, before attempts to run a native campaign.
                if not info["native_platform"] and not args.host_test:
                    break
        except (OSError, ValueError, KeyboardInterrupt) as exc:
            had_error = True
            reason = "interrupted by user; completed case results retained" if isinstance(exc, KeyboardInterrupt) else str(exc)
            failed.append({"status": "REJECTED", "diagnostic_only": True, "publishable": False,
                           "accepted": False, "rejection_reason": reason})
            write_json(campaign / "error.json", {"error": reason})
    checkpoint(campaign, all_rows, accepted, failed, had_error)
    print(str(campaign))
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
