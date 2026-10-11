#!/usr/bin/env bash
# Figure 6: complete heterogeneous RVV/IME campaign for the SpacemiT K1.
#
# This driver is additive. It does not modify benchmarking sources, existing
# Figure 6 scripts, old datasets, or plotting files. It follows the strict
# orchestration used by the successful Figure 2 driver, but builds the IME
# adapter with vectorization disabled to avoid the known SIGILL failure.
set -Eeuo pipefail

TASK_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" - "$TASK_SCRIPT_DIR" "$@" <<'PY'
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shlex
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
import uuid

ROOT = Path(os.environ.get("PROJECT_ROOT", sys.argv[1])).resolve()
BENCH = ROOT / "benchmarking"
N = 1024
TILES = ("8x4", "8x8")
UNROLLS = (1, 2, 4, 8)
MODES = ("rvv8", "static", "dynamic")
ROLES = ("primary", "profile", "counters")

SCOPE = """FIGURE 6: COMPLETE HETEROGENEOUS RVV/IME CAMPAIGN

Primary matrix: fixed INT8 M=N=K=1024, tiles 8x4 and 8x8, unroll factors
U1/U2/U4/U8, and RVV8/static-4RVV+4IME/dynamic-4RVV+4IME execution.
Each primary group uses two warmups and seven measured repetitions.

Primary timing is end-to-end and includes assignment, packing, kernel work,
applicable output handling, and timed barriers. Allocation, initialization,
reference construction, validation, and cleanup are outside the timer.

Profile values are worker-summed elapsed times, not wall-clock components.
Panel-phase fields are: packing_sec, kernel_sec, ime_output_sec, and
ime_boundary_sec. RVV output stores remain fused with the RVV kernel.
Synchronization is not separately timed. Missing fields are NA or an explicit
scope string; they are never converted to zero.

Counters are an independent diagnostic pass. Unavailable or multiplexed
counters do not invalidate accepted timing or phase measurements.

The driver deliberately avoids benchmarking/run.py's external small-shape IME
preflight. Each binary is tested on the real 1024^3 workload before the full
campaign. ime_adapter.c is compiled with -fno-tree-vectorize and
-fno-tree-slp-vectorize; all other source files use the normal target flags.
"""


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def csv_write(path, rows, initial=()):
    path = Path(path)
    rows = list(rows)
    fields = list(initial) + sorted({k for row in rows for k in row} - set(initial))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "NA" if v is None else json.dumps(v, sort_keys=True)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positive(name, value, allow_zero=False):
    text = str(value)
    pattern = r"(?:0|[1-9][0-9]*)" if allow_zero else r"[1-9][0-9]*"
    if not re.fullmatch(pattern, text):
        raise ValueError(f"{name} must be a nonnegative integer" if allow_zero else
                         f"{name} must be a positive integer")
    number = int(text)
    if number > 2_147_483_647:
        raise ValueError(f"{name} is too large")
    return number


def cpu_list(name, default):
    values = [int(x) for x in os.environ.get(name, default).split(",")]
    if len(values) != 4 or len(set(values)) != 4 or any(x < 0 or x >= 1024 for x in values):
        raise ValueError(f"{name} must contain four distinct CPU IDs in 0..1023")
    return values


def configuration():
    values = {
        "runs": positive("RUNS", os.environ.get("RUNS", "7")),
        "warmups": positive("WARMUPS", os.environ.get("WARMUPS", "2"), True),
        "seed": positive("SEED", os.environ.get("SEED", "42")),
        "timeout": positive("TIMEOUT_SEC", os.environ.get("TIMEOUT_SEC", "1800")),
        "weight": positive("STATIC_WEIGHT", os.environ.get("STATIC_WEIGHT", "4")),
        "chunk": positive("DYNAMIC_CHUNK", os.environ.get("DYNAMIC_CHUNK", "1")),
        "collect_counters": os.environ.get("COLLECT_COUNTERS", "1") == "1",
        "collect_profiles": os.environ.get("COLLECT_PROFILES", "1") == "1",
        "cc": os.environ.get("CC", "gcc"),
        "ime_cpus": cpu_list("IME_CPUS", "0,1,2,3"),
        "rvv_cpus": cpu_list("RVV_CPUS", "4,5,6,7"),
    }
    if values["seed"] > 4_294_967_295:
        raise ValueError("SEED exceeds uint32 range")
    if not shlex.split(values["cc"]):
        raise ValueError("CC must not be empty")
    if set(values["ime_cpus"]) & set(values["rvv_cpus"]):
        raise ValueError("IME_CPUS and RVV_CPUS must not overlap")
    return values


def resolve_rvv(tile, unroll):
    name = f"igemm_kernel_{tile}_zvl256b_lmul1_unroll{unroll}"
    path = (ROOT / f"GEMM_RVV_FP32_INT8_{tile}_Baseline" /
            f"RVV_IGEMM_INT8_I8I32_{tile}" / name / f"{name}_i8i32.c")
    if not path.is_file():
        raise ValueError(f"Missing RVV source: {path}")
    return name, path


def resolve_ime(tile, unroll):
    name = f"ime_kernel_{tile}_zvl256b_lmul1_unroll{unroll}"
    path = (ROOT / "IME_NATIVE_KERNELS" /
            f"IME_GEMM_INT8_I8I32_{tile}_NATIVE" / name / f"{name}.c")
    fallback = path.parent / "rvv_fallback.c"
    if not path.is_file() or not fallback.is_file():
        raise ValueError(f"Missing IME source or fallback: {name}")
    return name, path, fallback


def plan(config):
    all_cpus = config["ime_cpus"] + config["rvv_cpus"]
    points = []
    for tile in TILES:
        for unroll in UNROLLS:
            rvv, rvv_source = resolve_rvv(tile, unroll)
            ime, ime_source, fallback = resolve_ime(tile, unroll)
            for mode in MODES:
                if mode == "rvv8":
                    implementation, schedule, ime_workers = "rvv", "static", 0
                else:
                    implementation, schedule, ime_workers = "mixed", mode, 4
                points.append({
                    "point_id": f"{tile}_u{unroll}_{mode}",
                    "tile": tile, "unroll": unroll, "mode": mode,
                    "implementation": implementation, "schedule": schedule,
                    "threads": 8, "ime_workers": ime_workers,
                    "rvv_workers": 8 - ime_workers,
                    "cpus": all_cpus, "rvv_kernel": rvv, "ime_kernel": ime,
                    "rvv_source": str(rvv_source), "ime_source": str(ime_source),
                    "ime_fallback": str(fallback),
                })
    return points


def numeric(value, positive_value=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive_value else value >= 0))


def stats(values):
    values = sorted(value for value in values if numeric(value))
    if not values:
        return {"n": 0, "mean": None, "median": None, "sd": None,
                "min": None, "max": None}
    return {
        "n": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": values[0], "max": values[-1],
    }


def source_inventory(points):
    files = set(BENCH.glob("*.py")) | set((BENCH / "src").glob("*.[ch]"))
    files.add(ROOT / "HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h")
    for point in points:
        files.update((Path(point["rvv_source"]), Path(point["ime_source"]),
                      Path(point["ime_fallback"])))
    missing = [str(path) for path in sorted(files) if not path.is_file()]
    if missing:
        raise ValueError("Missing source files: " + "; ".join(missing))
    return files


def launch(argv, folder, timeout):
    folder = Path(folder)
    json_write(folder / "command.json", {"argv": argv, "started_utc": stamp()})
    with (folder / "stdout.log").open("w", encoding="utf-8") as out, \
            (folder / "stderr.log").open("w", encoding="utf-8") as err:
        process = subprocess.Popen(argv, cwd=folder, stdout=out, stderr=err,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            raise
    json_write(folder / "process.json", {
        "return_code": code, "finished_utc": stamp(), "timed_out": code == 124
    })
    return code


def compile_command(log, command, timeout):
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND: " + shlex.join(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def build_pair(point, result, config, builds, source_before):
    pair = (point["rvv_kernel"], point["ime_kernel"])
    if pair in builds:
        binary, meta = builds[pair]
        if meta.get("status") != "OK":
            raise ValueError(meta.get("error", f"Previous build failed for {pair}; see {meta.get('build_dir', 'build log')}"))
        if sha(binary) != meta["executable_sha256"]:
            raise ValueError("Cached benchmark binary changed")
        return binary, meta

    compiler = shlex.split(config["cc"])
    source_paths = [Path(point["rvv_source"]), Path(point["ime_source"]),
                    Path(point["ime_fallback"]), BENCH / "src" / "bench.c",
                    BENCH / "src" / "rvv_adapter.c", BENCH / "src" / "ime_adapter.c"]
    source_hashes = {str(path): sha(path) for path in source_paths}
    key_data = json.dumps({"pair": pair, "sources": source_hashes,
                           "compiler": compiler}, sort_keys=True).encode()
    key = hashlib.sha256(key_data).hexdigest()[:16]
    build_dir = result / "builds" / f"pair_{key}"
    build_dir.mkdir(parents=True, exist_ok=False)
    config_header = build_dir / "selected_sources.h"
    config_header.write_text(
        f"#define BENCH_RVV_NR {point['tile'][-1]}\n"
        f"#define BENCH_DISPATCH_SOURCE {json.dumps((ROOT / 'HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h').as_posix())}\n"
        f"#define BENCH_IME_SOURCE {json.dumps(Path(point['ime_source']).as_posix())}\n",
        encoding="utf-8")
    base_flags = compiler + ["-O3", "-std=c11", "-Wall", "-Wextra",
        "-Wno-unused-function", "-Wno-unknown-pragmas", "-march=rv64gcv_zvl256b",
        "-mabi=lp64d", "-fopenmp", "-fno-pie", "-I", str(BENCH / "src"),
        "-include", str(config_header)]
    sources = [
        (BENCH / "src" / "bench.c", []),
        (BENCH / "src" / "rvv_adapter.c", []),
        (BENCH / "src" / "ime_adapter.c", ["-fno-tree-vectorize", "-fno-tree-slp-vectorize"]),
        # Most RVV sources expose CNAME; the 8x8 sources expose their full
        # function name directly.  These sources export the precision-suffixed
        # symbol (<kernel>_i8i32), not the base directory name.  Rename the
        # exact exported symbol so rvv_adapter.c resolves bench_rvv_entry and
        # the IME fallback can retain its original symbol for ime_adapter.c.
        (Path(point["rvv_source"]), ["-DCNAME=bench_rvv_entry",
                                     f"-D{point['rvv_kernel']}=bench_rvv_entry",
                                     f"-D{point['rvv_kernel']}_i8i32=bench_rvv_entry"]),
        (Path(point["ime_fallback"]), []),
    ]
    objects = []
    commands = []
    log = build_dir / "build.log"
    try:
        for index, (source, extra) in enumerate(sources):
            obj = build_dir / f"unit_{index}.o"
            command = base_flags + extra + ["-c", str(source), "-o", str(obj)]
            commands.append(command)
            compile_command(log, command, config["timeout"])
            objects.append(str(obj))
        binary = build_dir / "bench"
        command = compiler + objects + ["-fopenmp", "-no-pie", "-o", str(binary)]
        commands.append(command)
        compile_command(log, command, config["timeout"])
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        error = f"Build failed for {pair}; see {log}: {exc}"
        builds[pair] = (None, {"status": "FAILED", "error": error, "build_dir": str(build_dir)})
        raise ValueError(error) from exc
    if any(not Path(path).is_file() or sha(path) != digest
           for path, digest in source_before.items()):
        raise ValueError("Source changed during compilation")
    executable_hash = sha(binary)
    meta = {
        "status": "OK", "rvv_kernel": point["rvv_kernel"],
        "ime_kernel": point["ime_kernel"], "binary": str(binary),
        "executable_sha256": executable_hash, "sources": source_hashes,
        "source_fingerprint": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "commands": commands, "compiler": compiler,
        "ime_adapter_flags": ["-fno-tree-vectorize", "-fno-tree-slp-vectorize"],
    }
    json_write(build_dir / "build.json", meta)
    (build_dir / "complete").write_text("built\n", encoding="utf-8")
    builds[pair] = (binary, meta)
    return binary, meta


def worker_gate(rows, point):
    issues = []
    expected = point["cpus"]
    for row in rows:
        if row.get("ime_workers") != point["ime_workers"] or row.get("rvv_workers") != point["rvv_workers"]:
            issues.append("RVV/IME worker split mismatch")
        workers = row.get("workers")
        if not isinstance(workers, list) or len(workers) != point["threads"]:
            issues.append("missing worker records")
            continue
        total_strips = 0
        for worker in workers:
            ident = worker.get("id")
            if not isinstance(ident, int) or ident < 0 or ident >= len(expected):
                issues.append("invalid worker identity")
                continue
            if worker.get("cpu_before") != expected[ident] or worker.get("cpu_after") != expected[ident]:
                issues.append("worker affinity mismatch")
            strips = worker.get("strips")
            if not isinstance(strips, int) or strips < 0:
                issues.append("invalid strip count")
            else:
                total_strips += strips
        if total_strips != 32:
            issues.append("worker strip assignment does not cover 32 output strips")
    return sorted(set(issues))


def phase_gate(rows, point, role):
    issues = []
    fields = ("packing_sec", "kernel_sec", "output_sec", "boundary_sec",
              "ime_output_sec", "ime_boundary_sec")
    for row in rows:
        if row.get("phase_aggregation") != "sum_worker_elapsed":
            issues.append("missing phase aggregation semantics")
        if row.get("rvv_output_fused") is not True:
            issues.append("RVV output scope is not marked fused")
        if role != "profile":
            if any(row.get(field) is not None for field in fields):
                issues.append("phase values present outside profile role")
            continue
        if not numeric(row.get("packing_sec"), True) or not numeric(row.get("kernel_sec"), True):
            issues.append("missing packing or kernel phase value")
        if row.get("output_sec") is not None or row.get("boundary_sec") is not None:
            issues.append("mixed/RVV output and boundary fields must remain null")
        if point["mode"] == "rvv8":
            if row.get("ime_output_sec") is not None or row.get("ime_boundary_sec") is not None:
                issues.append("IME phase fields must be null for RVV-only profile")
        else:
            if not numeric(row.get("ime_output_sec"), True):
                issues.append("missing IME output phase value")
            if not numeric(row.get("ime_boundary_sec")):
                issues.append("missing IME boundary phase value")
    return sorted(set(issues))


def counter_gate(rows, role):
    issues = []
    for row in rows:
        fields = ("cycles", "instructions", "ipc", "cache_references", "cache_misses")
        if role != "counters":
            if row.get("counters_status") != "DISABLED" or any(row.get(x) is not None for x in fields):
                issues.append("counter fields active outside counter role")
        elif row.get("counters_status") == "OK":
            if not all(numeric(row.get(x), True) for x in fields):
                issues.append("counter status OK without numeric values")
            elif not math.isclose(row["ipc"], row["instructions"] / row["cycles"], rel_tol=1e-8):
                issues.append("IPC does not equal instructions divided by cycles")
        elif row.get("counters_status") == "UNAVAILABLE_OR_MULTIPLEXED":
            if any(row.get(x) is not None for x in fields):
                issues.append("unavailable counters must be null")
        else:
            issues.append("invalid counter status")
    return sorted(set(issues))


def backend_validation_gate(stderr, point):
    if point["mode"] == "rvv8":
        return []
    observed = re.findall(r"^BACKEND_VALIDATION backend=(\w+) shape=(\d+x\d+x\d+) status=(\w+)$",
                          stderr, re.MULTILINE)
    expected = [("rvv", "16x16x64", "PASS"), ("rvv", "15x15x69", "PASS"),
                ("ime", "16x16x64", "PASS"), ("ime", "15x15x69", "PASS")]
    return [] if observed == expected else ["mixed backend validation evidence incomplete"]


def enrich(row, point, role, result, meta, config):
    return {
        **row, **{k: point[k] for k in ("point_id", "tile", "unroll", "mode", "implementation",
                                         "schedule", "threads", "ime_workers", "rvv_workers",
                                         "rvv_kernel", "ime_kernel")},
        "role": role, "campaign_id": result.name, "M": N, "N": N, "K": N,
        "input_a_bytes": N * N, "input_b_bytes": N * N,
        "output_c_bytes": 4 * N * N, "operations": 2 * N ** 3, "macs": N ** 3,
        "alpha": 1, "lda": N, "ldb": N, "ldc": N,
        "cpu_ids": point["cpus"], "static_weight": config["weight"],
        "dynamic_chunk": config["chunk"], "warmups": config["warmups"],
        "seed": config["seed"], "phase_time_unit": "seconds",
        "phase_aggregation": "sum_worker_elapsed",
        "rvv_output_scope": "fused_with_kernel",
        "synchronization_scope": "not_separately_measured",
        "source_sha256": meta.get("source_fingerprint"),
        "binary_sha256": meta["executable_sha256"],
        "validation_reference": "independent_INT64_full_output",
        "input_layout": "canonical_signed_int8_seeded",
        "output_layout": "column_major_int32",
        "unmeasured_reason": "allocation_initialization_reference_validation_synchronization_not_separate",
    }


def run_case(point, role, repetitions, result, config, builds, source_before,
             runner, allowed_cpus, append_rows=True):
    folder = result / ("smoke" if role == "smoke" else "cases") / point["point_id"] / role
    folder.mkdir(parents=True, exist_ok=False)
    attempt = {"point_id": point["point_id"], "role": role,
               "status": "FAILED", "started_utc": stamp(), "log_dir": str(folder)}
    try:
        if not set(point["cpus"]) <= allowed_cpus:
            raise ValueError("Requested CPU is outside the process affinity")
        binary, meta = build_pair(point, result, config, builds, source_before)
        expected = {"implementation": point["implementation"], "timing_mode": "end_to_end",
                    "threads": point["threads"], "cpu_ids": point["cpus"],
                    "profiled": role == "profile", "M": N, "N": N, "K": N,
                    "repetitions": repetitions}
        counters = role == "counters"
        argv = [str(binary), "--m", str(N), "--n", str(N), "--k", str(N),
                "--implementation", point["implementation"], "--timing", "end_to_end",
                "--threads", str(point["threads"]), "--ime-workers", str(point["ime_workers"]),
                "--cpus", ",".join(map(str, point["cpus"])), "--schedule", point["schedule"],
                "--weight", str(config["weight"]), "--chunk", str(config["chunk"]),
                "--warmups", str(1 if role == "smoke" else config["warmups"]),
                "--repetitions", str(repetitions), "--seed", str(config["seed"]),
                "--profile", str(int(role == "profile")), "--counters", str(int(counters)),
                "--validate-only", "0"]
        json_write(folder / "environment_before.json", {
            "platform": platform.platform(), "cpu_ids": point["cpus"],
            "role": role, "started_utc": stamp()})
        code = launch(argv, folder, config["timeout"])
        stdout = (folder / "stdout.log").read_text(errors="replace")
        stderr = (folder / "stderr.log").read_text(errors="replace")
        rows, issues = runner.parse_rows(stdout)
        issues += runner.gate_rows(rows, expected)
        issues += runner.gate_validation(stderr, (N, N, N))
        issues += backend_validation_gate(stderr, point)
        issues += worker_gate(rows, point)
        issues += phase_gate(rows, point, role)
        issues += counter_gate(rows, role)
        if code != 0:
            issues.append(f"process return code {code} (124=timeout)")
        if any(not Path(path).is_file() or sha(path) != digest
               for path, digest in source_before.items()):
            issues.append("source changed during measurement")
        issues = sorted(set(issues))
        if rows:
            enriched = [enrich(row, point, role, result, meta, config) for row in rows]
        else:
            enriched = []
        attempt.update({"return_code": code, "observed_rows": len(rows),
                        "accepted_rows": len(enriched) if not issues else 0,
                        "reason": "; ".join(issues)})
        json_write(folder / "checks.json", {"issues": issues,
                                             "accepted_samples": attempt["accepted_rows"]})
        if issues:
            return enriched, [], attempt
        attempt["status"] = "OK" if role != "counters" or all(
            row.get("counters_status") == "OK" for row in rows) else "COUNTERS_UNAVAILABLE"
        return enriched, enriched, attempt
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        attempt["reason"] = str(exc)
        json_write(folder / "checks.json", {"issues": [str(exc)], "accepted_samples": 0})
        return [], [], attempt


def summaries(points, accepted, config):
    output = []
    for point in points:
        row = {**point, "M": N, "N": N, "K": N,
               "expected_repetitions": config["runs"],
               "phase_aggregation": "sum_worker_elapsed",
               "rvv_output_scope": "fused_with_kernel",
               "synchronization_scope": "not_separately_measured"}
        for role in ("primary", "profile", "counters"):
            records = [x for x in accepted if x["point_id"] == point["point_id"] and x["role"] == role]
            row[f"{role}_samples"] = len(records)
            metrics = ("total_sec", "gops") if role == "primary" else (
                ("total_sec", "packing_sec", "kernel_sec", "ime_output_sec", "ime_boundary_sec")
                if role == "profile" else ("cycles", "instructions", "ipc", "cache_references", "cache_misses"))
            for metric in metrics:
                for name, value in stats([record.get(metric) for record in records]).items():
                    row[f"{role}_{metric}_{name}"] = value
        row["primary_complete"] = row["primary_samples"] == config["runs"]
        row["profile_complete"] = (not config["collect_profiles"] or
                                    row["profile_samples"] == config["runs"])
        row["counters_complete"] = (not config["collect_counters"] or
                                     row["counters_samples"] == config["runs"] and
                                     row.get("counters_ipc_n", 0) == config["runs"])
        output.append(row)
    baseline = {(x["tile"], x["unroll"]): x for x in output if x["mode"] == "rvv8"}
    for row in output:
        base = baseline[(row["tile"], row["unroll"])]
        if row["primary_complete"] and base["primary_complete"] and row.get("primary_total_sec_mean"):
            row["speedup_vs_rvv8"] = base["primary_total_sec_mean"] / row["primary_total_sec_mean"]
        else:
            row["speedup_vs_rvv8"] = None
    return output


def write_outputs(result, points, accepted, raw, attempts, failures, config, source_unchanged):
    initial = ("point_id", "role", "rep", "accepted", "tile", "unroll", "mode")
    csv_write(result / "measurements.csv", accepted, initial)
    csv_write(result / "raw_all_runs.csv", raw, initial)
    csv_write(result / "attempts.csv", attempts, ("point_id", "role", "status"))
    csv_write(result / "failures.csv", failures, ("point_id", "role", "reason"))
    summary = summaries(points, accepted, config)
    csv_write(result / "summary.csv", summary,
              ("point_id", "tile", "unroll", "mode", "primary_samples", "profile_samples", "counters_samples"))
    primary = [row for row in summary if row["mode"] in MODES]
    csv_write(result / "primary_comparison.csv", primary,
              ("tile", "unroll", "mode", "primary_samples", "primary_total_sec_mean",
               "primary_total_sec_median", "primary_total_sec_sd", "speedup_vs_rvv8"))
    phase = [row for row in summary if row["profile_samples"]]
    csv_write(result / "phase_profile_summary.csv", phase,
              ("tile", "unroll", "mode", "profile_samples", "profile_packing_sec_mean",
               "profile_kernel_sec_mean", "profile_ime_output_sec_mean", "profile_ime_boundary_sec_mean",
               "phase_aggregation", "rvv_output_scope", "synchronization_scope"))
    expected_groups = len(points)
    primary_ok = sum(row["primary_complete"] for row in summary)
    profile_ok = sum(row["profile_complete"] for row in summary)
    counters_ok = sum(row["counters_complete"] for row in summary)
    completion = {
        "finished_utc": stamp(), "expected_primary_groups": expected_groups,
        "observed_primary_groups": primary_ok,
        "expected_profile_groups": expected_groups if config["collect_profiles"] else 0,
        "observed_profile_groups": profile_ok if config["collect_profiles"] else 0,
        "expected_counter_groups": expected_groups if config["collect_counters"] else 0,
        "observed_counter_groups": counters_ok if config["collect_counters"] else 0,
        "primary_records_expected": expected_groups * config["runs"],
        "primary_records_observed": sum(row["primary_samples"] for row in summary),
        "timings_complete": primary_ok == expected_groups,
        "phase_complete": (not config["collect_profiles"] or profile_ok == expected_groups),
        "counters_complete": (not config["collect_counters"] or counters_ok == expected_groups),
        "source_unchanged": source_unchanged,
        "failed_attempts": len(failures),
        "hardware_execution": "native_riscv",
    }
    completion["exit_code"] = (0 if completion["timings_complete"] and completion["phase_complete"]
                                and source_unchanged and not failures else
                                3 if completion["timings_complete"] and completion["phase_complete"]
                                and source_unchanged else 1)
    json_write(result / "completion.json", completion)
    return completion


def self_test():
    point = {"mode": "rvv8"}
    row = {"phase_aggregation": "sum_worker_elapsed", "rvv_output_fused": True,
           "packing_sec": 1.0, "kernel_sec": 2.0, "output_sec": None,
           "boundary_sec": None, "ime_output_sec": None, "ime_boundary_sec": None}
    assert not phase_gate([row], point, "profile")
    assert phase_gate([{**row, "output_sec": 0.1}], point, "profile")
    assert stats([1.0, 3.0])["mean"] == 2.0
    print("Self-test PASS")


def main():
    parser = argparse.ArgumentParser(description="Complete Figure 6 K1 driver")
    parser.add_argument("--check", action="store_true", help="read-only source and configuration check")
    parser.add_argument("--self-test", action="store_true", help="offline gate/statistics test")
    parser.add_argument("--smoke-only", action="store_true", help="run the 24 full-size smoke cases only")
    args = parser.parse_args(sys.argv[2:])
    if args.self_test:
        self_test()
        return 0
    config = configuration()
    points = plan(config)
    source_files = source_inventory(points)
    before = {str(path): sha(path) for path in sorted(source_files)}
    print(f"Figure 6: {len(points)} primary groups; {config['runs']} repetitions/group; tiles=8x4,8x8; modes=RVV8,static,dynamic", flush=True)
    print("Build protection: ime_adapter.c uses -fno-tree-vectorize -fno-tree-slp-vectorize", flush=True)
    if args.check:
        for point in points:
            print(f"{point['point_id']} cpus={','.join(map(str, point['cpus']))} SOURCE_OK")
        print("Read-only check complete; no K1 execution performed.")
        return 0
    if platform.system() != "Linux" or not platform.machine().lower().startswith("riscv"):
        raise ValueError("Full Figure 6 execution requires native RISC-V Linux/K1; use --check here")
    os.environ.update(OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    allowed = set(os.sched_getaffinity(0))
    result_root = Path(os.environ.get("RESULT_ROOT", str(ROOT / "datasets/fig06_heterogeneous_complete"))).resolve()
    result = result_root / ("k1_1024_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    result.mkdir(parents=True, exist_ok=False)
    (result / "measurement_scope.txt").write_text(SCOPE, encoding="utf-8")
    json_write(result / "configuration.json", {**config, "project_root": str(ROOT),
                                                 "points": points, "roles": ROLES,
                                                 "M": N, "N": N, "K": N, "started_utc": stamp()})
    runner = load_module("figure6_existing_run", BENCH / "run.py")
    json_write(result / "system.json", runner.system_info(ROOT))
    builds = {}
    accepted, raw, attempts, failures = [], [], [], []
    smoke_points = list(points)
    print(f"Running {len(smoke_points)} full-size smoke cases before the campaign", flush=True)
    for point in smoke_points:
        rows, kept, attempt = run_case(point, "smoke", 1, result, config, builds, before,
                                       runner, allowed)
        for row in rows:
            row["accepted"] = bool(kept)
            raw.append(row)
        attempt["smoke"] = True
        attempts.append(attempt)
        if not kept:
            failures.append({"point_id": point["point_id"], "role": "smoke",
                             "reason": attempt.get("reason", "smoke failure"),
                             "log_dir": attempt.get("log_dir")})
            print(f"SMOKE FAILED: {point['point_id']}: {attempt.get('reason')}", flush=True)
    if failures:
        source_unchanged = all(Path(path).is_file() and sha(path) == digest
                                for path, digest in before.items())
        json_write(result / "completion.json", {
            "exit_code": 1, "smoke_complete": False,
            "smoke_cases_expected": len(smoke_points),
            "smoke_cases_observed": len(smoke_points) - len(failures),
            "failed_attempts": len(failures),
            "source_unchanged": source_unchanged,
        })
        csv_write(result / "attempts.csv", attempts, ("point_id", "role", "status"))
        csv_write(result / "failures.csv", failures, ("point_id", "role", "reason"))
        print("Smoke gate failed; full campaign was not started.", flush=True)
        return 1
    print("Smoke gate passed for all 24 primary configurations", flush=True)
    if args.smoke_only:
        source_unchanged = all(Path(path).is_file() and sha(path) == digest
                               for path, digest in before.items())
        csv_write(result / "smoke_attempts.csv", attempts, ("point_id", "role", "status"))
        json_write(result / "completion.json", {
            "exit_code": 0 if source_unchanged else 1,
            "mode": "smoke_only", "smoke_complete": source_unchanged,
            "smoke_cases_expected": len(smoke_points),
            "smoke_cases_observed": len(smoke_points),
            "failed_attempts": 0, "source_unchanged": source_unchanged,
        })
        print("Smoke-only campaign complete; full measurements were not started.", flush=True)
        return 0 if source_unchanged else 1
    jobs = [(point, role) for point in points for role in ROLES
            if role != "profile" or config["collect_profiles"]
            if role != "counters" or config["collect_counters"]]
    random.Random(config["seed"]).shuffle(jobs)
    json_write(result / "execution_order.json", [[point["point_id"], role] for point, role in jobs])
    for index, (point, role) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {point['point_id']} {role}", flush=True)
        rows, kept, attempt = run_case(point, role, config["runs"], result, config, builds,
                                       before, runner, allowed)
        for row in rows:
            row["accepted"] = bool(kept)
            raw.append(row)
            if kept:
                accepted.append(row)
        attempts.append(attempt)
        if not kept:
            failures.append({"point_id": point["point_id"], "role": role,
                             "reason": attempt.get("reason", "case rejected"),
                             "log_dir": attempt.get("log_dir")})
            print("  FAILED; diagnostics retained", flush=True)
        else:
            print(f"  {attempt.get('status', 'OK')}", flush=True)
        write_outputs(result, points, accepted, raw, attempts, failures, config,
                      all(Path(path).is_file() and sha(path) == digest
                          for path, digest in before.items()))
    source_unchanged = all(Path(path).is_file() and sha(path) == digest
                           for path, digest in before.items())
    completion = write_outputs(result, points, accepted, raw, attempts, failures,
                               config, source_unchanged)
    print(f"Results: {result}", flush=True)
    print(f"Exit {completion['exit_code']} | timings complete={completion['timings_complete']} | "
          f"phases complete={completion['phase_complete']} | counters complete={completion['counters_complete']} | "
          f"source unchanged={source_unchanged}", flush=True)
    return completion["exit_code"]


try:
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
except (ValueError, OSError) as error:
    print("ERROR: " + str(error), file=sys.stderr)
    sys.exit(2)
PY

