#!/usr/bin/env python3
"""Analyze the runner's accepted hardware repetitions; never import legacy data.

Usage: python3 analyze.py --input campaigns/NAME --output analysis/NAME
Only INPUT/accepted_runs.csv is an input. Output directories must not exist.
No result in this module is a hardware measurement or an empirical default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


PHASES = ("packing_sec", "kernel_sec", "output_sec", "boundary_sec")
METRICS = PHASES + ("total_sec", "gops")
OPTIONAL_METRICS = ("ime_output_sec", "ime_boundary_sec", "cycles", "instructions",
                    "cache_references", "cache_misses", "ipc")
ALL_METRICS = METRICS + OPTIONAL_METRICS
REQUIRED = (
    "M", "N", "K", "implementation", "timing_mode", "threads",
    "rvv_workers", "ime_workers", "schedule", "rep", "seed", "profiled",
    "status", "validation", "accepted", "publishable", "execution_backend",
    "datatype", "system_id", "source_digest", "build_metadata", "cpu_ids",
    "phase_aggregation", "timing_scope", "validation_scope", "total_sec", "gops",
    "rvv_kernel", "ime_kernel", "rvv_lmul", "ime_lmul", "rvv_tile",
    "ime_tile", "rvv_unroll", "ime_unroll",
)
# Every other input column is a configuration dimension. This conservative rule
# also protects newly introduced hardware/compiler/validation metadata from
# accidental pooling. Explicitly transient output locations are not dimensions.
TRANSIENT = frozenset((
    "rep", "run_id", "attempt", "timestamp", "timestamp_utc", "started_at",
    "finished_at", "command", "command_json", "stdout", "stderr",
    "stdout_path", "stderr_path", "result_path", "run_dir", "run_path",
    "system_path", "system_json_path", "build_path", "build_json_path",
    "campaign_path", "campaign_id", "campaign_dir", "case_id", "group", "validation_error",
    "error", "reason", "rejection_reason", "exit_code", "returncode", "max_abs_error",
    "max_rel_error", "mismatches", "_line", "_config_fields", "workers",
    "ime_output_sec", "ime_boundary_sec", "ipc", "cache_references", "counters_status",
)) | frozenset(ALL_METRICS)
COUNTER_METRICS = frozenset(("cycles", "instructions", "cache_references", "cache_misses", "ipc"))
INTEGER_FIELDS = ("M", "N", "K", "threads", "rvv_workers", "ime_workers", "rep", "seed")
BOOL_FIELDS = ("profiled", "accepted", "publishable")
OPTIONAL_BOOL_FIELDS = ("packing_in_scope", "phases_additive")
ACCEPTED_STATUS = {"OK", "PASS"}
NULLS = {"", "null", "none", "na", "n/a"}


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError("expected an explicit boolean")


def number(value: Any, *, nullable: bool = False) -> float | None:
    if value is None or str(value).strip().lower() in NULLS:
        if nullable:
            return None
        raise ValueError("missing numeric value")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("numeric value must be finite and nonnegative")
    return result


def frozen(value: Any) -> str:
    """Stable, typed configuration representation for keys and CSV cells."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def is_counter(field: str) -> bool:
    # Scope/enable flags and hardware counts are configuration, not readings.
    return field in COUNTER_METRICS


def config_fields(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(k for k in row if k not in TRANSIENT and not is_counter(k)))


def normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on incomplete provenance, rejected runs, or invalid metrics."""
    missing = [field for field in REQUIRED if field not in raw]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))
    row = {key: value.strip() if isinstance(value, str) else value
           for key, value in raw.items() if key is not None}
    if str(row["status"]).upper() not in ACCEPTED_STATUS:
        raise ValueError("run status is not OK/PASS")
    if str(row["validation"]).upper() != "PASS":
        raise ValueError("validation is not PASS")
    for field in BOOL_FIELDS:
        row[field] = boolean(row[field])
    if not row["accepted"] or not row["publishable"]:
        raise ValueError("run is not accepted and publishable")
    if str(row["execution_backend"]).lower() != "native":
        raise ValueError("reference/non-native runs are not hardware evidence")
    for field in ("host_test", "diagnostic_only"):
        if field in row and boolean(row[field]):
            raise ValueError("host/reference diagnostics are not hardware evidence")
    for field in ("datatype", "system_id", "source_digest", "build_metadata", "schedule", "timing_scope", "validation_scope"):
        if not row[field] or str(row[field]).lower() in NULLS:
            raise ValueError("missing configuration/provenance: " + field)
    for field in INTEGER_FIELDS:
        value = row[field]
        if isinstance(value, bool) or str(value).strip() != str(int(value)):
            raise ValueError("invalid integer: " + field)
        row[field] = int(value)
    if any(row[f] <= 0 for f in ("M", "N", "K", "threads")):
        raise ValueError("shape and thread count must be positive")
    if min(row[f] for f in ("rvv_workers", "ime_workers", "rep", "seed")) < 0:
        raise ValueError("workers, repetition and seed must be nonnegative")
    if row["rvv_workers"] + row["ime_workers"] != row["threads"]:
        raise ValueError("worker split does not equal total threads")
    impl = row["implementation"]
    if impl not in {"rvv", "ime", "mixed"}:
        raise ValueError("unknown implementation")
    if ((impl == "rvv" and row["ime_workers"] != 0)
            or (impl == "ime" and row["rvv_workers"] != 0)
            or (impl == "mixed" and min(row["rvv_workers"], row["ime_workers"]) == 0)):
        raise ValueError("implementation and worker split disagree")
    if row["timing_mode"] not in {"prepacked", "end_to_end"}:
        raise ValueError("unknown timing scope")
    if row["phase_aggregation"] != "sum_worker_elapsed":
        raise ValueError("unknown phase aggregation")
    for backend in ("rvv", "ime"):
        if row[backend + "_workers"]:
            for suffix in ("kernel", "lmul", "tile", "unroll"):
                field = backend + "_" + suffix
                if row[field] is None or str(row[field]).strip().lower() in NULLS:
                    raise ValueError("missing active-kernel configuration: " + field)
    for field in ("cpu_ids", "build_metadata"):
        if isinstance(row[field], str):
            try:
                row[field] = json.loads(row[field])
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON metadata: " + field) from exc
    cpus = row["cpu_ids"]
    if (not isinstance(cpus, list) or len(cpus) != row["threads"]
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or len(set(cpus)) != len(cpus)):
        raise ValueError("cpu_ids must identify every worker's distinct CPU")
    if not isinstance(row["build_metadata"], dict) or not row["build_metadata"]:
        raise ValueError("build_metadata must be a nonempty object")
    if boolean(row["build_metadata"].get("host_test", False)):
        raise ValueError("host-test build is not hardware evidence")
    build = row["build_metadata"]
    if "publishable" in build and not boolean(build["publishable"]):
        raise ValueError("build is not publishable")
    if "status" in build and build["status"] not in ACCEPTED_STATUS:
        raise ValueError("build status is not OK/PASS")
    if "source_digest" in build and build["source_digest"] != row["source_digest"]:
        raise ValueError("row and build source provenance disagree")
    if "build_flags" in build:
        if not isinstance(build["build_flags"], list):
            raise ValueError("build_flags must be a list")
        flags = []
        skip = False
        for flag in build["build_flags"]:
            if skip:
                skip = False
                continue
            if flag in ("-I", "-include"):
                skip = True
                continue
            flags.append(flag)
        if skip:
            raise ValueError("build flag is missing its path argument")
        # Absolute build paths, object names and selected-source lists are not
        # experimental dimensions. Their content is covered by source_digest.
        volatile = {"project_root", "binary", "commands", "sources", "source_universe",
                    "executable_sha256", "rvv_kernel", "ime_kernel"}
        volatile.update(backend + "_" + suffix for backend in ("rvv", "ime")
                        for suffix in ("lmul", "tile", "unroll"))
        row["build_metadata"] = {k: v for k, v in build.items() if k not in volatile}
        row["build_metadata"]["build_flags"] = flags
    for field in OPTIONAL_BOOL_FIELDS:
        if field in row:
            row[field] = boolean(row[field])
    for field in ALL_METRICS:
        row[field] = number(row.get(field), nullable=field not in ("total_sec", "gops"))
    if row["total_sec"] <= 0:
        raise ValueError("total_sec must be positive")
    expected = 2 * row["M"] * row["N"] * row["K"] / row["total_sec"] / 1e9
    if not math.isclose(row["gops"], expected, rel_tol=1e-4, abs_tol=1e-9):
        raise ValueError("gops disagrees with 2*M*N*K/total_sec/1e9")
    # These phases are fused into the RVV kernel. A zero is not a measurement
    # of the fused component and must not become a zero-height plotted bar.
    if impl == "rvv" and any(row[p] is not None for p in ("output_sec", "boundary_sec")):
        raise ValueError("RVV output and boundary phases must be null (fused)")
    if row["profiled"] and row["kernel_sec"] is None:
        raise ValueError("profiled run lacks kernel timing")
    if not row["profiled"] and any(row[p] is not None for p in PHASES):
        raise ValueError("unprofiled component timings must be null")
    row["_config_fields"] = config_fields(row)
    return row


def load_accepted(campaign: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Only the exact accepted_runs.csv filename is allowed as input."""
    if not campaign.is_dir():
        raise ValueError("--input must be a campaign directory")
    source = campaign / "accepted_runs.csv"
    if not source.is_file() or source.is_symlink():
        raise ValueError("campaign must contain a regular accepted_runs.csv (no symlink)")
    rows, rejected = [], []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows, rejected
        for line, raw in enumerate(reader, 2):
            try:
                if None in raw:
                    raise ValueError("CSV row has more cells than its header")
                row = normalize_row(raw)
                row["_line"] = line
                rows.append(row)
            except (ValueError, TypeError, OverflowError) as exc:
                rejected.append({"line": line, "reason": str(exc)})
    # A repeat identifier cannot count twice in the same fixed configuration.
    seen: set[tuple[Any, ...]] = set()
    unique = []
    for row in rows:
        identity = configuration_key(row) + (("case_id", frozen(row.get("case_id", ""))),
                                              ("rep", frozen(row["rep"])))
        if identity in seen:
            rejected.append({"line": row["_line"], "reason": "duplicate fixed-configuration repetition"})
        else:
            seen.add(identity)
            unique.append(row)
    return unique, rejected


def configuration_key(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    fields = row.get("_config_fields", config_fields(row))
    return tuple((field, frozen(row.get(field))) for field in fields)


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(configuration_key(row), []).append(row)
    summaries, repetitions = [], {}
    for key, samples in sorted(groups.items(), key=lambda item: repr(item[0])):
        identifier = hashlib.sha256(frozen(key).encode()).hexdigest()[:16]
        result = {field: samples[0][field] for field, _ in key}
        result.update(config_id=identifier, repetitions=len(samples))
        for metric in ALL_METRICS:
            values = [r[metric] for r in samples if r[metric] is not None]
            # Partial instrumentation is not silently presented as complete.
            result[metric + "_samples"] = len(values)
            if len(values) != len(samples):
                for suffix in ("mean", "median", "std", "min", "max", "q25", "q75"):
                    result[metric + "_" + suffix] = None
                continue
            result.update({
                metric + "_mean": statistics.mean(values),
                metric + "_median": statistics.median(values),
                metric + "_std": statistics.stdev(values) if len(values) > 1 else None,
                metric + "_min": min(values), metric + "_max": max(values),
                metric + "_q25": quantile(values, .25), metric + "_q75": quantile(values, .75),
            })
        if any("counters_status" in sample for sample in samples):
            result["counters_statuses"] = sorted({sample.get("counters_status", "unspecified") for sample in samples})
        summaries.append(result)
        repetitions[identifier] = samples
    return summaries, repetitions


BACKEND_FIELDS = {backend + "_" + suffix for backend in ("rvv", "ime")
                  for suffix in ("kernel", "lmul", "tile", "unroll")}
IMPLEMENTATION_FIELDS = BACKEND_FIELDS | {"implementation", "rvv_workers", "ime_workers", "schedule",
    "rvv_output_fused", "boundary_scope", "output_scope"}


def primary(summary):
    return not summary["profiled"] and not boolean(summary.get("counters_requested", False))


def comparison_context(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain all declared environment/scope fields, including unknown ones."""
    return {key: value for key, value in summary.items()
            if key not in IMPLEMENTATION_FIELDS | {"config_id", "repetitions", "counters_statuses"}
            and not any(key.startswith(metric + "_") for metric in ALL_METRICS)}


def compatible_baselines(target: dict[str, Any], summaries: list[dict[str, Any]], backend: str = "rvv") -> list[dict[str, Any]]:
    if not primary(target):
        return []
    context = comparison_context(target)
    matches = []
    for candidate in summaries:
        if candidate["implementation"] != backend or not primary(candidate):
            continue
        if comparison_context(candidate) != context:
            continue
        if target["implementation"] == "mixed":
            if any(candidate.get(field) != target.get(field)
                   for field in BACKEND_FIELDS if field.startswith(backend + "_")):
                continue
        matches.append(candidate)
    return matches


def speedup_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for target in summaries:
        if target["implementation"] == "rvv":
            continue
        candidates = compatible_baselines(target, summaries)
        baseline = candidates[0] if len(candidates) == 1 else None
        reason = ("profiled runs excluded" if target["profiled"] else
                  "counter-instrumented runs excluded" if not primary(target) else
                  "unique matched RVV baseline" if baseline else
                  "ambiguous RVV baselines" if len(candidates) > 1 else
                  "no compatible RVV baseline")
        results.append({
            "config_id": target["config_id"], "implementation": target["implementation"],
            "baseline_config_id": baseline["config_id"] if baseline else None,
            "baseline_candidates": len(candidates), "speedup_reason": reason,
            "speedup_statistic": "RVV mean total_sec / target mean total_sec",
            "speedup_vs_rvv": baseline["total_sec_mean"] / target["total_sec_mean"] if baseline else None,
        })
    return results


def central_comparisons(summaries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The central eight-core comparison is RVV 8 vs the same static/dynamic 4+4."""
    result = []
    for mixed in summaries:
        if (mixed["implementation"] != "mixed" or mixed["threads"] != 8
                or not primary(mixed) or mixed["schedule"] != "static"
                or (mixed["rvv_workers"], mixed["ime_workers"]) != (4, 4)):
            continue
        rvv = compatible_baselines(mixed, summaries, "rvv")
        dynamic = [s for s in summaries
                   if s["implementation"] == "mixed" and s["schedule"] == "dynamic"
                   and (s["rvv_workers"], s["ime_workers"]) == (4, 4)
                   and primary(s) and comparison_context(s) == comparison_context(mixed)
                   and all(s.get(f) == mixed.get(f) for f in BACKEND_FIELDS)]
        if len(rvv) == 1 and len(dynamic) == 1:
            result.append([rvv[0], mixed, dynamic[0]])
    return result


def phase_shares(sample: dict[str, Any]) -> tuple[dict[str, float | None], str]:
    shares: dict[str, float | None] = {phase: None for phase in PHASES}
    if not sample["profiled"]:
        return shares, "not profiled"
    if sample["threads"] != 1:
        return shares, "NA: summed worker phases are not additive wall time"
    if not sample.get("phases_additive", True):
        return shares, "NA: phases not declared additive"
    scoped = list(PHASES)
    if not sample.get("packing_in_scope", sample["timing_mode"] == "end_to_end"):
        scoped.remove("packing_sec")
    phase_sum = sum(sample[phase] or 0 for phase in scoped)
    if phase_sum > sample["total_sec"] * 1.001 + 1e-9:
        return shares, "NA: measured phases exceed total elapsed scope"
    for phase in scoped:
        if sample[phase] is not None:
            shares[phase] = 100 * sample[phase] / sample["total_sec"]
    return shares, "single-thread total elapsed; prepacked setup excluded; RVV fused phases unavailable"


def section_v_rows(summaries: list[dict[str, Any]], repetitions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    speedups = {row["config_id"]: row for row in speedup_rows(summaries)}
    result = []
    for summary in summaries:
        row = dict(summary)
        scopes, shares = [], {phase: [] for phase in PHASES}
        for sample in repetitions[summary["config_id"]]:
            sample_shares, scope = phase_shares(sample)
            scopes.append(scope)
            for phase in PHASES:
                shares[phase].append(sample_shares[phase])
        row["phase_share_scope"] = "; ".join(sorted(set(scopes)))
        row["phase_time_interpretation"] = "sum of worker elapsed seconds; not a wall-time decomposition for multiple threads"
        row["kernel_label"] = "RVV kernel (includes fused output and boundary)" if summary["implementation"] == "rvv" else "kernel"
        for phase in PHASES:
            values = shares[phase]
            row[phase + "_share_pct_mean"] = statistics.mean(values) if all(v is not None for v in values) else None
        row.update(speedups.get(summary["config_id"], {}))
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], default_fields: tuple[str, ...]) -> None:
    fields = list(dict.fromkeys(default_fields + tuple(key for row in rows for key in row)))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (frozen(value) if isinstance(value, (dict, list, tuple)) else value)
                             for key, value in row.items()})


def plot_results(output: Path, summaries: list[dict[str, Any]], repetitions: dict[str, list[dict[str, Any]]]) -> list[str]:
    if not summaries:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["Matplotlib unavailable; CSV analysis completed. Install matplotlib to produce plots."]
    plots = output / "plots"
    plots.mkdir()
    colors = {"rvv": "#4878a8", "ime": "#ee854a", "mixed": "#6acc64"}

    def label(row: dict[str, Any]) -> str:
        return (f"{row['implementation']} {row['rvv_workers']}+{row['ime_workers']} "
                f"{row['schedule']} | {row['config_id'][:8]}")

    def save(fig: Any, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)

    # The box for each configuration uses only its repetitions. Full keys,
    # including affinity, seed, tuning and hardware, are in summary.csv.
    for mode in ("prepacked", "end_to_end"):
        fixed = [s for s in summaries if s["timing_mode"] == mode and primary(s)]
        if fixed:
            fig, ax = plt.subplots(figsize=(max(7, .55 * len(fixed)), 5))
            ax.boxplot([[r["total_sec"] * 1e3 for r in repetitions[s["config_id"]]] for s in fixed], showmeans=True)
            ax.set_xticks(list(range(1, len(fixed) + 1)), [label(s) for s in fixed])
            ax.tick_params(axis="x", rotation=70)
            ax.set(ylabel="Total elapsed time (ms)", title=f"Repeatability — {mode}; one box per fixed configuration")
            save(fig, f"repeatability_{mode}.png")

    contexts: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        context = frozen(comparison_context(summary))
        contexts.setdefault(context, []).append(summary)
    for context, group in contexts.items():
        first = group[0]
        tag = hashlib.sha256(context.encode()).hexdigest()[:12]
        title = f"{first['M']}×{first['N']}×{first['K']} — {first['timing_mode']} — {first['threads']} thread(s)"
        if first["threads"] == 1 and primary(first) and {g["implementation"] for g in group} >= {"rvv", "ime"}:
            chosen = [g for g in group if g["implementation"] in {"rvv", "ime"}]
            fig, ax = plt.subplots(figsize=(max(7, .8 * len(chosen)), 4.5))
            ax.bar([label(g) for g in chosen], [g["total_sec_mean"] * 1e3 for g in chosen],
                   color=[colors[g["implementation"]] for g in chosen])
            ax.tick_params(axis="x", rotation=45)
            ax.set(ylabel="Mean total elapsed time (ms)", title=title)
            save(fig, f"singlecore_rvv_ime_{first['timing_mode']}_{tag}.png")
        if first["threads"] == 1 and first["profiled"]:
            fig, ax = plt.subplots(figsize=(max(8, .8 * len(group)), 5))
            positions, bottom = list(range(len(group))), [0.] * len(group)
            for phase in PHASES:
                present, values = [], []
                for index, g in enumerate(group):
                    sample = repetitions[g["config_id"]][0]
                    in_scope = phase != "packing_sec" or sample.get("packing_in_scope", first["timing_mode"] == "end_to_end")
                    value = g[phase + "_mean"]
                    if value is not None and in_scope:
                        present.append(index)
                        values.append(value * 1e3)
                if present:
                    ax.bar(present, values, bottom=[bottom[index] for index in present],
                           label=phase.replace("_sec", ""))
                    for index, value in zip(present, values):
                        bottom[index] += value
            ax.set_xticks(positions, [label(g) + ("\nRVV output/boundary fused" if g["implementation"] == "rvv" else "") for g in group])
            ax.tick_params(axis="x", rotation=45)
            ax.set(ylabel="Profiled component elapsed time (ms)", title=title + "\nRVV kernel includes fused output/boundary; setup outside scope omitted")
            ax.legend()
            save(fig, f"singlecore_components_{first['timing_mode']}_{tag}.png")
        # Every tuning point is a complete configuration, never a pooled LMUL
        # mean from unrelated workloads, compilers, worker splits or machines.
        if primary(first) and len(group) > 1:
            fig, ax = plt.subplots(figsize=(max(7, .7 * len(group)), 4.5))
            ax.bar([label(g) for g in group], [g["total_sec_mean"] * 1e3 for g in group],
                   color=[colors[g["implementation"]] for g in group])
            ax.tick_params(axis="x", rotation=60)
            ax.set(ylabel="Mean total elapsed time (ms)", title="Fixed-configuration tuning — " + title)
            save(fig, f"tuning_{tag}.png")
    for chosen in central_comparisons(summaries):
        mixed = chosen[1]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(["RVV 8", "4 RVV + 4 IME\nstatic", "4 RVV + 4 IME\ndynamic"], [g["total_sec_mean"] * 1e3 for g in chosen],
               color=[colors[g["implementation"]] for g in chosen])
        ax.set(ylabel="Mean total elapsed time (ms)", title=f"8 cores — {mixed['M']}×{mixed['N']}×{mixed['K']} — {mixed['timing_mode']}\nMixed split {mixed['rvv_workers']} RVV + {mixed['ime_workers']} IME")
        save(fig, f"eight_core_three_way_{mixed['config_id']}.png")
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="campaign directory containing accepted_runs.csv")
    parser.add_argument("--output", required=True, type=Path, help="new output directory; never overwritten")
    parser.add_argument("--no-plots", action="store_true", help="produce tables without Matplotlib")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new directory")
        rows, rejected = load_accepted(args.input)
        summaries, repetitions = summarize(rows)
        args.output.mkdir(parents=True, exist_ok=False)
        write_csv(args.output / "summary.csv", summaries, ("config_id", "repetitions", "total_sec_mean", "gops_mean"))
        comparison = args.output / "comparison"
        comparison.mkdir()
        write_csv(comparison / "table_section_v.csv", section_v_rows(summaries, repetitions),
                  ("config_id", "implementation", "timing_mode", "total_sec_mean", "gops_mean"))
        write_csv(comparison / "speedups.csv", speedup_rows(summaries),
                  ("config_id", "baseline_config_id", "speedup_vs_rvv", "speedup_reason"))
        write_csv(args.output / "rejected_rows.csv", rejected, ("line", "reason"))
        notices = [] if args.no_plots else plot_results(args.output, summaries, repetitions)
        report = {
            "input": str((args.input / "accepted_runs.csv").resolve()),
            "accepted_repetitions": len(rows), "fixed_configurations": len(summaries),
            "rejected_rows": len(rejected), "notices": notices,
            "timing_statistic": "arithmetic mean of repetition total elapsed times",
            "gops_statistic": "arithmetic mean of validated per-repetition 2*M*N*K/total_sec/1e9",
            "dispersion": "sample standard deviation (NA for one sample); linearly interpolated quartiles",
            "speedup": "unique compatible unprofiled RVV mean total time / target mean total time",
            "phase_rule": "sum_worker_elapsed is not a multicore wall-time decomposition; RVV output/boundary are fused and null",
        }
        with (args.output / "analysis.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        if not rows:
            print("No accepted hardware data; no plots generated.")
        else:
            print(f"Analyzed {len(rows)} accepted repetitions in {len(summaries)} fixed configurations.")
        if rejected:
            print(f"Excluded {len(rejected)} rows; see rejected_rows.csv.")
        for notice in notices:
            print(notice, file=sys.stderr)
        return 0
    except (OSError, ValueError) as exc:
        print(f"Analysis refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
