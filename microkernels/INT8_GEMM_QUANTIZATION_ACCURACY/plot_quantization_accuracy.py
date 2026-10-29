#!/usr/bin/env python3
"""Validate a completed quantization experiment and plot its recorded errors.

This script consumes actual CSV observations and their sibling ``manifest.json``.
It never generates measurements or substitutes missing RVV/IME results. Install
the normal plotting dependencies with ``python -m pip install matplotlib numpy``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


CSV_COLUMNS = (
    "run_id", "execution_kind", "distribution", "scheme", "sample_index",
    "seed", "m", "n", "k", "backend", "status", "mismatches",
    "max_integer_error", "reference_overflow", "relative_frobenius_error",
    "rmse", "max_absolute_error", "normalized_max_error", "sqnr_db",
    "reference_frobenius_norm",
)
DISTRIBUTIONS = ("uniform", "gaussian", "outliers")
SCHEMES = ("per_tensor", "per_row_column")
BACKENDS = {"native_k1": ("rvv", "ime"), "software_verification": ("scalar",)}
METRIC_COLUMNS = (
    "relative_frobenius_error", "rmse", "max_absolute_error",
    "normalized_max_error", "sqnr_db", "reference_frobenius_norm",
)
SOFTWARE_WARNING = "SOFTWARE VERIFICATION - NOT RVV/IME MEASUREMENTS"
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")


class ValidationError(ValueError):
    """The CSV and manifest do not describe a complete, valid experiment."""


def _integer(value: str, field: str, context: str, minimum: int = 0) -> int:
    if not isinstance(value, str) or not _INTEGER.fullmatch(value.strip()):
        raise ValidationError(f"{context}: {field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValidationError(f"{context}: {field} must be >= {minimum}")
    return result


def _manifest_integer(manifest: dict[str, Any], field: str) -> int:
    value = manifest.get(field)
    if type(value) is not int or value < 1:
        raise ValidationError(f"manifest: {field} must be a positive integer")
    return value


def _manifest_members(manifest: dict[str, Any], field: str,
                      expected: tuple[str, ...]) -> None:
    value = manifest.get(field)
    if (not isinstance(value, list)
            or not all(isinstance(item, str) for item in value)
            or len(value) != len(expected) or set(value) != set(expected)):
        raise ValidationError(
            f"manifest: {field} must contain exactly {', '.join(expected)}"
        )


def _load_manifest(csv_path: Path, allow_software_verification: bool) -> dict[str, Any]:
    manifest_path = csv_path.parent / "manifest.json"
    try:
        with manifest_path.open(encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot read valid manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValidationError("manifest: the root must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValidationError("manifest: schema_version must be 1")
    if manifest.get("status") != "complete":
        raise ValidationError("manifest: experiment is incomplete (status must be 'complete')")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].strip():
        raise ValidationError("manifest: run_id must be a nonempty string")
    execution_kind = manifest.get("execution_kind")
    if not isinstance(execution_kind, str) or execution_kind not in BACKENDS:
        raise ValidationError("manifest: execution_kind must be native_k1 or software_verification")
    if execution_kind == "software_verification" and not allow_software_verification:
        raise ValidationError(
            "software-verification data are not native K1 measurements; pass "
            "--allow-software-verification to export explicitly labeled verification plots"
        )
    for field in ("m", "n", "k", "samples"):
        _manifest_integer(manifest, field)
    _manifest_members(manifest, "distributions", DISTRIBUTIONS)
    _manifest_members(manifest, "schemes", SCHEMES)
    _manifest_members(manifest, "backends", BACKENDS[execution_kind])
    expected_rows = manifest["samples"] * len(DISTRIBUTIONS) * len(SCHEMES) * len(BACKENDS[execution_kind])
    for field in ("planned_rows", "completed_rows"):
        if field in manifest and manifest[field] != expected_rows:
            raise ValidationError(f"manifest: {field} disagrees with expected complete coverage")
    return manifest


def _metric(value: str, field: str, context: str) -> Decimal:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValidationError(f"{context}: {field} is not numeric") from exc
    if number.is_nan():
        raise ValidationError(f"{context}: {field} must not be NaN")
    if field == "sqnr_db":
        if number == Decimal("-Infinity"):
            raise ValidationError(f"{context}: sqnr_db must be finite or +inf for zero error")
    elif not number.is_finite() or number < 0:
        raise ValidationError(f"{context}: {field} must be finite and nonnegative")
    # A finite Decimal outside double range cannot be represented by the plotter.
    if number.is_finite() and not math.isfinite(float(number)):
        raise ValidationError(f"{context}: {field} exceeds the supported numeric range")
    return number


def load_and_validate(csv_path: str | Path,
                      allow_software_verification: bool = False
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return parsed observations and manifest, or raise ``ValidationError``.

    Coverage is exact: each distribution/scheme/backend contains sample indices
    0..samples-1 once. Native backend metrics must agree numerically without any
    tolerance, and each sample uses the same seed across schemes and backends.
    Distinct sample indices use distinct seeds within each distribution.
    Decimal parsing preserves the precision actually recorded in the CSV.
    """
    csv_path = Path(csv_path)
    manifest = _load_manifest(csv_path, allow_software_verification)
    if "data_sha256" in manifest:
        expected_hash = manifest["data_sha256"]
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValidationError("manifest: data_sha256 must be a SHA-256 hexadecimal digest")
        digest = hashlib.sha256()
        try:
            with csv_path.open("rb") as hash_source:
                for chunk in iter(lambda: hash_source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValidationError(f"cannot read CSV {csv_path}: {exc}") from exc
        if digest.hexdigest() != expected_hash.lower():
            raise ValidationError("CSV SHA-256 does not match manifest data_sha256")
    expected_backends = BACKENDS[manifest["execution_kind"]]
    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str, int]] = set()
    counts: Counter[tuple[str, str, str]] = Counter()
    sample_seeds: dict[tuple[str, int], int] = {}
    seed_indices: dict[tuple[str, int], int] = {}
    paired: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    try:
        handle = csv_path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise ValidationError(f"cannot read CSV {csv_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValidationError("CSV header must match schema v1 exactly: " + ",".join(CSV_COLUMNS))
        for raw in reader:
            context = f"CSV line {reader.line_num}"
            if None in raw or any(value is None or value == "" for value in raw.values()):
                raise ValidationError(f"{context}: missing or extra CSV fields")
            row: dict[str, Any] = dict(raw)
            for field in ("run_id", "execution_kind"):
                if row[field] != manifest[field]:
                    raise ValidationError(f"{context}: {field} does not match manifest")
            for field in ("m", "n", "k"):
                row[field] = _integer(row[field], field, context, minimum=1)
                if row[field] != manifest[field]:
                    raise ValidationError(f"{context}: {field} does not match manifest")
            if row["distribution"] not in DISTRIBUTIONS:
                raise ValidationError(f"{context}: unexpected distribution {row['distribution']!r}")
            if row["scheme"] not in SCHEMES:
                raise ValidationError(f"{context}: unexpected scheme {row['scheme']!r}")
            if row["backend"] not in expected_backends:
                raise ValidationError(f"{context}: backend does not match execution_kind/manifest")
            if row["status"] != "PASS":
                raise ValidationError(f"{context}: only status=PASS observations can be plotted")
            for field in ("sample_index", "seed", "mismatches", "max_integer_error", "reference_overflow"):
                row[field] = _integer(row[field], field, context)
            if row["sample_index"] >= manifest["samples"]:
                raise ValidationError(f"{context}: sample_index is outside 0..samples-1")
            for field in ("mismatches", "max_integer_error", "reference_overflow"):
                if row[field] != 0:
                    raise ValidationError(f"{context}: correctness field {field} must be zero")
            for field in METRIC_COLUMNS:
                row[field] = _metric(row[field], field, context)
            if row["sqnr_db"].is_infinite() and row["relative_frobenius_error"] != 0:
                raise ValidationError(f"{context}: +inf sqnr_db is only valid for zero error")
            if row["reference_frobenius_norm"] == 0 and any(
                    row[field] != 0 for field in (
                        "relative_frobenius_error", "rmse", "max_absolute_error", "normalized_max_error")):
                raise ValidationError(f"{context}: zero reference norm requires zero error metrics")
            # Convert only after validation; percentages must also be finite.
            if not math.isfinite(float(row["relative_frobenius_error"]) * 100.0):
                raise ValidationError(f"{context}: relative_frobenius_error is too large to plot as a percentage")
            group = (row["distribution"], row["scheme"], row["backend"])
            key = (*group, row["sample_index"])
            if key in observed:
                raise ValidationError(f"{context}: duplicate observation for {key}")
            observed.add(key)
            counts[group] += 1
            seed_key = (row["distribution"], row["sample_index"])
            if seed_key in sample_seeds and sample_seeds[seed_key] != row["seed"]:
                raise ValidationError(f"{context}: sample seed differs across schemes/backends for {seed_key}")
            sample_seeds[seed_key] = row["seed"]
            seed_owner = (row["distribution"], row["seed"])
            if seed_owner in seed_indices and seed_indices[seed_owner] != row["sample_index"]:
                raise ValidationError(
                    f"{context}: duplicate seed {row['seed']} within {row['distribution']} "
                    f"for sample indices {seed_indices[seed_owner]} and {row['sample_index']}"
                )
            seed_indices[seed_owner] = row["sample_index"]
            paired[(row["distribution"], row["scheme"], row["sample_index"])][row["backend"]] = row
            rows.append(row)
    for distribution in DISTRIBUTIONS:
        for scheme in SCHEMES:
            for backend in expected_backends:
                group = (distribution, scheme, backend)
                if counts[group] != manifest["samples"]:
                    raise ValidationError(
                        f"incomplete case coverage for {group}: expected {manifest['samples']} "
                        f"samples, found {counts[group]}"
                    )
    if manifest["execution_kind"] == "native_k1":
        for key, backend_rows in paired.items():
            for field in METRIC_COLUMNS:
                if backend_rows["rvv"][field] != backend_rows["ime"][field]:
                    raise ValidationError(f"native RVV/IME metrics differ for {key}: {field}")
    return rows, manifest


def plot_accuracy(rows: list[dict[str, Any]], manifest: dict[str, Any],
                  output_dir: str | Path) -> list[Path]:
    """Export PNG and SVG boxplots of already-validated observations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib and numpy; install them with "
                           "python -m pip install matplotlib numpy") from exc

    software = manifest["execution_kind"] == "software_verification"
    backends = BACKENDS[manifest["execution_kind"]]
    colors = {"rvv": "#2563EB", "ime": "#F97316", "scalar": "#64748B"}
    labels = {"rvv": "RVV", "ime": "IME", "scalar": "Scalar (software verification)"}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["distribution"], row["scheme"], row["backend"])].append(
            float(row["relative_frobenius_error"]) * 100.0
        )
    all_values = [value for values in grouped.values() for value in values]
    upper = max(all_values)
    # A linear axis retains exact zero samples; avoid a collapsed all-zero range.
    y_upper = upper * 1.10 if upper > 0 else 1.0
    if not math.isfinite(y_upper):
        y_upper = upper
    rc = {
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=True)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28,
                            top=0.75 if software else 0.82, wspace=0.10)
        fig.suptitle("INT8 GEMM quantization accuracy", x=0.535, y=0.975,
                     fontsize=17, fontweight="semibold")
        if software:
            fig.text(0.535, 0.883, SOFTWARE_WARNING, ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#B91C1C",
                     bbox={"facecolor": "#FEF2F2", "edgecolor": "#B91C1C", "pad": 9})
        else:
            fig.text(0.535, 0.905, "Native K1 measurements", ha="center", color="#475569")
        for ax, scheme in zip(axes, SCHEMES):
            ax.set_title("Per-tensor quantization" if scheme == "per_tensor"
                         else "Per-row A / per-column B quantization", pad=12)
            centers = np.arange(len(DISTRIBUTIONS), dtype=float) + 1.0
            for backend_index, backend in enumerate(backends):
                offset = (backend_index - (len(backends) - 1) / 2.0) * 0.30
                values = [np.asarray(grouped[(distribution, scheme, backend)], dtype=float)
                          for distribution in DISTRIBUTIONS]
                boxes = ax.boxplot(
                    values, positions=centers + offset,
                    widths=0.25 if len(backends) == 2 else 0.34,
                    patch_artist=True, manage_ticks=False, whis=1.5,
                    boxprops={"facecolor": colors[backend], "edgecolor": colors[backend],
                              "alpha": 0.72, "linewidth": 1.3},
                    medianprops={"color": "#111827", "linewidth": 1.5},
                    whiskerprops={"color": colors[backend], "linewidth": 1.2},
                    capprops={"color": colors[backend], "linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "none", "markeredgecolor": colors[backend],
                                "alpha": 0.6},
                )
                for artist in boxes["boxes"]:
                    artist.set_zorder(3)
            ax.set_xticks(centers)
            ax.set_xticklabels(["Uniform", "Gaussian", "Outliers"])
            ax.set_xlim(0.5, len(DISTRIBUTIONS) + 0.5)
            ax.set_ylim(0, y_upper)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-3, 4))
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis="both", length=0, pad=7)
        axes[0].set_ylabel("Relative Frobenius error (%)", labelpad=10)
        legend = [Patch(facecolor=colors[backend], edgecolor=colors[backend],
                        alpha=0.72, label=labels[backend]) for backend in backends]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.535, 0.173),
                   ncol=len(backends), frameon=False, handlelength=1.7, columnspacing=2.2)
        dimensions = f"M × N × K = {manifest['m']} × {manifest['n']} × {manifest['k']}"
        fig.text(0.535, 0.127,
                 f"{dimensions}  |  n = {manifest['samples']} per distribution, scheme and backend",
                 ha="center", va="center", fontsize=10, color="#334155")
        fig.text(0.535, 0.079,
                 "Boxes: interquartile range; line: median; whiskers: 1.5 × IQR; circles: outliers.",
                 ha="center", va="center", fontsize=9, color="#475569")
        fig.text(0.535, 0.035,
                 "Error against FP64 GEMM on the original inputs; lower values indicate less quantization error.",
                 ha="center", va="center", fontsize=9, color="#475569")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "quantization_accuracy_" + manifest["execution_kind"]
        outputs = [output_dir / f"{stem}.{extension}" for extension in ("png", "svg")]
        description = SOFTWARE_WARNING if software else "Native K1 RVV and IME quantization measurements"
        try:
            fig.savefig(outputs[0], dpi=220,
                        metadata={"Title": "INT8 GEMM quantization accuracy", "Description": description})
            fig.savefig(outputs[1],
                        metadata={"Title": "INT8 GEMM quantization accuracy", "Description": description})
        finally:
            plt.close(fig)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path,
                        help="experiment CSV; its directory must contain a complete manifest.json")
    parser.add_argument("--output-dir", type=Path,
                        help="export directory (default: CSV directory / plots)")
    parser.add_argument("--allow-software-verification", action="store_true",
                        help="explicitly allow scalar verification data, with a warning on every export")
    args = parser.parse_args(argv)
    try:
        rows, manifest = load_and_validate(args.csv, args.allow_software_verification)
        outputs = plot_accuracy(rows, manifest, args.output_dir or args.csv.parent / "plots")
    except (ValidationError, OSError, RuntimeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if manifest["execution_kind"] == "software_verification":
        print(SOFTWARE_WARNING, file=sys.stderr)
    print(f"Validated {len(rows)} observations from run {manifest['run_id']}.")
    for path in outputs:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
