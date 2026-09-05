#!/usr/bin/env python3
"""Create K1 paper figures from measured CSV files; no synthetic data is used."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
PROJECT = MODULE.parent
DEFAULT_RESULTS = MODULE / "results" / "paper_experiments"
DEFAULT_ACCURACY = (
    PROJECT
    / "RVV_IME_GEMM_ACCURACY_VALIDATION"
    / "int8_ime_vs_rvv_accuracy_summary_k1_latest.csv"
)

COLORS = {
    "RVV": "#0072B2",
    "IME": "#D55E00",
    "HET_STATIC": "#009E73",
    "HET_DYNAMIC": "#CC79A7",
    "STATIC": "#009E73",
    "DYNAMIC": "#CC79A7",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "lines.markersize": 4.5,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )


def read_real_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV has no measured rows: {path}")
    return frame


def clean_openmp(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "series",
        "parameter_value",
        "requested_threads",
        "M",
        "N",
        "K",
        "tile_N",
        "run",
        "status",
        "time_sec",
        "metric_value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    data = frame.loc[frame["status"].eq("OK")].copy()
    numeric = [
        "requested_threads",
        "M",
        "N",
        "K",
        "tile_N",
        "run",
        "time_sec",
        "metric_value",
        "unroll",
        "perf_ipc",
        "perf_cache_miss_rate",
    ]
    for column in numeric:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["time_sec", "metric_value"])


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"SAVED: {output_dir / (stem + '.png')}")
    print(f"SAVED: {output_dir / (stem + '.pdf')}")


def plot_strong_scaling(path: Path, output_dir: Path) -> None:
    data = clean_openmp(read_real_csv(path))
    timing = (
        data.groupby(["series", "requested_threads"], as_index=False)
        .agg(time_sec=("time_sec", "median"))
        .sort_values(["series", "requested_threads"])
    )
    # Run 1 performs the correctness gate. Later repetitions omit the
    # independent reference, so their process-level counters are comparable.
    counter_rows = data.loc[data["run"].gt(1)].copy()
    if counter_rows.empty:
        counter_rows = data
    counters = (
        counter_rows.groupby(["series", "requested_threads"], as_index=False)
        .agg(
            ipc=("perf_ipc", "mean"),
            cache_miss_rate=("perf_cache_miss_rate", "mean"),
        )
    )
    grouped = timing.merge(counters, on=["series", "requested_threads"], how="left")

    have_ipc = grouped["ipc"].notna().any()
    if have_ipc:
        fig = plt.figure(figsize=(7.2, 4.3))
        ax = fig.add_subplot(111, projection="3d")
        for series, part in grouped.groupby("series", sort=False):
            ax.plot(
                part["requested_threads"],
                part["time_sec"],
                part["ipc"],
                marker="o",
                color=COLORS.get(series, "#444444"),
                label=series.replace("_", " "),
            )
            for row in part.itertuples():
                if pd.notna(row.cache_miss_rate):
                    ax.text(
                        row.requested_threads,
                        row.time_sec,
                        row.ipc,
                        f" {row.cache_miss_rate:.1f}%",
                        fontsize=6,
                    )
        ax.set_xlabel("Cores")
        ax.set_ylabel("Execution time (s)")
        ax.set_zlabel("Instructions per cycle")
        ax.set_title("K1 INT8 GEMM Strong Scaling")
        ax.legend(frameon=False, loc="best")
        ax.view_init(elev=22, azim=-55)
        fig.text(0.02, 0.02, "Point labels: cache-miss rate", fontsize=7)
    else:
        warnings.warn("No perf counters found; strong-scaling figure uses time only.")
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for series, part in grouped.groupby("series", sort=False):
            ax.plot(
                part["requested_threads"],
                part["time_sec"],
                marker="o",
                color=COLORS.get(series, "#444444"),
                label=series.replace("_", " "),
            )
        ax.set(xlabel="Cores", ylabel="Execution time (s)", title="K1 INT8 GEMM Strong Scaling")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    save_figure(fig, output_dir, "k1_strong_scaling")


def plot_weak_scaling(path: Path, output_dir: Path) -> None:
    data = clean_openmp(read_real_csv(path))
    grouped = (
        data.groupby(["series", "requested_threads", "M"], as_index=False)["time_sec"]
        .median()
        .sort_values(["series", "requested_threads"])
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for series, part in grouped.groupby("series", sort=False):
        ax.plot(
            part["requested_threads"],
            part["time_sec"],
            marker="o",
            color=COLORS.get(series, "#444444"),
            label=series,
        )
        for row in part.itertuples():
            ax.annotate(
                f"{int(row.M)}³",
                (row.requested_threads, row.time_sec),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6,
            )
    ax.set(xlabel="Cores", ylabel="Execution time (s)", title="K1 INT8 GEMM Weak Scaling")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "k1_weak_scaling")


def plot_partitioning(path: Path, output_dir: Path) -> None:
    data = clean_openmp(read_real_csv(path))
    grouped = (
        data.groupby(["series", "parameter_value"], as_index=False)
        .agg(time_sec=("time_sec", "median"), throughput=("metric_value", "median"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    static = grouped.loc[grouped["series"].eq("STATIC")].copy()
    if not static.empty:
        static["order"] = static["parameter_value"].map(
            lambda value: float(str(value).split(":")[0])
            / sum(float(x) for x in str(value).split(":"))
        )
        static = static.sort_values("order")
        axes[0].plot(static["parameter_value"], static["time_sec"], marker="o", color=COLORS["STATIC"])
    axes[0].set(xlabel="Static IME:RVV ratio", ylabel="Execution time (s)", title="Static Partitioning")

    dynamic = grouped.loc[grouped["series"].eq("DYNAMIC")].copy()
    if not dynamic.empty:
        dynamic["chunk"] = pd.to_numeric(dynamic["parameter_value"], errors="coerce")
        dynamic = dynamic.sort_values("chunk")
        axes[1].plot(dynamic["chunk"], dynamic["time_sec"], marker="o", color=COLORS["DYNAMIC"])
    axes[1].set(xlabel="Dynamic chunk (output strips)", ylabel="Execution time (s)", title="Dynamic Partitioning")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("K1 Heterogeneous INT8 GEMM Partitioning", y=1.02, fontsize=10)
    fig.tight_layout()
    save_figure(fig, output_dir, "k1_partitioning")


def plot_tuning(path: Path, output_dir: Path) -> None:
    data = clean_openmp(read_real_csv(path))
    required = {"tile_shape", "lmul", "unroll", "tile_N"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing tuning columns: {', '.join(missing)}")

    measured = (
        data.groupby(["series", "tile_shape", "lmul", "unroll", "tile_N"], as_index=False)
        ["metric_value"]
        .median()
    )
    best_index = measured.groupby(["series", "tile_shape", "lmul", "unroll"])["metric_value"].idxmax()
    best = measured.loc[best_index].sort_values(["series", "tile_shape", "lmul", "unroll"])

    panel_keys = [(series, tile) for series in ("RVV", "IME") for tile in ("8x4", "8x8")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for ax, (series, tile_shape) in zip(axes.flat, panel_keys):
        panel = best.loc[(best["series"] == series) & (best["tile_shape"] == tile_shape)]
        for lmul, part in panel.groupby("lmul", sort=False):
            part = part.sort_values("unroll")
            ax.plot(part["unroll"], part["metric_value"], marker="o", label=f"LMUL {lmul}")
            for row in part.itertuples():
                ax.annotate(
                    f"T{int(row.tile_N)}",
                    (row.unroll, row.metric_value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5.5,
                )
        ax.set_title(f"{series} {tile_shape}")
        ax.set_ylabel("Throughput (GOPS)")
        ax.grid(axis="y", alpha=0.25)
        if not panel.empty:
            ax.legend(frameon=False, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("K-loop unroll factor")
    fig.suptitle("K1 INT8 Kernel Tuning", y=1.01, fontsize=10)
    fig.tight_layout()
    save_figure(fig, output_dir, "k1_kernel_tuning")


def plot_accuracy(path: Path, output_dir: Path) -> None:
    frame = read_real_csv(path)
    required = {"path", "status", "exact_match_rate", "mismatch_count", "max_integer_difference", "overflow_count"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing accuracy columns: {', '.join(missing)}")

    data = frame.loc[frame["status"].eq("OK")].copy()
    if data.empty:
        raise ValueError(f"Accuracy CSV has no successful measured rows: {path}")
    for column in ("exact_match_rate", "mismatch_count", "max_integer_difference", "overflow_count"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    grouped = (
        data.groupby("path", as_index=False)
        .agg(
            exact_match_rate=("exact_match_rate", "mean"),
            mismatches=("mismatch_count", "sum"),
            max_error=("max_integer_difference", "max"),
            overflows=("overflow_count", "sum"),
        )
    )
    grouped["exact_percent"] = 100.0 * grouped["exact_match_rate"]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    bars = ax.bar(grouped["path"], grouped["exact_percent"], color=["#D55E00", "#0072B2"][: len(grouped)])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact output values (%)")
    ax.set_title("K1 INT8 GEMM Accuracy")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, grouped.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 1, 102),
            f"{bar.get_height():.2f}%\nerr={int(row.max_error)}, ovf={int(row.overflows)}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    save_figure(fig, output_dir, "k1_accuracy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--accuracy-csv", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    configure_style()
    jobs = [
        (plot_strong_scaling, args.results_root / "k1_strong_scaling_raw_latest.csv"),
        (plot_weak_scaling, args.results_root / "k1_weak_scaling_raw_latest.csv"),
        (plot_partitioning, args.results_root / "k1_partitioning_raw_latest.csv"),
        (plot_tuning, args.results_root / "k1_kernel_tuning_raw_latest.csv"),
        (plot_accuracy, args.accuracy_csv),
    ]

    completed = 0
    for function, csv_path in jobs:
        if not csv_path.is_file():
            print(f"SKIPPED: missing measured data {csv_path}")
            continue
        function(csv_path, args.output_dir)
        completed += 1

    if completed == 0:
        print("No measured K1 CSV files were found. Run the experiment scripts first.")
        return 1
    print(f"DONE: created {completed} K1 figure set(s) from measured data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
