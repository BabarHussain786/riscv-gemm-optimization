#!/usr/bin/env python3
"""Evaluate quantization error around the existing signed INT8 GEMM kernels.

Native K1 measurements and local software verification are deliberately distinct.
This experiment uses alpha=1, initial C=0, and symmetric [-127,127] quantization.
It measures matrix-level quantization error, not neural-network task accuracy.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
DISTRIBUTIONS = ("uniform", "gaussian", "outliers")
SCHEMES = ("per_tensor", "per_row_column")
FIELDS = (
    "run_id", "execution_kind", "distribution", "scheme", "sample_index", "seed",
    "m", "n", "k", "backend", "status", "mismatches", "max_integer_error",
    "reference_overflow", "relative_frobenius_error", "rmse", "max_absolute_error",
    "normalized_max_error", "sqnr_db", "reference_frobenius_norm",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    """Publish metadata atomically; interrupted runs cannot look complete."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def input_seed(base_seed, distribution_index, sample_index):
    state = np.random.SeedSequence([base_seed, distribution_index, sample_index])
    return int(state.generate_state(1, dtype=np.uint64)[0])


def generate_inputs(distribution, seed, m, n, k):
    """Draw one pair; reuse it for every scheme and backend in this sample."""
    rng = np.random.Generator(np.random.PCG64(seed))
    if distribution == "uniform":
        a = rng.uniform(-1.0, 1.0, size=(m, k))
        b = rng.uniform(-1.0, 1.0, size=(k, n))
    elif distribution in ("gaussian", "outliers"):
        a = rng.standard_normal((m, k))
        b = rng.standard_normal((k, n))
        if distribution == "outliers":
            # Large-amplitude rows/columns test output-axis scale granularity.
            # These are synthetic stress inputs, not claimed model activations.
            rows = rng.choice(m, size=max(1, math.ceil(0.05 * m)), replace=False)
            columns = rng.choice(n, size=max(1, math.ceil(0.05 * n)), replace=False)
            a[rows, :] *= 20.0
            b[:, columns] *= 20.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def quantize_inputs(a, b, scheme):
    """Round to nearest-even; zero-valued groups use scale 1 and code 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("A and B must be compatible two-dimensional matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Inputs must be finite")
    if scheme == "per_tensor":
        sa = np.full(a.shape[0], float(np.max(np.abs(a))) / 127.0)
        sb = np.full(b.shape[1], float(np.max(np.abs(b))) / 127.0)
    elif scheme == "per_row_column":
        sa = np.max(np.abs(a), axis=1) / 127.0
        sb = np.max(np.abs(b), axis=0) / 127.0
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    sa = np.where(sa == 0.0, 1.0, sa)
    sb = np.where(sb == 0.0, 1.0, sb)
    qa = np.clip(np.rint(a / sa[:, None]), -127, 127).astype(np.int8)
    qb = np.clip(np.rint(b / sb[None, :]), -127, 127).astype(np.int8)
    return qa, qb, sa, sb


def integer_reference(qa, qb):
    # INT64 products and accumulation are exact for the enforced K bound.
    value = qa.astype(np.int64) @ qb.astype(np.int64)
    limits = np.iinfo(np.int32)
    overflow = int(np.count_nonzero((value < limits.min) | (value > limits.max)))
    return value, overflow


def compare_integer(actual, expected):
    difference = actual.astype(np.int64) - expected
    return int(np.count_nonzero(difference)), int(np.max(np.abs(difference)))


def dequantize(output, sa, sb):
    return output.astype(np.float64) * sa[:, None] * sb[None, :]


def error_metrics(reference, actual):
    """Return dimensionless errors as fractions; the plot converts to percent."""
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Metric operands must have matching shapes and finite values")
    delta = actual - reference
    norm = float(np.linalg.norm(reference.ravel()))
    error_norm = float(np.linalg.norm(delta.ravel()))
    maximum = float(np.max(np.abs(delta)))
    reference_max = float(np.max(np.abs(reference)))
    if not all(math.isfinite(x) for x in (norm, error_norm, maximum, reference_max)):
        raise ValueError("Metric reduction overflowed")
    if norm == 0.0 and error_norm != 0.0:
        raise ValueError("Relative error is undefined for a zero reference with a nonzero result")
    relative = error_norm / norm if norm else 0.0
    return {
        "relative_frobenius_error": relative,
        "rmse": error_norm / math.sqrt(reference.size),
        "max_absolute_error": maximum,
        "normalized_max_error": maximum / reference_max if reference_max else 0.0,
        # Explicit reporting convention: all exact results have SQNR=+infinity.
        "sqnr_db": -20.0 * math.log10(relative) if relative else math.inf,
        "reference_frobenius_norm": norm,
    }


class SoftwareBackend:
    """Exercise the experiment math locally; never label this RVV or IME."""
    execution_kind = "software_verification"
    names = ("scalar",)

    def run(self, name, qa, qb):
        if name != "scalar":
            raise ValueError("Software mode has no RVV/IME measurements")
        result, overflow = integer_reference(qa, qb)
        if overflow:
            raise ValueError("Software result exceeds INT32")
        return result.astype(np.int32)


class NativeBackend:
    execution_kind = "native_k1"
    names = ("rvv", "ime")

    def __init__(self, build_dir, rvv_cpu=4, ime_cpu=0):
        if platform.system() != "Linux" or not platform.machine().lower().startswith("riscv64"):
            raise ValueError("Native measurements require the Linux RISC-V K1; use --software-verification only for local checks")
        build_dir = Path(build_dir).resolve()
        self.metadata = json.loads((build_dir / "build.json").read_text(encoding="utf-8"))
        if self.metadata.get("status") != "complete" or self.metadata.get("execution_kind") != "native_k1":
            raise ValueError("A complete native build is required")
        library = Path(self.metadata["library"])
        if sha256(library) != self.metadata["library_sha256"]:
            raise ValueError("Native library does not match its recorded build hash")
        self.library = ctypes.CDLL(str(library))
        self.library.qa_abi_version.restype = ctypes.c_int
        if self.library.qa_abi_version() != 1:
            raise ValueError("Unsupported native bridge ABI")
        self.library.qa_run.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_long,
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.library.qa_run.restype = ctypes.c_int
        self.cpus = {"rvv": rvv_cpu, "ime": ime_cpu}

    def run(self, name, qa, qb):
        m, k = qa.shape
        _, n = qb.shape
        # Match the existing adapters' actual contracts, not nominal comments.
        a = np.asfortranarray(qa, dtype=np.int8)
        b = np.ascontiguousarray(qb, dtype=np.int8)
        c = np.zeros((m, n), dtype=np.int32, order="F")
        rc = self.library.qa_run(
            self.names.index(name), self.cpus[name], m, n, k,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if rc:
            raise RuntimeError(f"{name} native kernel failed (bridge code {rc}, CPU {self.cpus[name]})")
        return c


def run_experiment(args):
    if args.pilot:
        args.m = args.n = args.k = 64
        args.samples = 3
    for label in ("m", "n", "k", "samples"):
        if getattr(args, label) <= 0:
            raise ValueError(f"{label} must be positive")
    if args.m % 8 or args.n % 4 or args.k % 8:
        raise ValueError("Use M divisible by 8, N by 4 and K by 8; this experiment excludes scalar tail cleanup")
    if args.k > 131071:
        raise ValueError("K exceeds the conservative overflow-free INT32 bound")
    if not 0 <= args.seed <= 0xFFFFFFFF:
        raise ValueError("The base seed must be a uint32 integer")
    if args.rvv_cpu < 0 or args.ime_cpu < 0 or args.rvv_cpu == args.ime_cpu:
        raise ValueError("Specify distinct nonnegative RVV and IME CPU numbers")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = (args.output_dir or HERE / "results" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    kind = "software_verification" if args.software_verification else "native_k1"
    names = ["scalar"] if args.software_verification else ["rvv", "ime"]
    manifest = {
        "schema_version": 1, "run_id": run_id, "status": "running", "execution_kind": kind,
        "m": args.m, "n": args.n, "k": args.k, "samples": args.samples,
        "distributions": list(DISTRIBUTIONS), "schemes": list(SCHEMES), "backends": names,
        "base_seed": args.seed, "alpha": 1, "initial_C": 0,
        "tile": "8x4", "quantized_range": [-127, 127], "rounding": "nearest_even",
        "calibration": "dynamic absmax per input pair; no intentional clipping",
        "reference": "NumPy float64 A@B on original real inputs; not exact real arithmetic",
        "integer_reference": "NumPy int64 A_q@B_q; exact within enforced range",
        "metric": "relative Frobenius error as a fraction; plot multiplies by 100",
        "zero_convention": "zero reference and zero result: error=0; nonzero result: undefined and rejected; exact-result SQNR=+inf",
        "rng": "PCG64; seeds from SeedSequence([base_seed, distribution_index, sample_index])",
        "input_parameters": {
            "uniform": "independent Uniform[-1,1] A and B",
            "gaussian": "independent standard normal A and B",
            "outliers": "standard normal; ceil(5% of A rows) and ceil(5% of B columns), at least one each, multiplied by 20",
        },
        "planned_rows": args.samples * len(DISTRIBUTIONS) * len(SCHEMES) * len(names),
        "completed_rows": 0, "pilot": bool(args.pilot),
        "software": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "evaluation_source_hashes": {p.name: sha256(p) for p in HERE.glob("*.py")},
        "note": "Correct backends should have identical quantization error; no task-level accuracy claim.",
    }
    save_json(manifest_path, manifest)
    try:
        if args.software_verification:
            backend = SoftwareBackend()
        else:
            build_dir = args.build_dir.resolve() if args.build_dir else run_dir / "build"
            if not args.build_dir:
                subprocess.run([sys.executable, str(HERE / "build_native.py"),
                                "--output-dir", str(build_dir)], check=True)
            backend = NativeBackend(build_dir, args.rvv_cpu, args.ime_cpu)
            manifest["native_build"] = backend.metadata
            manifest["cpu_affinity"] = {"rvv": args.rvv_cpu, "ime": args.ime_cpu}
            manifest["cpuinfo"] = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            save_json(manifest_path, manifest)

        csv_path = run_dir / "quantization_accuracy.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for distribution_index, distribution in enumerate(DISTRIBUTIONS):
                for sample_index in range(args.samples):
                    seed = input_seed(args.seed, distribution_index, sample_index)
                    a, b = generate_inputs(distribution, seed, args.m, args.n, args.k)
                    high_precision = a @ b
                    for scheme in SCHEMES:
                        qa, qb, sa, sb = quantize_inputs(a, b, scheme)
                        reference, overflow = integer_reference(qa, qb)
                        paired_outputs = None
                        for name in names:
                            manifest["last_case"] = {
                                "distribution": distribution, "scheme": scheme,
                                "sample_index": sample_index, "seed": seed, "backend": name,
                            }
                            row = dict(run_id=run_id, execution_kind=kind, distribution=distribution,
                                       scheme=scheme, sample_index=sample_index, seed=seed,
                                       m=args.m, n=args.n, k=args.k, backend=name)
                            if overflow:
                                raise RuntimeError("INT64 result does not fit INT32")
                            output = backend.run(name, qa, qb)
                            mismatches, maximum = compare_integer(output, reference)
                            row.update(status="PASS" if mismatches == 0 else "FAIL",
                                       mismatches=mismatches, max_integer_error=maximum,
                                       reference_overflow=overflow)
                            if mismatches:
                                writer.writerow(row)
                                stream.flush()
                                raise RuntimeError(f"Kernel mismatch: {distribution}, {scheme}, sample {sample_index}, {name}")
                            if paired_outputs is not None and not np.array_equal(paired_outputs, output):
                                raise RuntimeError("RVV/IME integer outputs disagree")
                            paired_outputs = output.copy()
                            row.update(error_metrics(high_precision, dequantize(output, sa, sb)))
                            writer.writerow(row)
                            stream.flush()
                            manifest["completed_rows"] += 1
                    save_json(manifest_path, manifest)
                    print(f"{kind}: {distribution}, input {sample_index + 1}/{args.samples}; both schemes checked", flush=True)
        manifest["status"] = "complete"
        manifest["data_sha256"] = sha256(csv_path)
        save_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_json(manifest_path, manifest)
        raise

    print(f"Data saved: {csv_path}", flush=True)
    if not args.no_plot:
        command = [sys.executable, str(HERE / "plot_quantization_accuracy.py"), "--csv", str(csv_path)]
        if args.software_verification:
            command.append("--allow-software-verification")
        subprocess.run(command, check=True)
    return run_dir


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", action="store_true", help="Use 64x64x64 and three independent pairs per distribution")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rvv-cpu", type=int, default=4)
    p.add_argument("--ime-cpu", type=int, default=0)
    p.add_argument("--output-dir", type=Path, help="A new, previously nonexistent run directory")
    p.add_argument("--build-dir", type=Path, help="Reuse a completed, hash-verified native build")
    p.add_argument("--no-plot", action="store_true", help="Collect data without requiring matplotlib")
    p.add_argument("--software-verification", action="store_true", help="Test pipeline math locally; never measures RVV/IME")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run_experiment(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
