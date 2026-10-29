#!/usr/bin/env python3
"""Build the existing K1 RVV/IME adapters as a native accuracy shared library.

--check verifies required source files only; it does not create files, compile,
load a library, inspect hardware, or execute kernels. Actual builds require
Linux on riscv64 and an empty output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RVV_KERNEL = "igemm_kernel_8x4_zvl256b_lmulmf8_unroll2"
IME_KERNEL = "ime_kernel_8x4_zvl256b_lmul1_unroll1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_sources(root: Path) -> dict[str, Path]:
    rvv = (root / "GEMM_RVV_FP32_INT8_8x4_Baseline" /
           "RVV_IGEMM_INT8_I8I32_8x4" / RVV_KERNEL /
           f"{RVV_KERNEL}_i8i32.c")
    ime = (root / "IME_NATIVE_KERNELS" /
           "IME_GEMM_INT8_I8I32_8x4_NATIVE" / IME_KERNEL /
           f"{IME_KERNEL}.c")
    return {
        "bridge": HERE / "kernel_bridge.c",
        "build_script": Path(__file__).resolve(),
        "bench_header": root / "benchmarking/src/bench.h",
        "rvv_adapter": root / "benchmarking/src/rvv_adapter.c",
        "ime_adapter": root / "benchmarking/src/ime_adapter.c",
        "dispatch": root / "HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h",
        "rvv_kernel": rvv,
        "ime_kernel": ime,
        "ime_fallback": ime.parent / "rvv_fallback.c",
    }


def verify_sources(sources: dict[str, Path]) -> None:
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError("Required source files are missing:\n" + "\n".join(missing))


def relative_key(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build(args: argparse.Namespace) -> int:
    root = args.project_root.expanduser().resolve()
    sources = selected_sources(root)
    verify_sources(sources)
    if args.check:
        print(json.dumps({
            "schema_version": 1,
            "status": "source_check_passed",
            "execution_kind": "source_check_only",
            "project_root": str(root),
            "rvv_kernel": RVV_KERNEL,
            "ime_kernel": IME_KERNEL,
            "sources_checked": [str(path) for path in sources.values()],
        }, indent=2))
        return 0

    if platform.system() != "Linux" or platform.machine().lower() != "riscv64":
        raise ValueError("Native builds require Linux on riscv64. Use --check for source validation on this host.")
    compiler = shlex.split(args.cc)
    if not compiler:
        raise ValueError("The compiler command is empty")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Build output must be empty; choose a fresh directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = output / "selected_sources.h"
    library = output / "libquant_accuracy.so"
    compile_flags = [
        "-O3", "-std=c11", "-Wall", "-Wextra", "-Wno-unused-function",
        "-Wno-unknown-pragmas", "-march=rv64gcv_zvl256b", "-mabi=lp64d",
        "-fopenmp", "-fPIC", "-DSPACEMIT_IME_REQUIRE_HARDWARE=1",
        "-I", str(root), "-I", str(root / "benchmarking/src"),
        "-include", str(config),
    ]
    link_flags = ["-shared", "-fopenmp", "-Wl,-z,defs", "-lm"]
    metadata = {
        "schema_version": 1,
        "status": "building",
        "execution_kind": "native_k1",
        "abi_version": 1,
        "project_root": str(root),
        "library": str(library),
        "rvv_kernel": RVV_KERNEL,
        "ime_kernel": IME_KERNEL,
        "compiler_command": compiler,
        "compile_flags": compile_flags,
        "link_flags": link_flags,
        "source_hashes": {},
        "commands": [],
    }
    manifest = output / "build.json"
    try:
        metadata["source_hashes"] = {relative_key(path, root): sha256(path)
                                     for path in sources.values()}
        config.write_text(
            "#define BENCH_RVV_NR 4\n"
            "#define BENCH_DISPATCH_SOURCE " + json.dumps(sources["dispatch"].as_posix()) + "\n"
            "#define BENCH_IME_SOURCE " + json.dumps(sources["ime_kernel"].as_posix()) + "\n",
            encoding="utf-8",
        )
        metadata["generated_config"] = {
            "path": str(config), "sha256": sha256(config),
        }
        version = subprocess.run(compiler + ["--version"], check=True,
                                 capture_output=True, text=True)
        metadata["compiler_version"] = version.stdout
        units = [
            (sources["bridge"], []),
            (sources["rvv_adapter"], []),
            (sources["ime_adapter"], []),
            # Some preserved variants use CNAME, others spell their entry name.
            # Both aliases identify the exact original RVV translation unit.
            (sources["rvv_kernel"], ["-DCNAME=bench_rvv_entry",
                                    f"-D{RVV_KERNEL}_i8i32=bench_rvv_entry"]),
            (sources["ime_fallback"], []),
        ]
        objects = []
        with (output / "build.log").open("w", encoding="utf-8") as log:
            for index, (source, extra_flags) in enumerate(units):
                obj = output / f"unit_{index}.o"
                command = compiler + compile_flags + extra_flags + [
                    "-c", str(source), "-o", str(obj)]
                metadata["commands"].append(command)
                log.write(shlex.join(command) + "\n")
                log.flush()
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                objects.append(str(obj))
            command = compiler + objects + link_flags + ["-o", str(library)]
            metadata["commands"].append(command)
            log.write(shlex.join(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        metadata["library_sha256"] = sha256(library)
        metadata["status"] = "complete"
    except (OSError, subprocess.CalledProcessError) as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(library)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parent)
    parser.add_argument("--output-dir", type=Path, default=HERE / "build")
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        return build(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
