#!/usr/bin/env python3
"""Snapshot/check files outside this additive benchmarking directory (read only)."""
import argparse
import hashlib
import json
from pathlib import Path


def inventory(root):
    entries = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or relative.parts[0] in ("benchmarking", ".git"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries[relative.as_posix()] = digest.hexdigest()
    return entries


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--check", action="store_true")
    args = cli.parse_args()
    current = inventory(args.project_root.resolve())
    if args.check:
        original = json.loads(args.snapshot.read_text(encoding="utf-8"))["sha256"]
        changes = [name for name in sorted(set(current) | set(original)) if current.get(name) != original.get(name)]
        print(json.dumps({"files_checked": len(current), "changed_added_or_missing": changes}, indent=2))
        return bool(changes)
    with args.snapshot.open("x", encoding="utf-8") as stream:
        json.dump({"project_root": str(args.project_root.resolve()), "sha256": current}, stream, indent=2)
    print(f"Snapshotted {len(current)} original files; no project files edited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
