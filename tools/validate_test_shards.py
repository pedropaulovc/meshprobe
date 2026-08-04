"""Prove pytest-split groups are a disjoint, complete test partition."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _collect(extra_args: list[str], split_args: list[str]) -> set[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--disable-warnings",
        *split_args,
        *extra_args,
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }


def _profile_keys(path: Path) -> set[str]:
    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or not profile:
        raise ValueError(f"duration profile must be a non-empty object: {path}")
    return set(profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("durations", type=Path)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    collected = _collect(pytest_args, [])
    profile = _profile_keys(args.durations)
    groups: list[set[str]] = []
    assigned: set[str] = set()
    for group in range(1, args.splits + 1):
        split_args = [
            "--splits",
            str(args.splits),
            "--group",
            str(group),
            "--splitting-algorithm",
            "least_duration",
            "--durations-path",
            str(args.durations),
        ]
        selected = _collect(pytest_args, split_args)
        overlap = assigned & selected
        if overlap:
            sample = ", ".join(sorted(overlap)[:3])
            raise SystemExit(f"group {group} duplicates {len(overlap)} tests: {sample}")
        groups.append(selected)
        assigned.update(selected)

    missing = collected - assigned
    unexpected = assigned - collected
    if missing or unexpected:
        raise SystemExit(f"invalid partition: {len(missing)} missing, {len(unexpected)} unexpected")

    counts = ", ".join(f"group {index}={len(group)}" for index, group in enumerate(groups, 1))
    print(
        f"validated {len(collected)} tests ({counts}); "
        f"{len(collected - profile)} new, {len(profile - collected)} stale profile entries"
    )


if __name__ == "__main__":
    main()
