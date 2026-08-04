"""Convert pytest JUnit timing output to a pytest-split duration profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree


def _node_id(classname: str, test_name: str) -> str:
    parts = classname.split(".")
    module_index = next(
        (index for index in range(len(parts) - 1, -1, -1) if parts[index].startswith("test_")),
        None,
    )
    if module_index is None:
        raise ValueError(f"cannot find a test module in JUnit classname {classname!r}")

    path = "/".join(parts[: module_index + 1]) + ".py"
    qualifiers = [*parts[module_index + 1 :], test_name]
    return "::".join([path, *qualifiers])


def convert(junit_path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    for testcase in ElementTree.parse(junit_path).iterfind(".//testcase"):
        classname = testcase.get("classname")
        test_name = testcase.get("name")
        raw_duration = testcase.get("time")
        if classname is None or test_name is None or raw_duration is None:
            raise ValueError("every JUnit testcase must include classname, name, and time")

        node_id = _node_id(classname, test_name)
        duration = float(raw_duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"invalid duration for {node_id}: {raw_duration!r}")
        if node_id in durations:
            raise ValueError(f"duplicate JUnit testcase: {node_id}")
        durations[node_id] = duration

    if not durations:
        raise ValueError(f"no testcases found in {junit_path}")
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    durations = convert(args.junit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(durations, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(durations)} durations to {args.output}")


if __name__ == "__main__":
    main()
