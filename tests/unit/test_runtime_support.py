"""Audit proofs for the documented runtime support floors."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "source_path",
    [
        Path("src/meshprobe/artifacts.py"),
        Path("src/meshprobe/models.py"),
        Path("src/meshprobe/protocol.py"),
        Path("src/meshprobe/evals/harness/adapters.py"),
    ],
)
def test_pep_695_aliases_do_not_parse_as_python_3_11(source_path: Path) -> None:
    source = (PROJECT_ROOT / source_path).read_text(encoding="utf-8")

    with pytest.raises(SyntaxError):
        ast.parse(source, filename=str(source_path), feature_version=(3, 11))
