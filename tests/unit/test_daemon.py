from __future__ import annotations

import json
from pathlib import Path

from meshprobe.daemon import _remove_matching_metadata
from meshprobe.workspace import atomic_json


def test_daemon_exit_preserves_replacement_metadata(tmp_path: Path) -> None:
    original = {"pid": 10, "token": "old"}
    replacement = {"pid": 11, "token": "new"}
    path = tmp_path / "daemon.json"
    atomic_json(path, replacement)

    _remove_matching_metadata(tmp_path, original)

    assert json.loads(path.read_text(encoding="utf-8")) == replacement


def test_daemon_exit_removes_its_own_metadata(tmp_path: Path) -> None:
    metadata = {"pid": 10, "token": "owned"}
    path = tmp_path / "daemon.json"
    atomic_json(path, metadata)

    _remove_matching_metadata(tmp_path, metadata)

    assert not path.exists()
