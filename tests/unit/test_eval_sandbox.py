from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from meshprobe.evals.harness.sandbox import (
    IsolationLimits,
    SandboxUnavailable,
    run_isolated,
)

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is required")


def roots(tmp_path: Path) -> tuple[Path, Path]:
    public = tmp_path / "public"
    artifacts = tmp_path / "artifacts"
    public.mkdir()
    (public / "model.glb").write_bytes(b"model")
    return public, artifacts


def test_sandbox_exposes_only_read_only_input_and_writable_artifacts(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    result = run_isolated(
        (
            "/bin/sh",
            "-c",
            "test -r /workspace/input/model.glb && "
            "! test -e /home/pedro/src/meshprobe/pyproject.toml && "
            "! printf changed > /workspace/input/model.glb && "
            "printf evidence > evidence.txt",
        ),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode == 0, result.stderr
    assert (artifacts / "evidence.txt").read_text(encoding="utf-8") == "evidence"
    assert (public / "model.glb").read_bytes() == b"model"


def test_sandbox_has_no_network_namespace(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    program = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=0.2)"
    result = run_isolated(
        ("/usr/bin/python3", "-c", program),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode != 0
    assert "Network is unreachable" in result.stderr


def test_sandbox_timeout_terminates_agent(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    result = run_isolated(
        ("/bin/sh", "-c", "while :; do :; done"),
        input_root=public,
        artifact_root=artifacts,
        limits=IsolationLimits(wall_seconds=0.05, cpu_seconds=5),
    )

    assert result.timed_out
    assert result.returncode != 0


def test_sandbox_rejects_missing_bubblewrap_and_overlapping_roots(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    with pytest.raises(SandboxUnavailable, match="bubblewrap is required"):
        run_isolated(
            ("/bin/true",),
            input_root=public,
            artifact_root=artifacts,
            bubblewrap="does-not-exist",
        )
    with pytest.raises(ValueError, match="disjoint"):
        run_isolated(
            ("/bin/true",),
            input_root=public,
            artifact_root=public / "artifacts",
        )
