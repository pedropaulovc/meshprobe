from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from meshprobe.evals.harness.sandbox import (
    IsolationLimits,
    SandboxUnavailable,
    _user_task_count,
    run_isolated,
    visible_input_path,
)

pytestmark = pytest.mark.skipif(
    os.name != "nt" and shutil.which("bwrap") is None,
    reason="a platform sandbox is required",
)


def sandbox_python() -> str:
    return sys.executable if os.name == "nt" else "/usr/bin/python3"


def roots(tmp_path: Path) -> tuple[Path, Path]:
    public = tmp_path / "public"
    artifacts = tmp_path / "artifacts"
    public.mkdir()
    (public / "model.glb").write_bytes(b"model")
    return public, artifacts


def test_sandbox_exposes_only_read_only_input_and_writable_artifacts(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    private = tmp_path / "evaluator-private.txt"
    private.write_text("secret", encoding="utf-8")
    model = visible_input_path(public / "model.glb")
    evidence = (
        str((artifacts / "evidence.txt").resolve())
        if os.name == "nt"
        else "/workspace/artifacts/evidence.txt"
    )
    program = (
        "from pathlib import Path\n"
        f"model=Path({model!r})\n"
        f"private=Path({str(private.resolve())!r})\n"
        f"evidence=Path({evidence!r})\n"
        "assert model.read_bytes() == b'model'\n"
        "try:\n"
        "    private_visible=private.exists()\n"
        "except OSError:\n"
        "    private_visible=False\n"
        "assert not private_visible\n"
        "try:\n"
        "    model.write_bytes(b'changed')\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('assigned model was writable')\n"
        "evidence.write_text('evidence', encoding='utf-8')\n"
    )
    result = run_isolated(
        (sandbox_python(), "-c", program),
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
        (sandbox_python(), "-c", program),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode != 0
    assert any(
        marker in result.stderr
        for marker in ("Network is unreachable", "PermissionError", "WinError 10013")
    ), result.stderr


def test_sandbox_timeout_terminates_agent(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    result = run_isolated(
        (sandbox_python(), "-c", "while True: pass"),
        input_root=public,
        artifact_root=artifacts,
        limits=IsolationLimits(wall_seconds=0.05, cpu_seconds=5),
    )

    assert result.timed_out
    assert result.returncode != 0


def test_sandbox_enforces_aggregate_artifact_bytes(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    program = (
        "from pathlib import Path\n"
        "root=Path.cwd()\n"
        "[(root / f'artifact-{index}.bin').write_bytes(b'x' * 60) for index in range(3)]\n"
    )

    result = run_isolated(
        (sandbox_python(), "-c", program),
        input_root=public,
        artifact_root=artifacts,
        limits=IsolationLimits(output_bytes=100),
    )

    assert result.returncode != 0
    assert "aggregate artifact output limit exceeded" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="Bubblewrap runtime binding is POSIX-specific")
def test_sandbox_binds_agent_virtualenv_or_checkout_directory(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    runtime = tmp_path / "agent-runtime"
    runtime.mkdir()
    agent = runtime / "agent"
    agent.write_text("#!/usr/bin/python3\nprint('bound-agent')\n", encoding="utf-8")
    agent.chmod(0o755)

    result = run_isolated((str(agent),), input_root=public, artifact_root=artifacts)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bound-agent"


@pytest.mark.skipif(os.name == "nt", reason="Bubblewrap runtime binding is POSIX-specific")
def test_sandbox_translates_absolute_virtualenv_shebang(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    runtime = tmp_path / "agent-venv"
    executable_root = runtime / "bin"
    executable_root.mkdir(parents=True)
    (runtime / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    python = executable_root / "python"
    python.symlink_to("/usr/bin/python3")
    agent = executable_root / "agent"
    agent.write_text(f"#!{python}\nprint('venv-agent')\n", encoding="utf-8")
    agent.chmod(0o755)

    result = run_isolated((str(agent),), input_root=public, artifact_root=artifacts)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "venv-agent"


def test_sandbox_rejects_missing_bubblewrap_and_overlapping_roots(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    if os.name == "nt":
        with pytest.raises(ValueError, match="bubblewrap cannot be configured"):
            run_isolated(
                (sandbox_python(), "-c", "pass"),
                input_root=public,
                artifact_root=artifacts,
                bubblewrap="does-not-exist",
            )
    else:
        with pytest.raises(SandboxUnavailable, match="bubblewrap is required"):
            run_isolated(
                ("/bin/true",),
                input_root=public,
                artifact_root=artifacts,
                bubblewrap="does-not-exist",
            )
    with pytest.raises(ValueError, match="disjoint"):
        run_isolated(
            (sandbox_python(), "-c", "pass"),
            input_root=public,
            artifact_root=public / "artifacts",
        )


@pytest.mark.skipif(os.name == "nt", reason="Windows process limits use a Job Object")
def test_posix_process_limit_baseline_counts_threads_not_only_processes() -> None:
    own_processes = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            own_processes += entry.stat().st_uid == os.getuid()
        except FileNotFoundError:
            continue
    assert _user_task_count() >= own_processes
