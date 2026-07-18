from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path, PurePosixPath

import pytest

from meshprobe.evals.harness.sandbox import (
    IsolationLimits,
    SandboxUnavailable,
    _sandbox_agent_command,
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


def test_sandbox_keeps_python_available_for_batch_operations(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    (public / "first.json").write_text('{"value": 2}\n', encoding="utf-8")
    (public / "second.json").write_text('{"value": 3}\n', encoding="utf-8")
    input_directory = str(public.resolve()) if os.name == "nt" else "/workspace/input"
    output = (
        str((artifacts / "batch.json").resolve())
        if os.name == "nt"
        else "/workspace/artifacts/batch.json"
    )
    program = (
        "import json\n"
        "from pathlib import Path\n"
        f"root=Path({input_directory!r})\n"
        "values=[json.loads(path.read_text())['value'] "
        "for path in sorted(root.glob('*.json'))]\n"
        f"Path({output!r}).write_text(json.dumps({{'sum': sum(values)}}))\n"
    )

    result = run_isolated(
        (sandbox_python(), "-c", program),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((artifacts / "batch.json").read_text(encoding="utf-8")) == {"sum": 5}


@pytest.mark.skipif(os.name == "nt", reason="POSIX runtime identity files are Linux-specific")
def test_sandbox_exposes_read_only_runtime_identity(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    program = (
        "from pathlib import Path\n"
        "passwd=Path('/etc/passwd')\n"
        "assert passwd.is_file() and passwd.read_bytes()\n"
        "try:\n"
        "    passwd.write_text('changed')\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('runtime identity was writable')\n"
    )

    result = run_isolated(
        (sandbox_python(), "-c", program),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode == 0, result.stderr


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


@pytest.mark.skipif(os.name != "nt", reason="AppContainer ACL serialization is Windows-specific")
def test_windows_parallel_sandboxes_share_python_runtime(tmp_path: Path) -> None:
    workers = 4
    barrier = threading.Barrier(workers)

    def launch(index: int) -> tuple[int, str, str]:
        episode = tmp_path / f"episode-{index}"
        episode.mkdir()
        public, artifacts = roots(episode)
        barrier.wait()
        result = run_isolated(
            (sandbox_python(), "-c", f"print('agent-{index}')"),
            input_root=public,
            artifact_root=artifacts,
        )
        return result.returncode, result.stdout, result.stderr

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(launch, range(workers)))

    assert results == tuple((0, f"agent-{index}\n", "") for index in range(workers))


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
def test_sandbox_enables_pipefail_for_agent_shells(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)

    result = run_isolated(
        ("/usr/bin/bash", "-c", "false | true"),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode != 0


@pytest.mark.skipif(os.name == "nt", reason="Bubblewrap argument binding is POSIX-specific")
def test_sandbox_binds_direct_script_arguments(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    script = tmp_path / "external-agent.py"
    script.write_text("print('bound-script')\n", encoding="utf-8")

    result = run_isolated(
        ("/usr/bin/python3", str(script)),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bound-script"


@pytest.mark.skipif(os.name == "nt", reason="Bubblewrap runtime binding is POSIX-specific")
def test_sandbox_resolves_symlinked_agent_executable(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    runtime = tmp_path / "agent-runtime"
    runtime.mkdir()
    target = runtime / "real-agent"
    target.write_text("#!/usr/bin/python3\nprint('resolved-agent')\n", encoding="utf-8")
    target.chmod(0o755)
    link = tmp_path / "agent-link"
    link.symlink_to(target)

    result = run_isolated((str(link),), input_root=public, artifact_root=artifacts)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "resolved-agent"


@pytest.mark.skipif(os.name == "nt", reason="Bubblewrap runtime binding is POSIX-specific")
def test_sandbox_preserves_node_package_runtime_for_bin_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "node-v25"
    executable_root = runtime / "bin"
    package_bin = runtime / "lib" / "node_modules" / "agent" / "bin"
    executable_root.mkdir(parents=True)
    package_bin.mkdir(parents=True)
    (executable_root / "node").write_bytes(b"node")
    agent = package_bin / "agent.js"
    agent.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    entrypoint = executable_root / "agent"
    entrypoint.symlink_to(Path("../lib/node_modules/agent/bin/agent.js"))
    monkeypatch.setattr(shutil, "which", lambda name: str(entrypoint) if name == "agent" else None)

    mounts, translated = _sandbox_agent_command(("agent", "--version"))

    assert mounts == ((runtime, PurePosixPath("/opt/meshprobe-agent")),)
    assert translated == ("/opt/meshprobe-agent/bin/agent", "--version")


@pytest.mark.skipif(os.name == "nt", reason="Bubblewrap runtime binding is POSIX-specific")
def test_sandbox_treats_long_prompt_as_an_argument_not_a_path() -> None:
    prompt = "inspect the assigned model " * 200

    mounts, translated = _sandbox_agent_command(("echo", prompt))

    assert mounts == ()
    assert translated == ("/usr/bin/echo", prompt)


@pytest.mark.skipif(os.name == "nt", reason="Bubblewrap runtime binding is POSIX-specific")
def test_sandbox_mounts_external_virtualenv_base_runtime(tmp_path: Path) -> None:
    public, artifacts = roots(tmp_path)
    base_runtime = tmp_path / "hosted-python-3.13.12"
    base_bin = base_runtime / "bin"
    base_bin.mkdir(parents=True)
    base_python = base_bin / "python3"
    shutil.copy2("/usr/bin/python3", base_python)
    runtime_alias = tmp_path / "hosted-python-3.13"
    runtime_alias.symlink_to(base_runtime, target_is_directory=True)
    runtime = tmp_path / "agent-venv"
    executable_root = runtime / "bin"
    executable_root.mkdir(parents=True)
    (runtime / "pyvenv.cfg").write_text(
        f"home = {base_bin}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    python = executable_root / "python"
    python.symlink_to(runtime_alias / "bin" / "python3")

    result = run_isolated(
        (str(python), "-c", "print('external-base-runtime')"),
        input_root=public,
        artifact_root=artifacts,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "external-base-runtime"


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


def test_windows_poll_releases_process_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    from meshprobe.evals.harness import windows_sandbox

    closed: list[int] = []

    class FakeKernel32:
        @staticmethod
        def GetExitCodeProcess(handle: int, output: ctypes._CArgObject) -> int:
            del handle
            from ctypes import wintypes

            ctypes.cast(output, ctypes.POINTER(wintypes.DWORD)).contents.value = 0
            return 1

        @staticmethod
        def CloseHandle(handle: int) -> int:
            closed.append(handle)
            return 1

    cleaned: list[bool] = []
    monkeypatch.setattr(windows_sandbox, "_kernel32", FakeKernel32())
    process = windows_sandbox.WindowsProcess(
        command=("agent",),
        process_handle=11,
        job_handle=12,
        process_id=13,
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
        cleanup=lambda: cleaned.append(True),
    )

    assert process.poll() == 0
    assert process._finished
    assert closed == [11, 12]
    assert cleaned == [True]
