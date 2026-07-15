"""Platform process isolation for untrusted evaluation agents."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Protocol

if os.name == "posix":  # pragma: no cover
    import resource


class SandboxUnavailable(RuntimeError):
    """The host cannot provide the required network and filesystem isolation."""


class InteractiveProcess(Protocol):
    """Process surface shared by subprocess.Popen and the Windows launcher."""

    returncode: int | None
    stdin: IO[str] | None
    stdout: IO[str] | None
    stderr: IO[str] | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...

    def communicate(
        self, input: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]: ...


@dataclass(frozen=True)
class IsolationLimits:
    wall_seconds: float = 600
    cpu_seconds: int = 600
    memory_bytes: int = 8 * 1024**3
    output_bytes: int = 1 * 1024**3
    processes: int = 128

    def __post_init__(self) -> None:
        values = (
            self.wall_seconds,
            self.cpu_seconds,
            self.memory_bytes,
            self.output_bytes,
            self.processes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("isolation limits must all be positive")


@dataclass(frozen=True)
class IsolatedProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool


@dataclass(frozen=True)
class IsolatedProcess:
    command: tuple[str, ...]
    process: InteractiveProcess
    started_monotonic: float
    wall_seconds: float

    def terminate(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.kill()
        self.process.wait(timeout=2)


def visible_input_path(path: Path) -> str:
    """Return the assigned input path as seen inside the platform sandbox."""

    resolved = path.expanduser().resolve(strict=True)
    if os.name == "nt":  # pragma: no cover
        return str(resolved)
    return f"/workspace/input/{resolved.name}"


def visible_artifact_path(path: Path) -> str:
    """Return one artifact path as seen inside the platform sandbox."""

    resolved = path.expanduser().resolve()
    if os.name == "nt":  # pragma: no cover
        return str(resolved)
    return f"/workspace/artifacts/{resolved.name}"


def run_isolated(
    command: tuple[str, ...],
    *,
    input_root: Path,
    artifact_root: Path,
    stdin: str = "",
    environment: Mapping[str, str] | None = None,
    limits: IsolationLimits | None = None,
    bubblewrap: str | Path | None = None,
) -> IsolatedProcessResult:
    """Run one agent with read-only inputs, writable artifacts, and no network."""

    isolated = spawn_isolated(
        command,
        input_root=input_root,
        artifact_root=artifact_root,
        environment=environment,
        limits=limits,
        bubblewrap=bubblewrap,
    )
    process = isolated.process
    timed_out = False
    try:
        stdout, stderr = process.communicate(stdin, timeout=isolated.wall_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        isolated.terminate()
        stdout, stderr = process.communicate()
    if process.returncode is None:
        raise RuntimeError("isolated process completed without a return code")
    return IsolatedProcessResult(
        command=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=time.monotonic() - isolated.started_monotonic,
        timed_out=timed_out,
    )


def spawn_isolated(
    command: tuple[str, ...],
    *,
    input_root: Path,
    artifact_root: Path,
    environment: Mapping[str, str] | None = None,
    limits: IsolationLimits | None = None,
    bubblewrap: str | Path | None = None,
) -> IsolatedProcess:
    """Start an interactive isolated process with piped standard streams."""

    if not command or not command[0]:
        raise ValueError("isolated command cannot be empty")
    public = input_root.expanduser().resolve(strict=True)
    artifacts = artifact_root.expanduser().resolve()
    if artifacts == public or artifacts.is_relative_to(public) or public.is_relative_to(artifacts):
        raise ValueError("input and artifact roots must be disjoint")
    artifacts.mkdir(parents=True, exist_ok=True)
    active_environment = environment if environment is not None else {}
    active_limits = limits or IsolationLimits()
    if os.name == "nt":
        if bubblewrap is not None:
            raise ValueError("bubblewrap cannot be configured on Windows")
        from meshprobe.evals.harness.windows_sandbox import spawn_windows

        windows_process = spawn_windows(
            command,
            input_root=public,
            artifact_root=artifacts,
            environment=active_environment,
            limits=active_limits,
        )
        return IsolatedProcess(
            command=command,
            process=windows_process,
            started_monotonic=time.monotonic(),
            wall_seconds=active_limits.wall_seconds,
        )
    if os.name != "posix":  # pragma: no cover
        raise SandboxUnavailable(f"unsupported sandbox host: {os.name}")
    executable = _bubblewrap_path(bubblewrap)
    existing_user_tasks = _user_task_count()
    sandbox_command = _sandbox_command(executable, command, public, artifacts, active_environment)
    started = time.monotonic()
    process = subprocess.Popen(
        sandbox_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        preexec_fn=lambda: _set_limits(active_limits, existing_user_tasks),
        bufsize=1,
    )
    return IsolatedProcess(
        command=command,
        process=process,
        started_monotonic=started,
        wall_seconds=active_limits.wall_seconds,
    )


def _bubblewrap_path(configured: str | Path | None) -> Path:
    candidate = shutil.which(str(configured or "bwrap"))
    if candidate is None:
        raise SandboxUnavailable(
            "bubblewrap is required; install bwrap rather than running qualification unsandboxed"
        )
    return Path(candidate).resolve(strict=True)


def _sandbox_command(
    bubblewrap: Path,
    command: tuple[str, ...],
    input_root: Path,
    artifact_root: Path,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    runtime_root, translated_command = _sandbox_agent_command(command)
    args = [
        str(bubblewrap),
        "--unshare-all",
        "--new-session",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/home",
        "--ro-bind",
        str(input_root),
        "/workspace/input",
        "--bind",
        str(artifact_root),
        "/workspace/artifacts",
        "--chdir",
        "/workspace/artifacts",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp/home",
        "--setenv",
        "LANG",
        "C.UTF-8",
    ]
    if runtime_root is not None:
        mount = PurePosixPath("/opt/meshprobe-agent")
        executable_parent = str(PurePosixPath(translated_command[0]).parent)
        args.extend(("--ro-bind", str(runtime_root), str(mount)))
        args.extend(("--setenv", "PATH", f"{executable_parent}:/usr/bin:/bin"))
    for name, value in sorted(environment.items()):
        if not name or "=" in name or "\x00" in name or "\x00" in value:
            raise ValueError(f"invalid sandbox environment entry: {name!r}")
        args.extend(("--setenv", name, value))
    args.extend(("--", *translated_command))
    return tuple(args)


def _sandbox_agent_command(command: tuple[str, ...]) -> tuple[Path | None, tuple[str, ...]]:
    resolved_name = shutil.which(command[0])
    if resolved_name is None:
        raise FileNotFoundError(f"agent executable not found: {command[0]}")
    executable = Path(resolved_name).absolute()
    try:
        executable.relative_to("/usr")
    except ValueError:
        pass
    else:
        return None, (str(executable), *command[1:])
    runtime_root = executable.parent
    if runtime_root.name == "bin" and (runtime_root.parent / "pyvenv.cfg").is_file():
        runtime_root = runtime_root.parent
    relative_executable = executable.relative_to(runtime_root)
    translated = PurePosixPath("/opt/meshprobe-agent") / PurePosixPath(
        relative_executable.as_posix()
    )
    return runtime_root, (str(translated), *command[1:])


def _user_task_count() -> int:
    """Count UID-owned kernel tasks because RLIMIT_NPROC includes threads."""

    user_id = os.getuid()
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            owned_by_user = entry.stat().st_uid == user_id
        except FileNotFoundError:
            continue
        if not owned_by_user:
            continue
        try:
            count += sum(child.name.isdigit() for child in (entry / "task").iterdir())
        except FileNotFoundError:
            continue
    return count


def _set_limits(limits: IsolationLimits, existing_user_tasks: int) -> None:
    if os.name != "posix":  # pragma: no cover
        raise RuntimeError("POSIX resource limits requested on a non-POSIX host")
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.output_bytes, limits.output_bytes))
    process_ceiling = existing_user_tasks + limits.processes
    resource.setrlimit(resource.RLIMIT_NPROC, (process_ceiling, process_ceiling))
    os.umask(0o077)
