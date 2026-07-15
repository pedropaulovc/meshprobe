"""Platform process isolation for untrusted evaluation agents."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Protocol


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


class ArtifactBudgetProcess:
    """Popen-compatible wrapper enforcing aggregate bytes in the artifact tree."""

    def __init__(self, process: InteractiveProcess, artifact_root: Path, limit: int) -> None:
        self._process = process
        self._artifact_root = artifact_root
        self._limit = limit
        self._stop = threading.Event()
        self._exceeded = threading.Event()
        self.returncode: int | None = None
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.stderr = process.stderr
        self._monitor = threading.Thread(target=self._watch, daemon=True)
        self._monitor.start()

    @property
    def output_limit_exceeded(self) -> bool:
        return self._exceeded.is_set()

    def poll(self) -> int | None:
        returncode = self._process.poll()
        if returncode is not None:
            self._finish_monitor()
            self._sync_returncode()
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._process.wait(timeout=timeout)
        self._finish_monitor()
        self._sync_returncode()
        return self.returncode or 0

    def kill(self) -> None:
        self._process.kill()

    def communicate(
        self, input: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]:
        stdout, stderr = self._process.communicate(input, timeout=timeout)
        self._finish_monitor()
        self._sync_returncode()
        if self.output_limit_exceeded:
            stderr += "\nmeshprobe sandbox: aggregate artifact output limit exceeded\n"
        return stdout, stderr

    def _watch(self) -> None:
        while not self._stop.wait(0.05):
            if _artifact_tree_bytes(self._artifact_root) > self._limit:
                self._exceeded.set()
                if self._process.poll() is None:
                    self._process.kill()
                return
            if self._process.poll() is not None:
                return

    def _finish_monitor(self) -> None:
        if _artifact_tree_bytes(self._artifact_root) > self._limit:
            self._exceeded.set()
        self._stop.set()
        if threading.current_thread() is not self._monitor:
            self._monitor.join(timeout=1)

    def _sync_returncode(self) -> None:
        observed = self._process.returncode
        self.returncode = 1 if observed == 0 and self.output_limit_exceeded else observed


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
            process=ArtifactBudgetProcess(windows_process, artifacts, active_limits.output_bytes),
            started_monotonic=time.monotonic(),
            wall_seconds=active_limits.wall_seconds,
        )
    if os.name != "posix":  # pragma: no cover
        raise SandboxUnavailable(f"unsupported sandbox host: {os.name}")
    executable = _bubblewrap_path(bubblewrap)
    sandbox_command = _sandbox_command(executable, command, public, artifacts, active_environment)
    limited_command = _limit_command(
        sandbox_command,
        active_limits,
        existing_user_tasks=_user_task_count(),
    )
    started = time.monotonic()
    process = subprocess.Popen(
        limited_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        bufsize=1,
    )
    return IsolatedProcess(
        command=command,
        process=ArtifactBudgetProcess(process, artifacts, active_limits.output_bytes),
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
    mounts, translated_command = _sandbox_agent_command(command)
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
    for host, mount in mounts:
        args.extend(("--ro-bind", str(host), str(mount)))
    if any(mount == PurePosixPath("/opt/meshprobe-agent") for _host, mount in mounts):
        executable_parent = str(PurePosixPath(translated_command[0]).parent)
        args.extend(("--setenv", "PATH", f"{executable_parent}:/usr/bin:/bin"))
    for name, value in sorted(environment.items()):
        if not name or "=" in name or "\x00" in name or "\x00" in value:
            raise ValueError(f"invalid sandbox environment entry: {name!r}")
        args.extend(("--setenv", name, value))
    args.extend(("--", *translated_command))
    return tuple(args)


def _sandbox_agent_command(
    command: tuple[str, ...],
) -> tuple[tuple[tuple[Path, PurePosixPath], ...], tuple[str, ...]]:
    resolved_name = shutil.which(command[0])
    if resolved_name is None:
        raise FileNotFoundError(f"agent executable not found: {command[0]}")
    located_executable = Path(resolved_name).absolute()
    resolved_executable = located_executable.resolve(strict=True)
    mounts: list[tuple[Path, PurePosixPath]] = []
    runtime_root: Path | None = None
    virtualenv_root = located_executable.parent.parent
    executable = (
        located_executable
        if located_executable.parent.name == "bin" and (virtualenv_root / "pyvenv.cfg").is_file()
        else resolved_executable
    )
    if executable == located_executable and not resolved_executable.is_relative_to("/usr"):
        base_runtime = (
            resolved_executable.parent.parent
            if resolved_executable.parent.name == "bin"
            else resolved_executable.parent
        )
        guest_runtimes = {base_runtime}
        link_target = Path(os.readlink(located_executable))
        if link_target.is_absolute():
            guest_runtimes.add(
                link_target.parent.parent
                if link_target.parent.name == "bin"
                else link_target.parent
            )
        mounts.extend(
            (base_runtime, PurePosixPath(guest_runtime.as_posix()))
            for guest_runtime in sorted(guest_runtimes)
        )
    try:
        executable.relative_to("/usr")
    except ValueError:
        runtime_root = executable.parent
        if runtime_root.name == "bin" and (runtime_root.parent / "pyvenv.cfg").is_file():
            runtime_root = runtime_root.parent
        runtime_mount = PurePosixPath("/opt/meshprobe-agent")
        mounts.append((runtime_root, runtime_mount))
        relative_executable = executable.relative_to(runtime_root)
        translated_executable = runtime_mount / PurePosixPath(relative_executable.as_posix())
    else:
        translated_executable = PurePosixPath(str(executable))
    translated_arguments = list(command[1:])
    for index, argument in enumerate(translated_arguments, start=1):
        candidate = Path(argument).expanduser()
        if argument.startswith("/workspace/") or not candidate.is_file():
            continue
        resolved_argument = candidate.resolve(strict=True)
        if resolved_argument.is_relative_to("/usr"):
            translated_arguments[index - 1] = str(resolved_argument)
            continue
        if runtime_root is not None and resolved_argument.is_relative_to(runtime_root):
            relative_argument = resolved_argument.relative_to(runtime_root)
            translated_arguments[index - 1] = str(
                PurePosixPath("/opt/meshprobe-agent") / PurePosixPath(relative_argument.as_posix())
            )
            continue
        argument_mount = PurePosixPath(
            f"/opt/meshprobe-agent-arguments/{index:04d}-{resolved_argument.name}"
        )
        mounts.append((resolved_argument, argument_mount))
        translated_arguments[index - 1] = str(argument_mount)
    try:
        with executable.open("rb") as stream:
            first_line = stream.read(4_096).split(b"\n", 1)[0].decode("utf-8", errors="replace")
    except OSError:
        first_line = ""
    if first_line.startswith("#!") and runtime_root is not None:
        shebang = shlex.split(first_line[2:].strip())
        if shebang:
            interpreter = Path(shebang[0])
            try:
                relative_interpreter = interpreter.relative_to(runtime_root)
            except ValueError:
                pass
            else:
                translated_interpreter = PurePosixPath("/opt/meshprobe-agent") / PurePosixPath(
                    relative_interpreter.as_posix()
                )
                return tuple(mounts), (
                    str(translated_interpreter),
                    *shebang[1:],
                    str(translated_executable),
                    *translated_arguments,
                )
    return tuple(mounts), (str(translated_executable), *translated_arguments)


def _limit_command(
    command: tuple[str, ...],
    limits: IsolationLimits,
    *,
    existing_user_tasks: int,
) -> tuple[str, ...]:
    executable = shutil.which("prlimit")
    if executable is None:
        raise SandboxUnavailable("util-linux prlimit is required for POSIX sandbox limits")
    process_ceiling = existing_user_tasks + limits.processes
    return (
        str(Path(executable).resolve(strict=True)),
        f"--cpu={limits.cpu_seconds}:{limits.cpu_seconds}",
        f"--as={limits.memory_bytes}:{limits.memory_bytes}",
        f"--fsize={limits.output_bytes}:{limits.output_bytes}",
        f"--nproc={process_ceiling}:{process_ceiling}",
        "--",
        *command,
    )


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


def _artifact_tree_bytes(root: Path) -> int:
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except FileNotFoundError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    return total + 2**63
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
    return total
