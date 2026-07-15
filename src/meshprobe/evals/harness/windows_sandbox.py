"""Native Windows AppContainer and Job Object launcher."""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from ctypes import wintypes
from pathlib import Path
from typing import IO, Any

from meshprobe.evals.harness.sandbox import IsolationLimits, SandboxUnavailable

_HRESULT_ALREADY_EXISTS = 0x800700B7
_STILL_ACTIVE = 259
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_INFINITE = 0xFFFFFFFF

_STARTF_USESTDHANDLES = 0x00000100
_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_NO_WINDOW = 0x08000000

_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009

_JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

_get_last_error = getattr(ctypes, "get_last_error", lambda: 0)
_format_error = getattr(ctypes, "FormatError", lambda code: f"Windows error {code}")
_win_error = getattr(ctypes, "WinError", OSError)
_win_dll = getattr(ctypes, "WinDLL", None)


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", _STARTUPINFOW), ("lpAttributeList", wintypes.LPVOID)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class _SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", wintypes.LPVOID),
        ("Capabilities", ctypes.POINTER(_SID_AND_ATTRIBUTES)),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsProcess:
    """Small Popen-compatible wrapper around an AppContainer process."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        process_handle: int,
        job_handle: int,
        process_id: int,
        stdin: IO[str],
        stdout: IO[str],
        stderr: IO[str],
        cleanup: Callable[[], None],
    ) -> None:
        self.args = command
        self.pid = process_id
        self.returncode: int | None = None
        self.stdin: IO[str] | None = stdin
        self.stdout: IO[str] | None = stdout
        self.stderr: IO[str] | None = stderr
        self._process_handle = process_handle
        self._job_handle = job_handle
        self._cleanup_callback = cleanup
        self._finished = False
        self._lock = threading.Lock()
        self._communication_started = False
        self._stdout_parts: list[str] = []
        self._stderr_parts: list[str] = []
        self._reader_threads: tuple[threading.Thread, threading.Thread] | None = None
        atexit.register(self._atexit_cleanup)

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        exit_code = wintypes.DWORD()
        _check_bool(
            _kernel32.GetExitCodeProcess(self._process_handle, ctypes.byref(exit_code)),
            "GetExitCodeProcess",
        )
        if exit_code.value == _STILL_ACTIVE:
            return None
        self.returncode = _signed_exit_code(exit_code.value)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        milliseconds = _INFINITE if timeout is None else max(0, int(timeout * 1000))
        result = _kernel32.WaitForSingleObject(self._process_handle, milliseconds)
        if result == _WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired(self.args, timeout or 0)
        if result != _WAIT_OBJECT_0:
            raise _win_error(_last_error())
        returncode = self.poll()
        if returncode is None:
            raise RuntimeError("Windows process signaled without an exit code")
        self._finish()
        return returncode

    def kill(self) -> None:
        if self.poll() is not None:
            self._finish()
            return
        _check_bool(_kernel32.TerminateJobObject(self._job_handle, 1), "TerminateJobObject")

    def communicate(
        self, input: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]:
        if not self._communication_started:
            self._communication_started = True
            if self.stdin is not None:
                if input:
                    self.stdin.write(input)
                    self.stdin.flush()
                self.stdin.close()
                self.stdin = None
            self._reader_threads = (
                threading.Thread(
                    target=_read_to_end,
                    args=(self.stdout, self._stdout_parts),
                    daemon=True,
                ),
                threading.Thread(
                    target=_read_to_end,
                    args=(self.stderr, self._stderr_parts),
                    daemon=True,
                ),
            )
            for reader in self._reader_threads:
                reader.start()
        elif input:
            raise ValueError("cannot send input after communication started")
        self.wait(timeout=timeout)
        if self._reader_threads is not None:
            for reader in self._reader_threads:
                reader.join(timeout=2)
        return "".join(self._stdout_parts), "".join(self._stderr_parts)

    def _finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            _kernel32.CloseHandle(self._process_handle)
            _kernel32.CloseHandle(self._job_handle)
            self._cleanup_callback()
            atexit.unregister(self._atexit_cleanup)

    def _atexit_cleanup(self) -> None:
        if self._finished:
            return
        try:
            self.kill()
            _kernel32.WaitForSingleObject(self._process_handle, 2_000)
        finally:
            self._finish()


def spawn_windows(
    command: tuple[str, ...],
    *,
    input_root: Path,
    artifact_root: Path,
    environment: Mapping[str, str],
    limits: IsolationLimits,
) -> WindowsProcess:
    """Launch a native Windows process with AppContainer and Job Object isolation."""

    if os.name != "nt":
        raise SandboxUnavailable("the Windows sandbox can only run on Windows")
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(f"agent executable not found: {command[0]}")
    resolved_command = (str(Path(executable).resolve(strict=True)), *command[1:])
    profile_name = f"MeshProbe.Agent.{uuid.uuid4().hex}"
    appcontainer_sid = _create_profile(profile_name)
    sid_string: str | None = None
    granted: list[Path] = []
    try:
        sid_string = _sid_string(appcontainer_sid)
        profile_root = _profile_folder(sid_string)
        acl_roots = _runtime_roots(Path(executable))
        for root in (input_root, *acl_roots):
            _grant_acl(root, sid_string, "RX")
            granted.append(root)
        _grant_acl(artifact_root, sid_string, "M")
        granted.append(artifact_root)
        process = _create_process(
            resolved_command,
            appcontainer_sid=appcontainer_sid,
            artifact_root=artifact_root,
            profile_root=profile_root,
            environment=environment,
            limits=limits,
            cleanup=lambda: _cleanup_profile(profile_name, sid_string, granted),
        )
    except BaseException:
        if sid_string is None:
            _userenv.DeleteAppContainerProfile(profile_name)
        else:
            _cleanup_profile(profile_name, sid_string, granted)
        raise
    finally:
        _advapi32.FreeSid(appcontainer_sid)
    return process


def _create_process(
    command: tuple[str, ...],
    *,
    appcontainer_sid: int,
    artifact_root: Path,
    profile_root: Path,
    environment: Mapping[str, str],
    limits: IsolationLimits,
    cleanup: Callable[[], None],
) -> WindowsProcess:
    parent_fds, child_fds = _pipes()
    job_handle = _create_job(limits)
    attribute_buffer: ctypes.Array[ctypes.c_char] | None = None
    process_info = _PROCESS_INFORMATION()
    try:
        handles = (wintypes.HANDLE * 3)(
            _os_handle(child_fds[0]),
            _os_handle(child_fds[1]),
            _os_handle(child_fds[2]),
        )
        security = _SECURITY_CAPABILITIES(
            AppContainerSid=appcontainer_sid,
            Capabilities=None,
            CapabilityCount=0,
            Reserved=0,
        )
        attribute_buffer = _attribute_list(handles, security)
        startup = _STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = handles[0]
        startup.StartupInfo.hStdOutput = handles[1]
        startup.StartupInfo.hStdError = handles[2]
        startup.lpAttributeList = ctypes.cast(attribute_buffer, wintypes.LPVOID)
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        env_block = _environment_block(command[0], artifact_root, profile_root, environment)
        flags = (
            _CREATE_SUSPENDED
            | _CREATE_UNICODE_ENVIRONMENT
            | _EXTENDED_STARTUPINFO_PRESENT
            | _CREATE_NO_WINDOW
        )
        created = _kernel32.CreateProcessW(
            command[0],
            command_line,
            None,
            None,
            True,
            flags,
            env_block,
            str(artifact_root),
            ctypes.byref(startup),
            ctypes.byref(process_info),
        )
        _check_bool(created, "CreateProcessW")
        _check_bool(
            _kernel32.AssignProcessToJobObject(job_handle, process_info.hProcess),
            "AssignProcessToJobObject",
        )
        if _kernel32.ResumeThread(process_info.hThread) == 0xFFFFFFFF:
            raise _win_error(_last_error())
    except BaseException:
        if process_info.hProcess:
            _kernel32.TerminateProcess(process_info.hProcess, 1)
            _kernel32.CloseHandle(process_info.hProcess)
        _kernel32.CloseHandle(job_handle)
        _close_fds((*parent_fds, *child_fds))
        raise
    finally:
        if attribute_buffer is not None:
            _kernel32.DeleteProcThreadAttributeList(attribute_buffer)
        if process_info.hThread:
            _kernel32.CloseHandle(process_info.hThread)
    _close_fds(child_fds)
    stdin = os.fdopen(parent_fds[0], "w", encoding="utf-8", errors="replace", buffering=1)
    stdout = os.fdopen(parent_fds[1], "r", encoding="utf-8", errors="replace", buffering=1)
    stderr = os.fdopen(parent_fds[2], "r", encoding="utf-8", errors="replace", buffering=1)
    return WindowsProcess(
        command=command,
        process_handle=process_info.hProcess,
        job_handle=job_handle,
        process_id=process_info.dwProcessId,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cleanup=cleanup,
    )


def _pipes() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    child_stdin, parent_stdin = os.pipe()
    parent_stdout, child_stdout = os.pipe()
    parent_stderr, child_stderr = os.pipe()
    parent = (parent_stdin, parent_stdout, parent_stderr)
    child = (child_stdin, child_stdout, child_stderr)
    for descriptor in parent:
        os.set_inheritable(descriptor, False)
    for descriptor in child:
        os.set_inheritable(descriptor, True)
    return parent, child


def _attribute_list(
    handles: ctypes.Array[wintypes.HANDLE], security: _SECURITY_CAPABILITIES
) -> ctypes.Array[ctypes.c_char]:
    size = ctypes.c_size_t()
    _kernel32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    _check_bool(
        _kernel32.InitializeProcThreadAttributeList(buffer, 2, 0, ctypes.byref(size)),
        "InitializeProcThreadAttributeList",
    )
    _check_bool(
        _kernel32.UpdateProcThreadAttribute(
            buffer,
            0,
            _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            handles,
            ctypes.sizeof(handles),
            None,
            None,
        ),
        "UpdateProcThreadAttribute(handle list)",
    )
    _check_bool(
        _kernel32.UpdateProcThreadAttribute(
            buffer,
            0,
            _PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
            ctypes.byref(security),
            ctypes.sizeof(security),
            None,
            None,
        ),
        "UpdateProcThreadAttribute(security capabilities)",
    )
    return buffer


def _create_job(limits: IsolationLimits) -> int:
    handle = _kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise _win_error(_last_error())
    information = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    basic = information.BasicLimitInformation
    basic.PerJobUserTimeLimit = limits.cpu_seconds * 10_000_000
    basic.ActiveProcessLimit = limits.processes
    basic.LimitFlags = (
        _JOB_OBJECT_LIMIT_JOB_TIME
        | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | _JOB_OBJECT_LIMIT_JOB_MEMORY
        | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    information.JobMemoryLimit = limits.memory_bytes
    try:
        _check_bool(
            _kernel32.SetInformationJobObject(
                handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ),
            "SetInformationJobObject",
        )
    except BaseException:
        _kernel32.CloseHandle(handle)
        raise
    return int(handle)


def _create_profile(name: str) -> int:
    sid = wintypes.LPVOID()
    result = _userenv.CreateAppContainerProfile(
        name,
        "MeshProbe evaluation agent",
        "Per-episode read-only 3D inspection agent",
        None,
        0,
        ctypes.byref(sid),
    )
    if result & 0xFFFFFFFF == _HRESULT_ALREADY_EXISTS:
        raise RuntimeError(f"AppContainer profile unexpectedly exists: {name}")
    if result != 0:
        error = result & 0xFFFF
        raise OSError(error, _format_error(error))
    if sid.value is None:
        raise RuntimeError("CreateAppContainerProfile returned no SID")
    return int(sid.value)


def _sid_string(sid: int) -> str:
    value = wintypes.LPWSTR()
    _check_bool(
        _advapi32.ConvertSidToStringSidW(sid, ctypes.byref(value)),
        "ConvertSidToStringSidW",
    )
    try:
        if value.value is None:
            raise RuntimeError("ConvertSidToStringSidW returned no SID string")
        return value.value
    finally:
        _kernel32.LocalFree(value)


def _profile_folder(sid: str) -> Path:
    value = wintypes.LPWSTR()
    result = _userenv.GetAppContainerFolderPath(sid, ctypes.byref(value))
    if result != 0:
        error = result & 0xFFFF
        raise OSError(error, f"GetAppContainerFolderPath: {_format_error(error)}")
    try:
        if value.value is None:
            raise RuntimeError("GetAppContainerFolderPath returned no path")
        return Path(value.value).resolve(strict=True)
    finally:
        _ole32.CoTaskMemFree(value)


def _grant_acl(path: Path, sid: str, rights: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    ace = f"*{sid}:(OI)(CI){rights}" if path.is_dir() else f"*{sid}:{rights}"
    completed = subprocess.run(
        ("icacls.exe", str(path), "/grant", ace, "/Q"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise SandboxUnavailable(
            f"failed to grant AppContainer access to {path}: {completed.stderr.strip()}"
        )


def _cleanup_profile(name: str, sid: str, roots: list[Path]) -> None:
    for root in reversed(roots):
        subprocess.run(
            ("icacls.exe", str(root), "/remove:g", f"*{sid}", "/Q"),
            capture_output=True,
            check=False,
        )
    _userenv.DeleteAppContainerProfile(name)


def _runtime_roots(executable: Path) -> tuple[Path, ...]:
    root = executable.resolve(strict=True).parent
    if root.name.casefold() == "scripts" and (root.parent / "pyvenv.cfg").is_file():
        root = root.parent
    roots = [root]
    configuration = root / "pyvenv.cfg"
    if configuration.is_file():
        for line in configuration.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip().casefold() == "home":
                home = Path(value.strip()).resolve()
                if home.exists() and home not in roots:
                    roots.append(home)
    return tuple(roots)


def _environment_block(
    executable: str,
    artifact_root: Path,
    profile_root: Path,
    configured: Mapping[str, str],
) -> object:
    system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    drive = artifact_root.drive.upper()
    values = {
        f"={drive}": str(artifact_root),
        "APPDATA": str(profile_root),
        "COMSPEC": os.environ.get("COMSPEC", str(Path(system_root) / "System32" / "cmd.exe")),
        "HOME": str(artifact_root),
        "LANG": "C.UTF-8",
        "LOCALAPPDATA": str(profile_root),
        "PATH": os.pathsep.join(
            (str(Path(executable).parent), str(Path(system_root) / "System32"))
        ),
        "SystemRoot": system_root,
        "TEMP": str(profile_root / "Temp"),
        "TMP": str(profile_root / "Temp"),
        "USERPROFILE": str(artifact_root),
    }
    for name, value in configured.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("sandbox environment names and values must be strings")
        if not name or "=" in name or "\x00" in name or "\x00" in value:
            raise ValueError(f"invalid sandbox environment entry: {name!r}")
        values[name] = value
    serialized = (
        "\x00".join(
            f"{name}={value}"
            for name, value in sorted(values.items(), key=lambda item: item[0].upper())
        )
        + "\x00"
    )
    return ctypes.create_unicode_buffer(serialized)


def _os_handle(descriptor: int) -> int:
    import msvcrt

    return int(msvcrt.get_osfhandle(descriptor))  # type: ignore[attr-defined]


def _close_fds(descriptors: tuple[int, ...]) -> None:
    for descriptor in descriptors:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _read_to_end(stream: IO[str] | None, destination: list[str]) -> None:
    if stream is not None:
        destination.append(stream.read())


def _check_bool(value: int, operation: str) -> None:
    if not value:
        error = _last_error()
        raise OSError(error, f"{operation}: {_format_error(error)}")


def _last_error() -> int:
    return int(_get_last_error())


def _signed_exit_code(value: int) -> int:
    return ctypes.c_long(value).value


_kernel32: Any
_advapi32: Any
_userenv: Any
_ole32: Any

if os.name == "nt":
    if _win_dll is None:
        raise RuntimeError("ctypes.WinDLL is unavailable on Windows")
    _kernel32 = _win_dll("kernel32", use_last_error=True)
    _advapi32 = _win_dll("advapi32", use_last_error=True)
    _userenv = _win_dll("userenv", use_last_error=True)
    _ole32 = _win_dll("ole32", use_last_error=True)

    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPCWSTR,
        wintypes.LPVOID,
        ctypes.POINTER(_PROCESS_INFORMATION),
    ]
    _kernel32.CreateProcessW.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateProcess.restype = wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    _kernel32.ResumeThread.restype = wintypes.DWORD
    _kernel32.InitializeProcThreadAttributeList.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
    _kernel32.UpdateProcThreadAttribute.argtypes = [
        wintypes.LPVOID,
        ctypes.c_size_t,
        ctypes.c_size_t,
        wintypes.LPVOID,
        ctypes.c_size_t,
        wintypes.LPVOID,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
    _kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
    _kernel32.DeleteProcThreadAttributeList.restype = None
    _kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    _kernel32.LocalFree.restype = wintypes.HLOCAL
    _advapi32.ConvertSidToStringSidW.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    _advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    _advapi32.FreeSid.argtypes = [wintypes.LPVOID]
    _advapi32.FreeSid.restype = wintypes.LPVOID
    _userenv.CreateAppContainerProfile.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.POINTER(_SID_AND_ATTRIBUTES),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
    ]
    _userenv.CreateAppContainerProfile.restype = ctypes.c_long
    _userenv.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
    _userenv.DeleteAppContainerProfile.restype = ctypes.c_long
    _userenv.GetAppContainerFolderPath.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    _userenv.GetAppContainerFolderPath.restype = ctypes.c_long
    _ole32.CoTaskMemFree.argtypes = [wintypes.LPVOID]
    _ole32.CoTaskMemFree.restype = None
else:
    _kernel32 = None
    _advapi32 = None
    _userenv = None
    _ole32 = None
