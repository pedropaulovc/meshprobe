"""Client for the project-local MeshProbe daemon."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from meshprobe.controller import DEFAULT_WORKER_TIMEOUT_SECONDS
from meshprobe.protocol import Command, SceneOpenCommand
from meshprobe.workspace import (
    OperationReceipt,
    SessionFiles,
    SessionMetadata,
    atomic_json,
    utc_now,
    workspace_root,
)

CONNECT_TIMEOUT_SECONDS = 10.0
DAEMON_START_TIMEOUT_SECONDS = 10.0
DAEMON_SHUTDOWN_TIMEOUT_SECONDS = 10.0
OPERATION_READ_TIMEOUT_SECONDS = DEFAULT_WORKER_TIMEOUT_SECONDS + 30.0
KILL_RPC_TIMEOUT_SECONDS = 1.0


class MeshProbeClient:
    """Execute operations against one authenticated workspace daemon."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        blender: str | None = None,
    ) -> None:
        self.root = workspace_root(workspace)
        resolved_blender = shutil.which(blender) if blender is not None else None
        self.blender = str(Path(resolved_blender).resolve()) if resolved_blender else blender

    def execute(self, session: str, command: Command) -> OperationReceipt:
        # The `aspect_ratio` field was added after clients could already talk to a
        # persistent daemon. Omit only that *new* field when its default was not
        # explicitly requested: older daemons reject unknown fields, while known
        # defaults such as render width and height must still travel over the wire.
        # Dumping the full model also keeps nested discriminators intact.
        full_command = command.model_dump(mode="json")
        if isinstance(command, SceneOpenCommand) and "aspect_ratio" not in command.model_fields_set:
            full_command.pop("aspect_ratio")
        payload = self.request(
            "execute",
            session=session,
            command=full_command,
            blender=self.blender,
        )
        return OperationReceipt.model_validate(payload)

    def resolve_component(self, session: str, value: str) -> str:
        payload = self.request("resolve", session=session, value=value)
        return str(payload["component_id"])

    def resolve_component_ids(self, session: str, value: str) -> tuple[str, ...]:
        try:
            payload = self.request("resolve-ids", session=session, value=value)
        except ValueError as error:
            if "unsupported daemon action" not in str(error):
                raise
            # A daemon started before resolve-ids only speaks the exact `resolve`
            # action, so exact tokens keep working across an in-place upgrade;
            # globs surface the daemon's usual unknown-component error.
            return (self.resolve_component(session, value),)
        return tuple(str(component_id) for component_id in payload["component_ids"])

    def list_sessions(self) -> list[dict[str, Any]]:
        metadata = self._metadata(optional=True)
        if metadata is None or not self._alive(metadata):
            return self._persisted_sessions()
        try:
            payload = self.request("list", start=False, read_timeout=CONNECT_TIMEOUT_SECONDS)
        except (OSError, RuntimeError, TimeoutError, ValueError):
            return self._persisted_sessions()
        sessions = payload.get("sessions", [])
        return sessions if isinstance(sessions, list) else []

    def close(self, session: str) -> OperationReceipt:
        return OperationReceipt.model_validate(self.request("close", session=session, start=False))

    def kill(self, session: str) -> OperationReceipt:
        return OperationReceipt.model_validate(self.request("kill", session=session, start=False))

    def close_all(self) -> list[OperationReceipt]:
        metadata = self._metadata()
        assert metadata is not None
        payload = self.request(
            "close_all",
            start=False,
            read_timeout=OPERATION_READ_TIMEOUT_SECONDS,
        )
        self._wait_for_shutdown(metadata, force=False)
        return [OperationReceipt.model_validate(item) for item in payload.get("sessions", [])]

    def kill_all(self) -> list[OperationReceipt]:
        metadata = self._metadata()
        assert metadata is not None
        persisted = self._persisted_sessions()
        if not self._authenticated_daemon(metadata):
            self._remove_matching_metadata(metadata)
            return self._killed_receipts(persisted)
        payload: dict[str, Any] | None = None
        with suppress(OSError, RuntimeError, TimeoutError, ValueError):
            payload = self.request(
                "kill_all",
                start=False,
                read_timeout=KILL_RPC_TIMEOUT_SECONDS,
            )
        self._wait_for_shutdown(metadata, force=True, authenticated=True)
        if payload is not None:
            return [OperationReceipt.model_validate(item) for item in payload.get("sessions", [])]
        return self._killed_receipts(persisted)

    def _authenticated_daemon(self, metadata: dict[str, Any]) -> bool:
        try:
            payload = self.request(
                "ping",
                start=False,
                read_timeout=CONNECT_TIMEOUT_SECONDS,
            )
            return int(payload["pid"]) == int(metadata["pid"])
        except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            return False

    def _killed_receipts(self, persisted: list[dict[str, Any]]) -> list[OperationReceipt]:
        self._mark_sessions_killed(persisted)
        return [
            OperationReceipt(
                session=str(item["name"]),
                op="session.kill",
                state_path=SessionFiles(self.root, str(item["name"])).relative(
                    SessionFiles(self.root, str(item["name"])).state
                ),
            )
            for item in persisted
        ]

    def delete_data(self) -> Path:
        metadata = self._metadata(optional=True)
        if metadata is not None and self._alive(metadata):
            raise ValueError("close --all or kill --all before deleting workspace data")
        if self.root.exists():
            shutil.rmtree(self.root)
        return self.root

    def read_result(self, receipt: OperationReceipt) -> object:
        if receipt.result_path is None:
            return receipt.model_dump(mode="json")
        return json.loads((self.root.parent / receipt.result_path).read_text(encoding="utf-8"))

    def request(
        self,
        action: str,
        *,
        start: bool = True,
        read_timeout: float = OPERATION_READ_TIMEOUT_SECONDS,
        **arguments: object,
    ) -> dict[str, Any]:
        metadata = self._ensure_daemon() if start else self._metadata()
        if metadata is None:
            raise ValueError("meshprobe daemon is not running")
        if not self._alive(metadata):
            if not start:
                raise ValueError("meshprobe daemon is not running")
            metadata = self._start_daemon()
        request = {"token": metadata["token"], "action": action, **arguments}
        try:
            connection = socket.create_connection(
                (str(metadata["host"]), int(metadata["port"])),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except OSError:
            if not start:
                raise
            self._remove_matching_metadata(metadata)
            metadata = self._start_daemon()
            request["token"] = metadata["token"]
            connection = socket.create_connection(
                (str(metadata["host"]), int(metadata["port"])),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        with connection:
            connection.settimeout(read_timeout)
            connection.sendall((json.dumps(request, separators=(",", ":")) + "\n").encode())
            response = self._read_line(connection)
        if not response.get("ok"):
            error = response.get("error", {})
            raise ValueError(str(error.get("message", "daemon request failed")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError("daemon returned a non-object result")
        return result

    def _ensure_daemon(self) -> dict[str, Any]:
        metadata = self._metadata(optional=True)
        if metadata is not None and self._alive(metadata):
            return metadata
        return self._start_daemon()

    def _start_daemon(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / "daemon.lock"
        acquired = self._acquire_start_lock(lock)
        if acquired:
            metadata = self._metadata(optional=True)
            if metadata is not None and not self._alive(metadata):
                self._remove_stale_metadata(metadata)
            command = [
                sys.executable,
                "-m",
                "meshprobe.daemon",
                "--workspace",
                str(self.root),
            ]
            if self.blender is not None:
                command.extend(("--blender", self.blender))
            log = (self.root / "daemon.log").open("a", encoding="utf-8")
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=creation_flags,
                close_fds=True,
            )
            log.close()
        deadline = time.monotonic() + DAEMON_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            metadata = self._metadata(optional=True)
            if metadata is not None and self._alive(metadata):
                with suppress(FileNotFoundError):
                    lock.unlink()
                return metadata
            time.sleep(0.05)
        if acquired:
            with suppress(FileNotFoundError):
                lock.unlink()
        elif self._start_lock_stale(lock):
            with suppress(FileNotFoundError):
                lock.unlink()
            return self._start_daemon()
        log_text = ""
        log_path = self.root / "daemon.log"
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8")[-4000:]
        raise RuntimeError(
            f"meshprobe daemon did not start within {DAEMON_START_TIMEOUT_SECONDS:g} seconds\n"
            f"{log_text}"
        )

    def _acquire_start_lock(self, lock: Path) -> bool:
        for _ in range(2):
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if not self._start_lock_stale(lock):
                    return False
                with suppress(FileNotFoundError):
                    lock.unlink()
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, stream)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            return True
        return False

    @staticmethod
    def _start_lock_stale(lock: Path) -> bool:
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            owner = int(payload["pid"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            try:
                return time.time() - lock.stat().st_mtime >= DAEMON_START_TIMEOUT_SECONDS
            except FileNotFoundError:
                return True
        return not MeshProbeClient._pid_alive(owner)

    def _metadata(self, *, optional: bool = False) -> dict[str, Any] | None:
        path = self.root / "daemon.json"
        if not path.is_file():
            if optional:
                return None
            raise ValueError("meshprobe daemon is not running")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid daemon metadata")
        return payload

    @staticmethod
    def _alive(metadata: dict[str, Any]) -> bool:
        try:
            return MeshProbeClient._pid_alive(int(metadata["pid"]))
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if os.name == "nt":
            return MeshProbeClient._windows_pid_alive(pid)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _windows_pid_alive(pid: int) -> bool:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        process_query_limited_information = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    def _remove_stale_metadata(self, metadata: dict[str, Any]) -> None:
        if self._alive(metadata):
            return
        self._remove_matching_metadata(metadata)

    def _remove_matching_metadata(self, expected: dict[str, Any]) -> None:
        current = self._metadata(optional=True)
        if current is None:
            return
        if current.get("pid") != expected.get("pid") or current.get("token") != expected.get(
            "token"
        ):
            return
        with suppress(FileNotFoundError):
            (self.root / "daemon.json").unlink()

    def _persisted_sessions(self) -> list[dict[str, Any]]:
        sessions_root = self.root / "sessions"
        if not sessions_root.is_dir():
            return []
        sessions: list[dict[str, Any]] = []
        for path in sorted(sessions_root.iterdir()):
            metadata = path / "metadata.json"
            if not path.is_dir() or not metadata.is_file():
                continue
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"invalid session metadata: {metadata}")
            sessions.append(payload)
        return sessions

    def _mark_sessions_killed(self, sessions: list[dict[str, Any]]) -> None:
        for item in sessions:
            files = SessionFiles(self.root, str(item["name"]))
            if not files.metadata.is_file():
                continue
            metadata = SessionMetadata.model_validate_json(
                files.metadata.read_text(encoding="utf-8")
            )
            metadata.status = "killed"
            metadata.worker_pid = None
            metadata.updated_at = utc_now()
            atomic_json(files.metadata, metadata.model_dump(mode="json"))

    @staticmethod
    def _read_line(connection: socket.socket) -> dict[str, Any]:
        chunks: list[bytes] = []
        while True:
            chunk = connection.recv(64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        line = b"".join(chunks).split(b"\n", 1)[0]
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("daemon returned an invalid response")
        return payload

    def _wait_for_shutdown(
        self,
        metadata: dict[str, Any],
        *,
        force: bool,
        authenticated: bool = False,
    ) -> None:
        pid = int(metadata["pid"])
        wait_seconds = KILL_RPC_TIMEOUT_SECONDS if force else DAEMON_SHUTDOWN_TIMEOUT_SECONDS
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            metadata_gone = self._metadata(optional=True) is None
            pid_stopped = not self._pid_alive(pid)
            if pid_stopped or (metadata_gone and not force):
                self._remove_matching_metadata(metadata)
                return
            time.sleep(0.05)
        if not force:
            raise RuntimeError(
                f"meshprobe daemon {pid} did not stop within "
                f"{DAEMON_SHUTDOWN_TIMEOUT_SECONDS:g} seconds"
            )
        if not authenticated and not self._authenticated_daemon(metadata):
            self._remove_matching_metadata(metadata)
            return
        self._kill_process_tree(pid)
        self._remove_matching_metadata(metadata)

    @staticmethod
    def _kill_process_tree(pid: int) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
