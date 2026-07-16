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
from pathlib import Path
from typing import Any

from meshprobe.protocol import Command
from meshprobe.workspace import OperationReceipt, workspace_root


class MeshProbeClient:
    """Execute operations against one authenticated workspace daemon."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        blender: str | None = None,
    ) -> None:
        self.root = workspace_root(workspace)
        self.blender = blender

    def execute(self, session: str, command: Command) -> OperationReceipt:
        payload = self.request(
            "execute", session=session, command=command.model_dump(mode="json")
        )
        return OperationReceipt.model_validate(payload)

    def resolve_component(self, session: str, value: str) -> str:
        payload = self.request("resolve", session=session, value=value)
        return str(payload["component_id"])

    def list_sessions(self) -> list[dict[str, Any]]:
        payload = self.request("list", start=False)
        sessions = payload.get("sessions", [])
        return sessions if isinstance(sessions, list) else []

    def close(self, session: str) -> OperationReceipt:
        return OperationReceipt.model_validate(self.request("close", session=session, start=False))

    def kill(self, session: str) -> OperationReceipt:
        return OperationReceipt.model_validate(self.request("kill", session=session, start=False))

    def close_all(self) -> list[OperationReceipt]:
        payload = self.request("close_all", start=False)
        return [OperationReceipt.model_validate(item) for item in payload.get("sessions", [])]

    def kill_all(self) -> list[OperationReceipt]:
        metadata = self._metadata()
        payload = self.request("kill_all", start=False)
        self._ensure_stopped(int(metadata["pid"]))
        return [OperationReceipt.model_validate(item) for item in payload.get("sessions", [])]

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

    def request(self, action: str, *, start: bool = True, **arguments: object) -> dict[str, Any]:
        metadata = self._ensure_daemon() if start else self._metadata()
        if not self._alive(metadata):
            if not start:
                raise ValueError("meshprobe daemon is not running")
            metadata = self._start_daemon()
        request = {"token": metadata["token"], "action": action, **arguments}
        try:
            with socket.create_connection(
                (str(metadata["host"]), int(metadata["port"])), timeout=10
            ) as connection:
                connection.sendall((json.dumps(request, separators=(",", ":")) + "\n").encode())
                response = self._read_line(connection)
        except OSError:
            if not start:
                raise
            self._remove_stale_metadata(metadata)
            metadata = self._start_daemon()
            request["token"] = metadata["token"]
            with socket.create_connection(
                (str(metadata["host"]), int(metadata["port"])), timeout=10
            ) as connection:
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
        acquired = False
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            acquired = True
        except FileExistsError:
            pass
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
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            metadata = self._metadata(optional=True)
            if metadata is not None and self._alive(metadata):
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
                return metadata
            time.sleep(0.05)
        if acquired:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
        log_text = ""
        log_path = self.root / "daemon.log"
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8")[-4000:]
        raise RuntimeError(f"meshprobe daemon did not start within 10 seconds\n{log_text}")

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
            os.kill(int(metadata["pid"]), 0)
        except (KeyError, OSError, TypeError, ValueError):
            return False
        return True

    def _remove_stale_metadata(self, metadata: dict[str, Any]) -> None:
        if self._alive(metadata):
            return
        try:
            (self.root / "daemon.json").unlink()
        except FileNotFoundError:
            pass

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

    @staticmethod
    def _ensure_stopped(pid: int) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.05)
        if os.name == "nt":
            os.kill(pid, signal.SIGTERM)
            return
        os.killpg(pid, signal.SIGKILL)
