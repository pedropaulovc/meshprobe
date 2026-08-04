from __future__ import annotations

import io
import json
from collections.abc import Buffer
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest

from meshprobe.daemon import DaemonServer, RequestHandler, _remove_matching_metadata
from meshprobe.workspace import atomic_json


class _RecordingWriter(io.BytesIO):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def write(self, data: Buffer, /) -> int:
        self.events.append("write")
        return super().write(data)

    def flush(self) -> None:
        self.events.append("flush")
        super().flush()


class _RecordingReader:
    def __init__(self, request: bytes, events: list[str]) -> None:
        self.stream = io.BytesIO(request)
        self.events = events

    def readline(self, limit: int = -1) -> bytes:
        return self.stream.readline(limit)

    def read(self, size: int = -1) -> bytes:
        self.events.append("client.eof")
        return self.stream.read(size)


class _TimeoutReader(_RecordingReader):
    def read(self, size: int = -1) -> bytes:
        del size
        self.events.append("client.timeout")
        raise TimeoutError


class _RecordingConnection:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def settimeout(self, timeout: float | None) -> None:
        self.events.append(f"settimeout:{timeout}")


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


@pytest.mark.parametrize("action", ["close_all", "kill_all"])
def test_daemon_flushes_shutdown_response_before_stopping_server(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def record_close_all() -> list[object]:
        events.append("close_all")
        return []

    def record_kill_all() -> list[object]:
        events.append("kill_all")
        return []

    def record_manager_shutdown(*, force: bool) -> None:
        events.append(f"manager.shutdown:{force}")

    def record_server_shutdown() -> None:
        events.append("server.shutdown")

    manager = Mock()
    manager.close_all.side_effect = record_close_all
    manager.kill_all.side_effect = record_kill_all
    manager.shutdown.side_effect = record_manager_shutdown

    server = object.__new__(DaemonServer)
    untyped_server = cast(Any, server)
    untyped_server.token = "test-token"
    untyped_server.manager = manager
    monkeypatch.setattr(server, "shutdown", record_server_shutdown)

    handler = object.__new__(RequestHandler)
    untyped_handler = cast(Any, handler)
    writer = _RecordingWriter(events)
    untyped_handler.server = server
    untyped_handler.rfile = _RecordingReader(
        json.dumps({"token": "test-token", "action": action}).encode() + b"\n", events
    )
    untyped_handler.wfile = writer
    untyped_handler.connection = _RecordingConnection(events)

    handler.handle()

    assert json.loads(writer.getvalue()) == {
        "ok": True,
        "result": {"sessions": []},
    }
    force = action == "kill_all"
    assert events == [
        action,
        f"manager.shutdown:{force}",
        "write",
        "flush",
        "settimeout:1.0",
        "client.eof",
        "server.shutdown",
    ]


def test_daemon_shutdown_does_not_wait_forever_for_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manager = Mock()
    manager.close_all.return_value = []

    def record_server_shutdown() -> None:
        events.append("server.shutdown")

    server = object.__new__(DaemonServer)
    untyped_server = cast(Any, server)
    untyped_server.token = "test-token"
    untyped_server.manager = manager
    monkeypatch.setattr(server, "shutdown", record_server_shutdown)

    handler = object.__new__(RequestHandler)
    untyped_handler = cast(Any, handler)
    untyped_handler.server = server
    untyped_handler.rfile = _TimeoutReader(
        json.dumps({"token": "test-token", "action": "close_all"}).encode() + b"\n",
        events,
    )
    untyped_handler.wfile = _RecordingWriter(events)
    untyped_handler.connection = _RecordingConnection(events)

    handler.handle()

    assert events == [
        "write",
        "flush",
        "settimeout:1.0",
        "client.timeout",
        "server.shutdown",
    ]
