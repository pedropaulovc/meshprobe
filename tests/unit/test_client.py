from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from meshprobe.client import MeshProbeClient
from meshprobe.workspace import SessionMetadata, atomic_json, utc_now


def _daemon_metadata(pid: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "host": "127.0.0.1",
        "port": 12345,
        "pid": pid,
        "token": "secret",
    }


def _session_metadata(name: str = "review") -> SessionMetadata:
    now = utc_now()
    return SessionMetadata(
        name=name,
        source_path="/models/assembly.glb",
        source_sha256="a" * 64,
        blender="/opt/blender/blender",
        status="closed",
        created_at=now,
        updated_at=now,
    )


def test_list_sessions_reads_durable_metadata_without_daemon(tmp_path: Path) -> None:
    client = MeshProbeClient(tmp_path)
    metadata = _session_metadata()
    path = client.root / "sessions" / metadata.name / "metadata.json"
    atomic_json(path, metadata.model_dump(mode="json"))

    assert client.list_sessions() == [metadata.model_dump(mode="json")]


def test_list_sessions_falls_back_when_live_pid_has_no_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    metadata = _session_metadata()
    atomic_json(
        client.root / "sessions" / metadata.name / "metadata.json",
        metadata.model_dump(mode="json"),
    )
    atomic_json(client.root / "daemon.json", _daemon_metadata(os.getpid()), mode=0o600)
    monkeypatch.setattr(
        client,
        "request",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError()),
    )

    assert client.list_sessions() == [metadata.model_dump(mode="json")]


def test_stale_start_lock_is_reclaimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = MeshProbeClient(tmp_path)
    client.root.mkdir(parents=True)
    lock = client.root / "daemon.lock"
    lock.write_text('{"pid":999999,"created_at":0}\n', encoding="utf-8")
    monkeypatch.setattr(
        MeshProbeClient,
        "_pid_alive",
        staticmethod(lambda pid: pid == os.getpid()),
    )

    assert client._acquire_start_lock(lock) is True
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_request_does_not_retry_after_read_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TimeoutConnection:
        def __enter__(self) -> TimeoutConnection:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def settimeout(self, timeout: float) -> None:
            assert timeout > 10

        def sendall(self, payload: bytes) -> None:
            assert b'"action":"execute"' in payload

        def recv(self, size: int) -> bytes:
            raise TimeoutError("renderer still running")

    client = MeshProbeClient(tmp_path)
    atomic_json(client.root / "daemon.json", _daemon_metadata(os.getpid()), mode=0o600)
    connections = 0

    def connect(*args: object, **kwargs: object) -> TimeoutConnection:
        nonlocal connections
        connections += 1
        return TimeoutConnection()

    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(
        client,
        "_start_daemon",
        lambda: pytest.fail("a delivered operation must not be retried"),
    )

    with pytest.raises(TimeoutError, match="renderer still running"):
        client.request("execute", command={})
    assert connections == 1


def test_connection_refusal_replaces_metadata_even_when_pid_was_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ResponseConnection:
        def __enter__(self) -> ResponseConnection:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def settimeout(self, timeout: float) -> None:
            assert timeout > 0

        def sendall(self, payload: bytes) -> None:
            assert b'"token":"fresh"' in payload

        def recv(self, size: int) -> bytes:
            return b'{"ok":true,"result":{"recovered":true}}\n'

    client = MeshProbeClient(tmp_path)
    stale = _daemon_metadata(os.getpid())
    fresh = {**stale, "pid": os.getpid() + 1, "token": "fresh", "port": 54321}
    atomic_json(client.root / "daemon.json", stale, mode=0o600)
    attempts = 0

    def connect(*args: object, **kwargs: object) -> ResponseConnection:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionRefusedError("stale daemon socket")
        return ResponseConnection()

    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(client, "_start_daemon", lambda: fresh)
    monkeypatch.setattr(client, "_alive", lambda metadata: True)

    assert client.request("list") == {"recovered": True}
    assert attempts == 2
    assert not (client.root / "daemon.json").exists()


def test_resolve_component_ids_falls_back_to_resolve_on_old_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    actions: list[str] = []

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        actions.append(action)
        if action == "resolve-ids":
            raise ValueError("unsupported daemon action: resolve-ids")
        return {"component_id": "component-42"}

    monkeypatch.setattr(client, "request", request)

    assert client.resolve_component_ids("review", "c2") == ("component-42",)
    assert actions == ["resolve-ids", "resolve"]


def test_resolve_component_ids_propagates_unrelated_daemon_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        raise ValueError("no components match glob pattern: **/missing")

    monkeypatch.setattr(client, "request", request)

    with pytest.raises(ValueError, match="no components match glob pattern"):
        client.resolve_component_ids("review", "**/missing")


def test_close_all_waits_for_graceful_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    metadata = _daemon_metadata(1234)
    waited: list[bool] = []
    monkeypatch.setattr(client, "_metadata", lambda optional=False: metadata)
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: {"sessions": []})
    monkeypatch.setattr(
        client,
        "_wait_for_shutdown",
        lambda expected, *, force: waited.append(force),
    )

    assert client.close_all() == []
    assert waited == [False]


def test_kill_all_falls_back_to_process_tree_and_updates_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    daemon = _daemon_metadata(1234)
    session = _session_metadata()
    atomic_json(
        client.root / "sessions" / session.name / "metadata.json",
        session.model_dump(mode="json"),
    )
    forced: list[bool] = []
    monkeypatch.setattr(client, "_metadata", lambda optional=False: daemon)
    monkeypatch.setattr(
        client,
        "request",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("wedged")),
    )
    monkeypatch.setattr(client, "_authenticated_daemon", lambda metadata: True)
    monkeypatch.setattr(
        client,
        "_wait_for_shutdown",
        lambda expected, *, force, authenticated=False: forced.append(force),
    )

    receipts = client.kill_all()

    persisted = json.loads(
        (client.root / "sessions" / session.name / "metadata.json").read_text(encoding="utf-8")
    )
    assert forced == [True]
    assert [receipt.session for receipt in receipts] == [session.name]
    assert persisted["status"] == "killed"
    assert persisted["worker_pid"] is None


def test_force_shutdown_signals_daemon_authenticated_before_wedged_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    daemon = _daemon_metadata(1234)
    killed: list[int] = []
    monkeypatch.setattr("meshprobe.client.KILL_RPC_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(client, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(client, "_authenticated_daemon", lambda metadata: False)
    monkeypatch.setattr(client, "_kill_process_tree", killed.append)

    client._wait_for_shutdown(daemon, force=True, authenticated=True)

    assert killed == [1234]


def test_kill_all_does_not_signal_a_reused_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    daemon = _daemon_metadata(1234)
    session = _session_metadata()
    atomic_json(client.root / "daemon.json", daemon)
    atomic_json(
        client.root / "sessions" / session.name / "metadata.json",
        session.model_dump(mode="json"),
    )
    monkeypatch.setattr(client, "_authenticated_daemon", lambda metadata: False)
    monkeypatch.setattr(
        client,
        "_kill_process_tree",
        lambda pid: pytest.fail(f"must not signal unverified PID {pid}"),
    )

    receipts = client.kill_all()

    persisted = json.loads(
        (client.root / "sessions" / session.name / "metadata.json").read_text(encoding="utf-8")
    )
    assert [receipt.session for receipt in receipts] == [session.name]
    assert persisted["status"] == "killed"
    assert not (client.root / "daemon.json").exists()


def test_windows_pid_liveness_does_not_send_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_kill(pid: int, signal_number: int) -> None:
        pytest.fail(f"os.kill({pid}, {signal_number}) must not probe a Windows process")

    monkeypatch.setattr("meshprobe.client.os.name", "nt")
    monkeypatch.setattr("meshprobe.client.os.kill", unexpected_kill)
    monkeypatch.setattr(MeshProbeClient, "_windows_pid_alive", lambda pid: pid == 42)

    assert MeshProbeClient._pid_alive(42)
    assert not MeshProbeClient._pid_alive(41)
