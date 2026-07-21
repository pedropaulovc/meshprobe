from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, cast

import pytest

from meshprobe.client import MeshProbeClient
from meshprobe.models import DisplayMode, IsolationOperation, OrbitSweep, PerspectiveProjection
from meshprobe.protocol import (
    ComponentDisplayCommand,
    RenderComparisonRequest,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionResetCommand,
    SessionUndoCommand,
    ViewOrbitCommand,
)
from meshprobe.workspace import (
    OperationReceipt,
    SessionFiles,
    SessionMetadata,
    atomic_json,
    utc_now,
)


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


def test_execute_omits_default_unit_scale_from_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    captured: list[dict[str, Any]] = []

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        captured.append(arguments["command"])
        return {"session": "review", "op": "scene.open"}

    monkeypatch.setattr(client, "request", request)

    command = SceneOpenCommand(request_id="open", op="scene.open", source_path="/tmp/model.glb")
    client.execute("review", command)

    # A daemon that predates the unit_scale field validates with extra="forbid"; an ordinary
    # open must not send the new key or it would reject every open until restarted.
    assert "unit_scale" not in captured[0]


def test_execute_sends_overridden_unit_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    captured: list[dict[str, Any]] = []

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        captured.append(arguments["command"])
        return {"session": "review", "op": "scene.open"}

    monkeypatch.setattr(client, "request", request)

    command = SceneOpenCommand(
        request_id="open", op="scene.open", source_path="/tmp/model.glb", unit_scale=0.001
    )
    client.execute("review", command)

    assert captured[0]["unit_scale"] == 0.001


def test_execute_restarts_old_daemon_for_overridden_unit_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    captured: list[dict[str, Any]] = []
    close_all_calls = 0

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        captured.append({"action": action, **arguments})
        if (
            action == "execute"
            and len([call for call in captured if call["action"] == "execute"]) == 1
        ):
            raise ValueError("unit_scale: Extra inputs are not permitted")
        return {"session": "review", "op": "scene.open"}

    def close_all() -> list[OperationReceipt]:
        nonlocal close_all_calls
        close_all_calls += 1
        return []

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "close_all", close_all)

    command = SceneOpenCommand(
        request_id="open", op="scene.open", source_path="/tmp/model.glb", unit_scale=0.001
    )
    client.execute("review", command)

    assert close_all_calls == 1
    execute_commands = [call["command"] for call in captured if call["action"] == "execute"]
    assert execute_commands == [
        {
            "request_id": "open",
            "op": "scene.open",
            "source_path": "/tmp/model.glb",
            "unit_scale": 0.001,
        },
        {
            "request_id": "open",
            "op": "scene.open",
            "source_path": "/tmp/model.glb",
            "unit_scale": 0.001,
        },
    ]


def test_execute_restarts_old_daemon_for_explicit_aspect_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    captured: list[dict[str, Any]] = []
    close_all_calls = 0

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        captured.append({"action": action, **arguments})
        if (
            action == "execute"
            and len([call for call in captured if call["action"] == "execute"]) == 1
        ):
            raise ValueError("aspect_ratio: Extra inputs are not permitted")
        return {"session": "review", "op": "session.reset"}

    def close_all() -> list[OperationReceipt]:
        nonlocal close_all_calls
        close_all_calls += 1
        return []

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "close_all", close_all)

    client.execute(
        "review",
        SessionResetCommand(request_id="reset", op="session.reset", aspect_ratio=2.5),
    )

    assert close_all_calls == 1
    execute_commands = [call["command"] for call in captured if call["action"] == "execute"]
    assert execute_commands == [
        {"request_id": "reset", "op": "session.reset", "aspect_ratio": 2.5},
        {"request_id": "reset", "op": "session.reset", "aspect_ratio": 2.5},
    ]


def test_execute_restarts_old_daemon_for_additive_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    captured: list[dict[str, Any]] = []
    close_all_calls = 0

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        captured.append({"action": action, **arguments})
        if (
            action == "execute"
            and len([call for call in captured if call["action"] == "execute"]) == 1
        ):
            raise ValueError("isolation_operation: Extra inputs are not permitted")
        return {"session": "review", "op": "component.display"}

    def close_all() -> list[OperationReceipt]:
        nonlocal close_all_calls
        close_all_calls += 1
        return []

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "close_all", close_all)

    client.execute(
        "review",
        ComponentDisplayCommand(
            request_id="isolate-add",
            op="component.display",
            component_ids=("cmp-a",),
            mode=DisplayMode.ISOLATED,
            isolation_operation=IsolationOperation.ADD,
        ),
    )

    assert close_all_calls == 1
    execute_commands = [call["command"] for call in captured if call["action"] == "execute"]
    assert execute_commands == [
        {
            "request_id": "isolate-add",
            "op": "component.display",
            "component_ids": ["cmp-a"],
            "mode": "isolated",
            "isolation_operation": "add",
        },
        {
            "request_id": "isolate-add",
            "op": "component.display",
            "component_ids": ["cmp-a"],
            "mode": "isolated",
            "isolation_operation": "add",
        },
    ]


@pytest.mark.parametrize(
    ("command", "field", "response_op"),
    [
        (
            RenderImageCommand(
                request_id="comparison",
                op="render.image",
                output_path="evidence.png",
                comparison=RenderComparisonRequest(
                    reference_image_path="reference.png",
                    mode="side_by_side",
                    output_path="comparison.png",
                ),
            ),
            "comparison",
            "render.image",
        ),
        (
            RenderImageCommand(
                request_id="comparison-timeout",
                op="render.image",
                output_path="evidence.png",
                timeout_seconds=600,
                comparison=RenderComparisonRequest(
                    reference_image_path="reference.png",
                    mode="side_by_side",
                    output_path="comparison.png",
                ),
            ),
            "timeout_seconds",
            "render.image",
        ),
        (
            RenderContactSheetCommand(
                request_id="sweep",
                op="render.contact_sheet",
                output_path="sweep.png",
                recipe="orbit_sweep",
                orbit_sweep=OrbitSweep(azimuth_degrees=(0,), elevation_degrees=(0,)),
            ),
            "orbit_sweep",
            "render.contact_sheet",
        ),
    ],
)
def test_execute_restarts_old_daemon_for_new_render_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: RenderImageCommand | RenderContactSheetCommand,
    field: str,
    response_op: str,
) -> None:
    client = MeshProbeClient(tmp_path)
    attempts = 0
    close_all_calls = 0

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        nonlocal attempts
        if action == "execute":
            attempts += 1
            if attempts == 1:
                raise ValueError(f"{field}: Extra inputs are not permitted")
        return {"session": "review", "op": response_op}

    def close_all() -> list[OperationReceipt]:
        nonlocal close_all_calls
        close_all_calls += 1
        return []

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "close_all", close_all)

    receipt = client.execute("review", command)

    assert receipt.op == response_op
    assert attempts == 2
    assert close_all_calls == 1


def test_execute_restarts_old_daemon_that_does_not_recognize_session_undo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    attempts = 0
    close_all_calls = 0

    def request(action: str, **arguments: Any) -> dict[str, Any]:
        nonlocal attempts
        if action == "execute":
            attempts += 1
            if attempts == 1:
                raise ValueError(
                    "Input tag 'session.undo' does not match any expected discriminator tags"
                )
        return {"session": "review", "op": "session.undo"}

    def close_all() -> list[OperationReceipt]:
        nonlocal close_all_calls
        close_all_calls += 1
        return []

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "close_all", close_all)

    receipt = client.execute("review", SessionUndoCommand(request_id="undo", op="session.undo"))

    assert receipt.op == "session.undo"
    assert attempts == 2
    assert close_all_calls == 1


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


def test_execute_omits_only_the_new_default_aspect_ratio_from_the_wire_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An already-running daemon from a previous version validates commands with
    # extra="forbid" and may not know about a newly added field (e.g.
    # aspect_ratio); sending it unconditionally would reject every command from
    # an upgraded client until the daemon is restarted.
    client = MeshProbeClient(tmp_path)
    captured: dict[str, Any] = {}

    def fake_request(action: str, **arguments: object) -> dict[str, Any]:
        captured.update(arguments)
        return {"session": "review", "op": "scene.open"}

    monkeypatch.setattr(client, "request", fake_request)

    client.execute(
        "review",
        SceneOpenCommand(request_id="open", op="scene.open", source_path="/models/assembly.glb"),
    )
    assert "aspect_ratio" not in captured["command"]

    client.execute(
        "review",
        SceneOpenCommand(
            request_id="open",
            op="scene.open",
            source_path="/models/assembly.glb",
            aspect_ratio=2.5,
        ),
    )
    assert captured["command"]["aspect_ratio"] == 2.5

    client.execute(
        "review",
        SessionResetCommand(request_id="reset", op="session.reset"),
    )
    assert "aspect_ratio" not in captured["command"]

    client.execute(
        "review",
        SessionResetCommand(request_id="reset", op="session.reset", aspect_ratio=2.5),
    )
    assert captured["command"]["aspect_ratio"] == 2.5


def test_execute_keeps_established_default_values_on_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    captured: dict[str, Any] = {}

    def fake_request(action: str, **arguments: object) -> dict[str, Any]:
        captured.update(arguments)
        return {"session": "review", "op": "render.image"}

    monkeypatch.setattr(client, "request", fake_request)
    client.execute(
        "review",
        RenderImageCommand(request_id="render", op="render.image", output_path="/tmp/render.png"),
    )
    assert captured["command"]["width"] == 2576
    assert captured["command"]["height"] == 2576


def test_render_execution_extends_the_daemon_read_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MeshProbeClient(tmp_path)
    captured: dict[str, object] = {}

    def fake_request(action: str, **arguments: object) -> dict[str, object]:
        captured["action"] = action
        captured.update(arguments)
        return {"session": "review", "op": "render.image"}

    monkeypatch.setattr(client, "request", fake_request)
    client.execute(
        "review",
        RenderImageCommand(
            request_id="render",
            op="render.image",
            output_path="/tmp/render.png",
            timeout_seconds=600,
        ),
    )

    assert captured["read_timeout"] == 1230

    client.execute(
        "review",
        RenderImageCommand(
            request_id="render-short",
            op="render.image",
            output_path="/tmp/render-short.png",
            timeout_seconds=1,
        ),
    )
    assert cast(float, captured["read_timeout"]) == 631


def test_render_execution_budgets_time_for_checkpoint_replay(tmp_path: Path) -> None:
    client = MeshProbeClient(tmp_path)
    files = SessionFiles(client.root, "review")
    files.create_directories()
    files.checkpoint.write_text(
        json.dumps(
            {
                "replay_prefix": [{"op": "view.set"}],
                "accepted_commands": [
                    {"op": "view.orbit"},
                    {"op": "component.display"},
                ],
            }
        ),
        encoding="utf-8",
    )
    command = RenderImageCommand(
        request_id="render",
        op="render.image",
        output_path="evidence.png",
        timeout_seconds=60,
    )

    assert client._command_read_timeout("review", command) == 1320


def test_execute_keeps_nested_discriminator_fields_at_their_default_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A field left at its declared default is safe to omit only at the TOP level of
    # the command — a naive recursive `exclude_defaults` would also strip a nested
    # submodel's own default-valued fields, including a discriminator field like
    # `Projection.mode` (defaults to "perspective"), leaving
    # `COMMAND_ADAPTER.validate_python()` unable to tell which union member to parse.
    client = MeshProbeClient(tmp_path)
    captured: dict[str, Any] = {}

    def fake_request(action: str, **arguments: object) -> dict[str, Any]:
        captured.update(arguments)
        return {"session": "review", "op": "view.orbit"}

    monkeypatch.setattr(client, "request", fake_request)

    client.execute(
        "review",
        ViewOrbitCommand(
            request_id="orbit",
            op="view.orbit",
            target_mm=(0.0, 0.0, 0.0),
            azimuth_degrees=0.0,
            elevation_degrees=0.0,
            distance_mm=1000.0,
            projection=PerspectiveProjection(),
        ),
    )

    assert captured["command"]["projection"]["mode"] == "perspective"
