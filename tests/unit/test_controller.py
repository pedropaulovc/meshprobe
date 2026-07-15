from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from meshprobe.controller import (
    BlenderController,
    BlenderWorkerCrashed,
    BlenderWorkerError,
    BlenderWorkerTimeout,
    sha256_file,
)
from meshprobe.identity import stable_component_id
from meshprobe.models import MarkMode, SceneManifest
from meshprobe.protocol import (
    ComponentMarkCommand,
    SceneDescribeCommand,
    SceneOpenCommand,
    SessionResetCommand,
)
from meshprobe.session import InspectionSession, SessionSnapshot
from meshprobe.sources import snapshot_source


def make_fake_blender(tmp_path: Path, protocol_version: int = 1) -> Path:
    executable = tmp_path / "fake-blender"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

print(json.dumps({
    "event": "ready",
    "protocol_version": __PROTOCOL_VERSION__,
    "blender_version": "test",
    "pid": os.getpid(),
}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command["op"] == "protocol.error":
        print(json.dumps({
            "request_id": command["request_id"],
            "ok": False,
            "error": {"code": "worker.test", "message": "requested failure"},
        }), flush=True)
        continue
    if command["op"] == "protocol.scalar":
        print(json.dumps({
            "request_id": command["request_id"],
            "ok": True,
            "result": 1,
        }), flush=True)
        continue
    result = {"operation": command["op"]}
    print(json.dumps({
        "request_id": command["request_id"],
        "ok": True,
        "result": result,
    }), flush=True)
    if command["op"] == "session.shutdown":
        break
""".replace("__PROTOCOL_VERSION__", str(protocol_version)),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def manifest_for_hash(scene_manifest: SceneManifest, source_hash: str) -> SceneManifest:
    payload = scene_manifest.model_dump(mode="json")
    replacements = {
        component["id"]: stable_component_id(source_hash, component["path"])
        for component in payload["components"]
    }
    payload["source_sha256"] = source_hash
    for component in payload["components"]:
        component["id"] = replacements[component["id"]]
        if component["parent_id"] is not None:
            component["parent_id"] = replacements[component["parent_id"]]
        component["child_ids"] = [replacements[item] for item in component["child_ids"]]
    for warning in payload["warnings"]:
        warning["component_ids"] = [replacements[item] for item in warning["component_ids"]]
    return SceneManifest.model_validate(payload)


def test_sha256_file_reads_in_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    source.write_bytes(b"meshprobe")
    assert sha256_file(source) == "0ed04bc93d5b02521d55782c50a9bda9d342f820bf945afd2dc474e62919429c"


def test_missing_blender_executable_fails() -> None:
    controller = BlenderController(executable="meshprobe-blender-does-not-exist")
    with pytest.raises(BlenderWorkerError, match="not found"):
        controller.start()


def test_request_requires_started_worker() -> None:
    controller = BlenderController()
    with pytest.raises(BlenderWorkerError, match="not started"):
        controller.request("scene.describe")


def test_start_twice_fails(tmp_path: Path) -> None:
    controller = BlenderController(executable=make_fake_blender(tmp_path))
    try:
        controller.start()
        with pytest.raises(BlenderWorkerError, match="already started"):
            controller.start()
    finally:
        controller.close()


def test_start_failure_closes_half_started_process(tmp_path: Path) -> None:
    controller = BlenderController(executable=make_fake_blender(tmp_path, protocol_version=2))
    with pytest.raises(BlenderWorkerError, match="unsupported worker protocol"):
        controller.start()
    assert controller._process is None
    assert controller._reader_thread is None


def test_fake_worker_request_round_trip(tmp_path: Path) -> None:
    with BlenderController(executable=make_fake_blender(tmp_path)) as controller:
        result = controller.request("protocol.ping")
    assert result == {"operation": "protocol.ping"}


def test_worker_error_envelope_is_raised(tmp_path: Path) -> None:
    with (
        BlenderController(executable=make_fake_blender(tmp_path)) as controller,
        pytest.raises(BlenderWorkerError, match=r"worker\.test: requested failure"),
    ):
        controller.request("protocol.error")


def test_worker_nonobject_result_is_rejected(tmp_path: Path) -> None:
    with (
        BlenderController(executable=make_fake_blender(tmp_path)) as controller,
        pytest.raises(BlenderWorkerError, match="non-object"),
    ):
        controller.request("protocol.scalar")


def test_open_scene_validates_hash_and_manifest(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model bytes")
    source_hash = sha256_file(source)
    expected = manifest_for_hash(scene_manifest, source_hash)
    controller = BlenderController()
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: expected.model_dump(mode="json"),
    )

    assert controller.open_scene(source) == expected


def test_open_scene_detects_source_mutation(tmp_path: Path, scene_manifest, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model bytes")
    controller = BlenderController()

    def mutate_source(operation, **arguments):  # type: ignore[no-untyped-def]
        source.write_bytes(b"changed model bytes")
        return scene_manifest.model_dump(mode="json")

    monkeypatch.setattr(controller, "request", mutate_source)
    with pytest.raises(BlenderWorkerError, match="source asset bundle changed"):
        controller.open_scene(source)


def test_gltf_bundle_hash_and_immutability_cover_external_assets(
    tmp_path: Path, scene_manifest: SceneManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = tmp_path / "mesh data.bin"
    buffer.write_bytes(b"original buffer")
    source = tmp_path / "fixture.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "mesh%20data.bin", "byteLength": buffer.stat().st_size}],
            }
        ),
        encoding="utf-8",
    )
    original = snapshot_source(source)
    buffer.write_bytes(b"replacement buf")
    changed = snapshot_source(source)
    assert original.sha256 != changed.sha256

    buffer.write_bytes(b"original buffer")
    controller = BlenderController()

    def mutate_dependency(operation: str, **arguments: object) -> dict[str, object]:
        buffer.write_bytes(b"changed by worker")
        source_hash = cast(str, arguments["source_sha256"])
        return manifest_for_hash(scene_manifest, source_hash).model_dump(mode="json")

    monkeypatch.setattr(controller, "request", mutate_dependency)
    with pytest.raises(BlenderWorkerError, match="source asset bundle changed"):
        controller.open_scene(source)


def test_open_scene_rejects_worker_hash(tmp_path: Path, scene_manifest, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model bytes")
    controller = BlenderController()
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: scene_manifest.model_dump(mode="json"),
    )
    with pytest.raises(BlenderWorkerError, match="worker source hash"):
        controller.open_scene(source)


def test_logs_are_immutable_and_close_is_idempotent() -> None:
    controller = BlenderController()
    controller._logs.append("Blender banner")
    assert controller.logs == ("Blender banner",)
    controller.close()
    controller.close()


def test_wait_for_ignores_noise_and_unmatched_json() -> None:
    controller = BlenderController(timeout_seconds=1)
    controller._lines.put("Blender banner")
    controller._lines.put('{"event":"progress"}')
    controller._lines.put('{"event":"ready","protocol_version":1}')

    result = controller._wait_for(lambda payload: payload.get("event") == "ready")

    assert result["protocol_version"] == 1
    assert controller.logs == ("Blender banner", '{"event":"progress"}')


def test_wait_for_reports_process_end() -> None:
    controller = BlenderController(timeout_seconds=1)
    controller._lines.put(None)
    with pytest.raises(BlenderWorkerCrashed, match="exited with code"):
        controller._wait_for(lambda payload: False)


def test_wait_for_times_out() -> None:
    controller = BlenderController(timeout_seconds=0.001)
    with pytest.raises(BlenderWorkerTimeout, match="did not respond"):
        controller._wait_for(lambda payload: False)


def test_output_reader_is_bound_to_its_worker_generation() -> None:
    controller = BlenderController()
    old_queue: queue.Queue[str | None] = queue.Queue()
    current_queue: queue.Queue[str | None] = queue.Queue()
    controller._lines = current_queue
    old_process = SimpleNamespace(stdout=iter(["old worker\n"]))

    controller._read_output(cast(Any, old_process), old_queue)

    assert old_queue.get_nowait() == "old worker"
    assert old_queue.get_nowait() is None
    assert current_queue.empty()


def test_execute_records_state_and_compacts_reset(scene_manifest, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    snapshot = InspectionSession(scene_manifest).snapshot()
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: snapshot.model_dump(mode="json"),
    )
    target = scene_manifest.components[-1].id

    marked = controller.execute(
        ComponentMarkCommand(
            request_id="mark",
            op="component.mark",
            component_ids=(target,),
            mode=MarkMode.HIGHLIGHTED,
        )
    )
    assert isinstance(marked, SessionSnapshot)
    assert len(controller._accepted_commands) == 1

    reset = controller.execute(SessionResetCommand(request_id="reset", op="session.reset"))
    assert isinstance(reset, SessionSnapshot)
    assert controller._accepted_commands == []


def test_execute_returns_nonstate_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: {"operation": operation},
    )
    result = controller.execute(SceneDescribeCommand(request_id="describe", op="scene.describe"))
    assert result == {"operation": "scene.describe"}


def test_execute_routes_scene_open_through_checked_import(
    tmp_path: Path, scene_manifest: SceneManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = BlenderController()
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    opened: list[Path] = []

    def open_scene(path: str | Path) -> SceneManifest:
        opened.append(Path(path))
        return scene_manifest

    monkeypatch.setattr(controller, "open_scene", open_scene)
    result = controller.execute(
        SceneOpenCommand(request_id="open", op="scene.open", source_path=str(source))
    )

    assert result is scene_manifest
    assert opened == [source]


def test_execute_recovers_once_after_crash(scene_manifest, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    snapshot = InspectionSession(scene_manifest).snapshot()
    calls = 0
    recovered = False

    def request(operation, **arguments):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BlenderWorkerCrashed("test crash")
        return snapshot.model_dump(mode="json")

    def recover() -> None:
        nonlocal recovered
        recovered = True

    monkeypatch.setattr(controller, "request", request)
    monkeypatch.setattr(controller, "_recover_session", recover)
    target = scene_manifest.components[-1].id
    result = controller.execute(
        ComponentMarkCommand(
            request_id="mark",
            op="component.mark",
            component_ids=(target,),
            mode=MarkMode.SELECTED,
        )
    )
    assert isinstance(result, SessionSnapshot)
    assert recovered
    assert calls == 2


def test_recover_session_replays_commands_in_order(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    controller._source_path = source
    controller._source_sha256 = "source-hash"
    controller._accepted_commands = [
        ("component.mark", {"component_ids": ["one"], "mode": "selected"}),
        ("component.display", {"component_ids": ["one"], "mode": "hidden"}),
    ]
    calls: list[object] = []
    monkeypatch.setattr(controller, "close", lambda: calls.append("close"))
    monkeypatch.setattr(controller, "start", lambda: calls.append("start"))
    monkeypatch.setattr(
        controller,
        "open_scene",
        lambda path: calls.append(("open", path)) or SimpleNamespace(source_sha256="source-hash"),
    )
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: calls.append((operation, arguments)),
    )

    controller._recover_session()

    assert calls == [
        "close",
        "start",
        ("open", source),
        ("component.mark", {"component_ids": ["one"], "mode": "selected"}),
        ("component.display", {"component_ids": ["one"], "mode": "hidden"}),
    ]
    assert len(controller._accepted_commands) == 2


def test_recover_session_requires_open_scene() -> None:
    controller = BlenderController()
    with pytest.raises(BlenderWorkerCrashed, match="before a scene is open"):
        controller._recover_session()


def test_recover_session_rejects_replaced_source(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    controller._source_path = source
    controller._source_sha256 = "original-hash"
    calls: list[str] = []
    monkeypatch.setattr(controller, "close", lambda: calls.append("close"))
    monkeypatch.setattr(controller, "start", lambda: calls.append("start"))
    monkeypatch.setattr(
        controller,
        "open_scene",
        lambda path: SimpleNamespace(source_sha256="replacement-hash"),
    )
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: pytest.fail("state must not replay"),
    )

    with pytest.raises(BlenderWorkerError, match="changed since the session opened"):
        controller._recover_session()

    assert calls == ["close", "start", "close"]
    assert controller._source_path is None
    assert controller._source_sha256 is None
