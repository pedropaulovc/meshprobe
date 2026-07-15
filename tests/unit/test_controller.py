from __future__ import annotations

from pathlib import Path

import pytest

from meshprobe.controller import (
    BlenderController,
    BlenderWorkerCrashed,
    BlenderWorkerError,
    BlenderWorkerTimeout,
    sha256_file,
)


def make_fake_blender(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-blender"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

print(json.dumps({
    "event": "ready",
    "protocol_version": 1,
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
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


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
    tmp_path: Path, scene_manifest, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model bytes")
    source_hash = sha256_file(source)
    expected = scene_manifest.model_copy(update={"source_sha256": source_hash})
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
    with pytest.raises(BlenderWorkerError, match="source file changed"):
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
