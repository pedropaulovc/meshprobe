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


def test_sha256_file_reads_in_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    source.write_bytes(b"meshprobe")
    assert sha256_file(source) == "0ed04bc93d5b02521d55782c50a9bda9d342f820bf945afd2dc474e62919429c"


def test_missing_blender_executable_fails() -> None:
    with pytest.raises(BlenderWorkerError, match="not found"):
        BlenderController(executable="meshprobe-blender-does-not-exist")


def test_request_requires_started_worker() -> None:
    controller = BlenderController()
    with pytest.raises(BlenderWorkerError, match="not started"):
        controller.request("scene.describe")


def test_start_twice_fails() -> None:
    controller = BlenderController()
    try:
        controller.start()
        with pytest.raises(BlenderWorkerError, match="already started"):
            controller.start()
    finally:
        controller.close()


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
