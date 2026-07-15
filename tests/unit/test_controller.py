from __future__ import annotations

from pathlib import Path

import pytest

from meshprobe.controller import BlenderController, BlenderWorkerError, sha256_file


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
