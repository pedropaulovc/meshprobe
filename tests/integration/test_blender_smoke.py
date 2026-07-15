from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")
def test_blender_starts_factory_clean_in_background() -> None:
    completed = subprocess.run(
        [
            "blender",
            "--background",
            "--factory-startup",
            "--python-expr",
            "import bpy; print('MESHPROBE_BLENDER', bpy.app.version_string)",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "MESHPROBE_BLENDER" in completed.stdout
