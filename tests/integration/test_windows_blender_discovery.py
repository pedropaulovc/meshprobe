"""Real (non-mocked) exercise of the Windows Blender auto-discovery path.

Every other Blender-discovery test lives in `tests/unit/test_blender_discovery.py`
and is pure/platform-independent (fabricated paths, no filesystem or registry
access). This test is the one place that touches the real filesystem, and it
only runs on Windows, where CI (`.github/workflows/ci.yml`, the `windows` job)
MSI-installs a real Blender 5.2 before the test suite runs -- so on Windows CI
this proves `discover_windows_blender` finds that install for real, not just in
a fabricated-path unit test.
"""

from __future__ import annotations

import os
import sys

import pytest

from meshprobe.blender_discovery import discover_windows_blender

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="exercises the real Windows Program Files scan / registry lookup",
)


def test_discover_windows_blender_finds_a_real_install(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scrub PATH so `shutil.which("blender")` would fail, mirroring issue #95 --
    # BlenderController.start() only calls discover_windows_blender() in
    # exactly that situation.
    monkeypatch.setenv("PATH", os.environ.get("SYSTEMROOT", r"C:\Windows"))
    discovered = discover_windows_blender()
    assert discovered is not None
    assert discovered.name.lower() == "blender.exe"
    assert discovered.is_file()
