from __future__ import annotations

import hashlib

from meshprobe.models import DisplayMode, SceneManifest
from meshprobe.session import InspectionSession


def test_session_operations_do_not_touch_source_file(
    tmp_path, scene_manifest: SceneManifest
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"immutable model bytes")
    before = hashlib.sha256(source.read_bytes()).digest()
    before_stat = source.stat()

    session = InspectionSession(scene_manifest)
    session.display([scene_manifest.components[-1].id], DisplayMode.ISOLATED)
    session.reset()

    assert hashlib.sha256(source.read_bytes()).digest() == before
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns
