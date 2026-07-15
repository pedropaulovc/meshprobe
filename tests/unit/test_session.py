from __future__ import annotations

import pytest

from meshprobe.models import DisplayMode, MarkMode, SceneManifest
from meshprobe.session import InspectionSession


def test_isolate_hides_every_other_component(scene_manifest: SceneManifest) -> None:
    session = InspectionSession(scene_manifest)
    target = scene_manifest.components[-1].id
    snapshot = session.display([target], DisplayMode.ISOLATED)

    assert snapshot.components[target].display is DisplayMode.ISOLATED
    assert all(
        state.display is DisplayMode.HIDDEN
        for component_id, state in snapshot.components.items()
        if component_id != target
    )


def test_reset_restores_imported_state_and_hash(scene_manifest: SceneManifest) -> None:
    session = InspectionSession(scene_manifest)
    initial = session.snapshot()
    target = scene_manifest.components[-1].id

    changed = session.mark([target], MarkMode.HIGHLIGHTED)
    assert changed.state_sha256 != initial.state_sha256
    assert session.reset() == initial


def test_empty_display_selection_fails(scene_manifest: SceneManifest) -> None:
    session = InspectionSession(scene_manifest)
    with pytest.raises(ValueError, match="at least one"):
        session.display([], DisplayMode.HIDDEN)


def test_unknown_display_selection_fails(scene_manifest: SceneManifest) -> None:
    session = InspectionSession(scene_manifest)
    with pytest.raises(ValueError, match="unknown component ids"):
        session.display(["not-present"], DisplayMode.HIDDEN)
