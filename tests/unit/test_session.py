from __future__ import annotations

import pytest

from meshprobe.models import CameraPoseFrame, DisplayMode, MarkMode, SceneManifest
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


def test_custom_mark_color_is_hashed_and_cleared_when_unmarked(
    scene_manifest: SceneManifest,
) -> None:
    session = InspectionSession(scene_manifest)
    target = scene_manifest.components[-1].id

    colored = session.mark([target], MarkMode.HIGHLIGHTED, "#FF00ff")
    cleared = session.mark([target], MarkMode.UNMARKED)

    assert colored.components[target].mark_color == "#ff00ff"
    assert colored.state_sha256 != cleared.state_sha256
    assert cleared.components[target].mark_color is None


@pytest.mark.parametrize(
    ("mode", "color", "message"),
    [
        (MarkMode.HIGHLIGHTED, "magenta", "match pattern"),
        (MarkMode.UNMARKED, "#ff00ff", "requires a visible mark"),
    ],
)
def test_invalid_custom_mark_does_not_mutate_session(
    scene_manifest: SceneManifest,
    mode: MarkMode,
    color: str,
    message: str,
) -> None:
    session = InspectionSession(scene_manifest)
    target = scene_manifest.components[-1].id
    before = session.mark([target], MarkMode.SELECTED)

    with pytest.raises(ValueError, match=message):
        session.mark([target], mode, color)

    assert session.snapshot() == before


def test_source_frame_camera_does_not_mutate_in_memory_session(
    scene_manifest: SceneManifest,
) -> None:
    session = InspectionSession(scene_manifest)
    before = session.snapshot()
    source_camera = before.camera.model_copy(
        update={"pose": before.camera.pose.model_copy(update={"frame": CameraPoseFrame.SOURCE})}
    )

    with pytest.raises(ValueError, match="world-frame camera"):
        session.set_camera(source_camera)

    assert session.snapshot() == before


def test_empty_display_selection_fails(scene_manifest: SceneManifest) -> None:
    session = InspectionSession(scene_manifest)
    with pytest.raises(ValueError, match="at least one"):
        session.display([], DisplayMode.HIDDEN)


def test_unknown_display_selection_fails(scene_manifest: SceneManifest) -> None:
    session = InspectionSession(scene_manifest)
    with pytest.raises(ValueError, match="unknown component ids"):
        session.display(["not-present"], DisplayMode.HIDDEN)
