from __future__ import annotations

import json
import math
import os
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from meshprobe.camera import camera_diagnostics, orbit_camera
from meshprobe.controller import (
    BlenderController,
    BlenderWorkerCrashed,
    BlenderWorkerError,
    BlenderWorkerTimeout,
    sha256_file,
)
from meshprobe.identity import stable_component_id
from meshprobe.models import (
    Bounds,
    CameraMotionResult,
    CameraRotationReceipt,
    CameraTranslationReceipt,
    CameraViewResult,
    ComponentVisualStateResult,
    CoordinateFrame,
    CustomIllumination,
    DisplayMode,
    EnvironmentMap,
    IlluminationResult,
    MarkMode,
    OccluderRemovalStep,
    OcclusionQueryResult,
    OrthographicProjection,
    OrthonormalBasis,
    PerspectiveProjection,
    PresetIllumination,
    RenderManifest,
    SceneManifest,
    SessionSnapshot,
)
from meshprobe.protocol import (
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentInspectCommand,
    ComponentMarkCommand,
    ComponentOcclusionCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionResetCommand,
    SessionSnapshotCommand,
    ViewFrameCommand,
    ViewMoveCommand,
    ViewRotateCommand,
    ViewSetCommand,
)
from meshprobe.selectors import ComponentSelector, SelectorKind
from meshprobe.session import InspectionSession
from meshprobe.sources import snapshot_source


def make_fake_blender(tmp_path: Path, protocol_version: int = 2) -> Path:
    script = tmp_path / "fake_blender_worker.py"
    script.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

print(json.dumps({
    "event": "ready",
    "protocol_version": __PROTOCOL_VERSION__,
    "blender_version": "test",
    "pid": os.getpid(),
    "graphics": {
        "vendor": "Test",
        "renderer": "Test GPU",
        "version": "1.0",
        "backend": "OPENGL",
        "blender_device_type": "NVIDIA",
        "device_class": "hardware",
        "warnings": [],
    },
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
    if command["op"] == "protocol.truncated":
        sys.stdout.write('{"request_id":"' + command["request_id"])
        sys.stdout.flush()
        os._exit(23)
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
    if os.name == "nt":
        executable = tmp_path / "fake-blender.cmd"
        executable.write_text(
            f'@"{sys.executable}" "{script}"\r\n',
            encoding="utf-8",
        )
        return executable
    executable = tmp_path / "fake-blender"
    script.replace(executable)
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
        controller.request("session.snapshot")


def test_start_twice_fails(tmp_path: Path) -> None:
    controller = BlenderController(executable=make_fake_blender(tmp_path))
    try:
        controller.start()
        with pytest.raises(BlenderWorkerError, match="already started"):
            controller.start()
    finally:
        controller.close()


def test_start_failure_closes_half_started_process(tmp_path: Path) -> None:
    controller = BlenderController(executable=make_fake_blender(tmp_path, protocol_version=1))
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


def test_truncated_worker_response_fails_as_a_crash(tmp_path: Path) -> None:
    with (
        BlenderController(executable=make_fake_blender(tmp_path)) as controller,
        pytest.raises(BlenderWorkerCrashed, match="exited with code 23"),
    ):
        controller.request("protocol.truncated")
    assert any(line.startswith('{"request_id":') for line in controller.logs)


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

    assert controller.open_scene(source, aspect_ratio=2.5) == expected
    # Stored so a later crash recovery reopens with the same non-square framing.
    assert controller._aspect_ratio == 2.5


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
    controller._lines.put('{"event":"ready","protocol_version":2}')

    result = controller._wait_for(lambda payload: payload.get("event") == "ready")

    assert result["protocol_version"] == 2
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
    calls: list[tuple[str, dict[str, object]]] = []

    def request(operation: str, **arguments: object) -> dict[str, object]:
        calls.append((operation, arguments))
        return snapshot.model_dump(mode="json")

    monkeypatch.setattr(
        controller,
        "request",
        request,
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
    assert marked == {
        "components": {target: snapshot.components[target].model_dump(mode="json")},
        "state_sha256": snapshot.state_sha256,
    }
    assert len(controller._accepted_commands) == 1

    reset = controller.execute(SessionResetCommand(request_id="reset", op="session.reset"))
    assert reset == {"reset": True, "state_sha256": snapshot.state_sha256}
    assert controller._accepted_commands == []
    assert calls[-1] == ("session.reset", {})

    controller.execute(
        SessionResetCommand(request_id="reset", op="session.reset", aspect_ratio=2.5)
    )
    assert calls[-1] == ("session.reset", {"aspect_ratio": 2.5})
    assert controller._aspect_ratio == 2.5


def test_execute_returns_operation_local_state(scene_manifest, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    snapshot = InspectionSession(scene_manifest).snapshot()
    move_receipt = CameraTranslationReceipt(
        requested_world_delta_mm=(0, 0, 0),
        requested_camera_delta_mm=(1, 2, 3),
        resolved_world_delta_mm=(1, 2, 3),
        previous_pose=snapshot.camera.pose,
        resulting_pose=snapshot.camera.pose,
    )
    moved_snapshot = snapshot.model_copy(update={"camera_operation": move_receipt})
    rotation_receipt = CameraRotationReceipt(
        frame=CoordinateFrame.WORLD,
        target_mm=(0, 0, 0),
        axis="z",
        basis=OrthonormalBasis(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1)),
        axis_world=(0, 0, 1),
        requested_visual_degrees=90,
        applied_camera_orbit_degrees=-90,
        previous_pose=snapshot.camera.pose,
        resulting_pose=snapshot.camera.pose,
    )
    rotated_snapshot = snapshot.model_copy(update={"camera_operation": rotation_receipt})

    def request(operation: str, **arguments: object) -> dict[str, object]:
        resolved = {
            "view.move": moved_snapshot,
            "view.rotate": rotated_snapshot,
        }.get(operation, snapshot)
        return resolved.model_dump(mode="json")

    monkeypatch.setattr(
        controller,
        "request",
        request,
    )
    target = scene_manifest.components[-1].id

    view = controller.execute(
        ViewSetCommand(request_id="view", op="view.set", camera=snapshot.camera)
    )
    moved = controller.execute(
        ViewMoveCommand(
            request_id="move",
            op="view.move",
            camera_delta_mm=(1, 2, 3),
        )
    )
    rotated = controller.execute(
        ViewRotateCommand(
            request_id="rotate",
            op="view.rotate",
            target_mm=(0, 0, 0),
            axis="z",
            degrees=90,
            frame=CoordinateFrame.WORLD,
        )
    )
    illumination = controller.execute(
        IlluminationSetCommand(
            request_id="light",
            op="illumination.set",
            illumination=snapshot.illumination,
        )
    )
    isolated = controller.execute(
        ComponentDisplayCommand(
            request_id="isolate",
            op="component.display",
            component_ids=(target,),
            mode=DisplayMode.ISOLATED,
        )
    )

    assert view == {
        "camera": snapshot.camera.model_dump(mode="json"),
        "camera_diagnostics": snapshot.camera_diagnostics.model_dump(mode="json"),
        "state_sha256": snapshot.state_sha256,
    }
    assert moved == {**view, "camera_operation": move_receipt.model_dump(mode="json")}
    assert rotated == {**view, "camera_operation": rotation_receipt.model_dump(mode="json")}
    assert illumination == {
        "illumination": snapshot.illumination.model_dump(mode="json"),
        "state_sha256": snapshot.state_sha256,
    }
    assert isolated == {
        "components": {
            component_id: state.model_dump(mode="json")
            for component_id, state in snapshot.components.items()
        },
        "state_sha256": snapshot.state_sha256,
    }

    # The published result envelopes match the documented result contracts (schema lockstep).
    CameraViewResult.model_validate(view)
    CameraMotionResult.model_validate(moved)
    CameraMotionResult.model_validate(rotated)
    IlluminationResult.model_validate(illumination)
    ComponentVisualStateResult.model_validate(isolated)


def test_execute_returns_nonstate_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: {"operation": operation},
    )
    result = controller.execute(
        SessionSnapshotCommand(request_id="describe", op="session.snapshot")
    )
    assert result == {"operation": "session.snapshot"}


def test_execute_resolves_component_queries_without_worker(
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = BlenderController()
    controller._manifest = scene_manifest
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: pytest.fail("component lookup reached worker"),
    )
    target = scene_manifest.components[-1]

    found = controller.execute(
        ComponentFindCommand(
            request_id="find",
            op="component.find",
            selector=ComponentSelector(kind=SelectorKind.EXACT_NAME, pattern=target.display_name),
        )
    )
    inspected = controller.execute(
        ComponentInspectCommand(
            request_id="inspect",
            op="component.inspect",
            component_id=target.id,
        )
    )

    assert found == [target.model_dump(mode="json")]
    assert inspected == target.model_dump(mode="json")


def test_execute_rejects_component_query_before_open() -> None:
    command = ComponentInspectCommand(
        request_id="inspect",
        op="component.inspect",
        component_id="missing",
    )
    with pytest.raises(BlenderWorkerError, match="before a scene is open"):
        BlenderController().execute(command)


def test_removed_occluder_caption_retains_every_full_blocker_name() -> None:
    names = tuple(f"blocker-{index}-" + "x" * 230 for index in range(32))
    steps = tuple(
        OccluderRemovalStep(
            component_id=f"blocker-{index}",
            display_name=name,
            component_path=f"/blocker-{index}",
            ray_hit_count=1,
            visible_pixel_count_after=index,
            visible_fraction_after=index / 32,
        )
        for index, name in enumerate(names)
    )

    caption = BlenderController._removed_occluder_caption(steps)

    assert len(caption) > 256
    assert all(name in caption for name in names)


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("threshold_met_initial", "visibility threshold already met"),
        ("threshold_met", "visibility threshold met"),
        ("budget_exhausted", "no occluders removed within budget"),
        ("no_blockers", "no blocking components ranked"),
        ("focus_not_projected", "focus not projected"),
    ],
)
def test_empty_occlusion_caption_reports_stop_reason(
    stop_reason: str,
    expected: str,
) -> None:
    caption = BlenderController._empty_occlusion_caption(stop_reason)  # type: ignore[arg-type]

    assert expected in caption
    assert "previous focused view" in caption
    assert "no occluders found" not in caption


def test_focused_context_camera_stays_outside_scene_and_preserves_framing() -> None:
    bounds = Bounds(
        minimum_mm=(-1_000.0, -1_000.0, -1_000.0),
        maximum_mm=(1_000.0, 1_000.0, 1_000.0),
    )
    focus_center = (0.0, 0.0, 0.0)
    projection, distance = BlenderController._focused_context_camera(
        bounds,
        focus_center,
        focus_span=1.0,
        aspect_ratio=1.0,
    )
    context_radius = math.sqrt(3 * 1_000.0**2)
    framing_fov = min(
        projection.horizontal_fov_degrees(1.0),
        projection.vertical_fov_degrees(1.0),
    )
    framed_distance = (4.0 / 2) / math.tan(math.radians(framing_fov / 2)) * 1.25

    assert distance > context_radius
    assert framed_distance == pytest.approx(distance)
    assert projection.focal_length_mm > 50
    assert projection.far_clip_mm > distance + context_radius


def test_frame_view_orbits_onto_focus_component_bounds(scene_manifest, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    controller._manifest = scene_manifest
    snapshot = InspectionSession(scene_manifest).snapshot()
    captured: dict[str, Any] = {}

    def request(operation: str, **arguments: object) -> dict[str, object]:
        captured["operation"] = operation
        captured["arguments"] = arguments
        return snapshot.model_dump(mode="json")

    monkeypatch.setattr(controller, "request", request)
    target = scene_manifest.components[-1]

    result = controller.execute(
        ViewFrameCommand(request_id="frame", op="view.frame", focus_component_ids=(target.id,))
    )

    assert captured["operation"] == "view.orbit"
    arguments = captured["arguments"]
    assert arguments["target_mm"] == [0.0, 0.0, 0.0]
    assert arguments["focus_component_ids"] == [target.id]
    assert cast(float, arguments["distance_mm"]) > 0.0
    assert result == {
        "camera": snapshot.camera.model_dump(mode="json"),
        "camera_diagnostics": snapshot.camera_diagnostics.model_dump(mode="json"),
        "state_sha256": snapshot.state_sha256,
    }
    assert [operation for operation, _ in controller._accepted_commands] == ["view.orbit"]


def test_frame_view_rejects_unknown_focus_component(scene_manifest) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    controller._manifest = scene_manifest
    with pytest.raises(BlenderWorkerError, match="unknown component ids"):
        controller.frame_view(
            ViewFrameCommand(request_id="frame", op="view.frame", focus_component_ids=("missing",))
        )


def test_frame_view_requires_open_scene() -> None:
    controller = BlenderController()
    with pytest.raises(BlenderWorkerError, match="cannot frame the camera"):
        controller.frame_view(
            ViewFrameCommand(request_id="frame", op="view.frame", focus_component_ids=("c1",))
        )


def test_frame_camera_fits_perspective_and_orthographic_projections() -> None:
    bounds = Bounds(minimum_mm=(-20.0, -20.0, -20.0), maximum_mm=(20.0, 20.0, 20.0))
    target = (0.0, 0.0, 0.0)
    perspective, distance = BlenderController._frame_camera(
        PerspectiveProjection(),
        bounds,
        target,
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        aspect_ratio=1.0,
        margin=1.25,
    )
    assert isinstance(perspective, PerspectiveProjection)
    assert distance > perspective.near_clip_mm

    orthographic, ortho_distance = BlenderController._frame_camera(
        OrthographicProjection(scale_mm=1.0),
        bounds,
        target,
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        aspect_ratio=1.0,
        margin=1.5,
    )
    assert isinstance(orthographic, OrthographicProjection)
    span = math.sqrt(3 * 40.0**2)
    assert orthographic.scale_mm == pytest.approx(span * 1.5)
    assert ortho_distance == pytest.approx(span)


def test_frame_camera_orthographic_scale_compensates_for_portrait_aspect() -> None:
    bounds = Bounds(minimum_mm=(-20.0, -20.0, -20.0), maximum_mm=(20.0, 20.0, 20.0))
    target = (0.0, 0.0, 0.0)
    span = math.sqrt(3 * 40.0**2)
    landscape, _ = BlenderController._frame_camera(
        OrthographicProjection(scale_mm=1.0),
        bounds,
        target,
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        aspect_ratio=1.6,
        margin=1.2,
    )
    portrait, _ = BlenderController._frame_camera(
        OrthographicProjection(scale_mm=1.0),
        bounds,
        target,
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        aspect_ratio=0.5,
        margin=1.2,
    )
    assert isinstance(landscape, OrthographicProjection)
    assert isinstance(portrait, OrthographicProjection)
    # Landscape/square frames the vertical extent directly; portrait must enlarge scale_mm so
    # the horizontal extent (scale_mm * aspect_ratio) still covers the bounds.
    assert landscape.scale_mm == pytest.approx(span * 1.2)
    assert portrait.scale_mm == pytest.approx(span * 1.2 / 0.5)
    assert portrait.scale_mm * 0.5 == pytest.approx(span * 1.2)


def test_frame_camera_keeps_wide_fov_camera_outside_bounds() -> None:
    bounds = Bounds(minimum_mm=(-200.0, -200.0, -200.0), maximum_mm=(200.0, 200.0, 200.0))
    projection, distance = BlenderController._frame_camera(
        PerspectiveProjection(focal_length_mm=1.0),
        bounds,
        (0.0, 0.0, 0.0),
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        aspect_ratio=1.0,
        margin=1.25,
    )
    assert isinstance(projection, PerspectiveProjection)
    # A ~1mm focal length yields an extreme FOV whose framing distance collapses well inside
    # the bounds. The camera must nevertheless stay beyond its closest face and preserve clips.
    assert distance >= 200.0 + projection.near_clip_mm
    assert projection.far_clip_mm > distance + 200.0


def test_frame_camera_pushes_past_a_large_near_clip() -> None:
    bounds = Bounds(minimum_mm=(-10.0, -10.0, -10.0), maximum_mm=(10.0, 10.0, 10.0))
    near_clip = 1_000.0
    projection, distance = BlenderController._frame_camera(
        PerspectiveProjection(near_clip_mm=near_clip, far_clip_mm=200_000.0),
        bounds,
        (0.0, 0.0, 0.0),
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        aspect_ratio=1.0,
        margin=1.25,
    )
    # The whole bounds must stay behind the caller's near plane and before the far plane.
    assert distance >= near_clip + 10.0
    assert distance - 10.0 >= projection.near_clip_mm
    assert projection.far_clip_mm >= distance + 10.0


def test_frame_camera_corner_projection_tightly_fits_depth_elongated_bounds() -> None:
    # Looking along the long X axis, only the 212 mm Y and 628 mm Z silhouette should
    # determine the fit. The old bounding-sphere calculation also charged the full 628 mm
    # depth, leaving roughly a fifth of the limiting axis unused.
    bounds = Bounds(
        minimum_mm=(-314.0, -106.0, -314.0),
        maximum_mm=(314.0, 106.0, 314.0),
    )
    margin = 1.25
    projection, distance = BlenderController._frame_camera(
        PerspectiveProjection(focal_length_mm=50.0),
        bounds,
        (0.0, 0.0, 0.0),
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        aspect_ratio=1.0,
        margin=margin,
    )
    assert isinstance(projection, PerspectiveProjection)
    camera = orbit_camera(
        target_mm=(0.0, 0.0, 0.0),
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        roll_degrees=0.0,
        distance_mm=distance,
        projection=projection,
    )
    diagnostics = camera_diagnostics(camera, target_mm=(0.0, 0.0, 0.0), aspect_ratio=1.0)
    horizontal_tan = math.tan(math.radians(cast(float, diagnostics.horizontal_fov_degrees) / 2))
    vertical_tan = math.tan(math.radians(cast(float, diagnostics.vertical_fov_degrees) / 2))
    normalized_extents: list[float] = []
    for corner in BlenderController._bounds_corners(bounds):
        relative = cast(
            tuple[float, float, float],
            tuple(corner[axis] - camera.pose.position_mm[axis] for axis in range(3)),
        )
        depth = BlenderController._dot(relative, diagnostics.forward)
        horizontal = abs(BlenderController._dot(relative, diagnostics.right)) / (
            depth * horizontal_tan
        )
        vertical = abs(BlenderController._dot(relative, diagnostics.up)) / (depth * vertical_tan)
        normalized_extents.extend((horizontal, vertical))
        assert depth >= projection.near_clip_mm
    assert max(normalized_extents) == pytest.approx(1 / margin)

    legacy_span = math.sqrt(628.0**2 + 212.0**2 + 628.0**2)
    legacy_distance = (
        legacy_span / 2 / math.sin(math.radians(projection.vertical_fov_degrees(1.0) / 2)) * margin
    )
    legacy_fill = 314.0 / ((legacy_distance - 314.0) * vertical_tan)
    assert legacy_fill < 0.7


def test_occlusion_query_reports_current_camera_samples_and_named_blockers(
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = BlenderController()
    controller._manifest = scene_manifest
    snapshot = InspectionSession(scene_manifest).snapshot()
    focus = scene_manifest.components[-1]
    blocker = scene_manifest.components[-2]
    query_diagnostics = snapshot.camera_diagnostics.model_copy(
        update={"target_depth_mm": snapshot.camera_diagnostics.target_depth_mm + 100}
    )

    def request(operation: str, **arguments: object) -> dict[str, Any]:
        if operation == "session.snapshot":
            return {"session": snapshot.model_dump(mode="json")}
        if operation == "component.occluders":
            assert arguments["component_ids"] == [focus.id]
            assert arguments["aspect_ratio"] == snapshot.camera_diagnostics.aspect_ratio
            return {
                "projection": snapshot.camera.projection.mode,
                "aspect_ratio": snapshot.camera_diagnostics.aspect_ratio,
                "sample_count": 9,
                "visible_rays": 5,
                "occluded_rays": 4,
                "unresolved_rays": 0,
                "camera_diagnostics": query_diagnostics.model_dump(mode="json"),
                "occluders": [{"component_id": blocker.id, "blocked_rays": 4}],
            }
        raise AssertionError(f"unexpected operation: {operation}")

    monkeypatch.setattr(controller, "request", request)
    result = controller.execute(
        ComponentOcclusionCommand(
            request_id="occlusion",
            op="component.occlusion",
            component_ids=(focus.id,),
            max_samples_per_component=64,
        )
    )

    assert isinstance(result, OcclusionQueryResult)
    assert result.camera == snapshot.camera
    assert result.camera_diagnostics == query_diagnostics
    assert result.aspect_ratio == snapshot.camera_diagnostics.aspect_ratio
    assert result.camera_source == "current_session"
    assert result.state_sha256 == snapshot.state_sha256
    assert result.visible_sample_count == 5
    assert result.occluded_sample_count == 4
    assert result.unresolved_sample_count == 0
    assert result.visible_fraction == pytest.approx(5 / 9)
    assert result.sample_count == 9
    assert result.blockers[0].component_id == blocker.id
    assert result.blockers[0].display_name == blocker.display_name
    assert result.blockers[0].component_path == blocker.path


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"unresolved_rays": 1}, "inconsistent occlusion sample counts"),
        ({"visible_rays": -1}, "invalid occlusion sample counts"),
        ({"unresolved_rays": None}, "invalid occlusion sample counts"),
        ({"occluders": []}, "blocker hit counts do not equal occluded rays"),
        (
            {"occluders": [{"blocked_rays": 0}]},
            "invalid blocker hit count",
        ),
    ],
)
def test_occlusion_query_rejects_malformed_worker_accounting(
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    message: str,
) -> None:
    controller = BlenderController()
    controller._manifest = scene_manifest
    snapshot = InspectionSession(scene_manifest).snapshot()
    focus = scene_manifest.components[-1]
    blocker = scene_manifest.components[-2]
    ranking: dict[str, object] = {
        "projection": snapshot.camera.projection.mode,
        "aspect_ratio": snapshot.camera_diagnostics.aspect_ratio,
        "sample_count": 9,
        "visible_rays": 5,
        "occluded_rays": 4,
        "unresolved_rays": 0,
        "camera_diagnostics": snapshot.camera_diagnostics.model_dump(mode="json"),
        "occluders": [{"component_id": blocker.id, "blocked_rays": 4}],
    }
    ranking.update(updates)
    occluders = ranking.get("occluders", [])
    assert isinstance(occluders, list)
    for item in occluders:
        assert isinstance(item, dict)
        item.setdefault("component_id", blocker.id)

    def request(operation: str, **arguments: object) -> dict[str, Any]:
        if operation == "session.snapshot":
            return {"session": snapshot.model_dump(mode="json")}
        if operation == "component.occluders":
            return ranking
        raise AssertionError(f"unexpected operation: {operation}")

    monkeypatch.setattr(controller, "request", request)
    command = ComponentOcclusionCommand(
        request_id="malformed-occlusion",
        op="component.occlusion",
        component_ids=(focus.id,),
    )

    with pytest.raises(BlenderWorkerError, match=message):
        controller.execute(command)


def test_occlusion_query_validates_open_scene_and_focus(scene_manifest: SceneManifest) -> None:
    command = ComponentOcclusionCommand(
        request_id="occlusion",
        op="component.occlusion",
        component_ids=("missing",),
    )

    with pytest.raises(BlenderWorkerError, match="before a scene is open"):
        BlenderController().execute(command)
    controller = BlenderController()
    controller._manifest = scene_manifest
    with pytest.raises(BlenderWorkerError, match="unknown component"):
        controller.execute(command)


def test_execute_routes_scene_open_through_checked_import(
    tmp_path: Path, scene_manifest: SceneManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = BlenderController()
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    opened: list[tuple[Path, float]] = []

    def open_scene(
        path: str | Path, *, aspect_ratio: float = 1.0, unit_scale: float = 1.0
    ) -> SceneManifest:
        assert unit_scale == 1.0
        opened.append((Path(path), aspect_ratio))
        return scene_manifest

    monkeypatch.setattr(controller, "open_scene", open_scene)
    result = controller.execute(
        SceneOpenCommand(
            request_id="open", op="scene.open", source_path=str(source), aspect_ratio=2.5
        )
    )

    assert result is scene_manifest
    assert opened == [(source, 2.5)]


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
    assert result == {
        "components": {target: snapshot.components[target].model_dump(mode="json")},
        "state_sha256": snapshot.state_sha256,
    }
    assert recovered
    assert calls == 2


def test_recover_session_replays_commands_in_order(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    controller = BlenderController()
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    controller._source_path = source
    controller._source_sha256 = "source-hash"
    controller._aspect_ratio = 2.5
    controller._accepted_commands = [
        ("component.mark", {"component_ids": ["one"], "mode": "selected"}),
        ("component.display", {"component_ids": ["one"], "mode": "hidden"}),
    ]
    calls: list[object] = []
    monkeypatch.setattr(controller, "close", lambda: calls.append("close"))
    monkeypatch.setattr(controller, "start", lambda: calls.append("start"))

    def open_scene(
        path: Path, *, aspect_ratio: float = 1.0, unit_scale: float = 1.0
    ) -> SimpleNamespace:
        assert unit_scale == 1.0
        calls.append(("open", path, aspect_ratio))
        return SimpleNamespace(source_sha256="source-hash")

    monkeypatch.setattr(controller, "open_scene", open_scene)
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: calls.append((operation, arguments)),
    )

    controller._recover_session()

    assert calls == [
        "close",
        "start",
        ("open", source, 2.5),
        ("component.mark", {"component_ids": ["one"], "mode": "selected"}),
        ("component.display", {"component_ids": ["one"], "mode": "hidden"}),
    ]
    assert len(controller._accepted_commands) == 2


def test_recover_session_requires_open_scene() -> None:
    controller = BlenderController()
    with pytest.raises(BlenderWorkerCrashed, match="before a scene is open"):
        controller._recover_session()


def test_recover_session_retains_replay_log_when_replay_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = BlenderController()
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    controller._source_path = source
    controller._source_sha256 = "source-hash"
    expected: list[tuple[str, dict[str, object]]] = [
        ("component.mark", {"component_ids": ["one"], "mode": "selected"})
    ]
    controller._accepted_commands = list(expected)
    monkeypatch.setattr(controller, "close", lambda: None)
    monkeypatch.setattr(controller, "start", lambda: {})

    def reopen(
        path: Path, *, aspect_ratio: float = 1.0, unit_scale: float = 1.0
    ) -> SimpleNamespace:
        del path, aspect_ratio, unit_scale
        controller._accepted_commands.clear()
        return SimpleNamespace(source_sha256="source-hash")

    def crash(operation: str, **arguments: object) -> dict[str, object]:
        raise BlenderWorkerCrashed("replay crashed")

    monkeypatch.setattr(controller, "open_scene", reopen)
    monkeypatch.setattr(controller, "request", crash)

    with pytest.raises(BlenderWorkerCrashed, match="replay crashed"):
        controller._recover_session()
    assert controller._accepted_commands == expected


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
        lambda path, *, aspect_ratio=1.0, unit_scale=1.0: SimpleNamespace(
            source_sha256="replacement-hash"
        ),
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


def render_manifest_for(
    path: Path,
    source_hash: str,
    snapshot: SessionSnapshot | None = None,
) -> RenderManifest:
    session: dict[str, object] = (
        snapshot.model_dump(mode="json")
        if snapshot is not None
        else {
            "camera": {
                "pose": {
                    "position_mm": [100, 100, 100],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "projection": {"mode": "perspective", "focal_length_mm": 50},
            },
            "camera_diagnostics": {
                "aspect_ratio": 1,
                "horizontal_fov_degrees": 40,
                "vertical_fov_degrees": 40,
                "right": [1, 0, 0],
                "up": [0, 1, 0],
                "forward": [0, 0, -1],
                "frustum_corners_mm": [[0, 0, 0]] * 8,
                "target_depth_mm": 100,
                "projected_bounds": {},
            },
            "illumination": {"preset": "neutral_studio"},
            "components": {},
            "state_sha256": "b" * 64,
        }
    )
    state_sha256 = cast(str, session["state_sha256"])
    return RenderManifest.model_validate(
        {
            "source_sha256": source_hash,
            "state_sha256": state_sha256,
            "width": 128,
            "height": 128,
            "samples": 1,
            "engine": "eevee",
            "device": "graphics_hardware",
            "graphics_policy": "software_allowed",
            "graphics": {
                "vendor": "Test",
                "renderer": "Test GPU",
                "version": "1.0",
                "backend": "OPENGL",
                "blender_device_type": "NVIDIA",
                "device_class": "hardware",
            },
            "blender_version": "5.2.0",
            "session": session,
            "color": {
                "path": str(path),
                "media_type": "image/png",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            },
            "luminance": {
                "minimum": 0.1,
                "median": 0.2,
                "maximum": 0.3,
                "crushed_fraction": 0,
                "clipped_fraction": 0,
            },
            "foreground": {
                "visible_fraction": 0.25,
                "visible_pixels": 4096,
                "sampled_pixels": 16384,
                "sample_width": 128,
                "sample_height": 128,
            },
        }
    )


def test_render_requires_open_scene() -> None:
    controller = BlenderController()
    command = RenderImageCommand(request_id="render", op="render.image", output_path="evidence.png")
    with pytest.raises(BlenderWorkerError, match="before a scene is open"):
        controller.render_image(command)


def test_render_rejects_source_asset_as_output(tmp_path: Path) -> None:
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    controller = BlenderController()
    controller._source_snapshot = snapshot_source(source)
    command = RenderImageCommand(request_id="render", op="render.image", output_path=str(source))
    with pytest.raises(BlenderWorkerError, match="must not overwrite"):
        controller.render_image(command)


def test_render_rejects_evaluator_output_that_overwrites_source_dependency(
    tmp_path: Path,
) -> None:
    texture = tmp_path / "evidence.components.png"
    texture.write_bytes(b"texture")
    source = tmp_path / "fixture.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "images": [{"uri": texture.name}],
            }
        ),
        encoding="utf-8",
    )
    controller = BlenderController()
    controller._source_snapshot = snapshot_source(source)
    command = RenderImageCommand(
        request_id="render",
        op="render.image",
        output_path=str(tmp_path / "evidence.png"),
    )

    with pytest.raises(BlenderWorkerError, match="must not overwrite"):
        controller.render_image(command, evaluator_output_dir=tmp_path)


def test_render_validates_worker_artifact_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    output = tmp_path / "evidence.png"
    output.write_bytes(b"png bytes")
    source_snapshot = snapshot_source(source)
    manifest = render_manifest_for(output, source_snapshot.sha256)
    controller = BlenderController()
    controller._source_snapshot = source_snapshot
    controller._source_sha256 = source_snapshot.sha256
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: manifest.model_dump(mode="json"),
    )
    command = RenderImageCommand(
        request_id="render",
        op="render.image",
        output_path=str(output),
        width=128,
        height=128,
        samples=1,
    )

    assert controller.render_image(command) == manifest
    output.write_bytes(b"tampered!")
    with pytest.raises(BlenderWorkerError, match="digest mismatch"):
        controller._verify_render_artifacts(manifest)


@pytest.mark.parametrize(
    ("visibility", "expected_removed", "expected_stop", "third_caption"),
    [
        ((0.2, 0.8), ("blocker",), "threshold_met", "Occluders removed:"),
        ((0.8,), (), "threshold_met_initial", "Alternate focused context"),
    ],
)
def test_contact_sheet_orchestrates_evidence_and_restores_state(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
    visibility: tuple[float, ...],
    expected_removed: tuple[str, ...],
    expected_stop: str,
    third_caption: str,
) -> None:
    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    controller = BlenderController()
    controller._manifest = scene_manifest
    source_snapshot = snapshot_source(source)
    controller._source_snapshot = source_snapshot
    controller._source_sha256 = source_snapshot.sha256
    session = InspectionSession(scene_manifest)
    focus_id = scene_manifest.components[-1].id
    blocker_id = scene_manifest.components[-2].id
    session.mark((focus_id,), MarkMode.SELECTED, "#ff00ff")
    expected_restored = session.snapshot()
    controller._accepted_commands = [
        (
            "component.mark",
            {"component_ids": [focus_id], "mode": "selected", "color": "#ff00ff"},
        )
    ]
    calls: list[str] = []
    visibility_measurements = iter(visibility)
    visibility_pixels = iter(round(fraction * 10) for fraction in visibility)

    def request(operation: str, **arguments: object) -> dict[str, Any]:
        calls.append(operation)
        if operation == "session.reset":
            return session.reset().model_dump(mode="json")
        if operation == "illumination.set":
            illumination = PresetIllumination.model_validate(arguments["illumination"])
            return session.set_illumination(illumination).model_dump(mode="json")
        if operation == "view.orbit":
            projection_payload = cast(dict[str, object], arguments["projection"])
            projection = (
                PerspectiveProjection.model_validate(projection_payload)
                if projection_payload["mode"] == "perspective"
                else OrthographicProjection.model_validate(projection_payload)
            )
            camera = orbit_camera(
                target_mm=cast(tuple[float, float, float], arguments["target_mm"]),
                azimuth_degrees=cast(float, arguments["azimuth_degrees"]),
                elevation_degrees=cast(float, arguments["elevation_degrees"]),
                roll_degrees=cast(float, arguments["roll_degrees"]),
                distance_mm=cast(float, arguments["distance_mm"]),
                projection=projection,
            )
            return session.set_camera(camera).model_dump(mode="json")
        if operation == "component.mark":
            snapshot = session.mark(
                cast(list[str], arguments["component_ids"]),
                MarkMode(cast(str, arguments["mode"])),
                cast(str | None, arguments.get("color")),
            )
            return snapshot.model_dump(mode="json")
        if operation == "component.display":
            snapshot = session.display(
                cast(list[str], arguments["component_ids"]),
                DisplayMode(cast(str, arguments["mode"])),
            )
            return snapshot.model_dump(mode="json")
        if operation == "component.occluders":
            return {
                "sample_count": 6,
                "occluders": [{"component_id": blocker_id, "blocked_rays": 3, "fraction": 0.5}],
            }
        if operation == "component.visibility":
            fraction = next(visibility_measurements)
            return {
                "visible_pixels": next(visibility_pixels),
                "isolated_pixels": 10,
                "visible_fraction": fraction,
                "projected": True,
            }
        if operation == "render.image":
            output = Path(cast(str, arguments["output_path"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (120, 130, 140)).save(output)
            return render_manifest_for(
                output,
                source_snapshot.sha256,
                session.snapshot(),
            ).model_dump(mode="json")
        if operation == "render.style":
            return {}
        raise AssertionError(f"unexpected operation: {operation}")

    monkeypatch.setattr(controller, "request", request)
    output = tmp_path / "contact.png"
    sheet = controller.render_contact_sheet(
        RenderContactSheetCommand(
            request_id="sheet",
            op="render.contact_sheet",
            output_path=str(output),
            focus_component_ids=(focus_id, focus_id),
            panel_width=128,
            panel_height=128,
            samples=1,
        )
    )

    assert sheet.focus_component_ids == (focus_id,)
    resolved_removed = tuple(blocker_id if item == "blocker" else item for item in expected_removed)
    assert sheet.removed_occluder_ids == resolved_removed
    assert sheet.occlusion is not None
    assert sheet.occlusion.visible_fraction_before == visibility[0]
    assert sheet.occlusion.visible_fraction_after == visibility[-1]
    assert sheet.occlusion.camera_source == "generated_focus_context"
    assert sheet.occlusion.camera == sheet.panels[1].render.session.camera
    assert sheet.occlusion.visibility_width_px == 128
    assert sheet.occlusion.visibility_height_px == 128
    assert sheet.occlusion.visible_pixel_count_before == round(visibility[0] * 10)
    assert sheet.occlusion.visible_pixel_count_after == round(visibility[-1] * 10)
    assert sheet.occlusion.isolated_pixel_count == 10
    assert sheet.occlusion.stop_reason == expected_stop
    if resolved_removed:
        assert sheet.occlusion.steps[0].ray_hit_count == 3
        assert sheet.occlusion.steps[0].visible_pixel_count_after == round(visibility[-1] * 10)
        blocker = next(
            component for component in scene_manifest.components if component.id == blocker_id
        )
        assert sheet.occlusion.steps[0].display_name == blocker.display_name
        assert sheet.occlusion.steps[0].component_path == blocker.path
    assert sheet.panels[2].caption.startswith(third_caption)
    if resolved_removed:
        assert sheet.panels[1].render.session.camera == sheet.panels[2].render.session.camera
    else:
        assert sheet.panels[1].render.session.camera != sheet.panels[2].render.session.camera
    assert sheet.panels[0].render.session.camera != sheet.panels[1].render.session.camera
    assert len({panel.render.state_sha256 for panel in sheet.panels}) == 9
    assert [panel.caption for panel in sheet.panels[3:]] == ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    assert all(
        panel.render.session.camera.projection.mode == "orthographic" for panel in sheet.panels[3:]
    )
    assert all(
        panel.render.session.components[focus_id].mark_color == "#ff00ff"
        for panel in sheet.panels[1:]
    )
    assert output.is_file()
    assert session.snapshot() == expected_restored
    assert calls.count("render.image") == 9
    assert calls[-2:] == ["session.reset", "component.mark"]


@pytest.mark.parametrize("aspect_ratio", [1.0, 32.0, 1 / 32])
def test_focused_orbit_extends_far_clip_past_large_target(
    monkeypatch: pytest.MonkeyPatch,
    aspect_ratio: float,
) -> None:
    controller = BlenderController()
    request: dict[str, object] = {}
    monkeypatch.setattr(
        controller,
        "request",
        lambda operation, **arguments: request.update(operation=operation, **arguments),
    )

    controller._set_orbit(
        (0, 0, 0),
        80_000,
        45,
        30,
        PerspectiveProjection(),
        aspect_ratio,
        minimum_distance_mm=1,
    )

    projection = PerspectiveProjection.model_validate(request["projection"])
    distance = cast(float, request["distance_mm"])
    assert projection.far_clip_mm > distance + 80_000


@pytest.mark.parametrize(
    ("panel_dimensions", "visibility_dimensions"),
    [
        ((4096, 128), (256, 8)),
        ((128, 4096), (8, 256)),
        ((4096, 136), (256, 8)),
        ((4096, 16), (256, 4)),
        ((128, 128), (128, 128)),
    ],
)
def test_visibility_dimensions_scale_to_supported_render_bounds(
    panel_dimensions: tuple[int, int],
    visibility_dimensions: tuple[int, int],
) -> None:
    dimensions = BlenderController._visibility_dimensions(*panel_dimensions)

    assert dimensions == visibility_dimensions
    assert min(dimensions) >= 4
    assert max(dimensions) <= 256


def test_occlusion_evidence_reports_actual_visibility_resolution(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = BlenderController()
    camera = InspectionSession(scene_manifest).snapshot().camera
    request: dict[str, object] = {}

    def visibility(operation: str, **arguments: object) -> dict[str, object]:
        request.update(operation=operation, **arguments)
        return {
            "visible_pixels": 8,
            "isolated_pixels": 8,
            "visible_fraction": 1.0,
            "projected": True,
        }

    monkeypatch.setattr(controller, "request", visibility)
    evidence = controller._remove_occluders(
        RenderContactSheetCommand(
            request_id="sheet",
            op="render.contact_sheet",
            output_path=str(tmp_path / "contact.png"),
            focus_component_ids=(scene_manifest.components[-1].id,),
            panel_width=4096,
            panel_height=136,
        ),
        (scene_manifest.components[-1].id,),
        camera=camera,
    )

    assert request["width"] == 256
    assert request["height"] == 8
    assert evidence.visibility_width_px == 256
    assert evidence.visibility_height_px == 8
    assert evidence.camera == camera


def test_contact_sheet_validates_scene_focus_and_output(
    tmp_path: Path,
    scene_manifest: SceneManifest,
) -> None:
    command = RenderContactSheetCommand(
        request_id="sheet",
        op="render.contact_sheet",
        output_path=str(tmp_path / "contact.jpg"),
        focus_component_ids=("missing",),
    )
    with pytest.raises(BlenderWorkerError, match="before a scene is open"):
        BlenderController().render_contact_sheet(command)

    source = tmp_path / "fixture.glb"
    source.write_bytes(b"model")
    controller = BlenderController()
    controller._manifest = scene_manifest
    controller._source_snapshot = snapshot_source(source)
    with pytest.raises(BlenderWorkerError, match="unknown component"):
        controller.render_contact_sheet(command)

    command = command.model_copy(
        update={"focus_component_ids": (scene_manifest.components[-1].id,)}
    )
    with pytest.raises(BlenderWorkerError, match=r"must end in \.png"):
        controller.render_contact_sheet(command)


def test_contact_sheet_rejects_staging_path_that_overwrites_source_dependency(
    tmp_path: Path,
    scene_manifest: SceneManifest,
) -> None:
    output = tmp_path / "contact.png"
    staging = tmp_path / ".contact.png.part"
    staging.write_bytes(b"source dependency")
    source = tmp_path / "fixture.gltf"
    source.write_text(
        json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": staging.name}]}),
        encoding="utf-8",
    )
    controller = BlenderController()
    controller._manifest = scene_manifest
    controller._source_snapshot = snapshot_source(source)
    command = RenderContactSheetCommand(
        request_id="sheet",
        op="render.contact_sheet",
        output_path=str(output),
        focus_component_ids=(scene_manifest.components[-1].id,),
    )

    with pytest.raises(BlenderWorkerError, match="must not overwrite"):
        controller.render_contact_sheet(command)


def test_perspective_orbit_framing_uses_limiting_panel_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = BlenderController()
    distances: list[float] = []

    def request(operation: str, **arguments: object) -> dict[str, object]:
        assert operation == "view.orbit"
        distances.append(cast(float, arguments["distance_mm"]))
        return {}

    monkeypatch.setattr(controller, "request", request)
    projection = PerspectiveProjection()
    controller._set_orbit((0, 0, 0), 100, 45, 30, projection, 2.0)
    controller._set_orbit((0, 0, 0), 100, 45, 30, projection, 0.5)

    assert distances[1] > distances[0]


def test_environment_maps_are_verified_and_copied_into_content_cache(tmp_path: Path) -> None:
    source = tmp_path / "studio.exr"
    source.write_bytes(b"declared environment content")
    source_hash = sha256_file(source)
    controller = BlenderController(artifact_cache_root=tmp_path / "cache")
    illumination = CustomIllumination(
        background_rgb=(0, 0, 0),
        ambient_strength=0,
        environment_map=EnvironmentMap(path=str(source), sha256=source_hash),
    )

    cached = controller._cache_environment_map(illumination)

    assert isinstance(cached, CustomIllumination)
    assert cached.environment_map is not None
    cached_path = Path(cached.environment_map.path)
    assert cached_path != source
    assert cached_path.read_bytes() == source.read_bytes()
    assert cached.environment_map.sha256 == source_hash


def test_environment_map_declaration_must_match_file_hash(tmp_path: Path) -> None:
    source = tmp_path / "studio.hdr"
    source.write_bytes(b"actual")
    controller = BlenderController(artifact_cache_root=tmp_path / "cache")
    illumination = CustomIllumination(
        background_rgb=(0, 0, 0),
        ambient_strength=0,
        environment_map=EnvironmentMap(path=str(source), sha256="a" * 64),
    )

    with pytest.raises(BlenderWorkerError, match="hash does not match"):
        controller._cache_environment_map(illumination)
