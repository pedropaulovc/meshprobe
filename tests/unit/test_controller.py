from __future__ import annotations

import json
import os
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from meshprobe.camera import orbit_camera
from meshprobe.controller import (
    BlenderController,
    BlenderWorkerCrashed,
    BlenderWorkerError,
    BlenderWorkerTimeout,
    sha256_file,
)
from meshprobe.identity import stable_component_id
from meshprobe.models import (
    CustomIllumination,
    DisplayMode,
    EnvironmentMap,
    MarkMode,
    OrthographicProjection,
    PerspectiveProjection,
    PresetIllumination,
    RenderManifest,
    SceneManifest,
    SessionSnapshot,
)
from meshprobe.protocol import (
    ComponentFindCommand,
    ComponentInspectCommand,
    ComponentMarkCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneDescribeCommand,
    SceneOpenCommand,
    SessionResetCommand,
)
from meshprobe.selectors import ComponentSelector, SelectorKind
from meshprobe.session import InspectionSession
from meshprobe.sources import snapshot_source


def make_fake_blender(tmp_path: Path, protocol_version: int = 1) -> Path:
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

    def open_scene(path: Path) -> SimpleNamespace:
        calls.append(("open", path))
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
        ("open", source),
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

    def reopen(path: Path) -> SimpleNamespace:
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
            "device": "graphics",
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


def test_contact_sheet_orchestrates_evidence_and_restores_state(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
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
    session.mark((focus_id,), MarkMode.SELECTED)
    expected_restored = session.snapshot()
    controller._accepted_commands = [
        ("component.mark", {"component_ids": [focus_id], "mode": "selected"})
    ]
    calls: list[str] = []

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
            )
            return snapshot.model_dump(mode="json")
        if operation == "component.display":
            snapshot = session.display(
                cast(list[str], arguments["component_ids"]),
                DisplayMode(cast(str, arguments["mode"])),
            )
            return snapshot.model_dump(mode="json")
        if operation == "component.occluders":
            return {"occluders": [{"component_id": blocker_id, "blocked_rays": 3, "fraction": 0.5}]}
        if operation == "render.image":
            output = Path(cast(str, arguments["output_path"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (120, 130, 140)).save(output)
            return render_manifest_for(
                output,
                source_snapshot.sha256,
                session.snapshot(),
            ).model_dump(mode="json")
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
    assert sheet.removed_occluder_ids == (blocker_id,)
    assert [panel.caption for panel in sheet.panels[3:]] == ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    assert all(
        panel.render.session.camera.projection.mode == "orthographic" for panel in sheet.panels[3:]
    )
    assert output.is_file()
    assert session.snapshot() == expected_restored
    assert calls.count("render.image") == 9
    assert calls[-2:] == ["session.reset", "component.mark"]


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
