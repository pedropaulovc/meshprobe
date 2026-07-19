from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from pydantic import JsonValue

from meshprobe.camera import orbit_angles_from_orientation
from meshprobe.models import (
    CapabilityWarning,
    CustomIllumination,
    DisplayMode,
    EdgeType,
    EnvironmentMap,
    GraphicsDeviceClass,
    GraphicsPlatform,
    MarkMode,
    PerspectiveProjection,
    RenderStyle,
    RenderStyleState,
    SceneManifest,
    ShadedEdgesStyle,
)
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentMarkCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionResetCommand,
    SessionSnapshotCommand,
    SessionUndoCommand,
    ViewFrameCommand,
    ViewRotateCommand,
    ViewSetCommand,
)
from meshprobe.selectors import ComponentIndex, ComponentSelector, SelectorKind
from meshprobe.service import CommandResponse
from meshprobe.session import InspectionSession
from meshprobe.workspace import (
    DurableSessionState,
    SessionFiles,
    SessionManager,
    atomic_json,
)


class FakeSessionService:
    def __init__(
        self, manifest: SceneManifest, *, graphics: GraphicsPlatform | None = None
    ) -> None:
        self.manifest = manifest
        self.session: InspectionSession | None = None
        self.closed = False
        self.killed = False
        self.graphics = graphics
        self.received_commands: list[Command] = []
        self.fail_on_request_id: str | None = None

    @property
    def worker_pid(self) -> int | None:
        return None if self.closed or self.killed else 4242

    def execute(self, command: Command) -> CommandResponse:
        self.received_commands.append(command)
        if command.request_id == self.fail_on_request_id:
            raise RuntimeError(f"forced replay failure: {command.request_id}")
        if isinstance(command, SceneOpenCommand):
            self.session = InspectionSession(self.manifest, aspect_ratio=command.aspect_ratio)
            result: JsonValue = self.manifest.model_dump(mode="json")
        elif isinstance(command, SessionSnapshotCommand):
            assert self.session is not None
            result = {
                "scene": self.manifest.model_dump(mode="json"),
                "session": self.session.snapshot().model_dump(mode="json"),
            }
        elif isinstance(command, ComponentDisplayCommand):
            assert self.session is not None
            result = self.session.display(command.component_ids, command.mode).model_dump(
                mode="json"
            )
        elif isinstance(command, ComponentMarkCommand):
            assert self.session is not None
            result = self.session.mark(
                command.component_ids, command.mode, color=command.color
            ).model_dump(mode="json")
        elif isinstance(command, ComponentFindCommand):
            assert self.session is not None
            result = [
                component.model_dump(mode="json")
                for component in ComponentIndex(self.manifest).find(command.selector)
            ]
        elif isinstance(command, (ViewFrameCommand, ViewRotateCommand)):
            assert self.session is not None
            result = {"state_sha256": self.session.snapshot().state_sha256}
        elif isinstance(command, ViewSetCommand):
            assert self.session is not None
            result = self.session.set_camera(command.camera).model_dump(mode="json")
        elif isinstance(command, SessionResetCommand):
            assert self.session is not None
            result = self.session.reset().model_dump(mode="json")
        elif isinstance(command, (RenderImageCommand, RenderContactSheetCommand)):
            assert self.session is not None
            result = {}
        else:
            raise AssertionError(f"unexpected command: {command.op}")
        return CommandResponse(request_id=command.request_id, op=command.op, result=result)

    def close(self) -> None:
        self.closed = True

    def kill(self) -> None:
        self.killed = True


def test_session_manager_writes_compact_queryable_state(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    receipt = manager.open("review", source)
    session_root = tmp_path / ".meshprobe" / "sessions" / "review"

    assert receipt.result_path is not None
    assert (session_root / "scene.json").is_file()
    assert (session_root / "components.yml").is_file()
    assert (session_root / "state.yml").is_file()
    assert (session_root / "checkpoint.json").is_file()
    assert (session_root / "events.jsonl").is_file()
    components = yaml.safe_load((session_root / "components.yml").read_text(encoding="utf-8"))
    paths = [component["path"] for component in components["components"]]
    assert paths == sorted(paths)
    assert [component["ref"] for component in components["components"]] == ["c1", "c2", "c3"]

    component_id = manager.resolve_component("review", "c2")
    assert manager.resolve_component("review", "retaining clip") == component_id
    changed = manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    state = yaml.safe_load((session_root / "state.yml").read_text(encoding="utf-8"))
    assert changed.state_sha256 == state["state_sha256"]
    assert len(changed.components) == 1
    assert changed.components[0].id == component_id
    assert changed.components[0].name == "retaining clip"
    assert changed.components[0].path == "assembly/cover/clip"
    assert state["components"]["default"] == {"display": "shown", "mark": "unmarked"}
    assert state["components"]["overrides"]["c2"] == {
        "name": "retaining clip",
        "display": "hidden",
    }
    expected_azimuth, expected_elevation = orbit_angles_from_orientation(
        tuple(state["camera"]["pose"]["orientation_xyzw"])
    )
    assert state["camera_orbit_angles"] == {
        "azimuth_degrees": pytest.approx(expected_azimuth),
        "elevation_degrees": pytest.approx(expected_elevation),
    }
    # A legacy v1 state.yml written before this field existed has no camera_orbit_angles;
    # it must still validate and be backfilled from the camera orientation on read.
    legacy = {key: value for key, value in state.items() if key != "camera_orbit_angles"}
    restored = DurableSessionState.model_validate(legacy)
    assert restored.camera_orbit_angles is not None
    assert restored.camera_orbit_angles.azimuth_degrees == pytest.approx(expected_azimuth)
    assert restored.camera_orbit_angles.elevation_degrees == pytest.approx(expected_elevation)
    assert state["render_style"] == {
        "style": "screen_edges",
        "shaded_edges": {
            "line_color": "#202020",
            "line_width": 1.5,
            "crease_angle_degrees": 120.0,
            "edge_types": ["silhouette", "border", "crease", "material_boundary"],
        },
    }
    assert services[0].session is not None
    SessionManager._write_state(
        SessionFiles(manager.root, "review"),
        services[0].session.snapshot(),
        render_style=RenderStyleState(
            style=RenderStyle.SHADED_EDGES,
            shaded_edges=ShadedEdgesStyle(
                line_color="#101820",
                line_width=2,
                crease_angle_degrees=45,
                edge_types=(EdgeType.SILHOUETTE, EdgeType.CREASE),
            ),
        ),
    )
    styled_state = yaml.safe_load((session_root / "state.yml").read_text(encoding="utf-8"))
    assert styled_state["render_style"] == {
        "style": "shaded_edges",
        "shaded_edges": {
            "line_color": "#101820",
            "line_width": 2.0,
            "crease_angle_degrees": 45.0,
            "edge_types": ["silhouette", "crease"],
        },
    }
    checkpoint = json.loads((session_root / "checkpoint.json").read_text(encoding="utf-8"))
    assert [command["op"] for command in checkpoint["accepted_commands"]] == ["component.display"]


def test_undo_counts_mutation_commands_not_components_or_reads(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", source)
    component_ids = tuple(component.id for component in scene_manifest.components[:2])
    manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide-two",
            op="component.display",
            component_ids=component_ids,
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.execute(
        "review",
        ComponentMarkCommand(
            request_id="mark-one",
            op="component.mark",
            component_ids=(component_ids[0],),
            mode=MarkMode.HIGHLIGHTED,
        ),
    )
    manager.execute(
        "review",
        SessionSnapshotCommand(request_id="read-only", op="session.snapshot"),
    )

    first = manager.execute("review", SessionUndoCommand(request_id="undo-mark", op="session.undo"))
    first_result = manager.raw_result(first)
    assert first_result["result"]["undone"] == 1  # type: ignore[index]
    assert first_result["result"]["remaining"] == 1  # type: ignore[index]
    assert services[-1].session is not None
    first_snapshot = services[-1].session.snapshot()
    assert first_snapshot.components[component_ids[0]].mark is MarkMode.UNMARKED
    assert all(
        first_snapshot.components[component_id].display is DisplayMode.HIDDEN
        for component_id in component_ids
    )

    manager.execute("review", SessionUndoCommand(request_id="undo-display", op="session.undo"))
    assert services[-1].session is not None
    restored = services[-1].session.snapshot()
    assert all(
        restored.components[component_id].display is DisplayMode.SHOWN
        for component_id in component_ids
    )

    manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide-two-again",
            op="component.display",
            component_ids=component_ids,
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.execute(
        "review",
        ComponentMarkCommand(
            request_id="mark-one-again",
            op="component.mark",
            component_ids=(component_ids[0],),
            mode=MarkMode.HIGHLIGHTED,
        ),
    )
    counted = manager.execute(
        "review", SessionUndoCommand(request_id="undo-two", op="session.undo", count=2)
    )
    counted_result = manager.raw_result(counted)
    assert counted_result["result"]["undone"] == 2  # type: ignore[index]
    assert counted_result["result"]["remaining"] == 0  # type: ignore[index]
    with pytest.raises(ValueError, match="only 0 available"):
        manager.execute("review", SessionUndoCommand(request_id="too-many", op="session.undo"))


def test_undo_survives_manager_restart_and_reset_is_a_hard_boundary(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("review", source)
    component_id = scene_manifest.components[0].id
    manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.close("review")

    recovered = SessionManager(root, service_factory=factory)
    recovered.execute(
        "review", SessionUndoCommand(request_id="undo-after-restart", op="session.undo")
    )
    assert services[-1].session is not None
    assert services[-1].session.snapshot().components[component_id].display is DisplayMode.SHOWN

    recovered.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide-again",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    recovered.execute("review", SessionResetCommand(request_id="reset", op="session.reset"))
    with pytest.raises(ValueError, match="only 0 available since the last reset"):
        recovered.execute("review", SessionUndoCommand(request_id="past-reset", op="session.undo"))


def test_undo_after_restart_does_not_replay_the_discarded_command(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []
    reject_discarded_replay = False

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        if reject_discarded_replay:
            service.fail_on_request_id = "discarded-mutation"
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("review", source)
    component_id = scene_manifest.components[0].id
    manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="discarded-mutation",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.close("review")
    reject_discarded_replay = True

    recovered = SessionManager(root, service_factory=factory)
    recovered.execute("review", SessionUndoCommand(request_id="undo-discarded", op="session.undo"))

    assert len(services) == 2
    assert services[-1].session is not None
    assert services[-1].session.snapshot().components[component_id].display is DisplayMode.SHOWN
    assert all(
        command.request_id != "discarded-mutation" for command in services[-1].received_commands
    )


def test_undo_rebuild_resets_render_style_to_worker_default(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", source)
    manager.execute(
        "review",
        RenderImageCommand(
            request_id="shaded-edges",
            op="render.image",
            output_path=str(tmp_path / "render.png"),
            style=RenderStyle.SHADED_EDGES,
        ),
    )
    component_id = scene_manifest.components[0].id
    manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )

    manager.execute("review", SessionUndoCommand(request_id="undo", op="session.undo"))

    state = yaml.safe_load(SessionFiles(manager.root, "review").state.read_text())
    assert state["render_style"] == RenderStyleState().model_dump(mode="json")


def test_undo_replay_failure_keeps_original_service_and_durable_state(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        if services:
            service.fail_on_request_id = "hide"
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", source)
    component_id = scene_manifest.components[0].id
    manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.execute(
        "review",
        ComponentMarkCommand(
            request_id="mark",
            op="component.mark",
            component_ids=(component_id,),
            mode=MarkMode.HIGHLIGHTED,
        ),
    )
    files = SessionFiles(manager.root, "review")
    checkpoint_before = files.checkpoint.read_bytes()
    state_before = files.state.read_bytes()

    with pytest.raises(RuntimeError, match="forced replay failure: hide"):
        manager.execute("review", SessionUndoCommand(request_id="undo-fails", op="session.undo"))

    assert len(services) == 2
    assert services[1].killed
    assert not services[0].closed
    assert not services[0].killed
    assert files.checkpoint.read_bytes() == checkpoint_before
    assert files.state.read_bytes() == state_before
    events = [json.loads(line) for line in files.events.read_text().splitlines()]
    assert events[-1]["request_id"] == "undo-fails"
    assert events[-1]["status"] == "rejected"
    assert not (files.results / "undo-fails.json").exists()

    manager.execute(
        "review", SessionSnapshotCommand(request_id="original-still-live", op="session.snapshot")
    )
    assert len(services) == 2
    assert services[0].session is not None
    original = services[0].session.snapshot().components[component_id]
    assert original.display is DisplayMode.HIDDEN
    assert original.mark is MarkMode.HIGHLIGHTED


def test_undo_checkpoint_write_failure_rolls_back_derived_files(
    tmp_path: Path, scene_manifest: SceneManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", source)
    component_id = scene_manifest.components[0].id
    manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    files = SessionFiles(manager.root, "review")
    checkpoint_before = files.checkpoint.read_bytes()
    state_before = files.state.read_bytes()

    def fail_checkpoint(path: Path, value: object, *, mode: int | None = None) -> None:
        if path == files.checkpoint:
            raise OSError("forced checkpoint write failure")
        atomic_json(path, value, mode=mode)

    monkeypatch.setattr("meshprobe.workspace.atomic_json", fail_checkpoint)
    with pytest.raises(OSError, match="forced checkpoint write failure"):
        manager.execute("review", SessionUndoCommand(request_id="undo-write", op="session.undo"))

    assert services[-1].killed
    assert not services[0].closed
    assert files.checkpoint.read_bytes() == checkpoint_before
    assert files.state.read_bytes() == state_before
    assert not (files.results / "undo-write.json").exists()
    events = [json.loads(line) for line in files.events.read_text().splitlines()]
    assert events[-1]["request_id"] == "undo-write"
    assert events[-1]["status"] == "rejected"


def test_session_manager_execute_propagates_open_aspect_ratio_to_the_worker(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.execute(
        "review",
        SceneOpenCommand(
            request_id="open",
            op="scene.open",
            source_path=str(source),
            aspect_ratio=2.5,
        ),
    )

    assert len(services) == 1
    received = services[0].received_commands[0]
    assert isinstance(received, SceneOpenCommand)
    assert received.aspect_ratio == 2.5


def test_find_receipt_reports_match_count(tmp_path: Path, scene_manifest: SceneManifest) -> None:
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)

    hit = manager.execute(
        "review",
        ComponentFindCommand(
            request_id="hit",
            op="component.find",
            selector=ComponentSelector(kind=SelectorKind.EXACT_NAME, pattern="idler"),
        ),
    )
    miss = manager.execute(
        "review",
        ComponentFindCommand(
            request_id="miss",
            op="component.find",
            selector=ComponentSelector(kind=SelectorKind.EXACT_NAME, pattern="nonexistent-widget"),
        ),
    )
    snapshot = manager.execute(
        "review",
        SessionSnapshotCommand(request_id="snap", op="session.snapshot"),
    )

    assert hit.match_count == 1
    assert [(component.ref, component.name, component.path) for component in hit.components] == [
        ("c3", "idler", "assembly/drive/idler")
    ]
    assert miss.match_count == 0
    assert miss.components == ()
    assert snapshot.match_count is None


def test_open_receipt_surfaces_all_manifest_warnings(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    manifest = scene_manifest.model_copy(
        update={
            "warnings": (
                CapabilityWarning(code="illumination.generated", message="Generated lighting."),
                CapabilityWarning(code="hierarchy.flattened", message="Flattened hierarchy."),
            )
        }
    )
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")

    receipt = manager.open("review", source)

    assert receipt.warnings == (
        "illumination.generated: Generated lighting.",
        "hierarchy.flattened: Flattened hierarchy.",
    )


def test_view_frame_is_recorded_for_checkpoint_recovery(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    manager.open("review", source)
    component_id = manager.resolve_component("review", "c2")

    manager.execute(
        "review",
        ViewFrameCommand(
            request_id="frame",
            op="view.frame",
            focus_component_ids=(component_id,),
        ),
    )

    checkpoint = json.loads(
        (tmp_path / ".meshprobe" / "sessions" / "review" / "checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert [command["op"] for command in checkpoint["accepted_commands"]] == ["view.frame"]


def test_contact_sheet_records_worker_default_render_style(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    manager.execute(
        "review",
        RenderImageCommand(
            request_id="edges",
            op="render.image",
            output_path=str(tmp_path / "edges.png"),
            style=RenderStyle.SHADED_EDGES,
        ),
    )
    files = SessionFiles(manager.root, "review")
    edged_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert edged_state["render_style"]["style"] == "shaded_edges"

    manager.execute(
        "review",
        SessionSnapshotCommand(request_id="same-renderer", op="session.snapshot"),
    )
    live_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert live_state["render_style"]["style"] == "shaded_edges"

    manager.execute(
        "review",
        RenderContactSheetCommand(
            request_id="sheet",
            op="render.contact_sheet",
            output_path=str(tmp_path / "sheet.png"),
            focus_component_ids=(scene_manifest.components[0].id,),
        ),
    )

    sheet_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert sheet_state["render_style"] == RenderStyleState().model_dump(mode="json")


def test_render_image_records_the_workers_resolved_style(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    class FallbackStyleService(FakeSessionService):
        def execute(self, command: Command) -> CommandResponse:
            if isinstance(command, RenderImageCommand):
                assert self.session is not None
                return CommandResponse(
                    request_id=command.request_id,
                    op=command.op,
                    result={"style": "shaded"},
                )
            return super().execute(command)

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FallbackStyleService(scene_manifest),
    )
    manager.open("review", source)
    manager.execute(
        "review",
        RenderImageCommand(
            request_id="render",
            op="render.image",
            output_path=str(tmp_path / "frame.png"),
        ),
    )

    files = SessionFiles(manager.root, "review")
    state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert state["render_style"]["style"] == "shaded"


def _fake_graphics_platform() -> GraphicsPlatform:
    return GraphicsPlatform(
        vendor="Microsoft",
        renderer="D3D12 (Vendor GPU)",
        version="1.0",
        backend="d3d12",
        blender_device_type="CPU",
        device_class=GraphicsDeviceClass.SOFTWARE,
        warnings=(
            "Blender labels this adapter SOFTWARE, but its renderer identifies a "
            "hardware-backed D3D12 adapter.",
        ),
    )


def test_graphics_warning_surfaces_once_and_not_on_every_command(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    graphics = _fake_graphics_platform()
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest, graphics=graphics),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")

    open_receipt = manager.open("review", source)
    assert open_receipt.warnings == graphics.warnings

    snapshot_receipt = manager.execute(
        "review",
        SessionSnapshotCommand(request_id="snapshot", op="session.snapshot"),
    )
    assert snapshot_receipt.warnings == ()


def test_graphics_warning_resurfaces_when_the_worker_is_recreated(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    graphics = _fake_graphics_platform()

    class RecoveringSessionService(FakeSessionService):
        worker_generation = 1

        @property
        def worker_pid(self) -> int | None:
            if self.closed or self.killed:
                return None
            return 4_000 + self.worker_generation

        def execute(self, command: Command) -> CommandResponse:
            if command.request_id == "recreate-worker":
                self.worker_generation += 1
            return super().execute(command)

    service = RecoveringSessionService(scene_manifest, graphics=graphics)
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: service,
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)

    preserved_receipt = manager.execute(
        "review",
        SessionSnapshotCommand(request_id="same-worker", op="session.snapshot"),
    )
    assert preserved_receipt.warnings == ()

    recreated_receipt = manager.execute(
        "review",
        SessionSnapshotCommand(request_id="recreate-worker", op="session.snapshot"),
    )
    assert recreated_receipt.warnings == graphics.warnings


def test_graphics_warning_resurfaces_after_a_recovered_workers_first_command_is_rejected(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    graphics = _fake_graphics_platform()

    class RejectingFirstCommandService(FakeSessionService):
        def execute(self, command: Command) -> CommandResponse:
            if command.request_id == "will-fail":
                raise ValueError("rejected")
            return super().execute(command)

    def factory() -> RejectingFirstCommandService:
        return RejectingFirstCommandService(scene_manifest, graphics=graphics)

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", source)

    # A fresh manager with no in-memory service simulates a daemon restart: the
    # next command recovers a brand-new worker before running the user's command.
    recovered = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    with pytest.raises(ValueError):
        recovered.execute(
            "review",
            SessionSnapshotCommand(request_id="will-fail", op="session.snapshot"),
        )

    receipt = recovered.execute(
        "review",
        SessionSnapshotCommand(request_id="now-succeeds", op="session.snapshot"),
    )
    assert receipt.warnings == graphics.warnings


def test_render_image_warns_when_requested_aspect_diverges_from_the_framed_camera(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    # FakeSessionService/InspectionSession compute camera_diagnostics with the default
    # aspect ratio of 1.0 (no orbit/frame command overrides it here), so a square render
    # matches that framing and a wide render diverges from it.
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)

    matched = manager.execute(
        "review",
        RenderImageCommand(
            request_id="matched",
            op="render.image",
            output_path=str(tmp_path / "matched.png"),
            width=512,
            height=512,
        ),
    )
    assert matched.warnings == ()

    mismatched = manager.execute(
        "review",
        RenderImageCommand(
            request_id="mismatched",
            op="render.image",
            output_path=str(tmp_path / "mismatched.png"),
            width=1024,
            height=512,
        ),
    )
    assert len(mismatched.warnings) == 1
    assert "requested 1024x512" in mismatched.warnings[0]
    assert "aspect ratio 2" in mismatched.warnings[0]
    assert "framed for aspect ratio 1" in mismatched.warnings[0]


def test_render_image_does_not_warn_within_the_aspect_ratio_tolerance(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)

    # 1.005 is within the 1% relative tolerance of the framed 1.0 aspect ratio.
    receipt = manager.execute(
        "review",
        RenderImageCommand(
            request_id="within-tolerance",
            op="render.image",
            output_path=str(tmp_path / "within-tolerance.png"),
            width=1005,
            height=1000,
        ),
    )
    assert receipt.warnings == ()


def test_render_image_uses_the_checkpointed_open_aspect_before_any_view_refit(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source, aspect_ratio=2.0)

    receipt = manager.execute(
        "review",
        RenderImageCommand(
            request_id="matches-open-framing",
            op="render.image",
            output_path=str(tmp_path / "matches-open-framing.png"),
            width=1024,
            height=512,
        ),
    )

    assert receipt.warnings == ()


def test_render_image_reads_framing_aspect_from_the_last_refit_not_the_snapshot(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    # The framing aspect must come from the last camera-refitting command's aspect_ratio
    # (recorded in the checkpoint), NOT from snapshot.camera_diagnostics: view.move/
    # view.rotate overwrite that diagnostics field with their own default while leaving the
    # projection framed as before, so trusting it would false-warn a render that still
    # matches the framing. FakeSessionService's snapshot reports the default 1.0 aspect
    # regardless of the view.frame below, so a render matching the framed 2.0 only stays
    # quiet if the warning reads the checkpoint.
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    manager.execute(
        "review",
        ViewFrameCommand(
            request_id="frame-wide",
            op="view.frame",
            focus_component_ids=(scene_manifest.components[0].id,),
            aspect_ratio=2.0,
        ),
    )

    receipt = manager.execute(
        "review",
        RenderImageCommand(
            request_id="matches-framing",
            op="render.image",
            output_path=str(tmp_path / "matches-framing.png"),
            width=1024,
            height=512,
        ),
    )
    assert receipt.warnings == ()


def test_view_rotate_with_a_projection_updates_the_tracked_framing_aspect(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    # A view.rotate that supplies its own projection replaces the camera projection, so it
    # reframes at its aspect_ratio (unlike a bare rotate, which only nudges the pose). The
    # warning must adopt that 2.0 as the new framing, so a matching 2:1 render stays quiet
    # even though an earlier view.frame framed for 1.0.
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    manager.execute(
        "review",
        ViewFrameCommand(
            request_id="frame-square",
            op="view.frame",
            focus_component_ids=(scene_manifest.components[0].id,),
            aspect_ratio=1.0,
        ),
    )
    manager.execute(
        "review",
        ViewRotateCommand(
            request_id="rotate-wide",
            op="view.rotate",
            target_mm=(0, 0, 0),
            axis="z",
            degrees=15,
            projection=PerspectiveProjection(),
            aspect_ratio=2.0,
        ),
    )

    receipt = manager.execute(
        "review",
        RenderImageCommand(
            request_id="matches-rotate",
            op="render.image",
            output_path=str(tmp_path / "matches-rotate.png"),
            width=1024,
            height=512,
        ),
    )
    assert receipt.warnings == ()


def test_bare_view_rotate_does_not_change_the_tracked_framing_aspect(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    # A bare view.rotate (no projection) reuses the current projection, so its aspect_ratio
    # must NOT be read as the framing aspect: the framing stays the view.frame's 2.0, and a
    # square render still warns even though the bare rotate carried a 1.0 default.
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    manager.execute(
        "review",
        ViewFrameCommand(
            request_id="frame-wide",
            op="view.frame",
            focus_component_ids=(scene_manifest.components[0].id,),
            aspect_ratio=2.0,
        ),
    )
    manager.execute(
        "review",
        ViewRotateCommand(
            request_id="rotate-bare",
            op="view.rotate",
            target_mm=(0, 0, 0),
            axis="z",
            degrees=15,
        ),
    )

    receipt = manager.execute(
        "review",
        RenderImageCommand(
            request_id="square-after-bare-rotate",
            op="render.image",
            output_path=str(tmp_path / "square-after-bare-rotate.png"),
            width=512,
            height=512,
        ),
    )
    assert len(receipt.warnings) == 1
    assert "framed for aspect ratio 2" in receipt.warnings[0]


def test_recreated_worker_resets_durable_render_style(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    class RecoveringSessionService(FakeSessionService):
        worker_generation = 1

        @property
        def worker_pid(self) -> int | None:
            if self.closed or self.killed:
                return None
            return 4_000 + self.worker_generation

        def execute(self, command: Command) -> CommandResponse:
            if command.request_id == "recreate-worker":
                self.worker_generation += 1
            return super().execute(command)

    service = RecoveringSessionService(scene_manifest)
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: service,
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    manager.execute(
        "review",
        RenderImageCommand(
            request_id="edges",
            op="render.image",
            output_path=str(tmp_path / "edges.png"),
            style=RenderStyle.SHADED_EDGES,
        ),
    )
    files = SessionFiles(manager.root, "review")

    manager.execute(
        "review",
        SessionSnapshotCommand(request_id="recreate-worker", op="session.snapshot"),
    )

    recovered_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert recovered_state["render_style"] == RenderStyleState().model_dump(mode="json")


def test_worker_recreated_after_render_resets_durable_render_style(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    class SnapshotRecoveringSessionService(FakeSessionService):
        worker_generation = 1
        recreate_on_checkpoint = False

        @property
        def worker_pid(self) -> int | None:
            if self.closed or self.killed:
                return None
            return 5_000 + self.worker_generation

        def execute(self, command: Command) -> CommandResponse:
            if command.request_id == "checkpoint" and self.recreate_on_checkpoint:
                self.worker_generation += 1
                self.recreate_on_checkpoint = False
            return super().execute(command)

    service = SnapshotRecoveringSessionService(scene_manifest)
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: service,
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    service.recreate_on_checkpoint = True

    manager.execute(
        "review",
        RenderImageCommand(
            request_id="edges",
            op="render.image",
            output_path=str(tmp_path / "edges.png"),
            style=RenderStyle.SHADED_EDGES,
        ),
    )

    files = SessionFiles(manager.root, "review")
    recovered_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert recovered_state["render_style"] == RenderStyleState().model_dump(mode="json")


def test_reused_request_ids_preserve_every_result(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    command = SessionSnapshotCommand(request_id="same", op="session.snapshot")

    first = manager.execute("review", command)
    second = manager.execute("review", command)

    assert first.result_path != second.result_path
    assert first.result_path is not None
    assert second.result_path is not None
    assert (tmp_path / first.result_path).is_file()
    assert (tmp_path / second.result_path).is_file()


def test_checkpoint_persists_cached_environment_map_path(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager.open("review", source)
    assert services[0].session is not None
    original = CustomIllumination(
        background_rgb=(0.0, 0.0, 0.0),
        ambient_strength=0.0,
        environment_map=EnvironmentMap(path="/tmp/upload.hdr", sha256="a" * 64),
    )
    assert original.environment_map is not None
    cached = original.model_copy(
        update={
            "environment_map": original.environment_map.model_copy(
                update={"path": "/durable/cache/environment.hdr"}
            )
        }
    )
    snapshot = services[0].session.snapshot().model_copy(update={"illumination": cached})
    command = IlluminationSetCommand(
        request_id="light",
        op="illumination.set",
        illumination=original,
    )

    SessionManager._update_checkpoint(SessionFiles(manager.root, "review"), command, snapshot)

    checkpoint = json.loads(
        (manager.root / "sessions" / "review" / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert (
        checkpoint["accepted_commands"][0]["illumination"]["environment_map"]["path"]
        == "/durable/cache/environment.hdr"
    )


def test_reopening_a_session_replaces_all_prior_session_artifacts(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", first, request_id="first-open")
    files = SessionFiles(manager.root, "review")
    (files.results / "stale.json").write_text("{}", encoding="utf-8")
    (files.snapshots / "stale.json").write_text("{}", encoding="utf-8")
    (files.artifacts / "stale.png").write_bytes(b"stale")

    manager.open("review", second, request_id="second-open")

    assert services[0].closed
    assert not (files.results / "stale.json").exists()
    assert not (files.snapshots / "stale.json").exists()
    assert not (files.artifacts / "stale.png").exists()
    events = [json.loads(line) for line in files.events.read_text(encoding="utf-8").splitlines()]
    assert [event["request_id"] for event in events] == ["second-open"]
    metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
    assert metadata["source_path"] == str(second.resolve())


def test_rejected_reopen_preserves_existing_session(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", source)
    files = SessionFiles(manager.root, "review")
    (files.artifacts / "preserved.png").write_bytes(b"evidence")

    with pytest.raises(FileNotFoundError):
        manager.open("review", tmp_path / "missing.glb")

    assert not services[0].closed
    assert (files.artifacts / "preserved.png").read_bytes() == b"evidence"
    assert json.loads(files.metadata.read_text(encoding="utf-8"))["source_path"] == str(
        source.resolve()
    )


def test_import_failure_preserves_existing_session(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    class FailingSessionService(FakeSessionService):
        def execute(self, command: Command) -> CommandResponse:
            if isinstance(command, SceneOpenCommand):
                raise RuntimeError("import rejected")
            return super().execute(command)

    def factory() -> FakeSessionService:
        service: FakeSessionService
        if services:
            service = FailingSessionService(scene_manifest)
        else:
            service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("review", first)
    files = SessionFiles(manager.root, "review")
    (files.artifacts / "preserved.png").write_bytes(b"evidence")

    with pytest.raises(RuntimeError, match="import rejected"):
        manager.open("review", second)

    assert not services[0].closed
    assert services[1].killed
    assert (files.artifacts / "preserved.png").read_bytes() == b"evidence"


@pytest.mark.parametrize("termination", ["close", "kill"])
def test_closed_and_killed_sessions_recover_checkpoint_with_default_render_style(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    termination: str,
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("default", source)
    component_id = manager.resolve_component("default", "c1")
    manager.execute(
        "default",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.execute(
        "default",
        RenderImageCommand(
            request_id="edges",
            op="render.image",
            output_path=str(tmp_path / "edges.png"),
            style=RenderStyle.SHADED_EDGES,
        ),
    )
    files = SessionFiles(root, "default")
    rendered_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert rendered_state["render_style"]["style"] == "shaded_edges"

    getattr(manager, termination)("default")
    assert services[0].closed is (termination == "close")
    assert services[0].killed is (termination == "kill")

    recovered = SessionManager(root, service_factory=factory)
    receipt = recovered.execute(
        "default",
        SessionSnapshotCommand(request_id="after-kill", op="session.snapshot"),
    )
    payload = recovered.raw_result(receipt)
    assert isinstance(payload, dict)
    session = payload["result"]["session"]
    assert session["components"][component_id]["display"] == "hidden"
    recovered_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert recovered_state["render_style"] == RenderStyleState().model_dump(mode="json")
    recovered.close("default")
    assert services[-1].closed


def test_recovery_reopens_the_worker_with_the_checkpointed_aspect_ratio(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("default", source, aspect_ratio=2.5)
    manager.kill("default")

    recovered = SessionManager(root, service_factory=factory)
    recovered.execute(
        "default",
        SessionSnapshotCommand(request_id="after-kill", op="session.snapshot"),
    )

    recover_open = services[-1].received_commands[0]
    assert isinstance(recover_open, SceneOpenCommand)
    assert recover_open.aspect_ratio == 2.5


def test_recovery_reopens_with_a_later_session_reset_aspect_ratio(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("default", source, aspect_ratio=2.5)
    manager.execute(
        "default",
        SessionResetCommand(request_id="reset", op="session.reset", aspect_ratio=0.5),
    )
    manager.kill("default")

    recovered = SessionManager(root, service_factory=factory)
    recovered.execute(
        "default",
        SessionSnapshotCommand(request_id="after-kill", op="session.snapshot"),
    )

    recover_open = services[-1].received_commands[0]
    assert isinstance(recover_open, SceneOpenCommand)
    assert recover_open.aspect_ratio == 0.5


def test_bare_reset_preserves_the_checkpointed_open_aspect_ratio(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("default", source, aspect_ratio=2.5)
    # A reset that omits --aspect-ratio must not clobber the open aspect with 1.0.
    manager.execute(
        "default",
        SessionResetCommand(request_id="reset", op="session.reset"),
    )
    manager.kill("default")

    recovered = SessionManager(root, service_factory=factory)
    recovered.execute(
        "default",
        SessionSnapshotCommand(request_id="after-kill", op="session.snapshot"),
    )

    recover_open = services[-1].received_commands[0]
    assert isinstance(recover_open, SceneOpenCommand)
    assert recover_open.aspect_ratio == 2.5


def test_bare_reset_stays_omitted_in_durable_events(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    def factory() -> FakeSessionService:
        return FakeSessionService(scene_manifest)

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("default", source, aspect_ratio=2.5)
    manager.execute("default", SessionResetCommand(request_id="reset", op="session.reset"))

    events = [
        json.loads(line)
        for line in (root / "sessions" / "default" / "events.jsonl").read_text().splitlines()
    ]
    assert "aspect_ratio" not in events[-1]["command"]


@pytest.mark.parametrize("schema_version", (1, 2, 3))
def test_checkpoint_without_framing_marker_restores_the_source_camera_before_replaying(
    tmp_path: Path, scene_manifest: SceneManifest, schema_version: int
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("review", source)
    manager.close("review")
    checkpoint_path = root / "sessions" / "review" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["schema_version"] = schema_version
    checkpoint.pop("aspect_ratio")
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    recovered = SessionManager(root, service_factory=factory)
    recovered.execute(
        "review",
        SessionSnapshotCommand(request_id="recover", op="session.snapshot"),
    )

    recovered_commands = services[-1].received_commands
    assert isinstance(recovered_commands[0], SceneOpenCommand)
    assert isinstance(recovered_commands[1], ViewSetCommand)
    assert recovered_commands[1].camera == scene_manifest.imported_camera

    upgraded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert upgraded["schema_version"] == 3
    assert upgraded["accepted_commands"] == []
    assert upgraded["replay_prefix"][0] == {
        "request_id": "recover-v1-source-camera",
        "op": "view.set",
        "camera": scene_manifest.imported_camera.model_dump(mode="json"),
        "focus_component_ids": [],
        "aspect_ratio": 1.0,
    }

    recovered.kill("review")
    restarted = SessionManager(root, service_factory=factory)
    restarted.execute(
        "review",
        SessionSnapshotCommand(request_id="recover-again", op="session.snapshot"),
    )
    restarted_commands = services[-1].received_commands
    assert isinstance(restarted_commands[0], SceneOpenCommand)
    assert isinstance(restarted_commands[1], ViewSetCommand)
    assert restarted_commands[1].camera == scene_manifest.imported_camera


def test_failed_checkpoint_upgrade_kills_untracked_recovery_worker(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("review", source)
    manager.close("review")
    checkpoint_path = root / "sessions" / "review" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["schema_version"] = 1
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    recovered = SessionManager(root, service_factory=factory)

    def fail_checkpoint_write(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("meshprobe.workspace.atomic_json", fail_checkpoint_write)
    with pytest.raises(OSError, match="disk full"):
        recovered.execute(
            "review",
            SessionSnapshotCommand(request_id="recover", op="session.snapshot"),
        )

    assert services[-1].killed
    assert recovered._services == {}


def test_manager_lists_resolves_and_stops_all_sessions(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    first = tmp_path / "first.glb"
    second = tmp_path / "second.glb"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    manager.open("alpha", first)
    manager.open("beta", second)

    listed = manager.list_sessions()
    component_id = manager.resolve_component("alpha", scene_manifest.components[0].path)

    assert [item["name"] for item in listed] == ["alpha", "beta"]
    assert component_id == scene_manifest.components[0].id
    with pytest.raises(ValueError, match="unknown component"):
        manager.resolve_component("alpha", "c999")
    closed = manager.close_all()
    assert [item.session for item in closed] == ["alpha", "beta"]
    assert all(service.closed for service in services)

    recovered = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    recovered.execute(
        "alpha",
        SessionSnapshotCommand(request_id="recover", op="session.snapshot"),
    )
    killed = recovered.kill_all()
    assert [item.session for item in killed] == ["alpha", "beta"]
    assert services[-1].killed


def test_component_resolution_prefers_short_refs_over_paths(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    manager.open("review", source)
    files = SessionFiles(manager.root, "review")
    payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
    path_collision = payload["components"][0]
    ref_target = payload["components"][1]
    path_collision["path"] = "c2"
    ref_target["ref"] = "c2"
    files.components.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert manager.resolve_component("review", "c2") == ref_target["id"]


def test_ambiguous_component_name_reports_matching_paths(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    manager.open("review", source)
    files = SessionFiles(manager.root, "review")
    payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
    payload["components"][1]["name"] = "duplicate"
    payload["components"][2]["name"] = "duplicate"
    files.components.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous component display name") as error:
        manager.resolve_component("review", "duplicate")

    assert "assembly/cover/clip" in str(error.value)
    assert "assembly/drive/idler" in str(error.value)
    assert "exact path or stable ID" in str(error.value)


def test_resolve_component_ids_expands_globs_and_falls_back_to_exact(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    manager.open("review", source)
    by_path = {component.path: component.id for component in scene_manifest.components}

    # A bare glob is anchored like `find`, matches at any depth, and returns every
    # match ordered by path.
    assert manager.resolve_component_ids("review", "**/*") == (
        by_path["assembly"],
        by_path["assembly/cover/clip"],
        by_path["assembly/drive/idler"],
    )
    assert manager.resolve_component_ids("review", "idler") == (by_path["assembly/drive/idler"],)
    assert manager.resolve_component_ids("review", "assembly/*/clip") == (
        by_path["assembly/cover/clip"],
    )

    # Non-glob tokens keep exact ref/ID/name/path resolution, one id each.
    assert manager.resolve_component_ids("review", "c2") == (by_path["assembly/cover/clip"],)
    assert manager.resolve_component_ids("review", "assembly/drive/idler") == (
        by_path["assembly/drive/idler"],
    )

    with pytest.raises(ValueError, match="no components match glob pattern"):
        manager.resolve_component_ids("review", "**/missing*")
    with pytest.raises(ValueError, match="unknown component"):
        manager.resolve_component_ids("review", "c999")


def test_resolve_component_ids_prefers_exact_names_with_glob_metacharacters(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    manager.open("review", source)
    files = SessionFiles(manager.root, "review")
    payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
    payload["components"][1]["name"] = "panel[1]"
    payload["components"][2]["path"] = "assembly/drive/gear?"
    files.components.write_text(yaml.safe_dump(payload), encoding="utf-8")

    # A literal name/path that happens to contain glob metacharacters is matched
    # exactly, not interpreted as a pattern.
    assert manager.resolve_component_ids("review", "panel[1]") == (payload["components"][1]["id"],)
    assert manager.resolve_component_ids("review", "assembly/drive/gear?") == (
        payload["components"][2]["id"],
    )


def test_resolve_component_ids_reports_ambiguous_glob_looking_names(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: FakeSessionService(scene_manifest),
    )
    manager.open("review", source)
    files = SessionFiles(manager.root, "review")
    payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
    payload["components"][1]["name"] = "panel[1]"
    payload["components"][2]["name"] = "panel[1]"
    files.components.write_text(yaml.safe_dump(payload), encoding="utf-8")

    # A glob-looking literal shared by several components stays an ambiguous-name
    # error instead of being silently reinterpreted as a glob pattern.
    with pytest.raises(ValueError, match="ambiguous component display name") as error:
        manager.resolve_component_ids("review", "panel[1]")
    assert "assembly/cover/clip" in str(error.value)
    assert "assembly/drive/idler" in str(error.value)


def test_reset_clears_replay_and_artifact_detection_is_render_only(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    service = FakeSessionService(scene_manifest)
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(
        tmp_path / ".meshprobe",
        service_factory=lambda: service,
    )
    manager.open("default", source)
    component_id = manager.resolve_component("default", "c1")
    manager.execute(
        "default",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.execute(
        "default",
        RenderImageCommand(
            request_id="edges",
            op="render.image",
            output_path=str(tmp_path / "edges.png"),
            style=RenderStyle.SHADED_EDGES,
        ),
    )
    files = SessionFiles(manager.root, "default")
    rendered_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))
    assert rendered_state["render_style"]["style"] == "shaded_edges"

    manager.execute(
        "default",
        SessionResetCommand(request_id="reset", op="session.reset"),
    )
    checkpoint = json.loads(files.checkpoint.read_text(encoding="utf-8"))
    reset_state = yaml.safe_load(files.state.read_text(encoding="utf-8"))

    assert checkpoint["accepted_commands"] == []
    assert reset_state["render_style"] == RenderStyleState().model_dump(mode="json")
    assert SessionManager._artifact_paths("session.snapshot", {"path": "component/path"}) == ()
    assert SessionManager._artifact_paths(
        "render.image",
        {"color": {"path": "image.png", "sha256": "a" * 64, "bytes": 10}},
    ) == ("image.png",)
    manager.shutdown(force=True)
    assert service.killed


def test_render_warnings_extracts_image_and_sheet_warnings() -> None:
    assert SessionManager._render_warnings("render.image", {"warnings": ["blank"]}) == ("blank",)
    assert SessionManager._render_warnings("render.image", {"warnings": []}) == ()
    assert SessionManager._render_warnings("component.display", {"warnings": ["ignored"]}) == ()
    assert SessionManager._render_warnings("render.image", "not-a-dict") == ()
    sheet = {
        "panels": [
            {"render": {"warnings": ["blank"]}},
            {"render": {"warnings": ["blank", "askew"]}},
            {"render": {"warnings": []}},
            {"caption": "no render key"},
        ]
    }
    assert SessionManager._render_warnings("render.contact_sheet", sheet) == ("blank", "askew")


@pytest.mark.parametrize(
    ("render_warnings", "expected"),
    [(["frame is empty"], ("frame is empty",)), ([], ())],
)
def test_render_receipt_surfaces_worker_warnings(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    render_warnings: list[JsonValue],
    expected: tuple[str, ...],
) -> None:
    class WarningService(FakeSessionService):
        def execute(self, command: Command) -> CommandResponse:
            if isinstance(command, RenderImageCommand):
                assert self.session is not None
                return CommandResponse(
                    request_id=command.request_id,
                    op=command.op,
                    result={"warnings": render_warnings},
                )
            return super().execute(command)

    service = WarningService(scene_manifest)
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=lambda: service)
    manager.open("review", source)
    receipt = manager.execute(
        "review",
        RenderImageCommand(
            request_id="render",
            op="render.image",
            output_path=str(tmp_path / "frame.png"),
        ),
    )
    assert receipt.warnings == expected


def test_invalid_session_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session name"):
        SessionFiles(tmp_path / ".meshprobe", "bad/name")


def test_renderer_choice_is_persisted_and_reused_for_recovery(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[str | None] = []

    def service(*, blender: str | None, timeout_seconds: float) -> FakeSessionService:
        selected.append(blender)
        return FakeSessionService(scene_manifest)

    monkeypatch.setattr("meshprobe.workspace.MeshProbeService", service)
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root)
    manager.open("review", source, blender="/opt/pinned/blender")
    manager.close("review")

    recovered = SessionManager(root, blender="/different/default")
    recovered.execute(
        "review",
        SessionSnapshotCommand(request_id="recover", op="session.snapshot"),
    )
    metadata = json.loads(
        (root / "sessions" / "review" / "metadata.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (root / "sessions" / "review" / "checkpoint.json").read_text(encoding="utf-8")
    )

    assert selected == ["/opt/pinned/blender", "/opt/pinned/blender"]
    assert metadata["blender"] == "/opt/pinned/blender"
    assert checkpoint["blender"] == "/opt/pinned/blender"


@pytest.mark.skipif(os.name == "nt", reason="Windows does not expose POSIX file mode bits")
def test_atomic_json_enforces_private_mode(tmp_path: Path) -> None:
    path = tmp_path / "daemon.json"

    atomic_json(path, {"token": "secret"}, mode=0o600)

    assert path.stat().st_mode & 0o777 == 0o600
