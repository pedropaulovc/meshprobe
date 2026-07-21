from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import JsonValue

import meshprobe.evals.harness.broker as broker_module
from meshprobe.evals.harness.broker import (
    EvaluationBroker,
    _comparison_artifact_virtual_path,
    _contact_sheet_minimum_output_reservation,
    _render_output_reservation,
)
from meshprobe.evals.harness.sandbox import visible_artifact_path, visible_input_path
from meshprobe.evals.schemas import EpisodeBudgets, Operation, TraceStatus
from meshprobe.models import OrbitSweep
from meshprobe.protocol import (
    Command,
    ComponentOcclusionCommand,
    IlluminationSetCommand,
    RenderComparisonRequest,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionResetCommand,
    SessionSnapshotCommand,
    SessionUndoCommand,
    ViewFrameCommand,
    ViewRotateCommand,
)
from meshprobe.service import CommandResponse
from meshprobe.sources import sha256_file


class FakeEvaluationService:
    def __init__(self) -> None:
        self.commands: list[object] = []
        self.evaluator_directories: list[str] = []

    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        self.commands.append(command)
        self.evaluator_directories.append(evaluator_output_dir)
        request_id = command.request_id
        operation = command.op
        if operation == "scene.open":
            result: JsonValue = {
                "source_sha256": "a" * 64,
                "normalized_geometry": {
                    "path": "/private/cache/normalized.glb",
                    "sha256": "b" * 64,
                },
            }
        elif operation == "session.snapshot":
            result = {"session": {"state_sha256": "1" * 64}}
        elif operation in {"illumination.set", "session.reset", "view.rotate", "view.frame"}:
            result = {"state_sha256": "2" * 64}
        elif operation == "component.occlusion":
            result = {
                "camera_source": "current_session",
                "visible_fraction": 0.5,
                "state_sha256": "1" * 64,
            }
        elif operation == "render.image":
            assert isinstance(command, RenderImageCommand)
            output_path = command.output_path
            color_path = Path(output_path)
            color_path.parent.mkdir(parents=True, exist_ok=True)
            color_path.write_bytes(b"c" * 12)
            component_path = Path(evaluator_output_dir) / "components.png"
            component_path.parent.mkdir(parents=True, exist_ok=True)
            component_path.write_bytes(b"m" * 8)
            result = {
                "state_sha256": "2" * 64,
                "color": {"path": output_path, "bytes": 12},
                "evaluator": {
                    "component_ids": {
                        "path": str(component_path),
                        "bytes": 8,
                    }
                },
            }
            if command.comparison is not None:
                reference_path = Path(command.comparison.reference_image_path)
                assert reference_path.is_file()
                comparison_path = Path(command.comparison.output_path)
                comparison_path.parent.mkdir(parents=True, exist_ok=True)
                comparison_path.write_bytes(b"d" * 16)
                result["comparison"] = {
                    "artifact": {"path": str(comparison_path), "bytes": 16},
                    "reference": {"path": str(reference_path)},
                }
        else:
            raise AssertionError(operation)
        return CommandResponse(request_id=request_id, op=operation, result=result)


def broker(
    tmp_path: Path,
    *,
    budgets: EpisodeBudgets | None = None,
    service: FakeEvaluationService | None = None,
) -> EvaluationBroker:
    model = tmp_path / "input" / "model.glb"
    model.parent.mkdir()
    model.write_bytes(b"model")
    return EvaluationBroker(
        service=service or FakeEvaluationService(),
        model_path=model,
        artifact_root=tmp_path / "agent" / "artifacts",
        evaluator_root=tmp_path / "evaluator" / "passes",
        trace_path=tmp_path / "evaluator" / "trace.jsonl",
        budgets=budgets or EpisodeBudgets(),
    )


def test_broker_translates_paths_redacts_private_passes_and_checkpoints(tmp_path: Path) -> None:
    active = broker(tmp_path)
    assert active.execute(
        SceneOpenCommand(
            request_id="open",
            op="scene.open",
            source_path=active.visible_model_path,
        )
    ).ok
    assert active.execute(SessionSnapshotCommand(request_id="describe", op="session.snapshot")).ok
    assert active.execute(
        IlluminationSetCommand(
            request_id="light",
            op="illumination.set",
            illumination={"preset": "raking_left"},  # type: ignore[arg-type]
        )
    ).ok
    rendered = active.execute(
        RenderImageCommand(
            request_id="render",
            op="render.image",
            output_path="views/evidence.png",
            width=64,
            height=64,
            samples=1,
        )
    )

    assert rendered.ok and rendered.response is not None
    artifact_path = tmp_path / "agent" / "artifacts" / "views" / "evidence.png"
    visible_path = (
        str(artifact_path.resolve())
        if os.name == "nt"
        else "/workspace/artifacts/views/evidence.png"
    )
    assert rendered.response.result == {
        "state_sha256": "2" * 64,
        "color": {
            "path": visible_path,
            "bytes": 12,
        },
    }
    assert active.events[-1].result != rendered.response.result
    assert active.events[-1].state_before_sha256 == "2" * 64
    assert active.metrics.renders == 1
    assert active.metrics.total_pixels == 4_096
    assert active.metrics.output_bytes == 20
    assert artifact_path.read_bytes() == b"c" * 12
    trace = (tmp_path / "evaluator" / "trace.jsonl").read_text(encoding="utf-8")
    assert len(trace.splitlines()) == 4
    assert json.loads(trace.splitlines()[-1])["status"] == "accepted"


def test_broker_caps_render_timeout_to_the_remaining_episode_wall_budget(
    tmp_path: Path,
) -> None:
    service = FakeEvaluationService()
    active = broker(
        tmp_path,
        service=service,
        budgets=EpisodeBudgets(wall_seconds=30),
    )

    rendered = active.execute(
        RenderImageCommand(
            request_id="bounded-render",
            op="render.image",
            output_path="bounded.png",
            width=64,
            height=64,
            timeout_seconds=86_400,
        )
    )

    assert rendered.ok
    translated = service.commands[-1]
    assert isinstance(translated, RenderImageCommand)
    assert 0 < translated.timeout_seconds <= 30
    assert active.events[-1].arguments["timeout_seconds"] == 86_400


def test_broker_translates_comparison_paths_and_charges_the_second_artifact(
    tmp_path: Path,
) -> None:
    service = FakeEvaluationService()
    active = broker(tmp_path, service=service)
    reference = tmp_path / "input" / "historical.png"
    reference.write_bytes(b"reference")

    rendered = active.execute(
        RenderImageCommand(
            request_id="comparison",
            op="render.image",
            output_path="cad.png",
            width=64,
            height=64,
            comparison=RenderComparisonRequest(
                reference_image_path=visible_input_path(reference),
                mode="side_by_side",
                output_path="comparison.png",
            ),
        )
    )

    assert rendered.ok and rendered.response is not None
    translated = service.commands[-1]
    assert isinstance(translated, RenderImageCommand)
    assert translated.comparison is not None
    assert translated.comparison.reference_image_path == str(reference.resolve())
    assert Path(translated.comparison.output_path).is_relative_to(tmp_path / "evaluator" / "passes")
    assert isinstance(rendered.response.result, dict)
    comparison = rendered.response.result["comparison"]
    assert isinstance(comparison, dict)
    artifact = comparison["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["path"] == visible_artifact_path(
        tmp_path / "agent" / "artifacts" / "comparison.png"
    )
    reference_manifest = comparison["reference"]
    assert isinstance(reference_manifest, dict)
    assert reference_manifest["path"] == visible_input_path(reference)
    private_result = active.events[-1].result
    assert isinstance(private_result, dict)
    private_comparison = private_result["comparison"]
    assert isinstance(private_comparison, dict)
    private_artifact = private_comparison["artifact"]
    assert isinstance(private_artifact, dict)
    assert private_artifact["path"] == str(tmp_path / "agent" / "artifacts" / "comparison.png")
    assert active.metrics.output_bytes == 36


def test_broker_charges_orbit_sweeps_by_their_actual_panel_count(tmp_path: Path) -> None:
    active = broker(
        tmp_path,
        budgets=EpisodeBudgets(tool_calls=1, renders=1, total_pixels=128 * 128, output_bytes=10**9),
    )
    command = RenderContactSheetCommand(
        request_id="sweep",
        op="render.contact_sheet",
        output_path="sweep.png",
        recipe="orbit_sweep",
        orbit_sweep=OrbitSweep(azimuth_degrees=(0,), elevation_degrees=(0,)),
        panel_width=128,
        panel_height=128,
        samples=1,
    )

    _translated, renders, pixels, _bytes, _staging = active._translate_and_reserve(command, 1)

    assert renders == 1
    assert pixels == 128 * 128


def test_broker_reserves_orbit_sweep_bytes_by_actual_panel_count(tmp_path: Path) -> None:
    one_panel_reservation = _contact_sheet_minimum_output_reservation(128, 128, 1)
    assert one_panel_reservation < _contact_sheet_minimum_output_reservation(128, 128, 9)
    active = broker(
        tmp_path,
        budgets=EpisodeBudgets(
            tool_calls=1,
            renders=1,
            total_pixels=128 * 128,
            output_bytes=one_panel_reservation,
        ),
    )
    command = RenderContactSheetCommand(
        request_id="one-panel-sweep",
        op="render.contact_sheet",
        output_path="sweep.png",
        recipe="orbit_sweep",
        orbit_sweep=OrbitSweep(azimuth_degrees=(0,), elevation_degrees=(0,)),
        panel_width=128,
        panel_height=128,
        samples=1,
    )

    _translated, renders, pixels, byte_reservation, _staging = active._translate_and_reserve(
        command, 1
    )

    assert (renders, pixels, byte_reservation) == (1, 128 * 128, one_panel_reservation)


def test_broker_records_occlusion_queries(tmp_path: Path) -> None:
    active = broker(tmp_path)
    active.execute(
        SceneOpenCommand(
            request_id="open",
            op="scene.open",
            source_path=active.visible_model_path,
        )
    )

    result = active.execute(
        ComponentOcclusionCommand(
            request_id="occlusion",
            op="component.occlusion",
            component_ids=("target",),
        )
    )

    assert result.ok
    assert active.events[-1].operation.value == "component.occlusion"


def test_broker_omits_normalized_geometry_cache_metadata(tmp_path: Path) -> None:
    active = broker(tmp_path)

    opened = active.execute(
        SceneOpenCommand(
            request_id="open",
            op="scene.open",
            source_path=active.visible_model_path,
        )
    )

    assert opened.ok and opened.response is not None
    assert opened.response.result == {"source_sha256": "a" * 64}


def test_broker_records_view_rotate_as_a_normal_trace_event(tmp_path: Path) -> None:
    active = broker(tmp_path)

    rotated = active.execute(
        ViewRotateCommand(
            request_id="rotate",
            op="view.rotate",
            target_mm=(0, 0, 0),
            axis="z",
            degrees=15,
        )
    )

    assert rotated.ok
    assert active.events[-1].operation is Operation.VIEW_ROTATE
    assert active.events[-1].status is TraceStatus.ACCEPTED


def test_broker_trace_preserves_an_omitted_reset_aspect_ratio(tmp_path: Path) -> None:
    active = broker(tmp_path)

    reset = active.execute(SessionResetCommand(request_id="reset", op="session.reset"))

    assert reset.ok
    arguments = active.events[-1].arguments
    assert isinstance(arguments, dict)
    assert "aspect_ratio" not in arguments


def test_broker_records_view_frame_as_a_normal_trace_event(tmp_path: Path) -> None:
    active = broker(tmp_path)

    framed = active.execute(
        ViewFrameCommand(
            request_id="frame",
            op="view.frame",
            focus_component_ids=("component-1",),
        )
    )

    assert framed.ok
    assert active.events[-1].operation is Operation.VIEW_FRAME
    assert active.events[-1].status is TraceStatus.ACCEPTED


def test_broker_rejects_but_traces_durable_undo(tmp_path: Path) -> None:
    active = broker(tmp_path)

    undone = active.execute(SessionUndoCommand(request_id="undo", op="session.undo"))

    assert not undone.ok
    assert undone.error is not None
    assert undone.error.code == "broker.unsupported_operation"
    assert active.events[-1].operation is Operation.SESSION_UNDO
    assert active.events[-1].status is TraceStatus.REJECTED


def test_broker_mediates_custom_environment_map_paths(tmp_path: Path) -> None:
    service = FakeEvaluationService()
    active = broker(tmp_path, service=service)
    environment_map = tmp_path / "input" / "studio.hdr"
    environment_map.write_bytes(b"hdr")
    visible_path = (
        str(environment_map.resolve()) if os.name == "nt" else "/workspace/input/studio.hdr"
    )

    accepted = active.execute(
        IlluminationSetCommand.model_validate(
            {
                "request_id": "environment",
                "op": "illumination.set",
                "illumination": {
                    "preset": "custom",
                    "background_rgb": [1, 1, 1],
                    "ambient_strength": 1,
                    "environment_map": {
                        "path": visible_path,
                        "sha256": sha256_file(environment_map),
                    },
                },
            }
        )
    )

    assert accepted.ok
    translated = service.commands[-1]
    assert isinstance(translated, IlluminationSetCommand)
    assert translated.illumination.environment_map is not None  # type: ignore[union-attr]
    assert translated.illumination.environment_map.path == str(environment_map.resolve())  # type: ignore[union-attr]

    rejected = active.execute(
        IlluminationSetCommand.model_validate(
            {
                "request_id": "escaped-environment",
                "op": "illumination.set",
                "illumination": {
                    "preset": "custom",
                    "background_rgb": [1, 1, 1],
                    "ambient_strength": 1,
                    "environment_map": {
                        "path": str((tmp_path / "outside.hdr").resolve()),
                        "sha256": "a" * 64,
                    },
                },
            }
        )
    )
    assert rejected.error is not None
    assert rejected.error.code == "broker.input_path"

    missing = active.execute(
        IlluminationSetCommand.model_validate(
            {
                "request_id": "missing-environment",
                "op": "illumination.set",
                "illumination": {
                    "preset": "custom",
                    "background_rgb": [1, 1, 1],
                    "ambient_strength": 1,
                    "environment_map": {
                        "path": (
                            str((tmp_path / "input" / "missing.hdr").resolve())
                            if os.name == "nt"
                            else "/workspace/input/missing.hdr"
                        ),
                        "sha256": "a" * 64,
                    },
                },
            }
        )
    )
    assert missing.error is not None
    assert missing.error.code == "broker.input_path"


def test_broker_rejects_wrong_model_output_escape_duplicates_and_budgets(tmp_path: Path) -> None:
    active = broker(
        tmp_path,
        budgets=EpisodeBudgets(tool_calls=3, renders=1, total_pixels=4_096),
    )
    wrong_model = active.execute(
        SceneOpenCommand(
            request_id="wrong",
            op="scene.open",
            source_path="/workspace/input/other.glb",
        )
    )
    duplicate = active.execute(
        SceneOpenCommand(
            request_id="wrong",
            op="scene.open",
            source_path=active.visible_model_path,
        )
    )
    escaped = active.execute(
        RenderImageCommand(
            request_id="escape",
            op="render.image",
            output_path="../outside.png",
            width=64,
            height=64,
        )
    )
    exhausted = active.execute(
        SessionSnapshotCommand(request_id="over-budget", op="session.snapshot")
    )

    assert wrong_model.error is not None and wrong_model.error.code == "broker.model_path"
    assert duplicate.error is not None and duplicate.error.code == "broker.duplicate_request"
    assert escaped.error is not None and escaped.error.code == "broker.output_path"
    assert exhausted.error is not None and exhausted.error.code == "budget.tool_calls"
    assert all(event.status is TraceStatus.REJECTED for event in active.events)
    assert len(active.public_errors) == 4


def test_broker_reserves_output_bytes_before_calling_renderer(tmp_path: Path) -> None:
    service = FakeEvaluationService()
    active = broker(
        tmp_path,
        service=service,
        budgets=EpisodeBudgets(
            tool_calls=2,
            renders=1,
            total_pixels=4_096,
            output_bytes=1,
        ),
    )

    rejected = active.execute(
        RenderImageCommand(
            request_id="too-large",
            op="render.image",
            output_path="evidence.png",
            width=64,
            height=64,
            samples=1,
        )
    )

    assert rejected.error is not None
    assert rejected.error.code == "budget.output_bytes"
    assert service.commands == []
    assert active.metrics.output_bytes == 0
    assert not (tmp_path / "agent" / "artifacts" / "evidence.png").exists()


def test_broker_rejects_undersized_comparison_before_calling_renderer(tmp_path: Path) -> None:
    service = FakeEvaluationService()
    required = _render_output_reservation(64 * 64)
    active = broker(
        tmp_path,
        service=service,
        budgets=EpisodeBudgets(
            tool_calls=1,
            renders=1,
            total_pixels=64 * 64,
            output_bytes=required - 1,
        ),
    )
    reference = tmp_path / "input" / "reference.png"
    reference.write_bytes(b"reference")

    rejected = active.execute(
        RenderImageCommand(
            request_id="comparison-too-large",
            op="render.image",
            output_path="evidence.png",
            width=64,
            height=64,
            comparison=RenderComparisonRequest(
                reference_image_path=visible_input_path(reference),
                mode="side_by_side",
                output_path="comparison.png",
            ),
        )
    )

    assert rejected.error is not None
    assert rejected.error.code == "budget.output_bytes"
    assert service.commands == []


def test_comparison_artifact_path_stays_host_visible_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact = artifact_root / "comparison.png"
    monkeypatch.setattr(broker_module, "os", SimpleNamespace(name="nt"))

    visible = _comparison_artifact_virtual_path(
        {"comparison": {"artifact": {"path": str(artifact)}}}, artifact_root
    )

    assert visible == str(artifact)


def test_broker_reserves_remaining_bytes_for_dynamic_contact_sheet_captions(
    tmp_path: Path,
) -> None:
    output_budget = 123_456_789
    active = broker(
        tmp_path,
        budgets=EpisodeBudgets(
            tool_calls=1,
            renders=9,
            total_pixels=9 * 128 * 128,
            output_bytes=output_budget,
        ),
    )
    command = RenderContactSheetCommand(
        request_id="sheet",
        op="render.contact_sheet",
        output_path="sheet.png",
        focus_component_ids=("target",),
        panel_width=128,
        panel_height=128,
        samples=1,
    )

    _translated, _renders, _pixels, byte_reservation, _staging = active._translate_and_reserve(
        command, 1
    )

    assert byte_reservation == output_budget


def test_broker_rejects_undersized_contact_sheet_budget_before_rendering(
    tmp_path: Path,
) -> None:
    service = FakeEvaluationService()
    active = broker(
        tmp_path,
        service=service,
        budgets=EpisodeBudgets(
            tool_calls=2,
            renders=18,
            total_pixels=18 * 128 * 128,
            output_bytes=1,
        ),
    )

    for sequence in range(2):
        rejected = active.execute(
            RenderContactSheetCommand(
                request_id=f"undersized-{sequence}",
                op="render.contact_sheet",
                output_path=f"sheet-{sequence}.png",
                focus_component_ids=("target",),
                panel_width=128,
                panel_height=128,
                samples=1,
            )
        )
        assert rejected.error is not None
        assert rejected.error.code == "budget.output_bytes"

    assert service.commands == []
    assert active.metrics.renders == 0
    assert active.metrics.total_pixels == 0
    assert active.metrics.output_bytes == 0


def test_default_budget_fits_default_resolution_evidence_workflow() -> None:
    # A default render-image plus a default contact sheet must fit inside the
    # default output-byte budget under the harness's conservative per-render
    # reservations. Otherwise an agent following the standard render +
    # contact-sheet evidence flow trips budget.output_bytes purely from the
    # inspection-grade default resolutions (issue #57).
    image = RenderImageCommand(request_id="img", op="render.image", output_path="evidence.png")
    sheet = RenderContactSheetCommand(
        request_id="sheet",
        op="render.contact_sheet",
        output_path="sheet.png",
        focus_component_ids=("target",),
    )
    workflow_reservation = _render_output_reservation(
        image.width * image.height
    ) + _contact_sheet_minimum_output_reservation(sheet.panel_width, sheet.panel_height)

    assert workflow_reservation <= EpisodeBudgets().output_bytes


def test_broker_accepts_the_sandbox_visible_artifact_prefix(tmp_path: Path) -> None:
    active = broker(tmp_path)
    artifact = tmp_path / "agent" / "artifacts" / "visible.png"

    rendered = active.execute(
        RenderImageCommand(
            request_id="visible-output",
            op="render.image",
            output_path=visible_artifact_path(artifact),
            width=64,
            height=64,
            samples=1,
        )
    )

    assert rendered.ok
    assert artifact.is_file()


class PrivatePathErrorService(FakeEvaluationService):
    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        del command
        raise ValueError(f"private pass failed at {evaluator_output_dir}/depth.exr")


def test_broker_redacts_private_paths_from_tool_errors(tmp_path: Path) -> None:
    active = broker(tmp_path, service=PrivatePathErrorService())

    rejected = active.execute(
        SessionSnapshotCommand(request_id="private-error", op="session.snapshot")
    )

    assert rejected.error is not None
    assert "<private evaluator path>" in rejected.error.message
    assert str(tmp_path / "evaluator") not in rejected.error.message


class OversizeEvaluationService(FakeEvaluationService):
    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        assert isinstance(command, RenderImageCommand)
        output_path = Path(command.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x" * (5 * 1024**2))
        return CommandResponse(
            request_id=command.request_id,
            op=command.op,
            result={
                "state_sha256": "2" * 64,
                "color": {"path": str(output_path), "bytes": output_path.stat().st_size},
            },
        )


def test_broker_discards_output_that_exceeds_reserved_bound(tmp_path: Path) -> None:
    active = broker(
        tmp_path,
        service=OversizeEvaluationService(),
        budgets=EpisodeBudgets(
            tool_calls=2,
            renders=1,
            total_pixels=4_096,
            output_bytes=10 * 1024**2,
        ),
    )

    rejected = active.execute(
        RenderImageCommand(
            request_id="unexpected-size",
            op="render.image",
            output_path="evidence.png",
            width=64,
            height=64,
            samples=1,
        )
    )

    assert rejected.error is not None
    assert rejected.error.code == "budget.output_bytes"
    assert active.metrics.renders == 1
    assert active.metrics.total_pixels == 4_096
    assert active.metrics.output_bytes == 0
    assert not (tmp_path / "agent" / "artifacts" / "evidence.png").exists()
    assert not any((tmp_path / "agent" / "artifacts").rglob("*.png"))


class DestinationSwapService(FakeEvaluationService):
    def __init__(self, artifact_root: Path, outside: Path) -> None:
        super().__init__()
        self._artifact_root = artifact_root
        self._outside = outside

    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        response = super().execute_for_evaluation(
            command,
            evaluator_output_dir=evaluator_output_dir,
        )
        self._outside.mkdir()
        (self._artifact_root / "views").symlink_to(self._outside, target_is_directory=True)
        return response


@pytest.mark.skipif(os.name != "posix", reason="dirfd publication is POSIX-specific")
def test_broker_rejects_destination_parent_swapped_to_symlink(tmp_path: Path) -> None:
    artifact_root = tmp_path / "agent" / "artifacts"
    service = DestinationSwapService(artifact_root, tmp_path / "outside")
    active = broker(tmp_path, service=service)

    rejected = active.execute(
        RenderImageCommand(
            request_id="destination-swap",
            op="render.image",
            output_path="views/evidence.png",
            width=64,
            height=64,
        )
    )

    assert rejected.error is not None
    assert rejected.error.code == "broker.output_path"
    assert not (tmp_path / "outside" / "evidence.png").exists()
