from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import JsonValue

from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.harness.sandbox import visible_artifact_path
from meshprobe.evals.schemas import EpisodeBudgets, TraceStatus
from meshprobe.protocol import (
    Command,
    IlluminationSetCommand,
    RenderImageCommand,
    SceneDescribeCommand,
    SceneOpenCommand,
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
        elif operation == "scene.describe":
            result = {"session": {"state_sha256": "1" * 64}}
        elif operation == "illumination.set":
            result = {"state_sha256": "2" * 64}
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
    assert active.execute(SceneDescribeCommand(request_id="describe", op="scene.describe")).ok
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
    exhausted = active.execute(SceneDescribeCommand(request_id="over-budget", op="scene.describe"))

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

    rejected = active.execute(SceneDescribeCommand(request_id="private-error", op="scene.describe"))

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
