from __future__ import annotations

import json
from pathlib import Path

from pydantic import JsonValue

from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.schemas import EpisodeBudgets, TraceStatus
from meshprobe.protocol import (
    Command,
    IlluminationSetCommand,
    RenderImageCommand,
    SceneDescribeCommand,
    SceneOpenCommand,
)
from meshprobe.service import CommandResponse


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
            result: JsonValue = {"source_sha256": "a" * 64}
        elif operation == "scene.describe":
            result = {"session": {"state_sha256": "1" * 64}}
        elif operation == "illumination.set":
            result = {"state_sha256": "2" * 64}
        elif operation == "render.image":
            assert isinstance(command, RenderImageCommand)
            output_path = command.output_path
            result = {
                "state_sha256": "2" * 64,
                "color": {"path": output_path, "bytes": 12},
                "evaluator": {
                    "component_ids": {
                        "path": f"{evaluator_output_dir}/components.png",
                        "bytes": 8,
                    }
                },
            }
        else:
            raise AssertionError(operation)
        return CommandResponse(request_id=request_id, op=operation, result=result)


def broker(tmp_path: Path, *, budgets: EpisodeBudgets | None = None) -> EvaluationBroker:
    model = tmp_path / "input" / "model.glb"
    model.parent.mkdir()
    model.write_bytes(b"model")
    return EvaluationBroker(
        service=FakeEvaluationService(),
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
    assert rendered.response.result == {
        "state_sha256": "2" * 64,
        "color": {"path": "/workspace/artifacts/views/evidence.png", "bytes": 12},
    }
    assert active.events[-1].result != rendered.response.result
    assert active.events[-1].state_before_sha256 == "2" * 64
    assert active.metrics.renders == 1
    assert active.metrics.total_pixels == 4_096
    assert active.metrics.output_bytes == 20
    trace = (tmp_path / "evaluator" / "trace.jsonl").read_text(encoding="utf-8")
    assert len(trace.splitlines()) == 4
    assert json.loads(trace.splitlines()[-1])["status"] == "accepted"


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
