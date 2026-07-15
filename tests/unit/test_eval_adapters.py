from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import JsonValue

from meshprobe.evals.generators import (
    GeneratorFamily,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.harness.adapters import CliJsonlAdapter
from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.protocol import Command
from meshprobe.service import CommandResponse

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is required")


class AdapterService:
    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        del evaluator_output_dir
        result: JsonValue = {"source_sha256": "a" * 64}
        if command.op == "scene.describe":
            result = {"session": {"state_sha256": "1" * 64}}
        return CommandResponse(request_id=command.request_id, op=command.op, result=result)


def test_cli_adapter_runs_interactive_tool_calls_and_submission(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), public)
    episode = generate_episodes(model)[0]
    answer = episode.ground_truth.answer.model_dump(mode="json")
    script = public / "agent.py"
    script.write_text(
        "import json, sys\n"
        "episode = json.loads(sys.stdin.readline())\n"
        "def send(value):\n"
        "    print(json.dumps(value), flush=True)\n"
        "send({'type': 'tool_call', 'command': {"
        "'request_id': 'open', 'op': 'scene.open', "
        "'source_path': episode['model_path']}})\n"
        "json.loads(sys.stdin.readline())\n"
        "send({'type': 'tool_call', 'command': {"
        "'request_id': 'describe', 'op': 'scene.describe'}})\n"
        "json.loads(sys.stdin.readline())\n"
        f"answer = {answer!r}\n"
        "send({'type': 'submission', 'submission': {"
        "'episode_id': episode['episode_id'], 'answer': answer}})\n",
        encoding="utf-8",
    )
    broker = EvaluationBroker(
        service=AdapterService(),
        model_path=model.path,
        artifact_root=tmp_path / "agent-artifacts",
        evaluator_root=tmp_path / "evaluator" / "passes",
        trace_path=tmp_path / "evaluator" / "trace.jsonl",
        budgets=episode.spec.budgets,
    )

    run = CliJsonlAdapter(("/usr/bin/python3", "/workspace/input/agent.py")).run(
        spec=episode.spec,
        broker=broker,
        input_root=public,
        artifact_root=tmp_path / "agent-artifacts",
    )

    assert run.protocol_error is None
    assert run.returncode == 0
    assert run.submission is not None
    assert run.submission.answer == episode.ground_truth.answer
    assert [event.operation for event in broker.events] == ["scene.open", "scene.describe"]


def test_cli_adapter_reports_invalid_protocol_without_tool_execution(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), public)
    episode = generate_episodes(model)[0]
    script = public / "invalid.py"
    script.write_text("print('not-json', flush=True)\n", encoding="utf-8")
    broker = EvaluationBroker(
        service=AdapterService(),
        model_path=model.path,
        artifact_root=tmp_path / "agent-artifacts",
        evaluator_root=tmp_path / "evaluator" / "passes",
        trace_path=tmp_path / "evaluator" / "trace.jsonl",
        budgets=episode.spec.budgets,
    )

    run = CliJsonlAdapter(("/usr/bin/python3", "/workspace/input/invalid.py")).run(
        spec=episode.spec,
        broker=broker,
        input_root=public,
        artifact_root=tmp_path / "agent-artifacts",
    )

    assert run.submission is None
    assert run.protocol_error is not None and "invalid agent protocol" in run.protocol_error
    assert broker.events == ()
