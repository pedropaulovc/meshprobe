from __future__ import annotations

import io
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

from meshprobe.evals.generators import (
    GeneratorFamily,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.harness.adapters import CliJsonlAdapter, McpStdioAdapter
from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.harness.sandbox import visible_input_path
from meshprobe.evals.schemas import EpisodeBudgets
from meshprobe.protocol import Command
from meshprobe.service import CommandResponse

pytestmark = pytest.mark.skipif(
    os.name != "nt" and shutil.which("bwrap") is None,
    reason="a platform sandbox is required",
)


def agent_command(script: Path) -> tuple[str, str]:
    executable = sys.executable if os.name == "nt" else "/usr/bin/python3"
    return executable, visible_input_path(script)


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

    run = CliJsonlAdapter(agent_command(script)).run(
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

    run = CliJsonlAdapter(agent_command(script)).run(
        spec=episode.spec,
        broker=broker,
        input_root=public,
        artifact_root=tmp_path / "agent-artifacts",
    )

    assert run.submission is None
    assert run.protocol_error is not None and "invalid agent protocol" in run.protocol_error
    assert broker.events == ()


def test_mcp_adapter_serves_same_broker_contract_and_accepts_submission(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), public)
    episode = generate_episodes(model)[0]
    answer = episode.ground_truth.answer.model_dump(mode="json")
    script = public / "mcp_agent.py"
    script.write_text(
        "import json, sys\n"
        "def request(i, method, params=None):\n"
        "    value={'jsonrpc':'2.0','id':i,'method':method}\n"
        "    if params is not None: value['params']=params\n"
        "    print(json.dumps(value), flush=True)\n"
        "    return json.loads(sys.stdin.readline())\n"
        "init=request(1,'initialize',{'protocolVersion':'2025-06-18',"
        "'capabilities':{},'clientInfo':{'name':'test-agent','version':'1'}})\n"
        "episode=json.loads(init['result']['instructions'])\n"
        "print(json.dumps({'jsonrpc':'2.0','method':'notifications/initialized'}), flush=True)\n"
        "tools=request(2,'tools/list')['result']['tools']\n"
        "assert [tool['name'] for tool in tools] == ['meshprobe','submit']\n"
        "request(3,'tools/call',{'name':'meshprobe','arguments':{'command':{"
        "'request_id':'open','op':'scene.open','source_path':episode['model_path']}}})\n"
        "request(4,'tools/call',{'name':'meshprobe','arguments':{'command':{"
        "'request_id':'describe','op':'scene.describe'}}})\n"
        f"answer={answer!r}\n"
        "request(5,'tools/call',{'name':'submit','arguments':{"
        "'episode_id':episode['episode_id'],'answer':answer}})\n",
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

    run = McpStdioAdapter(agent_command(script)).run(
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


def test_mcp_adapter_json_rpc_validation_and_tool_error_paths(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), public)
    episode = generate_episodes(model)[0]
    broker = EvaluationBroker(
        service=AdapterService(),
        model_path=model.path,
        artifact_root=tmp_path / "agent-artifacts",
        evaluator_root=tmp_path / "evaluator" / "passes",
        trace_path=tmp_path / "evaluator" / "trace.jsonl",
        budgets=episode.spec.budgets,
    )
    stream = io.StringIO()

    def handle(payload: object, initialized: bool = True):  # type: ignore[no-untyped-def]
        return McpStdioAdapter._handle_message(
            json.dumps(payload).encode(),
            stream,
            broker,
            episode.spec,
            initialized,
        )

    assert McpStdioAdapter._handle_message(b"not-json", stream, broker, episode.spec, False)[
        1
    ].startswith("invalid MCP message")
    assert handle({"jsonrpc": "1.0"})[1] == "invalid MCP JSON-RPC envelope"
    assert handle({"jsonrpc": "2.0", "method": "ping"})[1] == "MCP request has no id"

    _submission, error, initialized = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "unsupported"},
        },
        False,
    )
    assert error is None and initialized
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, initialized) == (
        None,
        None,
        True,
    )
    assert handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})[1] is None
    assert handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})[1] is None
    assert handle({"jsonrpc": "2.0", "id": 4, "method": "unknown"})[1] is None
    assert handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": []})[1] is None
    for request_id, arguments in enumerate(
        (
            {},
            {"command": {"op": "bad"}},
            {
                "command": {
                    "request_id": "open",
                    "op": "scene.open",
                    "source_path": broker.visible_model_path,
                }
            },
        ),
        start=6,
    ):
        assert (
            handle(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": "meshprobe", "arguments": arguments},
                }
            )[1]
            is None
        )
    assert (
        handle(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "submit", "arguments": {}},
            }
        )[0]
        is None
    )
    submission, error, _initialized = handle(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "submit",
                "arguments": {
                    "episode_id": episode.spec.episode_id,
                    "answer": episode.ground_truth.answer.model_dump(mode="json"),
                },
            },
        }
    )
    assert error is None and submission is not None
    assert (
        handle(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {"name": "unknown", "arguments": {}},
            }
        )[1]
        is None
    )
    assert '"isError":true' in stream.getvalue()


def test_agent_adapters_reject_empty_commands_and_enforce_wall_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        CliJsonlAdapter(())
    with pytest.raises(ValueError, match="cannot be empty"):
        McpStdioAdapter(())

    public = tmp_path / "public"
    public.mkdir()
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), public)
    generated = generate_episodes(model)[0]
    spec = generated.spec.model_copy(update={"budgets": EpisodeBudgets(wall_seconds=0.05)})
    script = public / "slow.py"
    script.write_text("import time; time.sleep(10)\n", encoding="utf-8")
    broker = EvaluationBroker(
        service=AdapterService(),
        model_path=model.path,
        artifact_root=tmp_path / "agent-artifacts",
        evaluator_root=tmp_path / "evaluator" / "passes",
        trace_path=tmp_path / "evaluator" / "trace.jsonl",
        budgets=spec.budgets,
    )

    run = CliJsonlAdapter(agent_command(script)).run(
        spec=spec,
        broker=broker,
        input_root=public,
        artifact_root=tmp_path / "agent-artifacts",
    )

    assert run.timed_out
    assert run.protocol_error == "agent exceeded the episode wall-time budget"
