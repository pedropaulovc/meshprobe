"""Interactive agent adapters sharing the MeshProbe command contract."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from mcp.types import LATEST_PROTOCOL_VERSION
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.harness.sandbox import IsolationLimits, spawn_isolated
from meshprobe.evals.schemas import EpisodeSpec, EpisodeSubmission
from meshprobe.protocol import COMMAND_ADAPTER, command_json_schema

_MAX_STREAM_BYTES = 1_000_000


class AdapterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCallMessage(AdapterModel):
    type: Literal["tool_call"]
    command: dict[str, JsonValue]


class SubmissionMessage(AdapterModel):
    type: Literal["submission"]
    submission: EpisodeSubmission


AgentMessage = Annotated[ToolCallMessage | SubmissionMessage, Field(discriminator="type")]
AGENT_MESSAGE_ADAPTER: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)


@dataclass(frozen=True)
class AdapterRun:
    submission: EpisodeSubmission | None
    returncode: int
    stderr: str
    elapsed_seconds: float
    timed_out: bool
    protocol_error: str | None


class CliJsonlAdapter:
    """Drive a command-line agent over a strict line-delimited JSON protocol."""

    def __init__(self, command: tuple[str, ...]) -> None:
        if not command:
            raise ValueError("agent command cannot be empty")
        self.command = command

    def run(
        self,
        *,
        spec: EpisodeSpec,
        broker: EvaluationBroker,
        input_root: Path,
        artifact_root: Path,
    ) -> AdapterRun:
        limits = IsolationLimits(
            wall_seconds=spec.budgets.wall_seconds,
            cpu_seconds=max(1, int(spec.budgets.wall_seconds)),
            output_bytes=spec.budgets.output_bytes,
        )
        isolated = spawn_isolated(
            self.command,
            input_root=input_root,
            artifact_root=artifact_root,
            limits=limits,
            environment={"MESHPROBE_AGENT_PROTOCOL": "jsonl-v1"},
        )
        process = isolated.process
        if process.stdin is None or process.stdout is None or process.stderr is None:
            isolated.terminate()
            raise RuntimeError("isolated process did not expose standard streams")
        initialization = {
            "type": "episode",
            "protocol_version": 1,
            "episode_id": spec.episode_id,
            "prompt": spec.prompt,
            "answer_schema": spec.answer_schema,
            "tool_schema": command_json_schema(),
            "model_path": broker.visible_model_path,
            "budgets": spec.budgets.model_dump(mode="json"),
        }
        try:
            process.stdin.write(json.dumps(initialization, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            process.wait(timeout=1)
            stderr = process.stderr.read(_MAX_STREAM_BYTES)
            stdout = process.stdout.read(_MAX_STREAM_BYTES).encode("utf-8", errors="replace")
            _remainder, messages = _complete_lines(stdout)
            startup_error: str | None = "agent exited before accepting the episode envelope"
            startup_submission: EpisodeSubmission | None = None
            for raw_message in messages:
                startup_submission, startup_error = self._handle_message(
                    raw_message,
                    process.stdin,
                    broker,
                )
                if startup_submission is not None or startup_error is not None:
                    break
            return AdapterRun(
                submission=startup_submission,
                returncode=process.returncode,
                stderr=stderr,
                elapsed_seconds=time.monotonic() - isolated.started_monotonic,
                timed_out=False,
                protocol_error=startup_error,
            )
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        stdout_buffer = b""
        stderr_buffer = b""
        submission: EpisodeSubmission | None = None
        protocol_error: str | None = None
        timed_out = False
        deadline = isolated.started_monotonic + isolated.wall_seconds
        try:
            while submission is None and protocol_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    protocol_error = "agent exceeded the episode wall-time budget"
                    break
                if process.poll() is not None and not selector.select(timeout=0):
                    protocol_error = "agent exited before submitting an answer"
                    break
                ready = selector.select(timeout=min(remaining, 0.25))
                for key, _ in ready:
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        stderr_buffer = (stderr_buffer + chunk)[-_MAX_STREAM_BYTES:]
                        continue
                    stdout_buffer += chunk
                    if len(stdout_buffer) > _MAX_STREAM_BYTES:
                        protocol_error = "agent protocol line exceeded 1000000 bytes"
                        break
                    stdout_buffer, messages = _complete_lines(stdout_buffer)
                    for raw_message in messages:
                        submission, protocol_error = self._handle_message(
                            raw_message,
                            process.stdin,
                            broker,
                        )
                        if submission is not None or protocol_error is not None:
                            break
        finally:
            selector.close()
            if process.stdin is not None:
                process.stdin.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    isolated.terminate()
            if process.stderr is not None:
                remainder = process.stderr.read(_MAX_STREAM_BYTES).encode("utf-8", errors="replace")
                stderr_buffer = (stderr_buffer + remainder)[-_MAX_STREAM_BYTES:]
        return AdapterRun(
            submission=submission,
            returncode=process.returncode,
            stderr=stderr_buffer.decode("utf-8", errors="replace"),
            elapsed_seconds=time.monotonic() - isolated.started_monotonic,
            timed_out=timed_out,
            protocol_error=protocol_error,
        )

    @staticmethod
    def _handle_message(
        raw_message: bytes,
        agent_stdin: object,
        broker: EvaluationBroker,
    ) -> tuple[EpisodeSubmission | None, str | None]:
        try:
            payload = json.loads(raw_message.decode("utf-8"))
            message = AGENT_MESSAGE_ADAPTER.validate_python(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            return None, f"invalid agent protocol message: {error}"
        if isinstance(message, SubmissionMessage):
            return message.submission, None
        try:
            command = COMMAND_ADAPTER.validate_python(message.command)
        except ValidationError as error:
            response: object = {
                "type": "tool_result",
                "ok": False,
                "error": {"code": "protocol.invalid_command", "message": str(error)},
            }
        else:
            response = {
                "type": "tool_result",
                **broker.execute(command).model_dump(mode="json"),
            }
        if not hasattr(agent_stdin, "write") or not hasattr(agent_stdin, "flush"):
            return None, "agent stdin became unavailable"
        agent_stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
        agent_stdin.flush()
        return None, None


class McpStdioAdapter:
    """Expose the broker as an MCP server to an isolated tool-calling agent."""

    def __init__(self, command: tuple[str, ...]) -> None:
        if not command:
            raise ValueError("agent command cannot be empty")
        self.command = command

    def run(
        self,
        *,
        spec: EpisodeSpec,
        broker: EvaluationBroker,
        input_root: Path,
        artifact_root: Path,
    ) -> AdapterRun:
        limits = IsolationLimits(
            wall_seconds=spec.budgets.wall_seconds,
            cpu_seconds=max(1, int(spec.budgets.wall_seconds)),
            output_bytes=spec.budgets.output_bytes,
        )
        isolated = spawn_isolated(
            self.command,
            input_root=input_root,
            artifact_root=artifact_root,
            limits=limits,
            environment={"MESHPROBE_AGENT_PROTOCOL": "mcp-stdio-v1"},
        )
        process = isolated.process
        if process.stdin is None or process.stdout is None or process.stderr is None:
            isolated.terminate()
            raise RuntimeError("isolated process did not expose standard streams")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        stdout_buffer = b""
        stderr_buffer = b""
        submission: EpisodeSubmission | None = None
        protocol_error: str | None = None
        timed_out = False
        initialized = False
        deadline = isolated.started_monotonic + isolated.wall_seconds
        try:
            while submission is None and protocol_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    protocol_error = "agent exceeded the episode wall-time budget"
                    break
                if process.poll() is not None and not selector.select(timeout=0):
                    protocol_error = "agent exited before submitting an answer"
                    break
                ready = selector.select(timeout=min(remaining, 0.25))
                for key, _ in ready:
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        stderr_buffer = (stderr_buffer + chunk)[-_MAX_STREAM_BYTES:]
                        continue
                    stdout_buffer += chunk
                    if len(stdout_buffer) > _MAX_STREAM_BYTES:
                        protocol_error = "agent protocol line exceeded 1000000 bytes"
                        break
                    stdout_buffer, messages = _complete_lines(stdout_buffer)
                    for raw_message in messages:
                        submission, message_error, initialized = self._handle_message(
                            raw_message,
                            process.stdin,
                            broker,
                            spec,
                            initialized,
                        )
                        protocol_error = message_error
                        if submission is not None or protocol_error is not None:
                            break
        finally:
            selector.close()
            if process.stdin is not None:
                process.stdin.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    isolated.terminate()
            if process.stderr is not None:
                remainder = process.stderr.read(_MAX_STREAM_BYTES).encode("utf-8", errors="replace")
                stderr_buffer = (stderr_buffer + remainder)[-_MAX_STREAM_BYTES:]
        return AdapterRun(
            submission=submission,
            returncode=process.returncode,
            stderr=stderr_buffer.decode("utf-8", errors="replace"),
            elapsed_seconds=time.monotonic() - isolated.started_monotonic,
            timed_out=timed_out,
            protocol_error=protocol_error,
        )

    @staticmethod
    def _handle_message(
        raw_message: bytes,
        agent_stdin: object,
        broker: EvaluationBroker,
        spec: EpisodeSpec,
        initialized: bool,
    ) -> tuple[EpisodeSubmission | None, str | None, bool]:
        try:
            message = json.loads(raw_message.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return None, f"invalid MCP message: {error}", initialized
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return None, "invalid MCP JSON-RPC envelope", initialized
        method = message.get("method")
        request_id = message.get("id")
        if method == "notifications/initialized" and request_id is None:
            return None, None, initialized
        if request_id is None:
            return None, "MCP request has no id", initialized
        if method == "initialize":
            params = message.get("params")
            requested_version = params.get("protocolVersion") if isinstance(params, dict) else None
            protocol_version = (
                requested_version
                if requested_version in SUPPORTED_PROTOCOL_VERSIONS
                else LATEST_PROTOCOL_VERSION
            )
            response = {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "MeshProbe Evaluation", "version": "1"},
                "instructions": json.dumps(
                    {
                        "episode_id": spec.episode_id,
                        "prompt": spec.prompt,
                        "answer_schema": spec.answer_schema,
                        "model_path": broker.visible_model_path,
                        "budgets": spec.budgets.model_dump(mode="json"),
                    },
                    separators=(",", ":"),
                ),
            }
            _write_rpc(agent_stdin, _rpc_result(request_id, response))
            return None, None, True
        if not initialized:
            _write_rpc(agent_stdin, _rpc_error(request_id, -32002, "server is not initialized"))
            return None, None, initialized
        if method == "ping":
            _write_rpc(agent_stdin, _rpc_result(request_id, {}))
            return None, None, initialized
        if method == "tools/list":
            _write_rpc(
                agent_stdin,
                _rpc_result(request_id, {"tools": _mcp_tools()}),
            )
            return None, None, initialized
        if method != "tools/call":
            _write_rpc(agent_stdin, _rpc_error(request_id, -32601, "method not found"))
            return None, None, initialized
        params = message.get("params")
        if not isinstance(params, dict):
            _write_rpc(agent_stdin, _rpc_error(request_id, -32602, "invalid tool parameters"))
            return None, None, initialized
        name = params.get("name")
        arguments = params.get("arguments")
        if name == "meshprobe":
            result = _call_meshprobe(arguments, broker)
            _write_rpc(agent_stdin, _rpc_result(request_id, result))
            return None, None, initialized
        if name == "submit":
            try:
                submission = EpisodeSubmission.model_validate(arguments)
            except ValidationError as error:
                result = _mcp_tool_result(
                    {"code": "protocol.invalid_submission", "message": str(error)},
                    is_error=True,
                )
                _write_rpc(agent_stdin, _rpc_result(request_id, result))
                return None, None, initialized
            _write_rpc(
                agent_stdin,
                _rpc_result(request_id, _mcp_tool_result({"accepted": True})),
            )
            return submission, None, initialized
        result = _mcp_tool_result(
            {"code": "protocol.unknown_tool", "message": f"unknown tool: {name}"},
            is_error=True,
        )
        _write_rpc(agent_stdin, _rpc_result(request_id, result))
        return None, None, initialized


def _mcp_tools() -> list[dict[str, object]]:
    return [
        {
            "name": "meshprobe",
            "description": "Execute one typed read-only 3D inspection operation.",
            "inputSchema": {
                "type": "object",
                "properties": {"command": command_json_schema()},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit",
            "description": "Submit the final structured episode answer and evidence paths.",
            "inputSchema": EpisodeSubmission.model_json_schema(),
        },
    ]


def _call_meshprobe(arguments: object, broker: EvaluationBroker) -> dict[str, object]:
    if not isinstance(arguments, dict) or "command" not in arguments:
        return _mcp_tool_result(
            {"code": "protocol.invalid_command", "message": "command is required"},
            is_error=True,
        )
    try:
        command = COMMAND_ADAPTER.validate_python(arguments["command"])
    except ValidationError as error:
        return _mcp_tool_result(
            {"code": "protocol.invalid_command", "message": str(error)},
            is_error=True,
        )
    return _mcp_tool_result(broker.execute(command).model_dump(mode="json"))


def _mcp_tool_result(payload: object, *, is_error: bool = False) -> dict[str, object]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            }
        ],
        "structuredContent": payload,
        "isError": is_error,
    }


def _rpc_result(request_id: object, result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(request_id: object, code: int, message: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _write_rpc(stream: object, payload: object) -> None:
    if not hasattr(stream, "write") or not hasattr(stream, "flush"):
        raise RuntimeError("agent stdin became unavailable")
    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stream.flush()


def _complete_lines(buffer: bytes) -> tuple[bytes, tuple[bytes, ...]]:
    parts = buffer.split(b"\n")
    return parts[-1], tuple(part for part in parts[:-1] if part.strip())
