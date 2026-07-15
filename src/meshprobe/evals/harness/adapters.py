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
        process.stdin.write(json.dumps(initialization, separators=(",", ":")) + "\n")
        process.stdin.flush()
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


def _complete_lines(buffer: bytes) -> tuple[bytes, tuple[bytes, ...]]:
    parts = buffer.split(b"\n")
    return parts[-1], tuple(part for part in parts[:-1] if part.strip())
