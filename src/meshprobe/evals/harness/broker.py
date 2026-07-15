"""Budgeted tool broker that records private, replayable evaluation traces."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from meshprobe.controller import BlenderWorkerError
from meshprobe.evals.harness.sandbox import visible_input_path
from meshprobe.evals.schemas import EpisodeBudgets, Operation, TraceEvent, TraceStatus
from meshprobe.protocol import (
    Command,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
)
from meshprobe.service import CommandResponse


class EvaluationService(Protocol):
    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse: ...


class BrokerError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str


class BrokerReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    response: CommandResponse | None = None
    error: BrokerError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> BrokerReply:
        if self.ok != (self.response is not None) or self.ok == (self.error is not None):
            raise ValueError("broker reply must contain exactly one matching outcome")
        return self


@dataclass(frozen=True)
class BrokerMetrics:
    tool_calls: int
    renders: int
    total_pixels: int
    output_bytes: int
    wall_seconds: float


class _RejectedCommand(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EvaluationBroker:
    """Translate sandbox paths, enforce budgets, and checkpoint every tool call."""

    def __init__(
        self,
        *,
        service: EvaluationService,
        model_path: Path,
        artifact_root: Path,
        evaluator_root: Path,
        trace_path: Path,
        budgets: EpisodeBudgets,
    ) -> None:
        self._service = service
        self._model_path = model_path.expanduser().resolve(strict=True)
        self._artifact_root = artifact_root.expanduser().resolve()
        self._evaluator_root = evaluator_root.expanduser().resolve()
        self._trace_path = trace_path.expanduser().resolve()
        roots = (self._artifact_root, self._evaluator_root, self._trace_path.parent)
        if any(root == self._model_path.parent for root in roots):
            raise ValueError("agent and evaluator storage must not share the model directory")
        if self._artifact_root == self._evaluator_root:
            raise ValueError("agent and evaluator artifact roots must be separate")
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._evaluator_root.mkdir(parents=True, exist_ok=True)
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._budgets = budgets
        self._events: list[TraceEvent] = []
        self._request_ids: set[str] = set()
        self._state_sha256: str | None = None
        self._renders = 0
        self._total_pixels = 0
        self._output_bytes = 0
        self._started = time.monotonic()
        self._public_errors: list[str] = []

    @property
    def visible_model_path(self) -> str:
        return visible_input_path(self._model_path)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    @property
    def public_errors(self) -> tuple[str, ...]:
        return tuple(self._public_errors)

    @property
    def metrics(self) -> BrokerMetrics:
        return BrokerMetrics(
            tool_calls=len(self._events),
            renders=self._renders,
            total_pixels=self._total_pixels,
            output_bytes=self._output_bytes,
            wall_seconds=time.monotonic() - self._started,
        )

    def execute(self, command: Command) -> BrokerReply:
        sequence = len(self._events) + 1
        started = time.monotonic()
        state_before = self._state_sha256
        private_result: JsonValue = None
        error_code: str | None = None
        status = TraceStatus.ACCEPTED
        try:
            self._reserve_request(command)
            translated, render_count, pixel_count = self._translate_and_reserve(command)
            private_dir = self._evaluator_root / f"{sequence:04d}-{command.request_id}"
            response = self._service.execute_for_evaluation(
                translated,
                evaluator_output_dir=str(private_dir),
            )
            private_result = response.result
            self._state_sha256 = _state_hash(private_result) or state_before
            self._renders += render_count
            self._total_pixels += pixel_count
            self._output_bytes += _artifact_bytes(private_result)
            public_result = _public_result(
                private_result,
                self._artifact_root,
                self._evaluator_root,
            )
            public_response = response.model_copy(update={"result": public_result})
            reply = BrokerReply(ok=True, response=public_response)
        except _RejectedCommand as error:
            status = TraceStatus.REJECTED
            error_code = error.code
            reply = self._error_reply(error.code, str(error))
        except (BlenderWorkerError, OSError, ValueError) as error:
            status = TraceStatus.REJECTED
            error_code = f"tool.{type(error).__name__}"
            reply = self._error_reply(error_code, str(error))
        except Exception as error:
            status = TraceStatus.CRASHED
            error_code = f"tool.{type(error).__name__}"
            reply = self._error_reply(error_code, "tool execution crashed")
        event = TraceEvent(
            sequence=sequence,
            request_id=command.request_id,
            operation=Operation(command.op),
            status=status,
            arguments=command.model_dump(mode="json", exclude={"request_id", "op"}),
            result=private_result,
            error_code=error_code,
            state_before_sha256=state_before,
            state_after_sha256=self._state_sha256,
            started_monotonic=started,
            elapsed_seconds=time.monotonic() - started,
        )
        self._events.append(event)
        self._checkpoint_trace()
        return reply

    def _reserve_request(self, command: Command) -> None:
        if command.request_id in self._request_ids:
            raise _RejectedCommand("broker.duplicate_request", "request_id must be unique")
        self._request_ids.add(command.request_id)
        if len(self._events) >= self._budgets.tool_calls:
            raise _RejectedCommand("budget.tool_calls", "tool-call budget exhausted")
        if time.monotonic() - self._started > self._budgets.wall_seconds:
            raise _RejectedCommand("budget.wall_seconds", "episode wall-time budget exhausted")

    def _translate_and_reserve(self, command: Command) -> tuple[Command, int, int]:
        if isinstance(command, SceneOpenCommand):
            if command.source_path != self.visible_model_path:
                raise _RejectedCommand(
                    "broker.model_path",
                    f"scene.open must use assigned model {self.visible_model_path}",
                )
            return command.model_copy(update={"source_path": str(self._model_path)}), 0, 0
        if isinstance(command, RenderImageCommand):
            render_count = 1
            pixels = command.width * command.height
            translated_image = command.model_copy(
                update={"output_path": str(self._artifact_path(command.output_path))}
            )
            self._check_render_budget(render_count, pixels)
            return translated_image, render_count, pixels
        if isinstance(command, RenderContactSheetCommand):
            render_count = 9
            pixels = render_count * command.panel_width * command.panel_height
            translated_sheet = command.model_copy(
                update={"output_path": str(self._artifact_path(command.output_path))}
            )
            self._check_render_budget(render_count, pixels)
            return translated_sheet, render_count, pixels
        return command, 0, 0

    def _artifact_path(self, visible_path: str) -> Path:
        relative = PurePosixPath(visible_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise _RejectedCommand(
                "broker.output_path",
                "render output must be a relative path under the assigned artifact directory",
            )
        resolved = (self._artifact_root / Path(*relative.parts)).resolve()
        if not resolved.is_relative_to(self._artifact_root):
            raise _RejectedCommand("broker.output_path", "render output escapes artifact root")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def _check_render_budget(self, render_count: int, pixels: int) -> None:
        if self._renders + render_count > self._budgets.renders:
            raise _RejectedCommand("budget.renders", "render budget exhausted")
        if self._total_pixels + pixels > self._budgets.total_pixels:
            raise _RejectedCommand("budget.total_pixels", "render pixel budget exhausted")

    def _error_reply(self, code: str, message: str) -> BrokerReply:
        self._public_errors.append(message)
        return BrokerReply(ok=False, error=BrokerError(code=code, message=message))

    def _checkpoint_trace(self) -> None:
        payload = "".join(event.model_dump_json() + "\n" for event in self._events)
        pending = self._trace_path.with_name(f".{self._trace_path.name}.pending")
        pending.write_text(payload, encoding="utf-8")
        os.replace(pending, self._trace_path)


def _state_hash(value: JsonValue) -> str | None:
    if not isinstance(value, dict):
        return None
    direct = value.get("state_sha256")
    if isinstance(direct, str) and len(direct) == 64:
        return direct
    session = value.get("session")
    if isinstance(session, dict):
        nested = session.get("state_sha256")
        if isinstance(nested, str) and len(nested) == 64:
            return nested
    return None


def _artifact_bytes(value: JsonValue) -> int:
    paths: set[str] = set()

    def visit(item: JsonValue) -> int:
        if isinstance(item, list):
            return sum(visit(nested) for nested in item)
        if not isinstance(item, dict):
            return 0
        path = item.get("path")
        byte_count = item.get("bytes")
        total = 0
        if isinstance(path, str) and isinstance(byte_count, int) and path not in paths:
            paths.add(path)
            total += byte_count
        return total + sum(visit(nested) for nested in item.values())

    return visit(value)


def _public_result(value: JsonValue, artifact_root: Path, evaluator_root: Path) -> JsonValue:
    if isinstance(value, list):
        return [_public_result(item, artifact_root, evaluator_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _public_result(item, artifact_root, evaluator_root)
            for key, item in value.items()
            if key != "evaluator"
        }
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_relative_to(evaluator_root):
        raise ValueError("private evaluator path appeared in a public result")
    if candidate.is_absolute() and candidate.is_relative_to(artifact_root):
        relative = candidate.relative_to(artifact_root).as_posix()
        if os.name == "nt":
            return str(candidate)
        return f"/workspace/artifacts/{relative}"
    return value
