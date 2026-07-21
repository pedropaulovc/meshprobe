"""Shared application service for CLI and MCP clients."""

from __future__ import annotations

import time
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from meshprobe.controller import (
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    BlenderController,
    BlenderWorkerCrashed,
    BlenderWorkerTimeout,
    WorkerRecoveryPolicy,
)
from meshprobe.models import GraphicsPlatform
from meshprobe.protocol import (
    Command,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
)


class CommandResponse(BaseModel):
    """JSON-safe result envelope returned by every public adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    op: str
    result: JsonValue


class MeshProbeService:
    """Own a persistent renderer and dispatch typed public commands."""

    def __init__(
        self,
        *,
        controller: BlenderController | None = None,
        blender: str | None = None,
        timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    ) -> None:
        if controller is not None and blender is not None:
            raise ValueError("controller and blender cannot both be provided")
        self._controller = controller or BlenderController(
            executable=blender,
            timeout_seconds=timeout_seconds,
        )
        self._started = False

    def execute(self, command: Command) -> CommandResponse:
        """Execute one command and serialize its result through the public contract."""

        self._ensure_started(command)
        return self._response(command, self._controller.execute(command))

    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
        wall_timeout_seconds: float | None = None,
    ) -> CommandResponse:
        """Execute while keeping private render passes outside agent storage."""

        self._ensure_started(command)
        if isinstance(command, RenderImageCommand):
            try:
                image_result = self._controller.render_image(
                    command,
                    evaluator_output_dir=evaluator_output_dir,
                    recovery_policy=WorkerRecoveryPolicy.CLOSE,
                    request_deadline_monotonic=(
                        time.monotonic() + wall_timeout_seconds
                        if wall_timeout_seconds is not None
                        else None
                    ),
                )
            except (BlenderWorkerCrashed, BlenderWorkerTimeout):
                self._started = False
                raise
            return self._response(command, image_result)
        if isinstance(command, RenderContactSheetCommand):
            try:
                sheet_result = self._controller.render_contact_sheet(
                    command,
                    evaluator_output_dir=evaluator_output_dir,
                    recovery_policy=WorkerRecoveryPolicy.CLOSE,
                    request_deadline_monotonic=(
                        time.monotonic() + wall_timeout_seconds
                        if wall_timeout_seconds is not None
                        else None
                    ),
                )
            except (BlenderWorkerCrashed, BlenderWorkerTimeout):
                self._started = False
                raise
            return self._response(command, sheet_result)
        return self._response(command, self._controller.execute(command))

    def _ensure_started(self, command: Command) -> None:
        if self._started:
            return
        if not isinstance(command, SceneOpenCommand):
            raise ValueError("scene.open must be the first command in a session")
        self._controller.start()
        self._started = True

    @staticmethod
    def _response(command: Command, result: object) -> CommandResponse:
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        serialized: JsonValue = TypeAdapter(JsonValue).validate_python(result)
        return CommandResponse(request_id=command.request_id, op=command.op, result=serialized)

    def close(self) -> None:
        self._controller.close()
        self._started = False

    @property
    def worker_pid(self) -> int | None:
        return self._controller.worker_pid

    @property
    def graphics(self) -> GraphicsPlatform | None:
        return self._controller.graphics

    def kill(self) -> None:
        self._controller.kill()
        self._started = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
