"""Budgeted tool broker that records private, replayable evaluation traces."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from meshprobe.controller import BlenderWorkerError
from meshprobe.evals.harness.sandbox import visible_input_path
from meshprobe.evals.schemas import EpisodeBudgets, Operation, TraceEvent, TraceStatus
from meshprobe.models import CustomIllumination
from meshprobe.protocol import (
    Command,
    CommandEffect,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    command_payload,
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
        staging_root: Path | None = None
        request_token = hashlib.sha256(command.request_id.encode()).hexdigest()[:16]
        private_dir = self._evaluator_root / f"{sequence:04d}-{request_token}"
        try:
            self._reserve_request(command)
            translated, render_count, pixel_count, byte_reservation, staging_root = (
                self._translate_and_reserve(command, sequence)
            )
            response = self._service.execute_for_evaluation(
                translated,
                evaluator_output_dir=str(private_dir),
            )
            private_result = response.result
            produced_bytes = _directory_bytes(private_dir) + _directory_bytes(staging_root)
            if produced_bytes > byte_reservation:
                self._renders += render_count
                self._total_pixels += pixel_count
                self._discard_request_outputs(staging_root, private_dir)
                raise _RejectedCommand(
                    "budget.output_bytes",
                    "tool output exceeded its pre-reserved byte bound",
                )
            if staging_root is not None:
                self._publish_staged_artifacts(staging_root)
                private_result = _rewrite_path_root(
                    private_result,
                    staging_root,
                    self._artifact_root,
                )
            self._state_sha256 = _state_hash(private_result) or state_before
            self._renders += render_count
            self._total_pixels += pixel_count
            self._output_bytes += produced_bytes
            public_result = _public_result(
                private_result,
                self._artifact_root,
                self._evaluator_root,
            )
            comparison_path = _comparison_artifact_virtual_path(
                private_result,
                self._artifact_root,
            )
            if comparison_path is not None:
                public_result = _with_comparison_artifact_path(public_result, comparison_path)
            public_response = response.model_copy(update={"result": public_result})
            reply = BrokerReply(ok=True, response=public_response)
        except _RejectedCommand as error:
            self._discard_request_outputs(staging_root, private_dir)
            status = TraceStatus.REJECTED
            error_code = error.code
            reply = self._error_reply(error.code, str(error))
        except (BlenderWorkerError, OSError, ValueError) as error:
            self._discard_request_outputs(staging_root, private_dir)
            status = TraceStatus.REJECTED
            error_code = f"tool.{type(error).__name__}"
            reply = self._error_reply(
                error_code,
                str(error).replace(str(self._evaluator_root), "<private evaluator path>"),
            )
        except Exception as error:
            self._discard_request_outputs(staging_root, private_dir)
            status = TraceStatus.CRASHED
            error_code = f"tool.{type(error).__name__}"
            reply = self._error_reply(error_code, "tool execution crashed")
        event = TraceEvent(
            sequence=sequence,
            request_id=command.request_id,
            operation=Operation(command.op),
            status=status,
            arguments=command_payload(command, exclude={"request_id", "op"}),
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

    def _translate_and_reserve(
        self,
        command: Command,
        sequence: int,
    ) -> tuple[Command, int, int, int, Path | None]:
        if command.effect is CommandEffect.HISTORY:
            raise _RejectedCommand(
                "broker.unsupported_operation",
                f"{command.op} requires a durable session and is unavailable in evaluations",
            )
        if isinstance(command, SceneOpenCommand):
            if command.source_path != self.visible_model_path:
                raise _RejectedCommand(
                    "broker.model_path",
                    f"scene.open must use assigned model {self.visible_model_path}",
                )
            return command.model_copy(update={"source_path": str(self._model_path)}), 0, 0, 0, None
        if isinstance(command, IlluminationSetCommand):
            illumination = command.illumination
            if not isinstance(illumination, CustomIllumination):
                return command, 0, 0, 0, None
            environment_map = illumination.environment_map
            if environment_map is None:
                return command, 0, 0, 0, None
            translated_map = environment_map.model_copy(
                update={"path": str(self._readable_path(environment_map.path))}
            )
            translated_illumination = illumination.model_copy(
                update={"environment_map": translated_map}
            )
            return (
                command.model_copy(update={"illumination": translated_illumination}),
                0,
                0,
                0,
                None,
            )
        if isinstance(command, RenderImageCommand):
            render_count = 1
            pixels = command.width * command.height
            output_path = self._artifact_path(command.output_path)
            staging_root, staging_output = self._staging_output(
                output_path,
                sequence,
                command.request_id,
            )
            comparison = command.comparison
            if comparison is not None:
                reference_path = self._readable_path(comparison.reference_image_path)
                comparison_output = self._artifact_path(comparison.output_path)
                comparison_staging = staging_root / comparison_output.relative_to(
                    self._artifact_root
                )
                comparison = comparison.model_copy(
                    update={
                        "reference_image_path": str(reference_path),
                        "output_path": str(comparison_staging),
                    }
                )
            translated_image = command.model_copy(
                update={"output_path": str(staging_output), "comparison": comparison}
            )
            byte_reservation = _render_output_reservation(pixels)
            if comparison is not None:
                byte_reservation = self._budgets.output_bytes - self._output_bytes
            self._check_render_budget(render_count, pixels, byte_reservation)
            return translated_image, render_count, pixels, byte_reservation, staging_root
        if isinstance(command, RenderContactSheetCommand):
            render_count = 9
            if command.recipe == "orbit_sweep" and command.orbit_sweep is not None:
                render_count = (
                    len(command.orbit_sweep.azimuth_degrees)
                    * len(command.orbit_sweep.elevation_degrees)
                    * len(command.orbit_sweep.roll_degrees)
                )
            pixels = render_count * command.panel_width * command.panel_height
            output_path = self._artifact_path(command.output_path)
            staging_root, staging_output = self._staging_output(
                output_path,
                sequence,
                command.request_id,
            )
            translated_sheet = command.model_copy(update={"output_path": str(staging_output)})
            # Caption rows wrap renderer-owned component names and can grow beyond
            # dimensions derivable from the public command. Reserve the remaining
            # episode allowance; sandbox limits and post-render accounting still
            # enforce that hard cap while charging only the bytes actually written.
            byte_reservation = self._budgets.output_bytes - self._output_bytes
            minimum_reservation = _contact_sheet_minimum_output_reservation(
                command.panel_width,
                command.panel_height,
            )
            self._check_render_budget(render_count, pixels, minimum_reservation)
            return translated_sheet, render_count, pixels, byte_reservation, staging_root
        return command, 0, 0, 0, None

    def _readable_path(self, visible_path: str) -> Path:
        input_root = self._model_path.parent
        if os.name == "nt":
            host_candidate = Path(visible_path)
            if not host_candidate.is_absolute():
                raise _RejectedCommand("broker.input_path", "environment map path must be absolute")
            assigned_roots = (input_root, self._artifact_root)
            if not any(host_candidate.is_relative_to(root) for root in assigned_roots):
                raise _RejectedCommand(
                    "broker.input_path",
                    "environment map must be under the assigned input or artifact directory",
                )
            try:
                resolved = host_candidate.resolve(strict=True)
            except OSError as error:
                raise _RejectedCommand(
                    "broker.input_path", "environment map is unavailable"
                ) from error
        else:
            sandbox_candidate = PurePosixPath(visible_path)
            prefixes = {
                PurePosixPath("/workspace/input"): input_root,
                PurePosixPath("/workspace/artifacts"): self._artifact_root,
            }
            for prefix, root in prefixes.items():
                try:
                    relative = sandbox_candidate.relative_to(prefix)
                except ValueError:
                    continue
                if not relative.parts or ".." in relative.parts:
                    raise _RejectedCommand(
                        "broker.input_path", "environment map escapes assigned storage"
                    )
                try:
                    resolved = (root / Path(*relative.parts)).resolve(strict=True)
                except OSError as error:
                    raise _RejectedCommand(
                        "broker.input_path", "environment map is unavailable"
                    ) from error
                break
            else:
                raise _RejectedCommand(
                    "broker.input_path",
                    "environment map must be under the assigned input or artifact directory",
                )
        if not resolved.is_file():
            raise _RejectedCommand("broker.input_path", "environment map must be a regular file")
        if not (
            resolved.is_relative_to(input_root) or resolved.is_relative_to(self._artifact_root)
        ):
            raise _RejectedCommand("broker.input_path", "environment map escapes assigned storage")
        return resolved

    def _artifact_path(self, visible_path: str) -> Path:
        relative: PurePosixPath
        if os.name == "nt":
            windows_path = Path(visible_path)
            if windows_path.is_absolute():
                resolved_windows = windows_path.resolve()
                if not resolved_windows.is_relative_to(self._artifact_root):
                    raise _RejectedCommand(
                        "broker.output_path",
                        "render output escapes the assigned artifact directory",
                    )
                relative = PurePosixPath(
                    resolved_windows.relative_to(self._artifact_root).as_posix()
                )
            else:
                relative = PurePosixPath(visible_path)
        else:
            candidate = PurePosixPath(visible_path)
            artifact_prefix = PurePosixPath("/workspace/artifacts")
            if candidate.is_absolute():
                try:
                    relative = candidate.relative_to(artifact_prefix)
                except ValueError as error:
                    raise _RejectedCommand(
                        "broker.output_path",
                        "render output escapes the assigned artifact directory",
                    ) from error
            else:
                relative = candidate
        if not relative.parts or ".." in relative.parts:
            raise _RejectedCommand(
                "broker.output_path",
                "render output must be under the assigned artifact directory",
            )
        resolved = (self._artifact_root / Path(*relative.parts)).resolve()
        if not resolved.is_relative_to(self._artifact_root):
            raise _RejectedCommand("broker.output_path", "render output escapes artifact root")
        if resolved.exists():
            raise _RejectedCommand("broker.output_exists", "render output already exists")
        return resolved

    def _staging_output(
        self,
        output_path: Path,
        sequence: int,
        request_id: str,
    ) -> tuple[Path, Path]:
        request_digest = hashlib.sha256(request_id.encode()).hexdigest()[:16]
        staging_root = self._evaluator_root / "staging" / f"{sequence:04d}-{request_digest}"
        if staging_root.exists():
            raise _RejectedCommand("broker.staging_exists", "request staging directory exists")
        relative = output_path.relative_to(self._artifact_root)
        return staging_root, staging_root / relative

    def _check_render_budget(self, render_count: int, pixels: int, output_bytes: int) -> None:
        if self._renders + render_count > self._budgets.renders:
            raise _RejectedCommand("budget.renders", "render budget exhausted")
        if self._total_pixels + pixels > self._budgets.total_pixels:
            raise _RejectedCommand("budget.total_pixels", "render pixel budget exhausted")
        if self._output_bytes + output_bytes > self._budgets.output_bytes:
            raise _RejectedCommand(
                "budget.output_bytes",
                "render output cannot fit within the remaining output-byte budget",
            )

    def _publish_staged_artifacts(self, staging_root: Path) -> None:
        staged_files = tuple(path for path in staging_root.rglob("*") if path.is_file())
        if not staged_files:
            raise ValueError("render produced no staged artifacts")
        for source in staged_files:
            relative = source.relative_to(staging_root)
            if os.name == "posix":
                self._publish_staged_file_posix(source, relative)
                continue
            destination = self._artifact_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if any(parent.is_symlink() for parent in (destination, *destination.parents)):
                raise _RejectedCommand("broker.output_path", "render destination uses a symlink")
            resolved_parent = destination.parent.resolve(strict=True)
            if not resolved_parent.is_relative_to(self._artifact_root):
                raise _RejectedCommand(
                    "broker.output_path", "render destination escapes artifact root"
                )
            if destination.exists():
                raise _RejectedCommand("broker.output_exists", "render artifact already exists")
            os.replace(source, destination)
        shutil.rmtree(staging_root)

    def _publish_staged_file_posix(self, source: Path, relative: Path) -> None:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(self._artifact_root, flags)
        parent_fd = root_fd
        try:
            for part in relative.parts[:-1]:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                next_fd = os.open(part, flags, dir_fd=parent_fd)
                if parent_fd != root_fd:
                    os.close(parent_fd)
                parent_fd = next_fd
            name = relative.name
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise _RejectedCommand("broker.output_exists", "render artifact already exists")
            os.replace(source, name, dst_dir_fd=parent_fd)
        except OSError as error:
            raise _RejectedCommand(
                "broker.output_path", "render destination is not a safe artifact path"
            ) from error
        finally:
            if parent_fd != root_fd:
                os.close(parent_fd)
            os.close(root_fd)

    @staticmethod
    def _discard_request_outputs(staging_root: Path | None, private_dir: Path) -> None:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(private_dir, ignore_errors=True)

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


def _render_output_reservation(pixels: int) -> int:
    # Color and two RGBA masks plus a float depth/normal EXR. The fixed allowance
    # covers PNG/EXR headers and scanline tables at small resolutions.
    return pixels * 48 + 4 * 1024**2


def _contact_sheet_minimum_output_reservation(panel_width: int, panel_height: int) -> int:
    panel_pixels = panel_width * panel_height
    caption_height = max(48, panel_height // 10)
    sheet_pixels = panel_width * 3 * (panel_height + caption_height) * 3
    sheet_png_bound = sheet_pixels * 4 + 1024**2
    return 9 * _render_output_reservation(panel_pixels) + sheet_png_bound


def _directory_bytes(root: Path | None) -> int:
    if root is None or not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("render output must not contain symbolic links")
        if path.is_file():
            total += path.stat().st_size
    return total


def _rewrite_path_root(value: JsonValue, old_root: Path, new_root: Path) -> JsonValue:
    if isinstance(value, list):
        return [_rewrite_path_root(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_path_root(item, old_root, new_root) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if not candidate.is_absolute() or not candidate.is_relative_to(old_root):
        return value
    return str(new_root / candidate.relative_to(old_root))


def _public_result(value: JsonValue, artifact_root: Path, evaluator_root: Path) -> JsonValue:
    if isinstance(value, list):
        return [_public_result(item, artifact_root, evaluator_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _public_result(item, artifact_root, evaluator_root)
            for key, item in value.items()
            if key not in {"evaluator", "normalized_geometry"}
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


def _comparison_artifact_virtual_path(value: JsonValue, artifact_root: Path) -> str | None:
    if not isinstance(value, dict):
        return None
    comparison = value.get("comparison")
    if not isinstance(comparison, dict):
        return None
    artifact = comparison.get("artifact")
    if not isinstance(artifact, dict):
        return None
    path = artifact.get("path")
    if not isinstance(path, str):
        return None
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_relative_to(artifact_root):
        return None
    relative = candidate.relative_to(artifact_root).as_posix()
    return f"/workspace/artifacts/{relative}"


def _with_comparison_artifact_path(value: JsonValue, path: str) -> JsonValue:
    if not isinstance(value, dict):
        return value
    comparison = value.get("comparison")
    if not isinstance(comparison, dict):
        return value
    artifact = comparison.get("artifact")
    if not isinstance(artifact, dict):
        return value
    return {
        **value,
        "comparison": {
            **comparison,
            "artifact": {**artifact, "path": path},
        },
    }
