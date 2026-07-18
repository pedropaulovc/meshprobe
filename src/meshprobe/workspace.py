"""Durable project-local session state and renderer lifecycle management."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from meshprobe.camera import orbit_angles_from_orientation
from meshprobe.controller import DEFAULT_WORKER_TIMEOUT_SECONDS
from meshprobe.models import (
    UNIT_SUSPICION_WARNING_CODES,
    Camera,
    CameraOrbitAngles,
    DisplayMode,
    GraphicsPlatform,
    Illumination,
    MarkMode,
    RenderStyleState,
    SceneManifest,
    SessionSnapshot,
    SrgbHexColor,
)
from meshprobe.protocol import (
    Command,
    ComponentFindCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionResetCommand,
)
from meshprobe.selectors import is_glob_pattern, normalize_glob, path_glob_match
from meshprobe.service import CommandResponse, MeshProbeService

STATE_OPERATIONS = frozenset(
    {
        "view.set",
        "view.orbit",
        "view.frame",
        "view.move",
        "view.rotate",
        "illumination.set",
        "component.display",
        "component.mark",
    }
)

# The ops that always refit the camera to an aspect ratio, so their aspect_ratio is
# the intended framing. view.move/view.rotate reuse the current projection without
# refitting, so their aspect_ratio only feeds diagnostics and must NOT be read as the
# framing aspect — a bare move/rotate after a wide orbit would otherwise reset the
# tracked framing to its 1.0 default and false-warn a matching render. The exception is
# a view.rotate that carries its own projection (see _refits_camera): that replaces the
# projection, so it reframes at the given aspect just like the always-framing ops.
FRAMING_OPERATIONS = frozenset({"view.set", "view.orbit", "view.frame"})


def _refits_camera(command: dict[str, Any]) -> bool:
    op = command.get("op")
    if op in FRAMING_OPERATIONS:
        return True
    # A bare view.rotate reuses the current projection (a nudge, like view.move); one that
    # supplies a projection replaces it, reframing at the command's aspect_ratio.
    return op == "view.rotate" and command.get("projection") is not None


class RendererContinuity(StrEnum):
    PRESERVED = "preserved"
    RECREATED_BEFORE_SNAPSHOT = "recreated_before_snapshot"
    RECREATED_DURING_SNAPSHOT = "recreated_during_snapshot"


class SessionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str
    source_path: str
    source_sha256: str
    blender: str | None
    status: Literal["active", "closed", "killed"] = "active"
    created_at: str
    updated_at: str
    worker_pid: int | None = None
    graphics: GraphicsPlatform | None = None


class SessionCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2] = 2
    source_path: str
    source_sha256: str
    blender: str | None
    state_sha256: str
    accepted_commands: list[dict[str, Any]] = Field(default_factory=list)
    unit_scale: float = 1.0


def _upgrade_checkpoint(checkpoint: SessionCheckpoint) -> tuple[SessionCheckpoint, bool]:
    """Translate persisted commands whose meaning changed between checkpoint schemas."""
    if checkpoint.schema_version == 2:
        return checkpoint, False

    accepted_commands: list[dict[str, Any]] = []
    for raw in checkpoint.accepted_commands:
        command = dict(raw)
        if command.get("op") == "view.rotate":
            command["degrees"] = -float(command["degrees"])
        accepted_commands.append(command)
    return (
        checkpoint.model_copy(update={"schema_version": 2, "accepted_commands": accepted_commands}),
        True,
    )


class ComponentIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    id: str
    path: str
    parent: str | None
    name: str


class ComponentIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    components: list[ComponentIndexEntry]


class ComponentStateDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display: DisplayMode = DisplayMode.SHOWN
    mark: MarkMode = MarkMode.UNMARKED


class ComponentStateOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    display: DisplayMode | None = None
    mark: MarkMode | None = None
    mark_color: SrgbHexColor | None = None


class ComponentStateIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: ComponentStateDefaults
    overrides: dict[str, ComponentStateOverride]


class DurableSessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    state_sha256: str
    camera: Camera
    # Derived from the camera orientation; optional on read so v1 state.yml files written
    # before this field existed still validate, then backfilled from the camera below.
    camera_orbit_angles: CameraOrbitAngles | None = None
    illumination: Illumination
    render_style: RenderStyleState = RenderStyleState()
    components: ComponentStateIndex
    results: str
    artifacts: str

    @model_validator(mode="after")
    def backfill_orbit_angles(self) -> Self:
        if self.camera_orbit_angles is None:
            azimuth_degrees, elevation_degrees = orbit_angles_from_orientation(
                self.camera.pose.orientation_xyzw
            )
            self.camera_orbit_angles = CameraOrbitAngles(
                azimuth_degrees=azimuth_degrees,
                elevation_degrees=elevation_degrees,
            )
        return self


class SessionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: str
    op: str
    status: Literal["accepted", "rejected"]
    request_id: str | None = None
    command: dict[str, Any] | None = None
    result_path: str | None = None
    error: str | None = None


class ResolvedComponentReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str
    id: str
    name: str
    path: str


class OperationReceipt(BaseModel):
    """Small stdout contract pointing to queryable durable output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool = True
    session: str
    op: str
    state_sha256: str | None = None
    result_path: str | None = None
    state_path: str | None = None
    artifact_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    components: tuple[ResolvedComponentReference, ...] = ()
    match_count: int | None = None


def durable_state_json_schema() -> dict[str, object]:
    """Describe every stable file an agent queries in a durable session."""
    return {
        "title": "MeshProbe durable session state",
        "schema_version": 1,
        "files": {
            "metadata.json": SessionMetadata.model_json_schema(),
            "scene.json": SceneManifest.model_json_schema(),
            "components.yml": ComponentIndex.model_json_schema(),
            "state.yml": DurableSessionState.model_json_schema(),
            "checkpoint.json": SessionCheckpoint.model_json_schema(),
            "events.jsonl": SessionEvent.model_json_schema(),
            "results/*.json": CommandResponse.model_json_schema(),
        },
    }


def durable_state_schema_summary() -> dict[str, object]:
    """Return a compact field map suitable for an agent's first schema read."""
    return {
        "schema_version": 1,
        "query_with": {"json": "jq", "yaml": "yq", "jsonl": "jq"},
        "files": {
            "metadata.json": {
                "purpose": "session source and renderer lifecycle",
                "fields": "schema_version name source_path source_sha256 blender status "
                "created_at updated_at worker_pid",
            },
            "scene.json": {
                "purpose": "complete imported scene manifest",
                "fields": "schema_version source_sha256 source_format units root_bounds "
                "components imported_camera imported_illumination capabilities warnings "
                "normalized_geometry",
                "capabilities": {
                    "hierarchy.preserved": "parent and child component links preserve source "
                    "component ancestry",
                    "hierarchy.flattened": "component paths remain authoritative and retain "
                    "source ancestry, but intermediate non-mesh nodes are encoded only in paths",
                },
            },
            "components.yml": {
                "purpose": "compact stable component index",
                "fields": "schema_version components[].{ref,id,path,parent,name}",
            },
            "state.yml": {
                "purpose": "current camera, lighting, and sparse visual overrides",
                "fields": "schema_version state_sha256 camera "
                "camera_orbit_angles.{azimuth_degrees,elevation_degrees} illumination "
                "components.{default,overrides} results artifacts",
            },
            "checkpoint.json": {
                "purpose": "last acknowledged replay state",
                "fields": "schema_version source_path source_sha256 blender state_sha256 "
                "accepted_commands[]",
            },
            "events.jsonl": {
                "purpose": "one accepted or rejected operation per line",
                "fields": "at request_id op status command result_path error",
            },
            "results/*.json": {
                "purpose": "full operation result envelopes",
                "fields": "request_id op result",
                "result_shape": "the per-op shape of `result` is documented by "
                "`meshprobe schema --kind results`",
            },
        },
        "full_schema": "meshprobe schema --kind state --full",
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _string_warnings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def atomic_text(path: Path, content: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o666 if mode is None else mode,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if mode is not None:
            os.chmod(path, mode)
    except Exception:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def atomic_json(path: Path, value: object, *, mode: int | None = None) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n", mode=mode)


def workspace_root(path: Path | None = None) -> Path:
    base = (path or Path.cwd()).expanduser().resolve()
    return base if base.name == ".meshprobe" else base / ".meshprobe"


class SessionFiles:
    def __init__(self, root: Path, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("session name must contain only letters, digits, '.', '_' or '-'")
        self.workspace = root
        self.name = name
        self.root = root / "sessions" / name
        self.metadata = self.root / "metadata.json"
        self.scene = self.root / "scene.json"
        self.components = self.root / "components.yml"
        self.state = self.root / "state.yml"
        self.checkpoint = self.root / "checkpoint.json"
        self.events = self.root / "events.jsonl"
        self.results = self.root / "results"
        self.snapshots = self.root / "snapshots"
        self.artifacts = self.root / "artifacts"

    def create_directories(self) -> None:
        for path in (self.results, self.snapshots, self.artifacts):
            path.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.workspace.parent))


class SessionService(Protocol):
    @property
    def worker_pid(self) -> int | None: ...

    @property
    def graphics(self) -> GraphicsPlatform | None: ...

    def execute(self, command: Command) -> CommandResponse: ...

    def close(self) -> None: ...

    def kill(self) -> None: ...


class SessionManager:
    """Own durable sessions and lazily recover their Blender workers."""

    def __init__(
        self,
        root: Path,
        *,
        blender: str | None = None,
        timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
        service_factory: Callable[[], SessionService] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.blender = blender
        self.timeout_seconds = timeout_seconds
        self._service_factory = service_factory
        self._services: dict[str, SessionService] = {}
        self._graphics_warned_worker_pids: dict[str, int | None] = {}
        self._lock = threading.RLock()

    def open(
        self,
        name: str,
        source: Path,
        *,
        request_id: str = "open",
        blender: str | None = None,
        unit_scale: float = 1.0,
    ) -> OperationReceipt:
        with self._lock:
            files = SessionFiles(self.root, name)
            selected_blender = blender if blender is not None else self.blender
            command = SceneOpenCommand(
                request_id=request_id,
                op="scene.open",
                source_path=str(source.expanduser().resolve(strict=True)),
                unit_scale=unit_scale,
            )
            service = self._new_service(selected_blender)
            try:
                response = service.execute(command)
                manifest = SceneManifest.model_validate(response.result)
                snapshot = self._snapshot(service)
            except Exception:
                service.kill()
                raise
            existing = self._services.pop(name, None)
            if existing is not None:
                existing.close()
            if files.root.exists():
                shutil.rmtree(files.root)
            files.create_directories()
            self._services[name] = service
            now = utc_now()
            metadata = SessionMetadata(
                name=name,
                source_path=command.source_path,
                source_sha256=manifest.source_sha256,
                blender=selected_blender,
                created_at=now,
                updated_at=now,
                worker_pid=service.worker_pid,
                graphics=getattr(service, "graphics", None),
            )
            checkpoint = SessionCheckpoint(
                source_path=command.source_path,
                source_sha256=manifest.source_sha256,
                blender=selected_blender,
                state_sha256=snapshot.state_sha256,
                unit_scale=command.unit_scale,
            )
            atomic_json(files.metadata, metadata.model_dump(mode="json"))
            atomic_json(files.scene, manifest.model_dump(mode="json"))
            self._write_components(files, manifest)
            self._write_state(files, snapshot)
            atomic_json(files.checkpoint, checkpoint.model_dump(mode="json"))
            result_path = self._write_result(files, command.request_id, response)
            self._event(files, command, "accepted", result_path=result_path)
            graphics = getattr(service, "graphics", None)
            graphics_warnings = graphics.warnings if graphics is not None else ()
            unit_warnings = tuple(
                warning.message
                for warning in manifest.warnings
                if warning.code in UNIT_SUSPICION_WARNING_CODES
            )
            warnings = unit_warnings + graphics_warnings
            self._graphics_warned_worker_pids[name] = service.worker_pid
            return self._receipt(files, command.op, snapshot, result_path, warnings=warnings)

    def execute(
        self,
        name: str,
        command: Command,
        *,
        blender: str | None = None,
    ) -> OperationReceipt:
        if isinstance(command, SceneOpenCommand):
            return self.open(
                name,
                Path(command.source_path),
                request_id=command.request_id,
                blender=blender,
                unit_scale=command.unit_scale,
            )
        with self._lock:
            files = self._require_files(name)
            previous_service = self._services.get(name)
            previous_worker_pid = (
                previous_service.worker_pid if previous_service is not None else None
            )
            try:
                service = self._service(name, files)
                response = service.execute(command)
                command_worker_pid = service.worker_pid
                snapshot = self._snapshot(service)
            except Exception as error:
                self._event(files, command, "rejected", error=str(error))
                raise
            renderer_continuity = RendererContinuity.RECREATED_BEFORE_SNAPSHOT
            if command_worker_pid != service.worker_pid:
                renderer_continuity = RendererContinuity.RECREATED_DURING_SNAPSHOT
            elif service is previous_service and service.worker_pid == previous_worker_pid:
                renderer_continuity = RendererContinuity.PRESERVED
            result_path = self._write_result(files, command.request_id, response)
            self._write_state(
                files,
                snapshot,
                render_style=self._render_style(command, renderer_continuity),
            )
            self._update_checkpoint(files, command, snapshot)
            self._update_metadata(files, status="active", worker_pid=service.worker_pid)
            self._event(files, command, "accepted", result_path=result_path)
            artifacts = self._artifact_paths(command.op, response.result)
            warnings: tuple[str, ...] = ()
            if isinstance(command, RenderImageCommand):
                aspect_warning = self._aspect_ratio_warning(
                    command, self._framed_aspect_ratio(files)
                )
                if aspect_warning is not None:
                    warnings = (*warnings, aspect_warning)
            if self._graphics_warned_worker_pids.get(name) != service.worker_pid:
                graphics = getattr(service, "graphics", None)
                warnings = (*warnings, *(graphics.warnings if graphics is not None else ()))
                self._graphics_warned_worker_pids[name] = service.worker_pid
            match_count = (
                len(response.result)
                if isinstance(command, ComponentFindCommand) and isinstance(response.result, list)
                else None
            )
            warnings = (*warnings, *self._render_warnings(command.op, response.result))
            return self._receipt(
                files,
                command.op,
                snapshot,
                result_path,
                artifacts,
                warnings=warnings,
                components=self._component_references(files, command),
                match_count=match_count,
            )

    def close(self, name: str) -> OperationReceipt:
        with self._lock:
            files = self._require_files(name)
            service = self._services.pop(name, None)
            if service is not None:
                service.close()
            self._update_metadata(files, status="closed", worker_pid=None)
            self._event_raw(files, {"at": utc_now(), "op": "session.close", "status": "accepted"})
            return OperationReceipt(
                session=name,
                op="session.close",
                state_path=files.relative(files.state),
            )

    def kill(self, name: str) -> OperationReceipt:
        with self._lock:
            files = self._require_files(name)
            service = self._services.pop(name, None)
            if service is not None:
                service.kill()
            self._update_metadata(files, status="killed", worker_pid=None)
            self._event_raw(files, {"at": utc_now(), "op": "session.kill", "status": "accepted"})
            return OperationReceipt(
                session=name,
                op="session.kill",
                state_path=files.relative(files.state),
            )

    def close_all(self) -> list[OperationReceipt]:
        return [self.close(name) for name in self.list_names()]

    def kill_all(self) -> list[OperationReceipt]:
        return [self.kill(name) for name in self.list_names()]

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for name in self.list_names():
            files = SessionFiles(self.root, name)
            sessions.append(json.loads(files.metadata.read_text(encoding="utf-8")))
        return sessions

    def list_names(self) -> list[str]:
        sessions_root = self.root / "sessions"
        if not sessions_root.is_dir():
            return []
        return sorted(
            path.name
            for path in sessions_root.iterdir()
            if path.is_dir() and (path / "metadata.json").is_file()
        )

    def resolve_component(self, name: str, value: str) -> str:
        components = self._load_components(name)
        resolved = self._exact_component_id(components, value)
        if resolved is not None:
            return resolved
        raise self._unknown_component_error(value)

    def resolve_component_ids(self, name: str, value: str) -> tuple[str, ...]:
        components = self._load_components(name)
        # An exact ref/ID/name/path wins even when it contains glob metacharacters
        # (e.g. `panel[1]`, `gear?`), which are legal in real component names and
        # paths. An ambiguous literal raises here rather than falling through, so it
        # keeps its ambiguous-name error instead of being reinterpreted as a glob;
        # only a token that matches nothing exactly is read as a glob.
        resolved = self._exact_component_id(components, value)
        if resolved is not None:
            return (resolved,)
        if is_glob_pattern(value):
            pattern = normalize_glob(value)
            matched = sorted(
                (
                    component
                    for component in components
                    if path_glob_match(component["path"], pattern)
                ),
                key=lambda component: str(component["path"]),
            )
            if matched:
                return tuple(str(component["id"]) for component in matched)
            raise ValueError(f"no components match glob pattern: {value}")
        raise self._unknown_component_error(value)

    def _load_components(self, name: str) -> list[dict[str, Any]]:
        files = self._require_files(name)
        payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
        return payload.get("components", []) if isinstance(payload, dict) else []

    @staticmethod
    def _exact_component_id(components: list[dict[str, Any]], value: str) -> str | None:
        """Return the id of an exact ref/ID/path/name match, or None if nothing matches.

        A display name shared by several components is unresolvable, not unmatched, so
        it raises the ambiguous-name error rather than returning None (which a caller
        would otherwise reinterpret as a glob).
        """

        for field in ("ref", "id", "path"):
            for component in components:
                if value == component[field]:
                    return str(component["id"])
        named = [component for component in components if value == component["name"]]
        if len(named) == 1:
            return str(named[0]["id"])
        if named:
            paths = ", ".join(sorted(str(component["path"]) for component in named))
            raise ValueError(
                f"ambiguous component display name {value!r}; matches: {paths}; "
                "use an exact path or stable ID"
            )
        return None

    @staticmethod
    def _unknown_component_error(value: str) -> ValueError:
        return ValueError(
            f"unknown component reference, stable ID, exact display name, or exact path: {value}"
        )

    @staticmethod
    def _component_references(
        files: SessionFiles, command: Command
    ) -> tuple[ResolvedComponentReference, ...]:
        component_ids: dict[str, None] = {}

        def visit(value: object, field: str | None = None) -> None:
            if field == "component_id" and isinstance(value, str):
                component_ids[value] = None
                return
            if field in {"component_ids", "focus_component_ids"} and isinstance(value, list):
                for component_id in value:
                    if isinstance(component_id, str):
                        component_ids[component_id] = None
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, key)
                return
            if isinstance(value, list):
                for child in value:
                    visit(child)

        visit(command.model_dump(mode="json"))
        if not component_ids:
            return ()
        payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
        by_id = {component["id"]: component for component in payload["components"]}
        return tuple(
            ResolvedComponentReference(
                ref=by_id[component_id]["ref"],
                id=component_id,
                name=by_id[component_id]["name"],
                path=by_id[component_id]["path"],
            )
            for component_id in component_ids
        )

    def raw_result(self, receipt: OperationReceipt) -> object:
        if receipt.result_path is None:
            return receipt.model_dump(mode="json")
        return json.loads((self.root.parent / receipt.result_path).read_text(encoding="utf-8"))

    def shutdown(self, *, force: bool = False) -> None:
        with self._lock:
            for service in self._services.values():
                service.kill() if force else service.close()
            self._services.clear()

    def _service(self, name: str, files: SessionFiles) -> SessionService:
        current = self._services.get(name)
        if current is not None:
            return current
        checkpoint, upgraded = _upgrade_checkpoint(
            SessionCheckpoint.model_validate_json(files.checkpoint.read_text(encoding="utf-8"))
        )
        service = self._new_service(checkpoint.blender)
        try:
            opened = service.execute(
                SceneOpenCommand(
                    request_id="recover-open",
                    op="scene.open",
                    source_path=checkpoint.source_path,
                    unit_scale=checkpoint.unit_scale,
                )
            )
            manifest = SceneManifest.model_validate(opened.result)
            if manifest.source_sha256 != checkpoint.source_sha256:
                raise ValueError("source asset bundle changed since the session checkpoint")
            from meshprobe.protocol import COMMAND_ADAPTER

            for raw in checkpoint.accepted_commands:
                service.execute(COMMAND_ADAPTER.validate_python(raw))
            if upgraded:
                atomic_json(files.checkpoint, checkpoint.model_dump(mode="json"))
        except Exception:
            service.kill()
            raise
        self._services[name] = service
        return service

    def _new_service(self, blender: str | None) -> SessionService:
        if self._service_factory is not None:
            return self._service_factory()
        return MeshProbeService(blender=blender, timeout_seconds=self.timeout_seconds)

    @staticmethod
    def _snapshot(service: SessionService) -> SessionSnapshot:
        from meshprobe.protocol import SessionSnapshotCommand

        response = service.execute(
            SessionSnapshotCommand(request_id="checkpoint", op="session.snapshot")
        )
        payload = response.result
        if not isinstance(payload, dict) or "session" not in payload:
            raise ValueError("renderer returned an invalid session snapshot")
        return SessionSnapshot.model_validate(payload["session"])

    # A render's width/height diverging from the aspect ratio the camera was framed with
    # (view.orbit/view.frame/view.set take an aspect_ratio that refits the camera) does not
    # stretch or crop pixels: Blender derives the render's real field of view from the actual
    # resolution, so a mismatch silently reframes the shot instead (extra padding, or a
    # tighter crop, on whichever axis the framing aspect didn't pin). >1% relative difference
    # is treated as a deliberate divergence worth flagging.
    _ASPECT_RATIO_RELATIVE_TOLERANCE = 0.01

    @staticmethod
    def _framed_aspect_ratio(files: SessionFiles) -> float:
        # The framing aspect is the aspect_ratio of the last command that refit the camera
        # (see _refits_camera); the snapshot's camera_diagnostics.aspect_ratio can't be used
        # because view.move and bare view.rotate overwrite it with their own default while
        # leaving the projection framed as before. Defaults to 1.0 (the framing a freshly
        # opened session's camera carries) when nothing has refit the camera yet.
        checkpoint, _ = _upgrade_checkpoint(
            SessionCheckpoint.model_validate_json(files.checkpoint.read_text(encoding="utf-8"))
        )
        for raw in reversed(checkpoint.accepted_commands):
            if _refits_camera(raw):
                return float(raw.get("aspect_ratio", 1.0))
        return 1.0

    @classmethod
    def _aspect_ratio_warning(cls, command: RenderImageCommand, framed_aspect: float) -> str | None:
        render_aspect = command.width / command.height
        relative_difference = abs(render_aspect - framed_aspect) / framed_aspect
        if relative_difference <= cls._ASPECT_RATIO_RELATIVE_TOLERANCE:
            return None
        return (
            f"render.image requested {command.width}x{command.height} (aspect ratio "
            f"{render_aspect:.4g}), but the session camera was last framed for aspect ratio "
            f"{framed_aspect:.4g} by view.orbit/view.frame/view.set (or a view.rotate that set "
            "a projection). The render will reframe to the requested resolution instead of "
            "matching that framing; re-run whichever framing command you used with a matching "
            "--aspect-ratio first if the tight framing matters."
        )

    def _require_files(self, name: str) -> SessionFiles:
        files = SessionFiles(self.root, name)
        if not files.metadata.is_file():
            raise ValueError(f"session does not exist: {name}; run meshprobe open first")
        return files

    @staticmethod
    def _write_components(files: SessionFiles, manifest: SceneManifest) -> None:
        ordered = sorted(manifest.components, key=lambda component: component.path)
        refs = {component.id: f"c{index}" for index, component in enumerate(ordered, start=1)}
        payload = ComponentIndex(
            components=[
                ComponentIndexEntry(
                    ref=refs[component.id],
                    id=component.id,
                    path=component.path,
                    parent=refs.get(component.parent_id) if component.parent_id else None,
                    name=component.display_name,
                )
                for component in ordered
            ]
        )
        atomic_text(
            files.components,
            yaml.safe_dump(payload.model_dump(mode="json"), sort_keys=False),
        )

    @staticmethod
    def _render_style(
        command: Command,
        renderer_continuity: RendererContinuity,
    ) -> RenderStyleState | None:
        if renderer_continuity is RendererContinuity.RECREATED_DURING_SNAPSHOT:
            return RenderStyleState()
        if isinstance(command, RenderImageCommand):
            return RenderStyleState(style=command.style, shaded_edges=command.shaded_edges)
        if isinstance(command, (RenderContactSheetCommand, SessionResetCommand)):
            return RenderStyleState()
        if renderer_continuity is RendererContinuity.RECREATED_BEFORE_SNAPSHOT:
            return RenderStyleState()
        return None

    @staticmethod
    def _write_state(
        files: SessionFiles,
        snapshot: SessionSnapshot,
        *,
        render_style: RenderStyleState | None = None,
    ) -> None:
        default_display = DisplayMode.SHOWN
        default_mark = MarkMode.UNMARKED
        overrides: dict[str, dict[str, str]] = {}
        components_payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
        refs = {item["id"]: item["ref"] for item in components_payload["components"]}
        names = {item["id"]: item["name"] for item in components_payload["components"]}
        for component_id, visual in sorted(snapshot.components.items()):
            entry: dict[str, str] = {}
            if visual.display is not default_display:
                entry["display"] = visual.display.value
            if visual.mark is not default_mark:
                entry["mark"] = visual.mark.value
            if visual.mark_color is not None:
                entry["mark_color"] = visual.mark_color
            if entry:
                entry["name"] = names[component_id]
                overrides[refs[component_id]] = entry
        resolved_render_style = render_style
        if resolved_render_style is None and files.state.exists():
            current = DurableSessionState.model_validate(
                yaml.safe_load(files.state.read_text(encoding="utf-8"))
            )
            resolved_render_style = current.render_style
        payload = DurableSessionState(
            state_sha256=snapshot.state_sha256,
            camera=snapshot.camera,
            illumination=snapshot.illumination,
            render_style=resolved_render_style or RenderStyleState(),
            components=ComponentStateIndex(
                default=ComponentStateDefaults(display=default_display, mark=default_mark),
                overrides={
                    ref: ComponentStateOverride.model_validate(value)
                    for ref, value in overrides.items()
                },
            ),
            results=files.relative(files.results),
            artifacts=files.relative(files.artifacts),
        )
        atomic_text(
            files.state,
            yaml.safe_dump(
                payload.model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
        )

    @staticmethod
    def _write_result(files: SessionFiles, request_id: str, response: CommandResponse) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)[:80] or "result"
        root = files.snapshots if response.op == "session.snapshot" else files.results
        path = root / f"{safe_id}.json"
        if path.exists():
            path = root / f"{safe_id}-{uuid.uuid4().hex[:12]}.json"
        atomic_json(path, response.model_dump(mode="json"))
        return path

    @staticmethod
    def _artifact_paths(operation: str, result: object) -> tuple[str, ...]:
        if operation not in {"render.image", "render.contact_sheet"}:
            return ()
        if not isinstance(result, dict):
            return ()
        found: list[str] = []

        def visit(value: object) -> None:
            if isinstance(value, dict):
                path = value.get("path")
                if isinstance(path, str) and "sha256" in value and "bytes" in value:
                    found.append(path)
                for child in value.values():
                    visit(child)
            if isinstance(value, list):
                for child in value:
                    visit(child)

        visit(result)
        return tuple(dict.fromkeys(found))

    @staticmethod
    def _render_warnings(operation: str, result: object) -> tuple[str, ...]:
        if not isinstance(result, dict):
            return ()
        if operation == "render.image":
            return _string_warnings(result.get("warnings"))
        if operation == "render.contact_sheet":
            collected: list[str] = []
            panels = result.get("panels")
            for panel in panels if isinstance(panels, list) else ():
                if isinstance(panel, dict) and isinstance(panel.get("render"), dict):
                    collected.extend(_string_warnings(panel["render"].get("warnings")))
            return tuple(dict.fromkeys(collected))
        return ()

    @staticmethod
    def _update_checkpoint(
        files: SessionFiles,
        command: Command,
        snapshot: SessionSnapshot,
    ) -> None:
        checkpoint, _ = _upgrade_checkpoint(
            SessionCheckpoint.model_validate_json(files.checkpoint.read_text(encoding="utf-8"))
        )
        if isinstance(command, SessionResetCommand):
            checkpoint.accepted_commands.clear()
        elif command.op in STATE_OPERATIONS:
            accepted = command
            if isinstance(command, IlluminationSetCommand):
                accepted = command.model_copy(update={"illumination": snapshot.illumination})
            checkpoint.accepted_commands.append(accepted.model_dump(mode="json"))
        checkpoint.state_sha256 = snapshot.state_sha256
        atomic_json(files.checkpoint, checkpoint.model_dump(mode="json"))

    @staticmethod
    def _update_metadata(
        files: SessionFiles,
        *,
        status: Literal["active", "closed", "killed"],
        worker_pid: int | None,
    ) -> None:
        metadata = SessionMetadata.model_validate_json(files.metadata.read_text(encoding="utf-8"))
        metadata.status = status
        metadata.worker_pid = worker_pid
        metadata.updated_at = utc_now()
        atomic_json(files.metadata, metadata.model_dump(mode="json"))

    @staticmethod
    def _event(
        files: SessionFiles,
        command: Command,
        status: Literal["accepted", "rejected"],
        *,
        result_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        event: dict[str, object] = {
            "at": utc_now(),
            "request_id": command.request_id,
            "op": command.op,
            "status": status,
            "command": command.model_dump(mode="json"),
        }
        if result_path is not None:
            event["result_path"] = files.relative(result_path)
        if error is not None:
            event["error"] = error
        SessionManager._event_raw(files, event)

    @staticmethod
    def _event_raw(files: SessionFiles, event: dict[str, object]) -> None:
        files.events.parent.mkdir(parents=True, exist_ok=True)
        with files.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

    @staticmethod
    def _receipt(
        files: SessionFiles,
        operation: str,
        snapshot: SessionSnapshot,
        result_path: Path,
        artifacts: tuple[str, ...] = (),
        *,
        warnings: tuple[str, ...] = (),
        components: tuple[ResolvedComponentReference, ...] = (),
        match_count: int | None = None,
    ) -> OperationReceipt:
        return OperationReceipt(
            session=files.name,
            op=operation,
            state_sha256=snapshot.state_sha256,
            result_path=files.relative(result_path),
            state_path=files.relative(files.state),
            artifact_paths=artifacts,
            warnings=warnings,
            components=components,
            match_count=match_count,
        )
