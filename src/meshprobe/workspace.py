"""Durable project-local session state and renderer lifecycle management."""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from meshprobe.controller import DEFAULT_WORKER_TIMEOUT_SECONDS
from meshprobe.models import SceneManifest, SessionSnapshot
from meshprobe.protocol import Command, SceneOpenCommand, SessionResetCommand
from meshprobe.service import CommandResponse, MeshProbeService

STATE_OPERATIONS = frozenset(
    {
        "view.set",
        "view.orbit",
        "illumination.set",
        "component.display",
        "component.mark",
    }
)


class SessionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str
    source_path: str
    source_sha256: str
    status: Literal["active", "closed", "killed"] = "active"
    created_at: str
    updated_at: str
    worker_pid: int | None = None


class SessionCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source_path: str
    source_sha256: str
    state_sha256: str
    accepted_commands: list[dict[str, Any]] = Field(default_factory=list)


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


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


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
        self._lock = threading.RLock()

    def open(self, name: str, source: Path, *, request_id: str = "open") -> OperationReceipt:
        with self._lock:
            files = SessionFiles(self.root, name)
            files.create_directories()
            existing = self._services.pop(name, None)
            if existing is not None:
                existing.close()
            service = self._new_service()
            command = SceneOpenCommand(
                request_id=request_id,
                op="scene.open",
                source_path=str(source.expanduser().resolve(strict=True)),
            )
            try:
                response = service.execute(command)
                manifest = SceneManifest.model_validate(response.result)
                snapshot = self._snapshot(service)
            except Exception:
                service.kill()
                raise
            self._services[name] = service
            now = utc_now()
            metadata = SessionMetadata(
                name=name,
                source_path=command.source_path,
                source_sha256=manifest.source_sha256,
                created_at=now,
                updated_at=now,
                worker_pid=service.worker_pid,
            )
            checkpoint = SessionCheckpoint(
                source_path=command.source_path,
                source_sha256=manifest.source_sha256,
                state_sha256=snapshot.state_sha256,
            )
            atomic_json(files.metadata, metadata.model_dump(mode="json"))
            atomic_json(files.scene, manifest.model_dump(mode="json"))
            self._write_components(files, manifest)
            self._write_state(files, snapshot)
            atomic_json(files.checkpoint, checkpoint.model_dump(mode="json"))
            result_path = self._write_result(files, command.request_id, response)
            self._event(files, command, "accepted", result_path=result_path)
            return self._receipt(files, command.op, snapshot, result_path)

    def execute(self, name: str, command: Command) -> OperationReceipt:
        if isinstance(command, SceneOpenCommand):
            return self.open(name, Path(command.source_path), request_id=command.request_id)
        with self._lock:
            files = self._require_files(name)
            try:
                service = self._service(name, files)
                response = service.execute(command)
                snapshot = self._snapshot(service)
            except Exception as error:
                self._event(files, command, "rejected", error=str(error))
                raise
            result_path = self._write_result(files, command.request_id, response)
            self._write_state(files, snapshot)
            self._update_checkpoint(files, command, snapshot)
            self._update_metadata(files, status="active", worker_pid=service.worker_pid)
            self._event(files, command, "accepted", result_path=result_path)
            artifacts = self._artifact_paths(command.op, response.result)
            return self._receipt(files, command.op, snapshot, result_path, artifacts)

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
        files = self._require_files(name)
        payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
        components = payload.get("components", []) if isinstance(payload, dict) else []
        for component in components:
            if value in {component["ref"], component["id"], component["path"]}:
                return str(component["id"])
        raise ValueError(f"unknown component reference, id, or exact path: {value}")

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
        checkpoint = SessionCheckpoint.model_validate_json(
            files.checkpoint.read_text(encoding="utf-8")
        )
        service = self._new_service()
        try:
            opened = service.execute(
                SceneOpenCommand(
                    request_id="recover-open",
                    op="scene.open",
                    source_path=checkpoint.source_path,
                )
            )
            manifest = SceneManifest.model_validate(opened.result)
            if manifest.source_sha256 != checkpoint.source_sha256:
                raise ValueError("source asset bundle changed since the session checkpoint")
            from meshprobe.protocol import COMMAND_ADAPTER

            for raw in checkpoint.accepted_commands:
                service.execute(COMMAND_ADAPTER.validate_python(raw))
        except Exception:
            service.kill()
            raise
        self._services[name] = service
        return service

    def _new_service(self) -> SessionService:
        if self._service_factory is not None:
            return self._service_factory()
        return MeshProbeService(blender=self.blender, timeout_seconds=self.timeout_seconds)

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

    def _require_files(self, name: str) -> SessionFiles:
        files = SessionFiles(self.root, name)
        if not files.metadata.is_file():
            raise ValueError(f"session does not exist: {name}; run meshprobe open first")
        return files

    @staticmethod
    def _write_components(files: SessionFiles, manifest: SceneManifest) -> None:
        ordered = sorted(manifest.components, key=lambda component: component.path)
        refs = {component.id: f"c{index}" for index, component in enumerate(ordered, start=1)}
        payload = {
            "schema_version": 1,
            "components": [
                {
                    "ref": refs[component.id],
                    "id": component.id,
                    "path": component.path,
                    "parent": refs.get(component.parent_id) if component.parent_id else None,
                    "name": component.display_name,
                }
                for component in ordered
            ],
        }
        atomic_text(files.components, yaml.safe_dump(payload, sort_keys=False))

    @staticmethod
    def _write_state(files: SessionFiles, snapshot: SessionSnapshot) -> None:
        default_display = "shown"
        default_mark = "unmarked"
        overrides: dict[str, dict[str, str]] = {}
        components_payload = yaml.safe_load(files.components.read_text(encoding="utf-8"))
        refs = {item["id"]: item["ref"] for item in components_payload["components"]}
        for component_id, visual in sorted(snapshot.components.items()):
            entry: dict[str, str] = {}
            if visual.display.value != default_display:
                entry["display"] = visual.display.value
            if visual.mark.value != default_mark:
                entry["mark"] = visual.mark.value
            if entry:
                overrides[refs[component_id]] = entry
        payload = {
            "schema_version": 1,
            "state_sha256": snapshot.state_sha256,
            "camera": snapshot.camera.model_dump(mode="json"),
            "illumination": snapshot.illumination.model_dump(mode="json"),
            "components": {
                "default": {"display": default_display, "mark": default_mark},
                "overrides": overrides,
            },
            "results": files.relative(files.results),
            "artifacts": files.relative(files.artifacts),
        }
        atomic_text(files.state, yaml.safe_dump(payload, sort_keys=False))

    @staticmethod
    def _write_result(files: SessionFiles, request_id: str, response: CommandResponse) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)[:80] or "result"
        root = files.snapshots if response.op == "session.snapshot" else files.results
        path = root / f"{safe_id}.json"
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
    def _update_checkpoint(
        files: SessionFiles,
        command: Command,
        snapshot: SessionSnapshot,
    ) -> None:
        checkpoint = SessionCheckpoint.model_validate_json(
            files.checkpoint.read_text(encoding="utf-8")
        )
        if isinstance(command, SessionResetCommand):
            checkpoint.accepted_commands.clear()
        elif command.op in STATE_OPERATIONS:
            checkpoint.accepted_commands.append(command.model_dump(mode="json"))
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
    ) -> OperationReceipt:
        return OperationReceipt(
            session=files.name,
            op=operation,
            state_sha256=snapshot.state_sha256,
            result_path=files.relative(result_path),
            state_path=files.relative(files.state),
            artifact_paths=artifacts,
        )
