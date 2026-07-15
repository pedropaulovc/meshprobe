"""Persistent Blender process controller."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from meshprobe.models import RenderManifest, SceneManifest
from meshprobe.protocol import (
    Command,
    ComponentFindCommand,
    ComponentInspectCommand,
    RenderImageCommand,
    SceneOpenCommand,
)
from meshprobe.selectors import ComponentIndex
from meshprobe.session import SessionSnapshot
from meshprobe.sources import SourceSnapshot, sha256_file, snapshot_source

__all__ = [
    "BlenderController",
    "BlenderWorkerCrashed",
    "BlenderWorkerError",
    "BlenderWorkerTimeout",
    "sha256_file",
]

STATE_OPERATIONS = {
    "view.set",
    "view.orbit",
    "illumination.set",
    "component.display",
    "component.mark",
}


class BlenderWorkerError(RuntimeError):
    """A structured Blender worker error."""


class BlenderWorkerCrashed(BlenderWorkerError):
    """The Blender process stopped before producing a response."""


class BlenderWorkerTimeout(BlenderWorkerError):
    """The Blender process exceeded an operation deadline."""


class BlenderController:
    """Own one factory-clean Blender worker and its line-oriented protocol."""

    def __init__(self, executable: str | Path | None = None, timeout_seconds: float = 30) -> None:
        configured = str(executable or os.environ.get("MESHPROBE_BLENDER", "blender"))
        self.executable = Path(configured)
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._logs: list[str] = []
        self.ready_event: dict[str, Any] | None = None
        self._source_path: Path | None = None
        self._source_sha256: str | None = None
        self._manifest: SceneManifest | None = None
        self._source_snapshot: SourceSnapshot | None = None
        self._accepted_commands: list[tuple[str, dict[str, object]]] = []

    @property
    def logs(self) -> tuple[str, ...]:
        return tuple(self._logs)

    def start(self) -> dict[str, Any]:
        if self._process is not None:
            raise BlenderWorkerError("Blender worker is already started")
        resolved = shutil.which(str(self.executable))
        if resolved is None:
            raise BlenderWorkerError(f"Blender executable not found: {self.executable}")
        self.executable = Path(resolved)
        output_queue: queue.Queue[str | None] = queue.Queue()
        self._lines = output_queue
        worker_path = Path(__file__).with_name("blender") / "worker.py"
        self._process = subprocess.Popen(
            [
                str(self.executable),
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python",
                str(worker_path),
                "--",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        process = self._process
        thread = threading.Thread(
            target=self._read_output,
            args=(process, output_queue),
            name="meshprobe-blender-output",
            daemon=True,
        )
        self._reader_thread = thread
        thread.start()
        try:
            event = self._wait_for(lambda payload: payload.get("event") == "ready")
        except (BlenderWorkerCrashed, BlenderWorkerTimeout):
            self.close()
            raise
        if event.get("protocol_version") != 1:
            self.close()
            raise BlenderWorkerError(
                f"unsupported worker protocol: {event.get('protocol_version')}"
            )
        self.ready_event = event
        return event

    def request(self, operation: str, **arguments: object) -> dict[str, Any]:
        process = self._require_process()
        if process.stdin is None:
            raise BlenderWorkerCrashed("Blender worker stdin is unavailable")
        request_id = uuid.uuid4().hex
        command = {"request_id": request_id, "op": operation, **arguments}
        try:
            process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise BlenderWorkerCrashed(self._crash_message()) from error
        response = self._wait_for(lambda payload: payload.get("request_id") == request_id)
        if not response.get("ok"):
            worker_error = response.get("error", {})
            raise BlenderWorkerError(
                f"{worker_error.get('code', 'worker.error')}: {worker_error.get('message', '')}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise BlenderWorkerError("worker returned a non-object result")
        return result

    def open_scene(self, source_path: str | Path) -> SceneManifest:
        source = Path(source_path).expanduser().resolve(strict=True)
        before = snapshot_source(source)
        self._source_path = None
        self._source_sha256 = None
        self._manifest = None
        self._source_snapshot = None
        self._accepted_commands.clear()
        try:
            result = self.request(
                "scene.open", source_path=str(source), source_sha256=before.sha256
            )
        except (BlenderWorkerCrashed, BlenderWorkerTimeout):
            self.close()
            self.start()
            result = self.request(
                "scene.open", source_path=str(source), source_sha256=before.sha256
            )
        after = snapshot_source(source)
        if before != after:
            raise BlenderWorkerError("source asset bundle changed during read-only import")
        manifest = SceneManifest.model_validate(result)
        if manifest.source_sha256 != before.sha256:
            raise BlenderWorkerError("worker source hash does not match controller source hash")
        self._source_path = source
        self._source_sha256 = manifest.source_sha256
        self._manifest = manifest
        self._source_snapshot = before
        return manifest

    def execute(self, command: Command) -> object:
        if isinstance(command, SceneOpenCommand):
            return self.open_scene(command.source_path)
        if isinstance(command, (ComponentFindCommand, ComponentInspectCommand)):
            if self._manifest is None:
                raise BlenderWorkerError("cannot inspect components before a scene is open")
            index = ComponentIndex(self._manifest)
            if isinstance(command, ComponentFindCommand):
                return [
                    component.model_dump(mode="json") for component in index.find(command.selector)
                ]
            return index.by_id(command.component_id).model_dump(mode="json")
        if isinstance(command, RenderImageCommand):
            return self.render_image(command)
        operation = command.op
        arguments = command.model_dump(mode="json", exclude={"request_id", "op"})
        try:
            result = self.request(operation, **arguments)
        except (BlenderWorkerCrashed, BlenderWorkerTimeout):
            self._recover_session()
            result = self.request(operation, **arguments)
        if operation == "session.reset":
            self._accepted_commands.clear()
            return SessionSnapshot.model_validate(result)
        if operation in STATE_OPERATIONS:
            self._accepted_commands.append((operation, arguments))
            return SessionSnapshot.model_validate(result)
        return result

    def render_image(self, command: RenderImageCommand) -> RenderManifest:
        if self._source_snapshot is None:
            raise BlenderWorkerError("cannot render before a scene is open")
        output = Path(command.output_path).expanduser().resolve()
        source_paths = {asset.path for asset in self._source_snapshot.assets}
        if output in source_paths:
            raise BlenderWorkerError("render output must not overwrite a source asset")
        arguments = command.model_dump(mode="json", exclude={"request_id", "op"})
        arguments["output_path"] = str(output)
        try:
            result = self.request(command.op, **arguments)
        except (BlenderWorkerCrashed, BlenderWorkerTimeout):
            self._recover_session()
            result = self.request(command.op, **arguments)
        after = snapshot_source(self._source_snapshot.assets[0].path)
        if after != self._source_snapshot:
            raise BlenderWorkerError("source asset bundle changed during render")
        manifest = RenderManifest.model_validate(result)
        if manifest.source_sha256 != self._source_sha256:
            raise BlenderWorkerError("render manifest source hash does not match the open scene")
        self._verify_render_artifacts(manifest)
        return manifest

    @staticmethod
    def _verify_render_artifacts(manifest: RenderManifest) -> None:
        artifacts = [manifest.color]
        if manifest.evaluator is not None:
            artifacts.extend(
                (
                    manifest.evaluator.multilayer,
                    manifest.evaluator.component_ids,
                    manifest.evaluator.highlighted,
                )
            )
        for artifact_record in artifacts:
            path = Path(artifact_record.path)
            if not path.is_file():
                raise BlenderWorkerError(f"render artifact is missing: {path}")
            if (
                path.stat().st_size != artifact_record.bytes
                or sha256_file(path) != artifact_record.sha256
            ):
                raise BlenderWorkerError(f"render artifact digest mismatch: {path}")

    def _recover_session(self) -> None:
        if self._source_path is None or self._source_sha256 is None:
            raise BlenderWorkerCrashed("cannot recover a worker before a scene is open")
        source_path = self._source_path
        source_sha256 = self._source_sha256
        accepted_commands = list(self._accepted_commands)
        self.close()
        self.start()
        reopened = self.open_scene(source_path)
        if reopened.source_sha256 != source_sha256:
            self.close()
            self._source_path = None
            self._source_sha256 = None
            self._manifest = None
            self._source_snapshot = None
            raise BlenderWorkerError("source asset bundle changed since the session opened")
        self._accepted_commands = accepted_commands
        for operation, arguments in accepted_commands:
            self.request(operation, **arguments)

    def close(self) -> None:
        process = self._process
        self._process = None
        self.ready_event = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps({"request_id": uuid.uuid4().hex, "op": "session.shutdown"}) + "\n"
                )
                process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        reader_thread = self._reader_thread
        self._reader_thread = None
        if reader_thread is not None and reader_thread is not threading.current_thread():
            reader_thread.join(timeout=2)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _read_output(
        self,
        process: subprocess.Popen[str],
        output_queue: queue.Queue[str | None],
    ) -> None:
        if process.stdout is None:
            output_queue.put(None)
            return
        for line in process.stdout:
            output_queue.put(line.rstrip("\n"))
        output_queue.put(None)

    def _wait_for(self, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BlenderWorkerTimeout(
                    f"Blender worker did not respond within {self.timeout_seconds:g} seconds"
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as error:
                raise BlenderWorkerTimeout(
                    f"Blender worker did not respond within {self.timeout_seconds:g} seconds"
                ) from error
            if line is None:
                raise BlenderWorkerCrashed(self._crash_message())
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._logs.append(line)
                continue
            if isinstance(payload, dict) and predicate(payload):
                return payload
            self._logs.append(line)

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise BlenderWorkerError("Blender worker is not started")
        if self._process.poll() is not None:
            raise BlenderWorkerCrashed(self._crash_message())
        return self._process

    def _crash_message(self) -> str:
        process = self._process
        return_code = process.poll() if process is not None else None
        recent_logs = "\n".join(self._logs[-20:])
        return f"Blender worker exited with code {return_code}. Recent output:\n{recent_logs}"
