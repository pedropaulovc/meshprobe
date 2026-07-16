from __future__ import annotations

import json
from pathlib import Path

import yaml

from meshprobe.models import DisplayMode, SceneManifest
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    SceneOpenCommand,
    SessionSnapshotCommand,
)
from meshprobe.service import CommandResponse
from meshprobe.session import InspectionSession
from meshprobe.workspace import SessionManager


class FakeSessionService:
    def __init__(self, manifest: SceneManifest) -> None:
        self.manifest = manifest
        self.session: InspectionSession | None = None
        self.closed = False
        self.killed = False

    @property
    def worker_pid(self) -> int | None:
        return None if self.closed or self.killed else 4242

    def execute(self, command: Command) -> CommandResponse:
        if isinstance(command, SceneOpenCommand):
            self.session = InspectionSession(self.manifest)
            result: object = self.manifest.model_dump(mode="json")
        elif isinstance(command, SessionSnapshotCommand):
            assert self.session is not None
            result = {
                "scene": self.manifest.model_dump(mode="json"),
                "session": self.session.snapshot().model_dump(mode="json"),
            }
        elif isinstance(command, ComponentDisplayCommand):
            assert self.session is not None
            result = self.session.display(command.component_ids, command.mode).model_dump(
                mode="json"
            )
        else:
            raise AssertionError(f"unexpected command: {command.op}")
        return CommandResponse(request_id=command.request_id, op=command.op, result=result)

    def close(self) -> None:
        self.closed = True

    def kill(self) -> None:
        self.killed = True


def test_session_manager_writes_compact_queryable_state(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    manager = SessionManager(tmp_path / ".meshprobe", service_factory=factory)
    receipt = manager.open("review", source)
    session_root = tmp_path / ".meshprobe" / "sessions" / "review"

    assert receipt.result_path is not None
    assert (session_root / "scene.json").is_file()
    assert (session_root / "components.yml").is_file()
    assert (session_root / "state.yml").is_file()
    assert (session_root / "checkpoint.json").is_file()
    assert (session_root / "events.jsonl").is_file()
    components = yaml.safe_load((session_root / "components.yml").read_text(encoding="utf-8"))
    paths = [component["path"] for component in components["components"]]
    assert paths == sorted(paths)
    assert [component["ref"] for component in components["components"]] == ["c1", "c2", "c3"]

    component_id = manager.resolve_component("review", "c2")
    changed = manager.execute(
        "review",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    state = yaml.safe_load((session_root / "state.yml").read_text(encoding="utf-8"))
    assert changed.state_sha256 == state["state_sha256"]
    assert state["components"]["default"] == {"display": "shown", "mark": "unmarked"}
    assert state["components"]["overrides"]["c2"] == {"display": "hidden"}
    checkpoint = json.loads((session_root / "checkpoint.json").read_text(encoding="utf-8"))
    assert [command["op"] for command in checkpoint["accepted_commands"]] == ["component.display"]


def test_closed_and_killed_sessions_recover_from_acknowledged_checkpoint(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    services: list[FakeSessionService] = []

    def factory() -> FakeSessionService:
        service = FakeSessionService(scene_manifest)
        services.append(service)
        return service

    source = tmp_path / "assembly.glb"
    source.write_bytes(b"fixture")
    root = tmp_path / ".meshprobe"
    manager = SessionManager(root, service_factory=factory)
    manager.open("default", source)
    component_id = manager.resolve_component("default", "c1")
    manager.execute(
        "default",
        ComponentDisplayCommand(
            request_id="hide",
            op="component.display",
            component_ids=(component_id,),
            mode=DisplayMode.HIDDEN,
        ),
    )
    manager.kill("default")
    assert services[0].killed

    recovered = SessionManager(root, service_factory=factory)
    receipt = recovered.execute(
        "default",
        SessionSnapshotCommand(request_id="after-kill", op="session.snapshot"),
    )
    payload = recovered.raw_result(receipt)
    assert isinstance(payload, dict)
    session = payload["result"]["session"]
    assert session["components"][component_id]["display"] == "hidden"
    recovered.close("default")
    assert services[-1].closed
