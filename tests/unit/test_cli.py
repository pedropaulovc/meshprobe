from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from meshprobe.cli import app
from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    SceneOpenCommand,
    SessionSnapshotCommand,
)
from meshprobe.workspace import OperationReceipt

runner = CliRunner()


def test_schema_command_emits_discriminated_union() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["discriminator"]["propertyName"] == "op"


def test_eval_commands_generate_and_validate_a_corpus(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "generate",
            str(tmp_path),
            "--version",
            "cli-v1",
            "--family",
            "hidden_clip",
            "--seed-count",
            "1",
        ],
    )
    generated = json.loads(result.stdout)
    assert result.exit_code == 0
    assert generated["models"] == 1
    assert generated["episodes"] == 4

    validated_result = runner.invoke(
        app,
        ["eval", "validate", str(tmp_path / "cli-v1")],
    )
    validated = json.loads(validated_result.stdout)
    assert validated_result.exit_code == 0
    assert validated == generated


def test_eval_orchestration_commands_emit_compact_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = build_corpus(
        tmp_path / "inputs",
        corpus_version="source-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("meshprobe.cli.load_catalog", lambda path: path)
    monkeypatch.setattr("meshprobe.cli.ingest_curated_sources", lambda *args, **kwargs: {})
    monkeypatch.setattr("meshprobe.cli.build_curated_variants", lambda *args, **kwargs: object())
    monkeypatch.setattr("meshprobe.cli.build_curated_corpus", lambda *args, **kwargs: corpus)

    curated = runner.invoke(
        app,
        [
            "eval",
            "curated-generate",
            str(catalog_path),
            str(tmp_path / "work"),
            str(tmp_path / "output"),
        ],
    )
    assert curated.exit_code == 0
    assert json.loads(curated.stdout)["episodes"] == 4

    monkeypatch.setattr("meshprobe.cli.merge_corpora", lambda *args, **kwargs: corpus)
    merged = runner.invoke(
        app,
        [
            "eval",
            "merge",
            str(tmp_path / "merged"),
            str(corpus.root),
            str(corpus.root),
        ],
    )
    assert merged.exit_code == 0
    assert json.loads(merged.stdout)["models"] == 1

    fake_manifest = SimpleNamespace(
        tier="smoke",
        corpus_version="source-v1",
        episodes=("episode",),
        model_sha256={"model.glb": "a" * 64},
    )
    monkeypatch.setattr("meshprobe.cli.current_runtime_pin", lambda blender: object())
    monkeypatch.setattr(
        "meshprobe.cli.pin_standard_tiers",
        lambda *args, **kwargs: (fake_manifest,),
    )
    pinned = runner.invoke(
        app,
        ["eval", "pin", str(corpus.root), str(tmp_path / "manifests")],
    )
    assert pinned.exit_code == 0
    assert json.loads(pinned.stdout)["manifests"][0]["episodes"] == 1

    fake_report = SimpleNamespace(passed_thresholds=True)
    fake_run = SimpleNamespace(
        root=tmp_path / "run",
        completed=1,
        report_path=tmp_path / "report.json",
        report=fake_report,
    )
    run_arguments: dict[str, object] = {}

    def fake_run_tier(**kwargs: object) -> object:
        run_arguments.update(kwargs)
        return fake_run

    monkeypatch.setattr("meshprobe.cli.run_tier", fake_run_tier)
    run = runner.invoke(
        app,
        [
            "eval",
            "run-tier",
            str(corpus.root),
            str(catalog_path),
            str(tmp_path / "runs"),
            "--adapter",
            "mcp",
            "--agent-command-json",
            '["/bin/true"]',
            "--blender",
            "pinned-blender",
        ],
    )
    assert run.exit_code == 0
    assert json.loads(run.stdout)["passed_thresholds"] is True
    runtime_provider = run_arguments["runtime_provider"]
    service_factory = run_arguments["service_factory"]
    assert callable(runtime_provider)
    assert callable(service_factory)

    reference = runner.invoke(
        app,
        [
            "eval",
            "run-tier",
            str(corpus.root),
            str(catalog_path),
            str(tmp_path / "reference-runs"),
            "--adapter",
            "reference",
            "--blender",
            "pinned-blender",
        ],
    )
    assert reference.exit_code == 0


def test_eval_migration_and_ergonomics_commands_emit_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = build_corpus(
        tmp_path / "corpus",
        corpus_version="source-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    audit = SimpleNamespace(source_version="v1", destination_version="v2")
    monkeypatch.setattr("meshprobe.cli.migrate_corpus_v2", lambda *args, **kwargs: (corpus, audit))
    monkeypatch.setattr("meshprobe.cli.audit_migration", lambda *args, **kwargs: audit)
    ergonomics = SimpleNamespace(
        root=tmp_path / "ergonomics",
        attempts=(object(), object()),
        report_path=tmp_path / "ergonomics" / "report.json",
    )
    monkeypatch.setattr("meshprobe.cli.run_ergonomics_pilot", lambda *args, **kwargs: ergonomics)

    migrated = runner.invoke(
        app,
        [
            "eval",
            "migrate",
            str(corpus.root),
            str(tmp_path / "output"),
            "--version",
            "destination-v2",
        ],
    )
    audited = runner.invoke(
        app,
        ["eval", "audit-migration", str(corpus.root), str(corpus.root)],
    )
    pilot = runner.invoke(
        app,
        ["eval", "ergonomics", str(corpus.root), str(tmp_path / "runs")],
    )

    assert migrated.exit_code == 0
    assert json.loads(migrated.stdout)["audit"]["destination_version"] == "v2"
    assert audited.exit_code == 0
    assert json.loads(audited.stdout)["source_version"] == "v1"
    assert pilot.exit_code == 0
    assert json.loads(pilot.stdout)["attempts"] == 2
    assert json.loads(pilot.stdout)["qualification"] is False


class FakeClient:
    def __init__(self) -> None:
        self.commands: list[Command] = []
        self.closed: list[str] = []
        self.killed: list[str] = []

    def execute(self, session: str, command: Command) -> OperationReceipt:
        self.commands.append(command)
        return OperationReceipt(
            session=session,
            op=command.op,
            state_sha256="a" * 64,
            result_path=f".meshprobe/sessions/{session}/results/result.json",
            state_path=f".meshprobe/sessions/{session}/state.yml",
        )

    def resolve_component(self, session: str, value: str) -> str:
        assert session == "review"
        return {"c2": "component-id", "assembly/idler": "component-id"}[value]

    def read_result(self, receipt: OperationReceipt) -> object:
        return {"result": {"op": receipt.op}}

    def close(self, session: str) -> OperationReceipt:
        self.closed.append(session)
        return OperationReceipt(session=session, op="session.close")

    def close_all(self) -> list[OperationReceipt]:
        self.closed.append("all")
        return [OperationReceipt(session="review", op="session.close")]

    def kill(self, session: str) -> OperationReceipt:
        self.killed.append(session)
        return OperationReceipt(session=session, op="session.kill")

    def kill_all(self) -> list[OperationReceipt]:
        self.killed.append("all")
        return [OperationReceipt(session="review", op="session.kill")]

    def list_sessions(self) -> list[dict[str, object]]:
        return [{"name": "review", "status": "active", "source_path": "assembly.glb"}]

    def delete_data(self) -> Path:
        return Path(".meshprobe")


def test_flat_cli_uses_named_session_and_compact_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"model")
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    opened = runner.invoke(app, ["--session", "review", "open", str(source)])
    snapshotted = runner.invoke(app, ["--session", "review", "snapshot"])

    assert opened.exit_code == 0
    assert "session=review op=scene.open" in opened.stdout
    assert snapshotted.exit_code == 0
    assert isinstance(client.commands[0], SceneOpenCommand)
    assert isinstance(client.commands[1], SessionSnapshotCommand)


def test_component_commands_resolve_short_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["--session", "review", "display", "c2", "--mode", "isolated"],
    )

    assert result.exit_code == 0
    command = client.commands[-1]
    assert isinstance(command, ComponentDisplayCommand)
    assert command.component_ids == ("component-id",)


def test_close_and_kill_have_distinct_selected_and_all_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    assert runner.invoke(app, ["--session", "review", "close"]).exit_code == 0
    assert runner.invoke(app, ["--session", "review", "kill"]).exit_code == 0
    assert runner.invoke(app, ["close", "--all"]).exit_code == 0
    assert runner.invoke(app, ["kill", "--all"]).exit_code == 0

    assert client.closed == ["review", "all"]
    assert client.killed == ["review", "all"]


def test_json_and_raw_output_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    json_result = runner.invoke(app, ["--json", "snapshot"])
    raw_result = runner.invoke(app, ["--raw", "snapshot"])

    assert json_result.exit_code == 0
    assert json.loads(json_result.stdout)["result_path"].endswith("result.json")
    assert raw_result.exit_code == 0
    assert json.loads(raw_result.stdout) == {"op": "session.snapshot"}


def test_flat_operation_commands_build_typed_protocol_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    camera = json.dumps(
        {
            "pose": {"position_mm": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]},
            "projection": {"mode": "perspective", "focal_length_mm": 50},
        }
    )
    commands = (
        ["--session", "review", "find", "**/idler*"],
        ["--session", "review", "inspect", "c2"],
        ["--session", "review", "view-set", "--camera-json", camera, "--focus", "c2"],
        [
            "--session",
            "review",
            "view-orbit",
            "--target",
            "0",
            "0",
            "0",
            "--azimuth",
            "45",
            "--elevation",
            "30",
            "--distance",
            "100",
            "--projection-json",
            '{"mode":"orthographic","scale_mm":100}',
        ],
        ["--session", "review", "illumination-set", "raking_left"],
        ["--session", "review", "mark", "c2", "--mode", "highlighted"],
        [
            "--session",
            "review",
            "render-image",
            "--output",
            str(tmp_path / "image.png"),
            "--samples",
            "1",
        ],
        [
            "--session",
            "review",
            "render-sheet",
            "c2",
            "--output",
            str(tmp_path / "sheet.png"),
            "--samples",
            "1",
        ],
        ["--session", "review", "reset"],
    )

    results = [runner.invoke(app, command) for command in commands]

    assert all(result.exit_code == 0 for result in results), [result.output for result in results]
    assert [command.op for command in client.commands] == [
        "component.find",
        "component.inspect",
        "view.set",
        "view.orbit",
        "illumination.set",
        "component.mark",
        "render.image",
        "render.contact_sheet",
        "session.reset",
    ]


def test_list_and_delete_data_have_text_and_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    listed = runner.invoke(app, ["list"])
    listed_json = runner.invoke(app, ["--json", "list"])
    deleted = runner.invoke(app, ["delete-data"])
    deleted_json = runner.invoke(app, ["--json", "delete-data"])

    assert "review\tactive\tassembly.glb" in listed.stdout
    assert json.loads(listed_json.stdout)["sessions"][0]["name"] == "review"
    assert deleted.stdout == "deleted .meshprobe\n"
    assert json.loads(deleted_json.stdout) == {"deleted": ".meshprobe"}


def test_global_output_modes_are_mutually_exclusive() -> None:
    result = runner.invoke(app, ["--json", "--raw", "snapshot"])

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


@pytest.mark.parametrize(
    ("target", "arguments"),
    (
        ("build_corpus", ["eval", "generate", "{root}"]),
        ("validate_corpus", ["eval", "validate", "{root}"]),
        (
            "migrate_corpus_v2",
            ["eval", "migrate", "{root}", "{output}", "--version", "v2"],
        ),
        ("audit_migration", ["eval", "audit-migration", "{root}", "{root}"]),
        (
            "load_catalog",
            ["eval", "curated-generate", "{file}", "{root}", "{output}"],
        ),
        ("merge_corpora", ["eval", "merge", "{output}", "{root}", "{root}"]),
        ("current_runtime_pin", ["eval", "pin", "{root}", "{output}"]),
        ("run_ergonomics_pilot", ["eval", "ergonomics", "{root}", "{output}"]),
    ),
)
def test_eval_commands_surface_domain_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    arguments: list[str],
) -> None:
    source_file = tmp_path / "catalog.json"
    source_file.write_text("{}", encoding="utf-8")

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("deliberate domain error")

    monkeypatch.setattr(f"meshprobe.cli.{target}", fail)
    replacements = {
        "{root}": str(tmp_path),
        "{output}": str(tmp_path / "output"),
        "{file}": str(source_file),
    }
    command = [replacements.get(argument, argument) for argument in arguments]

    result = runner.invoke(app, command)

    assert result.exit_code == 2
    assert "deliberate domain error" in result.output


def test_session_commands_surface_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingClient(FakeClient):
        def execute(self, session: str, command: Command) -> OperationReceipt:
            raise ValueError("execute failed")

        def resolve_component(self, session: str, value: str) -> str:
            raise ValueError("resolve failed")

        def list_sessions(self) -> list[dict[str, object]]:
            raise ValueError("list failed")

        def close(self, session: str) -> OperationReceipt:
            raise ValueError("close failed")

        def kill(self, session: str) -> OperationReceipt:
            raise ValueError("kill failed")

        def delete_data(self) -> Path:
            raise ValueError("delete failed")

    client = FailingClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    results = (
        runner.invoke(app, ["snapshot"]),
        runner.invoke(app, ["--session", "review", "display", "c2", "--mode", "hidden"]),
        runner.invoke(app, ["list"]),
        runner.invoke(app, ["close"]),
        runner.invoke(app, ["kill"]),
        runner.invoke(app, ["delete-data"]),
    )

    assert all(result.exit_code == 2 for result in results)
