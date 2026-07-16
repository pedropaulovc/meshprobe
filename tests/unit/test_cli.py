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
