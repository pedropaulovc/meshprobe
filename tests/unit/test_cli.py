from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click import unstyle
from typer.testing import CliRunner

from meshprobe.cli import app
from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    ComponentFindCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionSnapshotCommand,
    ViewMoveCommand,
    ViewSetCommand,
)
from meshprobe.workspace import OperationReceipt, ResolvedComponentReference

runner = CliRunner()


def test_schema_command_emits_discriminated_union() -> None:
    result = runner.invoke(app, ["schema", "--kind", "commands"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["discriminator"]["propertyName"] == "op"


def test_schema_command_documents_durable_state_files() -> None:
    result = runner.invoke(app, ["schema", "--kind", "state"])

    assert result.exit_code == 0
    schemas = json.loads(result.stdout)["files"]
    assert "components.yml" in schemas
    assert "state.yml" in schemas
    assert "camera" in schemas["state.yml"]["fields"]

    full = runner.invoke(app, ["schema", "--kind", "state", "--full"])
    assert full.exit_code == 0
    assert json.loads(full.stdout)["files"]["state.yml"]["properties"]["camera"]


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
        report_markdown_path=tmp_path / "ergonomics" / "report.md",
    )
    ergonomics_options: dict[str, object] = {}

    def run_ergonomics(*args: object, **kwargs: object) -> object:
        ergonomics_options.update(kwargs)
        return ergonomics

    monkeypatch.setattr("meshprobe.cli.run_ergonomics_pilot", run_ergonomics)

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
        [
            "eval",
            "ergonomics",
            str(corpus.root),
            str(tmp_path / "runs"),
            "--max-pairs",
            "1",
        ],
    )

    assert migrated.exit_code == 0
    assert json.loads(migrated.stdout)["audit"]["destination_version"] == "v2"
    assert audited.exit_code == 0
    assert json.loads(audited.stdout)["source_version"] == "v1"
    assert pilot.exit_code == 0
    assert json.loads(pilot.stdout)["attempts"] == 2
    assert json.loads(pilot.stdout)["qualification"] is False
    assert ergonomics_options["max_pairs"] == 1


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
        return {
            "c2": "component-id",
            "assembly/idler": "component-id",
            "crank-pinion-1": "component-id",
        }[value]

    def read_result(self, receipt: OperationReceipt) -> object:
        return {"result": {"op": receipt.op}}

    def close(self, session: str) -> OperationReceipt:
        self.closed.append(session)
        return OperationReceipt(session=session, op="session.close")

    def close_all(self) -> list[OperationReceipt]:
        self.closed.append("all")
        return [
            OperationReceipt(session="review", op="session.close"),
            OperationReceipt(session="secondary", op="session.close"),
        ]

    def kill(self, session: str) -> OperationReceipt:
        self.killed.append(session)
        return OperationReceipt(session=session, op="session.kill")

    def kill_all(self) -> list[OperationReceipt]:
        self.killed.append("all")
        return [
            OperationReceipt(session="review", op="session.kill"),
            OperationReceipt(session="secondary", op="session.kill"),
        ]

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


def test_cli_resolves_source_and_render_paths_before_daemon_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"model")
    client = FakeClient()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    opened = runner.invoke(app, ["open", "assembly.glb"])
    rendered = runner.invoke(
        app,
        ["render-image", "--output", "relative/render.png", "--samples", "1"],
    )

    assert opened.exit_code == 0
    assert rendered.exit_code == 0
    open_command, render_command = client.commands
    assert isinstance(open_command, SceneOpenCommand)
    assert isinstance(render_command, RenderImageCommand)
    assert open_command.source_path == str(source.resolve())
    assert render_command.output_path == str((tmp_path / "relative/render.png").resolve())


def test_default_render_path_accepts_meshprobe_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".meshprobe"
    root.mkdir()
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    rendered = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "--session",
            "review",
            "render-image",
            "--samples",
            "1",
        ],
    )
    sheet = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "--session",
            "review",
            "render-sheet",
            "c2",
            "--samples",
            "1",
        ],
    )

    assert rendered.exit_code == 0
    assert sheet.exit_code == 0
    image_command = client.commands[-2]
    sheet_command = client.commands[-1]
    assert isinstance(image_command, RenderImageCommand)
    assert isinstance(sheet_command, RenderContactSheetCommand)
    artifacts = root / "sessions" / "review" / "artifacts"
    assert Path(image_command.output_path).parent == artifacts
    assert Path(sheet_command.output_path).parent == artifacts


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

    named = runner.invoke(
        app,
        ["--session", "review", "mark", "crank-pinion-1", "--mode", "highlighted"],
    )
    assert named.exit_code == 0
    assert client.commands[-1].component_ids == ("component-id",)


def test_compact_receipt_includes_readable_and_stable_component_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    component = ResolvedComponentReference(
        ref="c2",
        id="cmp_stable",
        name="crank-pinion-1",
        path="harmonic-analyzer/drive-train/crank-pinion-1",
    )
    monkeypatch.setattr(
        client,
        "execute",
        lambda session, command: OperationReceipt(
            session=session,
            op=command.op,
            components=(component,),
        ),
    )
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["--session", "review", "display", "crank-pinion-1", "--mode", "isolated"],
    )

    assert result.exit_code == 0
    assert "c2 crank-pinion-1" in result.stdout
    assert "harmonic-analyzer/drive-train/crank-pinion-1" in result.stdout
    assert "id=cmp_stable" in result.stdout


def test_view_set_applies_depth_of_field_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    camera = {
        "pose": {"position_mm": [0, 0, 1_000], "orientation_xyzw": [0, 0, 0, 1]},
        "projection": {"mode": "perspective", "focal_length_mm": 20},
    }

    result = runner.invoke(
        app,
        [
            "view-set",
            "--camera-json",
            json.dumps(camera),
            "--aperture-fstop",
            "3.8",
            "--focus-distance",
            "75cm",
        ],
    )

    assert result.exit_code == 0
    command = client.commands[-1]
    assert isinstance(command, ViewSetCommand)
    projection = command.camera.projection
    assert projection.mode == "perspective"
    assert projection.depth_of_field.aperture_fstop == 3.8
    assert projection.depth_of_field.focus_distance_mm == 750


def test_view_move_parses_units_and_camera_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "view-move",
            "--world-z",
            "10cm",
            "--right",
            "-25mm",
            "--forward",
            "0.05m",
        ],
    )

    assert result.exit_code == 0
    command = client.commands[-1]
    assert isinstance(command, ViewMoveCommand)
    assert command.world_delta_mm == (0, 0, 100)
    assert command.camera_delta_mm == (-25, 0, 50)

    invalid = runner.invoke(app, ["view-move", "--up", "ten centimeters"])
    assert invalid.exit_code == 2
    assert "use mm, cm, or m" in invalid.output


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


def test_all_session_lifecycle_output_is_one_machine_readable_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    closed = runner.invoke(app, ["--json", "close", "--all"])
    killed = runner.invoke(app, ["--yaml", "kill", "--all"])
    raw = runner.invoke(app, ["--raw", "close", "--all"])

    assert closed.exit_code == 0
    assert [item["session"] for item in json.loads(closed.stdout)["receipts"]] == [
        "review",
        "secondary",
    ]
    assert [item["session"] for item in yaml.safe_load(killed.stdout)["receipts"]] == [
        "review",
        "secondary",
    ]
    assert json.loads(raw.stdout) == {"results": [{"op": "session.close"}, {"op": "session.close"}]}


def test_json_and_raw_output_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    json_result = runner.invoke(app, ["--json", "snapshot"])
    raw_result = runner.invoke(app, ["--raw", "snapshot"])

    assert json_result.exit_code == 0
    assert json.loads(json_result.stdout)["result_path"].endswith("result.json")
    assert raw_result.exit_code == 0
    assert json.loads(raw_result.stdout) == {"op": "session.snapshot"}


def test_yaml_output_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, ["--yaml", "snapshot"])

    assert result.exit_code == 0
    assert yaml.safe_load(result.stdout)["result_path"].endswith("result.json")


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
    find = client.commands[0]
    assert isinstance(find, ComponentFindCommand)
    assert find.selector.kind.value == "glob"


def test_mark_cli_accepts_srgb_highlight_color(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "--session",
            "review",
            "mark",
            "c2",
            "--mode",
            "highlighted",
            "--color",
            "#FF00ff",
        ],
    )

    assert result.exit_code == 0, result.output
    command = client.commands[0]
    assert command.op == "component.mark"
    assert command.color == "#ff00ff"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--mode", "highlighted", "--color", "magenta"],
        ["--mode", "unmarked", "--color", "#ff00ff"],
    ],
)
def test_mark_cli_reports_invalid_color_as_parameter_error(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, ["--session", "review", "mark", "c2", *arguments])

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert "validation error for ComponentMarkCommand" in result.output
    assert client.commands == []


def test_illumination_cli_accepts_custom_json_and_preset_background_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    illumination_path = tmp_path / "illumination.json"
    illumination_path.write_text(
        json.dumps(
            {
                "preset": "custom",
                "background_rgb": [1, 1, 1],
                "background_strength": 1,
                "ambient_rgb": [0.1, 0.1, 0.1],
                "ambient_strength": 0.15,
            }
        ),
        encoding="utf-8",
    )

    custom = runner.invoke(
        app,
        ["illumination-set", "custom", "--illumination-json", str(illumination_path)],
    )
    preset = runner.invoke(
        app,
        ["illumination-set", "neutral_studio", "--background-rgb", "1", "1", "1"],
    )

    assert custom.exit_code == 0, custom.output
    assert preset.exit_code == 0, preset.output
    custom_command, preset_command = client.commands
    assert isinstance(custom_command, IlluminationSetCommand)
    assert custom_command.illumination.preset == "custom"
    assert custom_command.illumination.background_rgb == (1, 1, 1)
    assert isinstance(preset_command, IlluminationSetCommand)
    assert preset_command.illumination.preset == "neutral_studio"
    assert preset_command.illumination.background_rgb == (1, 1, 1)


@pytest.mark.parametrize(
    "payload",
    [
        {"preset": "custom", "background_rgb": [1, 1, 1]},
        {
            "preset": "custom",
            "background_rgb": [1, 1, 1],
            "ambient_rgb": [0, 0, 0],
            "ambient_strength": 0,
        },
    ],
    ids=["missing-required-field", "zero-output"],
)
def test_illumination_cli_reports_invalid_custom_json_as_parameter_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    illumination_path = tmp_path / "illumination.json"
    illumination_path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        app,
        ["illumination-set", "custom", "--illumination-json", str(illumination_path)],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert client.commands == []


def test_illumination_cli_reports_invalid_preset_override_as_parameter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["illumination-set", "neutral_studio", "--background-rgb", "-1", "0", "0"],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert "background_rgb" in result.output
    assert client.commands == []


def test_illumination_cli_resolves_relative_environment_from_json_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    specification_dir = tmp_path / "specification"
    environment_dir = specification_dir / "environment"
    environment_dir.mkdir(parents=True)
    environment_path = environment_dir / "studio.hdr"
    environment_bytes = b"environment-map"
    environment_path.write_bytes(environment_bytes)
    illumination_path = specification_dir / "illumination.json"
    environment_hash = sha256(environment_bytes).hexdigest()
    illumination_path.write_text(
        json.dumps(
            {
                "preset": "custom",
                "background_rgb": [0, 0, 0],
                "ambient_strength": 0,
                "environment_map": {
                    "path": "environment/studio.hdr",
                    "sha256": environment_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()
    monkeypatch.chdir(invocation_dir)

    result = runner.invoke(
        app,
        ["illumination-set", "custom", "--illumination-json", str(illumination_path)],
    )

    assert result.exit_code == 0, result.output
    command = client.commands[0]
    assert isinstance(command, IlluminationSetCommand)
    environment = command.illumination.environment_map  # type: ignore[union-attr]
    assert environment is not None
    assert environment.path == str(environment_path.resolve())
    assert environment.sha256 == environment_hash


def test_illumination_cli_keeps_absolute_environment_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    environment_path = tmp_path / "studio.hdr"
    environment_bytes = b"absolute-environment-map"
    environment_path.write_bytes(environment_bytes)
    illumination_path = tmp_path / "illumination.json"
    illumination_path.write_text(
        json.dumps(
            {
                "preset": "custom",
                "background_rgb": [0, 0, 0],
                "ambient_strength": 0,
                "environment_map": {
                    "path": str(environment_path),
                    "sha256": sha256(environment_bytes).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["illumination-set", "custom", "--illumination-json", str(illumination_path)],
    )

    assert result.exit_code == 0, result.output
    command = client.commands[0]
    assert isinstance(command, IlluminationSetCommand)
    environment = command.illumination.environment_map  # type: ignore[union-attr]
    assert environment is not None
    assert environment.path == str(environment_path)


def test_find_auto_detects_exact_names_and_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    results = [
        runner.invoke(app, ["find", "target__ControlPoint_Art"]),
        runner.invoke(app, ["find", "assembly/group/target"]),
        runner.invoke(app, ["find", "*Airship*", "--kind", "glob"]),
        runner.invoke(app, ["find", "**/idler*", "--kind", "glob"]),
        runner.invoke(app, ["find", "--name", "target__ControlPoint_Art"]),
    ]

    assert all(result.exit_code == 0 for result in results)
    selectors = [
        command.selector for command in client.commands if isinstance(command, ComponentFindCommand)
    ]
    assert [selector.kind.value for selector in selectors] == [
        "exact_name",
        "exact_path",
        "glob",
        "glob",
        "exact_name",
    ]
    assert [selector.pattern for selector in selectors] == [
        "target__ControlPoint_Art",
        "assembly/group/target",
        "**/*Airship*",
        "**/idler*",
        "target__ControlPoint_Art",
    ]


def test_find_rejects_conflicting_selector_options() -> None:
    both = runner.invoke(app, ["find", "target", "--name", "reference"])
    missing = runner.invoke(app, ["find"])

    assert both.exit_code == 2
    assert "provide either PATTERN or --name" in unstyle(both.output)
    assert missing.exit_code == 2
    assert "provide PATTERN or --name" in unstyle(missing.output)


def test_list_and_delete_data_have_text_and_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    listed = runner.invoke(app, ["list"])
    listed_json = runner.invoke(app, ["--json", "list"])
    deleted = runner.invoke(app, ["delete-data"])
    deleted_json = runner.invoke(app, ["--json", "delete-data"])

    assert "review\tactive\tgraphics=unknown\tassembly.glb" in listed.stdout
    assert json.loads(listed_json.stdout)["sessions"][0]["name"] == "review"
    assert deleted.stdout == "deleted .meshprobe\n"
    assert json.loads(deleted_json.stdout) == {"deleted": ".meshprobe"}


def test_global_output_modes_are_mutually_exclusive() -> None:
    result = runner.invoke(app, ["--json", "--yaml", "snapshot"])

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


@pytest.mark.parametrize(
    "adapter_arguments",
    (
        ["--adapter", "reference", "--agent-command-json", '["agent"]'],
        ["--adapter", "cli"],
        ["--adapter", "mcp", "--agent-command-json", '{"command":"agent"}'],
    ),
)
def test_run_tier_rejects_invalid_adapter_commands(
    tmp_path: Path,
    adapter_arguments: list[str],
) -> None:
    corpus = build_corpus(
        tmp_path / "corpus",
        corpus_version="source-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    manifest = tmp_path / "tier.json"
    manifest.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "eval",
            "run-tier",
            str(corpus.root),
            str(manifest),
            str(tmp_path / "runs"),
            *adapter_arguments,
        ],
    )

    assert result.exit_code == 2
