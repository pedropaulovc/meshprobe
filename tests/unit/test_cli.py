from __future__ import annotations

import json
import re
import subprocess
import sys
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
from meshprobe.models import (
    CoordinateFrame,
    OrthographicProjection,
    PerspectiveProjection,
    PresetIllumination,
    RenderStyle,
    SensorFit,
)
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentMarkCommand,
    ComponentOcclusionCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SceneOpenCommand,
    SessionResetCommand,
    SessionSnapshotCommand,
    ViewFrameCommand,
    ViewMoveCommand,
    ViewOrbitCommand,
    ViewRotateCommand,
    ViewSetCommand,
)
from meshprobe.selectors import is_glob_pattern, normalize_glob, path_glob_match
from meshprobe.workspace import OperationReceipt, ResolvedComponentReference

runner = CliRunner()


def test_schema_command_emits_discriminated_union() -> None:
    result = runner.invoke(app, ["schema", "--kind", "commands"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["discriminator"]["propertyName"] == "op"


def test_render_commands_accept_orbit_sweep_and_reference_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    reference = tmp_path / "historical.png"
    from PIL import Image

    Image.new("RGB", (8, 8), "black").save(reference)

    sweep = runner.invoke(
        app,
        [
            "--session",
            "review",
            "render-sheet",
            "--azimuth",
            "-5,-20,-35",
            "--elevation",
            "-10,-25",
            "--samples",
            "1",
        ],
    )
    comparison = runner.invoke(
        app,
        [
            "--session",
            "review",
            "render-image",
            "--reference-image",
            str(reference),
            "--comparison",
            "side-by-side",
            "--comparison-output",
            str(tmp_path / "comparison.png"),
        ],
    )

    assert sweep.exit_code == 0, sweep.output
    assert comparison.exit_code == 0, comparison.output
    sheet_command, image_command = client.commands
    assert isinstance(sheet_command, RenderContactSheetCommand)
    assert sheet_command.recipe == "orbit_sweep"
    assert sheet_command.orbit_sweep is not None
    assert sheet_command.orbit_sweep.roll_degrees == (0.0,)
    assert isinstance(image_command, RenderImageCommand)
    assert image_command.comparison is not None
    assert image_command.comparison.mode == "side_by_side"


def test_schema_per_command_lookup_returns_only_that_command() -> None:
    """Regression test for #98: `schema view-orbit` used to not exist at all, forcing a
    grep of the whole `--kind commands` dump for e.g. PerspectiveProjection."""
    result = runner.invoke(app, ["schema", "view-orbit"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    # Just the ViewOrbitCommand fragment, not the whole discriminated union.
    assert payload["title"] == "ViewOrbitCommand"
    assert "discriminator" not in payload
    assert "PerspectiveProjection" in payload["$defs"]

    full_dump = runner.invoke(app, ["schema", "--kind", "commands"])
    assert result.stdout != full_dump.stdout
    assert len(result.stdout) < len(full_dump.stdout)


def test_schema_per_command_lookup_accepts_the_protocol_op_name() -> None:
    result = runner.invoke(app, ["schema", "component.occlusion"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["title"] == "ComponentOcclusionCommand"


def test_schema_per_command_lookup_accepts_a_cli_alias() -> None:
    # "occlusion" is the CLI subcommand name; the op is "component.occlusion".
    result = runner.invoke(app, ["schema", "occlusion"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["title"] == "ComponentOcclusionCommand"


def test_schema_per_command_lookup_defaults_to_results_kind_when_asked() -> None:
    result = runner.invoke(app, ["schema", "view-orbit", "--kind", "results"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["required"] == ["request_id", "op", "result"]
    assert payload["properties"]["op"]["const"] == "view.orbit"


def test_schema_per_command_lookup_supports_yaml_output() -> None:
    result = runner.invoke(app, ["schema", "view-orbit", "--yaml"])
    assert result.exit_code == 0
    payload = yaml.safe_load(result.stdout)
    assert payload["title"] == "ViewOrbitCommand"


def test_schema_unknown_command_name_reports_a_clear_error() -> None:
    result = runner.invoke(app, ["schema", "bogus-command"])
    assert result.exit_code != 0
    output = unstyle(result.output)
    assert "bogus-command" in output
    assert "view-orbit" in output
    assert "schema --kind commands" in output
    # No stack trace leaked to the user.
    assert "Traceback" not in output


def test_schema_kind_state_rejects_a_per_command_lookup() -> None:
    result = runner.invoke(app, ["schema", "view-orbit", "--kind", "state"])
    assert result.exit_code != 0
    assert "Traceback" not in unstyle(result.output)


def _property_description(envelope: dict[str, object], model: str, field: str) -> str:
    defs = envelope.get("$defs", {})
    assert isinstance(defs, dict)
    properties = defs[model]["properties"]
    description = properties[field].get("description", "")
    assert isinstance(description, str)
    return description


def test_schema_results_kind_documents_result_envelope_fields() -> None:
    result = runner.invoke(app, ["schema", "--kind", "results"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    # Every command op has a documented result shape.
    assert {
        "render.image",
        "render.contact_sheet",
        "component.occlusion",
        "view.orbit",
        "scene.open",
    } <= payload.keys()

    # The per-op schema is the CommandResponse envelope pinned to that op.
    render_envelope = payload["render.image"]
    assert render_envelope["required"] == ["request_id", "op", "result"]
    assert render_envelope["properties"]["op"]["const"] == "render.image"

    # Render luminance exposure block.
    assert _property_description(render_envelope, "LuminanceSummary", "median").strip()

    # Projected component frame bounds and camera intrinsics from an occlusion query.
    occlusion = payload["component.occlusion"]
    assert _property_description(occlusion, "CameraDiagnostics", "projected_bounds").strip()
    assert _property_description(occlusion, "ProjectedComponentBounds", "minimum_image_xy").strip()
    assert _property_description(occlusion, "PerspectiveProjection", "focal_length_mm").strip()

    # Contact-sheet panel captions, callouts, and the occlusion-removal trace.
    sheet = payload["render.contact_sheet"]
    assert _property_description(sheet, "ContactSheetPanel", "caption").strip()
    assert _property_description(sheet, "ContactSheetCallout", "image_xy").strip()
    assert _property_description(sheet, "OccluderRemovalStep", "visible_fraction_after").strip()


def test_schema_results_have_no_dangling_definition_references() -> None:
    result = runner.invoke(app, ["schema", "--kind", "results"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    for op, envelope in payload.items():
        root_definitions = set(envelope.get("$defs", {}))
        referenced = set(re.findall(r"#/\$defs/(\w+)", json.dumps(envelope)))
        missing = sorted(referenced - root_definitions)
        assert not missing, f"{op} result schema references undefined $defs: {missing}"


def test_schema_all_full_surfaces_result_envelope_fields() -> None:
    result = runner.invoke(app, ["schema", "--kind", "all", "--full"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"commands", "results", "state"}

    rendered = result.stdout
    for field in ("luminance", "projected_bounds", "caption", "callouts", "image_xy"):
        assert field in rendered


def test_render_image_help_advertises_luminance() -> None:
    result = runner.invoke(app, ["render-image", "--help"])
    assert result.exit_code == 0
    assert "luminance" in " ".join(unstyle(result.stdout).split())


def test_occlusion_help_advertises_projected_bounds() -> None:
    result = runner.invoke(app, ["occlusion", "--help"])
    assert result.exit_code == 0
    assert "projected_bounds" in " ".join(unstyle(result.stdout).split())


def test_view_orbit_help_advertises_camera_intrinsics() -> None:
    result = runner.invoke(app, ["view-orbit", "--help"])
    assert result.exit_code == 0
    assert "focal_length_mm" in " ".join(unstyle(result.stdout).replace("│", " ").split())


def test_render_sheet_help_advertises_occlusion_analysis() -> None:
    result = runner.invoke(app, ["render-sheet", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(unstyle(result.stdout).split())
    assert "automatic occlusion analysis" in help_text
    assert "Deep pink marks the focus, not the occluders" in help_text
    assert "one combined focus" in help_text
    assert "--raw" in help_text


def test_mark_help_explains_mode_semantics() -> None:
    result = runner.invoke(app, ["mark", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(unstyle(result.stdout).split())
    for description in (
        "unmarked restores source materials",
        "selected applies cyan",
        "highlighted applies deep pink",
        "labeled applies gold and adds the component name",
    ):
        assert description in help_text


def test_illumination_help_lists_every_preset() -> None:
    result = runner.invoke(app, ["illumination-set", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(unstyle(result.stdout).split())
    for preset in (
        "neutral_studio",
        "high_key",
        "raking_left",
        "raking_right",
        "backlit",
        "flat_diagnostic",
        "custom",
    ):
        assert preset in help_text


@pytest.mark.parametrize(
    "command",
    (
        "open",
        "view-set",
        "view-orbit",
        "view-frame",
        "view-move",
        "view-rotate",
        "illumination-set",
        "display",
        "mark",
        "reset",
    ),
)
def test_visual_mutation_help_offers_opt_in_render(command: str) -> None:
    result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "140"})

    assert result.exit_code == 0
    help_text = " ".join(unstyle(result.stdout).split())
    assert "--render" in help_text
    assert "resulting state in this CLI call" in help_text


def test_render_image_style_help_documents_shaded_edges_cost() -> None:
    result = runner.invoke(app, ["render-image", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(unstyle(result.stdout).split())
    assert "shaded_edges" in help_text
    assert "slower" in help_text.casefold()
    assert "visible components" in help_text


def test_occlusion_command_uses_current_camera_query(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["--session", "review", "occlusion", "c2", "--max-samples", "512"],
    )

    assert result.exit_code == 0
    command = client.commands[-1]
    assert isinstance(command, ComponentOcclusionCommand)
    assert command.component_ids == ("component-id",)
    assert command.max_samples_per_component == 512


def test_raw_render_still_surfaces_receipt_warnings_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --raw emits only the operation result JSON on stdout, but receipt warnings (e.g. the
    # aspect-ratio mismatch) must still reach the user on stderr rather than being silently
    # dropped by the raw early return.
    client = FakeClient()
    client.warnings = ("render.image requested 512x512 (aspect ratio 1), but ...",)
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["--session", "review", "--raw", "render-image", "--width", "512", "--height", "512"],
    )

    assert result.exit_code == 0, result.output
    assert "warning: render.image requested 512x512" in result.stderr
    assert "warning:" not in result.stdout


def test_occlusion_command_expands_glob_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, ["--session", "review", "occlusion", "**/*"])

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, ComponentOcclusionCommand)
    assert command.component_ids == ("id-root", "id-clip", "id-idler")


def test_render_sheet_command_expands_glob_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["--session", "review", "render-sheet", "**/idler", "--output", str(tmp_path / "s.png")],
    )

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, RenderContactSheetCommand)
    assert command.focus_component_ids == ("id-idler",)


def test_occlusion_glob_with_no_match_reports_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, ["--session", "review", "occlusion", "**/missing*"])

    assert result.exit_code != 0
    assert "no components match glob pattern" in result.output


def test_schema_command_documents_durable_state_files() -> None:
    result = runner.invoke(app, ["schema", "--kind", "state"])

    assert result.exit_code == 0
    schemas = json.loads(result.stdout)["files"]
    assert "components.yml" in schemas
    assert "state.yml" in schemas
    assert "camera" in schemas["state.yml"]["fields"]
    assert "camera_orbit_angles" in schemas["state.yml"]["fields"]

    full = runner.invoke(app, ["schema", "--kind", "state", "--full"])
    assert full.exit_code == 0
    properties = json.loads(full.stdout)["files"]["state.yml"]["properties"]
    assert properties["camera"]
    assert properties["camera_orbit_angles"]


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
    monkeypatch.setattr("meshprobe.cli.migrate_corpus_v3", lambda *args, **kwargs: (corpus, audit))
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
        self.match_count: int | None = None
        self.find_results: list[object] | None = None
        self.warnings: tuple[str, ...] = ()
        self.framed_aspect_ratio_value = 1.0

    def execute(self, session: str, command: Command) -> OperationReceipt:
        self.commands.append(command)
        return OperationReceipt(
            session=session,
            op=command.op,
            state_sha256="a" * 64,
            result_path=f".meshprobe/sessions/{session}/results/result.json",
            state_path=f".meshprobe/sessions/{session}/state.yml",
            match_count=self.match_count if isinstance(command, ComponentFindCommand) else None,
            warnings=self.warnings,
        )

    GLOB_COMPONENTS = (
        ("id-root", "assembly"),
        ("id-clip", "assembly/cover/clip"),
        ("id-idler", "assembly/drive/idler"),
    )

    def resolve_component(self, session: str, value: str) -> str:
        assert session == "review"
        return {
            "c2": "component-id",
            "assembly/idler": "component-id",
            "crank-pinion-1": "component-id",
        }[value]

    def resolve_component_ids(self, session: str, value: str) -> tuple[str, ...]:
        assert session == "review"
        if not is_glob_pattern(value):
            return (self.resolve_component(session, value),)
        pattern = normalize_glob(value)
        matched = tuple(
            component_id
            for component_id, path in sorted(self.GLOB_COMPONENTS, key=lambda entry: entry[1])
            if path_glob_match(path, pattern)
        )
        if not matched:
            raise ValueError(f"no components match glob pattern: {value}")
        return matched

    def read_result(self, receipt: OperationReceipt) -> object:
        if receipt.op == "component.find" and self.find_results is not None:
            return {"result": self.find_results}
        if receipt.op == "session.undo":
            return {"result": {"undone": 1, "remaining": 2, "state_sha256": "a" * 64}}
        return {"result": {"op": receipt.op}}

    def framed_aspect_ratio(self, session: str) -> float:
        assert session == "review"
        return self.framed_aspect_ratio_value

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


def test_open_omits_aspect_ratio_unless_explicitly_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"model")
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    default_open = runner.invoke(app, ["--session", "review", "open", str(source)])
    explicit_open = runner.invoke(
        app, ["--session", "review", "open", str(source), "--aspect-ratio", "2.0"]
    )

    assert default_open.exit_code == 0
    assert explicit_open.exit_code == 0
    default_command, explicit_command = client.commands
    # Omitting --aspect-ratio must leave the field unset so the client filters it out
    # of the wire payload — an older daemon validating with extra="forbid" still accepts.
    assert "aspect_ratio" not in default_command.model_fields_set
    assert isinstance(explicit_command, SceneOpenCommand)
    assert "aspect_ratio" in explicit_command.model_fields_set
    assert explicit_command.aspect_ratio == 2.0


def test_reset_omits_aspect_ratio_unless_explicitly_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    default_reset = runner.invoke(app, ["--session", "review", "reset"])
    explicit_reset = runner.invoke(app, ["--session", "review", "reset", "--aspect-ratio", "0.5"])

    assert default_reset.exit_code == 0
    assert explicit_reset.exit_code == 0
    default_command, explicit_command = client.commands
    assert "aspect_ratio" not in default_command.model_fields_set
    assert isinstance(explicit_command, SessionResetCommand)
    assert "aspect_ratio" in explicit_command.model_fields_set
    assert explicit_command.aspect_ratio == 0.5


def test_undo_uses_an_optional_positional_count_and_compact_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, ["--session", "review", "undo"])
    counted = runner.invoke(app, ["--session", "review", "undo", "3"])

    assert result.exit_code == 0
    assert "op=session.undo undone=1 remaining=2" in result.stdout
    assert counted.exit_code == 0
    assert client.commands[0].count == 1  # type: ignore[union-attr]
    assert client.commands[1].count == 3  # type: ignore[union-attr]


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
    assert render_command.style is RenderStyle.SCREEN_EDGES


@pytest.mark.parametrize("terminal_width", [80, 120])
def test_render_help_explains_style_policy_at_common_widths(terminal_width: int) -> None:
    result = runner.invoke(
        app,
        ["render-image", "--help"],
        env={"COLUMNS": str(terminal_width)},
    )

    assert result.exit_code == 0
    help_text = " ".join(unstyle(result.output).split())
    assert "Choose a style:" in help_text
    assert "screen_edges (default): Fast GPU depth/normal edge pass" in help_text
    assert "shaded_edges: Slower geometry-aware Freestyle lines" in help_text
    assert "shaded: Unmodified shaded image with no edge overlay" in help_text
    assert "--style STYLE" in help_text
    assert "See the style guide" in help_text
    assert (
        "cost increases with the number of visible components, not output resolution" in help_text
    )
    assert "[default: screen_edges]" in help_text


def test_render_cli_builds_resolved_shaded_edges_style(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "render-image",
            "--output",
            str(tmp_path / "edges.png"),
            "--style",
            "shaded_edges",
            "--edge-color",
            "#101820",
            "--edge-width",
            "2.5",
            "--crease-angle",
            "45",
            "--edge-types",
            "silhouette,crease,material_boundary",
        ],
    )

    assert result.exit_code == 0, result.output
    command = client.commands[0]
    assert isinstance(command, RenderImageCommand)
    assert command.style == "shaded_edges"
    assert command.shaded_edges.line_color == "#101820"
    assert command.shaded_edges.line_width == 2.5
    assert command.shaded_edges.crease_angle_degrees == 45
    assert command.shaded_edges.edge_types == (
        "silhouette",
        "crease",
        "material_boundary",
    )


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
    assert image_command.shaded_edges.crease_angle_degrees == 120
    artifacts = root / "sessions" / "review" / "artifacts"
    assert Path(image_command.output_path).parent == artifacts
    assert Path(sheet_command.output_path).parent == artifacts

    render_help = runner.invoke(app, ["render-image", "--help"])
    assert render_help.exit_code == 0
    render_help_text = " ".join(unstyle(render_help.output).split())
    assert "Maximum face angle in" in render_help_text
    assert "degrees classified as a" in render_help_text
    assert "[default: 120]" in render_help_text


def test_render_defaults_favor_inspection_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    image = runner.invoke(app, ["--session", "review", "render-image", "--samples", "1"])
    sheet = runner.invoke(app, ["--session", "review", "render-sheet", "c2", "--samples", "1"])

    assert image.exit_code == 0, image.output
    assert sheet.exit_code == 0, sheet.output
    image_command, sheet_command = client.commands
    assert isinstance(image_command, RenderImageCommand)
    assert isinstance(sheet_command, RenderContactSheetCommand)
    assert (image_command.width, image_command.height) == (2576, 2576)
    assert (sheet_command.panel_width, sheet_command.panel_height) == (1200, 1200)


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
    mark_command = client.commands[-1]
    assert isinstance(mark_command, ComponentMarkCommand)
    assert mark_command.component_ids == ("component-id",)


@pytest.mark.parametrize("output_option", ("--json", "--yaml", "--raw"))
def test_visual_mutation_can_render_in_one_machine_readable_invocation(
    output_option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [output_option, "--session", "review", "mark", "c2", "--mode", "selected", "--render"],
    )

    assert result.exit_code == 0, result.output
    assert [command.op for command in client.commands] == ["component.mark", "render.image"]
    render_command = client.commands[-1]
    assert isinstance(render_command, RenderImageCommand)
    assert Path(render_command.output_path).name.startswith("render-")
    if output_option == "--yaml":
        payload = yaml.safe_load(result.stdout)
    else:
        payload = json.loads(result.stdout)
    key = "results" if output_option == "--raw" else "receipts"
    assert [item["op"] for item in payload[key]] == ["component.mark", "render.image"]


@pytest.mark.parametrize(
    ("aspect_ratio", "dimensions"),
    (
        (2.0, (2576, 1288)),
        (0.5, (1288, 2576)),
        (40.25, (2576, 64)),
        (1 / 40.25, (64, 2576)),
    ),
)
def test_visual_mutation_render_preserves_landscape_and_portrait_framing(
    aspect_ratio: float,
    dimensions: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    client.framed_aspect_ratio_value = aspect_ratio
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "--session",
            "review",
            "view-frame",
            "c2",
            "--aspect-ratio",
            str(aspect_ratio),
            "--render",
        ],
    )

    assert result.exit_code == 0, result.output
    frame_command, render_command = client.commands
    assert isinstance(frame_command, ViewFrameCommand)
    assert frame_command.aspect_ratio == aspect_ratio
    assert isinstance(render_command, RenderImageCommand)
    assert (render_command.width, render_command.height) == dimensions


@pytest.mark.parametrize("aspect_ratio", (0.01, 100.0))
def test_visual_mutation_render_rejects_aspects_outside_default_dimension_budget(
    aspect_ratio: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    client.framed_aspect_ratio_value = aspect_ratio
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "--session",
            "review",
            "view-frame",
            "c2",
            "--aspect-ratio",
            str(aspect_ratio),
            "--render",
        ],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).split())
    assert "view.frame succeeded" in output
    assert "outside the" in output
    assert "supported by" in output
    assert "--render" in output
    assert "render-image" in output
    assert "with explicit" in output
    assert [command.op for command in client.commands] == ["view.frame"]


def test_visual_mutation_reports_partial_success_when_requested_render_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    execute = client.execute

    def fail_render(session: str, command: Command) -> OperationReceipt:
        if isinstance(command, RenderImageCommand):
            raise RuntimeError("renderer unavailable")
        return execute(session, command)

    monkeypatch.setattr(client, "execute", fail_render)
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["--session", "review", "mark", "c2", "--mode", "selected", "--render"],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).split())
    assert "component.mark succeeded" in output
    assert "requested render failed:" in output
    assert "renderer unavailable" in output
    assert [command.op for command in client.commands] == ["component.mark"]


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


def test_view_rotate_exposes_source_frame_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "--session",
            "review",
            "view-rotate",
            "--target",
            "0",
            "0",
            "0",
            "--axis",
            "y",
            "--degrees",
            "145",
            "--frame",
            "source",
            "--basis-json",
            '{"x":[0,1,0],"y":[-1,0,0],"z":[0,0,1]}',
        ],
    )

    assert result.exit_code == 0
    command = client.commands[-1]
    assert isinstance(command, ViewRotateCommand)
    assert command.frame == CoordinateFrame.SOURCE
    assert command.axis == "y"
    assert command.basis.x == (0, 1, 0)
    assert command.degrees == 145

    help_result = runner.invoke(app, ["view-orbit", "--help"])
    assert help_result.exit_code == 0
    help_text = " ".join(unstyle(help_result.stdout).replace("│", " ").split())
    assert "Set an absolute orbit in right-handed, Z-up MeshProbe world coordinates." in help_text
    assert "Absolute degrees from +X toward +Y about world +Z." in help_text
    assert "Angles are degrees, not relative deltas." in help_text


def test_view_rotate_defaults_to_source_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "--session",
            "review",
            "view-rotate",
            "--target",
            "0",
            "0",
            "0",
            "--axis",
            "y",
            "--degrees",
            "145",
        ],
    )

    assert result.exit_code == 0
    command = client.commands[-1]
    assert isinstance(command, ViewRotateCommand)
    assert command.frame == CoordinateFrame.SOURCE

    world = runner.invoke(
        app,
        [
            "--session",
            "review",
            "view-rotate",
            "--target",
            "0",
            "0",
            "0",
            "--axis",
            "y",
            "--degrees",
            "145",
            "--frame",
            "world",
        ],
    )
    assert world.exit_code == 0
    world_command = client.commands[-1]
    assert isinstance(world_command, ViewRotateCommand)
    assert world_command.frame == CoordinateFrame.WORLD

    help_result = runner.invoke(app, ["view-rotate", "--help"])
    assert help_result.exit_code == 0
    help_text = " ".join(unstyle(help_result.stdout).replace("│", " ").split())
    assert "the model's own axes, the default" in help_text


def test_view_rotate_accepts_camera_frame_and_rejects_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    base = ["--session", "review", "view-rotate", "--target", "0", "0", "0", "--degrees", "15"]

    camera = runner.invoke(app, [*base, "--axis", "x", "--frame", "camera"])
    assert camera.exit_code == 0
    camera_command = client.commands[-1]
    assert isinstance(camera_command, ViewRotateCommand)
    assert camera_command.frame == CoordinateFrame.CAMERA

    component = runner.invoke(app, [*base, "--axis", "x", "--frame", "component"])
    assert component.exit_code == 2
    assert "component" in component.output

    # --help advertises exactly the honored --frame choices; it no longer promises
    # component there (the word can still appear elsewhere, e.g. --focus's
    # view-frame disclaimer).
    help_result = runner.invoke(app, ["view-rotate", "--help"])
    help_text = " ".join(unstyle(help_result.stdout).replace("│", " ").split())
    assert "camera" in help_text
    assert "[source|world|camera]" in help_text


def test_view_frame_builds_component_framing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "--session",
            "review",
            "view-frame",
            "c2",
            "--azimuth",
            "60",
            "--elevation",
            "20",
            "--margin",
            "1.4",
            "--projection-json",
            '{"mode":"orthographic","scale_mm":100}',
        ],
    )

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, ViewFrameCommand)
    assert command.focus_component_ids == ("component-id",)
    assert command.azimuth_degrees == 60
    assert command.elevation_degrees == 20
    assert command.margin == 1.4
    assert command.projection.mode == "orthographic"


def test_view_frame_defaults_to_isometric_perspective(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, ["--session", "review", "view-frame", "c2"])

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, ViewFrameCommand)
    assert command.azimuth_degrees == 45
    assert command.elevation_degrees == 30
    assert command.margin == 1.25
    assert command.projection.mode == "perspective"


def test_view_focus_help_disclaims_reframing(monkeypatch: pytest.MonkeyPatch) -> None:
    for verb in ("view-set", "view-orbit", "view-move", "view-rotate"):
        result = runner.invoke(app, [verb, "--help"])
        assert result.exit_code == 0
        help_text = " ".join(unstyle(result.stdout).replace("│", " ").split())
        assert "does NOT move or reframe the camera" in help_text
        assert "use view-frame to frame a component" in help_text


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


def test_version_flag_prints_installed_version_and_exits() -> None:
    from meshprobe import __version__

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"meshprobe {__version__}"


def test_version_flag_takes_precedence_over_a_subcommand() -> None:
    result = runner.invoke(app, ["--version", "schema"])

    assert result.exit_code == 0
    assert result.stdout.startswith("meshprobe ")


def test_python_dash_m_meshprobe_matches_the_console_script() -> None:
    """`python -m meshprobe` must work like `meshprobe` (issue #94)."""

    module_result = subprocess.run(
        [sys.executable, "-m", "meshprobe", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    console_result = subprocess.run(
        ["meshprobe", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert module_result.returncode == 0
    assert console_result.returncode == 0

    def _without_usage_line(stdout: str) -> list[str]:
        # The usage line's program name differs (`python -m meshprobe` vs
        # `meshprobe`); everything else about the rendered help text is identical.
        return [line for line in stdout.splitlines() if "Usage:" not in line]

    assert _without_usage_line(module_result.stdout) == _without_usage_line(console_result.stdout)


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


def _orbit_args(*extra: str) -> list[str]:
    return [
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
        *extra,
    ]


def test_view_orbit_without_projection_flags_defers_to_session_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #96: the common case orbits the existing camera without restating it."""

    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, _orbit_args())

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, ViewOrbitCommand)
    assert command.projection is None


def test_view_orbit_focal_length_shorthand_builds_perspective_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, _orbit_args("--focal-length", "100"))

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, ViewOrbitCommand)
    assert command.projection == PerspectiveProjection(focal_length_mm=100)


def test_view_orbit_ortho_scale_shorthand_builds_orthographic_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, _orbit_args("--ortho-scale", "600"))

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, ViewOrbitCommand)
    assert command.projection == OrthographicProjection(scale_mm=600)


def test_view_orbit_projection_json_still_builds_exact_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--projection-json keeps working unchanged for the exact-camera case."""

    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        _orbit_args(
            "--projection-json",
            '{"mode":"perspective","focal_length_mm":85,"sensor_width_mm":36,'
            '"sensor_height_mm":24,"sensor_fit":"horizontal","near_clip_mm":1,'
            '"far_clip_mm":10000,"depth_of_field":{"mode":"disabled"}}',
        ),
    )

    assert result.exit_code == 0, result.output
    command = client.commands[-1]
    assert isinstance(command, ViewOrbitCommand)
    assert command.projection == PerspectiveProjection(
        focal_length_mm=85,
        sensor_width_mm=36,
        sensor_height_mm=24,
        sensor_fit=SensorFit.HORIZONTAL,
        near_clip_mm=1,
        far_clip_mm=10_000,
    )


def test_view_orbit_projection_flags_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    both_shorthands = runner.invoke(
        app, _orbit_args("--focal-length", "50", "--ortho-scale", "600")
    )
    assert both_shorthands.exit_code != 0
    assert not client.commands

    json_and_shorthand = runner.invoke(
        app,
        _orbit_args(
            "--projection-json",
            '{"mode":"orthographic","scale_mm":100}',
            "--focal-length",
            "50",
        ),
    )
    assert json_and_shorthand.exit_code != 0
    assert not client.commands


def test_view_orbit_help_advertises_projection_shorthands() -> None:
    result = runner.invoke(app, ["view-orbit", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(unstyle(result.stdout).replace("│", " ").split())
    assert "--focal-length" in help_text
    assert "--ortho-scale" in help_text
    assert "current projection" in help_text


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


def test_illumination_cli_accepts_display_referred_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        ["illumination-set", "neutral_studio", "--background-srgb", "1", "1", "1"],
    )

    assert result.exit_code == 0, result.output
    (command,) = client.commands
    assert isinstance(command, IlluminationSetCommand)
    assert isinstance(command.illumination, PresetIllumination)
    assert command.illumination.background_srgb == (1, 1, 1)
    assert command.illumination.background_rgb is None


def test_illumination_cli_rejects_display_and_linear_background_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app,
        [
            "illumination-set",
            "neutral_studio",
            "--background-srgb",
            "1",
            "1",
            "1",
            "--background-rgb",
            "0.5",
            "0.5",
            "0.5",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert client.commands == []


def test_illumination_cli_rejects_display_background_with_custom_json(
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
                "ambient_rgb": [0.1, 0.1, 0.1],
                "ambient_strength": 0.15,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "illumination-set",
            "custom",
            "--illumination-json",
            str(illumination_path),
            "--background-srgb",
            "1",
            "1",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert client.commands == []


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


def test_find_receipt_distinguishes_zero_matches_from_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    client.match_count = 3
    hit = runner.invoke(app, ["find", "*idler*", "--kind", "glob"])
    client.match_count = 0
    miss = runner.invoke(app, ["find", "nonexistent-widget-xyz"])

    assert hit.exit_code == 0
    assert "matches=3" in hit.stdout
    assert "no components matched" not in hit.output

    assert miss.exit_code == 0
    assert "matches=0" in miss.stdout
    assert "warning: no components matched" in miss.stderr
    assert unstyle(miss.stdout) != unstyle(hit.stdout)


def test_find_derives_match_count_from_result_when_daemon_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    # A daemon still running the pre-upgrade protocol never sets match_count on the
    # receipt at all, so the CLI must fall back to the persisted result file.
    client.match_count = None
    client.find_results = []
    miss = runner.invoke(app, ["find", "nonexistent-widget-xyz"])
    client.find_results = ["component-id"]
    hit = runner.invoke(app, ["find", "*idler*", "--kind", "glob"])

    assert miss.exit_code == 0
    assert "matches=0" in miss.stdout
    assert "warning: no components matched" in miss.stderr

    assert hit.exit_code == 0
    assert "matches=1" in hit.stdout
    assert "no components matched" not in hit.output


def test_find_zero_matches_machine_readable_modes_expose_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)
    client.match_count = 0

    payload = json.loads(runner.invoke(app, ["--json", "find", "ghost"]).stdout)
    assert payload["ok"] is True
    assert payload["match_count"] == 0

    document = yaml.safe_load(runner.invoke(app, ["--yaml", "find", "ghost"]).stdout)
    assert document["match_count"] == 0


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


@pytest.mark.parametrize("flag", ["--json", "--yaml", "--raw"])
def test_output_flags_work_after_subcommand(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    before = runner.invoke(app, [flag, "snapshot"])
    after = runner.invoke(app, ["snapshot", flag])

    assert before.exit_code == 0
    assert after.exit_code == 0
    assert after.stdout == before.stdout


def test_session_and_workspace_work_after_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".meshprobe"
    root.mkdir()
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    before = runner.invoke(
        app,
        ["--workspace", str(root), "--session", "review", "display", "c2", "--mode", "isolated"],
    )
    after = runner.invoke(
        app,
        ["display", "c2", "--mode", "isolated", "--workspace", str(root), "--session", "review"],
    )

    assert before.exit_code == 0
    assert after.exit_code == 0
    assert after.stdout == before.stdout
    assert isinstance(client.commands[-1], ComponentDisplayCommand)


def test_global_and_subcommand_options_interleave_after_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"model")
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(
        app, ["open", str(source), "--json", "--blender", "pinned", "--session", "review"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["session"] == "review"
    assert isinstance(client.commands[-1], SceneOpenCommand)


def test_eval_subcommand_version_is_not_hoisted_to_the_root_flag(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["eval", "generate", str(tmp_path), "--version", "cli-v1", "--seed-count", "1"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["corpus_version"] == "cli-v1"


def test_double_dash_keeps_a_global_flag_as_a_literal_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr("meshprobe.cli._client", lambda *args, **kwargs: client)

    result = runner.invoke(app, ["--session", "review", "find", "--", "--json"])

    assert result.exit_code == 0
    command = client.commands[-1]
    assert isinstance(command, ComponentFindCommand)
    assert command.selector.pattern == "--json"


def test_subcommand_help_is_not_shadowed_by_global_hoisting() -> None:
    result = runner.invoke(app, ["view-orbit", "--help"])

    assert result.exit_code == 0
    assert "absolute orbit" in unstyle(result.stdout).lower()


@pytest.mark.parametrize(
    ("target", "arguments"),
    (
        ("build_corpus", ["eval", "generate", "{root}"]),
        ("validate_corpus", ["eval", "validate", "{root}"]),
        (
            "migrate_corpus_v3",
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
