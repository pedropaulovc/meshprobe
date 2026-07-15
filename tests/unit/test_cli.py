from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from meshprobe.cli import app
from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.models import SceneManifest

runner = CliRunner()


def write_manifest(tmp_path, scene_manifest: SceneManifest):  # type: ignore[no-untyped-def]
    path = tmp_path / "scene.json"
    path.write_text(scene_manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


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


def test_open_reports_missing_blender(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "assembly.glb"
    source.write_bytes(b"model")
    result = runner.invoke(
        app,
        ["open", str(source), "--blender", "meshprobe-blender-does-not-exist"],
    )
    assert result.exit_code == 2
    assert "Blender executable not found" in result.output


def test_validate_manifest_reports_source(tmp_path, scene_manifest: SceneManifest) -> None:  # type: ignore[no-untyped-def]
    path = write_manifest(tmp_path, scene_manifest)
    result = runner.invoke(app, ["validate-manifest", str(path)])
    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload == {
        "components": 3,
        "source_sha256": scene_manifest.source_sha256,
        "valid": True,
    }


def test_find_command_returns_component_paths(tmp_path, scene_manifest: SceneManifest) -> None:  # type: ignore[no-untyped-def]
    path = write_manifest(tmp_path, scene_manifest)
    result = runner.invoke(app, ["find", str(path), "assembly/**/idler*", "--kind", "glob"])
    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["count"] == 1
    assert payload["components"][0]["path"] == "assembly/drive/idler"


def test_apply_exercises_renderer_independent_operations(
    tmp_path: Path, scene_manifest: SceneManifest
) -> None:
    manifest_path = write_manifest(tmp_path, scene_manifest)
    target = scene_manifest.components[-1].id
    commands = [
        {"request_id": "describe", "op": "scene.describe"},
        {
            "request_id": "find",
            "op": "component.find",
            "selector": {"kind": "exact_name", "pattern": "idler"},
        },
        {"request_id": "inspect", "op": "component.inspect", "component_id": target},
        {
            "request_id": "camera",
            "op": "view.set",
            "camera": {
                "pose": {"position_mm": [200, 300, 400], "orientation_xyzw": [0, 0, 0, 1]},
                "projection": {"mode": "orthographic", "scale_mm": 500},
            },
        },
        {
            "request_id": "orbit",
            "op": "view.orbit",
            "target_mm": [0, 0, 0],
            "azimuth_degrees": 0,
            "elevation_degrees": 0,
            "distance_mm": 250,
            "projection": {"mode": "perspective", "focal_length_mm": 85},
        },
        {
            "request_id": "light",
            "op": "illumination.set",
            "illumination": {"preset": "raking_left"},
        },
        {
            "request_id": "display",
            "op": "component.display",
            "component_ids": [target],
            "mode": "isolated",
        },
        {
            "request_id": "mark",
            "op": "component.mark",
            "component_ids": [target],
            "mode": "highlighted",
        },
        {"request_id": "reset", "op": "session.reset"},
    ]
    commands_path = tmp_path / "commands.jsonl"
    commands_path.write_text(
        "\n".join(json.dumps(command) for command in commands), encoding="utf-8"
    )

    result = runner.invoke(app, ["apply", str(manifest_path), str(commands_path)])
    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert [item["request_id"] for item in payload["results"]] == [
        "describe",
        "find",
        "inspect",
        "camera",
        "orbit",
        "light",
        "display",
        "mark",
        "reset",
    ]
    assert payload["final_state"]["camera"] == scene_manifest.imported_camera.model_dump(
        mode="json"
    )


def test_apply_rejects_renderer_operation(tmp_path, scene_manifest: SceneManifest) -> None:  # type: ignore[no-untyped-def]
    manifest_path = write_manifest(tmp_path, scene_manifest)
    commands_path = tmp_path / "commands.jsonl"
    commands_path.write_text(
        json.dumps(
            {
                "request_id": "render",
                "op": "render.image",
                "output_path": "evidence.png",
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["apply", str(manifest_path), str(commands_path)])
    assert result.exit_code == 2
    assert "requires the Blender worker" in result.output


def test_describe_reports_current_session_state(tmp_path, scene_manifest: SceneManifest) -> None:  # type: ignore[no-untyped-def]
    manifest_path = write_manifest(tmp_path, scene_manifest)
    target = scene_manifest.components[-1].id
    commands_path = tmp_path / "commands.jsonl"
    commands_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "hide",
                        "op": "component.display",
                        "component_ids": [target],
                        "mode": "hidden",
                    }
                ),
                json.dumps({"request_id": "describe", "op": "scene.describe"}),
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["apply", str(manifest_path), str(commands_path)])
    payload = json.loads(result.stdout)
    described = payload["results"][1]["result"]
    assert result.exit_code == 0
    assert described["session"]["components"][target]["display"] == "hidden"
    assert described["scene"]["source_sha256"] == scene_manifest.source_sha256
