from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from meshprobe.cli import app
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
