"""Command-line access to MeshProbe's renderer-independent contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from meshprobe.models import SceneManifest
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentInspectCommand,
    ComponentMarkCommand,
    IlluminationSetCommand,
    SceneDescribeCommand,
    SessionResetCommand,
    ViewSetCommand,
    command_json_schema,
    parse_command_json,
)
from meshprobe.selectors import ComponentIndex, ComponentSelector, SelectorKind
from meshprobe.session import InspectionSession

app = typer.Typer(help="Read-only 3D model inspection for AI agents.", no_args_is_help=True)


def _load_manifest(path: Path) -> SceneManifest:
    return SceneManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _emit(value: object) -> None:
    if hasattr(value, "model_dump_json"):
        typer.echo(value.model_dump_json(indent=2))
        return
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


@app.command("schema")
def schema() -> None:
    """Print the JSON Schema for every public protocol command."""

    _emit(command_json_schema())


@app.command("validate-manifest")
def validate_manifest(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate a renderer-produced scene manifest."""

    scene = _load_manifest(manifest)
    _emit(
        {"valid": True, "components": len(scene.components), "source_sha256": scene.source_sha256}
    )


@app.command("find")
def find_components(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    pattern: Annotated[str, typer.Argument()],
    kind: Annotated[SelectorKind, typer.Option("--kind")] = SelectorKind.GLOB,
) -> None:
    """Find components in an existing scene manifest."""

    scene = _load_manifest(manifest)
    matches = ComponentIndex(scene).find(ComponentSelector(kind=kind, pattern=pattern))
    _emit(
        {
            "count": len(matches),
            "components": [
                {"id": component.id, "path": component.path, "name": component.display_name}
                for component in matches
            ],
        }
    )


@app.command("apply")
def apply_commands(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    commands: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate and apply JSONL inspection commands without a renderer."""

    scene = _load_manifest(manifest)
    index = ComponentIndex(scene)
    session = InspectionSession(scene)
    results: list[dict[str, object]] = []

    for line_number, line in enumerate(commands.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            command = parse_command_json(line)
            result = _apply_one(command, scene, index, session)
        except (ValidationError, ValueError) as error:
            raise typer.BadParameter(f"line {line_number}: {error}") from error
        results.append({"request_id": command.request_id, "result": result})

    _emit({"results": results, "final_state": session.snapshot().model_dump(mode="json")})


def _apply_one(
    command: Command,
    scene: SceneManifest,
    index: ComponentIndex,
    session: InspectionSession,
) -> object:
    if isinstance(command, SceneDescribeCommand):
        return {
            "scene": scene.model_dump(mode="json"),
            "session": session.snapshot().model_dump(mode="json"),
        }
    if isinstance(command, ComponentFindCommand):
        return [component.model_dump(mode="json") for component in index.find(command.selector)]
    if isinstance(command, ComponentInspectCommand):
        return index.by_id(command.component_id).model_dump(mode="json")
    if isinstance(command, ViewSetCommand):
        return session.set_camera(command.camera).model_dump(mode="json")
    if isinstance(command, IlluminationSetCommand):
        return session.set_illumination(command.illumination).model_dump(mode="json")
    if isinstance(command, ComponentDisplayCommand):
        return session.display(command.component_ids, command.mode).model_dump(mode="json")
    if isinstance(command, ComponentMarkCommand):
        return session.mark(command.component_ids, command.mode).model_dump(mode="json")
    if isinstance(command, SessionResetCommand):
        return session.reset().model_dump(mode="json")
    raise ValueError(f"operation {command.op} requires the Blender worker")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
