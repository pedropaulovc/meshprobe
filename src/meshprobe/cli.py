"""Command-line access to MeshProbe's renderer-independent contracts."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from meshprobe.camera import orbit_camera
from meshprobe.controller import DEFAULT_WORKER_TIMEOUT_SECONDS, BlenderWorkerError
from meshprobe.evals.curated import ingest_curated_sources, load_catalog
from meshprobe.evals.curated_build import build_curated_variants
from meshprobe.evals.curated_tasks import build_curated_corpus
from meshprobe.evals.factory import CorpusBuild, build_corpus, merge_corpora, validate_corpus
from meshprobe.evals.generators import PUBLIC_GENERATOR_FAMILIES, GeneratorFamily
from meshprobe.evals.harness.adapters import (
    CliJsonlAdapter,
    McpStdioAdapter,
    ReferenceAgentAdapter,
)
from meshprobe.evals.harness.suite import run_tier
from meshprobe.evals.tiers import current_runtime_pin, pin_private_tier, pin_standard_tiers
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
    ViewOrbitCommand,
    ViewSetCommand,
    command_json_schema,
    parse_command_json,
)
from meshprobe.selectors import ComponentIndex, ComponentSelector, SelectorKind
from meshprobe.service import MeshProbeService
from meshprobe.session import InspectionSession

app = typer.Typer(help="Read-only 3D model inspection for AI agents.", no_args_is_help=True)
eval_app = typer.Typer(help="Build and validate qualification corpora.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


class AgentAdapterKind(StrEnum):
    CLI = "cli"
    MCP = "mcp"
    REFERENCE = "reference"


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


@eval_app.command("generate")
def generate_eval_corpus(
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    corpus_version: Annotated[str, typer.Option("--version")] = "procedural-v1",
    families: Annotated[list[GeneratorFamily] | None, typer.Option("--family")] = None,
    seed_start: Annotated[int, typer.Option("--seed-start", min=0)] = 0,
    seed_count: Annotated[int, typer.Option("--seed-count", min=1)] = 32,
) -> None:
    """Generate an atomic, checkpointed procedural evaluation corpus."""

    selected_families = tuple(families) if families else PUBLIC_GENERATOR_FAMILIES
    try:
        build = build_corpus(
            output_root,
            corpus_version=corpus_version,
            families=selected_families,
            seeds=range(seed_start, seed_start + seed_count),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(_corpus_summary(build))


@eval_app.command("validate")
def validate_eval_corpus(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Verify corpus membership, hashes, leakage boundaries, and coverage."""

    try:
        build = validate_corpus(root.resolve())
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(_corpus_summary(build))


@eval_app.command("curated-generate")
def generate_curated_eval_corpus(
    catalog_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    work_root: Annotated[Path, typer.Argument(file_okay=False)],
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    build_version: Annotated[str, typer.Option("--build-version")] = "curated-v1",
    corpus_version: Annotated[str, typer.Option("--corpus-version")] = "curated-tasks-v1",
    blender: Annotated[str, typer.Option("--blender")] = "blender",
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 8,
) -> None:
    """Fetch, verify, build, inspect, and publish the pinned curated corpus."""

    try:
        catalog = load_catalog(catalog_path.resolve())
        sources = ingest_curated_sources(catalog, work_root / "cache", workers=workers)
        build = build_curated_variants(
            catalog,
            sources,
            work_root / "builds",
            version=build_version,
            blender=blender,
        )
        corpus = build_curated_corpus(
            build,
            output_root,
            corpus_version=corpus_version,
            blender=blender,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(_corpus_summary(corpus))


@eval_app.command("merge")
def merge_eval_corpora(
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    corpus_roots: Annotated[list[Path], typer.Argument(exists=True, file_okay=False)],
    corpus_version: Annotated[str, typer.Option("--version")] = "qualification-v1",
) -> None:
    """Combine validated procedural and curated corpora without rewriting artifacts."""

    try:
        build = merge_corpora(corpus_roots, output_root, corpus_version=corpus_version)
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(_corpus_summary(build))


@eval_app.command("pin")
def pin_eval_tiers(
    corpus_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    private: Annotated[bool, typer.Option("--private")] = False,
    blender: Annotated[str, typer.Option("--blender")] = "blender",
) -> None:
    """Pin exact public tiers or one held-out private tier to the current runtime."""

    try:
        runtime = current_runtime_pin(blender)
        manifests = (
            (pin_private_tier(corpus_root, output_root, runtime=runtime),)
            if private
            else pin_standard_tiers(corpus_root, output_root, runtime=runtime)
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(
        {
            "manifests": [
                {
                    "tier": manifest.tier,
                    "corpus_version": manifest.corpus_version,
                    "episodes": len(manifest.episodes),
                    "models": len(manifest.model_sha256),
                    "path": str(output_root.resolve() / f"{manifest.tier}.json"),
                }
                for manifest in manifests
            ]
        }
    )


@eval_app.command("run-tier")
def run_eval_tier(
    corpus_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    tier_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    agent_command_json: Annotated[str | None, typer.Option("--agent-command-json")] = None,
    adapter_kind: Annotated[AgentAdapterKind, typer.Option("--adapter")] = AgentAdapterKind.CLI,
    blender: Annotated[str, typer.Option("--blender")] = "blender",
) -> None:
    """Run an isolated agent over every episode in an exact pinned tier."""

    try:
        adapter: ReferenceAgentAdapter | CliJsonlAdapter | McpStdioAdapter
        if adapter_kind is AgentAdapterKind.REFERENCE:
            if agent_command_json is not None:
                raise ValueError("the reference adapter does not accept an agent command")
            adapter = ReferenceAgentAdapter()
        else:
            if agent_command_json is None:
                raise ValueError("CLI and MCP adapters require --agent-command-json")
            command_payload = json.loads(agent_command_json)
            if not isinstance(command_payload, list) or not all(
                isinstance(item, str) for item in command_payload
            ):
                raise ValueError("agent command must be a JSON array of strings")
            command = tuple(command_payload)
            adapter = (
                CliJsonlAdapter(command)
                if adapter_kind is AgentAdapterKind.CLI
                else McpStdioAdapter(command)
            )
        result = run_tier(
            corpus_root=corpus_root,
            tier_manifest_path=tier_manifest,
            adapter=adapter,
            output_root=output_root,
            service_factory=lambda: MeshProbeService(blender=blender),
            runtime_provider=lambda: current_runtime_pin(blender),
        )
    except (json.JSONDecodeError, OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(
        {
            "root": str(result.root),
            "completed": result.completed,
            "report": str(result.report_path),
            "passed_thresholds": result.report.passed_thresholds,
        }
    )


def _corpus_summary(build: CorpusBuild) -> dict[str, object]:
    return {
        "valid": True,
        "root": str(build.root),
        "corpus_version": build.manifest.corpus_version,
        "generator_sha256": build.manifest.generator_sha256,
        "models": build.model_count,
        "episodes": build.episode_count,
        "full_investigations": build.full_investigation_count,
    }


@app.command("open")
def open_scene(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    blender: Annotated[str | None, typer.Option("--blender")] = None,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1)] = (
        DEFAULT_WORKER_TIMEOUT_SECONDS
    ),
) -> None:
    """Open a model in a factory-clean Blender worker and print its manifest."""

    command = parse_command_json(
        json.dumps({"request_id": "open", "op": "scene.open", "source_path": str(source)})
    )
    try:
        with MeshProbeService(blender=blender, timeout_seconds=timeout_seconds) as service:
            manifest = service.execute(command).result
    except (BlenderWorkerError, OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(manifest)


@app.command("run")
def run_commands(
    commands: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    blender: Annotated[str | None, typer.Option("--blender")] = None,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1)] = (
        DEFAULT_WORKER_TIMEOUT_SECONDS
    ),
) -> None:
    """Execute JSONL commands in one persistent Blender session."""

    results: list[dict[str, object]] = []
    try:
        with MeshProbeService(blender=blender, timeout_seconds=timeout_seconds) as service:
            for line_number, line in enumerate(
                commands.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    command = parse_command_json(line)
                    response = service.execute(command)
                except (ValidationError, ValueError) as error:
                    raise typer.BadParameter(f"line {line_number}: {error}") from error
                results.append(response.model_dump(mode="json"))
    except BlenderWorkerError as error:
        raise typer.BadParameter(str(error)) from error
    _emit({"results": results})


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
    if isinstance(command, ViewOrbitCommand):
        camera = orbit_camera(
            target_mm=command.target_mm,
            azimuth_degrees=command.azimuth_degrees,
            elevation_degrees=command.elevation_degrees,
            roll_degrees=command.roll_degrees,
            distance_mm=command.distance_mm,
            projection=command.projection,
        )
        return session.set_camera(camera).model_dump(mode="json")
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
