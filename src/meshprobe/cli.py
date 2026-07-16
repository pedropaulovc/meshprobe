"""Command-line access to MeshProbe's renderer-independent contracts."""

from __future__ import annotations

import json
import subprocess
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from meshprobe.client import MeshProbeClient
from meshprobe.evals.curated import ingest_curated_sources, load_catalog
from meshprobe.evals.curated_build import build_curated_variants
from meshprobe.evals.curated_tasks import build_curated_corpus
from meshprobe.evals.ergonomics import run_ergonomics_pilot
from meshprobe.evals.factory import CorpusBuild, build_corpus, merge_corpora, validate_corpus
from meshprobe.evals.generators import PUBLIC_GENERATOR_FAMILIES, GeneratorFamily
from meshprobe.evals.harness.adapters import (
    CliJsonlAdapter,
    McpStdioAdapter,
    ReferenceAgentAdapter,
)
from meshprobe.evals.harness.suite import run_tier
from meshprobe.evals.migration import audit_migration, migrate_corpus_v2
from meshprobe.evals.tiers import current_runtime_pin, pin_private_tier, pin_standard_tiers
from meshprobe.models import DisplayMode, IlluminationPreset, MarkMode, Projection, RenderEngine
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentInspectCommand,
    ComponentMarkCommand,
    IlluminationSetCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    SessionResetCommand,
    SessionSnapshotCommand,
    ViewOrbitCommand,
    ViewSetCommand,
    command_json_schema,
)
from meshprobe.selectors import ComponentSelector, SelectorKind
from meshprobe.service import MeshProbeService
from meshprobe.workspace import OperationReceipt

app = typer.Typer(help="Read-only 3D model inspection for AI agents.", no_args_is_help=True)
eval_app = typer.Typer(help="Build and validate qualification corpora.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
DEFAULT_WORKSPACE = Path.cwd()


class CliOptions:
    def __init__(self, session: str, workspace: Path, output: str) -> None:
        self.session = session
        self.workspace = workspace
        self.output = output


@app.callback()
def global_options(
    ctx: typer.Context,
    session: Annotated[str, typer.Option("--session", "-s")] = "default",
    workspace: Annotated[Path, typer.Option("--workspace", file_okay=False)] = DEFAULT_WORKSPACE,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    raw: Annotated[bool, typer.Option("--raw")] = False,
) -> None:
    """Select a durable named session and receipt output format."""

    if json_output and raw:
        raise typer.BadParameter("--json and --raw are mutually exclusive")
    output = "raw" if raw else "json" if json_output else "receipt"
    ctx.obj = CliOptions(session, workspace, output)


class AgentAdapterKind(StrEnum):
    CLI = "cli"
    MCP = "mcp"
    REFERENCE = "reference"


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
    corpus_version: Annotated[str, typer.Option("--version")] = "procedural-v6",
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


@eval_app.command("migrate")
def migrate_eval_corpus(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    corpus_version: Annotated[str, typer.Option("--version")],
    opaque_family: Annotated[str | None, typer.Option("--opaque-family")] = None,
) -> None:
    """Migrate a schema-v1 corpus without regenerating models or task identities."""

    try:
        build, audit = migrate_corpus_v2(
            source,
            output_root,
            corpus_version=corpus_version,
            opaque_family=opaque_family,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit({**_corpus_summary(build), "audit": audit.__dict__})


@eval_app.command("audit-migration")
def audit_eval_migration(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    destination: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Prove identity and evaluator truth were preserved by a corpus migration."""

    try:
        audit = audit_migration(source.resolve(), destination.resolve())
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(audit.__dict__)


@eval_app.command("curated-generate")
def generate_curated_eval_corpus(
    catalog_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    work_root: Annotated[Path, typer.Argument(file_okay=False)],
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    build_version: Annotated[str, typer.Option("--build-version")] = "curated-v2",
    corpus_version: Annotated[str, typer.Option("--corpus-version")] = "curated-tasks-v6",
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
    corpus_version: Annotated[str, typer.Option("--version")] = "qualification-v6",
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
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
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
            workers=workers,
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


@eval_app.command("ergonomics")
def run_cli_ergonomics(
    corpus_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Argument(file_okay=False)],
    per_difficulty: Annotated[int, typer.Option("--per-difficulty", min=1, max=50)] = 12,
    canary_pairs: Annotated[int, typer.Option("--canary-pairs", min=1, max=24)] = 4,
    token_limit: Annotated[int, typer.Option("--token-limit", min=1_000)] = 768_000,
) -> None:
    """Run the paired Claude Opus and Codex Luna CLI ergonomics pilot."""

    try:
        result = run_ergonomics_pilot(
            corpus_root,
            output_root,
            per_difficulty=per_difficulty,
            canary_pairs=canary_pairs,
            token_limit=token_limit,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(
        {
            "root": str(result.root),
            "attempts": len(result.attempts),
            "report": str(result.report_path),
            "report_markdown": str(result.report_markdown_path),
            "qualification": False,
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


def _options(ctx: typer.Context) -> CliOptions:
    if not isinstance(ctx.obj, CliOptions):
        raise RuntimeError("CLI context is unavailable")
    return ctx.obj


def _client(ctx: typer.Context, *, blender: str | None = None) -> MeshProbeClient:
    return MeshProbeClient(_options(ctx).workspace, blender=blender)


def _request_id(operation: str) -> str:
    return f"{operation}-{uuid.uuid4().hex[:12]}"


def _emit_receipt(
    ctx: typer.Context,
    client: MeshProbeClient,
    receipt: OperationReceipt,
) -> None:
    output = _options(ctx).output
    if output == "json":
        typer.echo(receipt.model_dump_json(indent=2))
        return
    if output == "raw":
        envelope = client.read_result(receipt)
        result = envelope.get("result") if isinstance(envelope, dict) else envelope
        _emit(result)
        return
    fields = ["ok", f"session={receipt.session}", f"op={receipt.op}"]
    if receipt.result_path:
        fields.append(f"result={receipt.result_path}")
    if receipt.state_path:
        fields.append(f"state={receipt.state_path}")
    if receipt.artifact_paths:
        fields.append(f"artifacts={','.join(receipt.artifact_paths)}")
    typer.echo(" ".join(fields))


def _execute(ctx: typer.Context, command: Command, *, blender: str | None = None) -> None:
    client = _client(ctx, blender=blender)
    try:
        receipt = client.execute(_options(ctx).session, command)
    except (OSError, RuntimeError, ValueError, ValidationError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit_receipt(ctx, client, receipt)


def _component_ids(ctx: typer.Context, components: list[str]) -> tuple[str, ...]:
    client = _client(ctx)
    try:
        return tuple(
            client.resolve_component(_options(ctx).session, component) for component in components
        )
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("open")
def open_scene(
    ctx: typer.Context,
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    blender: Annotated[str | None, typer.Option("--blender")] = None,
) -> None:
    """Open a model in the selected durable session."""

    from meshprobe.protocol import SceneOpenCommand

    _execute(
        ctx,
        SceneOpenCommand(
            request_id=_request_id("open"),
            op="scene.open",
            source_path=str(source.expanduser().resolve(strict=True)),
        ),
        blender=blender,
    )


@app.command("snapshot")
def snapshot(ctx: typer.Context) -> None:
    """Write and return the current scene/session snapshot."""

    _execute(
        ctx,
        SessionSnapshotCommand(request_id=_request_id("snapshot"), op="session.snapshot"),
    )


@app.command("find")
def find_components(
    ctx: typer.Context,
    pattern: Annotated[str, typer.Argument()],
    kind: Annotated[SelectorKind, typer.Option("--kind")] = SelectorKind.GLOB,
) -> None:
    """Find components in the selected session."""

    _execute(
        ctx,
        ComponentFindCommand(
            request_id=_request_id("find"),
            op="component.find",
            selector=ComponentSelector(kind=kind, pattern=pattern),
        ),
    )


@app.command("inspect")
def inspect_component(ctx: typer.Context, component: Annotated[str, typer.Argument()]) -> None:
    """Inspect one component by ref, stable ID, or exact path."""

    component_id = _component_ids(ctx, [component])[0]
    _execute(
        ctx,
        ComponentInspectCommand(
            request_id=_request_id("inspect"),
            op="component.inspect",
            component_id=component_id,
        ),
    )


@app.command("view-set")
def view_set(
    ctx: typer.Context,
    camera_json: Annotated[str, typer.Option("--camera-json")],
    focus: Annotated[list[str] | None, typer.Option("--focus")] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
) -> None:
    """Set an exact camera from a JSON object."""

    from meshprobe.models import Camera

    camera = Camera.model_validate_json(camera_json)
    _execute(
        ctx,
        ViewSetCommand(
            request_id=_request_id("view-set"),
            op="view.set",
            camera=camera,
            focus_component_ids=_component_ids(ctx, focus or []) if focus else (),
            aspect_ratio=aspect_ratio,
        ),
    )


@app.command("view-orbit")
def view_orbit(
    ctx: typer.Context,
    target: Annotated[tuple[float, float, float], typer.Option("--target")],
    azimuth: Annotated[float, typer.Option("--azimuth")],
    elevation: Annotated[float, typer.Option("--elevation")],
    distance: Annotated[float, typer.Option("--distance", min=0.000001)],
    projection_json: Annotated[str, typer.Option("--projection-json")],
    roll: Annotated[float, typer.Option("--roll")] = 0.0,
    focus: Annotated[list[str] | None, typer.Option("--focus")] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
) -> None:
    """Orbit the camera around a target point."""

    from pydantic import TypeAdapter

    projection: Projection = TypeAdapter(Projection).validate_json(projection_json)
    _execute(
        ctx,
        ViewOrbitCommand(
            request_id=_request_id("view-orbit"),
            op="view.orbit",
            target_mm=target,
            azimuth_degrees=azimuth,
            elevation_degrees=elevation,
            roll_degrees=roll,
            distance_mm=distance,
            projection=projection,
            focus_component_ids=_component_ids(ctx, focus or []) if focus else (),
            aspect_ratio=aspect_ratio,
        ),
    )


@app.command("illumination-set")
def illumination_set(
    ctx: typer.Context,
    preset: Annotated[IlluminationPreset, typer.Argument()],
) -> None:
    """Apply a named illumination preset."""

    from meshprobe.models import PresetIllumination

    _execute(
        ctx,
        IlluminationSetCommand(
            request_id=_request_id("illumination"),
            op="illumination.set",
            illumination=PresetIllumination(preset=preset),
        ),
    )


@app.command("display")
def display_components(
    ctx: typer.Context,
    components: Annotated[list[str], typer.Argument()],
    mode: Annotated[DisplayMode, typer.Option("--mode")],
) -> None:
    """Change component visibility using short refs, IDs, or exact paths."""

    _execute(
        ctx,
        ComponentDisplayCommand(
            request_id=_request_id("display"),
            op="component.display",
            component_ids=_component_ids(ctx, components),
            mode=mode,
        ),
    )


@app.command("mark")
def mark_components(
    ctx: typer.Context,
    components: Annotated[list[str], typer.Argument()],
    mode: Annotated[MarkMode, typer.Option("--mode")],
) -> None:
    """Mark components using short refs, IDs, or exact paths."""

    _execute(
        ctx,
        ComponentMarkCommand(
            request_id=_request_id("mark"),
            op="component.mark",
            component_ids=_component_ids(ctx, components),
            mode=mode,
        ),
    )


@app.command("render-image")
def render_image(
    ctx: typer.Context,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    width: Annotated[int, typer.Option("--width", min=64, max=16_384)] = 1024,
    height: Annotated[int, typer.Option("--height", min=64, max=16_384)] = 1024,
    samples: Annotated[int, typer.Option("--samples", min=1, max=4_096)] = 64,
    engine: Annotated[RenderEngine, typer.Option("--engine")] = RenderEngine.EEVEE,
) -> None:
    """Render the selected session to an image artifact."""

    options = _options(ctx)
    destination = output or (
        options.workspace.resolve()
        / ".meshprobe"
        / "sessions"
        / options.session
        / "artifacts"
        / f"render-{uuid.uuid4().hex[:12]}.png"
    )
    _execute(
        ctx,
        RenderImageCommand(
            request_id=_request_id("render-image"),
            op="render.image",
            output_path=str(destination.expanduser().resolve()),
            width=width,
            height=height,
            samples=samples,
            engine=engine,
        ),
    )


@app.command("render-sheet")
def render_sheet(
    ctx: typer.Context,
    focus: Annotated[list[str], typer.Argument()],
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    panel_width: Annotated[int, typer.Option("--panel-width", min=128)] = 768,
    panel_height: Annotated[int, typer.Option("--panel-height", min=128)] = 768,
    samples: Annotated[int, typer.Option("--samples", min=1)] = 32,
    engine: Annotated[RenderEngine, typer.Option("--engine")] = RenderEngine.EEVEE,
) -> None:
    """Render the focused 3x3 diagnostic sheet."""

    options = _options(ctx)
    destination = output or (
        options.workspace.resolve()
        / ".meshprobe"
        / "sessions"
        / options.session
        / "artifacts"
        / f"sheet-{uuid.uuid4().hex[:12]}.png"
    )
    _execute(
        ctx,
        RenderContactSheetCommand(
            request_id=_request_id("render-sheet"),
            op="render.contact_sheet",
            output_path=str(destination.expanduser().resolve()),
            focus_component_ids=_component_ids(ctx, focus),
            panel_width=panel_width,
            panel_height=panel_height,
            samples=samples,
            engine=engine,
        ),
    )


@app.command("reset")
def reset(ctx: typer.Context) -> None:
    """Reset visual state to the imported scene defaults."""

    _execute(ctx, SessionResetCommand(request_id=_request_id("reset"), op="session.reset"))


@app.command("list")
def list_sessions(ctx: typer.Context) -> None:
    """List durable sessions and renderer status."""

    client = _client(ctx)
    try:
        sessions = client.list_sessions()
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if _options(ctx).output in {"json", "raw"}:
        _emit({"sessions": sessions})
        return
    for session in sessions:
        typer.echo(f"{session['name']}\t{session['status']}\t{session['source_path']}")


@app.command("close")
def close_session(
    ctx: typer.Context,
    all_sessions: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Gracefully checkpoint and stop one renderer, or all renderers and the daemon."""

    client = _client(ctx)
    try:
        receipts = client.close_all() if all_sessions else [client.close(_options(ctx).session)]
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    for receipt in receipts:
        _emit_receipt(ctx, client, receipt)


@app.command("kill")
def kill_session(
    ctx: typer.Context,
    all_sessions: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Force-stop one renderer, or the daemon and its complete process tree."""

    client = _client(ctx)
    try:
        receipts = client.kill_all() if all_sessions else [client.kill(_options(ctx).session)]
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    for receipt in receipts:
        _emit_receipt(ctx, client, receipt)


@app.command("delete-data")
def delete_data(ctx: typer.Context) -> None:
    """Delete the stopped workspace's `.meshprobe` state."""

    client = _client(ctx)
    try:
        deleted = client.delete_data()
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if _options(ctx).output in {"json", "raw"}:
        _emit({"deleted": str(deleted)})
        return
    typer.echo(f"deleted {deleted}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
