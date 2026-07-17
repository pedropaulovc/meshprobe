"""Command-line access to MeshProbe's renderer-independent contracts."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
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
from meshprobe.models import (
    Camera,
    ComponentFocus,
    CoordinateFrame,
    DepthOfField,
    DepthOfFieldMode,
    DisplayMode,
    GraphicsPolicy,
    IlluminationPreset,
    MarkMode,
    PerspectiveProjection,
    Projection,
    RenderEngine,
)
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
    ViewMoveCommand,
    ViewOrbitCommand,
    ViewRotateCommand,
    ViewSetCommand,
    command_json_schema,
)
from meshprobe.selectors import ComponentSelector, SelectorKind
from meshprobe.service import MeshProbeService
from meshprobe.workspace import (
    OperationReceipt,
    durable_state_json_schema,
    durable_state_schema_summary,
    workspace_root,
)

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
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a machine-readable JSON receipt.")
    ] = False,
    yaml_output: Annotated[
        bool, typer.Option("--yaml", help="Emit a machine-readable YAML receipt.")
    ] = False,
    raw: Annotated[
        bool, typer.Option("--raw", help="Emit the full operation result as JSON.")
    ] = False,
) -> None:
    """Select a durable named session and receipt output format."""

    if sum((json_output, yaml_output, raw)) > 1:
        raise typer.BadParameter("--json, --yaml, and --raw are mutually exclusive")
    output = "raw" if raw else "json" if json_output else "yaml" if yaml_output else "receipt"
    ctx.obj = CliOptions(session, workspace, output)


class AgentAdapterKind(StrEnum):
    CLI = "cli"
    MCP = "mcp"
    REFERENCE = "reference"


class SchemaKind(StrEnum):
    COMMANDS = "commands"
    STATE = "state"
    ALL = "all"


class FindKind(StrEnum):
    AUTO = "auto"
    EXACT_NAME = "exact_name"
    EXACT_PATH = "exact_path"
    GLOB = "glob"
    REGEX = "regex"


def _emit(value: object) -> None:
    if hasattr(value, "model_dump_json"):
        typer.echo(value.model_dump_json(indent=2))
        return
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _emit_yaml(value: object) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    typer.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())


@app.command("schema")
def schema(
    ctx: typer.Context,
    kind: Annotated[
        SchemaKind,
        typer.Option("--kind", help="Show operation commands, durable state files, or both."),
    ] = SchemaKind.STATE,
    full: Annotated[
        bool,
        typer.Option("--full", help="Expand durable state fields into formal JSON Schemas."),
    ] = False,
) -> None:
    """Print machine-readable schemas for commands or durable state files."""

    payload: object = command_json_schema()
    if kind is SchemaKind.STATE:
        payload = durable_state_json_schema() if full else durable_state_schema_summary()
    if kind is SchemaKind.ALL:
        state = durable_state_json_schema() if full else durable_state_schema_summary()
        payload = {"commands": command_json_schema(), "state": state}
    if _options(ctx).output == "yaml":
        _emit_yaml(payload)
        return
    _emit(payload)


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
    max_pairs: Annotated[int | None, typer.Option("--max-pairs", min=1, max=100)] = None,
) -> None:
    """Run the paired Claude Opus and Codex Luna CLI ergonomics pilot."""

    try:
        result = run_ergonomics_pilot(
            corpus_root,
            output_root,
            per_difficulty=per_difficulty,
            canary_pairs=canary_pairs,
            token_limit=token_limit,
            max_pairs=max_pairs,
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


def _distance_mm(value: str | None) -> float:
    if value is None:
        return 0.0
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(mm|cm|m)?\s*",
        value,
    )
    if match is None:
        raise typer.BadParameter(f"invalid distance {value!r}; use mm, cm, or m")
    magnitude = float(match.group(1))
    scale = {None: 1.0, "mm": 1.0, "cm": 10.0, "m": 1_000.0}[match.group(2)]
    return magnitude * scale


def _camera_with_depth_of_field(
    camera: Camera,
    *,
    aperture_fstop: float | None,
    focus_distance: str | None,
    focus_component_id: str | None,
    disable: bool,
) -> Camera:
    changed = (
        aperture_fstop is not None or focus_distance is not None or focus_component_id is not None
    )
    if not changed and not disable:
        return camera
    if not isinstance(camera.projection, PerspectiveProjection):
        raise typer.BadParameter("depth of field requires a perspective camera")
    if disable and changed:
        raise typer.BadParameter("--disable-depth-of-field cannot be combined with focus controls")
    if disable:
        projection = camera.projection.model_copy(update={"depth_of_field": DepthOfField()})
        return camera.model_copy(update={"projection": projection})
    if focus_distance is not None and focus_component_id is not None:
        raise typer.BadParameter("provide either --focus-distance or --focus-component")
    current = camera.projection.depth_of_field
    resolved_distance = (
        _distance_mm(focus_distance) if focus_distance is not None else current.focus_distance_mm
    )
    resolved_focus = (
        ComponentFocus(component_id=focus_component_id)
        if focus_component_id is not None
        else current.focus
    )
    if focus_distance is not None:
        resolved_focus = None
    if focus_component_id is not None:
        resolved_distance = None
    depth_of_field = DepthOfField(
        mode=DepthOfFieldMode.ENABLED,
        aperture_fstop=aperture_fstop or current.aperture_fstop,
        focus_distance_mm=resolved_distance,
        focus=resolved_focus,
    )
    projection = camera.projection.model_copy(update={"depth_of_field": depth_of_field})
    return camera.model_copy(update={"projection": projection})


def _emit_receipt(
    ctx: typer.Context,
    client: MeshProbeClient,
    receipt: OperationReceipt,
) -> None:
    output = _options(ctx).output
    if output == "json":
        typer.echo(receipt.model_dump_json(indent=2))
        return
    if output == "yaml":
        _emit_yaml(receipt)
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
    for warning in receipt.warnings:
        typer.echo(f"warning: {warning}", err=True)
    for component in receipt.components:
        typer.echo(
            f"component: {component.ref} {component.name} ({component.path}) id={component.id}"
        )


def _emit_receipts(
    ctx: typer.Context,
    client: MeshProbeClient,
    receipts: list[OperationReceipt],
) -> None:
    output = _options(ctx).output
    if output == "json":
        _emit({"receipts": [receipt.model_dump(mode="json") for receipt in receipts]})
        return
    if output == "yaml":
        _emit_yaml({"receipts": [receipt.model_dump(mode="json") for receipt in receipts]})
        return
    if output == "raw":
        results: list[object] = []
        for receipt in receipts:
            envelope = client.read_result(receipt)
            results.append(envelope.get("result") if isinstance(envelope, dict) else envelope)
        _emit({"results": results})
        return
    for receipt in receipts:
        _emit_receipt(ctx, client, receipt)


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
    pattern: Annotated[str | None, typer.Argument()] = None,
    kind: Annotated[
        FindKind,
        typer.Option(
            "--kind",
            help="Auto treats plain text as an exact name, paths as exact, and wildcards as glob.",
        ),
    ] = FindKind.AUTO,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Find an exact component display name."),
    ] = None,
) -> None:
    """Find components in the selected session."""

    selector = _find_selector_options(pattern=pattern, name=name, kind=kind)
    _execute(
        ctx,
        ComponentFindCommand(
            request_id=_request_id("find"),
            op="component.find",
            selector=selector,
        ),
    )


def _find_selector_options(
    *, pattern: str | None, name: str | None, kind: FindKind
) -> ComponentSelector:
    if name is not None and pattern is not None:
        raise typer.BadParameter("provide either PATTERN or --name, not both")
    if name is not None and kind is not FindKind.AUTO:
        raise typer.BadParameter("--name already selects exact-name matching; omit --kind")
    if name is not None:
        return ComponentSelector(kind=SelectorKind.EXACT_NAME, pattern=name)
    if pattern is None:
        raise typer.BadParameter("provide PATTERN or --name")
    return _find_selector(kind, pattern)


def _find_selector(kind: FindKind, pattern: str) -> ComponentSelector:
    selector_kind = _find_selector_kind(kind, pattern)
    if selector_kind is SelectorKind.GLOB and "/" not in pattern:
        pattern = f"**/{pattern}"
    return ComponentSelector(kind=selector_kind, pattern=pattern)


def _find_selector_kind(kind: FindKind, pattern: str) -> SelectorKind:
    if kind is not FindKind.AUTO:
        return SelectorKind(kind.value)
    if any(character in pattern for character in "*?["):
        return SelectorKind.GLOB
    if "/" in pattern:
        return SelectorKind.EXACT_PATH
    return SelectorKind.EXACT_NAME


@app.command("inspect")
def inspect_component(ctx: typer.Context, component: Annotated[str, typer.Argument()]) -> None:
    """Inspect one component by ref, stable ID, exact display name, or exact path."""

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
    focus: Annotated[
        list[str] | None,
        typer.Option("--focus", help="Component ref, stable ID, exact name, or exact path."),
    ] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
    aperture_fstop: Annotated[float | None, typer.Option("--aperture-fstop", min=0.000001)] = None,
    focus_distance: Annotated[
        str | None, typer.Option("--focus-distance", help="Exact focus distance (mm, cm, or m).")
    ] = None,
    focus_component: Annotated[
        str | None,
        typer.Option("--focus-component", help="Component ref, stable ID, or exact path."),
    ] = None,
    disable_depth_of_field: Annotated[bool, typer.Option("--disable-depth-of-field")] = False,
) -> None:
    """Set an exact camera from a JSON object."""

    camera = Camera.model_validate_json(camera_json)
    focus_component_id = (
        _component_ids(ctx, [focus_component])[0] if focus_component is not None else None
    )
    camera = _camera_with_depth_of_field(
        camera,
        aperture_fstop=aperture_fstop,
        focus_distance=focus_distance,
        focus_component_id=focus_component_id,
        disable=disable_depth_of_field,
    )
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
    focus: Annotated[
        list[str] | None,
        typer.Option("--focus", help="Component ref, stable ID, exact name, or exact path."),
    ] = None,
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


@app.command("view-move")
def view_move(
    ctx: typer.Context,
    world_x: Annotated[
        str | None, typer.Option("--world-x", help="World X delta (mm, cm, or m).")
    ] = None,
    world_y: Annotated[
        str | None, typer.Option("--world-y", help="World Y delta (mm, cm, or m).")
    ] = None,
    world_z: Annotated[
        str | None, typer.Option("--world-z", help="World Z delta (mm, cm, or m).")
    ] = None,
    right: Annotated[
        str | None, typer.Option("--right", help="Camera-right delta (mm, cm, or m).")
    ] = None,
    up: Annotated[str | None, typer.Option("--up", help="Camera-up delta (mm, cm, or m).")] = None,
    forward: Annotated[
        str | None, typer.Option("--forward", help="Camera-forward delta (mm, cm, or m).")
    ] = None,
    focus: Annotated[list[str] | None, typer.Option("--focus")] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
) -> None:
    """Move the current camera by world- and camera-frame deltas."""

    _execute(
        ctx,
        ViewMoveCommand(
            request_id=_request_id("view-move"),
            op="view.move",
            world_delta_mm=(
                _distance_mm(world_x),
                _distance_mm(world_y),
                _distance_mm(world_z),
            ),
            camera_delta_mm=(
                _distance_mm(right),
                _distance_mm(up),
                _distance_mm(forward),
            ),
            focus_component_ids=_component_ids(ctx, focus or []) if focus else (),
            aspect_ratio=aspect_ratio,
        ),
    )


@app.command("view-rotate")
def view_rotate(
    ctx: typer.Context,
    target: Annotated[
        tuple[float, float, float],
        typer.Option("--target", help="Rotation center in the selected coordinate frame, in mm."),
    ],
    axis: Annotated[
        str,
        typer.Option("--axis", help="Positive X, Y, or Z axis in the selected frame."),
    ],
    degrees: Annotated[
        float,
        typer.Option("--degrees", help="Signed right-hand-rule rotation in degrees."),
    ],
    frame: Annotated[
        CoordinateFrame,
        typer.Option("--frame", help="Interpret target and axis in source or world coordinates."),
    ] = CoordinateFrame.WORLD,
    projection_json: Annotated[
        str | None,
        typer.Option(
            "--projection-json", help="Optional replacement projection; current by default."
        ),
    ] = None,
    focus: Annotated[list[str] | None, typer.Option("--focus")] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
) -> None:
    """Rotate the current camera about a source- or world-frame axis."""

    if frame not in {CoordinateFrame.SOURCE, CoordinateFrame.WORLD}:
        raise typer.BadParameter("--frame must be source or world")
    normalized_axis = axis.casefold()
    if normalized_axis not in {"x", "y", "z"}:
        raise typer.BadParameter("--axis must be x, y, or z")
    projection: Projection | None = None
    if projection_json is not None:
        from pydantic import TypeAdapter

        projection = TypeAdapter(Projection).validate_json(projection_json)
    _execute(
        ctx,
        ViewRotateCommand(
            request_id=_request_id("view-rotate"),
            op="view.rotate",
            target_mm=target,
            axis=normalized_axis,  # type: ignore[arg-type]
            degrees=degrees,
            frame=frame,  # type: ignore[arg-type]
            projection=projection,
            focus_component_ids=_component_ids(ctx, focus or []) if focus else (),
            aspect_ratio=aspect_ratio,
        ),
    )


@app.command("illumination-set")
def illumination_set(
    ctx: typer.Context,
    preset: Annotated[str, typer.Argument(help="A named preset, or 'custom'.")],
    illumination_json: Annotated[
        Path | None,
        typer.Option("--illumination-json", exists=True, dir_okay=False, readable=True),
    ] = None,
    background_rgb: Annotated[
        tuple[float, float, float] | None,
        typer.Option("--background-rgb", help="Override the visible linear-RGB background."),
    ] = None,
    background_strength: Annotated[
        float | None,
        typer.Option("--background-strength", min=0),
    ] = None,
) -> None:
    """Apply a named preset or a complete custom illumination JSON object."""

    from pydantic import TypeAdapter

    from meshprobe.models import (
        CustomIllumination,
        EnvironmentMap,
        Illumination,
        PresetIllumination,
    )

    illumination: Illumination
    if preset == "custom":
        if illumination_json is None:
            raise typer.BadParameter("custom requires --illumination-json")
        try:
            illumination = TypeAdapter(Illumination).validate_json(
                illumination_json.read_text(encoding="utf-8")
            )
        except KeyError as error:
            raise typer.BadParameter(f"missing required field: {error.args[0]}") from error
        except (OSError, ValidationError) as error:
            raise typer.BadParameter(str(error)) from error
        if not isinstance(illumination, CustomIllumination):
            raise typer.BadParameter("--illumination-json must declare preset 'custom'")
        if background_rgb is not None or background_strength is not None:
            raise typer.BadParameter(
                "background overrides cannot be combined with --illumination-json"
            )
        environment: EnvironmentMap | None = illumination.environment_map
        if environment is not None:
            environment_path = Path(environment.path).expanduser()
            if not environment_path.is_absolute():
                environment_path = (illumination_json.parent / environment_path).resolve()
                illumination = illumination.model_copy(
                    update={
                        "environment_map": environment.model_copy(
                            update={"path": str(environment_path)}
                        )
                    }
                )
    else:
        if illumination_json is not None:
            raise typer.BadParameter("--illumination-json requires the custom preset")
        try:
            named_preset = IlluminationPreset(preset)
        except ValueError as error:
            choices = ", ".join(item.value for item in IlluminationPreset)
            raise typer.BadParameter(f"preset must be one of: {choices}, custom") from error
        try:
            illumination = PresetIllumination(
                preset=named_preset,
                background_rgb=background_rgb,
                background_strength=background_strength,
            )
        except ValidationError as error:
            raise typer.BadParameter(str(error)) from error

    _execute(
        ctx,
        IlluminationSetCommand(
            request_id=_request_id("illumination"),
            op="illumination.set",
            illumination=illumination,
        ),
    )


@app.command("display")
def display_components(
    ctx: typer.Context,
    components: Annotated[list[str], typer.Argument()],
    mode: Annotated[DisplayMode, typer.Option("--mode")],
) -> None:
    """Change visibility using refs, stable IDs, exact display names, or exact paths."""

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
    color: Annotated[
        str | None,
        typer.Option("--color", help="sRGB highlight color in #RRGGBB form."),
    ] = None,
) -> None:
    """Mark components using refs, stable IDs, exact display names, or exact paths."""

    try:
        command = ComponentMarkCommand(
            request_id=_request_id("mark"),
            op="component.mark",
            component_ids=_component_ids(ctx, components),
            mode=mode,
            color=color,
        )
    except ValidationError as error:
        raise typer.BadParameter(str(error)) from error
    _execute(ctx, command)


@app.command("render-image")
def render_image(
    ctx: typer.Context,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    width: Annotated[int, typer.Option("--width", min=64, max=16_384)] = 1024,
    height: Annotated[int, typer.Option("--height", min=64, max=16_384)] = 1024,
    samples: Annotated[int, typer.Option("--samples", min=1, max=4_096)] = 64,
    engine: Annotated[RenderEngine, typer.Option("--engine")] = RenderEngine.EEVEE,
    graphics_policy: Annotated[
        GraphicsPolicy, typer.Option("--graphics-policy")
    ] = GraphicsPolicy.SOFTWARE_ALLOWED,
) -> None:
    """Render the selected session to an image artifact."""

    options = _options(ctx)
    destination = output or (
        workspace_root(options.workspace)
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
            graphics_policy=graphics_policy,
        ),
    )


@app.command("render-sheet")
def render_sheet(
    ctx: typer.Context,
    focus: Annotated[
        list[str],
        typer.Argument(help="Component refs, stable IDs, exact names, or exact paths."),
    ],
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    panel_width: Annotated[int, typer.Option("--panel-width", min=128)] = 768,
    panel_height: Annotated[int, typer.Option("--panel-height", min=128)] = 768,
    samples: Annotated[int, typer.Option("--samples", min=1)] = 32,
    engine: Annotated[RenderEngine, typer.Option("--engine")] = RenderEngine.EEVEE,
    graphics_policy: Annotated[
        GraphicsPolicy, typer.Option("--graphics-policy")
    ] = GraphicsPolicy.SOFTWARE_ALLOWED,
) -> None:
    """Render the focused 3x3 diagnostic sheet."""

    options = _options(ctx)
    destination = output or (
        workspace_root(options.workspace)
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
            graphics_policy=graphics_policy,
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
    output = _options(ctx).output
    if output in {"json", "raw", "yaml"}:
        payload = {"sessions": sessions}
        _emit_yaml(payload) if output == "yaml" else _emit(payload)
        return
    for session in sessions:
        graphics = session.get("graphics")
        renderer = "graphics=unknown"
        if isinstance(graphics, dict):
            renderer = (
                f"graphics={graphics.get('device_class', 'unknown')}/"
                f"{graphics.get('backend', 'unknown')} "
                f"renderer={graphics.get('renderer', 'unknown')}"
            )
        typer.echo(f"{session['name']}\t{session['status']}\t{renderer}\t{session['source_path']}")


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
    if all_sessions:
        _emit_receipts(ctx, client, receipts)
        return
    _emit_receipt(ctx, client, receipts[0])


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
    if all_sessions:
        _emit_receipts(ctx, client, receipts)
        return
    _emit_receipt(ctx, client, receipts[0])


@app.command("delete-data")
def delete_data(ctx: typer.Context) -> None:
    """Delete the stopped workspace's `.meshprobe` state."""

    client = _client(ctx)
    try:
        deleted = client.delete_data()
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    output = _options(ctx).output
    if output in {"json", "raw", "yaml"}:
        payload = {"deleted": str(deleted)}
        _emit_yaml(payload) if output == "yaml" else _emit(payload)
        return
    typer.echo(f"deleted {deleted}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
