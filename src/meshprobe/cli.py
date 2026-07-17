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
from typer import _click
from typer.core import TyperGroup

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
from meshprobe.evals.migration import audit_migration, migrate_corpus_v3
from meshprobe.evals.tiers import current_runtime_pin, pin_private_tier, pin_standard_tiers
from meshprobe.models import (
    Camera,
    ComponentFocus,
    DepthOfField,
    DepthOfFieldMode,
    DisplayMode,
    EdgeType,
    GraphicsPolicy,
    IlluminationPreset,
    MarkMode,
    PerspectiveProjection,
    Projection,
    RenderEngine,
    RenderStyle,
    RotationFrame,
    ShadedEdgesStyle,
)
from meshprobe.protocol import (
    Command,
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentInspectCommand,
    ComponentMarkCommand,
    ComponentOcclusionCommand,
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
    command_result_json_schema,
)
from meshprobe.selectors import (
    ComponentSelector,
    SelectorKind,
    is_glob_pattern,
    normalize_glob,
)
from meshprobe.service import MeshProbeService
from meshprobe.workspace import (
    OperationReceipt,
    durable_state_json_schema,
    durable_state_schema_summary,
    workspace_root,
)


def _is_option(token: str) -> bool:
    return token.startswith("-") and token != "-"


def _matches_option(token: str, names: set[str]) -> bool:
    if token.split("=", 1)[0] in names:
        return True
    return not token.startswith("--") and token[:2] in names


def _consumes_separate_value(token: str, value_names: set[str]) -> bool:
    return "=" not in token and token in value_names


class GlobalOptionGroup(TyperGroup):
    """Root group that also accepts its global options after the subcommand.

    Click only parses group-level options that precede the subcommand token, so
    ``meshprobe find foo --json`` would otherwise fail with "No such option"
    while ``meshprobe --json find foo`` works. This hoists any root global
    option (and its value) that trails the subcommand back in front of it, so
    ``--session``/``--workspace``/``--json``/``--yaml``/``--raw`` work in either
    position (issue #48). Argv is returned untouched when nothing moves, so
    existing before-subcommand usage is unaffected.
    """

    def parse_args(self, ctx: _click.Context, args: list[str]) -> list[str]:
        return super().parse_args(ctx, self._hoist_global_options(ctx, args))

    def _hoist_global_options(self, ctx: _click.Context, args: list[str]) -> list[str]:
        global_names, global_value_names = self._option_names(self, ctx)
        command_index = self._command_index(args, global_value_names)
        if command_index is None:
            return args
        command = args[command_index]
        subcommand = self.get_command(ctx, command)
        subcommand_names, subcommand_value_names = self._option_names(subcommand, ctx)
        hoisted, trailing = self._split_global_options(
            args[command_index + 1 :],
            global_names - subcommand_names,
            global_value_names | subcommand_value_names,
        )
        if not hoisted:
            return args
        return [*args[:command_index], *hoisted, command, *trailing]

    @staticmethod
    def _option_names(
        command: _click.Command | None, ctx: _click.Context
    ) -> tuple[set[str], set[str]]:
        names: set[str] = set()
        value_names: set[str] = set()
        if command is None:
            return names, value_names
        for param in command.get_params(ctx):
            if param.param_type_name != "option":
                continue
            option_names = (*param.opts, *param.secondary_opts)
            names.update(option_names)
            if not getattr(param, "is_flag", False) and not getattr(param, "count", False):
                value_names.update(option_names)
        return names, value_names

    @staticmethod
    def _command_index(args: list[str], value_names: set[str]) -> int | None:
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--":
                return None
            if not _is_option(token):
                return index
            index += 1 + _consumes_separate_value(token, value_names)
        return None

    @staticmethod
    def _split_global_options(
        trailing: list[str], global_names: set[str], value_names: set[str]
    ) -> tuple[list[str], list[str]]:
        hoisted: list[str] = []
        remaining: list[str] = []
        index = 0
        while index < len(trailing):
            token = trailing[index]
            if token == "--":
                remaining.extend(trailing[index:])
                break
            if not _is_option(token):
                remaining.append(token)
                index += 1
                continue
            target = hoisted if _matches_option(token, global_names) else remaining
            target.append(token)
            index += 1
            if _consumes_separate_value(token, value_names) and index < len(trailing):
                target.append(trailing[index])
                index += 1
        return hoisted, remaining


app = typer.Typer(
    help="Read-only 3D model inspection for AI agents.",
    no_args_is_help=True,
    cls=GlobalOptionGroup,
)
eval_app = typer.Typer(help="Build and validate qualification corpora.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
DEFAULT_WORKSPACE = Path.cwd()


class CliOptions:
    def __init__(self, session: str, workspace: Path, output: str) -> None:
        self.session = session
        self.workspace = workspace
        self.output = output


def _print_version(value: bool) -> None:
    if not value:
        return
    from meshprobe import __version__

    typer.echo(f"meshprobe {__version__}")
    raise typer.Exit()


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
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the installed meshprobe version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
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
    RESULTS = "results"
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
        typer.Option(
            "--kind",
            help="Show command inputs, command result envelopes, durable state files, or all.",
        ),
    ] = SchemaKind.STATE,
    full: Annotated[
        bool,
        typer.Option("--full", help="Expand durable state fields into formal JSON Schemas."),
    ] = False,
) -> None:
    """Print machine-readable schemas for command inputs, result envelopes, or durable state.

    The ``results`` kind documents the full JSON shape each command returns — including
    otherwise-undocumented fields such as render luminance, projected component bounds,
    occlusion traces, and contact-sheet panel captions and callouts.
    """

    payload: object = command_json_schema()
    if kind is SchemaKind.RESULTS:
        payload = command_result_json_schema()
    if kind is SchemaKind.STATE:
        payload = durable_state_json_schema() if full else durable_state_schema_summary()
    if kind is SchemaKind.ALL:
        state = durable_state_json_schema() if full else durable_state_schema_summary()
        payload = {
            "commands": command_json_schema(),
            "results": command_result_json_schema(),
            "state": state,
        }
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
        build, audit = migrate_corpus_v3(
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
    corpus_version: Annotated[str, typer.Option("--version")] = "qualification-v7",
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
    match_count = receipt.match_count
    if match_count is None and receipt.op == "component.find" and receipt.result_path:
        # A daemon still running the pre-upgrade protocol omits match_count entirely; derive
        # it from the persisted result so the zero-match warning survives without a restart.
        envelope = client.read_result(receipt)
        result = envelope.get("result") if isinstance(envelope, dict) else envelope
        if isinstance(result, list):
            match_count = len(result)
    fields = ["ok", f"session={receipt.session}", f"op={receipt.op}"]
    if match_count is not None:
        fields.append(f"matches={match_count}")
    if receipt.result_path:
        fields.append(f"result={receipt.result_path}")
    if receipt.state_path:
        fields.append(f"state={receipt.state_path}")
    if receipt.artifact_paths:
        fields.append(f"artifacts={','.join(receipt.artifact_paths)}")
    typer.echo(" ".join(fields))
    if match_count == 0:
        typer.echo("warning: no components matched", err=True)
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
    """Resolve refs/IDs/names/paths and expand glob patterns, like `find`.

    Each glob token fans out to every matching component; exact tokens resolve to
    one. Order follows the arguments and, within a glob, component path; duplicates
    are dropped so an overlapping glob and ref map to a single entry.
    """

    client = _client(ctx)
    session = _options(ctx).session
    resolved: dict[str, None] = {}
    try:
        for component in components:
            for component_id in client.resolve_component_ids(session, component):
                resolved[component_id] = None
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    return tuple(resolved)


def _component_id(ctx: typer.Context, value: str) -> str:
    client = _client(ctx)
    try:
        return client.resolve_component(_options(ctx).session, value)
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
    if selector_kind is SelectorKind.GLOB:
        pattern = normalize_glob(pattern)
    return ComponentSelector(kind=selector_kind, pattern=pattern)


def _find_selector_kind(kind: FindKind, pattern: str) -> SelectorKind:
    if kind is not FindKind.AUTO:
        return SelectorKind(kind.value)
    if is_glob_pattern(pattern):
        return SelectorKind.GLOB
    if "/" in pattern:
        return SelectorKind.EXACT_PATH
    return SelectorKind.EXACT_NAME


@app.command("inspect")
def inspect_component(ctx: typer.Context, component: Annotated[str, typer.Argument()]) -> None:
    """Inspect one component by ref, stable ID, exact display name, or exact path."""

    component_id = _component_id(ctx, component)
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
        typer.Option("--focus", help="Component ref, stable ID, exact name, exact path, or glob."),
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
        _component_id(ctx, focus_component) if focus_component is not None else None
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
    target: Annotated[
        tuple[float, float, float],
        typer.Option("--target", help="Absolute world-space X Y Z target in millimeters."),
    ],
    azimuth: Annotated[
        float,
        typer.Option("--azimuth", help="Absolute degrees from +X toward +Y about world +Z."),
    ],
    elevation: Annotated[
        float,
        typer.Option("--elevation", help="Absolute degrees above the world XY plane toward +Z."),
    ],
    distance: Annotated[
        float,
        typer.Option("--distance", min=0.000001, help="Absolute target distance in millimeters."),
    ],
    projection_json: Annotated[
        str,
        typer.Option(
            "--projection-json",
            help="Complete perspective or orthographic projection, including camera intrinsics "
            "(focal_length_mm, sensor_width_mm/sensor_height_mm, sensor_fit, depth_of_field).",
        ),
    ],
    roll: Annotated[
        float,
        typer.Option("--roll", help="Signed right-hand-rule degrees around the forward axis."),
    ] = 0.0,
    focus: Annotated[
        list[str] | None,
        typer.Option("--focus", help="Component ref, stable ID, exact name, exact path, or glob."),
    ] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
) -> None:
    """Set an absolute orbit in right-handed, Z-up MeshProbe world coordinates.

    Angles are degrees, not relative deltas. Unlike the cumulative view-move and
    view-rotate, this command discards the current pose and replaces it wholesale, so
    it does not compose with a prior view call. To continue an orbit, read the current
    camera_orbit_angles (azimuth_degrees, elevation_degrees) from state.yml and pass
    them back here. Positive azimuth follows the right-hand rule around +Z. GLTF is
    right-handed and Y-up; scene.open reports the exact source_to_world mapping
    (GLTF +X -> world +X, +Y -> +Z, +Z -> -Y). The result echoes the resolved camera
    (including the intrinsics above) and its diagnostics; see `meshprobe schema --kind results`.
    """

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
    """Move the current camera by world- and camera-frame deltas.

    Cumulative: deltas compose on top of the current pose, like view-rotate and unlike
    the absolute view-orbit (which replaces the pose wholesale).
    """

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
        typer.Option(
            "--degrees",
            help="Signed right-hand-rule visual/model rotation; camera orbit is the inverse.",
        ),
    ],
    frame: Annotated[
        RotationFrame,
        typer.Option(
            "--frame",
            help=(
                "Interpret target and axis in source (the model's own axes, the default), "
                "world, or camera coordinates. A bare '--axis Y' spins a Y-up glTF in "
                "place; '--frame world' addresses MeshProbe's Z-up world axes; "
                "'--frame camera' rotates around the camera's own local axes, so the "
                "on-screen effect stays the same regardless of which way the camera faces."
            ),
        ),
    ] = RotationFrame.SOURCE,
    projection_json: Annotated[
        str | None,
        typer.Option(
            "--projection-json", help="Optional replacement projection; current by default."
        ),
    ] = None,
    basis_json: Annotated[
        str | None,
        typer.Option(
            "--basis-json",
            help="Right-handed orthonormal basis JSON with x, y, and z vectors.",
        ),
    ] = None,
    focus: Annotated[list[str] | None, typer.Option("--focus")] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
) -> None:
    """Show the visual equivalent of rotating the model around a basis axis.

    The axis and target are read in the source model's own frame by default, so
    '--axis Y' spins a Y-up glTF in place. Pass '--frame world' to address
    MeshProbe's Z-up world axes, or '--frame camera' to rotate around the camera's
    own local axes — a direction-independent tilt/pan/roll whose sign does not flip
    when the camera moves to the other side of the model.

    Cumulative: each rotation composes on top of the current pose, like view-move and
    unlike the absolute view-orbit (which replaces the pose wholesale).
    """

    normalized_axis = axis.casefold()
    if normalized_axis not in {"x", "y", "z"}:
        raise typer.BadParameter("--axis must be x, y, or z")
    projection: Projection | None = None
    if projection_json is not None:
        from pydantic import TypeAdapter

        projection = TypeAdapter(Projection).validate_json(projection_json)
    from meshprobe.models import OrthonormalBasis

    basis = (
        OrthonormalBasis.model_validate_json(basis_json)
        if basis_json is not None
        else OrthonormalBasis()
    )
    _execute(
        ctx,
        ViewRotateCommand(
            request_id=_request_id("view-rotate"),
            op="view.rotate",
            target_mm=target,
            axis=normalized_axis,  # type: ignore[arg-type]
            degrees=degrees,
            frame=frame,  # type: ignore[arg-type]
            basis=basis,
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
        typer.Option(
            "--background-rgb",
            help=(
                "Override the visible background as a linear-referred RGB triple; it is "
                "tone-mapped by the active view transform, so 1 1 1 is NOT pure white. "
                "For an exact PNG color use --background-srgb."
            ),
        ),
    ] = None,
    background_strength: Annotated[
        float | None,
        typer.Option(
            "--background-strength",
            min=0,
            help="Scale the linear --background-rgb emission before tone-mapping.",
        ),
    ] = None,
    background_srgb: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            "--background-srgb",
            help=(
                "Set the visible background to an exact display-referred sRGB color "
                "(0..1 per channel); 1 1 1 renders as pure white (255) in the saved PNG. "
                "Cannot be combined with --background-rgb/--background-strength."
            ),
        ),
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
        if (
            background_rgb is not None
            or background_strength is not None
            or background_srgb is not None
        ):
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
        if background_srgb is not None and (
            background_rgb is not None or background_strength is not None
        ):
            raise typer.BadParameter(
                "--background-srgb is display-referred and cannot be combined with the "
                "linear-referred --background-rgb/--background-strength"
            )
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
                background_srgb=background_srgb,
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
    """Change visibility using refs, stable IDs, exact names, exact paths, or globs."""

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
    """Mark components using refs, stable IDs, exact names, exact paths, or globs."""

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
    style: Annotated[RenderStyle, typer.Option("--style")] = RenderStyle.SHADED,
    edge_color: Annotated[str, typer.Option("--edge-color")] = "#202020",
    edge_width: Annotated[float, typer.Option("--edge-width", min=0.01, max=10)] = 1.5,
    crease_angle: Annotated[
        float,
        typer.Option(
            "--crease-angle",
            min=0.01,
            max=180,
            help="Maximum face angle in degrees classified as a crease.",
        ),
    ] = 120,
    edge_types: Annotated[
        str,
        typer.Option(
            "--edge-types",
            help="Comma-separated silhouette, border, crease, and material boundary types.",
        ),
    ] = "silhouette,border,crease,material_boundary",
    graphics_policy: Annotated[
        GraphicsPolicy, typer.Option("--graphics-policy")
    ] = GraphicsPolicy.SOFTWARE_ALLOWED,
) -> None:
    """Render the selected session to an image artifact.

    The result carries a `luminance` exposure summary (median, clipped/crushed fractions) so
    a too-dark or blown-out frame can be detected without inspecting pixels. See
    `meshprobe schema --kind results` for the full render result shape.
    """

    options = _options(ctx)
    destination = output or (
        workspace_root(options.workspace)
        / "sessions"
        / options.session
        / "artifacts"
        / f"render-{uuid.uuid4().hex[:12]}.png"
    )
    try:
        selected_edge_types = tuple(EdgeType(value.strip()) for value in edge_types.split(","))
    except ValueError as error:
        choices = ", ".join(item.value for item in EdgeType)
        raise typer.BadParameter(f"edge types must be selected from: {choices}") from error
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
            style=style,
            shaded_edges=ShadedEdgesStyle(
                line_color=edge_color,
                line_width=edge_width,
                crease_angle_degrees=crease_angle,
                edge_types=selected_edge_types,
            ),
            graphics_policy=graphics_policy,
        ),
    )


@app.command("render-sheet")
def render_sheet(
    ctx: typer.Context,
    focus: Annotated[
        list[str],
        typer.Argument(
            help="Component refs, stable IDs, exact names, exact paths, or glob patterns."
        ),
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
    """Render a focused 3x3 sheet with automatic occlusion analysis.

    Each panel result carries a structured `caption` and numbered `callouts` (with marker
    positions), and the sheet carries the full occlusion-removal trace. See
    `meshprobe schema --kind results` for the full contact-sheet result shape.
    """

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


@app.command("occlusion")
def occlusion(
    ctx: typer.Context,
    focus: Annotated[
        list[str],
        typer.Argument(
            help="Component refs, stable IDs, exact names, exact paths, or glob patterns."
        ),
    ],
    max_samples: Annotated[int, typer.Option("--max-samples", min=1, max=4_096)] = 128,
) -> None:
    """Measure focus visibility and blockers from the current session camera.

    The result's `camera_diagnostics.projected_bounds` reports each focus component's
    normalized frame coverage, and the `camera` block exposes the full camera intrinsics
    (focal length, sensor size and fit). See `meshprobe schema --kind results` for the full
    occlusion result shape.
    """

    _execute(
        ctx,
        ComponentOcclusionCommand(
            request_id=_request_id("occlusion"),
            op="component.occlusion",
            component_ids=_component_ids(ctx, focus),
            max_samples_per_component=max_samples,
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
