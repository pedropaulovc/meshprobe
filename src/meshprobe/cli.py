"""Command-line access to MeshProbe's renderer-independent contracts."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import typer
import yaml
from pydantic import ValidationError
from typer.core import TyperGroup

if TYPE_CHECKING:
    # Typer vendors its Click as ``typer._click``; used only to match the
    # overridden ``TyperGroup.parse_args`` signature for the type checker. It is
    # never imported at runtime, so Typer releases that predate the vendored
    # module still import cleanly.
    from typer import _click

from meshprobe.client import MeshProbeClient
from meshprobe.controller import DEFAULT_WORKER_TIMEOUT_SECONDS
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
    IsolationOperation,
    MarkMode,
    OrbitSweep,
    OrthographicProjection,
    PerspectiveProjection,
    Projection,
    RenderEngine,
    RenderStyle,
    RotationFrame,
    ShadedEdgesStyle,
)
from meshprobe.protocol import (
    COMMAND_MODELS,
    Command,
    ComponentDisplayCommand,
    ComponentFindCommand,
    ComponentInspectCommand,
    ComponentMarkCommand,
    ComponentOcclusionCommand,
    IlluminationSetCommand,
    RenderComparisonRequest,
    RenderContactSheetCommand,
    RenderImageCommand,
    SessionResetCommand,
    SessionSnapshotCommand,
    SessionUndoCommand,
    ViewFrameCommand,
    ViewMoveCommand,
    ViewOrbitCommand,
    ViewRotateCommand,
    ViewSetCommand,
    command_json_schema,
    command_json_schema_for,
    command_result_json_schema,
    command_result_json_schema_for,
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

    Only leaf subcommands are eligible: a nested group (``eval``) is left
    untouched, since its trailing options belong to a leaf we do not enumerate
    here and could share a name with a root global (e.g. ``eval generate
    --version`` vs the root ``--version``); those subcommands do not consume the
    global receipt/session options anyway.
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
        if subcommand is None or hasattr(subcommand, "get_command"):
            return args
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
    context_settings={"show_default": True},
    # Click's line-oriented formatter keeps every flag and command searchable
    # with standard shell tools; Rich's box tables wrap long descriptions.
    rich_markup_mode=None,
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


# CLI subcommand names (as registered via `@app.command(...)`) that don't reduce to
# their protocol `op` value by a plain dash-for-dot swap, e.g. "view-orbit" already
# matches "view.orbit" without an alias, but "occlusion" needs one to reach
# "component.occlusion". Kept minimal — every other command name is resolved by
# `_resolve_command_op` swapping "-" for "." against `COMMAND_MODELS`.
_CLI_COMMAND_OP_ALIASES: dict[str, str] = {
    "open": "scene.open",
    "snapshot": "session.snapshot",
    "find": "component.find",
    "inspect": "component.inspect",
    "occlusion": "component.occlusion",
    "display": "component.display",
    "mark": "component.mark",
    "render-sheet": "render.contact_sheet",
    "reset": "session.reset",
    "undo": "session.undo",
}


def _resolve_command_op(name: str) -> str | None:
    """Resolve a user-supplied command name to its protocol ``op`` value.

    Accepts the protocol ``op`` string itself (``view.orbit``), the CLI
    subcommand name shown in ``--help`` (``view-orbit``), or one of the
    small set of CLI names that don't reduce to ``op`` by a dash-for-dot
    swap (``occlusion`` -> ``component.occlusion``). Returns ``None`` if
    ``name`` doesn't match any known command.
    """
    if name in COMMAND_MODELS:
        return name
    alias = _CLI_COMMAND_OP_ALIASES.get(name)
    if alias is not None:
        return alias
    dotted = name.replace("-", ".")
    if dotted in COMMAND_MODELS:
        return dotted
    return None


def _cli_command_names() -> list[str]:
    """All command names accepted by the per-command schema lookup, CLI-style."""
    reverse_aliases = {op: cli_name for cli_name, op in _CLI_COMMAND_OP_ALIASES.items()}
    return sorted(reverse_aliases.get(op, op.replace(".", "-")) for op in COMMAND_MODELS)


# These fields cannot be recovered from Click's parameter tree.  Keeping them in a
# small registry means the Markdown and JSON cmdhelp renderers still have one
# canonical source for examples and stream semantics.
_CMDHELP_EXAMPLES: dict[str, tuple[tuple[str, str], ...]] = {
    "open": (("meshprobe open models/gearbox.glb", "Open a model in the default session."),),
    "snapshot": (("meshprobe snapshot", "Persist and print the current session snapshot."),),
    "find": (("meshprobe find '**/idler*'", "Find components by a wildcard path."),),
    "inspect": (("meshprobe inspect c7", "Inspect a component by ref or stable ID."),),
    "view-orbit": (
        (
            "meshprobe view-orbit --target 0 0 0 --azimuth 45 --elevation 30 --distance 500",
            "Set an absolute camera orbit.",
        ),
    ),
    "view-frame": (("meshprobe view-frame c7", "Frame a component tightly."),),
    "view-move": (("meshprobe view-move --forward 25mm", "Move the current camera."),),
    "view-rotate": (
        (
            "meshprobe view-rotate --target 0 0 0 --axis Y --degrees 15",
            "Apply a cumulative visual rotation.",
        ),
    ),
    "illumination-set": (
        ("meshprobe illumination-set raking_left", "Apply a diagnostic lighting preset."),
    ),
    "display": (("meshprobe display c7 --mode isolated", "Isolate a component."),),
    "mark": (("meshprobe mark c7 --mode highlighted", "Highlight a component."),),
    "render-image": (("meshprobe render-image --output idler.png", "Render the current view."),),
    "render-sheet": (
        (
            "meshprobe render-sheet '**/*bearing*' --output bearings.png",
            "Render a focused 3x3 inspection sheet.",
        ),
    ),
    "occlusion": (("meshprobe occlusion c7", "Measure visibility and blockers."),),
    "reset": (("meshprobe reset", "Restore imported visual state."),),
    "undo": (("meshprobe undo 2", "Undo the two latest state-changing commands."),),
    "list": (("meshprobe list", "List durable sessions."),),
    "close": (("meshprobe close", "Checkpoint and stop the selected renderer."),),
    "kill": (("meshprobe kill --all", "Force-stop all renderers."),),
    "delete-data": (("meshprobe delete-data", "Delete stopped session data."),),
    "schema": (("meshprobe schema --kind results", "Inspect command result schemas."),),
    "eval generate": (
        (
            "meshprobe eval generate .corpora --version procedural-v7",
            "Build the procedural evaluation corpus.",
        ),
    ),
    "eval validate": (("meshprobe eval validate .corpora/procedural-v7", "Validate a corpus."),),
    "eval migrate": (
        (
            "meshprobe eval migrate .corpora/procedural-v5 .corpora --version procedural-v7",
            "Migrate a corpus while preserving identities.",
        ),
    ),
    "eval merge": (
        (
            "meshprobe eval merge .corpora .corpora/procedural-v7 .corpora/curated-tasks-v7",
            "Combine validated corpora.",
        ),
    ),
}

_CMDHELP_SEE_ALSO: dict[str, tuple[str, ...]] = {
    "open": ("snapshot", "close", "delete-data"),
    "find": ("inspect", "occlusion"),
    "view-orbit": ("view-frame", "view-move", "view-rotate"),
    "view-frame": ("view-orbit", "render-image"),
    "render-image": ("render-sheet", "schema"),
    "render-sheet": ("occlusion", "render-image"),
    "schema": ("help",),
}

_CMDHELP_EXIT_CODES: dict[str, object] = {
    "0": "ok",
    "1": {"when": "unexpected runtime failure", "recovery": "Inspect the error and retry."},
    "2": {
        "when": "invalid arguments or command state",
        "recovery": "Check the command's arguments and flags, then retry.",
    },
}


def _cmdhelp_command_is_group(command: object) -> bool:
    return bool(getattr(command, "commands", None))


def _cmdhelp_type(param: object) -> tuple[str, list[str] | None]:
    parameter_type = getattr(param, "type", None)
    choices = getattr(parameter_type, "choices", None)
    if choices is not None:
        return "enum", [str(choice) for choice in choices]
    if getattr(param, "is_flag", False):
        return "bool", None
    type_name = str(getattr(parameter_type, "name", "string")).lower()
    class_name = type(parameter_type).__name__.lower()
    if "path" in class_name or type_name in {"path", "file", "directory"}:
        return "path", None
    if "int" in type_name or "intrange" in class_name:
        return "int", None
    if "float" in type_name or "floatrange" in class_name:
        return "float", None
    if "bool" in type_name:
        return "bool", None
    if "tuple" in class_name:
        return "string", None
    return {
        "integer": "int",
        "number": "float",
        "text": "string",
    }.get(
        type_name, type_name if type_name in {"string", "json", "url", "date"} else "string"
    ), None


def _cmdhelp_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None and isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if isinstance(value, (tuple, list)):
        return [_cmdhelp_json_value(item) for item in value]
    return str(value)


def _cmdhelp_parameter(param: object) -> dict[str, object]:
    is_argument = getattr(param, "is_flag", None) is None
    parameter_type, enum = _cmdhelp_type(param)
    names = list(getattr(param, "opts", ()))
    if is_argument:
        payload: dict[str, object] = {
            "name": getattr(param, "name", "argument"),
            "type": parameter_type,
            "required": bool(getattr(param, "required", False)),
            "description": getattr(param, "help", None)
            or f"Value for {getattr(param, 'name', 'argument')}.",
        }
    else:
        long_name = next((name for name in names if name.startswith("--")), names[0])
        key = long_name.lstrip("-").replace("-", "_")
        payload = {
            "flag": long_name,
            "type": parameter_type,
            "required": bool(getattr(param, "required", False)),
            "description": getattr(param, "help", None) or f"Set {key}.",
        }
        aliases = [name for name in names if name != long_name]
        if aliases:
            payload["aliases"] = aliases
        if getattr(param, "multiple", False):
            payload["repeatable"] = True
        if getattr(param, "is_flag", False):
            payload["presence_switch"] = True
    if enum is not None:
        payload["enum"] = enum
    nargs = getattr(param, "nargs", 1)
    if nargs > 1:
        payload["arity"] = nargs
    default = getattr(param, "default", None)
    if default is not None:
        if getattr(param, "name", None) == "workspace" and isinstance(default, Path):
            default = "current directory"
        payload["default"] = _cmdhelp_json_value(default)
    metavar = getattr(param, "metavar", None)
    if metavar:
        payload["metavar"] = metavar
    return payload


def _cmdhelp_command_document(
    path: tuple[str, ...], command: object, detail: bool
) -> dict[str, object]:
    name = " ".join(path)
    raw_help = (getattr(command, "help", None) or "").strip()
    summary = next((line.strip() for line in raw_help.splitlines() if line.strip()), name)
    document: dict[str, object] = {"summary": summary}
    if raw_help and raw_help != summary:
        document["description"] = raw_help
    children = tuple(getattr(command, "commands", {}).keys())
    if children:
        document["subcommands"] = [" ".join((*path, child)) for child in children]
    if not detail:
        return document
    params = tuple(getattr(command, "params", ()))
    arguments = [
        _cmdhelp_parameter(param) for param in params if getattr(param, "is_flag", None) is None
    ]
    flags = [
        _cmdhelp_parameter(param) for param in params if getattr(param, "is_flag", None) is not None
    ]
    document["usage"] = _cmdhelp_synopsis(path, params)
    document["args"] = arguments
    document["flags"] = {
        str(payload["flag"]).lstrip("-").replace("-", "_"): payload for payload in flags
    }
    document["stdin"] = {"accepted": False, "description": "This command does not read stdin."}
    document["stdout"] = {
        "formats": ["text", "json", "yaml"],
        "description": (
            "Writes a receipt or command result to stdout; use --json, --yaml, or --raw "
            "for structured output."
        ),
    }
    op = _resolve_command_op(name)
    if op is not None:
        stdout = cast(dict[str, object], document["stdout"])
        stdout["json_schema_command"] = f"meshprobe schema {path[-1]} --kind results"
    document["exit_codes"] = _CMDHELP_EXIT_CODES
    document["examples"] = [
        {"cmd": example, "note": note} for example, note in _CMDHELP_EXAMPLES.get(name, ())
    ]
    see_also = _CMDHELP_SEE_ALSO.get(name)
    if see_also:
        document["see_also"] = ["meshprobe " + item for item in see_also]
    context: dict[str, str] = {
        "session": "The durable session is selected with --session (default: default).",
        "workspace": (
            "Session data is stored under .meshprobe in --workspace (default: current directory)."
        ),
    }
    if name in {"find", "inspect", "display", "mark", "occlusion", "render-sheet", "view-frame"}:
        context["components"] = (
            "Component refs, IDs, names, paths, and globs resolve against the selected session."
        )
    document["context"] = context
    return document


def _cmdhelp_synopsis(path: tuple[str, ...], params: tuple[object, ...]) -> str:
    parts = ["meshprobe", *path]
    for param in params:
        if getattr(param, "is_flag", None) is None:
            name = str(getattr(param, "metavar", None) or getattr(param, "name", "ARG")).upper()
            if getattr(param, "required", False):
                parts.append(name)
            else:
                parts.append(f"[{name}]")
    parts.append("[OPTIONS]")
    return " ".join(parts)


def _cmdhelp_lookup(root: object, path: tuple[str, ...]) -> object | None:
    command = root
    for part in path:
        commands = getattr(command, "commands", {})
        command = commands.get(part)
        if command is None:
            return None
    return command


def _cmdhelp_documents(
    root: object, selected_path: tuple[str, ...], depth: int | None, all_commands: bool
) -> tuple[dict[str, object], object]:
    selected = _cmdhelp_lookup(root, selected_path)
    if selected is None:
        valid = ", ".join(_cmdhelp_paths(root))
        raise typer.BadParameter(
            f"Unknown command {' '.join(selected_path)!r}. Valid commands: {valid}",
            param_hint="COMMAND",
        )
    is_leaf = not _cmdhelp_command_is_group(selected)
    effective_depth = 99 if all_commands else depth
    if effective_depth is None:
        effective_depth = 99 if is_leaf else 0
    documents: dict[str, object] = {}
    selected_detail = is_leaf and (depth is None or all_commands or depth > 0)
    if selected_path:
        documents[" ".join(selected_path)] = _cmdhelp_command_document(
            selected_path, selected, selected_detail
        )
    max_relative = effective_depth + 1

    def visit(command: object, path: tuple[str, ...], relative: int) -> None:
        for child_name, child in getattr(command, "commands", {}).items():
            child_path = (*path, child_name)
            if relative > max_relative:
                continue
            child_is_leaf = not _cmdhelp_command_is_group(child)
            detail = relative <= effective_depth
            documents[" ".join(child_path)] = _cmdhelp_command_document(child_path, child, detail)
            if not child_is_leaf:
                visit(child, child_path, relative + 1)

    if _cmdhelp_command_is_group(selected) or not selected_path:
        visit(selected, selected_path, 1)
    return documents, selected


def _cmdhelp_paths(root: object) -> list[str]:
    paths: list[str] = []

    def visit(command: object, prefix: tuple[str, ...]) -> None:
        for name, child in getattr(command, "commands", {}).items():
            path = (*prefix, name)
            paths.append(" ".join(path))
            if _cmdhelp_command_is_group(child):
                visit(child, path)

    visit(root, ())
    return paths


def _cmdhelp_json(
    root: object, selected_path: tuple[str, ...], depth: int | None, all_commands: bool
) -> dict[str, object]:
    documents, _ = _cmdhelp_documents(root, selected_path, depth, all_commands)
    schemas: dict[str, object] = {}
    for name, raw_document in documents.items():
        document = cast(dict[str, object], raw_document)
        if "usage" not in document:
            continue
        op = _resolve_command_op(name)
        if op is None:
            continue
        schemas[op] = command_result_json_schema_for(op)
        stdout = cast(dict[str, object], document["stdout"])
        stdout["json_schema_ref"] = f"#/schemas/{op}"
    root_params = tuple(getattr(root, "params", ()))
    global_flags = {
        str(payload["flag"]).lstrip("-").replace("-", "_"): payload
        for payload in (_cmdhelp_parameter(param) for param in root_params)
        if "flag" in payload
    }
    from meshprobe import __version__

    return {
        "cmdhelp_version": "0.1",
        "binary": "meshprobe",
        "version": __version__,
        "summary": "Read-only 3D model inspection for AI agents.",
        "homepage": "https://github.com/pedropaulovc/meshprobe",
        "global_flags": global_flags,
        "commands": documents,
        "schemas": schemas,
        "context": {
            "workspace": "selected with --workspace; defaults to the current directory",
            "session": "selected with --session; defaults to default",
        },
    }


def _cmdhelp_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _cmdhelp_markdown(
    root: object, selected_path: tuple[str, ...], depth: int | None, all_commands: bool
) -> str:
    documents, _ = _cmdhelp_documents(root, selected_path, depth, all_commands)
    from meshprobe import __version__

    lines = [
        "---",
        'cmdhelp_version: "0.1"',
        "binary: meshprobe",
        f"version: {__version__}",
        "---",
        "",
        "# meshprobe",
        "",
        "Read-only 3D model inspection for AI agents.",
        "",
        "## Global flags",
        "",
    ]
    root_params = tuple(getattr(root, "params", ()))
    lines.extend(["| flag | type | default | description |", "| --- | --- | --- | --- |"])
    for param in root_params:
        payload = _cmdhelp_parameter(param)
        if "flag" not in payload:
            continue
        lines.append(
            "| "
            + " | ".join(
                _cmdhelp_escape(payload.get(key, ""))
                for key in ("flag", "type", "default", "description")
            )
            + " |"
        )
    for name, raw_document in documents.items():
        document = cast(dict[str, object], raw_document)
        lines.extend(["", f"## `meshprobe {name}`", "", str(document["summary"])])
        if "subcommands" in document:
            lines.extend(["", "### Subcommands", ""])
            for child in cast(list[str], document["subcommands"]):
                child_document = cast(dict[str, object], documents.get(child, {}))
                lines.append(f"- `meshprobe {child}` — {child_document.get('summary', '')}")
        if "usage" not in document:
            continue
        lines.extend(["", "### Synopsis", "", f"`{document['usage']}`", "", "### Arguments", ""])
        arguments = cast(list[dict[str, object]], document["args"])
        lines.extend(["| name | type | required | description |", "| --- | --- | --- | --- |"])
        for argument in arguments:
            lines.append(
                "| "
                + " | ".join(
                    _cmdhelp_escape(argument.get(key, ""))
                    for key in ("name", "type", "required", "description")
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Flags",
                "",
                "| flag | type | default | description |",
                "| --- | --- | --- | --- |",
            ]
        )
        for flag in cast(dict[str, dict[str, object]], document["flags"]).values():
            lines.append(
                "| "
                + " | ".join(
                    _cmdhelp_escape(flag.get(key, ""))
                    for key in ("flag", "type", "default", "description")
                )
                + " |"
            )
        lines.extend(
            ["", "### Stdin", "", str(cast(dict[str, object], document["stdin"])["description"])]
        )
        lines.extend(["", "### Examples", ""])
        for example in cast(list[dict[str, str]], document["examples"]):
            lines.extend([f"```bash\n{example['cmd']}\n```", f"{example['note']}", ""])
        if not document["examples"]:
            lines.append("No canonical examples are currently registered.")
        stdout = cast(dict[str, object], document["stdout"])
        lines.extend(["", "### Output", "", str(stdout["description"])])
        if stdout.get("json_schema_command"):
            lines.append(f"JSON result schema: `{stdout['json_schema_command']}`.")
        lines.extend(["", "### Exit codes", "", "| code | meaning |", "| --- | --- |"])
        for code, meaning in cast(dict[str, object], document["exit_codes"]).items():
            detail = (
                meaning
                if isinstance(meaning, str)
                else cast(dict[str, object], meaning).get("when", "")
            )
            lines.append(f"| {code} | {_cmdhelp_escape(detail)} |")
        lines.extend(["", "### Workspace context", ""])
        for key, value in cast(dict[str, str], document["context"]).items():
            lines.append(f"- **{key}:** {value}")
        if document.get("see_also"):
            lines.extend(["", "### See also", ""])
            lines.extend(f"- `{item}`" for item in cast(list[str], document["see_also"]))
    return "\n".join(lines).rstrip() + "\n"


@app.command("help")
def cmdhelp(
    ctx: typer.Context,
    command: Annotated[list[str] | None, typer.Argument(metavar="[COMMAND]...")] = None,
    format: Annotated[
        Literal["text", "md", "json", "llm"], typer.Option("--format", help="Output format.")
    ] = "text",
    depth: Annotated[
        int | None, typer.Option("--depth", min=0, help="Expand the command tree to N levels.")
    ] = None,
    all_commands: Annotated[
        bool, typer.Option("--all", help="Expand the complete command tree.")
    ] = False,
    capabilities: Annotated[
        bool, typer.Option("--capabilities", help="Print the stable cmdhelp capability line.")
    ] = False,
) -> None:
    """Emit concise, Markdown, or structured documentation for MeshProbe commands."""

    if capabilities:
        typer.echo("cmdhelp/0.1: text, md, json, llm")
        return
    selected_path = tuple(command or ())
    root = ctx.find_root().command
    if format == "text":
        selected = _cmdhelp_lookup(root, selected_path)
        if selected is None:
            valid = ", ".join(_cmdhelp_paths(root))
            raise typer.BadParameter(
                f"Unknown command {' '.join(selected_path)!r}. Valid commands: {valid}",
                param_hint="COMMAND",
            )
        info_name = "meshprobe" if not selected_path else "meshprobe " + " ".join(selected_path)
        command_object = cast(Any, selected)
        help_context = type(ctx)(command_object, info_name=info_name)
        typer.echo(command_object.get_help(help_context), nl=False)
        return
    if format == "llm":
        format = "md"
    if format == "json":
        typer.echo(json.dumps(_cmdhelp_json(root, selected_path, depth, all_commands), indent=2))
        return
    typer.echo(_cmdhelp_markdown(root, selected_path, depth, all_commands), nl=False)


@app.command("schema")
def schema(
    ctx: typer.Context,
    command_name: Annotated[
        str | None,
        typer.Argument(
            help="Look up a single command's schema by name (e.g. view-orbit), instead of "
            "dumping the whole --kind commands/results schema.",
            metavar="[COMMAND]",
        ),
    ] = None,
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

    The results kind documents the full JSON shape each command returns — including
    otherwise-undocumented fields such as render luminance, projected component bounds,
    occlusion traces, and contact-sheet panel captions and callouts.

    Pass a command name (its CLI name, e.g. view-orbit, or its protocol op, e.g.
    view.orbit) to print just that command's schema fragment instead of the whole dump —
    e.g. meshprobe schema view-orbit instead of grepping meshprobe schema --kind commands
    for PerspectiveProjection. Combine with --kind results to see that command's result
    envelope instead of its input shape.
    """

    if command_name is not None:
        op = _resolve_command_op(command_name)
        if op is None:
            valid = ", ".join(_cli_command_names())
            raise typer.BadParameter(
                f"Unknown command {command_name!r}. Valid commands: {valid}. "
                "See meshprobe schema --kind commands for the full schema dump.",
                param_hint="COMMAND",
            )
        kind_source = ctx.get_parameter_source("kind")
        effective_kind = kind
        if kind_source is None or kind_source.name == "DEFAULT":
            effective_kind = SchemaKind.COMMANDS
        if effective_kind not in (SchemaKind.COMMANDS, SchemaKind.RESULTS):
            raise typer.BadParameter(
                f"--kind {effective_kind.value} has no single-command schema; "
                "use --kind commands (default) or --kind results with a command name.",
                param_hint="--kind",
            )
        single_payload: object = (
            command_result_json_schema_for(op)
            if effective_kind is SchemaKind.RESULTS
            else command_json_schema_for(op)
        )
        if _options(ctx).output == "yaml":
            _emit_yaml(single_payload)
            return
        _emit(single_payload)
        return

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
    corpus_version: Annotated[str, typer.Option("--version")] = "procedural-v7",
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
    corpus_version: Annotated[str, typer.Option("--corpus-version")] = "curated-tasks-v7",
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
    corpus_version: Annotated[str, typer.Option("--version")] = "qualification-v8",
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


_FOCUS_DIAGNOSTIC_HELP = (
    "Component ref, stable ID, exact name, exact path, or glob to report projected image "
    "bounds for (camera diagnostics only). This does NOT move or reframe the camera; use "
    "view-frame to frame a component."
)


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


def _emit_warnings(receipt: OperationReceipt) -> None:
    for warning in receipt.warnings:
        typer.echo(f"warning: {warning}", err=True)


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
        # Warnings go to stderr, so they still surface without corrupting the raw JSON on
        # stdout — a raw render-image must not silently drop its aspect-ratio warning.
        _emit_warnings(receipt)
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
    if receipt.op == "session.undo" and receipt.result_path:
        envelope = client.read_result(receipt)
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if isinstance(result, dict):
            fields.extend(
                (
                    f"undone={result.get('undone')}",
                    f"remaining={result.get('remaining')}",
                )
            )
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
    _emit_warnings(receipt)
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
        for receipt in receipts:
            _emit_warnings(receipt)
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


DEFAULT_RENDER_MAX_DIMENSION = 2576
MINIMUM_RENDER_DIMENSION = 64


def _aspect_preserving_render_dimensions(aspect_ratio: float) -> tuple[int, int]:
    """Fit an aspect ratio inside the default render budget without reframing."""

    if aspect_ratio >= 1:
        width = DEFAULT_RENDER_MAX_DIMENSION
        height = round(width / aspect_ratio)
    else:
        height = DEFAULT_RENDER_MAX_DIMENSION
        width = round(height * aspect_ratio)
    if min(width, height) >= MINIMUM_RENDER_DIMENSION:
        return width, height
    minimum_aspect = MINIMUM_RENDER_DIMENSION / DEFAULT_RENDER_MAX_DIMENSION
    maximum_aspect = 1 / minimum_aspect
    raise ValueError(
        f"aspect ratio {aspect_ratio:g} is outside the {minimum_aspect:.5g}.."
        f"{maximum_aspect:.5g} range supported by --render's "
        f"{DEFAULT_RENDER_MAX_DIMENSION}px budget; use render-image with explicit "
        "--width/--height for this extreme aspect ratio"
    )


def _execute_visual(
    ctx: typer.Context,
    command: Command,
    *,
    render: bool,
    blender: str | None = None,
) -> None:
    """Execute a visual mutation and optionally capture its resulting frame.

    The opt-in render is deliberately a second durable operation, so its result, artifact,
    and failure remain independently inspectable in the session journal. One CLI invocation
    still saves an agent from making a second tool call.
    """

    if not render:
        _execute(ctx, command, blender=blender)
        return

    options = _options(ctx)
    destination = (
        workspace_root(options.workspace)
        / "sessions"
        / options.session
        / "artifacts"
        / f"render-{uuid.uuid4().hex[:12]}.png"
    )
    client = _client(ctx, blender=blender)
    receipts: list[OperationReceipt] = []
    try:
        receipts.append(client.execute(options.session, command))
        aspect_ratio = client.framed_aspect_ratio(options.session)
        width, height = _aspect_preserving_render_dimensions(aspect_ratio)
        render_command = RenderImageCommand(
            request_id=_request_id("render-image"),
            op="render.image",
            output_path=str(destination.resolve()),
            width=width,
            height=height,
        )
        receipts.append(client.execute(options.session, render_command))
    except (OSError, RuntimeError, ValueError, ValidationError) as error:
        if not receipts:
            raise typer.BadParameter(str(error)) from error
        applied = receipts[0]
        raise typer.BadParameter(
            f"{applied.op} succeeded (result: {applied.result_path}); "
            f"requested render failed: {error}"
        ) from error
    _emit_receipts(ctx, client, receipts)


_RENDER_AFTER_HELP = "Also render the resulting state in this CLI call."


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
    aspect_ratio: Annotated[float | None, typer.Option("--aspect-ratio", min=0.01)] = None,
    unit_scale: Annotated[
        float,
        typer.Option(
            "--unit-scale",
            min=0.000001,
            help="Multiplier applied to the imported geometry before bounds/units checks run. "
            "Use 0.001 to correct a millimeter-authored asset a buggy exporter reported as "
            "meters, or 1000 for the opposite mistake.",
        ),
    ] = 1.0,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
) -> None:
    """Open a model in the selected durable session."""

    from meshprobe.protocol import SceneOpenCommand

    overrides = {} if aspect_ratio is None else {"aspect_ratio": aspect_ratio}
    _execute_visual(
        ctx,
        SceneOpenCommand(
            request_id=_request_id("open"),
            op="scene.open",
            source_path=str(source.expanduser().resolve(strict=True)),
            **overrides,
            unit_scale=unit_scale,
        ),
        render=render,
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
        typer.Option("--focus", help=_FOCUS_DIAGNOSTIC_HELP),
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
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
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
    _execute_visual(
        ctx,
        ViewSetCommand(
            request_id=_request_id("view-set"),
            op="view.set",
            camera=camera,
            focus_component_ids=_component_ids(ctx, focus or []) if focus else (),
            aspect_ratio=aspect_ratio,
        ),
        render=render,
    )


def _orbit_projection(
    projection_json: str | None,
    focal_length: float | None,
    ortho_scale: float | None,
) -> Projection | None:
    """Resolve view-orbit's projection flags to a concrete Projection, or None.

    None tells the controller/worker to keep the session's current projection
    (defaulting to a standard perspective on the first orbit of a session). See
    issue #96: the full --projection-json object is heavy for the common case of
    orbiting the existing camera, so --focal-length/--ortho-scale offer shorthands
    for the two projection modes and both stay mutually exclusive with each other
    and with --projection-json.
    """

    provided = [
        flag
        for flag, value in (
            ("--projection-json", projection_json),
            ("--focal-length", focal_length),
            ("--ortho-scale", ortho_scale),
        )
        if value is not None
    ]
    if len(provided) > 1:
        raise typer.BadParameter(f"provide at most one of {', '.join(provided)}")
    if projection_json is not None:
        from pydantic import TypeAdapter

        return TypeAdapter(Projection).validate_json(projection_json)
    if focal_length is not None:
        return PerspectiveProjection(focal_length_mm=focal_length)
    if ortho_scale is not None:
        return OrthographicProjection(scale_mm=ortho_scale)
    return None


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
        str | None,
        typer.Option(
            "--projection-json",
            help="Complete perspective or orthographic projection, including camera intrinsics "
            "(focal_length_mm, sensor_width_mm/sensor_height_mm, sensor_fit, depth_of_field). "
            "Omit to keep the session's current projection (or a standard 50mm perspective on "
            "the first view call); mutually exclusive with --focal-length/--ortho-scale.",
        ),
    ] = None,
    focal_length: Annotated[
        float | None,
        typer.Option(
            "--focal-length",
            min=0.000001,
            help="Perspective shorthand: lens focal length in millimeters, replacing the "
            "current projection with a perspective one at this focal length. Mutually "
            "exclusive with --projection-json/--ortho-scale.",
        ),
    ] = None,
    ortho_scale: Annotated[
        float | None,
        typer.Option(
            "--ortho-scale",
            min=0.000001,
            help="Orthographic shorthand: vertical view extent in millimeters, replacing the "
            "current projection with an orthographic one at this scale. Mutually exclusive "
            "with --projection-json/--focal-length.",
        ),
    ] = None,
    roll: Annotated[
        float,
        typer.Option("--roll", help="Signed right-hand-rule degrees around the forward axis."),
    ] = 0.0,
    focus: Annotated[
        list[str] | None,
        typer.Option("--focus", help=_FOCUS_DIAGNOSTIC_HELP),
    ] = None,
    aspect_ratio: Annotated[float | None, typer.Option("--aspect-ratio", min=0.01)] = None,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
) -> None:
    """Set an absolute orbit in right-handed, Z-up MeshProbe world coordinates.

    Angles are degrees, not relative deltas. Unlike the cumulative view-move and
    view-rotate, this command discards the current pose and replaces it wholesale, so
    it does not compose with a prior view call. To continue an orbit, read the current
    camera_orbit_angles (azimuth_degrees, elevation_degrees) from state.yml and pass
    them back here. Positive azimuth follows the right-hand rule around +Z. GLTF is
    right-handed and Y-up; scene.open reports the exact source_to_world mapping
    (GLTF +X -> world +X, +Y -> +Z, +Z -> -Y). The projection carries over from the
    session's current camera unless --projection-json, --focal-length, or --ortho-scale
    replaces it (a fresh session defaults to a standard 50mm perspective); use
    --projection-json for exact control over the full intrinsics (sensor size, sensor
    fit, depth of field). The result echoes the resolved camera (including the
    intrinsics above) and its diagnostics; see meshprobe schema --kind results.
    """

    projection = _orbit_projection(projection_json, focal_length, ortho_scale)
    command = ViewOrbitCommand(
        request_id=_request_id("view-orbit"),
        op="view.orbit",
        target_mm=target,
        azimuth_degrees=azimuth,
        elevation_degrees=elevation,
        roll_degrees=roll,
        distance_mm=distance,
        projection=projection,
        focus_component_ids=_component_ids(ctx, focus or []) if focus else (),
    )
    if aspect_ratio is not None:
        command = command.model_copy(update={"aspect_ratio": aspect_ratio})
    _execute_visual(
        ctx,
        command,
        render=render,
    )


@app.command("view-frame")
def view_frame(
    ctx: typer.Context,
    components: Annotated[
        list[str],
        typer.Argument(
            help="Component refs, stable IDs, exact names, or exact paths to frame tightly."
        ),
    ],
    azimuth: Annotated[
        float,
        typer.Option("--azimuth", help="Absolute degrees from +X toward +Y about world +Z."),
    ] = 45.0,
    elevation: Annotated[
        float,
        typer.Option("--elevation", help="Absolute degrees above the world XY plane toward +Z."),
    ] = 30.0,
    roll: Annotated[
        float,
        typer.Option("--roll", help="Signed right-hand-rule degrees around the forward axis."),
    ] = 0.0,
    margin: Annotated[
        float,
        typer.Option(
            "--margin",
            min=0.01,
            help="Padding factor around the bounds; 1.0 is a tight fit, higher zooms out.",
        ),
    ] = 1.25,
    projection_json: Annotated[
        str | None,
        typer.Option(
            "--projection-json",
            help="Perspective or orthographic projection; a standard 50mm perspective by default.",
        ),
    ] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
) -> None:
    """Frame the camera tightly on one or more components.

    Computes the target, distance, and projection that fit the selected components' world
    bounds from an absolute azimuth/elevation, then persists that camera like view-orbit.
    This is the auto-framing render-sheet applies internally, exposed as a one-shot camera
    move so an isolated component actually fills a following render-image.
    """

    if projection_json is not None:
        from pydantic import TypeAdapter

        projection: Projection = TypeAdapter(Projection).validate_json(projection_json)
    else:
        projection = PerspectiveProjection()
    _execute_visual(
        ctx,
        ViewFrameCommand(
            request_id=_request_id("view-frame"),
            op="view.frame",
            focus_component_ids=_component_ids(ctx, components),
            azimuth_degrees=azimuth,
            elevation_degrees=elevation,
            roll_degrees=roll,
            margin=margin,
            projection=projection,
            aspect_ratio=aspect_ratio,
        ),
        render=render,
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
    focus: Annotated[list[str] | None, typer.Option("--focus", help=_FOCUS_DIAGNOSTIC_HELP)] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
) -> None:
    """Move the current camera by world- and camera-frame deltas.

    Cumulative: deltas compose on top of the current pose, like view-rotate and unlike
    the absolute view-orbit (which replaces the pose wholesale).
    """

    _execute_visual(
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
        render=render,
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
    focus: Annotated[list[str] | None, typer.Option("--focus", help=_FOCUS_DIAGNOSTIC_HELP)] = None,
    aspect_ratio: Annotated[float, typer.Option("--aspect-ratio", min=0.01)] = 1.0,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
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
    _execute_visual(
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
        render=render,
    )


@app.command("illumination-set")
def illumination_set(
    ctx: typer.Context,
    preset: Annotated[
        str,
        typer.Argument(
            help=(
                "Preset: neutral_studio, high_key, raking_left, raking_right, backlit, "
                "flat_diagnostic, or custom."
            )
        ),
    ],
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
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
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

    _execute_visual(
        ctx,
        IlluminationSetCommand(
            request_id=_request_id("illumination"),
            op="illumination.set",
            illumination=illumination,
        ),
        render=render,
    )


@app.command("display")
def display_components(
    ctx: typer.Context,
    components: Annotated[list[str], typer.Argument()],
    mode: Annotated[DisplayMode, typer.Option("--mode")],
    operation: Annotated[IsolationOperation | None, typer.Option("--operation")] = None,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
) -> None:
    """Change visibility using refs, stable IDs, exact names, exact paths, or globs."""

    _execute_visual(
        ctx,
        ComponentDisplayCommand(
            request_id=_request_id("display"),
            op="component.display",
            component_ids=_component_ids(ctx, components),
            mode=mode,
            isolation_operation=operation,
        ),
        render=render,
    )


@app.command("mark")
def mark_components(
    ctx: typer.Context,
    components: Annotated[list[str], typer.Argument()],
    mode: Annotated[
        MarkMode,
        typer.Option(
            "--mode",
            help=(
                "unmarked restores source materials; selected applies cyan; highlighted "
                "applies deep pink; labeled applies gold and adds the component name."
            ),
        ),
    ],
    color: Annotated[
        str | None,
        typer.Option("--color", help="sRGB highlight color in #RRGGBB form."),
    ] = None,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
) -> None:
    """Mark one or more components using refs, IDs, names, paths, or globs.

    Modes: unmarked restores source materials; selected applies cyan; highlighted applies
    deep pink; labeled applies gold and adds the component name.

    --color overrides the mode's default color for selected, highlighted, and labeled;
    labeled still adds a text annotation. Pass --mode unmarked to restore source materials.
    """

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
    _execute_visual(ctx, command, render=render)


@app.command("render-image")
def render_image(
    ctx: typer.Context,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    width: Annotated[int | None, typer.Option("--width", min=64, max=16_384)] = None,
    height: Annotated[int | None, typer.Option("--height", min=64, max=16_384)] = None,
    samples: Annotated[int, typer.Option("--samples", min=1, max=4_096)] = 64,
    engine: Annotated[RenderEngine, typer.Option("--engine")] = RenderEngine.EEVEE,
    style: Annotated[
        RenderStyle,
        typer.Option(
            "--style",
            metavar="STYLE",
            help="See the style guide above.",
        ),
    ] = RenderStyle.SCREEN_EDGES,
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
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.001, max=86_400, help="Maximum render time in seconds."),
    ] = DEFAULT_WORKER_TIMEOUT_SECONDS,
    reference_image: Annotated[
        Path | None, typer.Option("--reference-image", exists=True, dir_okay=False)
    ] = None,
    comparison: Annotated[
        str | None, typer.Option("--comparison", help="Comparison mode; currently side-by-side.")
    ] = None,
    comparison_output: Annotated[
        Path | None, typer.Option("--comparison-output", dir_okay=False)
    ] = None,
) -> None:
    """Render the selected session to an image artifact.

    Choose a style:

    screen_edges (default): Fast GPU depth/normal edge pass for routine inspection.
    shaded_edges: Slower geometry-aware Freestyle lines for final confirmation,
    especially when separating same-color adjacent parts.
    shaded: Unmodified shaded image with no edge overlay.

    Freestyle is single-threaded and CPU-bound. Its cost increases with the number of
    visible components, not output resolution. Narrow the visible set or --edge-types
    to reduce final-confirmation time.

    The result carries a luminance exposure summary (median, clipped/crushed fractions) so
    a too-dark or blown-out frame can be detected without inspecting pixels. See
    meshprobe schema --kind results for the full render result shape.
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
    comparison_fields = (reference_image, comparison, comparison_output)
    if any(field is not None for field in comparison_fields) and not all(
        field is not None for field in comparison_fields
    ):
        raise typer.BadParameter(
            "--reference-image, --comparison, and --comparison-output must be supplied together"
        )
    if comparison not in {None, "side-by-side"}:
        raise typer.BadParameter("--comparison must be side-by-side")
    comparison_request = None
    if reference_image is not None and comparison_output is not None:
        comparison_request = RenderComparisonRequest(
            reference_image_path=str(reference_image.expanduser().resolve()),
            mode="side_by_side",
            output_path=str(comparison_output.expanduser().resolve()),
        )
    client = _client(ctx)
    if width is None and height is None:
        try:
            width, height = _aspect_preserving_render_dimensions(
                client.framed_aspect_ratio(options.session)
            )
        except (OSError, RuntimeError, ValueError) as error:
            typer.echo(f"Render failed: {error}", err=True)
            raise typer.Exit(1) from error
    else:
        width = width or DEFAULT_RENDER_MAX_DIMENSION
        height = height or DEFAULT_RENDER_MAX_DIMENSION
    try:
        receipt = client.execute(
            options.session,
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
                timeout_seconds=timeout,
                comparison=comparison_request,
            ),
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as error:
        typer.echo(f"Render failed: {error}", err=True)
        raise typer.Exit(1) from error
    _emit_receipt(ctx, client, receipt)


@app.command("render-sheet")
def render_sheet(
    ctx: typer.Context,
    focus: Annotated[
        list[str] | None,
        typer.Argument(
            help="Component refs, stable IDs, exact names, exact paths, or glob patterns."
        ),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    panel_width: Annotated[int, typer.Option("--panel-width", min=128)] = 1200,
    panel_height: Annotated[int, typer.Option("--panel-height", min=128)] = 1200,
    samples: Annotated[int, typer.Option("--samples", min=1)] = 32,
    engine: Annotated[RenderEngine, typer.Option("--engine")] = RenderEngine.EEVEE,
    graphics_policy: Annotated[
        GraphicsPolicy, typer.Option("--graphics-policy")
    ] = GraphicsPolicy.SOFTWARE_ALLOWED,
    azimuth: Annotated[
        str | None, typer.Option("--azimuth", help="Comma-separated absolute azimuth degrees.")
    ] = None,
    elevation: Annotated[
        str | None, typer.Option("--elevation", help="Comma-separated absolute elevation degrees.")
    ] = None,
    roll: Annotated[
        str, typer.Option("--roll", help="Comma-separated absolute roll degrees.")
    ] = "0",
    target: Annotated[
        str, typer.Option("--target", help="Sweep target: scene or focus.")
    ] = "scene",
) -> None:
    """Render a focused 3x3 sheet with automatic occlusion analysis.

    The first row shows the full assembly, the highlighted focus in context, then the same
    focused view after automatically removing blockers. The other six panels isolate the
    focus from +X, -X, +Y, -Y, +Z, and -Z. Deep pink marks the focus, not the occluders.

    Multiple component arguments or a glob form one combined focus. For example,
    meshprobe render-sheet '**/*bearing*' --output bearings.png analyzes every matching
    bearing together. Each panel result carries its full caption, numbered callouts,
    camera, and render metadata; the sheet result carries the occlusion-removal trace. Use
    meshprobe --raw render-sheet ... to print that result directly, or
    meshprobe schema render-sheet --kind results for its schema.
    """

    options = _options(ctx)
    destination = output or (
        workspace_root(options.workspace)
        / "sessions"
        / options.session
        / "artifacts"
        / f"sheet-{uuid.uuid4().hex[:12]}.png"
    )
    sweep_requested = azimuth is not None or elevation is not None
    if sweep_requested and (azimuth is None or elevation is None):
        raise typer.BadParameter("--azimuth and --elevation are both required for an orbit sweep")
    if target not in {"scene", "focus"}:
        raise typer.BadParameter("--target must be scene or focus")
    recipe: Literal["focused_3x3", "custom_3x3", "orbit_sweep"] = (
        "orbit_sweep" if sweep_requested else "focused_3x3"
    )
    sweep: OrbitSweep | None = None
    if sweep_requested:
        try:
            assert azimuth is not None
            assert elevation is not None

            sweep = OrbitSweep(
                azimuth_degrees=tuple(float(value.strip()) for value in azimuth.split(",")),
                elevation_degrees=tuple(float(value.strip()) for value in elevation.split(",")),
                roll_degrees=tuple(float(value.strip()) for value in roll.split(",")),
                target=cast(Literal["scene", "focus"], target),
            )
        except (ValueError, ValidationError) as error:
            raise typer.BadParameter(str(error)) from error
    _execute(
        ctx,
        RenderContactSheetCommand(
            request_id=_request_id("render-sheet"),
            op="render.contact_sheet",
            output_path=str(destination.expanduser().resolve()),
            focus_component_ids=_component_ids(ctx, focus or []),
            recipe=recipe,
            orbit_sweep=sweep,
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

    The result's camera_diagnostics.projected_bounds reports each focus component's
    normalized frame coverage, and the camera block exposes the full camera intrinsics
    (focal length, sensor size and fit). See meshprobe schema --kind results for the full
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
def reset(
    ctx: typer.Context,
    aspect_ratio: Annotated[float | None, typer.Option("--aspect-ratio", min=0.01)] = None,
    render: Annotated[bool, typer.Option("--render", help=_RENDER_AFTER_HELP)] = False,
) -> None:
    """Reset visual state to the imported scene defaults."""

    overrides = {} if aspect_ratio is None else {"aspect_ratio": aspect_ratio}
    _execute_visual(
        ctx,
        SessionResetCommand(request_id=_request_id("reset"), op="session.reset", **overrides),
        render=render,
    )


@app.command("undo")
def undo(
    ctx: typer.Context,
    count: Annotated[int, typer.Argument(min=1)] = 1,
) -> None:
    """Undo COUNT state-changing commands (default: 1).

    History survives process restarts. Rendering and other read-only commands do not count.
    Reset starts a new history boundary, so undo never crosses it.
    """

    _execute(
        ctx,
        SessionUndoCommand(request_id=_request_id("undo"), op="session.undo", count=count),
    )


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
    """Delete the stopped workspace's .meshprobe state."""

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
