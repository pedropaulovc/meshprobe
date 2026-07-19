"""Line-oriented command contract shared by the CLI and future Blender worker."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from meshprobe.models import (
    Camera,
    CameraMotionResult,
    CameraViewResult,
    Component,
    ComponentVisualStateResult,
    ContactSheetManifest,
    ContactSheetPanelSpec,
    CoordinateFrame,
    DisplayMode,
    FiniteFloat,
    GraphicsPolicy,
    Illumination,
    IlluminationResult,
    IsolationOperation,
    MarkMode,
    OcclusionQueryResult,
    OrbitSweep,
    OrthonormalBasis,
    PerspectiveProjection,
    PositiveFiniteFloat,
    Projection,
    RenderEngine,
    RenderManifest,
    RenderStyle,
    SceneManifest,
    SessionResetResult,
    SessionSnapshotResult,
    SessionUndoResult,
    ShadedEdgesStyle,
    SrgbHexColor,
    Vec3,
)
from meshprobe.selectors import ComponentSelector


class CommandEffect(StrEnum):
    """How a command participates in durable session history."""

    UNDECLARED = "undeclared"
    OPEN = "open"
    READ_ONLY = "read_only"
    STATE_MUTATION = "state_mutation"
    RESET = "reset"
    HISTORY = "history"


class CommandModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    effect: ClassVar[CommandEffect] = CommandEffect.UNDECLARED

    request_id: str


class SceneOpenCommand(CommandModel):
    effect = CommandEffect.OPEN
    op: Literal["scene.open"]
    source_path: str
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0
    unit_scale: PositiveFiniteFloat = Field(
        default=1.0,
        description="Multiplier applied to the imported geometry at import time, before any "
        "bounds/units checks run. Use 0.001 to correct a millimeter-authored asset that a "
        "buggy exporter reported as meters, or 1000 for the opposite mistake.",
    )


class SessionSnapshotCommand(CommandModel):
    effect = CommandEffect.READ_ONLY
    op: Literal["session.snapshot"]


class ComponentFindCommand(CommandModel):
    effect = CommandEffect.READ_ONLY
    op: Literal["component.find"]
    selector: ComponentSelector


class ComponentInspectCommand(CommandModel):
    effect = CommandEffect.READ_ONLY
    op: Literal["component.inspect"]
    component_id: str


class ComponentOcclusionCommand(CommandModel):
    effect = CommandEffect.READ_ONLY
    op: Literal["component.occlusion"]
    component_ids: tuple[str, ...] = Field(min_length=1)
    max_samples_per_component: Annotated[int, Field(ge=1, le=4_096)] = 128


class ViewSetCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["view.set"]
    camera: Camera
    focus_component_ids: tuple[str, ...] = ()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class ViewOrbitCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["view.orbit"]
    target_mm: Vec3
    azimuth_degrees: FiniteFloat
    elevation_degrees: FiniteFloat
    roll_degrees: FiniteFloat = 0
    distance_mm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    projection: Projection | None = None
    focus_component_ids: tuple[str, ...] = ()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class ViewFrameCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["view.frame"]
    focus_component_ids: tuple[str, ...] = Field(min_length=1)
    azimuth_degrees: FiniteFloat = 45.0
    elevation_degrees: FiniteFloat = 30.0
    roll_degrees: FiniteFloat = 0.0
    margin: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)] = 1.25
    projection: Projection = PerspectiveProjection()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class ViewMoveCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["view.move"]
    world_delta_mm: Vec3 = (0.0, 0.0, 0.0)
    camera_delta_mm: Vec3 = (0.0, 0.0, 0.0)
    focus_component_ids: tuple[str, ...] = ()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0

    @model_validator(mode="after")
    def require_movement(self) -> Self:
        if not any((*self.world_delta_mm, *self.camera_delta_mm)):
            raise ValueError("view.move requires at least one non-zero delta")
        return self


class ViewRotateCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["view.rotate"]
    target_mm: Vec3
    axis: Literal["x", "y", "z"]
    degrees: FiniteFloat
    frame: Literal[CoordinateFrame.SOURCE, CoordinateFrame.WORLD, CoordinateFrame.CAMERA] = (
        CoordinateFrame.SOURCE
    )
    basis: OrthonormalBasis = OrthonormalBasis()
    projection: Projection | None = None
    focus_component_ids: tuple[str, ...] = ()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class IlluminationSetCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["illumination.set"]
    illumination: Illumination


class ComponentDisplayCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["component.display"]
    component_ids: tuple[str, ...] = Field(min_length=1)
    mode: DisplayMode
    isolation_operation: IsolationOperation | None = None

    @model_validator(mode="after")
    def reject_isolation_operation_without_isolation(self) -> Self:
        if self.isolation_operation is not None and self.mode is not DisplayMode.ISOLATED:
            raise ValueError("operation requires mode=isolated")
        return self


class ComponentMarkCommand(CommandModel):
    effect = CommandEffect.STATE_MUTATION
    op: Literal["component.mark"]
    component_ids: tuple[str, ...] = Field(min_length=1)
    mode: MarkMode
    color: SrgbHexColor | None = None

    @model_validator(mode="after")
    def reject_color_without_mark(self) -> Self:
        if self.mode is MarkMode.UNMARKED and self.color is not None:
            raise ValueError("color requires a visible mark mode")
        return self


class RenderComparisonRequest(BaseModel):
    """Local reference comparison requested after the ordinary CAD render."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_image_path: str
    mode: Literal["side_by_side"]
    output_path: str


class RenderImageCommand(CommandModel):
    effect = CommandEffect.READ_ONLY
    op: Literal["render.image"]
    output_path: str
    width: Annotated[int, Field(ge=64, le=16_384)] = 2576
    height: Annotated[int, Field(ge=64, le=16_384)] = 2576
    samples: Annotated[int, Field(ge=1, le=4_096)] = 64
    engine: RenderEngine = RenderEngine.EEVEE
    style: Annotated[
        RenderStyle,
        Field(
            description="Rendering policy: screen_edges is the fast default for inspection; "
            "use shaded_edges for slower Freestyle final confirmation, or shaded for no overlay."
        ),
    ] = RenderStyle.SCREEN_EDGES
    shaded_edges: ShadedEdgesStyle = ShadedEdgesStyle()
    graphics_policy: GraphicsPolicy = GraphicsPolicy.SOFTWARE_ALLOWED
    comparison: RenderComparisonRequest | None = None


class RenderContactSheetCommand(CommandModel):
    effect = CommandEffect.READ_ONLY
    op: Literal["render.contact_sheet"]
    output_path: str
    recipe: Literal["focused_3x3", "custom_3x3", "orbit_sweep"] = "focused_3x3"
    focus_component_ids: tuple[str, ...] = ()
    panels: tuple[ContactSheetPanelSpec, ...] = Field(default=(), max_length=9)
    orbit_sweep: OrbitSweep | None = None
    panel_width: Annotated[int, Field(ge=128, le=4_096)] = 1200
    panel_height: Annotated[int, Field(ge=128, le=4_096)] = 1200
    samples: Annotated[int, Field(ge=1, le=4_096)] = 32
    engine: RenderEngine = RenderEngine.EEVEE
    graphics_policy: GraphicsPolicy = GraphicsPolicy.SOFTWARE_ALLOWED
    visibility_threshold: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.65
    occluder_budget: Annotated[int, Field(ge=0, le=32)] = 3

    @model_validator(mode="after")
    def validate_recipe(self) -> Self:
        if self.recipe == "focused_3x3":
            if not self.focus_component_ids:
                raise ValueError("focused_3x3 requires at least one focus component")
            if self.panels:
                raise ValueError("focused_3x3 does not accept custom panels")
            if self.orbit_sweep is not None:
                raise ValueError("focused_3x3 does not accept orbit_sweep")
            return self
        if self.recipe == "orbit_sweep":
            if self.panels:
                raise ValueError("orbit_sweep does not accept custom panels")
            if self.orbit_sweep is None:
                raise ValueError("orbit_sweep requires sweep axes")
            if self.orbit_sweep.target == "focus" and not self.focus_component_ids:
                raise ValueError("focus-targeted orbit_sweep requires at least one focus component")
            return self
        if not self.focus_component_ids:
            raise ValueError("custom_3x3 requires at least one focus component")
        if self.orbit_sweep is not None:
            raise ValueError("custom_3x3 does not accept orbit_sweep")
        if len(self.panels) != 9:
            raise ValueError("custom_3x3 requires exactly nine panels")
        fixed_pose_panels = [
            panel for panel in self.panels if panel.experiment == "fixed_pose_focal_study"
        ]
        if len(fixed_pose_panels) > 1:
            poses = {panel.camera.pose for panel in fixed_pose_panels if panel.camera is not None}
            if len(poses) != 1:
                raise ValueError("fixed_pose_focal_study panels must share one camera pose")
        return self


class SessionResetCommand(CommandModel):
    effect = CommandEffect.RESET
    op: Literal["session.reset"]
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class SessionUndoCommand(CommandModel):
    effect = CommandEffect.HISTORY

    op: Literal["session.undo"]
    count: Annotated[int, Field(ge=1)] = 1


type Command = Annotated[
    SceneOpenCommand
    | SessionSnapshotCommand
    | ComponentFindCommand
    | ComponentInspectCommand
    | ComponentOcclusionCommand
    | ViewSetCommand
    | ViewOrbitCommand
    | ViewFrameCommand
    | ViewMoveCommand
    | ViewRotateCommand
    | IlluminationSetCommand
    | ComponentDisplayCommand
    | ComponentMarkCommand
    | RenderImageCommand
    | RenderContactSheetCommand
    | SessionResetCommand
    | SessionUndoCommand,
    Field(discriminator="op"),
]


def command_payload(command: Command, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Serialize commands while preserving omitted optional wire fields."""
    payload = command.model_dump(mode="json", exclude=exclude, exclude_none=True)
    if isinstance(command, (SceneOpenCommand, SessionResetCommand, ViewOrbitCommand)) and (
        "aspect_ratio" not in command.model_fields_set
    ):
        payload.pop("aspect_ratio")
    return payload


COMMAND_ADAPTER: TypeAdapter[Command] = TypeAdapter(Command)


def _command_model_registry() -> dict[str, type[CommandModel]]:
    """Map each command's ``op`` discriminator value to its model class.

    Derived from the ``Command`` discriminated union itself (via the same
    ``op: Literal[...]`` field the union discriminates on), so the registry
    can never drift out of sync with the actual set of commands.
    """
    union = get_args(Command.__value__)[0]
    registry: dict[str, type[CommandModel]] = {}
    for member in get_args(union):
        (op_value,) = get_args(member.model_fields["op"].annotation)
        registry[op_value] = member
    return registry


COMMAND_MODELS: dict[str, type[CommandModel]] = _command_model_registry()


def parse_command_json(payload: str) -> Command:
    return COMMAND_ADAPTER.validate_json(payload)


def command_json_schema() -> dict[str, object]:
    return COMMAND_ADAPTER.json_schema()


def command_json_schema_for(op: str) -> dict[str, object]:
    """Describe a single command's input schema, keyed by its ``op`` value.

    Raises ``KeyError`` (with the unknown ``op``) if ``op`` is not a known
    command; callers typically want :data:`COMMAND_MODELS` for the full set
    of valid values first.
    """
    return TypeAdapter(COMMAND_MODELS[op]).json_schema()


_RESULT_MODELS: dict[str, object] = {
    "scene.open": SceneManifest,
    "session.snapshot": SessionSnapshotResult,
    "component.find": list[Component],
    "component.inspect": Component,
    "component.occlusion": OcclusionQueryResult,
    "view.set": CameraViewResult,
    "view.orbit": CameraViewResult,
    "view.frame": CameraViewResult,
    "view.move": CameraMotionResult,
    "view.rotate": CameraMotionResult,
    "illumination.set": IlluminationResult,
    "component.display": ComponentVisualStateResult,
    "component.mark": ComponentVisualStateResult,
    "render.image": RenderManifest,
    "render.contact_sheet": ContactSheetManifest,
    "session.reset": SessionResetResult,
    "session.undo": SessionUndoResult,
}


def _result_envelope_schema(op: str, model: object) -> dict[str, object]:
    result_schema = TypeAdapter(model).json_schema()
    # Hoist the result's shared definitions to the envelope root so its
    # "#/$defs/..." references resolve against the standalone per-op schema.
    definitions = result_schema.pop("$defs", None)
    envelope: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["request_id", "op", "result"],
        "properties": {
            "request_id": {"type": "string"},
            "op": {"const": op},
            "result": result_schema,
        },
    }
    if definitions:
        envelope["$defs"] = definitions
    return envelope


def command_result_json_schema() -> dict[str, object]:
    """Describe the full result envelope each command op returns, keyed by op.

    Each entry is the ``CommandResponse`` envelope a caller reads back (from a
    ``results/*.json`` file or an adapter response): ``request_id``, ``op`` pinned to the
    command, and ``result`` expanded to that op's concrete result schema.
    """
    return {op: _result_envelope_schema(op, model) for op, model in _RESULT_MODELS.items()}


def command_result_json_schema_for(op: str) -> dict[str, object]:
    """Describe a single command's result envelope, keyed by its ``op`` value.

    Raises ``KeyError`` (with the unknown ``op``) if ``op`` is not a known
    command.
    """
    return _result_envelope_schema(op, _RESULT_MODELS[op])
