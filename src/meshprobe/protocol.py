"""Line-oriented command contract shared by the CLI and future Blender worker."""

from __future__ import annotations

from typing import Annotated, Literal, Self

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
    MarkMode,
    OcclusionQueryResult,
    OrthonormalBasis,
    PerspectiveProjection,
    Projection,
    RenderEngine,
    RenderManifest,
    RenderStyle,
    SceneManifest,
    SessionResetResult,
    SessionSnapshotResult,
    ShadedEdgesStyle,
    SrgbHexColor,
    Vec3,
)
from meshprobe.selectors import ComponentSelector


class CommandModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str


class SceneOpenCommand(CommandModel):
    op: Literal["scene.open"]
    source_path: str


class SessionSnapshotCommand(CommandModel):
    op: Literal["session.snapshot"]


class ComponentFindCommand(CommandModel):
    op: Literal["component.find"]
    selector: ComponentSelector


class ComponentInspectCommand(CommandModel):
    op: Literal["component.inspect"]
    component_id: str


class ComponentOcclusionCommand(CommandModel):
    op: Literal["component.occlusion"]
    component_ids: tuple[str, ...] = Field(min_length=1)
    max_samples_per_component: Annotated[int, Field(ge=1, le=4_096)] = 128


class ViewSetCommand(CommandModel):
    op: Literal["view.set"]
    camera: Camera
    focus_component_ids: tuple[str, ...] = ()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class ViewOrbitCommand(CommandModel):
    op: Literal["view.orbit"]
    target_mm: Vec3
    azimuth_degrees: FiniteFloat
    elevation_degrees: FiniteFloat
    roll_degrees: FiniteFloat = 0
    distance_mm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    projection: Projection
    focus_component_ids: tuple[str, ...] = ()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class ViewFrameCommand(CommandModel):
    op: Literal["view.frame"]
    focus_component_ids: tuple[str, ...] = Field(min_length=1)
    azimuth_degrees: FiniteFloat = 45.0
    elevation_degrees: FiniteFloat = 30.0
    roll_degrees: FiniteFloat = 0.0
    margin: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)] = 1.25
    projection: Projection = PerspectiveProjection()
    aspect_ratio: Annotated[float, Field(ge=0.01, le=100, allow_inf_nan=False)] = 1.0


class ViewMoveCommand(CommandModel):
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
    op: Literal["illumination.set"]
    illumination: Illumination


class ComponentDisplayCommand(CommandModel):
    op: Literal["component.display"]
    component_ids: tuple[str, ...] = Field(min_length=1)
    mode: DisplayMode


class ComponentMarkCommand(CommandModel):
    op: Literal["component.mark"]
    component_ids: tuple[str, ...] = Field(min_length=1)
    mode: MarkMode
    color: SrgbHexColor | None = None

    @model_validator(mode="after")
    def reject_color_without_mark(self) -> Self:
        if self.mode is MarkMode.UNMARKED and self.color is not None:
            raise ValueError("color requires a visible mark mode")
        return self


class RenderImageCommand(CommandModel):
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


class RenderContactSheetCommand(CommandModel):
    op: Literal["render.contact_sheet"]
    output_path: str
    recipe: Literal["focused_3x3", "custom_3x3"] = "focused_3x3"
    focus_component_ids: tuple[str, ...] = Field(min_length=1)
    panels: tuple[ContactSheetPanelSpec, ...] = Field(default=(), max_length=9)
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
            if self.panels:
                raise ValueError("focused_3x3 does not accept custom panels")
            return self
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
    op: Literal["session.reset"]


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
    | SessionResetCommand,
    Field(discriminator="op"),
]

COMMAND_ADAPTER: TypeAdapter[Command] = TypeAdapter(Command)


def parse_command_json(payload: str) -> Command:
    return COMMAND_ADAPTER.validate_json(payload)


def command_json_schema() -> dict[str, object]:
    return COMMAND_ADAPTER.json_schema()


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
}


def command_result_json_schema() -> dict[str, object]:
    """Describe the full result envelope each command op returns, keyed by op.

    Each entry is the ``CommandResponse`` envelope a caller reads back (from a
    ``results/*.json`` file or an adapter response): ``request_id``, ``op`` pinned to the
    command, and ``result`` expanded to that op's concrete result schema.
    """
    schemas: dict[str, object] = {}
    for op, model in _RESULT_MODELS.items():
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
        schemas[op] = envelope
    return schemas
