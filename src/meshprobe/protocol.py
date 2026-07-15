"""Line-oriented command contract shared by the CLI and future Blender worker."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from meshprobe.models import Camera, DisplayMode, Illumination, MarkMode, Projection, Vec3
from meshprobe.selectors import ComponentSelector


class CommandModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str


class SceneOpenCommand(CommandModel):
    op: Literal["scene.open"]
    source_path: str


class SceneDescribeCommand(CommandModel):
    op: Literal["scene.describe"]


class ComponentFindCommand(CommandModel):
    op: Literal["component.find"]
    selector: ComponentSelector


class ComponentInspectCommand(CommandModel):
    op: Literal["component.inspect"]
    component_id: str


class ViewSetCommand(CommandModel):
    op: Literal["view.set"]
    camera: Camera


class ViewOrbitCommand(CommandModel):
    op: Literal["view.orbit"]
    target_mm: Vec3
    azimuth_degrees: float
    elevation_degrees: float
    roll_degrees: float = 0
    distance_mm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    projection: Projection


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


class RenderImageCommand(CommandModel):
    op: Literal["render.image"]
    output_path: str
    width: Annotated[int, Field(ge=64, le=16_384)] = 1024
    height: Annotated[int, Field(ge=64, le=16_384)] = 1024


class RenderContactSheetCommand(CommandModel):
    op: Literal["render.contact_sheet"]
    output_path: str
    recipe: str
    focus_component_ids: tuple[str, ...] = Field(min_length=1)


class SessionResetCommand(CommandModel):
    op: Literal["session.reset"]


Command: TypeAlias = Annotated[
    SceneOpenCommand
    | SceneDescribeCommand
    | ComponentFindCommand
    | ComponentInspectCommand
    | ViewSetCommand
    | ViewOrbitCommand
    | IlluminationSetCommand
    | ComponentDisplayCommand
    | ComponentMarkCommand
    | RenderImageCommand
    | RenderContactSheetCommand
    | SessionResetCommand,
    Field(discriminator="op"),
]

COMMAND_ADAPTER = TypeAdapter(Command)


def parse_command_json(payload: str) -> Command:
    return COMMAND_ADAPTER.validate_json(payload)


def command_json_schema() -> dict[str, object]:
    return COMMAND_ADAPTER.json_schema()
