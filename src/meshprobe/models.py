"""Typed, renderer-independent contracts for MeshProbe."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=256)]
Vec3: TypeAlias = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Quaternion: TypeAlias = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
Matrix4x4: TypeAlias = tuple[
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
]


class ContractModel(BaseModel):
    """Base contract that rejects misspelled or undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Bounds(ContractModel):
    minimum_mm: Vec3
    maximum_mm: Vec3

    @model_validator(mode="after")
    def validate_extents(self) -> Self:
        if any(low > high for low, high in zip(self.minimum_mm, self.maximum_mm, strict=True)):
            raise ValueError("minimum_mm must not exceed maximum_mm")
        return self


class Pose(ContractModel):
    position_mm: Vec3
    orientation_xyzw: Quaternion

    @field_validator("orientation_xyzw")
    @classmethod
    def normalize_quaternion(cls, value: Quaternion) -> Quaternion:
        norm = math.sqrt(sum(component * component for component in value))
        if norm <= 1e-12:
            raise ValueError("orientation_xyzw must have non-zero magnitude")
        return tuple(component / norm for component in value)  # type: ignore[return-value]


class SensorFit(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class PerspectiveProjection(ContractModel):
    mode: Literal["perspective"] = "perspective"
    focal_length_mm: PositiveFiniteFloat = 50.0
    sensor_width_mm: PositiveFiniteFloat = 36.0
    sensor_fit: SensorFit = SensorFit.HORIZONTAL
    near_clip_mm: PositiveFiniteFloat = 0.5
    far_clip_mm: PositiveFiniteFloat = 100_000.0

    @model_validator(mode="after")
    def validate_clip_range(self) -> Self:
        if self.far_clip_mm <= self.near_clip_mm:
            raise ValueError("far_clip_mm must be greater than near_clip_mm")
        return self

    @property
    def horizontal_fov_degrees(self) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_width_mm / (2.0 * self.focal_length_mm)))

    def vertical_fov_degrees(self, aspect_ratio: PositiveFiniteFloat) -> float:
        sensor_height = self.sensor_width_mm / aspect_ratio
        return math.degrees(2.0 * math.atan(sensor_height / (2.0 * self.focal_length_mm)))


class OrthographicProjection(ContractModel):
    mode: Literal["orthographic"] = "orthographic"
    scale_mm: PositiveFiniteFloat
    near_clip_mm: PositiveFiniteFloat = 0.5
    far_clip_mm: PositiveFiniteFloat = 100_000.0

    @model_validator(mode="after")
    def validate_clip_range(self) -> Self:
        if self.far_clip_mm <= self.near_clip_mm:
            raise ValueError("far_clip_mm must be greater than near_clip_mm")
        return self


Projection: TypeAlias = Annotated[
    PerspectiveProjection | OrthographicProjection,
    Field(discriminator="mode"),
]


class Camera(ContractModel):
    pose: Pose
    projection: Projection


class LightType(StrEnum):
    AREA = "area"
    POINT = "point"
    SPOT = "spot"
    SUN = "sun"


class LightColor(ContractModel):
    linear_rgb: (
        tuple[NonNegativeFiniteFloat, NonNegativeFiniteFloat, NonNegativeFiniteFloat] | None
    ) = None
    color_temperature_k: (
        Annotated[float, Field(ge=1_000, le=40_000, allow_inf_nan=False)] | None
    ) = None

    @model_validator(mode="after")
    def validate_color_source(self) -> Self:
        declared = int(self.linear_rgb is not None) + int(self.color_temperature_k is not None)
        if declared != 1:
            raise ValueError("declare exactly one of linear_rgb or color_temperature_k")
        return self


class AreaLight(LightColor):
    id: Identifier
    type: Literal["area"] = "area"
    position_mm: Vec3
    orientation_xyzw: Quaternion
    power_w: PositiveFiniteFloat
    size_mm: PositiveFiniteFloat


class PointLight(LightColor):
    id: Identifier
    type: Literal["point"] = "point"
    position_mm: Vec3
    power_w: PositiveFiniteFloat


class SpotLight(LightColor):
    id: Identifier
    type: Literal["spot"] = "spot"
    position_mm: Vec3
    orientation_xyzw: Quaternion
    power_w: PositiveFiniteFloat
    spot_size_degrees: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)]
    blend: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.15


class SunLight(LightColor):
    id: Identifier
    type: Literal["sun"] = "sun"
    orientation_xyzw: Quaternion
    strength: PositiveFiniteFloat
    angle_degrees: Annotated[float, Field(ge=0, le=180, allow_inf_nan=False)] = 0.526


Light: TypeAlias = Annotated[
    AreaLight | PointLight | SpotLight | SunLight, Field(discriminator="type")
]


class IlluminationPreset(StrEnum):
    NEUTRAL_STUDIO = "neutral_studio"
    HIGH_KEY = "high_key"
    RAKING_LEFT = "raking_left"
    RAKING_RIGHT = "raking_right"
    BACKLIT = "backlit"
    FLAT_DIAGNOSTIC = "flat_diagnostic"


class PresetIllumination(ContractModel):
    preset: IlluminationPreset


class CustomIllumination(ContractModel):
    preset: Literal["custom"] = "custom"
    background_rgb: tuple[NonNegativeFiniteFloat, NonNegativeFiniteFloat, NonNegativeFiniteFloat]
    ambient_strength: NonNegativeFiniteFloat
    lights: tuple[Light, ...] = Field(min_length=1, max_length=32)

    @field_validator("lights")
    @classmethod
    def unique_light_ids(cls, lights: tuple[Light, ...]) -> tuple[Light, ...]:
        ids = [light.id for light in lights]
        if len(ids) != len(set(ids)):
            raise ValueError("light ids must be unique")
        return lights


Illumination: TypeAlias = PresetIllumination | CustomIllumination


class DisplayMode(StrEnum):
    SHOWN = "shown"
    HIDDEN = "hidden"
    ISOLATED = "isolated"
    GHOSTED = "ghosted"


class MarkMode(StrEnum):
    UNMARKED = "unmarked"
    SELECTED = "selected"
    HIGHLIGHTED = "highlighted"
    LABELED = "labeled"


class MaterialSummary(ContractModel):
    names: tuple[str, ...] = ()
    missing_textures: tuple[str, ...] = ()


class Component(ContractModel):
    id: Identifier
    path: Identifier
    display_name: Identifier
    source_name: Identifier
    parent_id: Identifier | None = None
    child_ids: tuple[Identifier, ...] = ()
    mesh_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    world_transform: Matrix4x4
    local_bounds: Bounds
    world_bounds: Bounds
    materials: MaterialSummary = MaterialSummary()


class CapabilityWarning(ContractModel):
    code: Identifier
    message: Annotated[str, StringConstraints(min_length=1, max_length=2_000)]
    component_ids: tuple[Identifier, ...] = ()


class SceneCapabilities(ContractModel):
    hierarchy: Literal["preserved", "flattened"]
    component_names: Literal["source", "generated"]
    materials: Literal["preserved", "partial", "absent"]
    textures: Literal["preserved", "partial", "absent"]


class SceneManifest(ContractModel):
    schema_version: Literal[1] = 1
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    source_format: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    units: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    root_bounds: Bounds
    components: tuple[Component, ...]
    imported_camera: Camera
    imported_illumination: Illumination
    capabilities: SceneCapabilities
    warnings: tuple[CapabilityWarning, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        by_id = {component.id: component for component in self.components}
        if len(by_id) != len(self.components):
            raise ValueError("component ids must be unique")
        paths = {component.path for component in self.components}
        if len(paths) != len(self.components):
            raise ValueError("component paths must be unique")
        for component in self.components:
            if component.parent_id is not None and component.parent_id not in by_id:
                raise ValueError(f"unknown parent id: {component.parent_id}")
            unknown_children = set(component.child_ids) - by_id.keys()
            if unknown_children:
                raise ValueError(f"unknown child ids: {sorted(unknown_children)}")
            for child_id in component.child_ids:
                if by_id[child_id].parent_id != component.id:
                    raise ValueError(
                        f"child {child_id} does not point back to parent {component.id}"
                    )
        return self
