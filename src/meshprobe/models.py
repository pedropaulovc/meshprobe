"""Typed, renderer-independent contracts for MeshProbe."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from meshprobe.identity import stable_component_id

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=256)]
type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
type Quaternion = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]


def normalize_quaternion(value: Quaternion) -> Quaternion:
    scale = max(abs(component) for component in value)
    if scale <= 1e-12:
        raise ValueError("quaternion must have non-zero magnitude")
    scaled = tuple(component / scale for component in value)
    norm = math.sqrt(sum(component * component for component in scaled))
    return cast(Quaternion, tuple(component / scale / norm for component in value))


type UnitQuaternion = Annotated[Quaternion, AfterValidator(normalize_quaternion)]
type Matrix4x4 = tuple[
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
    orientation_xyzw: UnitQuaternion


class SensorFit(StrEnum):
    AUTO = "auto"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class PerspectiveProjection(ContractModel):
    mode: Literal["perspective"] = "perspective"
    focal_length_mm: PositiveFiniteFloat = 50.0
    sensor_width_mm: PositiveFiniteFloat = 36.0
    sensor_height_mm: PositiveFiniteFloat = 24.0
    sensor_fit: SensorFit = SensorFit.AUTO
    near_clip_mm: PositiveFiniteFloat = 0.5
    far_clip_mm: PositiveFiniteFloat = 100_000.0

    @model_validator(mode="after")
    def validate_clip_range(self) -> Self:
        if self.far_clip_mm <= self.near_clip_mm:
            raise ValueError("far_clip_mm must be greater than near_clip_mm")
        return self

    def effective_sensor_fit(self, aspect_ratio: PositiveFiniteFloat) -> SensorFit:
        if self.sensor_fit is not SensorFit.AUTO:
            return self.sensor_fit
        return SensorFit.HORIZONTAL if aspect_ratio >= 1 else SensorFit.VERTICAL

    def horizontal_fov_degrees(self, aspect_ratio: PositiveFiniteFloat) -> float:
        sensor_width = self.sensor_width_mm
        if self.effective_sensor_fit(aspect_ratio) is SensorFit.VERTICAL:
            sensor_width = self.sensor_height_mm * aspect_ratio
        return math.degrees(2.0 * math.atan(sensor_width / (2.0 * self.focal_length_mm)))

    def vertical_fov_degrees(self, aspect_ratio: PositiveFiniteFloat) -> float:
        sensor_height = self.sensor_height_mm
        if self.effective_sensor_fit(aspect_ratio) is SensorFit.HORIZONTAL:
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


type Projection = Annotated[
    PerspectiveProjection | OrthographicProjection,
    Field(discriminator="mode"),
]


class Camera(ContractModel):
    pose: Pose
    projection: Projection


class ProjectedComponentBounds(ContractModel):
    projection_status: Literal["in_front", "behind", "crosses_camera_plane"]
    minimum_image_xy: tuple[FiniteFloat, FiniteFloat] | None = None
    maximum_image_xy: tuple[FiniteFloat, FiniteFloat] | None = None
    minimum_depth_mm: FiniteFloat
    maximum_depth_mm: FiniteFloat

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if self.minimum_depth_mm > self.maximum_depth_mm:
            raise ValueError("minimum_depth_mm must not exceed maximum_depth_mm")
        has_minimum = self.minimum_image_xy is not None
        if has_minimum != (self.maximum_image_xy is not None):
            raise ValueError("projected image bounds must be both present or both absent")
        if self.projection_status == "crosses_camera_plane":
            if has_minimum:
                raise ValueError("camera-plane intersections cannot have finite image bounds")
            return self
        if not has_minimum or self.maximum_image_xy is None or self.minimum_image_xy is None:
            raise ValueError(
                "finite projected image bounds are required away from the camera plane"
            )
        if any(
            low > high
            for low, high in zip(
                self.minimum_image_xy,
                self.maximum_image_xy,
                strict=True,
            )
        ):
            raise ValueError("minimum_image_xy must not exceed maximum_image_xy")
        return self


class CameraDiagnostics(ContractModel):
    aspect_ratio: PositiveFiniteFloat
    horizontal_fov_degrees: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)] | None
    vertical_fov_degrees: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)] | None
    right: Vec3
    up: Vec3
    forward: Vec3
    frustum_corners_mm: tuple[Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3]
    target_depth_mm: FiniteFloat
    projected_bounds: dict[Identifier, ProjectedComponentBounds]


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
    orientation_xyzw: UnitQuaternion
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
    orientation_xyzw: UnitQuaternion
    power_w: PositiveFiniteFloat
    spot_size_degrees: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)]
    blend: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.15


class SunLight(LightColor):
    id: Identifier
    type: Literal["sun"] = "sun"
    orientation_xyzw: UnitQuaternion
    strength: PositiveFiniteFloat
    angle_degrees: Annotated[float, Field(ge=0, le=180, allow_inf_nan=False)] = 0.526


type Light = Annotated[AreaLight | PointLight | SpotLight | SunLight, Field(discriminator="type")]


class IlluminationPreset(StrEnum):
    NEUTRAL_STUDIO = "neutral_studio"
    HIGH_KEY = "high_key"
    RAKING_LEFT = "raking_left"
    RAKING_RIGHT = "raking_right"
    BACKLIT = "backlit"
    FLAT_DIAGNOSTIC = "flat_diagnostic"


class PresetIllumination(ContractModel):
    preset: IlluminationPreset


class EnvironmentMap(ContractModel):
    path: Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    strength: PositiveFiniteFloat = 1.0
    rotation_degrees: FiniteFloat = 0.0
    projection: Literal["equirectangular"] = "equirectangular"


class CustomIllumination(ContractModel):
    preset: Literal["custom"] = "custom"
    background_rgb: tuple[NonNegativeFiniteFloat, NonNegativeFiniteFloat, NonNegativeFiniteFloat]
    ambient_strength: NonNegativeFiniteFloat
    lights: tuple[Light, ...] = Field(default=(), max_length=32)
    environment_map: EnvironmentMap | None = None

    @field_validator("lights")
    @classmethod
    def unique_light_ids(cls, lights: tuple[Light, ...]) -> tuple[Light, ...]:
        ids = [light.id for light in lights]
        if len(ids) != len(set(ids)):
            raise ValueError("light ids must be unique")
        return lights

    @model_validator(mode="after")
    def require_effective_output(self) -> Self:
        background_contributes = self.ambient_strength > 0 and any(self.background_rgb)
        environment_contributes = self.environment_map is not None
        light_contributes = any(
            light.color_temperature_k is not None
            or (light.linear_rgb is not None and any(light.linear_rgb))
            for light in self.lights
        )
        if not background_contributes and not light_contributes and not environment_contributes:
            raise ValueError("custom illumination must have non-zero light output")
        return self


type Illumination = PresetIllumination | CustomIllumination


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


class ComponentVisualState(ContractModel):
    display: DisplayMode = DisplayMode.SHOWN
    mark: MarkMode = MarkMode.UNMARKED


class SessionSnapshot(ContractModel):
    camera: Camera
    camera_diagnostics: CameraDiagnostics
    illumination: Illumination
    components: dict[Identifier, ComponentVisualState]
    state_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CapabilityWarning(ContractModel):
    code: Identifier
    message: Annotated[str, StringConstraints(min_length=1, max_length=2_000)]
    component_ids: tuple[Identifier, ...] = ()


class SceneCapabilities(ContractModel):
    hierarchy: Literal["preserved", "flattened"]
    component_names: Literal["source", "generated"]
    materials: Literal["preserved", "partial", "absent"]
    textures: Literal["preserved", "partial", "absent"]
    animations: Literal["absent", "static_pose"] = "absent"
    procedural_materials: Literal["absent", "unsupported"] = "absent"


class NormalizedGeometryArtifact(ContractModel):
    cache_key: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    path: Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    bytes: Annotated[int, Field(ge=1)]
    importer_version: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    normalization_parameters: dict[
        Annotated[str, StringConstraints(min_length=1, max_length=128)], JsonValue
    ]


class SceneManifest(ContractModel):
    schema_version: Literal[1] = 1
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    source_format: Literal["glb", "gltf", "obj", "stl"]
    units: Literal["millimeter"]
    root_bounds: Bounds
    components: tuple[Component, ...]
    imported_camera: Camera
    imported_illumination: Illumination
    capabilities: SceneCapabilities
    warnings: tuple[CapabilityWarning, ...] = ()
    normalized_geometry: NormalizedGeometryArtifact | None = None

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        by_id = {component.id: component for component in self.components}
        if len(by_id) != len(self.components):
            raise ValueError("component ids must be unique")
        paths = {component.path for component in self.components}
        if len(paths) != len(self.components):
            raise ValueError("component paths must be unique")
        for component in self.components:
            expected_id = stable_component_id(self.source_sha256, component.path)
            if component.id != expected_id:
                raise ValueError(
                    f"component {component.path} id must equal stable derivation {expected_id}"
                )
            if len(component.child_ids) != len(set(component.child_ids)):
                raise ValueError(f"component {component.id} child ids must be unique")
            if component.parent_id is not None and component.parent_id not in by_id:
                raise ValueError(f"unknown parent id: {component.parent_id}")
            if (
                component.parent_id is not None
                and component.id not in by_id[component.parent_id].child_ids
            ):
                raise ValueError(f"parent {component.parent_id} does not list child {component.id}")
            unknown_children = set(component.child_ids) - by_id.keys()
            if unknown_children:
                raise ValueError(f"unknown child ids: {sorted(unknown_children)}")
            for child_id in component.child_ids:
                if by_id[child_id].parent_id != component.id:
                    raise ValueError(
                        f"child {child_id} does not point back to parent {component.id}"
                    )
        for component in self.components:
            ancestors: set[str] = set()
            current: Component | None = component
            while current is not None:
                if current.id in ancestors:
                    raise ValueError(f"component hierarchy contains a cycle at {current.id}")
                ancestors.add(current.id)
                current = by_id[current.parent_id] if current.parent_id is not None else None
        for component in self.components:
            if component.parent_id is not None:
                parent_path = by_id[component.parent_id].path
                if not component.path.startswith(f"{parent_path}/"):
                    raise ValueError(
                        f"component {component.id} path must be under parent path {parent_path}"
                    )
            if any(
                child_bound < root_bound
                for child_bound, root_bound in zip(
                    component.world_bounds.minimum_mm,
                    self.root_bounds.minimum_mm,
                    strict=True,
                )
            ) or any(
                child_bound > root_bound
                for child_bound, root_bound in zip(
                    component.world_bounds.maximum_mm,
                    self.root_bounds.maximum_mm,
                    strict=True,
                )
            ):
                raise ValueError(f"root_bounds must enclose component {component.id} world_bounds")
        warning_ids = {
            component_id for warning in self.warnings for component_id in warning.component_ids
        }
        unknown_warning_ids = warning_ids - by_id.keys()
        if unknown_warning_ids:
            raise ValueError(
                f"warnings reference unknown component ids: {sorted(unknown_warning_ids)}"
            )
        return self


class RenderEngine(StrEnum):
    EEVEE = "eevee"
    CYCLES = "cycles"


class ImageArtifact(ContractModel):
    path: Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
    media_type: Literal["image/png", "image/x-exr"]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    bytes: Annotated[int, Field(ge=1)]


class LuminanceSummary(ContractModel):
    minimum: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    median: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    maximum: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    crushed_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    clipped_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if not self.minimum <= self.median <= self.maximum:
            raise ValueError("luminance values must be ordered minimum <= median <= maximum")
        return self


class EvaluatorPasses(ContractModel):
    multilayer: ImageArtifact
    component_ids: ImageArtifact
    highlighted: ImageArtifact
    component_colors: dict[Identifier, tuple[int, int, int]]

    @field_validator("component_colors")
    @classmethod
    def validate_component_colors(
        cls, colors: dict[str, tuple[int, int, int]]
    ) -> dict[str, tuple[int, int, int]]:
        if len(colors.values()) != len(set(colors.values())):
            raise ValueError("component pass colors must be unique")
        if any(channel < 0 or channel > 255 for color in colors.values() for channel in color):
            raise ValueError("component pass colors must use 8-bit channels")
        return colors


class RenderManifest(ContractModel):
    schema_version: Literal[1] = 1
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    state_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    width: Annotated[int, Field(ge=64, le=16_384)]
    height: Annotated[int, Field(ge=64, le=16_384)]
    samples: Annotated[int, Field(ge=1, le=4_096)]
    engine: RenderEngine
    device: Literal["graphics", "cuda"]
    blender_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    session: SessionSnapshot
    color: ImageArtifact
    evaluator: EvaluatorPasses | None = None
    luminance: LuminanceSummary

    @model_validator(mode="after")
    def validate_session_hash(self) -> Self:
        if self.state_sha256 != self.session.state_sha256:
            raise ValueError("render state hash must match the embedded session")
        return self


class ContactSheetPanel(ContractModel):
    index: Annotated[int, Field(ge=1)]
    caption: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    render: RenderManifest
    callouts: tuple[ContactSheetCallout, ...] = ()
    experiment: Literal["declared", "fixed_pose_focal_study", "dolly_zoom"] = "declared"


class ContactSheetCallout(ContractModel):
    number: Annotated[int, Field(ge=1, le=99)]
    component_id: Identifier
    label: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    image_xy: tuple[
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
    ]


class ContactSheetOrbit(ContractModel):
    target: Literal["focus", "scene"] = "focus"
    azimuth_degrees: FiniteFloat
    elevation_degrees: FiniteFloat
    roll_degrees: FiniteFloat = 0.0
    distance_mm: PositiveFiniteFloat
    projection: Projection
    reference_focal_length_mm: PositiveFiniteFloat | None = None
    reference_distance_mm: PositiveFiniteFloat | None = None


class ContactSheetPanelSpec(ContractModel):
    caption: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    camera: Camera | None = None
    orbit: ContactSheetOrbit | None = None
    illumination: Illumination | None = None
    display: Literal["context", "isolated"] = "context"
    mark: MarkMode = MarkMode.HIGHLIGHTED
    experiment: Literal["declared", "fixed_pose_focal_study", "dolly_zoom"] = "declared"

    @model_validator(mode="after")
    def validate_view_experiment(self) -> Self:
        if (self.camera is None) == (self.orbit is None):
            raise ValueError("custom panel must declare exactly one of camera or orbit")
        if self.experiment == "fixed_pose_focal_study" and (
            self.camera is None
            or not isinstance(self.camera.projection, PerspectiveProjection)
        ):
            raise ValueError("fixed_pose_focal_study requires a perspective camera")
        if self.experiment != "dolly_zoom":
            return self
        if self.orbit is None or not isinstance(self.orbit.projection, PerspectiveProjection):
            raise ValueError("dolly_zoom requires a perspective orbit")
        if (
            self.orbit.reference_focal_length_mm is None
            or self.orbit.reference_distance_mm is None
        ):
            raise ValueError("dolly_zoom requires reference focal length and distance")
        return self


class OccluderRemovalStep(ContractModel):
    component_id: Identifier
    ray_hit_count: Annotated[int, Field(ge=1)]
    visible_fraction_after: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class OcclusionEvidence(ContractModel):
    visibility_threshold: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    removal_budget: Annotated[int, Field(ge=0, le=32)]
    sample_count: Annotated[int, Field(ge=0)]
    visible_fraction_before: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    visible_fraction_after: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    steps: tuple[OccluderRemovalStep, ...] = ()
    stop_reason: Literal[
        "threshold_met_initial",
        "threshold_met",
        "budget_exhausted",
        "no_blockers",
        "focus_not_projected",
    ]


class ContactSheetManifest(ContractModel):
    schema_version: Literal[1] = 1
    recipe: Literal["focused_3x3", "custom_3x3"]
    focus_component_ids: tuple[Identifier, ...] = Field(min_length=1)
    removed_occluder_ids: tuple[Identifier, ...] = ()
    occlusion: OcclusionEvidence | None = None
    sheet: ImageArtifact
    panels: tuple[ContactSheetPanel, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def validate_panel_indices(self) -> Self:
        if tuple(panel.index for panel in self.panels) != tuple(range(1, 10)):
            raise ValueError("focused contact-sheet panel indices must be 1 through 9")
        return self
