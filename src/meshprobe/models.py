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
type SrgbHexColor = Annotated[
    str,
    StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$"),
    AfterValidator(str.lower),
]
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


class CoordinateFrame(StrEnum):
    SOURCE = "source"
    WORLD = "world"
    CAMERA = "camera"
    COMPONENT = "component"


class RotationFrame(StrEnum):
    """Frames a view-rotate axis/target may be expressed in.

    A strict subset of CoordinateFrame: the frames view-rotate actually honors. The
    component frame is intentionally absent — rotating around a target component's local
    axes is not implemented, so advertising it in the CLI would repeat the mismatch of
    issue #65.
    """

    SOURCE = "source"
    WORLD = "world"
    CAMERA = "camera"


class CameraPoseFrame(StrEnum):
    SOURCE = "source"
    WORLD = "world"


class Handedness(StrEnum):
    RIGHT = "right"


class AxisDirection(StrEnum):
    POSITIVE_X = "+x"
    POSITIVE_Y = "+y"
    POSITIVE_Z = "+z"


class CoordinateSystem(ContractModel):
    name: Identifier
    frame: CoordinateFrame
    handedness: Handedness
    up_axis: AxisDirection


class SceneCoordinateFrames(ContractModel):
    source: CoordinateSystem
    world: CoordinateSystem = CoordinateSystem(
        name="meshprobe_z_up",
        frame=CoordinateFrame.WORLD,
        handedness=Handedness.RIGHT,
        up_axis=AxisDirection.POSITIVE_Z,
    )
    source_to_world: Matrix4x4


class Bounds(ContractModel):
    minimum_mm: Vec3
    maximum_mm: Vec3
    frame: CoordinateFrame = CoordinateFrame.WORLD

    @model_validator(mode="after")
    def validate_extents(self) -> Self:
        if any(low > high for low, high in zip(self.minimum_mm, self.maximum_mm, strict=True)):
            raise ValueError("minimum_mm must not exceed maximum_mm")
        return self


class Pose(ContractModel):
    position_mm: Vec3
    orientation_xyzw: UnitQuaternion
    frame: CameraPoseFrame = CameraPoseFrame.WORLD


class SensorFit(StrEnum):
    AUTO = "auto"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class DepthOfFieldMode(StrEnum):
    DISABLED = "disabled"
    ENABLED = "enabled"


class ComponentFocus(ContractModel):
    component_id: Identifier


class DepthOfField(ContractModel):
    mode: DepthOfFieldMode = Field(
        default=DepthOfFieldMode.DISABLED,
        description="Whether the lens simulates finite-aperture defocus blur.",
    )
    aperture_fstop: PositiveFiniteFloat = Field(
        default=2.8,
        description="Lens f-number; smaller values widen the aperture and shrink the in-focus "
        "depth.",
    )
    focus_distance_mm: PositiveFiniteFloat | None = Field(
        default=None,
        description="Explicit focus-plane distance from the camera in millimetres.",
    )
    focus: ComponentFocus | None = Field(
        default=None,
        description="Component whose centroid sets the focus distance in lieu of "
        "focus_distance_mm.",
    )

    @model_validator(mode="after")
    def validate_focus(self) -> Self:
        focus_sources = int(self.focus_distance_mm is not None) + int(self.focus is not None)
        if self.mode is DepthOfFieldMode.ENABLED and focus_sources != 1:
            raise ValueError("enabled depth of field requires exactly one focus target")
        if self.mode is DepthOfFieldMode.DISABLED and focus_sources:
            raise ValueError("disabled depth of field cannot declare a focus target")
        return self


class PerspectiveProjection(ContractModel):
    mode: Literal["perspective"] = "perspective"
    focal_length_mm: PositiveFiniteFloat = Field(
        default=50.0,
        description="Camera lens focal length in millimetres; larger values narrow the field of "
        "view.",
    )
    sensor_width_mm: PositiveFiniteFloat = Field(
        default=36.0, description="Horizontal camera sensor size in millimetres."
    )
    sensor_height_mm: PositiveFiniteFloat = Field(
        default=24.0, description="Vertical camera sensor size in millimetres."
    )
    sensor_fit: SensorFit = Field(
        default=SensorFit.AUTO,
        description="Which sensor dimension the focal length fits when deriving the field of view.",
    )
    near_clip_mm: PositiveFiniteFloat = 0.5
    far_clip_mm: PositiveFiniteFloat = 100_000.0
    depth_of_field: DepthOfField = DepthOfField()

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


class CameraOrbitAngles(ContractModel):
    """View-orbit spherical angles derived from a world-frame camera orientation.

    Feeding these back to ``view-orbit`` (with a target and distance) reproduces the
    current viewing direction, so an orbit sequence stays continuous without decomposing
    the stored quaternion by hand. Both angles are target-independent.
    """

    azimuth_degrees: FiniteFloat
    elevation_degrees: FiniteFloat


class ProjectedComponentBounds(ContractModel):
    projection_status: Annotated[
        Literal["in_front", "behind", "crosses_camera_plane"],
        Field(
            description="Whether the component sits fully in front of, entirely behind, or "
            "straddling the camera plane."
        ),
    ]
    minimum_image_xy: tuple[FiniteFloat, FiniteFloat] | None = Field(
        default=None,
        description="Lower corner of the component's projected bounding box in normalized 0-1 "
        "frame coordinates; null when it crosses the camera plane.",
    )
    maximum_image_xy: tuple[FiniteFloat, FiniteFloat] | None = Field(
        default=None,
        description="Upper corner of the component's projected bounding box in normalized 0-1 "
        "frame coordinates; null when it crosses the camera plane.",
    )
    minimum_depth_mm: FiniteFloat = Field(
        description="Nearest component depth along the camera forward axis in millimetres."
    )
    maximum_depth_mm: FiniteFloat = Field(
        description="Farthest component depth along the camera forward axis in millimetres."
    )

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
    target_depth_mm: FiniteFloat = Field(
        description="Depth of the focus target (or scene centre) along the camera forward axis "
        "in millimetres."
    )
    projected_bounds: dict[Identifier, ProjectedComponentBounds] = Field(
        description="Per focus-component projected frame coverage keyed by stable id: the "
        "component's normalized image-space bounding box and depth range for the current camera. "
        "Populated for focus targets (e.g. an occlusion query); empty for an unfocused render."
    )


class CameraTranslationReceipt(ContractModel):
    operation: Literal["move"] = "move"
    requested_world_delta_mm: Vec3
    requested_camera_delta_mm: Vec3
    resolved_world_delta_mm: Vec3
    previous_pose: Pose
    resulting_pose: Pose


class OrthonormalBasis(ContractModel):
    x: Vec3 = (1.0, 0.0, 0.0)
    y: Vec3 = (0.0, 1.0, 0.0)
    z: Vec3 = (0.0, 0.0, 1.0)

    @model_validator(mode="after")
    def validate_orthonormal(self) -> Self:
        axes = (self.x, self.y, self.z)
        lengths = [math.sqrt(sum(value * value for value in axis)) for axis in axes]
        if any(not math.isclose(length, 1.0, abs_tol=1e-6) for length in lengths):
            raise ValueError("basis axes must have unit length")
        dot_products = [
            sum(left * right for left, right in zip(axes[first], axes[second], strict=True))
            for first, second in ((0, 1), (0, 2), (1, 2))
        ]
        if any(not math.isclose(value, 0.0, abs_tol=1e-6) for value in dot_products):
            raise ValueError("basis axes must be mutually perpendicular")
        cross_xy = (
            self.x[1] * self.y[2] - self.x[2] * self.y[1],
            self.x[2] * self.y[0] - self.x[0] * self.y[2],
            self.x[0] * self.y[1] - self.x[1] * self.y[0],
        )
        if any(
            not math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(cross_xy, self.z, strict=True)
        ):
            raise ValueError("basis must be right-handed (x cross y must equal z)")
        return self


class CameraRotationReceipt(ContractModel):
    operation: Literal["rotate"] = "rotate"
    frame: CoordinateFrame
    target_mm: Vec3
    axis: Literal["x", "y", "z"]
    basis: OrthonormalBasis
    axis_world: Vec3
    requested_visual_degrees: FiniteFloat
    applied_camera_orbit_degrees: FiniteFloat
    previous_pose: Pose
    resulting_pose: Pose


type CameraOperationReceipt = CameraTranslationReceipt | CameraRotationReceipt


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


class VisibleBackgroundMode(StrEnum):
    ENVIRONMENT = "environment"
    COLOR = "color"


class PresetIllumination(ContractModel):
    preset: IlluminationPreset
    background_rgb: (
        tuple[NonNegativeFiniteFloat, NonNegativeFiniteFloat, NonNegativeFiniteFloat] | None
    ) = None
    background_strength: NonNegativeFiniteFloat | None = None


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
    background_strength: NonNegativeFiniteFloat = Field(
        default_factory=lambda data: data["ambient_strength"]
    )
    ambient_rgb: tuple[NonNegativeFiniteFloat, NonNegativeFiniteFloat, NonNegativeFiniteFloat] = (
        Field(default_factory=lambda data: data["background_rgb"])
    )
    lights: tuple[Light, ...] = Field(default=(), max_length=32)
    environment_map: EnvironmentMap | None = None
    visible_background_mode: VisibleBackgroundMode = Field(
        default_factory=lambda data: (
            VisibleBackgroundMode.ENVIRONMENT
            if data["environment_map"] is not None
            else VisibleBackgroundMode.COLOR
        )
    )

    @field_validator("lights")
    @classmethod
    def unique_light_ids(cls, lights: tuple[Light, ...]) -> tuple[Light, ...]:
        ids = [light.id for light in lights]
        if len(ids) != len(set(ids)):
            raise ValueError("light ids must be unique")
        return lights

    @model_validator(mode="after")
    def require_effective_output(self) -> Self:
        if (
            self.visible_background_mode is VisibleBackgroundMode.ENVIRONMENT
            and self.environment_map is None
        ):
            raise ValueError("environment visible background requires an environment map")
        ambient_contributes = self.ambient_strength > 0 and any(self.ambient_rgb)
        environment_contributes = (
            self.environment_map is not None and self.environment_map.strength > 0
        )
        light_contributes = any(
            (light.strength if isinstance(light, SunLight) else light.power_w) > 0
            and (
                light.color_temperature_k is not None
                or (light.linear_rgb is not None and any(light.linear_rgb))
            )
            for light in self.lights
        )
        if not any(
            (
                ambient_contributes,
                light_contributes,
                environment_contributes,
            )
        ):
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
    world_transform_frame: CoordinateFrame = CoordinateFrame.WORLD
    local_bounds: Bounds
    world_bounds: Bounds
    materials: MaterialSummary = MaterialSummary()


class ComponentVisualState(ContractModel):
    display: DisplayMode = DisplayMode.SHOWN
    mark: MarkMode = MarkMode.UNMARKED
    mark_color: SrgbHexColor | None = None

    @model_validator(mode="after")
    def reject_color_without_mark(self) -> Self:
        if self.mark is MarkMode.UNMARKED and self.mark_color is not None:
            raise ValueError("mark_color requires a visible mark mode")
        return self


class SessionSnapshot(ContractModel):
    camera: Camera
    camera_diagnostics: CameraDiagnostics
    illumination: Illumination
    components: dict[Identifier, ComponentVisualState]
    state_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    camera_operation: CameraOperationReceipt | None = None


class CapabilityWarning(ContractModel):
    code: Identifier
    message: Annotated[str, StringConstraints(min_length=1, max_length=2_000)]
    component_ids: tuple[Identifier, ...] = ()


class SceneCapabilities(ContractModel):
    hierarchy: Annotated[
        Literal["preserved", "flattened"],
        Field(
            description="Preserved exposes component parent/child links. Flattened keeps "
            "authoritative source-ancestry paths but omits intermediate non-mesh nodes as "
            "addressable components."
        ),
    ]
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
    schema_version: Literal[2] = 2
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    source_format: Literal["glb", "gltf", "obj", "stl"]
    units: Literal["millimeter"]
    coordinate_frames: SceneCoordinateFrames
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


class RenderStyle(StrEnum):
    SHADED = "shaded"
    SHADED_EDGES = "shaded_edges"


class EdgeType(StrEnum):
    SILHOUETTE = "silhouette"
    BORDER = "border"
    CREASE = "crease"
    MATERIAL_BOUNDARY = "material_boundary"
    CONTOUR = "contour"
    EXTERNAL_CONTOUR = "external_contour"


class ShadedEdgesStyle(ContractModel):
    line_color: Annotated[str, StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$")] = "#202020"
    line_width: Annotated[float, Field(gt=0, le=10, allow_inf_nan=False)] = 1.5
    crease_angle_degrees: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)] = 120
    edge_types: tuple[EdgeType, ...] = (
        EdgeType.SILHOUETTE,
        EdgeType.BORDER,
        EdgeType.CREASE,
        EdgeType.MATERIAL_BOUNDARY,
    )

    @field_validator("edge_types")
    @classmethod
    def validate_edge_types(cls, edge_types: tuple[EdgeType, ...]) -> tuple[EdgeType, ...]:
        if not edge_types:
            raise ValueError("at least one edge type is required")
        if len(edge_types) != len(set(edge_types)):
            raise ValueError("edge types must be unique")
        return edge_types

    @field_validator("line_color")
    @classmethod
    def canonical_line_color(cls, line_color: str) -> str:
        return line_color.lower()


class RenderStyleState(ContractModel):
    style: RenderStyle = RenderStyle.SHADED
    shaded_edges: ShadedEdgesStyle = ShadedEdgesStyle()


class GraphicsPolicy(StrEnum):
    HARDWARE_REQUIRED = "hardware_required"
    SOFTWARE_ALLOWED = "software_allowed"


class GraphicsDeviceClass(StrEnum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    UNKNOWN = "unknown"


class GraphicsPlatform(ContractModel):
    vendor: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    renderer: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    version: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    backend: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    blender_device_type: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    device_class: GraphicsDeviceClass
    warnings: tuple[Annotated[str, StringConstraints(min_length=1, max_length=1_024)], ...] = ()


class ImageArtifact(ContractModel):
    path: Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
    media_type: Literal["image/png", "image/x-exr"]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    bytes: Annotated[int, Field(ge=1)]


class LuminanceSummary(ContractModel):
    """Exposure statistics of a rendered frame's luminance.

    Lets a caller detect an under- or over-exposed render programmatically, instead of
    visually inspecting the PNG.
    """

    minimum: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False, description="Darkest normalized luminance (0-1)."),
    ]
    median: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Median normalized luminance (0-1); a low value (e.g. 0.03) flags a "
            "near-black, underlit frame.",
        ),
    ]
    maximum: Annotated[
        float,
        Field(ge=0, le=1, allow_inf_nan=False, description="Brightest normalized luminance (0-1)."),
    ]
    crushed_fraction: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Fraction of pixels crushed to black (underexposed shadow clipping).",
        ),
    ]
    clipped_fraction: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Fraction of pixels clipped to white (overexposed highlight clipping).",
        ),
    ]

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


class ResolvedDepthOfField(ContractModel):
    aperture_fstop: PositiveFiniteFloat
    focus_distance_mm: PositiveFiniteFloat


class RenderManifest(ContractModel):
    schema_version: Literal[2] = 2
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    state_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    width: Annotated[int, Field(ge=64, le=16_384)]
    height: Annotated[int, Field(ge=64, le=16_384)]
    samples: Annotated[int, Field(ge=1, le=4_096)]
    engine: RenderEngine
    style: RenderStyle = RenderStyle.SHADED
    shaded_edges: ShadedEdgesStyle = ShadedEdgesStyle()
    device: Literal["graphics_hardware", "graphics_software", "cuda"]
    graphics_policy: GraphicsPolicy
    graphics: GraphicsPlatform
    blender_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    session: SessionSnapshot
    color: ImageArtifact
    evaluator: EvaluatorPasses | None = None
    luminance: LuminanceSummary = Field(
        description="Exposure summary of the rendered frame; use it to detect an under- or "
        "over-exposed render programmatically."
    )
    resolved_depth_of_field: ResolvedDepthOfField | None = None

    @model_validator(mode="after")
    def validate_session_hash(self) -> Self:
        if self.state_sha256 != self.session.state_sha256:
            raise ValueError("render state hash must match the embedded session")
        return self


class ContactSheetPanel(ContractModel):
    index: Annotated[
        int, Field(ge=1, description="1-based panel position within the 3x3 contact sheet.")
    ]
    caption: Annotated[
        str,
        StringConstraints(min_length=1, max_length=10_000),
        Field(
            description="Human-readable label describing the panel's view "
            "(e.g. 'Natural isometric', 'Focused in context')."
        ),
    ]
    render: RenderManifest
    callouts: tuple[ContactSheetCallout, ...] = Field(
        default=(),
        description="Numbered marker annotations placed on the panel, each pointing at a focus "
        "component.",
    )
    experiment: Annotated[
        Literal["declared", "fixed_pose_focal_study", "dolly_zoom"],
        Field(description="Which panel experiment produced this view."),
    ] = "declared"


class ContactSheetCallout(ContractModel):
    number: Annotated[
        int,
        Field(ge=1, le=99, description="Marker number drawn on the panel, matching the badge."),
    ]
    component_id: Identifier = Field(description="Stable id of the component the marker points at.")
    label: Identifier = Field(description="Short label shown beside the marker.")
    image_xy: tuple[
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
    ] = Field(description="Marker position in normalized 0-1 panel image coordinates.")


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
            self.camera is None or not isinstance(self.camera.projection, PerspectiveProjection)
        ):
            raise ValueError("fixed_pose_focal_study requires a perspective camera")
        if self.experiment != "dolly_zoom":
            return self
        if self.orbit is None or not isinstance(self.orbit.projection, PerspectiveProjection):
            raise ValueError("dolly_zoom requires a perspective orbit")
        if self.orbit.reference_focal_length_mm is None or self.orbit.reference_distance_mm is None:
            raise ValueError("dolly_zoom requires reference focal length and distance")
        return self


class OccluderRemovalStep(ContractModel):
    """One occluder removed while clearing the focus target, with its visibility effect."""

    component_id: Identifier = Field(description="Stable id of the occluder removed at this step.")
    display_name: Identifier = Field(description="Human-readable name of the removed occluder.")
    component_path: Identifier = Field(description="Full scene path of the removed occluder.")
    ray_hit_count: Annotated[
        int,
        Field(ge=1, description="Focus rays this occluder was blocking before removal."),
    ]
    visible_pixel_count_after: Annotated[
        int,
        Field(ge=0, description="Visible focus-mask pixels after removing this occluder."),
    ]
    visible_fraction_after: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Visible focus fraction after removing this occluder.",
        ),
    ]


class OcclusionEvidence(ContractModel):
    """Full trace of the occluder-removal algorithm behind a focused contact sheet."""

    camera: Camera = Field(description="Camera the visibility measurements were taken from.")
    camera_source: Annotated[
        Literal["generated_focus_context", "current_session"],
        Field(
            description="Whether the camera was auto-framed for the focus target or taken from "
            "the current session."
        ),
    ]
    visibility_width_px: Annotated[
        int,
        Field(
            ge=4,
            le=1_024,
            description="Render width used to measure pixel visibility for the attributed camera.",
        ),
    ]
    visibility_height_px: Annotated[
        int,
        Field(
            ge=4,
            le=1_024,
            description="Render height used to measure pixel visibility for the attributed camera.",
        ),
    ]
    visibility_threshold: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Minimum visible focus fraction the sheet aims to reach before it stops "
            "removing occluders.",
        ),
    ]
    removal_budget: Annotated[
        int,
        Field(ge=0, le=32, description="Maximum number of occluders the sheet may remove."),
    ]
    sample_count: Annotated[
        int,
        Field(
            ge=0,
            description="Mesh vertex and centroid rays used only to rank blocking components.",
        ),
    ]
    visible_pixel_count_before: Annotated[
        int, Field(ge=0, description="Visible focus-mask pixels before any occluder removal.")
    ]
    visible_pixel_count_after: Annotated[
        int, Field(ge=0, description="Visible focus-mask pixels after all removal steps.")
    ]
    isolated_pixel_count: Annotated[
        int,
        Field(
            ge=0,
            description="Focus-mask pixels rendered in isolation; the visibility denominator.",
        ),
    ]
    visible_fraction_before: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Visible focus-mask pixels divided by isolated focus-mask pixels.",
        ),
    ]
    visible_fraction_after: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Recomputed pixel visibility after all reported removal steps.",
        ),
    ]
    steps: tuple[OccluderRemovalStep, ...] = Field(
        default=(),
        description="Ordered trace of each occluder removal and its effect on focus visibility.",
    )
    stop_reason: Annotated[
        Literal[
            "threshold_met_initial",
            "threshold_met",
            "budget_exhausted",
            "no_blockers",
            "focus_not_projected",
        ],
        Field(description="Why occluder removal halted."),
    ]

    @model_validator(mode="after")
    def validate_visibility_progression(self) -> Self:
        denominator = self.isolated_pixel_count
        if self.visible_pixel_count_before > denominator:
            raise ValueError("visible pixels before removal cannot exceed isolated pixels")
        if self.visible_pixel_count_after > denominator:
            raise ValueError("visible pixels after removal cannot exceed isolated pixels")
        expected_before = self.visible_pixel_count_before / denominator if denominator else 0.0
        expected_after = self.visible_pixel_count_after / denominator if denominator else 0.0
        if not math.isclose(self.visible_fraction_before, expected_before, abs_tol=1e-12):
            raise ValueError("visible fraction before removal must match pixel counts")
        if not math.isclose(self.visible_fraction_after, expected_after, abs_tol=1e-12):
            raise ValueError("visible fraction after removal must match pixel counts")
        if self.visible_pixel_count_after < self.visible_pixel_count_before:
            raise ValueError("occluder removal cannot reduce visible focus pixels")

        previous_count = self.visible_pixel_count_before
        for step in self.steps:
            if step.visible_pixel_count_after < previous_count:
                raise ValueError("occluder removal steps must not reduce visible focus pixels")
            if step.visible_pixel_count_after > denominator:
                raise ValueError("step visible pixels cannot exceed isolated pixels")
            expected_step = step.visible_pixel_count_after / denominator if denominator else 0.0
            if not math.isclose(step.visible_fraction_after, expected_step, abs_tol=1e-12):
                raise ValueError("step visible fraction must match pixel counts")
            previous_count = step.visible_pixel_count_after
        if self.steps:
            last = self.steps[-1]
            if last.visible_pixel_count_after != self.visible_pixel_count_after or not math.isclose(
                last.visible_fraction_after,
                self.visible_fraction_after,
                abs_tol=1e-12,
            ):
                raise ValueError("last removal step must match final visibility")
        elif self.visible_pixel_count_after != self.visible_pixel_count_before or not math.isclose(
            self.visible_fraction_after,
            self.visible_fraction_before,
            abs_tol=1e-12,
        ):
            raise ValueError("visibility cannot change without a removal step")
        return self


class OcclusionBlocker(ContractModel):
    component_id: Identifier
    display_name: Identifier
    component_path: Identifier
    ray_hit_count: Annotated[int, Field(ge=1)]


class OcclusionQueryResult(ContractModel):
    focus_component_ids: tuple[Identifier, ...] = Field(min_length=1)
    camera: Camera = Field(description="Camera the occlusion query was evaluated from.")
    camera_diagnostics: CameraDiagnostics = Field(
        description="Camera framing for the query, including projected_bounds: the per-component "
        "normalized frame coverage of each focus target."
    )
    aspect_ratio: PositiveFiniteFloat
    camera_source: Literal["current_session"] = "current_session"
    projection_status: Literal["projected", "not_projected"]
    sample_count: Annotated[
        int,
        Field(
            ge=0,
            description="Projected mesh vertex and centroid rays evaluated from the camera.",
        ),
    ]
    visible_sample_count: Annotated[int, Field(ge=0)]
    occluded_sample_count: Annotated[int, Field(ge=0)]
    unresolved_sample_count: Annotated[int, Field(ge=0)]
    visible_fraction: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Visible focus rays divided by all projected focus rays.",
        ),
    ]
    blockers: tuple[OcclusionBlocker, ...] = Field(
        default=(),
        description="Components blocking the focus target from this camera, ranked by blocked "
        "ray count.",
    )
    state_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if not math.isclose(
            self.aspect_ratio,
            self.camera_diagnostics.aspect_ratio,
            abs_tol=1e-12,
        ):
            raise ValueError("evaluated aspect ratio must match camera diagnostics")
        classified = (
            self.visible_sample_count + self.occluded_sample_count + self.unresolved_sample_count
        )
        if classified != self.sample_count:
            raise ValueError("classified sample counts must equal sample count")
        expected_fraction = (
            self.visible_sample_count / self.sample_count if self.sample_count else 0.0
        )
        if not math.isclose(self.visible_fraction, expected_fraction, abs_tol=1e-12):
            raise ValueError("visible fraction must match sample counts")
        expected_status = "projected" if self.sample_count else "not_projected"
        if self.projection_status != expected_status:
            raise ValueError("projection status must match sample count")
        return self


class ContactSheetManifest(ContractModel):
    schema_version: Literal[5] = 5
    recipe: Literal["focused_3x3", "custom_3x3"]
    focus_component_ids: tuple[Identifier, ...] = Field(min_length=1)
    removed_occluder_ids: tuple[Identifier, ...] = ()
    occlusion: OcclusionEvidence | None = Field(
        default=None,
        description="Trace of the automatic occluder-removal analysis, or null when none ran.",
    )
    sheet: ImageArtifact
    panels: tuple[ContactSheetPanel, ...] = Field(
        min_length=9,
        max_length=9,
        description="The nine rendered panels, each with its caption and numbered callouts.",
    )

    @model_validator(mode="after")
    def validate_panel_indices(self) -> Self:
        if tuple(panel.index for panel in self.panels) != tuple(range(1, 10)):
            raise ValueError("focused contact-sheet panel indices must be 1 through 9")
        return self


StateHash = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
    Field(description="Digest of the durable session state after the operation."),
]


class CameraViewResult(ContractModel):
    """Result of an absolute view command (view.set / view.orbit)."""

    state_sha256: StateHash
    camera: Camera = Field(description="Resolved camera pose and projection after the view change.")
    camera_diagnostics: CameraDiagnostics = Field(
        description="Derived camera framing: field of view, basis axes, frustum corners, and "
        "projected focus bounds."
    )


class CameraMotionResult(CameraViewResult):
    """Result of a relative view command (view.move / view.rotate)."""

    camera_operation: CameraOperationReceipt = Field(
        description="Receipt describing the requested and applied camera translation or rotation."
    )


class IlluminationResult(ContractModel):
    """Result of illumination.set."""

    state_sha256: StateHash
    illumination: Illumination = Field(description="Resolved scene illumination after the change.")


class ComponentVisualStateResult(ContractModel):
    """Result of component.display / component.mark."""

    state_sha256: StateHash
    components: dict[Identifier, ComponentVisualState] = Field(
        description="Per-component visibility and mark state after the change, keyed by stable id."
    )


class SessionResetResult(ContractModel):
    """Result of session.reset."""

    reset: Literal[True] = Field(
        default=True,
        description="Always true; confirms the visual state was reset to the scene defaults.",
    )
    state_sha256: StateHash


class SessionSnapshotResult(ContractModel):
    """Result of session.snapshot."""

    scene: SceneManifest = Field(
        description="The imported scene manifest: components, coordinate frames, and capabilities."
    )
    session: SessionSnapshot = Field(
        description="Complete camera, illumination, and per-component visual state snapshot."
    )
