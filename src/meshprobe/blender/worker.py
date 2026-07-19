"""Factory-clean Blender worker using newline-delimited JSON over stdio.

This module runs inside Blender's bundled Python. Keep it free of third-party
dependencies; the controller validates its output with the package's Pydantic
contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import struct
import sys
import tempfile
import traceback
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any

import bpy  # type: ignore[import-not-found]
import gpu  # type: ignore[import-not-found]
from bpy_extras.object_utils import world_to_camera_view  # type: ignore[import-not-found]
from mathutils import Matrix, Quaternion, Vector  # type: ignore[import-not-found]

PROTOCOL_VERSION = 2
# Blender 4.2 runs in a bounded software-only compatibility mode. Its background
# GPU API lacks ``gpu.init()``, so graphics telemetry, hardware-required EEVEE,
# and the GPU compositor's ``screen_edges`` style require Blender 5.2 or newer.
MINIMUM_BLENDER_VERSION = (4, 2)
GPU_INITIALIZATION_MINIMUM_VERSION = (5, 2)
MILLIMETERS_PER_METER = 1_000.0
PRESET_REFERENCE_SPAN_MM = 5_000.0
# The classic CAD-export bug: glTF's spec unit is meters, but a source authored in
# millimeters (nearly every mechanical CAD package's native unit) gets exported without
# converting the numbers, so a real 0.5 m part shows up 1000x too large. The tools this
# package inspects — machines, assemblies, individual mechanical parts — essentially never
# exceed a few meters; 50 m is already an enormous machine (a large industrial press, a
# small building), comfortably clearing legitimate large assemblies while still catching
# the mm/m mixup long before it reaches the absurd 100s-of-meters range issue #100 reported.
PLAUSIBLE_MAX_SPAN_METERS = 50.0
# The mirror mistake — meters authored, millimeters assumed — would show up as a
# suspiciously TINY scene. This tool is also used to inspect genuinely small mechanical
# parts (fasteners, connector pins, watch components) that can legitimately span a few
# millimeters, so the floor sits at 1 mm: a whole asset's bounding box below that is well
# under anything meaningful to inspect standalone here, and far more likely a units (or a
# wrong --unit-scale) mistake than a real part.
PLAUSIBLE_MIN_SPAN_METERS = 0.001


class MaskMaterialMode(StrEnum):
    FLAT_BY_COMPONENT = "flat_by_component"
    MATERIAL_EXACT = "material_exact"
    UNIFORM_BACKFACE_CULLED = "uniform_backface_culled"
    UNIFORM_DOUBLE_SIDED = "uniform_double_sided"


DISPLAY_MODES = {"shown", "hidden", "isolated", "ghosted"}
MARK_MODES = {"unmarked", "selected", "highlighted", "labeled"}
MARK_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
MARK_COLORS = {
    "selected": (0.05, 0.8, 1.0, 1.0),
    # Linear RGBA of sRGB #FF1493 (deep pink); pops against brass/copper materials
    # that the former salmon default blended into. See issue #60.
    "highlighted": (1.0, 0.006995410187265387, 0.29177064981753587, 1.0),
    "labeled": (1.0, 0.85, 0.05, 1.0),
}
SUPPORTED_GLTF_EXTENSIONS = {
    "EXT_mesh_gpu_instancing",
    "KHR_draco_mesh_compression",
    "KHR_lights_punctual",
    "KHR_materials_anisotropy",
    "KHR_materials_clearcoat",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_iridescence",
    "KHR_materials_pbrSpecularGlossiness",
    "KHR_materials_sheen",
    "KHR_materials_specular",
    "KHR_materials_transmission",
    "KHR_materials_unlit",
    "KHR_materials_variants",
    "KHR_materials_volume",
    "KHR_mesh_quantization",
    "KHR_texture_basisu",
    "KHR_texture_transform",
    "KHR_xmp_json_ld",
}
MANIFEST: dict[str, Any] | None = None
COMPONENT_OBJECTS: dict[str, bpy.types.Object] = {}
ORIGINAL_MESHES: dict[str, bpy.types.Mesh] = {}
ORIGINAL_VISIBILITY: dict[str, tuple[bool, bool]] = {}
COMPONENT_STATES: dict[str, dict[str, Any]] = {}
IMPORTED_CAMERA: dict[str, Any] | None = None
# The aspect_ratio last established by scene.open or an explicit session.reset —
# an internal session.reset that omits the field (e.g. render_contact_sheet's
# cleanup pass) reuses this instead of silently reframing to a square default.
DEFAULT_ASPECT_RATIO: float = 1.0
CURRENT_CAMERA: dict[str, Any] | None = None
CURRENT_CAMERA_DIAGNOSTICS: dict[str, Any] | None = None
CURRENT_CAMERA_OPERATION: dict[str, Any] | None = None
# ``view.orbit --aspect-ratio`` derives one sensor dimension for the requested
# render shape. Keep the physical pair that preceded that derivation separately:
# the public camera state intentionally records the derived shape, but a later
# AUTO-fit orbit must not use that derived edge as a new physical intrinsic.
CURRENT_PERSPECTIVE_SENSOR_INTRINSICS: tuple[float, float] | None = None
IMPORTED_ILLUMINATION: dict[str, Any] | None = None
CURRENT_ILLUMINATION: dict[str, Any] | None = None
MARK_OBJECTS: dict[str, bpy.types.Object] = {}
OVERRIDE_MATERIALS: dict[str, bpy.types.Material] = {}
EMISSION_MATERIALS: dict[str, bpy.types.Material] = {}
GPU_PLATFORM: dict[str, Any] | None = None


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def stable_component_id(source_sha256: str, instance_path: str) -> str:
    digest = hashlib.sha256(f"{source_sha256}\0{instance_path}".encode()).hexdigest()
    return f"cmp_{digest[:24]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, media_type: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "media_type": media_type,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def object_path(obj: bpy.types.Object) -> str:
    names: list[str] = []
    current: bpy.types.Object | None = obj
    while current is not None:
        names.append(escape_path_segment(current.name))
        current = current.parent
    return "/".join(reversed(names))


def escape_path_segment(name: str) -> str:
    return name.replace("%", "%25").replace("/", "%2F")


def nearest_component_parent(
    obj: bpy.types.Object, component_objects: set[bpy.types.Object]
) -> bpy.types.Object | None:
    current = obj.parent
    while current is not None:
        if current in component_objects:
            return current
        current = current.parent
    return None


def bounds_of_points(points: list[Vector]) -> dict[str, list[float]]:
    return {
        "minimum_mm": [
            min(point[axis] for point in points) * MILLIMETERS_PER_METER for axis in range(3)
        ],
        "maximum_mm": [
            max(point[axis] for point in points) * MILLIMETERS_PER_METER for axis in range(3)
        ],
    }


def plausibility_warning(root_bounds: dict[str, list[float]]) -> dict[str, Any] | None:
    """Flag a root bounding box whose size suggests a millimeter/meter unit mistake.

    See the reasoning documented next to PLAUSIBLE_MAX_SPAN_METERS /
    PLAUSIBLE_MIN_SPAN_METERS. Only the long edge of the root bounds matters — a single
    absurd dimension is exactly the signature of an incorrectly scaled source.
    """
    minimum = root_bounds["minimum_mm"]
    maximum = root_bounds["maximum_mm"]
    span_mm = max(high - low for low, high in zip(minimum, maximum, strict=True))
    span_m = span_mm / MILLIMETERS_PER_METER
    if span_m > PLAUSIBLE_MAX_SPAN_METERS:
        return {
            "code": "units.suspected_millimeters",
            "message": (
                f"root bounds span {span_m:.1f} m — source coordinates may be millimetres "
                "interpreted as metres; consider --unit-scale 0.001"
            ),
            "component_ids": [],
        }
    if span_m < PLAUSIBLE_MIN_SPAN_METERS:
        return {
            "code": "units.suspected_meters",
            "message": (
                f"root bounds span {span_mm:.4f} mm — source coordinates may be metres "
                "interpreted as millimetres; consider --unit-scale 1000"
            ),
            "component_ids": [],
        }
    return None


def object_bounds(obj: bpy.types.Object) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    local_points = [Vector(corner) for corner in obj.bound_box]
    world_points = [obj.matrix_world @ point for point in local_points]
    local_bounds = bounds_of_points(local_points)
    local_bounds["frame"] = "component"  # type: ignore[assignment]
    world_bounds = bounds_of_points(world_points)
    world_bounds["frame"] = "world"  # type: ignore[assignment]
    return local_bounds, world_bounds


def matrix_in_millimeters(obj: bpy.types.Object) -> list[float]:
    matrix = obj.matrix_world.copy()
    for axis in range(3):
        matrix[axis][3] *= MILLIMETERS_PER_METER
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def mesh_hash(mesh: bpy.types.Mesh) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", len(mesh.vertices), len(mesh.polygons)))
    for vertex in mesh.vertices:
        digest.update(struct.pack("<ddd", *vertex.co))
    for polygon in mesh.polygons:
        digest.update(struct.pack("<I", len(polygon.vertices)))
        for vertex_index in polygon.vertices:
            digest.update(struct.pack("<I", vertex_index))
    return digest.hexdigest()


def material_summary(obj: bpy.types.Object) -> tuple[dict[str, list[str]], int]:
    names: list[str] = []
    missing_textures: list[str] = []
    texture_count = 0
    for slot in obj.material_slots:
        material = slot.material
        if material is None:
            continue
        names.append(material.name)
        if material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE":
                continue
            texture_count += 1
            image = node.image
            if image is None:
                missing_textures.append(f"{material.name}:{node.name}")
                continue
            if image.packed_file is not None or not image.filepath:
                continue
            resolved = Path(bpy.path.abspath(image.filepath))
            if not resolved.exists():
                missing_textures.append(str(resolved))
    return (
        {
            "names": sorted(set(names)),
            "missing_textures": sorted(set(missing_textures)),
        },
        texture_count,
    )


def quaternion_xyzw(obj: bpy.types.Object) -> list[float]:
    quaternion = obj.matrix_world.to_quaternion().normalized()
    return [quaternion.x, quaternion.y, quaternion.z, quaternion.w]


def imported_camera(
    root_bounds: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    cameras = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"), key=lambda obj: obj.name
    )
    if cameras:
        active_camera = bpy.context.scene.camera
        camera = active_camera if active_camera in cameras else cameras[0]
        data = camera.data
        projection: dict[str, Any]
        if data.type == "ORTHO":
            projection = {
                "mode": "orthographic",
                "scale_mm": data.ortho_scale * MILLIMETERS_PER_METER,
                "near_clip_mm": data.clip_start * MILLIMETERS_PER_METER,
                "far_clip_mm": data.clip_end * MILLIMETERS_PER_METER,
            }
        else:
            projection = {
                "mode": "perspective",
                "focal_length_mm": data.lens,
                "sensor_width_mm": data.sensor_width,
                "sensor_height_mm": data.sensor_height,
                "sensor_fit": data.sensor_fit.lower(),
                "near_clip_mm": data.clip_start * MILLIMETERS_PER_METER,
                "far_clip_mm": data.clip_end * MILLIMETERS_PER_METER,
                "depth_of_field": (
                    {
                        "mode": "enabled",
                        "aperture_fstop": data.dof.aperture_fstop,
                        "focus_distance_mm": data.dof.focus_distance * MILLIMETERS_PER_METER,
                    }
                    if data.dof.use_dof
                    else {"mode": "disabled"}
                ),
            }
        return (
            {
                "pose": {
                    "position_mm": [
                        value * MILLIMETERS_PER_METER for value in camera.matrix_world.translation
                    ],
                    "orientation_xyzw": quaternion_xyzw(camera),
                },
                "projection": projection,
            },
            None,
        )

    minimum = root_bounds["minimum_mm"]
    maximum = root_bounds["maximum_mm"]
    center = [(low + high) / 2 for low, high in zip(minimum, maximum, strict=True)]
    span = max(high - low for low, high in zip(minimum, maximum, strict=True))
    distance = max(span * 2, 100.0)
    position = Vector((center[0] + distance, center[1] + distance, center[2] + distance))
    direction = Vector(center) - position
    quaternion = direction.to_track_quat("-Z", "Y").normalized()
    return (
        {
            "pose": {
                "position_mm": list(position),
                "orientation_xyzw": [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            },
            "projection": {
                "mode": "perspective",
                "focal_length_mm": 50.0,
                "sensor_width_mm": 36.0,
                "sensor_height_mm": 24.0,
                "sensor_fit": "auto",
                "near_clip_mm": 0.5,
                "far_clip_mm": max(100_000.0, distance * 4.0 + span * 2.0),
            },
        },
        {
            "code": "camera.generated",
            "message": (
                "The source had no camera; MeshProbe created a deterministic inspection camera."
            ),
            "component_ids": [],
        },
    )


# Fraction of the frame's limiting axis the model fills once the default camera
# auto-frames the scene (see meshprobe.camera.DEFAULT_FRAME_FILL, kept in sync;
# the worker runs inside Blender and cannot import the package).
DEFAULT_FRAME_FILL = 0.9


def visible_root_bounds(fallback: dict[str, list[float]]) -> dict[str, list[float]]:
    """Bounds of only the originally-shown components, for default-view framing.

    ``root_bounds`` spans every imported mesh including source-hidden construction
    geometry or alternate configurations, so framing against it can centre and back
    off the camera for objects the default view never actually shows. Falls back to
    the full scene bounds if every component started hidden, so framing never
    degenerates to an empty/undefined bounding box.
    """
    points: list[Vector] = []
    for component_id, obj in COMPONENT_OBJECTS.items():
        hide_render, _ = ORIGINAL_VISIBILITY.get(component_id, (False, False))
        if hide_render:
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return fallback
    bounds = bounds_of_points(points)
    bounds["frame"] = "world"  # type: ignore[assignment]
    return bounds


def framed_default_camera(
    camera: dict[str, Any],
    root_bounds: dict[str, list[float]],
    frame_fill: float = DEFAULT_FRAME_FILL,
    aspect_ratio: float = 1.0,
) -> dict[str, Any]:
    """Re-frame an imported camera so the whole scene fills the default view.

    Keeps the source camera's viewing direction and lens but dollies it in (or
    out) along its own axis and re-centres it on ``root_bounds`` so the model
    occupies ``frame_fill`` of the frame instead of whatever margin the source
    happened to author. The faithful source camera is left untouched in the
    manifest; only the applied session view is tightened.

    ``aspect_ratio`` is the render width/height the caller intends to use with
    this default view. A perspective camera with ``sensor_fit="auto"`` narrows
    one FOV axis relative to the other away from a square render, so fitting
    against the wrong assumed aspect can under- or over-shoot the true frame
    and clip a tightly-fit model; it defaults to ``1.0`` (square) when the
    caller has no specific render size in mind yet.
    """

    framed = deepcopy(camera)
    projection = normalize_camera_projection(camera["projection"])
    pose = camera["pose"]
    x, y, z, w = pose["orientation_xyzw"]
    rotation = Quaternion((w, x, y, z)).normalized()
    forward = rotation @ Vector((0.0, 0.0, -1.0))
    right = rotation @ Vector((1.0, 0.0, 0.0))
    up = rotation @ Vector((0.0, 1.0, 0.0))
    minimum = root_bounds["minimum_mm"]
    maximum = root_bounds["maximum_mm"]
    center = Vector((low + high) / 2 for low, high in zip(minimum, maximum, strict=True))
    corners = [
        Vector((corner_x, corner_y, corner_z))
        for corner_x in (minimum[0], maximum[0])
        for corner_y in (minimum[1], maximum[1])
        for corner_z in (minimum[2], maximum[2])
    ]
    bounding_radius = max((corner - center).length for corner in corners)

    # The nearest bound's depth (most negative, i.e. closest to the camera along
    # -forward) sets a floor on how far back the camera must sit: fitting only
    # against a coarse aggregate extent can still
    # place the camera on or inside a bound that is very close along this specific
    # viewing axis (a corner or a thin, view-aligned object), which then gets
    # clipped regardless of the near_clip_mm floor below.
    nearest_depth = min((corner - center) @ forward for corner in corners)
    clearance = -nearest_depth + 1.0

    if projection["mode"] == "orthographic":
        half_width = max(abs((corner - center) @ right) for corner in corners)
        half_height = max(abs((corner - center) @ up) for corner in corners)
        framed_projection = deepcopy(projection)
        # Blender's ortho_scale, like a perspective sensor under sensor_fit="auto",
        # sets the FITTED axis directly and derives the other from aspect_ratio: a
        # landscape (>= 1) render fits horizontally (the vertical extent is scaled
        # down by aspect_ratio), a portrait render fits vertically. Fitting as if
        # aspect_ratio were always 1 ignores that derivation and can under-size the
        # non-fitted axis, clipping it in the requested render. Verified against a
        # live Blender `Camera.view_frame()` probe: an ORTHO camera with the
        # default (unset) sensor_fit="AUTO" genuinely halves the non-fitted axis
        # by aspect_ratio rather than treating scale_mm as always-vertical, which
        # is what this codebase's renderer-independent `camera_diagnostics` model
        # (camera.py) assumes — a separate, pre-existing inconsistency between
        # that model and the real renderer, out of scope here since fixing the
        # renderer to match the model breaks actual rendered output (confirmed:
        # doing so clips this file's landscape orthographic fixture vertically).
        if aspect_ratio >= 1:
            required_half = max(half_width, half_height * aspect_ratio, 1.0)
        else:
            required_half = max(half_height, half_width / aspect_ratio, 1.0)
        framed_projection["scale_mm"] = 2.0 * required_half / frame_fill
        standoff = max(clearance, bounding_radius + 1.0, 1.0)
        position = center - forward * standoff
    else:
        horizontal_half_tan, vertical_half_tan = _perspective_half_tangents(
            projection, aspect_ratio
        )
        distance = 0.0
        for corner in corners:
            offset = corner - center
            depth = offset @ forward
            horizontal = abs(offset @ right) / (horizontal_half_tan * frame_fill)
            vertical = abs(offset @ up) / (vertical_half_tan * frame_fill)
            distance = max(distance, horizontal - depth, vertical - depth)
        distance = max(distance, clearance)
        position = center - forward * distance
        framed_projection = deepcopy(projection)

    # Recompute clip planes from the actual camera position (not the aggregate
    # fitted aggregate distance) for both projections alike: `clearance` above guarantees
    # min(depths) >= 1.0, so near_clip_mm (half that) always sits strictly in
    # front of the nearest surface instead of relying on its own 0.1 mm floor to
    # save a degenerate fit.
    depths = [(corner - position) @ forward for corner in corners]
    framed_projection["near_clip_mm"] = max(0.1, min(depths) * 0.5)
    framed_projection["far_clip_mm"] = max(depths) * 1.5

    # An explicit numeric focus_distance_mm is an absolute distance from the SOURCE
    # camera position; re-derive it from the world-space point it originally named,
    # or the newly framed (dollied/re-centred) view keeps focusing where the source
    # camera used to sit instead of where it now points.
    depth_of_field = framed_projection.get("depth_of_field")
    if (
        depth_of_field is not None
        and depth_of_field.get("mode") == "enabled"
        and depth_of_field.get("focus_distance_mm") is not None
    ):
        original_position = Vector(pose["position_mm"])
        world_focus_point = original_position + forward * depth_of_field["focus_distance_mm"]
        framed_projection["depth_of_field"] = {
            **depth_of_field,
            "focus_distance_mm": max(0.1, (world_focus_point - position) @ forward),
        }

    framed["pose"] = {
        "position_mm": list(position),
        "orientation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
        "frame": "world",
    }
    framed["projection"] = framed_projection
    return framed


def _perspective_half_tangents(
    projection: dict[str, Any],
    aspect_ratio: float,
) -> tuple[float, float]:
    sensor_width = projection["sensor_width_mm"]
    sensor_height = projection.get("sensor_height_mm", 24.0)
    sensor_fit = projection["sensor_fit"]
    if sensor_fit == "auto":
        sensor_fit = "horizontal" if aspect_ratio >= 1 else "vertical"
    if sensor_fit == "vertical":
        sensor_width = sensor_height * aspect_ratio
    else:
        sensor_height = sensor_width / aspect_ratio
    focal_length = projection["focal_length_mm"]
    return sensor_width / (2 * focal_length), sensor_height / (2 * focal_length)


def light_color(data: bpy.types.Light) -> dict[str, list[float]]:
    return {"linear_rgb": [float(channel) for channel in data.color]}


def imported_illumination() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lights = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"), key=lambda obj: obj.name
    )
    if not lights:
        return (
            {"preset": "neutral_studio"},
            [
                {
                    "code": "illumination.generated",
                    "message": "The source had no lights; MeshProbe selected neutral_studio.",
                    "component_ids": [],
                }
            ],
        )

    converted: list[dict[str, Any]] = []
    for obj in lights:
        data = obj.data
        if data.energy <= 0 or not any(data.color):
            continue
        base: dict[str, Any] = {"id": obj.name, **light_color(data)}
        if data.type == "AREA":
            converted.append(
                {
                    **base,
                    "type": "area",
                    "position_mm": [
                        value * MILLIMETERS_PER_METER for value in obj.matrix_world.translation
                    ],
                    "orientation_xyzw": quaternion_xyzw(obj),
                    "power_w": data.energy,
                    "size_mm": data.size * MILLIMETERS_PER_METER,
                }
            )
            continue
        if data.type == "POINT":
            converted.append(
                {
                    **base,
                    "type": "point",
                    "position_mm": [
                        value * MILLIMETERS_PER_METER for value in obj.matrix_world.translation
                    ],
                    "power_w": data.energy,
                }
            )
            continue
        if data.type == "SPOT":
            converted.append(
                {
                    **base,
                    "type": "spot",
                    "position_mm": [
                        value * MILLIMETERS_PER_METER for value in obj.matrix_world.translation
                    ],
                    "orientation_xyzw": quaternion_xyzw(obj),
                    "power_w": data.energy,
                    "spot_size_degrees": math.degrees(data.spot_size),
                    "blend": data.spot_blend,
                }
            )
            continue
        if data.type == "SUN":
            converted.append(
                {
                    **base,
                    "type": "sun",
                    "orientation_xyzw": quaternion_xyzw(obj),
                    "strength": data.energy,
                    "angle_degrees": math.degrees(data.angle),
                }
            )

    if not converted:
        return (
            {"preset": "neutral_studio"},
            [
                {
                    "code": "illumination.generated",
                    "message": (
                        "The source had no positive-energy supported lights; MeshProbe selected "
                        "neutral_studio."
                    ),
                    "component_ids": [],
                }
            ],
        )

    world_color = bpy.context.scene.world.color if bpy.context.scene.world else (0.0, 0.0, 0.0)
    return (
        {
            "preset": "custom",
            "background_rgb": [float(channel) for channel in world_color],
            "background_strength": 1.0,
            "ambient_rgb": [1.0, 1.0, 1.0],
            "ambient_strength": 0.0,
            "lights": converted,
        },
        [],
    )


def require_session() -> dict[str, Any]:
    if MANIFEST is None:
        raise ValueError("no scene is open")
    return MANIFEST


def camera_object() -> bpy.types.Object:
    cameras = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"), key=lambda obj: obj.name
    )
    if cameras:
        active_camera = bpy.context.scene.camera
        camera = active_camera if active_camera in cameras else cameras[0]
        bpy.context.scene.camera = camera
        return camera
    data = bpy.data.cameras.new("MeshProbeCamera")
    camera = bpy.data.objects.new("MeshProbeCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def remember_perspective_sensor_intrinsics(projection: dict[str, Any]) -> None:
    """Record the physical sensor dimensions before aspect-ratio derivation."""
    global CURRENT_PERSPECTIVE_SENSOR_INTRINSICS
    if projection.get("mode") != "perspective":
        CURRENT_PERSPECTIVE_SENSOR_INTRINSICS = None
        return
    projection = normalize_camera_projection(projection)
    CURRENT_PERSPECTIVE_SENSOR_INTRINSICS = (
        projection["sensor_width_mm"],
        projection.get("sensor_height_mm", 24.0),
    )


def apply_camera(
    camera: dict[str, Any],
    focus_component_ids: list[str] | tuple[str, ...] = (),
    aspect_ratio: float = 1.0,
    target_mm: list[float] | tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    validate_camera_diagnostics_inputs(focus_component_ids, aspect_ratio)
    resolved_camera = deepcopy(camera)
    resolved_camera["projection"] = normalize_camera_projection(camera["projection"])
    obj = camera_object()
    pose = resolved_camera["pose"]
    position = Vector(value / MILLIMETERS_PER_METER for value in pose["position_mm"])
    x, y, z, w = pose["orientation_xyzw"]
    rotation = Quaternion((w, x, y, z)).normalized()
    if pose.get("frame", "world") == "source":
        source_to_world = source_to_world_matrix()
        position = source_to_world @ position
        rotation = source_to_world.to_quaternion() @ rotation
        resolved_camera["pose"] = {
            "position_mm": [value * MILLIMETERS_PER_METER for value in position],
            "orientation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
            "frame": "world",
        }
    elif pose.get("frame", "world") != "world":
        raise ValueError("camera pose frame must be source or world")
    else:
        resolved_camera["pose"]["frame"] = "world"
    obj.matrix_world = Matrix.Translation(position) @ rotation.to_matrix().to_4x4()
    projection = resolved_camera["projection"]
    data = obj.data
    data.clip_start = projection["near_clip_mm"] / MILLIMETERS_PER_METER
    data.clip_end = projection["far_clip_mm"] / MILLIMETERS_PER_METER
    mode = projection["mode"]
    if mode not in {"orthographic", "perspective"}:
        raise ValueError(f"unknown projection mode: {mode}")
    if mode == "orthographic":
        data.type = "ORTHO"
        data.ortho_scale = projection["scale_mm"] / MILLIMETERS_PER_METER
        data.dof.use_dof = False
    else:
        data.type = "PERSP"
        data.lens = projection["focal_length_mm"]
        data.sensor_width = projection["sensor_width_mm"]
        data.sensor_height = projection.get("sensor_height_mm", 24.0)
        sensor_fit = projection["sensor_fit"]
        if sensor_fit not in {"auto", "horizontal", "vertical"}:
            raise ValueError(f"unknown sensor fit: {sensor_fit}")
        data.sensor_fit = sensor_fit.upper()
        depth_of_field = projection.get("depth_of_field", {"mode": "disabled"})
        data.dof.use_dof = depth_of_field["mode"] == "enabled"
        data.dof.focus_object = None
        if data.dof.use_dof:
            data.dof.aperture_fstop = depth_of_field["aperture_fstop"]
            focus = depth_of_field.get("focus")
            if focus is not None:
                component_id = focus["component_id"]
                if component_id not in COMPONENT_OBJECTS:
                    raise ValueError(f"unknown depth-of-field component id: {component_id}")
                data.dof.focus_object = COMPONENT_OBJECTS[component_id]
            if depth_of_field.get("focus_distance_mm") is not None:
                data.dof.focus_distance = (
                    depth_of_field["focus_distance_mm"] / MILLIMETERS_PER_METER
                )
    global CURRENT_CAMERA, CURRENT_CAMERA_DIAGNOSTICS
    CURRENT_CAMERA = resolved_camera
    CURRENT_CAMERA_DIAGNOSTICS = camera_diagnostics(
        obj,
        focus_component_ids,
        aspect_ratio,
        target_mm,
    )
    orient_component_labels()
    return session_snapshot()


def _fit_reused_projection_to_scene(
    camera: dict[str, Any], position: Vector, rotation: Quaternion
) -> None:
    """Keep a reused projection's clip range valid after camera motion."""
    if MANIFEST is None:
        return
    bounds = MANIFEST["root_bounds"]
    forward = rotation @ Vector((0.0, 0.0, -1.0))
    depths = [
        (Vector((x, y, z)) / MILLIMETERS_PER_METER - position) @ forward
        for x in (bounds["minimum_mm"][0], bounds["maximum_mm"][0])
        for y in (bounds["minimum_mm"][1], bounds["maximum_mm"][1])
        for z in (bounds["minimum_mm"][2], bounds["maximum_mm"][2])
    ]
    visible_depths = [depth * MILLIMETERS_PER_METER for depth in depths if depth > 0]
    if not visible_depths:
        return
    projection = camera["projection"]
    projection["near_clip_mm"] = max(0.1, min(visible_depths) * 0.5)
    projection["far_clip_mm"] = max(projection["far_clip_mm"], max(visible_depths) * 1.5)


def orbit_camera(command: dict[str, Any]) -> dict[str, Any]:
    # Every dispatch path calls require_session() before orbit_camera, and
    # initialize_session always sets CURRENT_CAMERA (from the imported or a default
    # camera) in the same step it sets the manifest that require_session checks, so
    # CURRENT_CAMERA is guaranteed set here.
    if CURRENT_CAMERA is None:
        raise ValueError("session camera is not initialized")
    global CURRENT_CAMERA_OPERATION
    target = Vector(value / MILLIMETERS_PER_METER for value in command["target_mm"])
    azimuth = math.radians(command["azimuth_degrees"])
    elevation = math.radians(command["elevation_degrees"])
    distance = command["distance_mm"] / MILLIMETERS_PER_METER
    offset = Vector(
        (
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        )
    )
    position = target + offset
    direction = target - position
    if direction.length <= 1e-12:
        raise ValueError("orbit camera direction is degenerate")
    rotation = direction.to_track_quat("-Z", "Y").normalized()
    roll = math.radians(command.get("roll_degrees", 0.0))
    if roll:
        rotation = Quaternion(direction.normalized(), roll) @ rotation
    # No projection given: keep the session's current projection (mirrors
    # rotate_camera below), so a bare orbit doesn't force every caller to restate
    # camera intrinsics. A brand-new session's "current projection" is already a
    # standard 50mm perspective (imported_camera's fallback when the source has no
    # camera), so this alone covers "default to perspective on first use". See #96.
    requested_projection = command.get("projection")
    focus_component_ids = command.get("focus_component_ids", ())
    aspect_ratio = command.get("aspect_ratio", 1.0)
    validate_camera_diagnostics_inputs(focus_component_ids, aspect_ratio)
    projection = (
        requested_projection
        if requested_projection is not None
        else deepcopy(CURRENT_CAMERA["projection"])
    )
    if (
        requested_projection is None
        and "aspect_ratio" in command
        and projection["mode"] == "perspective"
    ):
        if CURRENT_PERSPECTIVE_SENSOR_INTRINSICS is None:
            remember_perspective_sensor_intrinsics(projection)
        assert CURRENT_PERSPECTIVE_SENSOR_INTRINSICS is not None
        projection["sensor_width_mm"], projection["sensor_height_mm"] = (
            CURRENT_PERSPECTIVE_SENSOR_INTRINSICS
        )
        sensor_fit = projection["sensor_fit"]
        effective_fit = (
            sensor_fit
            if sensor_fit != "auto"
            else ("horizontal" if aspect_ratio >= 1 else "vertical")
        )
        if effective_fit == "horizontal":
            projection["sensor_height_mm"] = projection["sensor_width_mm"] / aspect_ratio
        else:
            projection["sensor_width_mm"] = projection["sensor_height_mm"] * aspect_ratio
    camera = {
        "pose": {
            "position_mm": [value * MILLIMETERS_PER_METER for value in position],
            "orientation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
        },
        "projection": projection,
    }
    if requested_projection is None:
        _fit_reused_projection_to_scene(camera, position, rotation)
    apply_camera(
        camera,
        focus_component_ids,
        aspect_ratio,
        command["target_mm"],
    )
    # A supplied projection establishes the physical sensor pair that later bare
    # AUTO-fit orbits reuse.  Publish it only after ``apply_camera`` accepts the
    # complete camera request; otherwise a rejected orbit would poison the next
    # aspect-ratio derivation with intrinsics that never became session state.
    if requested_projection is not None:
        remember_perspective_sensor_intrinsics(projection)
    CURRENT_CAMERA_OPERATION = None
    return session_snapshot()


def move_camera(command: dict[str, Any]) -> dict[str, Any]:
    if CURRENT_CAMERA is None:
        raise ValueError("session camera is not initialized")
    validate_camera_diagnostics_inputs(
        command.get("focus_component_ids", ()), command.get("aspect_ratio", 1.0)
    )
    previous_pose = deepcopy(CURRENT_CAMERA["pose"])
    x, y, z, w = previous_pose["orientation_xyzw"]
    orientation = Quaternion((w, x, y, z))
    world_delta = Vector(command["world_delta_mm"])
    right, up, forward = command["camera_delta_mm"]
    world_delta += orientation @ Vector((right, up, -forward))
    position = Vector(previous_pose["position_mm"]) + world_delta
    resulting_pose = {
        **previous_pose,
        "position_mm": list(position),
    }
    camera = {
        "pose": resulting_pose,
        "projection": deepcopy(CURRENT_CAMERA["projection"]),
    }
    _fit_reused_projection_to_scene(
        camera,
        position / MILLIMETERS_PER_METER,
        orientation,
    )
    global CURRENT_CAMERA_OPERATION
    CURRENT_CAMERA_OPERATION = {
        "operation": "move",
        "requested_world_delta_mm": command["world_delta_mm"],
        "requested_camera_delta_mm": command["camera_delta_mm"],
        "resolved_world_delta_mm": list(world_delta),
        "previous_pose": previous_pose,
        "resulting_pose": resulting_pose,
    }
    return apply_camera(
        camera,
        command.get("focus_component_ids", ()),
        command.get("aspect_ratio", 1.0),
    )


def source_to_world_matrix() -> Matrix:
    frames = require_session()["coordinate_frames"]
    rows = tuple(
        tuple(frames["source_to_world"][row * 4 + column] for column in range(4))
        for row in range(4)
    )
    return Matrix(rows)


def transform_from_source(
    values: list[float] | tuple[float, float, float], *, point: bool
) -> Vector:
    matrix = source_to_world_matrix()
    vector = Vector(values)
    return matrix @ vector if point else matrix.to_3x3() @ vector


def validate_rotation_basis(value: Any) -> dict[str, list[float]]:
    defaults = {
        "x": [1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    }
    if not isinstance(value, dict) or set(value) - defaults.keys():
        raise ValueError("basis must contain only x, y, and z axes")

    basis: dict[str, list[float]] = {}
    for name, default in defaults.items():
        axis = value.get(name, default)
        if not isinstance(axis, (list, tuple)) or len(axis) != 3:
            raise ValueError("basis axes must contain three finite numbers")
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(component)
            for component in axis
        ):
            raise ValueError("basis axes must contain three finite numbers")
        basis[name] = [float(component) for component in axis]

    axes = (basis["x"], basis["y"], basis["z"])
    lengths = [math.sqrt(sum(component * component for component in axis)) for axis in axes]
    if any(not math.isclose(length, 1.0, abs_tol=1e-6) for length in lengths):
        raise ValueError("basis axes must have unit length")
    dot_products = [
        sum(left * right for left, right in zip(axes[first], axes[second], strict=True))
        for first, second in ((0, 1), (0, 2), (1, 2))
    ]
    if any(not math.isclose(dot, 0.0, abs_tol=1e-6) for dot in dot_products):
        raise ValueError("basis axes must be mutually perpendicular")
    cross_xy = (
        basis["x"][1] * basis["y"][2] - basis["x"][2] * basis["y"][1],
        basis["x"][2] * basis["y"][0] - basis["x"][0] * basis["y"][2],
        basis["x"][0] * basis["y"][1] - basis["x"][1] * basis["y"][0],
    )
    if any(
        not math.isclose(actual, expected, abs_tol=1e-6)
        for actual, expected in zip(cross_xy, basis["z"], strict=True)
    ):
        raise ValueError("basis must be right-handed (x cross y must equal z)")
    return basis


def rotate_camera(command: dict[str, Any]) -> dict[str, Any]:
    if CURRENT_CAMERA is None:
        raise ValueError("session camera is not initialized")
    focus_component_ids = command.get("focus_component_ids", ())
    aspect_ratio = command.get("aspect_ratio", 1.0)
    validate_camera_diagnostics_inputs(focus_component_ids, aspect_ratio)
    frame = command.get("frame", "source")
    if frame not in {"source", "world", "camera"}:
        raise ValueError("camera rotation frame must be source, world, or camera")
    degrees = command["degrees"]
    if isinstance(degrees, bool) or not isinstance(degrees, (int, float)):
        raise ValueError("degrees must be finite")
    if not math.isfinite(degrees):
        raise ValueError("degrees must be finite")
    basis = validate_rotation_basis(
        command.get(
            "basis",
            {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]},
        )
    )
    axis = Vector(basis[command["axis"]])
    target_mm = command["target_mm"]
    if not isinstance(target_mm, (list, tuple)) or len(target_mm) != 3:
        raise ValueError("target_mm must contain three finite numbers")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in target_mm
    ):
        raise ValueError("target_mm must contain three finite numbers")
    target = Vector(target_mm)
    if frame == "source":
        axis = transform_from_source(axis, point=False)
        target = transform_from_source(target, point=True)
    elif frame == "camera":
        cx, cy, cz, cw = CURRENT_CAMERA["pose"]["orientation_xyzw"]
        camera_orientation = Quaternion((cw, cx, cy, cz))
        axis = camera_orientation @ axis
        # The pivot is a camera-local point: place it relative to the camera pose so that
        # target (0, 0, 0) pivots at the camera itself (a zero-radius tilt/pan/roll).
        target = Vector(CURRENT_CAMERA["pose"]["position_mm"]) + camera_orientation @ target
    axis.normalize()
    camera_orbit_degrees = -degrees
    rotation = Quaternion(axis, math.radians(camera_orbit_degrees))
    previous_pose = deepcopy(CURRENT_CAMERA["pose"])
    position = Vector(previous_pose["position_mm"])
    x, y, z, w = previous_pose["orientation_xyzw"]
    orientation = rotation @ Quaternion((w, x, y, z))
    resulting_pose = {
        "position_mm": [round(value, 12) for value in target + rotation @ (position - target)],
        "orientation_xyzw": [
            round(value, 12)
            for value in (orientation.x, orientation.y, orientation.z, orientation.w)
        ],
        "frame": "world",
    }
    requested_projection = command.get("projection")
    projection = (
        deepcopy(CURRENT_CAMERA["projection"])
        if requested_projection is None
        else normalize_camera_projection(requested_projection)
    )
    if requested_projection is not None:
        remember_perspective_sensor_intrinsics(projection)
    camera = {
        "pose": resulting_pose,
        "projection": projection,
    }
    if requested_projection is None:
        position = Vector(resulting_pose["position_mm"]) / MILLIMETERS_PER_METER
        _fit_reused_projection_to_scene(camera, position, orientation)
    global CURRENT_CAMERA_OPERATION
    CURRENT_CAMERA_OPERATION = {
        "operation": "rotate",
        "frame": frame,
        "target_mm": command["target_mm"],
        "axis": command["axis"],
        "basis": basis,
        "axis_world": list(axis),
        "requested_visual_degrees": degrees,
        "applied_camera_orbit_degrees": camera_orbit_degrees,
        "previous_pose": previous_pose,
        "resulting_pose": resulting_pose,
    }
    return apply_camera(
        camera,
        focus_component_ids,
        aspect_ratio,
        list(target),
    )


def validate_camera_diagnostics_inputs(
    focus_component_ids: list[str] | tuple[str, ...],
    aspect_ratio: float,
) -> None:
    unknown = set(focus_component_ids) - COMPONENT_OBJECTS.keys()
    if unknown:
        raise ValueError(f"unknown focus component ids: {sorted(unknown)}")
    if not math.isfinite(aspect_ratio) or not 0.01 <= aspect_ratio <= 100:
        raise ValueError("aspect_ratio must be between 0.01 and 100")


def validate_positive_projection_number(projection: dict[str, Any], key: str) -> float:
    value = projection[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a positive finite number")
    return float(value)


def reject_unknown_fields(payload: dict[str, Any], allowed: set[str], contract: str) -> None:
    unknown = payload.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {contract} fields: {sorted(unknown)}")


def normalize_camera_projection(projection: dict[str, Any]) -> dict[str, Any]:
    mode = projection["mode"]
    if mode == "orthographic":
        reject_unknown_fields(
            projection,
            {"mode", "scale_mm", "near_clip_mm", "far_clip_mm"},
            "orthographic projection",
        )
        normalized = {
            "near_clip_mm": 0.5,
            "far_clip_mm": 100_000.0,
            **deepcopy(projection),
        }
        validate_camera_projection(normalized)
        return normalized
    if mode != "perspective":
        raise ValueError(f"unknown projection mode: {mode}")

    reject_unknown_fields(
        projection,
        {
            "mode",
            "focal_length_mm",
            "sensor_width_mm",
            "sensor_height_mm",
            "sensor_fit",
            "near_clip_mm",
            "far_clip_mm",
            "depth_of_field",
        },
        "perspective projection",
    )
    requested_depth_of_field = projection.get("depth_of_field", {})
    reject_unknown_fields(
        requested_depth_of_field,
        {"mode", "aperture_fstop", "focus_distance_mm", "focus"},
        "depth-of-field",
    )
    requested_focus = requested_depth_of_field.get("focus")
    if requested_focus is not None:
        reject_unknown_fields(requested_focus, {"component_id"}, "component focus")
    depth_of_field = {
        "mode": "disabled",
        "aperture_fstop": 2.8,
        "focus_distance_mm": None,
        "focus": None,
        **deepcopy(requested_depth_of_field),
    }
    normalized = {
        "mode": "perspective",
        "focal_length_mm": 50.0,
        "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0,
        "sensor_fit": "auto",
        "near_clip_mm": 0.5,
        "far_clip_mm": 100_000.0,
        **deepcopy(projection),
        "depth_of_field": depth_of_field,
    }
    validate_camera_projection(normalized)
    return normalized


def validate_camera_projection(projection: dict[str, Any]) -> None:
    mode = projection["mode"]
    if mode not in {"orthographic", "perspective"}:
        raise ValueError(f"unknown projection mode: {mode}")
    near_clip_mm = validate_positive_projection_number(projection, "near_clip_mm")
    far_clip_mm = validate_positive_projection_number(projection, "far_clip_mm")
    if far_clip_mm <= near_clip_mm:
        raise ValueError("far_clip_mm must be greater than near_clip_mm")
    if mode == "orthographic":
        validate_positive_projection_number(projection, "scale_mm")
        return

    validate_positive_projection_number(projection, "focal_length_mm")
    validate_positive_projection_number(projection, "sensor_width_mm")
    validate_positive_projection_number(projection, "sensor_height_mm")
    sensor_fit = projection["sensor_fit"]
    if sensor_fit not in {"auto", "horizontal", "vertical"}:
        raise ValueError(f"unknown sensor fit: {sensor_fit}")
    depth_of_field = projection.get("depth_of_field", {"mode": "disabled"})
    depth_of_field_mode = depth_of_field["mode"]
    if depth_of_field_mode not in {"disabled", "enabled"}:
        raise ValueError(f"unknown depth-of-field mode: {depth_of_field_mode}")
    focus = depth_of_field.get("focus")
    focus_distance_mm = depth_of_field.get("focus_distance_mm")
    if depth_of_field_mode == "disabled":
        if focus is not None or focus_distance_mm is not None:
            raise ValueError("disabled depth of field cannot declare a focus target")
        return
    validate_positive_projection_number(depth_of_field, "aperture_fstop")
    if (focus is None) == (focus_distance_mm is None):
        raise ValueError("enabled depth of field requires exactly one focus target")
    if focus is not None:
        component_id = focus["component_id"]
        if component_id not in COMPONENT_OBJECTS:
            raise ValueError(f"unknown depth-of-field component id: {component_id}")
        return
    validate_positive_projection_number(depth_of_field, "focus_distance_mm")


def camera_diagnostics(
    camera: bpy.types.Object,
    focus_component_ids: list[str] | tuple[str, ...],
    aspect_ratio: float,
    target_mm: list[float] | tuple[float, float, float] | None,
) -> dict[str, Any]:
    validate_camera_diagnostics_inputs(focus_component_ids, aspect_ratio)
    focus_ids = set(focus_component_ids)
    matrix = camera.matrix_world
    right = matrix.to_3x3().col[0].normalized()
    up = matrix.to_3x3().col[1].normalized()
    forward = -matrix.to_3x3().col[2].normalized()
    y_resolution = 1_000
    x_resolution = max(1, round(y_resolution * aspect_ratio))
    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=x_resolution,
        y=y_resolution,
        scale_x=1.0,
        scale_y=1.0,
    )
    inverse_projection = projection.inverted()
    frustum: list[list[float]] = []
    for clip_z in (-1.0, 1.0):
        for clip_y in (-1.0, 1.0):
            for clip_x in (-1.0, 1.0):
                local_homogeneous = inverse_projection @ Vector((clip_x, clip_y, clip_z, 1.0))
                local = local_homogeneous.xyz / local_homogeneous.w
                world = matrix @ local
                frustum.append([value * MILLIMETERS_PER_METER for value in world])

    if target_mm is None:
        target_mm = focus_center_mm(focus_ids) if focus_ids else scene_center_mm()
    target_world = Vector(value / MILLIMETERS_PER_METER for value in target_mm)
    target_camera = matrix.inverted() @ target_world
    projected = {
        component_id: projected_component_bounds(
            camera,
            projection,
            COMPONENT_OBJECTS[component_id],
        )
        for component_id in sorted(focus_ids)
    }
    horizontal_fov, vertical_fov = camera_fov_degrees(camera.data, aspect_ratio)
    return {
        "aspect_ratio": aspect_ratio,
        "horizontal_fov_degrees": horizontal_fov,
        "vertical_fov_degrees": vertical_fov,
        "right": list(right),
        "up": list(up),
        "forward": list(forward),
        "frustum_corners_mm": frustum,
        "target_depth_mm": -target_camera.z * MILLIMETERS_PER_METER,
        "projected_bounds": projected,
    }


def camera_fov_degrees(
    camera: bpy.types.Camera,
    aspect_ratio: float,
) -> tuple[float | None, float | None]:
    if camera.type == "ORTHO":
        return None, None
    sensor_fit = camera.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if aspect_ratio >= 1 else "VERTICAL"
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    if sensor_fit == "VERTICAL":
        sensor_width = sensor_height * aspect_ratio
    else:
        sensor_height = sensor_width / aspect_ratio
    horizontal = math.degrees(2 * math.atan(sensor_width / (2 * camera.lens)))
    vertical = math.degrees(2 * math.atan(sensor_height / (2 * camera.lens)))
    return horizontal, vertical


def scene_center_mm() -> list[float]:
    manifest = require_session()
    minimum = manifest["root_bounds"]["minimum_mm"]
    maximum = manifest["root_bounds"]["maximum_mm"]
    return [(low + high) / 2 for low, high in zip(minimum, maximum, strict=True)]


def focus_center_mm(focus_ids: set[str]) -> list[float]:
    components = {component["id"]: component for component in require_session()["components"]}
    minimum = [
        min(
            components[component_id]["world_bounds"]["minimum_mm"][axis]
            for component_id in focus_ids
        )
        for axis in range(3)
    ]
    maximum = [
        max(
            components[component_id]["world_bounds"]["maximum_mm"][axis]
            for component_id in focus_ids
        )
        for axis in range(3)
    ]
    return [(low + high) / 2 for low, high in zip(minimum, maximum, strict=True)]


def projected_component_bounds(
    camera: bpy.types.Object,
    projection: Matrix,
    obj: bpy.types.Object,
) -> dict[str, Any]:
    world_to_camera = camera.matrix_world.inverted()
    points = [world_to_camera @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    depths = [-point.z * MILLIMETERS_PER_METER for point in points]
    in_front = [depth > 0 for depth in depths]
    if any(in_front) and not all(in_front):
        return {
            "projection_status": "crosses_camera_plane",
            "minimum_image_xy": None,
            "maximum_image_xy": None,
            "minimum_depth_mm": min(depths),
            "maximum_depth_mm": max(depths),
        }
    image_points: list[tuple[float, float]] = []
    for point in points:
        clip = projection @ point.to_4d()
        if abs(clip.w) <= 1e-12:
            return {
                "projection_status": "crosses_camera_plane",
                "minimum_image_xy": None,
                "maximum_image_xy": None,
                "minimum_depth_mm": min(depths),
                "maximum_depth_mm": max(depths),
            }
        image_points.append(((clip.x / clip.w + 1) / 2, (clip.y / clip.w + 1) / 2))
    return {
        "projection_status": "in_front" if all(in_front) else "behind",
        "minimum_image_xy": [
            min(point[0] for point in image_points),
            min(point[1] for point in image_points),
        ],
        "maximum_image_xy": [
            max(point[0] for point in image_points),
            max(point[1] for point in image_points),
        ],
        "minimum_depth_mm": min(depths),
        "maximum_depth_mm": max(depths),
    }


def linear_rgb_from_temperature(kelvin: float) -> list[float]:
    temperature = kelvin / 100.0
    if temperature <= 66:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = (
            0.0
            if temperature <= 19
            else 138.5177312231 * math.log(temperature - 10) - 305.0447927307
        )
    else:
        red = 329.698727446 * ((temperature - 60) ** -0.1332047592)
        green = 288.1221695283 * ((temperature - 60) ** -0.0755148492)
        blue = 255.0

    def to_linear(channel: float) -> float:
        srgb = min(255.0, max(0.0, channel)) / 255.0
        if srgb <= 0.04045:
            return srgb / 12.92
        return float(((srgb + 0.055) / 1.055) ** 2.4)

    return [to_linear(red), to_linear(green), to_linear(blue)]


def light_rgb(spec: dict[str, Any]) -> list[float]:
    if spec.get("linear_rgb") is not None:
        return list(spec["linear_rgb"])
    return linear_rgb_from_temperature(spec["color_temperature_k"])


def clear_lights() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type != "LIGHT":
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data.users == 0:
            bpy.data.lights.remove(data)


def scene_center_and_span() -> tuple[Vector, float]:
    manifest = require_session()
    minimum = manifest["root_bounds"]["minimum_mm"]
    maximum = manifest["root_bounds"]["maximum_mm"]
    center = Vector((low + high) / 2 for low, high in zip(minimum, maximum, strict=True))
    span = max(high - low for low, high in zip(minimum, maximum, strict=True))
    return center, max(span, 100.0)


def scene_frame() -> tuple[Vector, Vector, Vector]:
    manifest = require_session()
    roots = [component for component in manifest["components"] if component["parent_id"] is None]
    if len(roots) != 1:
        return Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    transform = roots[0]["world_transform"]
    axes = tuple(
        Vector((transform[index], transform[4 + index], transform[8 + index])).normalized()
        for index in range(3)
    )
    if any(axis.length_squared == 0 for axis in axes):
        return Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    return axes


def frame_offset(
    frame: tuple[Vector, Vector, Vector], values: tuple[float, float, float]
) -> Vector:
    return sum((axis * value for axis, value in zip(frame, values, strict=True)), Vector())


def look_orientation(position_mm: Vector, target_mm: Vector) -> list[float]:
    rotation = (target_mm - position_mm).to_track_quat("-Z", "Y").normalized()
    return [rotation.x, rotation.y, rotation.z, rotation.w]


def preset_lights(preset: str) -> dict[str, Any]:
    center, span = scene_center_and_span()
    frame = scene_frame()
    power_scale = (span / PRESET_REFERENCE_SPAN_MM) ** 2
    background = [0.03, 0.03, 0.03]
    ambient = 0.15
    definitions: list[tuple[str, Vector, float, float]] = []
    if preset == "neutral_studio":
        # A dark backdrop with almost no ambient fill left the default preset for
        # light-less sources near-black from every angle (issue #53): only the faces
        # the three-point rig hit directly caught any light. Lift the neutral backdrop
        # and ambient fill and strengthen the rig so the default is legible at any
        # orbit while still reading softer than high_key.
        background = [0.10, 0.10, 0.10]
        ambient = 0.5
        definitions = [
            ("key", center + frame_offset(frame, (span, -span, span)), 1_400.0, span),
            (
                "fill",
                center + frame_offset(frame, (-span, -span / 2, span / 2)),
                800.0,
                span,
            ),
            ("rim", center + frame_offset(frame, (0, span, span)), 900.0, span / 2),
        ]
    elif preset == "high_key":
        background = [0.25, 0.25, 0.25]
        ambient = 0.45
        definitions = [
            (
                "key",
                center + frame_offset(frame, (span, -span, span)),
                1_200.0,
                span * 1.5,
            ),
            (
                "fill",
                center + frame_offset(frame, (-span, -span, span)),
                1_000.0,
                span * 1.5,
            ),
            (
                "under",
                center + frame_offset(frame, (0, 0, -span)),
                700.0,
                span * 1.5,
            ),
        ]
    elif preset in {"raking_left", "raking_right"}:
        side = -1 if preset == "raking_left" else 1
        definitions = [
            (
                "rake",
                center + frame_offset(frame, (side * span * 2, -span / 4, span / 8)),
                1_000.0,
                span / 3,
            )
        ]
    elif preset == "backlit":
        definitions = [("back", center + frame_offset(frame, (0, span * 2, 0)), 1_200.0, span)]
    elif preset == "flat_diagnostic":
        background = [0.18, 0.18, 0.18]
        ambient = 0.25
        definitions = [
            (
                f"axis-{index}",
                center + frame_offset(frame, tuple(direction * span)),
                500.0,
                span * 2,
            )
            for index, direction in enumerate(
                (
                    Vector((1, 0, 0)),
                    Vector((-1, 0, 0)),
                    Vector((0, 1, 0)),
                    Vector((0, -1, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 0, -1)),
                ),
                start=1,
            )
        ]
    else:
        raise ValueError(f"unknown illumination preset: {preset}")
    lights = [
        {
            "id": light_id,
            "type": "area",
            "position_mm": list(position),
            "orientation_xyzw": look_orientation(position, center),
            "power_w": power * power_scale,
            "linear_rgb": [1.0, 1.0, 1.0],
            "size_mm": max(size, 1.0),
        }
        for light_id, position, power, size in definitions
    ]
    return {
        "preset": "custom",
        "background_rgb": background,
        "background_strength": ambient,
        "ambient_rgb": background,
        "ambient_strength": ambient,
        "lights": lights,
        "visible_background_mode": "color",
    }


def configure_world(
    background_rgb: list[float],
    background_strength: float,
    ambient_rgb: list[float],
    ambient_strength: float,
    environment_map: dict[str, Any] | None,
    visible_background_mode: str,
) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("MeshProbeWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    visible = nodes.new("ShaderNodeBackground")
    visible.name = "MeshProbeVisibleBackground"
    visible.inputs["Color"].default_value = (*background_rgb, 1.0)
    visible.inputs["Strength"].default_value = background_strength
    ambient = nodes.new("ShaderNodeBackground")
    ambient.name = "MeshProbeAmbient"
    ambient.inputs["Color"].default_value = (*ambient_rgb, 1.0)
    ambient.inputs["Strength"].default_value = ambient_strength
    links = world.node_tree.links
    environment_output = ambient.outputs["Background"]
    if environment_map is not None:
        path = Path(environment_map["path"]).expanduser().resolve(strict=True)
        if sha256_file(path) != environment_map["sha256"]:
            raise ValueError("environment map hash does not match its declared content")
        texture_coordinates = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.vector_type = "POINT"
        mapping.inputs["Rotation"].default_value[2] = math.radians(
            environment_map["rotation_degrees"]
        )
        texture = nodes.new("ShaderNodeTexEnvironment")
        texture.name = "MeshProbeEnvironment"
        texture.projection = "EQUIRECTANGULAR"
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = f"MeshProbeEnvironment-{environment_map['sha256'][:16]}"
        texture.image = image
        environment = nodes.new("ShaderNodeBackground")
        environment.name = "MeshProbeEnvironmentBackground"
        environment.inputs["Strength"].default_value = environment_map["strength"]
        add = nodes.new("ShaderNodeAddShader")
        add.name = "MeshProbeEnvironmentLighting"
        links.new(texture_coordinates.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], environment.inputs["Color"])
        links.new(ambient.outputs["Background"], add.inputs[0])
        links.new(environment.outputs["Background"], add.inputs[1])
        environment_output = add.outputs["Shader"]
    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    mix.name = "MeshProbeVisibleBackgroundMix"
    camera_output = (
        environment_output
        if visible_background_mode == "environment"
        else visible.outputs["Background"]
    )
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(environment_output, mix.inputs[1])
    links.new(camera_output, mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def create_light(spec: dict[str, Any]) -> None:
    blender_type = {"area": "AREA", "point": "POINT", "spot": "SPOT", "sun": "SUN"}[spec["type"]]
    data = bpy.data.lights.new(f"MeshProbe-{spec['id']}", blender_type)
    data.color = light_rgb(spec)
    obj = bpy.data.objects.new(f"MeshProbe-{spec['id']}", data)
    bpy.context.scene.collection.objects.link(obj)
    if spec.get("position_mm") is not None:
        obj.location = [value / MILLIMETERS_PER_METER for value in spec["position_mm"]]
    if spec.get("orientation_xyzw") is not None:
        x, y, z, w = spec["orientation_xyzw"]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Quaternion((w, x, y, z)).normalized()
    if spec["type"] == "area":
        data.energy = spec["power_w"]
        data.shape = "DISK"
        data.size = spec["size_mm"] / MILLIMETERS_PER_METER
    elif spec["type"] == "point":
        data.energy = spec["power_w"]
    elif spec["type"] == "spot":
        data.energy = spec["power_w"]
        data.spot_size = math.radians(spec["spot_size_degrees"])
        data.spot_blend = spec["blend"]
    else:
        data.energy = spec["strength"]
        data.angle = math.radians(spec["angle_degrees"])


def custom_illumination_has_effective_output(illumination: dict[str, Any]) -> bool:
    if illumination["ambient_strength"] > 0 and any(illumination["ambient_rgb"]):
        return True
    environment_map = illumination.get("environment_map")
    if environment_map is not None and environment_map["strength"] > 0:
        return True
    for light in illumination["lights"]:
        intensity = light["strength"] if light["type"] == "sun" else light["power_w"]
        color_contributes = light.get("color_temperature_k") is not None or any(
            light.get("linear_rgb") or ()
        )
        if intensity > 0 and color_contributes:
            return True
    return False


def apply_illumination(illumination: dict[str, Any]) -> dict[str, Any]:
    if illumination["preset"] == "custom":
        resolved = deepcopy(illumination)
        resolved.setdefault("background_strength", resolved["ambient_strength"])
        resolved.setdefault("ambient_rgb", deepcopy(resolved["background_rgb"]))
        resolved.setdefault("lights", [])
        resolved.setdefault("environment_map", None)
        resolved.setdefault(
            "visible_background_mode",
            "environment" if resolved.get("environment_map") is not None else "color",
        )
        if resolved["visible_background_mode"] not in {"color", "environment"}:
            raise ValueError("unknown visible background mode")
        if (
            resolved["visible_background_mode"] == "environment"
            and resolved.get("environment_map") is None
        ):
            raise ValueError("environment visible background requires an environment map")
        if not custom_illumination_has_effective_output(resolved):
            raise ValueError("custom illumination must have non-zero light output")
        runtime = resolved
    else:
        runtime = preset_lights(illumination["preset"])
        srgb_override = illumination.get("background_srgb")
        background_override = illumination.get("background_rgb")
        strength_override = illumination.get("background_strength")
        if background_override is not None:
            runtime["background_rgb"] = background_override
            runtime["background_strength"] = 1.0
        if strength_override is not None:
            runtime["background_strength"] = strength_override
        if srgb_override is None:
            resolved = {
                **deepcopy(illumination),
                "background_rgb": deepcopy(runtime["background_rgb"]),
                "background_strength": runtime["background_strength"],
            }
        else:
            # A display-referred backdrop composites the exact color, so the resolved
            # snapshot carries only background_srgb (no shadowed linear override).
            resolved = {
                **deepcopy(illumination),
                "background_rgb": None,
                "background_strength": None,
            }
    # Normalize away an absent-vs-explicit-null difference: an imported preset omits the
    # key, an explicitly re-applied one carries background_srgb=None. Drop the null so both
    # hash to the same state_sha256 in session_snapshot.
    if resolved.get("background_srgb") is None:
        resolved.pop("background_srgb", None)
    clear_lights()
    configure_world(
        runtime["background_rgb"],
        runtime["background_strength"],
        runtime["ambient_rgb"],
        runtime["ambient_strength"],
        runtime.get("environment_map"),
        runtime["visible_background_mode"],
    )
    for light in runtime["lights"]:
        create_light(light)
    global CURRENT_ILLUMINATION
    CURRENT_ILLUMINATION = resolved
    return session_snapshot()


def restore_mesh(component_id: str) -> None:
    obj = COMPONENT_OBJECTS[component_id]
    original = ORIGINAL_MESHES[component_id]
    if obj.data is original:
        return
    replacement = obj.data
    obj.data = original
    if replacement.users == 0:
        bpy.data.meshes.remove(replacement)


def replacement_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = OVERRIDE_MATERIALS.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        OVERRIDE_MATERIALS[name] = material
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Alpha"].default_value = color[3]
    if color[3] < 1 and hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    return material


def replace_component_material(component_id: str, material: bpy.types.Material) -> None:
    obj = COMPONENT_OBJECTS[component_id]
    restore_mesh(component_id)
    obj.data = obj.data.copy()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0


def clear_component_label(component_id: str) -> None:
    label = MARK_OBJECTS.pop(component_id, None)
    if label is None:
        return
    data = label.data
    bpy.data.objects.remove(label, do_unlink=True)
    if data.users == 0:
        bpy.data.curves.remove(data)


def mark_rgba(mode: str, color: str | None) -> tuple[float, float, float, float]:
    if color is None:
        return MARK_COLORS[mode]
    return (
        srgb_channel_to_linear(int(color[1:3], 16)),
        srgb_channel_to_linear(int(color[3:5], 16)),
        srgb_channel_to_linear(int(color[5:7], 16)),
        1.0,
    )


def create_component_label(component_id: str, color: tuple[float, float, float, float]) -> None:
    component = next(item for item in require_session()["components"] if item["id"] == component_id)
    bounds = component["world_bounds"]
    minimum = bounds["minimum_mm"]
    maximum = bounds["maximum_mm"]
    _, scene_span = scene_center_and_span()
    data = bpy.data.curves.new(f"MeshProbeLabel-{component_id}", "FONT")
    data.body = component["display_name"]
    data.align_x = "CENTER"
    data.align_y = "BOTTOM"
    data.size = max(scene_span / MILLIMETERS_PER_METER / 30.0, 0.002)
    data.extrude = data.size * 0.01
    label = bpy.data.objects.new(f"MeshProbeLabel-{component_id}", data)
    bpy.context.scene.collection.objects.link(label)
    label.location = (
        (minimum[0] + maximum[0]) / 2 / MILLIMETERS_PER_METER,
        (minimum[1] + maximum[1]) / 2 / MILLIMETERS_PER_METER,
        (maximum[2] + scene_span * 0.02) / MILLIMETERS_PER_METER,
    )
    # Keep the semantic world-space anchor separately: the rendered text is projected onto a
    # camera-facing overlay plane below so dense surrounding geometry cannot hide it. Camera
    # moves recompute that projection from this stable anchor rather than compounding transforms.
    label["meshprobe_anchor_m"] = list(label.location)
    label.rotation_mode = "QUATERNION"
    color_suffix = COMPONENT_STATES[component_id]["mark_color"]
    material_name = "MeshProbeMark-labeled"
    if color_suffix is not None:
        material_name = f"{material_name}-{color_suffix[1:]}"
    label_material = emission_material(f"{material_name}-overlay", color)
    if hasattr(label_material, "surface_render_method"):
        # Labels render only in a transparent annotation pass. Blended surfaces preserve their
        # anti-aliased edge alpha when that pass is composited over the physical scene render.
        label_material.surface_render_method = "BLENDED"
    data.materials.append(label_material)
    MARK_OBJECTS[component_id] = label
    orient_component_labels()


def component_label_overlay_transform(
    camera: bpy.types.Object,
    anchor: Vector,
    basis: tuple[Vector, Vector, Vector, Vector],
) -> tuple[Vector, float] | None:
    position, right, up, forward = basis
    near = camera.data.clip_start
    far = camera.data.clip_end
    offset = anchor - position
    depth = offset @ forward
    if not near < depth < far:
        return None
    # Prefer a small gap behind the near plane, but never consume more than one percent of
    # the clip interval or half the distance to the anchor. The overlay therefore stays
    # strictly inside the clip range and strictly in front of every valid semantic anchor.
    preferred_clearance = max(near * 0.1, 0.0001)
    overlay_depth = near + min(
        preferred_clearance,
        (far - near) * 0.01,
        (depth - near) * 0.5,
    )
    horizontal = offset @ right
    vertical = offset @ up
    scale = overlay_depth / depth if camera.data.type == "PERSP" else 1.0
    location = (
        position + forward * overlay_depth + right * horizontal * scale + up * vertical * scale
    )
    return location, scale


def orient_component_labels() -> None:
    camera = camera_object()
    orientation = camera.matrix_world.to_quaternion()
    position = camera.matrix_world.translation
    right = orientation @ Vector((1.0, 0.0, 0.0))
    up = orientation @ Vector((0.0, 1.0, 0.0))
    forward = orientation @ Vector((0.0, 0.0, -1.0))
    basis = (position, right, up, forward)
    for label in MARK_OBJECTS.values():
        label.rotation_quaternion = orientation
        anchor = Vector(label["meshprobe_anchor_m"])
        transform = component_label_overlay_transform(camera, anchor, basis)
        label.hide_render = transform is None
        if transform is None:
            # A clipped semantic anchor has no meaningful annotation in this frame. Keep its
            # world transform deterministic so a later camera move can project it again.
            label.location = anchor
            label.scale = (1.0, 1.0, 1.0)
            continue
        label.location, scale = transform
        label.scale = (scale, scale, scale)


def refresh_component_appearance(component_id: str) -> None:
    restore_mesh(component_id)
    clear_component_label(component_id)
    state = COMPONENT_STATES[component_id]
    if state["display"] == "hidden":
        return
    mark = state["mark"]
    if mark != "unmarked":
        color = mark_rgba(mark, state["mark_color"])
        material_name = f"MeshProbeMark-{mark}"
        if state["mark_color"] is not None:
            material_name = f"{material_name}-{state['mark_color'][1:]}"
        replace_component_material(component_id, replacement_material(material_name, color))
        if mark == "labeled":
            create_component_label(component_id, color)
        return
    if state["display"] == "ghosted":
        replace_component_material(
            component_id, replacement_material("MeshProbeGhost", (0.35, 0.65, 1.0, 0.2))
        )


def set_component_visibility(component_id: str, mode: str) -> None:
    obj = COMPONENT_OBJECTS[component_id]
    hidden = mode == "hidden"
    obj.hide_render = hidden
    obj.hide_viewport = hidden
    COMPONENT_STATES[component_id]["display"] = mode
    refresh_component_appearance(component_id)


def selected_component_ids(command: dict[str, Any]) -> set[str]:
    selected = set(command["component_ids"])
    if not selected:
        raise ValueError("at least one component id is required")
    unknown = selected - COMPONENT_OBJECTS.keys()
    if unknown:
        raise ValueError(f"unknown component ids: {sorted(unknown)}")
    return selected


def component_display(command: dict[str, Any]) -> dict[str, Any]:
    selected = selected_component_ids(command)
    mode = command["mode"]
    if mode not in DISPLAY_MODES:
        raise ValueError(f"unknown display mode: {mode}")
    if mode == "isolated":
        operation = command.get("isolation_operation") or "replace"
        if operation not in {"replace", "add", "remove"}:
            raise ValueError(f"unknown isolation operation: {operation}")
        current = {
            component_id
            for component_id, state in COMPONENT_STATES.items()
            if state["display"] == "isolated"
        }
        if operation == "add":
            selected |= current
        if operation == "remove":
            selected = current - selected
        for component_id in COMPONENT_OBJECTS:
            set_component_visibility(
                component_id, "isolated" if component_id in selected else "hidden"
            )
        return session_snapshot()
    for component_id in selected:
        set_component_visibility(component_id, mode)
    return session_snapshot()


def component_mark(command: dict[str, Any]) -> dict[str, Any]:
    selected = selected_component_ids(command)
    mode = command["mode"]
    if mode not in MARK_MODES:
        raise ValueError(f"unknown mark mode: {mode}")
    color = command.get("color")
    if color is not None and mode == "unmarked":
        raise ValueError("color requires a visible mark mode")
    if color is not None and (
        not isinstance(color, str) or MARK_COLOR_PATTERN.fullmatch(color) is None
    ):
        raise ValueError("color must use sRGB #RRGGBB form")
    canonical_color = color.lower() if color is not None else None
    for component_id in selected:
        COMPONENT_STATES[component_id]["mark"] = mode
        COMPONENT_STATES[component_id]["mark_color"] = canonical_color
        refresh_component_appearance(component_id)
    return session_snapshot()


def session_snapshot() -> dict[str, Any]:
    if CURRENT_CAMERA is None or CURRENT_CAMERA_DIAGNOSTICS is None or CURRENT_ILLUMINATION is None:
        raise ValueError("session state is not initialized")
    state = {
        "camera": CURRENT_CAMERA,
        "illumination": CURRENT_ILLUMINATION,
        "components": {key: COMPONENT_STATES[key] for key in sorted(COMPONENT_STATES)},
    }
    hashed_state = deepcopy(state)
    environment = hashed_state["illumination"].get("environment_map")
    if environment is not None:
        environment.pop("path", None)
    canonical = json.dumps(hashed_state, sort_keys=True, separators=(",", ":")).encode()
    return {
        **deepcopy(state),
        "camera_diagnostics": deepcopy(CURRENT_CAMERA_DIAGNOSTICS),
        "state_sha256": hashlib.sha256(canonical).hexdigest(),
        "camera_operation": deepcopy(CURRENT_CAMERA_OPERATION),
    }


def reset_session(command: dict[str, Any]) -> dict[str, Any]:
    if IMPORTED_CAMERA is None or IMPORTED_ILLUMINATION is None:
        raise ValueError("session state is not initialized")
    global CURRENT_CAMERA_OPERATION, DEFAULT_ASPECT_RATIO
    aspect_ratio = command.get("aspect_ratio")
    if aspect_ratio is None:
        aspect_ratio = DEFAULT_ASPECT_RATIO
    validate_camera_diagnostics_inputs((), aspect_ratio)
    CURRENT_CAMERA_OPERATION = None
    if command.get("aspect_ratio") is not None:
        DEFAULT_ASPECT_RATIO = aspect_ratio
    for component_id, obj in COMPONENT_OBJECTS.items():
        restore_mesh(component_id)
        clear_component_label(component_id)
        hide_render, hide_viewport = ORIGINAL_VISIBILITY[component_id]
        obj.hide_render = hide_render
        obj.hide_viewport = hide_viewport
        COMPONENT_STATES[component_id] = {
            "display": "hidden" if hide_render else "shown",
            "mark": "unmarked",
            "mark_color": None,
        }
    default_camera = framed_default_camera(
        IMPORTED_CAMERA,
        visible_root_bounds(require_session()["root_bounds"]),
        aspect_ratio=aspect_ratio,
    )
    projection = default_camera.get("projection")
    if isinstance(projection, dict):
        remember_perspective_sensor_intrinsics(projection)
    apply_camera(
        default_camera,
        aspect_ratio=aspect_ratio,
    )
    apply_illumination(deepcopy(IMPORTED_ILLUMINATION))
    # Blender 4.2 cannot use the GPU compositor-backed default screen_edges
    # style. Resetting a session must remain available for the supported
    # shaded/contact-sheet flows without changing the modern runtime default.
    reset_style = "shaded" if uses_software_compatibility_mode(bpy.app.version) else "screen_edges"
    configure_render_style({"style": reset_style})
    return session_snapshot()


def runtime_diagnostics() -> dict[str, Any]:
    current_illumination = CURRENT_ILLUMINATION
    if current_illumination is None:
        raise RuntimeError("no active illumination")
    camera = camera_object()
    world_nodes = bpy.context.scene.world.node_tree.nodes
    environment = world_nodes.get("MeshProbeEnvironment")
    visible_background = world_nodes.get("MeshProbeVisibleBackground")
    ambient_background = world_nodes.get("MeshProbeAmbient")
    visible_background_mix = world_nodes.get("MeshProbeVisibleBackgroundMix")
    camera_background_source = visible_background_mix.inputs[2].links[0].from_node.name
    return {
        "components": {
            component_id: {
                "hide_render": obj.hide_render,
                "hide_viewport": obj.hide_viewport,
                "mesh_name": obj.data.name,
                "materials": [material.name for material in obj.data.materials],
                "material_colors": [
                    list(material.diffuse_color) for material in obj.data.materials
                ],
                "label": MARK_OBJECTS[component_id].name if component_id in MARK_OBJECTS else None,
                "label_rotation_wxyz": (
                    list(MARK_OBJECTS[component_id].rotation_quaternion)
                    if component_id in MARK_OBJECTS
                    else None
                ),
                "label_location_m": (
                    list(MARK_OBJECTS[component_id].location)
                    if component_id in MARK_OBJECTS
                    else None
                ),
                "label_hide_render": (
                    MARK_OBJECTS[component_id].hide_render if component_id in MARK_OBJECTS else None
                ),
            }
            for component_id, obj in COMPONENT_OBJECTS.items()
        },
        "camera": {
            "type": camera.data.type,
            "lens": camera.data.lens,
            "sensor_fit": camera.data.sensor_fit,
            "ortho_scale_mm": camera.data.ortho_scale * MILLIMETERS_PER_METER,
            "depth_of_field": resolved_depth_of_field(),
        },
        "lights": sorted(obj.name for obj in bpy.context.scene.objects if obj.type == "LIGHT"),
        "environment_map": (
            {
                "path": environment.image.filepath,
                "projection": environment.projection,
            }
            if environment is not None and environment.image is not None
            else None
        ),
        "world": {
            "visible_background_mode": current_illumination.get(
                "visible_background_mode",
                "color",
            ),
            "camera_background_source": camera_background_source,
            "background_rgb": list(visible_background.inputs["Color"].default_value[:3]),
            "background_strength": float(visible_background.inputs["Strength"].default_value),
            "background_srgb": current_illumination.get("background_srgb"),
            "ambient_rgb": list(ambient_background.inputs["Color"].default_value[:3]),
            "ambient_strength": float(ambient_background.inputs["Strength"].default_value),
        },
        "render": {
            "engine": bpy.context.scene.render.engine,
            "eevee_samples": eevee_render_samples(),
            "use_freestyle": bpy.context.scene.render.use_freestyle,
        },
    }


def configure_render(command: dict[str, Any]) -> str:
    scene = bpy.context.scene
    engine = command["engine"]
    scene.render.engine = eevee_engine() if engine == "eevee" else "CYCLES"
    scene.render.resolution_x = command["width"]
    scene.render.resolution_y = command["height"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_file_extension = False
    if engine == "cycles":
        configure_cycles_gpu(command["samples"])
        return "cuda"
    platform = graphics_platform()
    if command["graphics_policy"] == "hardware_required" and platform["device_class"] != "hardware":
        raise RuntimeError(
            "hardware graphics required, but Blender initialized "
            f"{platform['renderer']} as {platform['device_class']}"
        )
    configure_eevee_samples(command["samples"])
    return f"graphics_{platform['device_class']}"


@contextmanager
def temporary_render_style() -> Iterator[None]:
    scene = bpy.context.scene
    freestyle = bpy.context.view_layer.freestyle_settings
    line_set = freestyle.linesets[0]
    selectors = (
        "select_silhouette",
        "select_border",
        "select_crease",
        "select_material_boundary",
        "select_contour",
        "select_external_contour",
    )
    original_use_freestyle = scene.render.use_freestyle
    original_crease_angle = freestyle.crease_angle
    original_material_boundaries = freestyle.use_material_boundaries
    original_visibility_selection = line_set.select_by_visibility
    original_visibility = line_set.visibility
    original_edge_selection = line_set.select_by_edge_types
    original_selectors = {attribute: getattr(line_set, attribute) for attribute in selectors}
    original_edge_mark = line_set.select_edge_mark
    original_ridge_valley = line_set.select_ridge_valley
    original_suggestive_contour = line_set.select_suggestive_contour
    original_line_style = line_set.linestyle
    original_color = tuple(original_line_style.color) if original_line_style is not None else None
    original_thickness = original_line_style.thickness if original_line_style is not None else None
    try:
        yield
    finally:
        scene.render.use_freestyle = original_use_freestyle
        freestyle.crease_angle = original_crease_angle
        freestyle.use_material_boundaries = original_material_boundaries
        line_set.select_by_visibility = original_visibility_selection
        line_set.visibility = original_visibility
        line_set.select_by_edge_types = original_edge_selection
        for attribute, value in original_selectors.items():
            setattr(line_set, attribute, value)
        line_set.select_edge_mark = original_edge_mark
        line_set.select_ridge_valley = original_ridge_valley
        line_set.select_suggestive_contour = original_suggestive_contour
        if original_line_style is not None:
            original_line_style.color = original_color
            original_line_style.thickness = original_thickness


def configure_render_style(command: dict[str, Any]) -> None:
    style = command.get("style", "screen_edges")
    if style == "screen_edges" and uses_software_compatibility_mode(bpy.app.version):
        raise RuntimeError(
            "screen_edges requires Blender 5.2 or newer; use shaded or shaded_edges "
            "with Blender 4.2 software compatibility mode"
        )
    scene = bpy.context.scene
    scene.render.use_freestyle = style == "shaded_edges"
    if style in {"shaded", "screen_edges"}:
        return
    if style != "shaded_edges":
        raise ValueError(f"unknown render style: {style}")
    settings = command["shaded_edges"]
    freestyle = bpy.context.view_layer.freestyle_settings
    freestyle.crease_angle = math.radians(settings["crease_angle_degrees"])
    freestyle.use_material_boundaries = "material_boundary" in settings["edge_types"]
    line_set = freestyle.linesets[0]
    line_set.select_by_visibility = True
    line_set.visibility = "VISIBLE"
    line_set.select_by_edge_types = True
    selectors = {
        "silhouette": "select_silhouette",
        "border": "select_border",
        "crease": "select_crease",
        "material_boundary": "select_material_boundary",
        "contour": "select_contour",
        "external_contour": "select_external_contour",
    }
    selected = set(settings["edge_types"])
    for edge_type, attribute in selectors.items():
        setattr(line_set, attribute, edge_type in selected)
    line_set.select_edge_mark = False
    line_set.select_ridge_valley = False
    line_set.select_suggestive_contour = False
    line_style = line_set.linestyle
    if line_style is None:
        line_style = bpy.data.linestyles.new("MeshProbeShadedEdges")
        line_set.linestyle = line_style
    color = settings["line_color"]
    line_style.color = tuple(
        srgb_channel_to_linear(int(color[offset : offset + 2], 16)) for offset in (1, 3, 5)
    )
    line_style.thickness = settings["line_width"]


def require_supported_blender_version(version: tuple[int, int, int]) -> None:
    """Raise ``RuntimeError`` with a clear message if ``version`` is too old.

    ``version`` is the tuple Blender exposes as ``bpy.app.version``, e.g.
    ``(5, 1, 2)``. Only the major/minor components are compared against
    :data:`MINIMUM_BLENDER_VERSION`; the patch component is ignored. Pure
    function (no ``bpy``/``gpu`` access) so it stays unit-testable outside a
    real Blender process — see ``tests/unit/test_blender_worker.py``.

    Blender 4.2 through 5.1 run in bounded compatibility mode. They cannot call
    GPU telemetry or the GPU compositor path, which require Blender 5.2.
    """
    if version[:2] >= MINIMUM_BLENDER_VERSION:
        return
    detected = ".".join(str(component) for component in version)
    minimum = ".".join(str(component) for component in MINIMUM_BLENDER_VERSION)
    raise RuntimeError(
        f"Blender {detected} is unsupported — MeshProbe requires Blender >= {minimum}. "
        "Install Blender 4.2 LTS or newer."
    )


def uses_software_compatibility_mode(version: tuple[int, int, int]) -> bool:
    return version[:2] < GPU_INITIALIZATION_MINIMUM_VERSION


def graphics_platform() -> dict[str, Any]:
    if GPU_PLATFORM is None:
        raise RuntimeError("graphics platform is not initialized")
    return GPU_PLATFORM


def initialize_graphics_platform() -> dict[str, Any]:
    require_supported_blender_version(bpy.app.version)
    if uses_software_compatibility_mode(bpy.app.version):
        detected = bpy.app.version_string
        return {
            "vendor": "unavailable",
            "renderer": f"Blender {detected} compatibility mode (GPU unavailable)",
            "version": detected,
            "backend": "unavailable",
            "blender_device_type": "unavailable",
            "device_class": "unknown",
            "warnings": [
                f"Blender {detected} runs in compatibility mode without GPU telemetry: "
                "hardware_required and screen_edges require Blender 5.2 or newer."
            ],
        }
    gpu.init()
    vendor = str(gpu.platform.vendor_get())
    renderer = str(gpu.platform.renderer_get())
    version = str(gpu.platform.version_get())
    backend = str(gpu.platform.backend_type_get())
    blender_device_type = str(gpu.platform.device_type_get())
    renderer_folded = renderer.casefold()
    software_markers = (
        "llvmpipe",
        "softpipe",
        "swiftshader",
        "software rasterizer",
        "microsoft basic render driver",
        "warp",
    )
    reported_software = blender_device_type.casefold() == "software"
    if any(marker in renderer_folded for marker in software_markers) or (
        reported_software and "d3d12" not in renderer_folded
    ):
        device_class = "software"
    elif blender_device_type.casefold() == "unknown":
        device_class = "unknown"
    else:
        device_class = "hardware"
    warnings: list[str] = []
    if device_class == "software":
        warnings.append(f"Blender is using software graphics: {vendor} {renderer} ({backend}).")
    elif blender_device_type.casefold() == "software":
        warnings.append(
            "Blender labels this adapter SOFTWARE, but its renderer identifies a "
            f"hardware-backed D3D12 adapter: {renderer}."
        )
    return {
        "vendor": vendor,
        "renderer": renderer,
        "version": version,
        "backend": backend,
        "blender_device_type": blender_device_type,
        "device_class": device_class,
        "warnings": warnings,
    }


def eevee_engine() -> str:
    identifiers = {
        item.identifier for item in bpy.context.scene.render.bl_rna.properties["engine"].enum_items
    }
    if "BLENDER_EEVEE" in identifiers:
        return "BLENDER_EEVEE"
    if "BLENDER_EEVEE_NEXT" in identifiers:
        return "BLENDER_EEVEE_NEXT"
    raise RuntimeError(f"Blender exposes no Eevee render engine: {sorted(identifiers)}")


def configure_eevee_samples(samples: int) -> None:
    settings = bpy.context.scene.eevee
    if hasattr(settings, "taa_render_samples"):
        settings.taa_render_samples = samples
        return
    if hasattr(settings, "taa_samples"):
        settings.taa_samples = samples
        return
    raise RuntimeError("Blender exposes no supported Eevee render sample setting")


def eevee_render_samples() -> int:
    settings = bpy.context.scene.eevee
    if hasattr(settings, "taa_render_samples"):
        return int(settings.taa_render_samples)
    if hasattr(settings, "taa_samples"):
        return int(settings.taa_samples)
    raise RuntimeError("Blender exposes no supported Eevee render sample setting")


def configure_cycles_gpu(samples: int) -> None:
    scene = bpy.context.scene
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
    except Exception as error:
        raise RuntimeError(f"CUDA initialization failed: {error}") from error
    devices = [device for device in preferences.devices if device.type == "CUDA"]
    if not devices:
        raise RuntimeError("Cycles CUDA rendering requested but no CUDA device is available")
    for device in preferences.devices:
        device.use = device in devices
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True


def render_still(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file():
        raise RuntimeError(f"Blender did not publish render output: {path}")


@contextmanager
def hidden_component_labels() -> Iterator[None]:
    original_visibility = {label: label.hide_render for label in MARK_OBJECTS.values()}
    try:
        for label in original_visibility:
            label.hide_render = True
        yield
    finally:
        for label, hide_render in original_visibility.items():
            label.hide_render = hide_render


def composite_label_overlay(path: Path, overlay_path: Path) -> None:
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - OpenImageIO ships with Blender
        raise RuntimeError(
            "Label annotation compositing needs OpenImageIO, which is bundled with Blender but "
            "was not importable in this build"
        ) from error

    base = oiio.ImageBuf(str(path))
    overlay = oiio.ImageBuf(str(overlay_path))
    base_spec = base.spec()
    overlay_spec = overlay.spec()
    base_size = (base_spec.width, base_spec.height)
    overlay_size = (overlay_spec.width, overlay_spec.height)
    if base_size != overlay_size:
        raise RuntimeError(f"label overlay size {overlay_size} does not match render {base_size}")
    composited = oiio.ImageBufAlgo.over(overlay, base)
    if composited.has_error:
        raise RuntimeError(f"failed to composite label overlay: {composited.geterror()}")
    if not composited.write(str(path)):
        raise RuntimeError(f"failed to publish label overlay: {composited.geterror()}")


def render_label_overlay(path: Path) -> None:
    """Composite camera-facing labels separately from physical scene geometry.

    Rendering annotations alone through one-sample Eevee with DOF disabled keeps them sharp and
    unobstructed without letting text affect scene lighting, samples, or geometry evidence. The
    requested engine remains responsible for the physical scene render.
    """

    if not MARK_OBJECTS:
        return
    scene = bpy.context.scene
    camera = camera_object()
    labels = tuple(label for label in MARK_OBJECTS.values() if not label.hide_render)
    if not labels:
        return
    original_visibility = {obj: obj.hide_render for obj in scene.objects if obj.type != "CAMERA"}
    original_engine = scene.render.engine
    original_film_transparent = scene.render.film_transparent
    original_freestyle = scene.render.use_freestyle
    original_dof = camera.data.dof.use_dof
    original_samples = eevee_render_samples()
    with tempfile.TemporaryDirectory(prefix="meshprobe-label-overlay-") as directory:
        overlay_path = Path(directory) / "labels.png"
        try:
            for obj in original_visibility:
                obj.hide_render = obj not in labels
            scene.render.engine = eevee_engine()
            configure_eevee_samples(1)
            scene.render.film_transparent = True
            scene.render.use_freestyle = False
            camera.data.dof.use_dof = False
            render_still(overlay_path)
            composite_label_overlay(path, overlay_path)
        finally:
            for obj, hide_render in original_visibility.items():
                obj.hide_render = hide_render
            scene.render.engine = original_engine
            scene.render.film_transparent = original_film_transparent
            scene.render.use_freestyle = original_freestyle
            camera.data.dof.use_dof = original_dof
            configure_eevee_samples(original_samples)


@contextmanager
def temporary_screen_edge_compositor(
    path: Path,
    settings: dict[str, Any],
) -> Iterator[str]:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    original_group = scene.compositing_node_group
    original_compositor_device = scene.render.compositor_device
    original_pass_z = view_layer.use_pass_z
    original_pass_normal = view_layer.use_pass_normal
    group = None
    try:
        group = bpy.data.node_groups.new(
            f"MeshProbeScreenEdges-{path.stem}",
            "CompositorNodeTree",
        )
        scene.compositing_node_group = group
        scene.render.compositor_device = "GPU"
        view_layer.use_pass_z = True
        view_layer.use_pass_normal = True
        configure_screen_edge_nodes(group, path, settings)
        yield f"{path.stem}.screen-edges"
    finally:
        scene.compositing_node_group = original_group
        scene.render.compositor_device = original_compositor_device
        view_layer.use_pass_z = original_pass_z
        view_layer.use_pass_normal = original_pass_normal
        if group is not None:
            bpy.data.node_groups.remove(group)


def configure_screen_edge_nodes(
    group: bpy.types.NodeTree,
    path: Path,
    settings: dict[str, Any],
) -> None:
    """Build an experimental GPU compositor depth/normal edge pass."""

    render_layers = group.nodes.new("CompositorNodeRLayers")

    normalized_depth = group.nodes.new("CompositorNodeNormalize")
    depth_sobel = group.nodes.new("CompositorNodeFilter")
    # Blender 5.2 exposes the filter choice as a menu socket, not a node property.
    depth_sobel.inputs["Type"].default_value = "Sobel"
    depth_luminance = group.nodes.new("CompositorNodeRGBToBW")
    depth_threshold = group.nodes.new("ShaderNodeMath")
    depth_threshold.operation = "GREATER_THAN"
    depth_threshold.inputs[1].default_value = 0.01
    group.links.new(render_layers.outputs["Depth"], normalized_depth.inputs["Value"])
    group.links.new(normalized_depth.outputs["Value"], depth_sobel.inputs["Image"])
    group.links.new(depth_sobel.outputs["Image"], depth_luminance.inputs["Image"])
    group.links.new(depth_luminance.outputs["Val"], depth_threshold.inputs[0])

    normal_sobel = group.nodes.new("CompositorNodeFilter")
    normal_sobel.inputs["Type"].default_value = "Sobel"
    normal_luminance = group.nodes.new("CompositorNodeRGBToBW")
    normal_threshold = group.nodes.new("ShaderNodeMath")
    normal_threshold.operation = "GREATER_THAN"
    normal_threshold.inputs[1].default_value = 0.3
    group.links.new(render_layers.outputs["Normal"], normal_sobel.inputs["Image"])
    group.links.new(normal_sobel.outputs["Image"], normal_luminance.inputs["Image"])
    group.links.new(normal_luminance.outputs["Val"], normal_threshold.inputs[0])

    combined = group.nodes.new("ShaderNodeMath")
    combined.operation = "MAXIMUM"
    group.links.new(depth_threshold.outputs["Value"], combined.inputs[0])
    group.links.new(normal_threshold.outputs["Value"], combined.inputs[1])

    mask_output = combined.outputs["Value"]
    dilation_size = max(0, round(float(settings["line_width"]) - 1.0))
    if dilation_size:
        dilate = group.nodes.new("CompositorNodeDilateErode")
        dilate.inputs["Size"].default_value = dilation_size
        group.links.new(mask_output, dilate.inputs["Mask"])
        mask_output = dilate.outputs["Mask"]

    anti_alias = group.nodes.new("CompositorNodeAntiAliasing")
    group.links.new(mask_output, anti_alias.inputs["Image"])
    mask_output = anti_alias.outputs["Image"]

    opacity = group.nodes.new("ShaderNodeMath")
    opacity.operation = "MULTIPLY"
    opacity.inputs[1].default_value = 0.85
    group.links.new(mask_output, opacity.inputs[0])
    mask_output = opacity.outputs["Value"]

    line_color = group.nodes.new("CompositorNodeRGB")
    color = settings["line_color"]
    line_color.outputs["Color"].default_value = (
        *(srgb_channel_to_linear(int(color[offset : offset + 2], 16)) for offset in (1, 3, 5)),
        1.0,
    )
    mix = group.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    group.links.new(mask_output, mix.inputs[0])
    group.links.new(render_layers.outputs["Image"], mix.inputs[6])
    group.links.new(line_color.outputs["Color"], mix.inputs[7])

    file_output = group.nodes.new("CompositorNodeOutputFile")
    file_output.directory = str(path.parent)
    file_output.file_name = f"{path.stem}.screen-edges"
    file_output.use_file_extension = True
    file_output.format.file_format = "OPEN_EXR_MULTILAYER"
    file_output.file_output_items.new("RGBA", "Image")
    group.links.new(mix.outputs[2], file_output.inputs["Image"])


def render_screen_edges(
    path: Path,
    settings: dict[str, Any],
) -> None:
    """Render an experimental GPU compositor depth/normal edge pass."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="meshprobe-screen-edges-") as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        prefix = f"{temporary_path.stem}.screen-edges"
        with temporary_screen_edge_compositor(temporary_path, settings) as configured_prefix:
            if configured_prefix != prefix:
                raise RuntimeError(f"unexpected screen-edge output prefix: {configured_prefix}")
            bpy.ops.render.render()
        candidates = [
            candidate
            for candidate in temporary_path.parent.iterdir()
            if candidate.name.startswith(prefix) and candidate.suffix.lower() == ".exr"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one screen-edge output for {prefix}, found "
                f"{sorted(candidate.name for candidate in candidates)}"
            )
        composited = bpy.data.images.load(str(candidates[0]), check_existing=False)
        try:
            composited.save_render(str(path), scene=bpy.context.scene)
        finally:
            bpy.data.images.remove(composited)


def luminance_summary(path: Path) -> dict[str, float]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        if image.size[0] == 0 or image.size[1] == 0:
            raise RuntimeError("render result has no pixels")
        width, height = image.size
        x_stride = max(1, math.ceil(width / 512))
        y_stride = max(1, math.ceil(height / 512))
        values: list[float] = []
        for y in range(0, height, y_stride):
            row_start = y * width * 4
            row = image.pixels[row_start : row_start + width * 4]
            for x in range(0, width, x_stride):
                offset = x * 4
                value = 0.2126 * row[offset] + 0.7152 * row[offset + 1] + 0.0722 * row[offset + 2]
                values.append(min(1.0, max(0.0, value)))
        return {
            "minimum": min(values),
            "median": median(values),
            "maximum": max(values),
            "crushed_fraction": sum(value <= 0.01 for value in values) / len(values),
            "clipped_fraction": sum(value >= 0.99 for value in values) / len(values),
        }
    finally:
        bpy.data.images.remove(image)


# A rendered frame whose visible foreground covers less than this fraction of the
# pixels is treated as effectively empty and earns a warning, regardless of the cause
# (camera framing that misses the model, a leftover isolate/hidden display override,
# an empty shown set, ...). Measured post-render, so it catches every cause uniformly.
EMPTY_FRAME_COVERAGE_THRESHOLD = 0.001
FOREGROUND_MASK_MAX_DIMENSION = 256
# 8-bit channel value above which a foreground-mask pixel counts as covered; ignores the
# pure-black background plus anti-aliasing/dither fringe.
FOREGROUND_PIXEL_CUTOFF = 8


def count_foreground_pixels(path: Path) -> int:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = image.pixels[:]
        count = 0
        for offset in range(0, len(pixels), 4):
            if any(
                round(pixels[offset + channel] * 255) >= FOREGROUND_PIXEL_CUTOFF
                for channel in range(3)
            ):
                count += 1
        return count
    finally:
        bpy.data.images.remove(image)


def foreground_mask_material_mode() -> MaskMaterialMode:
    """Choose the cheapest mask material path that preserves visible coverage."""

    culling_modes: set[bool] = set()
    for obj in COMPONENT_OBJECTS.values():
        if obj.hide_render:
            continue
        if not obj.data.materials:
            culling_modes.add(False)
        for material in obj.data.materials:
            if material is None:
                culling_modes.add(False)
                continue
            culling_modes.add(bool(material.use_backface_culling))
            if material.diffuse_color[3] < 1.0:
                return MaskMaterialMode.MATERIAL_EXACT
            if not material.use_nodes:
                continue
            principled = material.node_tree.nodes.get("Principled BSDF")
            if principled is None:
                # Unknown node graphs may contain Transparent/Holdout shaders. Preserve them
                # instead of guessing that an arbitrary graph is opaque.
                return MaskMaterialMode.MATERIAL_EXACT
            alpha = principled.inputs.get("Alpha")
            if alpha is None or alpha.is_linked or float(alpha.default_value) < 1.0:
                return MaskMaterialMode.MATERIAL_EXACT
    if len(culling_modes) > 1:
        return MaskMaterialMode.MATERIAL_EXACT
    if culling_modes == {True}:
        return MaskMaterialMode.UNIFORM_BACKFACE_CULLED
    return MaskMaterialMode.UNIFORM_DOUBLE_SIDED


def foreground_coverage() -> dict[str, Any]:
    """Measure how much of the current frame any shown component actually covers.

    Renders the visible components as a flat white mask on black at a reduced
    resolution (coverage is resolution-independent) so an agent reading receipts can
    tell "nothing is here" from a real shot — which ``luminance`` cannot, since a
    blank frame reads as normal uniform background exposure.
    """

    scene = bpy.context.scene
    original_resolution = (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
    )
    original_engine = scene.render.engine
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    scale = min(1.0, FOREGROUND_MASK_MAX_DIMENSION / max(width, height))
    # A single scale factor keeps the mask's aspect ratio matched to the requested render (the
    # camera frame depends on it): independently flooring one axis to a minimum, as a per-axis
    # max(4, ...) would, can distort extreme aspect ratios (e.g. 64x16384 -> 4x256 is 1:64, not
    # 1:256) and sample a different view than the color render.
    scale = max(scale, 4 / min(width, height))
    sample_width = round(width * scale)
    sample_height = round(height * scale)
    colors = {component_id: (255, 255, 255) for component_id in COMPONENT_OBJECTS}
    with tempfile.TemporaryDirectory(prefix="meshprobe-foreground-") as directory:
        mask_path = Path(directory) / "foreground.png"
        try:
            scene.render.resolution_x = sample_width
            scene.render.resolution_y = sample_height
            scene.render.resolution_percentage = 100
            render_mask(
                mask_path,
                colors,
                material_mode=foreground_mask_material_mode(),
            )
            visible_pixels = count_foreground_pixels(mask_path)
        finally:
            (
                scene.render.resolution_x,
                scene.render.resolution_y,
                scene.render.resolution_percentage,
            ) = original_resolution
            scene.render.engine = original_engine
    sampled_pixels = sample_width * sample_height
    return {
        "visible_fraction": visible_pixels / sampled_pixels if sampled_pixels else 0.0,
        "visible_pixels": visible_pixels,
        "sampled_pixels": sampled_pixels,
        "sample_width": sample_width,
        "sample_height": sample_height,
    }


def empty_frame_warnings(coverage: dict[str, Any]) -> list[str]:
    if coverage["visible_fraction"] >= EMPTY_FRAME_COVERAGE_THRESHOLD:
        return []
    threshold_percent = EMPTY_FRAME_COVERAGE_THRESHOLD * 100
    return [
        "Rendered frame is effectively empty: visible foreground covers less than "
        f"{threshold_percent:g}% of the frame. No shown component occupies the view — "
        "check the camera framing and the component display/isolation state."
    ]


def emission_material(
    name: str,
    color: tuple[float, float, float, float],
) -> bpy.types.Material:
    material = EMISSION_MATERIALS.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        EMISSION_MATERIALS[name] = material
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def mask_alpha_variant_material(
    source: bpy.types.Material,
    color: tuple[float, float, float, float],
) -> bpy.types.Material:
    """A mask material that emits ``color`` where ``source`` is opaque and turns
    transparent where ``source`` is — mixing an Emission with a Transparent BSDF by the
    source's own Alpha. The source's whole node tree is duplicated (``source.copy()``),
    so the Alpha driver carries over verbatim whether it is a constant, a glTF MASK
    cutoff Blender baked to 0/1, or a per-pixel LINKED alpha texture; backface culling is
    preserved too. This keeps the foreground mask agreeing with the real color render:
    a fragment the color pass discards for transparency reads invisible here instead of
    being painted flat opaque and suppressing the empty-frame warning.

    Where the source is fully transparent (Alpha 0) the mix is pure Transparent BSDF, so
    it alpha-blends through to whatever is behind it — the world background, or another
    component's mask color — rather than erasing it. A Holdout shader was tried for that
    role first but always reveals the world background regardless of what else is behind
    the polygon, incorrectly erasing an opaque component's mask color when a transparent
    face sits in front of it on screen.
    """
    variant = source.copy()
    variant.name = f"MeshProbeMaskVariant-{source.name}"
    variant.use_nodes = True
    variant.use_backface_culling = source.use_backface_culling
    variant.blend_method = "BLEND"
    if hasattr(variant, "surface_render_method"):
        variant.surface_render_method = "BLENDED"
    tree = variant.node_tree
    nodes = tree.nodes
    output = next(
        (node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output),
        None,
    ) or nodes.get("Material Output")
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.get("Principled BSDF")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    alpha = principled.inputs.get("Alpha") if principled is not None else None
    if alpha is not None and alpha.is_linked:
        tree.links.new(alpha.links[0].from_socket, mix.inputs["Fac"])
    elif alpha is not None:
        mix.inputs["Fac"].default_value = float(alpha.default_value)
    else:
        mix.inputs["Fac"].default_value = float(source.diffuse_color[3])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return variant


def srgb_channel_to_linear(channel: int) -> float:
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return float(((value + 0.055) / 1.055) ** 2.4)


def component_pass_colors() -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    used: set[tuple[int, int, int]] = set()
    for component_id in sorted(COMPONENT_OBJECTS):
        seed = hashlib.sha256(component_id.encode()).digest()
        color = (max(seed[0], 16), max(seed[1], 16), max(seed[2], 16))
        counter = 0
        while color in used:
            counter += 1
            seed = hashlib.sha256(f"{component_id}:{counter}".encode()).digest()
            color = (max(seed[0], 16), max(seed[1], 16), max(seed[2], 16))
        colors[component_id] = color
        used.add(color)
    return colors


def render_mask(
    path: Path,
    colors: dict[str, tuple[int, int, int]],
    *,
    material_mode: MaskMaterialMode = MaskMaterialMode.FLAT_BY_COMPONENT,
) -> None:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    original_meshes = {component_id: obj.data for component_id, obj in COMPONENT_OBJECTS.items()}
    other_visibility = {
        obj: obj.hide_render for obj in scene.objects if obj.type not in {"MESH", "CAMERA"}
    }
    original_world = scene.world
    original_freestyle = scene.render.use_freestyle
    mask_world = bpy.data.worlds.get("MeshProbeMaskWorld") or bpy.data.worlds.new(
        "MeshProbeMaskWorld"
    )
    mask_world.use_nodes = True
    background = mask_world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    scene.world = mask_world
    scene.render.use_freestyle = False
    original_transform = scene.view_settings.view_transform
    original_look = scene.view_settings.look
    original_exposure = scene.view_settings.exposure
    original_gamma = scene.view_settings.gamma
    original_dither = scene.render.dither_intensity
    original_material_override = view_layer.material_override
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.dither_intensity = 0.0
    scene.render.engine = eevee_engine()
    original_samples = eevee_render_samples()
    # A binary coverage mask needs no anti-aliasing; inheriting the color render's
    # (possibly thousands of) samples would render the mask that many times over
    # just to answer a yes/no coverage question.
    configure_eevee_samples(1)
    mask_variants: list[bpy.types.Material] = []
    uniform_color = next(iter(set(colors.values()))) if len(set(colors.values())) == 1 else None
    try:
        for obj in other_visibility:
            obj.hide_render = True
        if uniform_color is not None and material_mode in {
            MaskMaterialMode.UNIFORM_BACKFACE_CULLED,
            MaskMaterialMode.UNIFORM_DOUBLE_SIDED,
        }:
            red, green, blue = uniform_color
            override = emission_material(
                f"MeshProbeMask-{red:02x}{green:02x}{blue:02x}-{material_mode.value}",
                (
                    srgb_channel_to_linear(red),
                    srgb_channel_to_linear(green),
                    srgb_channel_to_linear(blue),
                    1.0,
                ),
            )
            override.use_backface_culling = (
                material_mode is MaskMaterialMode.UNIFORM_BACKFACE_CULLED
            )
            view_layer.material_override = override
        else:
            for component_id, obj in COMPONENT_OBJECTS.items():
                source_materials = list(obj.data.materials)
                mesh = obj.data.copy()
                mesh.materials.clear()
                red, green, blue = colors[component_id]
                color = (
                    srgb_channel_to_linear(red),
                    srgb_channel_to_linear(green),
                    srgb_channel_to_linear(blue),
                    1.0,
                )
                base_name = f"MeshProbeMask-{red:02x}{green:02x}{blue:02x}"
                if material_mode is MaskMaterialMode.FLAT_BY_COMPONENT:
                    mesh.materials.append(emission_material(base_name, color))
                    for polygon in mesh.polygons:
                        polygon.material_index = 0
                elif material_mode is MaskMaterialMode.MATERIAL_EXACT:
                    # Preserve each source material's transparency/culling instead of forcing
                    # every polygon opaque and front-and-back visible: a component that is
                    # invisible in the real color render (transparent material — constant,
                    # glTF MASK cutoff, or a per-pixel alpha texture — or a culled backface)
                    # must read the same way in this mask, or the foreground check can't tell
                    # "geometry exists" from "geometry is actually visible". Each source
                    # material gets one alpha-preserving emissive variant; polygons with no
                    # material fall back to a flat opaque mask.
                    flat_index = -1
                    variant_index: dict[str, int] = {}
                    for polygon in mesh.polygons:
                        original = (
                            source_materials[polygon.material_index]
                            if polygon.material_index < len(source_materials)
                            else None
                        )
                        if original is None:
                            if flat_index < 0:
                                flat_index = len(mesh.materials)
                                mesh.materials.append(emission_material(f"{base_name}-flat", color))
                            polygon.material_index = flat_index
                            continue
                        if original.name not in variant_index:
                            variant = mask_alpha_variant_material(original, color)
                            mask_variants.append(variant)
                            variant_index[original.name] = len(mesh.materials)
                            mesh.materials.append(variant)
                        polygon.material_index = variant_index[original.name]
                else:
                    raise ValueError(
                        f"mask material mode requires one uniform color: {material_mode.value}"
                    )
                obj.data = mesh
        render_still(path)
    finally:
        view_layer.material_override = original_material_override
        for component_id, obj in COMPONENT_OBJECTS.items():
            replacement = obj.data
            obj.data = original_meshes[component_id]
            if replacement.users == 0:
                bpy.data.meshes.remove(replacement)
        for variant in mask_variants:
            if variant.users == 0:
                bpy.data.materials.remove(variant)
        for obj, hide_render in other_visibility.items():
            obj.hide_render = hide_render
        scene.world = original_world
        scene.render.use_freestyle = original_freestyle
        scene.view_settings.view_transform = original_transform
        scene.view_settings.look = original_look
        scene.view_settings.exposure = original_exposure
        scene.view_settings.gamma = original_gamma
        scene.render.dither_intensity = original_dither
        configure_eevee_samples(original_samples)


def count_mask_pixels(path: Path, selected_colors: set[tuple[int, int, int]]) -> int:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = image.pixels[:]
        count = 0
        for offset in range(0, len(pixels), 4):
            color = tuple(round(pixels[offset + channel] * 255) for channel in range(3))
            if color in selected_colors:
                count += 1
        return count
    finally:
        bpy.data.images.remove(image)


def component_visibility(command: dict[str, Any]) -> dict[str, Any]:
    focus_ids = selected_component_ids(command)
    width = int(command.get("width", 256))
    height = int(command.get("height", 256))
    if not 4 <= width <= 1_024 or not 4 <= height <= 1_024:
        raise ValueError("component.visibility dimensions must be between 4 and 1024")
    configure_render(
        {
            "engine": "eevee",
            "width": width,
            "height": height,
            "samples": 1,
            "graphics_policy": command.get("graphics_policy", "software_allowed"),
        }
    )
    colors = component_pass_colors()
    selected_colors = {colors[component_id] for component_id in focus_ids}
    original_visibility = {
        component_id: obj.hide_render for component_id, obj in COMPONENT_OBJECTS.items()
    }
    with tempfile.TemporaryDirectory(prefix="meshprobe-visibility-") as directory:
        root = Path(directory)
        context_path = root / "context.png"
        isolated_path = root / "isolated.png"
        try:
            render_mask(context_path, colors)
            visible_pixels = count_mask_pixels(context_path, selected_colors)
            for component_id, obj in COMPONENT_OBJECTS.items():
                obj.hide_render = component_id not in focus_ids
            render_mask(isolated_path, colors)
            isolated_pixels = count_mask_pixels(isolated_path, selected_colors)
        finally:
            for component_id, hidden in original_visibility.items():
                COMPONENT_OBJECTS[component_id].hide_render = hidden
    return {
        "focus_component_ids": sorted(focus_ids),
        "width": width,
        "height": height,
        "visible_pixels": visible_pixels,
        "isolated_pixels": isolated_pixels,
        "visible_fraction": visible_pixels / isolated_pixels if isolated_pixels else 0.0,
        "projected": isolated_pixels > 0,
    }


def render_evaluator_passes(output_dir: Path, stem: str) -> dict[str, Any]:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    original_group = scene.compositing_node_group
    group = bpy.data.node_groups.new(f"MeshProbePasses-{stem}", "CompositorNodeTree")
    scene.compositing_node_group = group
    render_layers = group.nodes.new("CompositorNodeRLayers")
    pass_output = compositor_file_output(group, output_dir, f"{stem}.passes")
    pass_output.file_output_items.new("FLOAT", "Depth")
    pass_output.file_output_items.new("VECTOR", "Normal")
    group.links.new(render_layers.outputs["Depth"], pass_output.inputs["Depth"])
    group.links.new(render_layers.outputs["Normal"], pass_output.inputs["Normal"])
    prefix = f"{stem}.passes"
    before = {path: file_version(path) for path in output_dir.glob(f"{prefix}*.exr")}
    # Freestyle strokes composite into the Combined pass only; the Depth and Normal
    # passes come straight from geometry rasterization and are byte-identical whether
    # Freestyle is on or off. Leaving it on here would pay a second single-threaded
    # view-map computation (the shaded_edges bottleneck) for output it never touches.
    original_freestyle = scene.render.use_freestyle
    scene.render.use_freestyle = False
    try:
        bpy.ops.render.render()
    finally:
        scene.render.use_freestyle = original_freestyle
        scene.compositing_node_group = original_group
        bpy.data.node_groups.remove(group)
    multilayer_path = published_compositor_path(output_dir, prefix, before)

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    component_path = output_dir / f"{stem}.components.png"
    component_colors = component_pass_colors()
    render_mask(component_path, component_colors)
    highlight_path = output_dir / f"{stem}.highlighted.png"
    highlight_colors = {
        component_id: (
            (255, 255, 255) if COMPONENT_STATES[component_id]["mark"] != "unmarked" else (0, 0, 0)
        )
        for component_id in COMPONENT_OBJECTS
    }
    render_mask(highlight_path, highlight_colors)
    return {
        "multilayer": artifact(multilayer_path, "image/x-exr"),
        "component_ids": artifact(component_path, "image/png"),
        "highlighted": artifact(highlight_path, "image/png"),
        "component_colors": component_colors,
    }


def compositor_file_output(
    group: bpy.types.NodeTree,
    output_dir: Path,
    file_name: str,
) -> bpy.types.Node:
    node = group.nodes.new("CompositorNodeOutputFile")
    node.directory = str(output_dir)
    node.file_name = file_name
    node.use_file_extension = True
    node.format.file_format = "OPEN_EXR_MULTILAYER"
    node.format.color_depth = "32"
    return node


def file_version(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def published_compositor_path(
    output_dir: Path,
    prefix: str,
    before: dict[Path, tuple[int, int, int]],
) -> Path:
    expected = output_dir / f"{prefix}.exr"
    candidates = [
        path
        for path in output_dir.glob(f"{prefix}*.exr")
        if path not in before or file_version(path) != before[path]
    ]
    if len(candidates) != 1:
        names = sorted(path.name for path in candidates)
        raise RuntimeError(f"expected one fresh compositor output for {prefix}, found {names}")
    candidate = candidates[0]
    if candidate == expected:
        return expected
    expected.unlink(missing_ok=True)
    candidate.replace(expected)
    return expected


def display_referred_background() -> tuple[float, float, float] | None:
    illumination = CURRENT_ILLUMINATION
    if illumination is None:
        return None
    srgb = illumination.get("background_srgb")
    if srgb is None or illumination.get("visible_background_mode", "color") != "color":
        return None
    return (float(srgb[0]), float(srgb[1]), float(srgb[2]))


def composite_over_background(path: Path, background_srgb: tuple[float, float, float]) -> None:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - numpy ships with Blender
        raise RuntimeError(
            "--background-srgb compositing needs numpy, which is bundled with Blender but "
            "was not importable in this build; drop the flag or install numpy for this Blender"
        ) from error

    # Composite the exact display-referred color behind the transparent (film_transparent)
    # backdrop in place, using the bulk pixel API into a float32 buffer so a max-resolution
    # render never materializes the whole image as a Python float list.
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        width, height = image.size
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        pixels = buffer.reshape(-1, 4)
        rgb = pixels[:, :3]
        alpha = pixels[:, 3]
        background = np.asarray(background_srgb, dtype=np.float32)
        # Composite one channel at a time, using `out=` to write into two reused
        # (width*height,) scratch buffers instead of `rgb * alpha + background * (1 - alpha)`
        # computed directly, which allocates several full (width*height, 3) float32
        # temporaries (each ~3 GiB at the 16,384x16,384 limit) — or per-channel scratch
        # allocated fresh on every loop iteration, which still peaks at ~1 GiB per
        # temporary. Two reused (width*height,) buffers keep the composite's extra
        # memory bounded regardless of how many channels it runs over.
        one_minus_alpha = np.empty_like(alpha)
        np.subtract(1.0, alpha, out=one_minus_alpha)
        scaled_background = np.empty_like(alpha)
        for channel in range(3):
            channel_view = rgb[:, channel]
            channel_view *= alpha
            np.multiply(one_minus_alpha, background[channel], out=scaled_background)
            channel_view += scaled_background
        np.clip(rgb, 0.0, 1.0, out=rgb)
        pixels[:, 3] = 1.0
        image.pixels.foreach_set(buffer)
        image.file_format = "PNG"
        image.filepath_raw = str(path)
        image.save()
    finally:
        bpy.data.images.remove(image)


def render_image(command: dict[str, Any]) -> dict[str, Any]:
    manifest = require_session()
    style = command.setdefault("style", "screen_edges")
    compatibility_warnings: list[str] = []
    if style == "screen_edges" and uses_software_compatibility_mode(bpy.app.version):
        # The public protocol's default is screen_edges. Blender 4.2 lacks its
        # GPU compositor, so honor the usable default-render path with the
        # closest portable style and make the downgrade visible in the result.
        style = "shaded"
        command["style"] = style
        compatibility_warnings.append(
            "Blender 4.2 software compatibility mode rendered shaded instead of "
            "screen_edges; screen_edges requires Blender 5.2 or newer."
        )
    shaded_edges = command.setdefault(
        "shaded_edges",
        {
            "line_color": "#202020",
            "line_width": 1.5,
            "crease_angle_degrees": 120,
            "edge_types": ["silhouette", "border", "crease", "material_boundary"],
        },
    )
    output = Path(command["output_path"]).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("render.image output_path must end in .png")
    if command.get("evaluator_output_dir") is not None and uses_software_compatibility_mode(
        bpy.app.version
    ):
        raise RuntimeError(
            "evaluator passes require Blender 5.2 or newer; Blender 4.2 software "
            "compatibility mode supports shaded and shaded_edges renders only"
        )
    with temporary_render_style():
        device = configure_render(command)
        configure_render_style(command)
        backdrop = display_referred_background()
        if backdrop is not None:
            bpy.context.scene.render.film_transparent = True

        def render_color() -> None:
            if style == "screen_edges":
                render_screen_edges(output, shaded_edges)
                return
            render_still(output)

        # Labels are annotations, never physical scene geometry. Keep them out of every
        # requested engine's lighting/sample pass and composite them uniformly afterward.
        with hidden_component_labels():
            render_color()
        if backdrop is not None:
            bpy.context.scene.render.film_transparent = False
            composite_over_background(output, backdrop)
        render_label_overlay(output)
        luminance = luminance_summary(output)
        foreground = foreground_coverage()
        evaluator = None
        if command.get("evaluator_output_dir") is not None:
            evaluator_dir = Path(command["evaluator_output_dir"]).expanduser().resolve()
            evaluator_dir.mkdir(parents=True, exist_ok=True)
            with hidden_component_labels():
                evaluator = render_evaluator_passes(evaluator_dir, output.stem)
        session = session_snapshot()
    return {
        "schema_version": 2,
        "source_sha256": manifest["source_sha256"],
        "state_sha256": session["state_sha256"],
        "width": command["width"],
        "height": command["height"],
        "samples": command["samples"],
        "engine": command["engine"],
        "style": style,
        "shaded_edges": shaded_edges,
        "device": device,
        "graphics_policy": command["graphics_policy"],
        "graphics": graphics_platform(),
        "blender_version": bpy.app.version_string,
        "session": session,
        "color": artifact(output, "image/png"),
        "evaluator": evaluator,
        "luminance": luminance,
        "foreground": foreground,
        "warnings": [*compatibility_warnings, *empty_frame_warnings(foreground)],
        "resolved_depth_of_field": resolved_depth_of_field(),
    }


def resolved_depth_of_field() -> dict[str, float] | None:
    camera = camera_object()
    data = camera.data
    if data.type != "PERSP" or not data.dof.use_dof:
        return None
    focus_distance = data.dof.focus_distance
    if data.dof.focus_object is not None:
        camera_position = camera.matrix_world.translation
        focus_position = data.dof.focus_object.matrix_world.translation
        focus_distance = (focus_position - camera_position).length
    return {
        "aperture_fstop": data.dof.aperture_fstop,
        "focus_distance_mm": focus_distance * MILLIMETERS_PER_METER,
    }


def export_normalized(command: dict[str, Any]) -> dict[str, Any]:
    require_session()
    output = Path(command["output_path"]).expanduser().resolve()
    if output.suffix.lower() != ".glb":
        raise ValueError("scene.export_normalized output_path must end in .glb")
    if output.exists():
        raise ValueError("scene.export_normalized refuses to overwrite an existing file")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_apply=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender exporter did not finish: {sorted(result)}")
    return artifact(output, "model/gltf-binary")


def material_is_fully_transparent(material: Any) -> bool:
    if not material.use_nodes or material.node_tree is None:
        return bool(float(material.diffuse_color[3]) <= 1e-6)
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        return False
    alpha = principled.inputs.get("Alpha")
    return bool(alpha is not None and not alpha.is_linked and alpha.default_value <= 1e-6)


def rank_occluders(command: dict[str, Any]) -> dict[str, Any]:
    focus_ids = selected_component_ids(command)
    scene = bpy.context.scene
    camera = camera_object()
    camera_origin = camera.matrix_world.translation
    camera_forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    camera_forward.normalize()
    projection = "orthographic" if camera.data.type == "ORTHO" else "perspective"
    object_ids = {obj: component_id for component_id, obj in COMPONENT_OBJECTS.items()}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[tuple[str, Vector]] = []
    max_samples = int(command.get("max_samples_per_component", 128))
    if not 1 <= max_samples <= 4_096:
        raise ValueError("component.occluders max_samples_per_component must be between 1 and 4096")
    default_aspect = (
        scene.render.resolution_x
        * scene.render.pixel_aspect_x
        / max(1e-12, scene.render.resolution_y * scene.render.pixel_aspect_y)
    )
    aspect_ratio = float(command.get("aspect_ratio", default_aspect))
    if not math.isfinite(aspect_ratio) or not 0.01 <= aspect_ratio <= 100:
        raise ValueError("component.occluders aspect_ratio must be between 0.01 and 100")
    render_settings = (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.pixel_aspect_x,
        scene.render.pixel_aspect_y,
    )

    def material_uses_transparency(material: Any) -> bool:
        if not material.use_nodes or material.node_tree is None:
            return bool(float(material.diffuse_color[3]) < 1 - 1e-6)
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is None:
            return False
        alpha = principled.inputs.get("Alpha")
        if alpha is None or alpha.is_linked:
            return False
        # Blender 5.2 resolves scalar glTF OPAQUE/MASK factors to 1/0 at import.
        return bool(float(alpha.default_value) < 1 - 1e-6)

    def hit_material(hit_object: Any, polygon_index: int) -> Any | None:
        evaluated = hit_object.evaluated_get(depsgraph)
        data = getattr(evaluated, "data", None)
        polygons = getattr(data, "polygons", ())
        if not 0 <= polygon_index < len(polygons):
            return None
        material_index = polygons[polygon_index].material_index
        material_slots = getattr(evaluated, "material_slots", ())
        if not 0 <= material_index < len(material_slots):
            return None
        return material_slots[material_index].material

    def is_inside_camera_frustum(point: Vector) -> bool:
        depth = (point - camera_origin).dot(camera_forward)
        if depth - camera.data.clip_start <= 1e-6 or camera.data.clip_end - depth <= 1e-6:
            return False
        projected = world_to_camera_view(scene, camera, point)
        if not 1e-9 < projected.x < 1 - 1e-9:
            return False
        return bool(1e-9 < projected.y < 1 - 1e-9)

    def source_material(obj: Any, polygon_index: int) -> Any | None:
        polygon = obj.data.polygons[polygon_index]
        material_slots = obj.material_slots
        if not 0 <= polygon.material_index < len(material_slots):
            return None
        return material_slots[polygon.material_index].material

    def polygon_renders_from_camera(obj: Any, polygon_index: int, point: Vector) -> bool:
        material = source_material(obj, polygon_index)
        if material is not None and material_is_fully_transparent(material):
            return False
        if material is None or not material.use_backface_culling:
            return True
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        world_normal = (normal_matrix @ obj.data.polygons[polygon_index].normal).normalized()
        direction = camera_forward if projection == "orthographic" else (point - camera_origin)
        if direction.length <= 1e-12:
            return False
        return bool(world_normal.dot(direction.normalized()) <= 1e-9)

    def eligible_focus_point(
        obj: Any,
        point: Vector,
        polygon_indices: tuple[int, ...],
    ) -> bool:
        if not is_inside_camera_frustum(point):
            return False
        return any(
            polygon_renders_from_camera(obj, polygon_index, point)
            for polygon_index in polygon_indices
        )

    def triangle_screen_sample(
        triangle: tuple[Vector, Vector, Vector],
        screen_x: float,
        screen_y: float,
    ) -> Vector | None:
        projected = tuple(world_to_camera_view(scene, camera, point) for point in triangle)
        denominator = (projected[1].y - projected[2].y) * (projected[0].x - projected[2].x) + (
            projected[2].x - projected[1].x
        ) * (projected[0].y - projected[2].y)
        if abs(denominator) <= 1e-12:
            return None
        weights = (
            (
                (projected[1].y - projected[2].y) * (screen_x - projected[2].x)
                + (projected[2].x - projected[1].x) * (screen_y - projected[2].y)
            )
            / denominator,
            (
                (projected[2].y - projected[0].y) * (screen_x - projected[2].x)
                + (projected[0].x - projected[2].x) * (screen_y - projected[2].y)
            )
            / denominator,
            0.0,
        )
        weights = (weights[0], weights[1], 1 - weights[0] - weights[1])
        if min(weights) < -1e-9:
            return None
        if projection == "perspective":
            depths = tuple((point - camera_origin).dot(camera_forward) for point in triangle)
            if min(depths) <= 0:
                return None
            corrected = tuple(weight / depth for weight, depth in zip(weights, depths, strict=True))
            total = sum(corrected)
            weights = tuple(weight / total for weight in corrected)
        point = triangle[0] * weights[0] + triangle[1] * weights[1] + triangle[2] * weights[2]
        return point if is_inside_camera_frustum(point) else None

    def edge_frustum_samples(start: Vector, end: Vector) -> list[Vector]:
        samples: list[Vector] = []
        start_projected = world_to_camera_view(scene, camera, start)
        end_projected = world_to_camera_view(scene, camera, end)
        for axis in ("x", "y"):
            start_value = float(getattr(start_projected, axis))
            end_value = float(getattr(end_projected, axis))
            for boundary in (0.0, 1.0):
                if (start_value - boundary) * (end_value - boundary) > 0:
                    continue
                if math.isclose(start_value, end_value, abs_tol=1e-12):
                    continue
                low = 0.0
                high = 1.0
                for _ in range(32):
                    middle = (low + high) / 2
                    point = start.lerp(end, middle)
                    value = float(getattr(world_to_camera_view(scene, camera, point), axis))
                    if (start_value - boundary) * (value - boundary) <= 0:
                        high = middle
                    else:
                        low = middle
                point = start.lerp(end, (low + high) / 2)
                start_is_interior = (boundary == 0.0 and start_value > 0.0) or (
                    boundary == 1.0 and start_value < 1.0
                )
                interior = start if start_is_interior else end
                samples.append(point.lerp(interior, 1e-6))
        start_depth = (start - camera_origin).dot(camera_forward)
        end_depth = (end - camera_origin).dot(camera_forward)
        for clip_depth in (camera.data.clip_start, camera.data.clip_end):
            if (start_depth - clip_depth) * (end_depth - clip_depth) > 0:
                continue
            if math.isclose(start_depth, end_depth, abs_tol=1e-12):
                continue
            factor = (clip_depth - start_depth) / (end_depth - start_depth)
            point = start.lerp(end, factor)
            start_is_interior = (
                clip_depth == camera.data.clip_start and start_depth > clip_depth
            ) or (clip_depth == camera.data.clip_end and start_depth < clip_depth)
            interior = start if start_is_interior else end
            samples.append(point.lerp(interior, 1e-6))
        return samples

    surface_triangles_scanned = 0

    def surface_frustum_samples(obj: Any, budget: int) -> list[Vector]:
        nonlocal surface_triangles_scanned
        if budget <= 0:
            return []
        samples: list[Vector] = []
        triangle_weights = (
            (1 / 3, 1 / 3, 1 / 3),
            (0.8, 0.1, 0.1),
            (0.1, 0.8, 0.1),
            (0.1, 0.1, 0.8),
            (0.45, 0.45, 0.1),
            (0.45, 0.1, 0.45),
            (0.1, 0.45, 0.45),
        )
        screen_samples = (
            (0.5, 0.5),
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
        )
        obj.data.calc_loop_triangles()
        triangles = obj.data.loop_triangles
        triangle_scan_budget = max(8, budget * 8)
        triangle_stride = max(1, math.ceil(len(triangles) / triangle_scan_budget))
        triangle_start = min(len(triangles) - 1, triangle_stride // 2) if triangles else 0
        for triangles_scanned, triangle_index in enumerate(
            range(triangle_start, len(triangles), triangle_stride)
        ):
            if triangles_scanned >= triangle_scan_budget:
                return samples
            surface_triangles_scanned += 1
            loop_triangle = triangles[triangle_index]
            polygon_index = loop_triangle.polygon_index
            triangle = tuple(
                obj.matrix_world @ obj.data.vertices[vertex_index].co
                for vertex_index in loop_triangle.vertices
            )
            for weights in triangle_weights:
                point = (
                    triangle[0] * weights[0] + triangle[1] * weights[1] + triangle[2] * weights[2]
                )
                if eligible_focus_point(obj, point, (polygon_index,)):
                    samples.append(point)
                if len(samples) >= budget:
                    return samples
            for screen_x, screen_y in screen_samples:
                point = triangle_screen_sample(triangle, screen_x, screen_y)
                if point is not None and eligible_focus_point(obj, point, (polygon_index,)):
                    samples.append(point)
                if len(samples) >= budget:
                    return samples
            for start, end in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                samples.extend(
                    point
                    for point in edge_frustum_samples(start, end)
                    if eligible_focus_point(obj, point, (polygon_index,))
                )
                if len(samples) >= budget:
                    return samples[:budget]
        return samples

    try:
        scene.render.resolution_x = 1_000
        scene.render.resolution_y = 1_000
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = max(1.0, aspect_ratio)
        scene.render.pixel_aspect_y = max(1.0, 1.0 / aspect_ratio)
        for component_id in sorted(focus_ids):
            if COMPONENT_STATES[component_id]["display"] == "hidden":
                continue
            obj = COMPONENT_OBJECTS[component_id].evaluated_get(depsgraph)
            vertices = obj.data.vertices
            vertex_polygons: list[list[int]] = [[] for _ in vertices]
            for polygon in obj.data.polygons:
                for vertex_index in polygon.vertices:
                    vertex_polygons[vertex_index].append(polygon.index)
            vertex_budget = max_samples
            stride = max(1, math.ceil(len(vertices) / vertex_budget)) if vertex_budget else 1
            indices = range(0, len(vertices), stride) if vertex_budget else ()
            candidates = [
                (
                    obj.matrix_world @ vertices[index].co,
                    tuple(vertex_polygons[index]),
                )
                for index in indices
            ]
            component_points = [
                point
                for point, polygon_indices in candidates
                if eligible_focus_point(obj, point, polygon_indices)
            ]
            remaining_budget = max_samples - len(component_points)
            component_points.extend(surface_frustum_samples(obj, remaining_budget))
            points.extend((component_id, point) for point in component_points[:max_samples])
    finally:
        (
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.resolution_percentage,
            scene.render.pixel_aspect_x,
            scene.render.pixel_aspect_y,
        ) = render_settings
    projected_points: dict[tuple[int, int], tuple[str, Vector, float]] = {}
    for component_id, point in points:
        projected = world_to_camera_view(scene, camera, point)
        key = (round(float(projected.x) * 1e9), round(float(projected.y) * 1e9))
        depth = (point - camera_origin).dot(camera_forward)
        previous = projected_points.get(key)
        if previous is None or depth < previous[2]:
            projected_points[key] = (component_id, point, depth)
    points = [(component_id, point) for component_id, point, _ in projected_points.values()]
    counts: dict[str, int] = {}
    visible_rays = 0
    unresolved_rays = 0
    annotation_objects = set(MARK_OBJECTS.values())
    annotation_names = {obj.name for obj in annotation_objects}
    for _sampled_component_id, point in points:
        depth = (point - camera_origin).dot(camera_forward)
        if projection == "orthographic":
            ray_origin = point - camera_forward * depth
            ray_origin += camera_forward * camera.data.clip_start
            direction = camera_forward
            distance = depth - camera.data.clip_start
        else:
            direction = point - camera_origin
            full_distance = direction.length
            near_fraction = camera.data.clip_start / depth
            ray_origin = camera_origin + direction * near_fraction
            distance = full_distance * (1 - near_fraction)
        if distance <= 1e-9:
            unresolved_rays += 1
            continue
        normalized_direction = direction.normalized()
        remaining_distance = distance + 1e-6
        hit_component_id = None
        hit_resolved = False
        while remaining_distance > 1e-9:
            hit, hit_location, hit_normal, polygon_index, hit_object, _ = (
                bpy.context.scene.ray_cast(
                    depsgraph,
                    ray_origin,
                    normalized_direction,
                    distance=remaining_distance,
                )
            )
            if not hit or hit_object is None:
                break
            original = hit_object.original if hasattr(hit_object, "original") else hit_object
            hit_component_id = object_ids.get(original)
            if hit_component_id is None:
                if original in annotation_objects or hit_object.name in annotation_names:
                    traveled = (hit_location - ray_origin).length
                    advance = traveled + 1e-6
                    next_remaining_distance = remaining_distance - advance
                    next_ray_origin = hit_location + normalized_direction * 1e-6
                    if (
                        next_remaining_distance >= remaining_distance
                        or (next_ray_origin - ray_origin).length <= 1e-12
                    ):
                        break
                    remaining_distance = next_remaining_distance
                    ray_origin = next_ray_origin
                    continue
                hit_resolved = True
                break
            hit_state = COMPONENT_STATES[hit_component_id]
            material = hit_material(hit_object, polygon_index)
            backface_culled = bool(
                material is not None
                and material.use_backface_culling
                and hit_normal.dot(normalized_direction) > 1e-9
            )
            if (
                hit_component_id in focus_ids
                and hit_state["display"] != "hidden"
                and not backface_culled
            ):
                hit_resolved = True
                break
            transparent_hit = (
                backface_culled
                or hit_state["display"] == "hidden"
                or (hit_state["display"] == "ghosted" and hit_state["mark"] == "unmarked")
                or (
                    hit_state["mark"] == "unmarked"
                    and material is not None
                    and material_uses_transparency(material)
                )
            )
            if not transparent_hit:
                hit_resolved = True
                break
            traveled = (hit_location - ray_origin).length
            advance = traveled + 1e-6
            next_remaining_distance = remaining_distance - advance
            next_ray_origin = hit_location + normalized_direction * 1e-6
            if (
                next_remaining_distance >= remaining_distance
                or (next_ray_origin - ray_origin).length <= 1e-12
            ):
                break
            remaining_distance = next_remaining_distance
            ray_origin = next_ray_origin
        if not hit_resolved:
            unresolved_rays += 1
            continue
        if hit_component_id in focus_ids:
            visible_rays += 1
            continue
        if hit_component_id is None:
            unresolved_rays += 1
            continue
        counts[hit_component_id] = counts.get(hit_component_id, 0) + 1
    sample_count = visible_rays + unresolved_rays + sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "focus_component_ids": sorted(focus_ids),
        "projection": projection,
        "aspect_ratio": aspect_ratio,
        "camera_diagnostics": camera_diagnostics(
            camera,
            sorted(focus_ids),
            aspect_ratio,
            None,
        ),
        "sample_count": sample_count,
        "surface_triangles_scanned": surface_triangles_scanned,
        "visible_rays": visible_rays,
        "occluded_rays": sum(counts.values()),
        "unresolved_rays": unresolved_rays,
        "occluders": [
            {
                "component_id": component_id,
                "blocked_rays": count,
                "fraction": count / sample_count if sample_count else 0.0,
            }
            for component_id, count in ranked
        ],
    }


def initialize_session(
    manifest: dict[str, Any], source_path: Path, aspect_ratio: float = 1.0
) -> None:
    global MANIFEST, COMPONENT_OBJECTS, ORIGINAL_MESHES, ORIGINAL_VISIBILITY
    global COMPONENT_STATES, IMPORTED_CAMERA, CURRENT_CAMERA, CURRENT_CAMERA_OPERATION
    global IMPORTED_ILLUMINATION, CURRENT_ILLUMINATION, MARK_OBJECTS
    global OVERRIDE_MATERIALS, EMISSION_MATERIALS, DEFAULT_ASPECT_RATIO

    DEFAULT_ASPECT_RATIO = aspect_ratio
    MANIFEST = manifest
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    path_objects = set(objects)
    for obj in objects:
        ancestor = obj.parent
        while ancestor is not None:
            path_objects.add(ancestor)
            ancestor = ancestor.parent
    source_names, _ = source_names_for_objects(source_path, sorted(path_objects, key=object_path))
    objects_by_path = {path: obj for obj, path in source_paths(objects, source_names).items()}
    COMPONENT_OBJECTS = {
        component["id"]: objects_by_path[component["path"]] for component in manifest["components"]
    }
    ORIGINAL_MESHES = {component_id: obj.data for component_id, obj in COMPONENT_OBJECTS.items()}
    ORIGINAL_VISIBILITY = {
        component_id: (obj.hide_render, obj.hide_viewport)
        for component_id, obj in COMPONENT_OBJECTS.items()
    }
    COMPONENT_STATES = {
        component_id: {
            "display": "hidden" if obj.hide_render else "shown",
            "mark": "unmarked",
            "mark_color": None,
        }
        for component_id, obj in COMPONENT_OBJECTS.items()
    }
    IMPORTED_CAMERA = deepcopy(manifest["imported_camera"])
    CURRENT_CAMERA = framed_default_camera(
        IMPORTED_CAMERA,
        visible_root_bounds(manifest["root_bounds"]),
        aspect_ratio=aspect_ratio,
    )
    remember_perspective_sensor_intrinsics(CURRENT_CAMERA["projection"])
    CURRENT_CAMERA_OPERATION = None
    IMPORTED_ILLUMINATION = deepcopy(manifest["imported_illumination"])
    CURRENT_ILLUMINATION = deepcopy(IMPORTED_ILLUMINATION)
    MARK_OBJECTS = {}
    OVERRIDE_MATERIALS = {}
    EMISSION_MATERIALS = {}
    apply_camera(deepcopy(CURRENT_CAMERA), aspect_ratio=aspect_ratio)
    apply_illumination(deepcopy(IMPORTED_ILLUMINATION))


def import_source(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    warnings: list[dict[str, Any]] = []
    if suffix in {".glb", ".gltf"}:
        result = bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        result = bpy.ops.wm.obj_import(filepath=str(path))
        warnings.append(
            {
                "code": "units.assumed",
                "message": (
                    "OBJ has no reliable unit declaration; imported coordinates are treated "
                    "as meters."
                ),
                "component_ids": [],
            }
        )
    elif suffix == ".stl":
        result = bpy.ops.wm.stl_import(filepath=str(path))
        warnings.append(
            {
                "code": "hierarchy.flattened",
                "message": "STL does not carry component hierarchy or source names.",
                "component_ids": [],
            }
        )
        warnings.append(
            {
                "code": "units.assumed",
                "message": (
                    "STL has no unit declaration; imported coordinates are treated as meters."
                ),
                "component_ids": [],
            }
        )
    else:
        raise ValueError(f"unsupported source format: {suffix or '<none>'}")
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender importer did not finish: {sorted(result)}")
    return warnings


def _parent_depth(obj: bpy.types.Object) -> int:
    depth = 0
    parent = obj.parent
    while parent is not None:
        depth += 1
        parent = parent.parent
    return depth


def apply_unit_scale(scale: float) -> None:
    """Bake a uniform scale about the world origin into the imported geometry, in place.

    The correction is baked into the mesh data (not left in the object transforms) so that
    *every* dimension the manifest reports — component-local bounds, world bounds, root
    bounds, camera pose and camera intrinsics — comes out in the corrected units. Scaling
    only the transforms would leave `obj.bound_box`-derived local bounds and the camera's
    scene-unit intrinsics (ortho scale, clip planes, focus distance) at the original,
    wrong scale.

    Mesh geometry is scaled once per unique datablock; each object's world *position* is
    scaled about the origin (rotation/object-scale untouched) via matrix_world, which
    reproduces an exact uniform scale about the origin for any hierarchy — including glTF's
    parent-inverse matrices — because it is expressed in world space directly.
    """
    if scale == 1.0:
        return
    scale_matrix = Matrix.Scale(scale, 4)
    scaled_meshes: set[str] = set()
    scaled_cameras: set[str] = set()
    for obj in bpy.context.scene.objects:
        data = obj.data
        if isinstance(data, bpy.types.Mesh) and data.name not in scaled_meshes:
            data.transform(scale_matrix, shape_keys=True)
            scaled_meshes.add(data.name)
    ordered = sorted(bpy.context.scene.objects, key=_parent_depth)
    targets = {obj: obj.matrix_world.copy() for obj in ordered}
    for obj in ordered:
        world = targets[obj]
        world.translation = world.translation * scale
        obj.matrix_world = world
    for obj in bpy.context.scene.objects:
        if obj.type != "CAMERA":
            continue
        camera = obj.data
        if camera.name in scaled_cameras:
            continue
        scaled_cameras.add(camera.name)
        camera.clip_start *= scale
        camera.clip_end *= scale
        if camera.type == "ORTHO":
            camera.ortho_scale *= scale
        camera.dof.focus_distance *= scale
    bpy.context.view_layer.update()


def coordinate_frames(source_format: str) -> dict[str, Any]:
    identity = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    source_to_world = identity
    source_name = f"{source_format}_z_up_assumed"
    up_axis = "+z"
    if source_format in {"glb", "gltf", "obj"}:
        source_to_world = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        source_name = "gltf_y_up" if source_format in {"glb", "gltf"} else "obj_y_up_assumed"
        up_axis = "+y"
    return {
        "source": {
            "name": source_name,
            "frame": "source",
            "handedness": "right",
            "up_axis": up_axis,
        },
        "world": {
            "name": "meshprobe_z_up",
            "frame": "world",
            "handedness": "right",
            "up_axis": "+z",
        },
        "source_to_world": source_to_world,
    }


def build_manifest(
    source_path: Path, source_sha256: str, import_warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    objects = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH"), key=object_path
    )
    if not objects:
        raise ValueError("the imported scene contains no mesh components")

    component_objects = set(objects)
    path_objects = set(objects)
    for obj in objects:
        current = obj.parent
        while current is not None:
            path_objects.add(current)
            current = current.parent
    source_names, names_preserved = source_names_for_objects(
        source_path, sorted(path_objects, key=object_path)
    )
    paths = source_paths(objects, source_names)
    objects.sort(key=paths.__getitem__)
    ids = {obj: stable_component_id(source_sha256, paths[obj]) for obj in objects}
    children: dict[bpy.types.Object, list[str]] = {obj: [] for obj in objects}
    parents: dict[bpy.types.Object, bpy.types.Object | None] = {}
    for obj in objects:
        parent = nearest_component_parent(obj, component_objects)
        parents[obj] = parent
        if parent is not None:
            children[parent].append(ids[obj])

    components: list[dict[str, Any]] = []
    all_world_points: list[Vector] = []
    missing_texture_count = 0
    components_with_materials = 0
    texture_count = 0
    for obj in objects:
        local_bounds, world_bounds = object_bounds(obj)
        all_world_points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
        materials, object_texture_count = material_summary(obj)
        components_with_materials += bool(materials["names"])
        texture_count += object_texture_count
        missing_texture_count += len(materials["missing_textures"])
        components.append(
            {
                "id": ids[obj],
                "path": paths[obj],
                "display_name": source_names[obj],
                "source_name": source_names[obj],
                "parent_id": ids[parents[obj]] if parents[obj] is not None else None,
                "child_ids": sorted(children[obj]),
                "mesh_hash": mesh_hash(obj.data),
                "world_transform": matrix_in_millimeters(obj),
                "world_transform_frame": "world",
                "local_bounds": local_bounds,
                "world_bounds": world_bounds,
                "materials": materials,
            }
        )

    root_bounds = bounds_of_points(all_world_points)
    root_bounds["frame"] = "world"  # type: ignore[assignment]
    camera, camera_warning = imported_camera(root_bounds)
    illumination, illumination_warnings = imported_illumination()
    source_format = source_path.suffix.lower().removeprefix(".")
    document = gltf_document(source_path) if source_format in {"glb", "gltf"} else {}
    animations = document.get("animations", [])
    has_animations = isinstance(animations, list) and bool(animations)
    extensions = document.get("extensionsUsed", [])
    extension_names = (
        {name for name in extensions if isinstance(name, str)}
        if isinstance(extensions, list)
        else set()
    )
    unverified_extensions = sorted(extension_names - SUPPORTED_GLTF_EXTENSIONS)
    procedural_extensions = [
        name
        for name in unverified_extensions
        if "procedural" in name.casefold() or "materialx" in name.casefold()
    ]
    hierarchy_flattened = source_format == "stl" or has_unaddressable_hierarchy(objects)
    warnings = [*import_warnings, *illumination_warnings]
    bounds_warning = plausibility_warning(root_bounds)
    if bounds_warning is not None:
        warnings.append(bounds_warning)
    if hierarchy_flattened and not any(
        warning["code"] == "hierarchy.flattened" for warning in warnings
    ):
        warnings.append(
            {
                "code": "hierarchy.flattened",
                "message": (
                    "The source contains non-mesh hierarchy nodes; component parent links "
                    "represent the addressable mesh hierarchy."
                ),
                "component_ids": [],
            }
        )
    if camera_warning is not None:
        warnings.append(camera_warning)

    if has_animations:
        warnings.append(
            {
                "code": "animation.static_pose",
                "message": (
                    "The source contains animation data; MeshProbe inspects the imported "
                    "rest state and does not evaluate animation timelines."
                ),
                "component_ids": [],
            }
        )
    if unverified_extensions:
        warnings.append(
            {
                "code": "extension.unverified",
                "message": (
                    "The source declares glTF extensions outside MeshProbe's verified import "
                    f"set: {', '.join(unverified_extensions)}."
                ),
                "component_ids": [],
            }
        )
    if procedural_extensions:
        warnings.append(
            {
                "code": "material.procedural_unsupported",
                "message": (
                    "Procedural material extensions are not evaluated: "
                    f"{', '.join(procedural_extensions)}."
                ),
                "component_ids": [],
            }
        )

    if source_format in {"glb", "gltf"} and camera_warning is None:
        warnings.append(
            {
                "code": "camera.lens_reconstructed",
                "message": (
                    "glTF stores angular field of view, not physical lens metadata; Blender "
                    "reconstructed an optically equivalent lens and sensor combination."
                ),
                "component_ids": [],
            }
        )
    texture_capability = "absent"
    if texture_count:
        texture_capability = "partial" if missing_texture_count else "preserved"
    material_capability = "absent"
    if components_with_materials:
        material_capability = (
            "preserved" if components_with_materials == len(components) else "partial"
        )
    return {
        "schema_version": 2,
        "source_sha256": source_sha256,
        "source_format": source_format,
        "units": "millimeter",
        "coordinate_frames": coordinate_frames(source_format),
        "root_bounds": root_bounds,
        "components": components,
        "imported_camera": camera,
        "imported_illumination": illumination,
        "capabilities": {
            "hierarchy": "flattened" if hierarchy_flattened else "preserved",
            "component_names": "source" if names_preserved else "generated",
            "materials": material_capability,
            "textures": texture_capability,
            "animations": "static_pose" if has_animations else "absent",
            "procedural_materials": "unsupported" if procedural_extensions else "absent",
        },
        "warnings": warnings,
    }


def has_unaddressable_hierarchy(objects: list[bpy.types.Object]) -> bool:
    component_objects = set(objects)
    for obj in objects:
        current = obj.parent
        while current is not None:
            if current not in component_objects:
                return True
            current = current.parent
    return False


def source_names_for_objects(
    source_path: Path,
    objects: list[bpy.types.Object],
) -> tuple[dict[bpy.types.Object, str], bool]:
    original_names = source_node_names(source_path)
    available = Counter(original_names)
    names: dict[bpy.types.Object, str] = {}
    preserved = source_path.suffix.lower() != ".stl"
    for obj in objects:
        if available[obj.name] <= 0:
            continue
        names[obj] = obj.name
        available[obj.name] -= 1
    for obj in objects:
        if obj in names:
            continue
        base, separator, suffix = obj.name.rpartition(".")
        if separator and suffix.isdigit() and len(suffix) == 3 and available[base] > 0:
            names[obj] = base
            available[base] -= 1
            continue
        names[obj] = obj.name
        preserved = False
    return names, preserved


def source_paths(
    components: list[bpy.types.Object], source_names: dict[bpy.types.Object, str]
) -> dict[bpy.types.Object, str]:
    nodes = set(components)
    for component in components:
        ancestor = component.parent
        while ancestor is not None:
            nodes.add(ancestor)
            ancestor = ancestor.parent

    siblings: dict[tuple[bpy.types.Object | None, str], list[bpy.types.Object]] = {}
    for node in nodes:
        name = source_names[node]
        siblings.setdefault((node.parent, name), []).append(node)

    segments: dict[bpy.types.Object, str] = {}
    for (_, name), matching_nodes in siblings.items():
        ordered = sorted(matching_nodes, key=lambda node: node.name)
        for index, node in enumerate(ordered, start=1):
            suffix = f"~{index}" if len(ordered) > 1 else ""
            segments[node] = escape_path_segment(f"{name}{suffix}")

    paths: dict[bpy.types.Object, str] = {}
    for component in components:
        path_segments: list[str] = []
        current: bpy.types.Object | None = component
        while current is not None:
            path_segments.append(segments[current])
            current = current.parent
        paths[component] = "/".join(reversed(path_segments))
    return paths


def source_node_names(source_path: Path) -> tuple[str, ...]:
    suffix = source_path.suffix.lower()
    if suffix == ".obj":
        names: list[str] = []
        for line in source_path.read_text(encoding="utf-8-sig").splitlines():
            tokens = shlex.split(line, comments=True, posix=True)
            if len(tokens) >= 2 and tokens[0].lower() in {"o", "g"}:
                names.extend(tokens[1:])
        return tuple(names)
    if suffix not in {".glb", ".gltf"}:
        return ()
    document = gltf_document(source_path)
    nodes = document.get("nodes", [])
    if not isinstance(nodes, list):
        return ()
    return tuple(
        node["name"]
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("name"), str)
    )


def gltf_document(source_path: Path) -> dict[str, Any]:
    if source_path.suffix.lower() == ".gltf":
        document = json.loads(source_path.read_text(encoding="utf-8"))
        return document if isinstance(document, dict) else {}
    payload = source_path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        return {}
    offset = 12
    while offset + 8 <= len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type != 0x4E4F534A:
            continue
        document = json.loads(chunk.rstrip(b"\x00 \t\r\n").decode("utf-8"))
        return document if isinstance(document, dict) else {}
    return {}


def scene_open(command: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(command["source_path"]).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"source is not a file: {source_path}")
    source_sha256 = command.get("source_sha256") or sha256_file(source_path)
    unit_scale = float(command.get("unit_scale", 1.0))
    if not math.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("unit_scale must be a positive finite number")
    aspect_ratio = command.get("aspect_ratio", 1.0)
    validate_camera_diagnostics_inputs((), aspect_ratio)
    clear_session_state()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_warnings = import_source(source_path)
    if unit_scale != 1.0:
        apply_unit_scale(unit_scale)
    manifest = build_manifest(source_path, source_sha256, import_warnings)
    initialize_session(manifest, source_path, aspect_ratio)
    return manifest


def clear_session_state() -> None:
    global MANIFEST, COMPONENT_OBJECTS, ORIGINAL_MESHES, ORIGINAL_VISIBILITY
    global COMPONENT_STATES, IMPORTED_CAMERA, CURRENT_CAMERA, CURRENT_CAMERA_DIAGNOSTICS
    global CURRENT_CAMERA_OPERATION, CURRENT_PERSPECTIVE_SENSOR_INTRINSICS
    global IMPORTED_ILLUMINATION, CURRENT_ILLUMINATION, MARK_OBJECTS
    MANIFEST = None
    COMPONENT_OBJECTS = {}
    ORIGINAL_MESHES = {}
    ORIGINAL_VISIBILITY = {}
    COMPONENT_STATES = {}
    IMPORTED_CAMERA = None
    CURRENT_CAMERA = None
    CURRENT_CAMERA_DIAGNOSTICS = None
    CURRENT_CAMERA_OPERATION = None
    CURRENT_PERSPECTIVE_SENSOR_INTRINSICS = None
    IMPORTED_ILLUMINATION = None
    CURRENT_ILLUMINATION = None
    MARK_OBJECTS = {}


def dispatch(command: dict[str, Any]) -> dict[str, Any]:
    operation = command.get("op")
    if operation == "scene.open":
        return scene_open(command)
    if operation == "session.snapshot":
        return {"scene": require_session(), "session": session_snapshot()}
    if operation == "view.set":
        require_session()
        global CURRENT_CAMERA_OPERATION
        apply_camera(
            command["camera"],
            command.get("focus_component_ids", ()),
            command.get("aspect_ratio", 1.0),
        )
        # ``apply_camera`` validates the full request before publishing camera state.
        # Keep the hidden physical sensor pair equally transactional: a rejected
        # view.set must not affect the next bare AUTO-fit orbit.
        remember_perspective_sensor_intrinsics(command["camera"]["projection"])
        CURRENT_CAMERA_OPERATION = None
        return session_snapshot()
    if operation == "view.orbit":
        require_session()
        return orbit_camera(command)
    if operation == "view.move":
        require_session()
        return move_camera(command)
    if operation == "view.rotate":
        require_session()
        return rotate_camera(command)
    if operation == "illumination.set":
        require_session()
        return apply_illumination(command["illumination"])
    if operation == "component.display":
        require_session()
        return component_display(command)
    if operation == "component.mark":
        require_session()
        return component_mark(command)
    if operation == "session.reset":
        require_session()
        return reset_session(command)
    if operation == "session.runtime":
        require_session()
        return runtime_diagnostics()
    if operation == "render.image":
        require_session()
        return render_image(command)
    if operation == "render.style":
        require_session()
        configure_render_style(command)
        return {"style": command.get("style", "screen_edges")}
    if operation == "scene.export_normalized":
        return export_normalized(command)
    if operation == "component.visibility":
        require_session()
        return component_visibility(command)
    if operation == "component.occluders":
        require_session()
        return rank_occluders(command)
    if operation == "session.shutdown":
        return {"shutdown": True}
    raise ValueError(f"unsupported worker operation: {operation}")


def main() -> None:
    global GPU_PLATFORM
    GPU_PLATFORM = initialize_graphics_platform()
    emit(
        {
            "event": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "blender_version": bpy.app.version_string,
            "pid": os.getpid(),
            "graphics": GPU_PLATFORM,
        }
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id: str | None = None
        try:
            command = json.loads(line)
            request_id = command.get("request_id")
            result = dispatch(command)
            emit({"request_id": request_id, "ok": True, "result": result})
            if command.get("op") == "session.shutdown":
                return
        except Exception as error:
            emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": {
                        "code": f"worker.{type(error).__name__}",
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            )


if __name__ == "__main__":
    main()
