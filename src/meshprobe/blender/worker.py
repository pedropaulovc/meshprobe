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
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any

import bpy  # type: ignore[import-not-found]
import gpu  # type: ignore[import-not-found]
from mathutils import Matrix, Quaternion, Vector  # type: ignore[import-not-found]

PROTOCOL_VERSION = 2
MILLIMETERS_PER_METER = 1_000.0
PRESET_REFERENCE_SPAN_MM = 5_000.0
DISPLAY_MODES = {"shown", "hidden", "isolated", "ghosted"}
MARK_MODES = {"unmarked", "selected", "highlighted", "labeled"}
MARK_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
MARK_COLORS = {
    "selected": (0.05, 0.8, 1.0, 1.0),
    "highlighted": (1.0, 0.2, 0.02, 1.0),
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
CURRENT_CAMERA: dict[str, Any] | None = None
CURRENT_CAMERA_DIAGNOSTICS: dict[str, Any] | None = None
CURRENT_CAMERA_OPERATION: dict[str, Any] | None = None
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


def apply_camera(
    camera: dict[str, Any],
    focus_component_ids: list[str] | tuple[str, ...] = (),
    aspect_ratio: float = 1.0,
    target_mm: list[float] | tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    validate_camera_diagnostics_inputs(focus_component_ids, aspect_ratio)
    obj = camera_object()
    pose = camera["pose"]
    position = Vector(value / MILLIMETERS_PER_METER for value in pose["position_mm"])
    x, y, z, w = pose["orientation_xyzw"]
    rotation = Quaternion((w, x, y, z)).normalized()
    resolved_camera = deepcopy(camera)
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


def orbit_camera(command: dict[str, Any]) -> dict[str, Any]:
    global CURRENT_CAMERA_OPERATION
    CURRENT_CAMERA_OPERATION = None
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
    camera = {
        "pose": {
            "position_mm": [value * MILLIMETERS_PER_METER for value in position],
            "orientation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
        },
        "projection": command["projection"],
    }
    return apply_camera(
        camera,
        command.get("focus_component_ids", ()),
        command.get("aspect_ratio", 1.0),
        command["target_mm"],
    )


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
        "projection": CURRENT_CAMERA["projection"],
    }
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


def rotate_camera(command: dict[str, Any]) -> dict[str, Any]:
    if CURRENT_CAMERA is None:
        raise ValueError("session camera is not initialized")
    focus_component_ids = command.get("focus_component_ids", ())
    aspect_ratio = command.get("aspect_ratio", 1.0)
    validate_camera_diagnostics_inputs(focus_component_ids, aspect_ratio)
    axis_index = {"x": 0, "y": 1, "z": 2}[command["axis"]]
    axis = Vector(tuple(1.0 if index == axis_index else 0.0 for index in range(3)))
    target = Vector(command["target_mm"])
    if command["frame"] == "source":
        axis = transform_from_source(axis, point=False)
        target = transform_from_source(target, point=True)
    axis.normalize()
    rotation = Quaternion(axis, math.radians(command["degrees"]))
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
    camera = {
        "pose": resulting_pose,
        "projection": command.get("projection") or CURRENT_CAMERA["projection"],
    }
    global CURRENT_CAMERA_OPERATION
    CURRENT_CAMERA_OPERATION = {
        "operation": "rotate",
        "frame": command["frame"],
        "target_mm": command["target_mm"],
        "axis": command["axis"],
        "axis_world": list(axis),
        "degrees": command["degrees"],
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
        definitions = [
            ("key", center + frame_offset(frame, (span, -span, span)), 900.0, span),
            (
                "fill",
                center + frame_offset(frame, (-span, -span / 2, span / 2)),
                450.0,
                span,
            ),
            ("rim", center + frame_offset(frame, (0, span, span)), 600.0, span / 2),
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
        background_override = illumination.get("background_rgb")
        strength_override = illumination.get("background_strength")
        if background_override is not None:
            runtime["background_rgb"] = background_override
            runtime["background_strength"] = 1.0
        if strength_override is not None:
            runtime["background_strength"] = strength_override
        resolved = {
            **deepcopy(illumination),
            "background_rgb": deepcopy(runtime["background_rgb"]),
            "background_strength": runtime["background_strength"],
        }
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
    label.rotation_mode = "QUATERNION"
    label.rotation_quaternion = camera_object().matrix_world.to_quaternion()
    color_suffix = COMPONENT_STATES[component_id]["mark_color"]
    material_name = "MeshProbeMark-labeled"
    if color_suffix is not None:
        material_name = f"{material_name}-{color_suffix[1:]}"
    data.materials.append(replacement_material(material_name, color))
    MARK_OBJECTS[component_id] = label


def orient_component_labels() -> None:
    orientation = camera_object().matrix_world.to_quaternion()
    for label in MARK_OBJECTS.values():
        label.rotation_quaternion = orientation


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


def reset_session() -> dict[str, Any]:
    if IMPORTED_CAMERA is None or IMPORTED_ILLUMINATION is None:
        raise ValueError("session state is not initialized")
    global CURRENT_CAMERA_OPERATION
    CURRENT_CAMERA_OPERATION = None
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
    apply_camera(deepcopy(IMPORTED_CAMERA))
    apply_illumination(deepcopy(IMPORTED_ILLUMINATION))
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
            "ambient_rgb": list(ambient_background.inputs["Color"].default_value[:3]),
            "ambient_strength": float(ambient_background.inputs["Strength"].default_value),
        },
        "render": {
            "engine": bpy.context.scene.render.engine,
            "eevee_samples": eevee_render_samples(),
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


def graphics_platform() -> dict[str, Any]:
    if GPU_PLATFORM is None:
        raise RuntimeError("graphics platform is not initialized")
    return GPU_PLATFORM


def initialize_graphics_platform() -> dict[str, Any]:
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


def emission_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = EMISSION_MATERIALS.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        EMISSION_MATERIALS[name] = material
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


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


def render_mask(path: Path, colors: dict[str, tuple[int, int, int]]) -> None:
    scene = bpy.context.scene
    original_meshes = {component_id: obj.data for component_id, obj in COMPONENT_OBJECTS.items()}
    other_visibility = {
        obj: obj.hide_render for obj in scene.objects if obj.type not in {"MESH", "CAMERA"}
    }
    original_world = scene.world
    mask_world = bpy.data.worlds.get("MeshProbeMaskWorld") or bpy.data.worlds.new(
        "MeshProbeMaskWorld"
    )
    mask_world.use_nodes = True
    background = mask_world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    scene.world = mask_world
    original_transform = scene.view_settings.view_transform
    original_look = scene.view_settings.look
    original_exposure = scene.view_settings.exposure
    original_gamma = scene.view_settings.gamma
    original_dither = scene.render.dither_intensity
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.dither_intensity = 0.0
    scene.render.engine = eevee_engine()
    try:
        for obj in other_visibility:
            obj.hide_render = True
        for component_id, obj in COMPONENT_OBJECTS.items():
            mesh = obj.data.copy()
            mesh.materials.clear()
            red, green, blue = colors[component_id]
            mesh.materials.append(
                emission_material(
                    f"MeshProbeMask-{red:02x}{green:02x}{blue:02x}",
                    (
                        srgb_channel_to_linear(red),
                        srgb_channel_to_linear(green),
                        srgb_channel_to_linear(blue),
                        1.0,
                    ),
                )
            )
            for polygon in mesh.polygons:
                polygon.material_index = 0
            obj.data = mesh
        render_still(path)
    finally:
        for component_id, obj in COMPONENT_OBJECTS.items():
            replacement = obj.data
            obj.data = original_meshes[component_id]
            if replacement.users == 0:
                bpy.data.meshes.remove(replacement)
        for obj, hide_render in other_visibility.items():
            obj.hide_render = hide_render
        scene.world = original_world
        scene.view_settings.view_transform = original_transform
        scene.view_settings.look = original_look
        scene.view_settings.exposure = original_exposure
        scene.view_settings.gamma = original_gamma
        scene.render.dither_intensity = original_dither


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
    if not 64 <= width <= 1_024 or not 64 <= height <= 1_024:
        raise ValueError("component.visibility dimensions must be between 64 and 1024")
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
    try:
        bpy.ops.render.render()
    finally:
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


def render_image(command: dict[str, Any]) -> dict[str, Any]:
    manifest = require_session()
    output = Path(command["output_path"]).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("render.image output_path must end in .png")
    device = configure_render(command)
    render_still(output)
    luminance = luminance_summary(output)
    evaluator = None
    if command.get("evaluator_output_dir") is not None:
        evaluator_dir = Path(command["evaluator_output_dir"]).expanduser().resolve()
        evaluator_dir.mkdir(parents=True, exist_ok=True)
        evaluator = render_evaluator_passes(evaluator_dir, output.stem)
    session = session_snapshot()
    return {
        "schema_version": 1,
        "source_sha256": manifest["source_sha256"],
        "state_sha256": session["state_sha256"],
        "width": command["width"],
        "height": command["height"],
        "samples": command["samples"],
        "engine": command["engine"],
        "device": device,
        "graphics_policy": command["graphics_policy"],
        "graphics": graphics_platform(),
        "blender_version": bpy.app.version_string,
        "session": session,
        "color": artifact(output, "image/png"),
        "evaluator": evaluator,
        "luminance": luminance,
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


def rank_occluders(command: dict[str, Any]) -> dict[str, Any]:
    focus_ids = selected_component_ids(command)
    camera_origin = camera_object().matrix_world.translation
    object_ids = {obj: component_id for component_id, obj in COMPONENT_OBJECTS.items()}
    points: list[Vector] = []
    for component_id in sorted(focus_ids):
        obj = COMPONENT_OBJECTS[component_id]
        vertices = obj.data.vertices
        stride = max(1, len(vertices) // 128)
        points.extend(
            obj.matrix_world @ vertices[index].co for index in range(0, len(vertices), stride)
        )
        points.append(obj.matrix_world.translation.copy())
    counts: dict[str, int] = {}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for point in points:
        direction = point - camera_origin
        distance = direction.length
        if distance <= 1e-9:
            continue
        hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(
            depsgraph, camera_origin, direction.normalized(), distance=distance + 1e-6
        )
        if not hit or hit_object is None:
            continue
        original = hit_object.original if hasattr(hit_object, "original") else hit_object
        hit_component_id = object_ids.get(original)
        if hit_component_id is None or hit_component_id in focus_ids:
            continue
        if COMPONENT_STATES[hit_component_id]["display"] == "hidden":
            continue
        counts[hit_component_id] = counts.get(hit_component_id, 0) + 1
    sample_count = len(points)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "focus_component_ids": sorted(focus_ids),
        "sample_count": sample_count,
        "occluders": [
            {
                "component_id": component_id,
                "blocked_rays": count,
                "fraction": count / sample_count if sample_count else 0.0,
            }
            for component_id, count in ranked
        ],
    }


def initialize_session(manifest: dict[str, Any], source_path: Path) -> None:
    global MANIFEST, COMPONENT_OBJECTS, ORIGINAL_MESHES, ORIGINAL_VISIBILITY
    global COMPONENT_STATES, IMPORTED_CAMERA, CURRENT_CAMERA, CURRENT_CAMERA_OPERATION
    global IMPORTED_ILLUMINATION, CURRENT_ILLUMINATION, MARK_OBJECTS
    global OVERRIDE_MATERIALS, EMISSION_MATERIALS

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
    CURRENT_CAMERA = deepcopy(IMPORTED_CAMERA)
    CURRENT_CAMERA_OPERATION = None
    IMPORTED_ILLUMINATION = deepcopy(manifest["imported_illumination"])
    CURRENT_ILLUMINATION = deepcopy(IMPORTED_ILLUMINATION)
    MARK_OBJECTS = {}
    OVERRIDE_MATERIALS = {}
    EMISSION_MATERIALS = {}
    apply_camera(deepcopy(IMPORTED_CAMERA))
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
        "schema_version": 1,
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
    clear_session_state()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_warnings = import_source(source_path)
    manifest = build_manifest(source_path, source_sha256, import_warnings)
    initialize_session(manifest, source_path)
    return manifest


def clear_session_state() -> None:
    global MANIFEST, COMPONENT_OBJECTS, ORIGINAL_MESHES, ORIGINAL_VISIBILITY
    global COMPONENT_STATES, IMPORTED_CAMERA, CURRENT_CAMERA, CURRENT_CAMERA_DIAGNOSTICS
    global CURRENT_CAMERA_OPERATION
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
        CURRENT_CAMERA_OPERATION = None
        return apply_camera(
            command["camera"],
            command.get("focus_component_ids", ()),
            command.get("aspect_ratio", 1.0),
        )
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
        return reset_session()
    if operation == "session.runtime":
        require_session()
        return runtime_diagnostics()
    if operation == "render.image":
        require_session()
        return render_image(command)
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
