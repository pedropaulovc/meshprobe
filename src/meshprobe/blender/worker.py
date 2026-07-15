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
import struct
import sys
import traceback
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

PROTOCOL_VERSION = 1
MILLIMETERS_PER_METER = 1_000.0


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
    return bounds_of_points(local_points), bounds_of_points(world_points)


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


def material_summary(obj: bpy.types.Object) -> dict[str, list[str]]:
    names: list[str] = []
    missing_textures: list[str] = []
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
            image = node.image
            if image is None:
                missing_textures.append(f"{material.name}:{node.name}")
                continue
            if image.packed_file is not None or not image.filepath:
                continue
            resolved = Path(bpy.path.abspath(image.filepath))
            if not resolved.exists():
                missing_textures.append(str(resolved))
    return {
        "names": sorted(set(names)),
        "missing_textures": sorted(set(missing_textures)),
    }


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
                "sensor_fit": "vertical" if data.sensor_fit == "VERTICAL" else "horizontal",
                "near_clip_mm": data.clip_start * MILLIMETERS_PER_METER,
                "far_clip_mm": data.clip_end * MILLIMETERS_PER_METER,
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
                "sensor_fit": "horizontal",
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
        if data.energy <= 0:
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
            "ambient_strength": 0.0,
            "lights": converted,
        },
        [],
    )


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


def build_manifest(
    source_path: Path, source_sha256: str, import_warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    objects = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH"), key=object_path
    )
    if not objects:
        raise ValueError("the imported scene contains no mesh components")

    component_objects = set(objects)
    ids = {obj: stable_component_id(source_sha256, object_path(obj)) for obj in objects}
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
    material_count = 0
    for obj in objects:
        local_bounds, world_bounds = object_bounds(obj)
        all_world_points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
        materials = material_summary(obj)
        material_count += len(materials["names"])
        missing_texture_count += len(materials["missing_textures"])
        components.append(
            {
                "id": ids[obj],
                "path": object_path(obj),
                "display_name": obj.name,
                "source_name": obj.name,
                "parent_id": ids[parents[obj]] if parents[obj] is not None else None,
                "child_ids": sorted(children[obj]),
                "mesh_hash": mesh_hash(obj.data),
                "world_transform": matrix_in_millimeters(obj),
                "local_bounds": local_bounds,
                "world_bounds": world_bounds,
                "materials": materials,
            }
        )

    root_bounds = bounds_of_points(all_world_points)
    camera, camera_warning = imported_camera(root_bounds)
    illumination, illumination_warnings = imported_illumination()
    warnings = [*import_warnings, *illumination_warnings]
    if camera_warning is not None:
        warnings.append(camera_warning)

    source_format = source_path.suffix.lower().removeprefix(".")
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
    flattened = source_format == "stl"
    texture_capability = "absent"
    if material_count:
        texture_capability = "partial" if missing_texture_count else "preserved"
    return {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "source_format": source_format,
        "units": "millimeter",
        "root_bounds": root_bounds,
        "components": components,
        "imported_camera": camera,
        "imported_illumination": illumination,
        "capabilities": {
            "hierarchy": "flattened" if flattened else "preserved",
            "component_names": "generated" if flattened else "source",
            "materials": "absent" if material_count == 0 else "preserved",
            "textures": texture_capability,
        },
        "warnings": warnings,
    }


def scene_open(command: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(command["source_path"]).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"source is not a file: {source_path}")
    source_sha256 = command["source_sha256"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_warnings = import_source(source_path)
    return build_manifest(source_path, source_sha256, import_warnings)


def dispatch(command: dict[str, Any]) -> dict[str, Any]:
    operation = command.get("op")
    if operation == "scene.open":
        return scene_open(command)
    if operation == "session.shutdown":
        return {"shutdown": True}
    raise ValueError(f"unsupported worker operation: {operation}")


def main() -> None:
    emit(
        {
            "event": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "blender_version": bpy.app.version_string,
            "pid": os.getpid(),
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
