"""Factory-clean Blender batch builder for curated evaluation assemblies."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]


def arguments() -> tuple[Path, Path]:
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        raise ValueError("curated builder expects jobs.json and checkpoint.json")
    return Path(args[0]).resolve(strict=True), Path(args[1]).resolve()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def imported_objects(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"curated source import failed: {path}")
    return [obj for obj in bpy.data.objects if obj not in before]


def parent_preserving_world(obj: bpy.types.Object, parent: bpy.types.Object) -> None:
    matrix = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = matrix


def mesh_span(objects: list[bpy.types.Object]) -> float:
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not points:
        raise ValueError("curated source contains no mesh objects")
    minimum = Vector(min(point[axis] for point in points) for axis in range(3))
    maximum = Vector(max(point[axis] for point in points) for axis in range(3))
    return float(max((maximum - minimum).length, 1e-6))


def add_camera_and_light(variant: str) -> None:
    camera_data = bpy.data.cameras.new("inspection_camera")
    camera = bpy.data.objects.new("inspection_camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (5.5, -8.0, 4.5)
    camera.rotation_euler = (math.radians(67), 0, math.radians(34))
    camera_data.lens = 24 if variant == "start_state" else 50
    bpy.context.scene.camera = camera
    light_data = bpy.data.lights.new("inspection_key", "AREA")
    light_data.energy = 1_000 if variant == "start_state" else 700
    light_data.shape = "DISK"
    light_data.size = 5
    light = bpy.data.objects.new("inspection_key", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (-4, -5, 6) if variant == "start_state" else (4, -5, 6)


def build(job: dict[str, Any]) -> dict[str, object]:
    clear_scene()
    variant = str(job["variant"])
    assembly_root = bpy.data.objects.new("assembly", None)
    bpy.context.scene.collection.objects.link(assembly_root)
    placements = list(job["placements"])
    if variant == "reorder_hierarchy":
        placements.reverse()
    roles: dict[str, list[str]] = {}
    for placement_index, placement in enumerate(placements):
        role = str(placement["role"])
        group = bpy.data.objects.new(f"group_{placement_index + 1:02d}", None)
        bpy.context.scene.collection.objects.link(group)
        group.parent = assembly_root
        objects = imported_objects(Path(str(placement["path"])).resolve(strict=True))
        top_level = [obj for obj in objects if obj.parent not in objects]
        for obj in top_level:
            parent_preserving_world(obj, group)
        scale = 1 / mesh_span(objects)
        group.scale = (scale, scale, scale)
        group.location = tuple(float(value) / 1_000 for value in placement["position_mm"])
        role_names: list[str] = []
        meshes = sorted((obj for obj in objects if obj.type == "MESH"), key=lambda obj: obj.name)
        for mesh_index, obj in enumerate(meshes, start=1):
            suffix = f"part_{mesh_index:02d}" if variant == "rename" else obj.name
            obj.name = f"{role}__{suffix}"
            role_names.append(obj.name)
        roles[role] = role_names

    if variant == "rigid_transform":
        assembly_root.location = (1.7, -0.8, 0.6)
        assembly_root.rotation_euler[2] = math.radians(37)
    if variant == "rescale":
        assembly_root.scale = (0.1, 0.1, 0.1)
    if variant == "mirror":
        assembly_root.scale.x = -1
    if variant == "material":
        material = bpy.data.materials.new("curated_diagnostic_blue")
        material.diffuse_color = (0.08, 0.25, 0.7, 1)
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(material)
    if variant == "extra_occluder":
        bpy.ops.mesh.primitive_cube_add(location=(0.2, -0.55, 0.25), scale=(0.45, 0.12, 0.45))
        occluder = bpy.context.active_object
        occluder.name = "distractor__irrelevant_panel"
        occluder.parent = assembly_root
        roles["distractor"] = [occluder.name]
    add_camera_and_light(variant)
    output = Path(str(job["output"])).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_cameras=True,
        export_lights=True,
    )
    if "FINISHED" not in result or not output.is_file():
        raise RuntimeError(f"curated assembly export failed: {output}")
    return {"roles": roles, "variant": variant, "model_file": output.name}


def atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(pending, path)


def main() -> None:
    jobs_path, checkpoint_path = arguments()
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    build_id = str(payload["build_id"])
    jobs = list(payload["jobs"])
    checkpoint: dict[str, Any] = {"build_id": build_id, "completed": []}
    if checkpoint_path.is_file():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if candidate.get("build_id") == build_id:
            checkpoint = candidate
    completed = set(checkpoint.get("completed", []))
    for job in jobs:
        key = str(job["key"])
        output = Path(str(job["output"]))
        metadata = Path(str(job["metadata"]))
        if key in completed and output.is_file() and metadata.is_file():
            continue
        result = build(job)
        atomic_json(metadata, result)
        completed.add(key)
        atomic_json(
            checkpoint_path,
            {"build_id": build_id, "completed": sorted(completed)},
        )


if __name__ == "__main__":
    main()
