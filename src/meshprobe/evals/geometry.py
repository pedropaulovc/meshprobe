"""Small deterministic GLB writer used by procedural evaluation families."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from meshprobe.sources import sha256_file

Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class MeshGeometry:
    vertices: tuple[Vec3, ...]
    triangles: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3 or not self.triangles:
            raise ValueError("mesh geometry requires vertices and triangles")
        if any(
            index < 0 or index >= len(self.vertices) for face in self.triangles for index in face
        ):
            raise ValueError("triangle index is outside the vertex array")


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    base_color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.55

    def __post_init__(self) -> None:
        if any(channel < 0 or channel > 1 for channel in self.base_color):
            raise ValueError("material base color channels must be between zero and one")
        if not 0 <= self.metallic <= 1 or not 0 <= self.roughness <= 1:
            raise ValueError("material metallic and roughness must be between zero and one")


@dataclass(frozen=True)
class ComponentSpec:
    role: str
    name: str
    geometry: MeshGeometry
    material: MaterialSpec | None
    parent_role: str | None = None
    translation_m: Vec3 = (0.0, 0.0, 0.0)
    rotation_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)
    scale: Vec3 = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position_m: Vec3
    target_m: Vec3
    projection: str = "perspective"
    focal_length_mm: float = 50.0
    sensor_height_mm: float = 24.0
    orthographic_height_m: float = 8.0

    def __post_init__(self) -> None:
        if self.projection not in {"perspective", "orthographic"}:
            raise ValueError(f"unknown camera projection: {self.projection}")
        if self.position_m == self.target_m:
            raise ValueError("camera position and target must differ")


@dataclass(frozen=True)
class PointLightSpec:
    name: str
    position_m: Vec3
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1_000.0


@dataclass
class GlbScene:
    components: list[ComponentSpec] = field(default_factory=list)
    camera: CameraSpec | None = None
    lights: list[PointLightSpec] = field(default_factory=list)

    def add(self, component: ComponentSpec) -> None:
        if component.role in {item.role for item in self.components}:
            raise ValueError(f"duplicate component role: {component.role}")
        if component.parent_role is not None and component.parent_role not in {
            item.role for item in self.components
        }:
            raise ValueError(f"parent role must be added first: {component.parent_role}")
        self.components.append(component)

    def write_glb(self, output: Path, *, role_order: tuple[str, ...] | None = None) -> str:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_glb(role_order=role_order)
        output.write_bytes(payload)
        return sha256_file(output)

    def to_glb(self, *, role_order: tuple[str, ...] | None = None) -> bytes:
        components = self._ordered_components(role_order)
        material_specs = sorted(
            {component.material for component in components if component.material is not None},
            key=lambda item: item.name,
        )
        material_indices = {material: index for index, material in enumerate(material_specs)}
        binary = bytearray()
        buffer_views: list[dict[str, object]] = []
        accessors: list[dict[str, object]] = []
        meshes: list[dict[str, object]] = []

        for component in components:
            position_bytes = b"".join(
                struct.pack("<fff", *vertex) for vertex in component.geometry.vertices
            )
            position_view = _append_buffer(binary, position_bytes, 34962, buffer_views)
            minimum = [
                min(vertex[axis] for vertex in component.geometry.vertices) for axis in range(3)
            ]
            maximum = [
                max(vertex[axis] for vertex in component.geometry.vertices) for axis in range(3)
            ]
            position_accessor = len(accessors)
            accessors.append(
                {
                    "bufferView": position_view,
                    "componentType": 5126,
                    "count": len(component.geometry.vertices),
                    "type": "VEC3",
                    "min": minimum,
                    "max": maximum,
                }
            )
            flat_indices = [
                index for triangle in component.geometry.triangles for index in triangle
            ]
            index_bytes = b"".join(struct.pack("<I", index) for index in flat_indices)
            index_view = _append_buffer(binary, index_bytes, 34963, buffer_views)
            index_accessor = len(accessors)
            accessors.append(
                {
                    "bufferView": index_view,
                    "componentType": 5125,
                    "count": len(flat_indices),
                    "type": "SCALAR",
                    "min": [min(flat_indices)],
                    "max": [max(flat_indices)],
                }
            )
            primitive: dict[str, object] = {
                "attributes": {"POSITION": position_accessor},
                "indices": index_accessor,
                "mode": 4,
            }
            if component.material is not None:
                primitive["material"] = material_indices[component.material]
            meshes.append({"name": f"{component.name}-mesh", "primitives": [primitive]})

        nodes: list[dict[str, object]] = []
        role_indices = {component.role: index for index, component in enumerate(components)}
        children: dict[str, list[int]] = {component.role: [] for component in components}
        for component in components:
            if component.parent_role is not None:
                children[component.parent_role].append(role_indices[component.role])
        for mesh_index, component in enumerate(components):
            node: dict[str, object] = {
                "name": component.name,
                "mesh": mesh_index,
                "translation": list(component.translation_m),
                "rotation": list(component.rotation_xyzw),
                "scale": list(component.scale),
            }
            if children[component.role]:
                node["children"] = sorted(children[component.role])
            nodes.append(node)

        root_nodes = [
            role_indices[component.role]
            for component in components
            if component.parent_role is None
        ]
        cameras: list[dict[str, object]] = []
        if self.camera is not None:
            camera_index = len(nodes)
            nodes.append(
                {
                    "name": self.camera.name,
                    "camera": 0,
                    "translation": list(self.camera.position_m),
                    "rotation": list(
                        look_at_quaternion(self.camera.position_m, self.camera.target_m)
                    ),
                }
            )
            root_nodes.append(camera_index)
            cameras.append(_gltf_camera(self.camera))

        extensions: dict[str, object] = {}
        extensions_used: list[str] = []
        if self.lights:
            extensions_used.append("KHR_lights_punctual")
            light_records: list[dict[str, object]] = []
            for light_index, light in enumerate(self.lights):
                light_records.append(
                    {
                        "name": light.name,
                        "type": "point",
                        "color": list(light.color),
                        "intensity": light.intensity,
                    }
                )
                node_index = len(nodes)
                nodes.append(
                    {
                        "name": light.name,
                        "translation": list(light.position_m),
                        "extensions": {"KHR_lights_punctual": {"light": light_index}},
                    }
                )
                root_nodes.append(node_index)
            extensions["KHR_lights_punctual"] = {"lights": light_records}

        document: dict[str, object] = {
            "asset": {"version": "2.0", "generator": "MeshProbe procedural eval factory"},
            "scene": 0,
            "scenes": [{"name": "qualification-scene", "nodes": root_nodes}],
            "nodes": nodes,
            "meshes": meshes,
            "materials": [_gltf_material(material) for material in material_specs],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }
        if cameras:
            document["cameras"] = cameras
        if extensions_used:
            document["extensionsUsed"] = extensions_used
            document["extensions"] = extensions
        return _encode_glb(document, bytes(binary))

    def component_paths(self, *, role_order: tuple[str, ...] | None = None) -> dict[str, str]:
        components = self._ordered_components(role_order)
        by_role = {component.role: component for component in components}
        sibling_groups: dict[tuple[str | None, str], list[str]] = {}
        for component in components:
            sibling_groups.setdefault((component.parent_role, component.name), []).append(
                component.role
            )
        segments: dict[str, str] = {}
        for (_, name), roles in sibling_groups.items():
            for index, role in enumerate(roles, start=1):
                suffix = f"~{index}" if len(roles) > 1 else ""
                segments[role] = _escape_segment(f"{name}{suffix}")
        paths: dict[str, str] = {}
        for component in components:
            path_segments = [segments[component.role]]
            parent_role = component.parent_role
            while parent_role is not None:
                parent = by_role[parent_role]
                path_segments.append(segments[parent.role])
                parent_role = parent.parent_role
            paths[component.role] = "/".join(reversed(path_segments))
        return paths

    def _ordered_components(self, role_order: tuple[str, ...] | None) -> list[ComponentSpec]:
        if role_order is None:
            return list(self.components)
        if set(role_order) != {component.role for component in self.components}:
            raise ValueError("role_order must contain every component role exactly once")
        by_role = {component.role: component for component in self.components}
        ordered = [by_role[role] for role in role_order]
        positions = {component.role: index for index, component in enumerate(ordered)}
        for component in ordered:
            if (
                component.parent_role is not None
                and positions[component.parent_role] > positions[component.role]
            ):
                raise ValueError("role_order must place parents before children")
        return ordered


def box(size: Vec3) -> MeshGeometry:
    x, y, z = (value / 2 for value in size)
    vertices = (
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    )
    triangles = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    return MeshGeometry(vertices=vertices, triangles=triangles)


def cylinder(radius: float, depth: float, segments: int = 24) -> MeshGeometry:
    if radius <= 0 or depth <= 0 or segments < 3:
        raise ValueError("cylinder dimensions and segment count must be positive")
    half = depth / 2
    vertices: list[Vec3] = []
    for z in (-half, half):
        vertices.extend(
            (
                radius * math.cos(2 * math.pi * index / segments),
                radius * math.sin(2 * math.pi * index / segments),
                z,
            )
            for index in range(segments)
        )
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -half))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, half))
    triangles: list[tuple[int, int, int]] = []
    for index in range(segments):
        following = (index + 1) % segments
        top = index + segments
        top_following = following + segments
        triangles.extend(((index, following, top_following), (index, top_following, top)))
        triangles.append((bottom_center, following, index))
        triangles.append((top_center, top, top_following))
    return MeshGeometry(vertices=tuple(vertices), triangles=tuple(triangles))


def extruded_polygon(points_xy: tuple[tuple[float, float], ...], depth: float) -> MeshGeometry:
    if len(points_xy) < 3 or depth <= 0:
        raise ValueError("an extruded polygon requires at least three points and positive depth")
    half = depth / 2
    vertices = tuple((x, y, z) for z in (-half, half) for x, y in points_xy)
    count = len(points_xy)
    triangles: list[tuple[int, int, int]] = []
    for index in range(1, count - 1):
        triangles.append((0, index + 1, index))
        triangles.append((count, count + index, count + index + 1))
    for index in range(count):
        following = (index + 1) % count
        triangles.extend(
            ((index, following, count + following), (index, count + following, count + index))
        )
    return MeshGeometry(vertices=vertices, triangles=tuple(triangles))


def arrow(
    direction: str, length: float = 0.9, width: float = 0.45, depth: float = 0.04
) -> MeshGeometry:
    if direction not in {"left", "right", "up", "down"}:
        raise ValueError(f"unknown arrow direction: {direction}")
    points = (
        (-length / 2, -width / 6),
        (length / 6, -width / 6),
        (length / 6, -width / 2),
        (length / 2, 0.0),
        (length / 6, width / 2),
        (length / 6, width / 6),
        (-length / 2, width / 6),
    )
    rotations = {"right": 0.0, "up": math.pi / 2, "left": math.pi, "down": -math.pi / 2}
    angle = rotations[direction]
    rotated = tuple(
        (
            x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle),
        )
        for x, y in points
    )
    return extruded_polygon(rotated, depth)


def look_at_quaternion(position: Vec3, target: Vec3) -> Quaternion:
    direction = _normalize(
        (
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        )
    )
    backward: Vec3 = (-direction[0], -direction[1], -direction[2])
    world_up: Vec3 = (0.0, 0.0, 1.0)
    if abs(_dot(backward, world_up)) > 0.999:
        world_up = (0.0, 1.0, 0.0)
    right = _normalize(_cross(world_up, backward))
    up = _cross(backward, right)
    matrix = (
        (right[0], up[0], backward[0]),
        (right[1], up[1], backward[1]),
        (right[2], up[2], backward[2]),
    )
    return _matrix_quaternion(matrix)


def _append_buffer(
    binary: bytearray,
    payload: bytes,
    target: int,
    views: list[dict[str, object]],
) -> int:
    binary.extend(b"\x00" * (-len(binary) % 4))
    offset = len(binary)
    binary.extend(payload)
    view_index = len(views)
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target})
    return view_index


def _gltf_material(material: MaterialSpec) -> dict[str, object]:
    record: dict[str, object] = {
        "name": material.name,
        "pbrMetallicRoughness": {
            "baseColorFactor": list(material.base_color),
            "metallicFactor": material.metallic,
            "roughnessFactor": material.roughness,
        },
        "doubleSided": True,
    }
    if material.base_color[3] < 1:
        record["alphaMode"] = "BLEND"
    return record


def _gltf_camera(camera: CameraSpec) -> dict[str, object]:
    if camera.projection == "orthographic":
        return {
            "name": camera.name,
            "type": "orthographic",
            "orthographic": {
                "xmag": camera.orthographic_height_m * 16 / 18,
                "ymag": camera.orthographic_height_m / 2,
                "znear": 0.001,
                "zfar": 1_000.0,
            },
        }
    yfov = 2 * math.atan(camera.sensor_height_mm / (2 * camera.focal_length_mm))
    return {
        "name": camera.name,
        "type": "perspective",
        "perspective": {"yfov": yfov, "znear": 0.001, "zfar": 1_000.0},
    }


def _encode_glb(document: dict[str, object], binary: bytes) -> bytes:
    binary += b"\x00" * (-len(binary) % 4)
    document["buffers"] = [{"byteLength": len(binary)}]
    json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _escape_segment(name: str) -> str:
    return name.replace("%", "%25").replace("/", "%2F")


def _normalize(vector: Vec3) -> Vec3:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _matrix_quaternion(matrix: tuple[Vec3, Vec3, Vec3]) -> Quaternion:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quaternion = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, scale / 4)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2
        quaternion = (scale / 4, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2
        quaternion = ((m01 + m10) / scale, scale / 4, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2
        quaternion = ((m02 + m20) / scale, (m12 + m21) / scale, scale / 4, (m10 - m01) / scale)
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]
