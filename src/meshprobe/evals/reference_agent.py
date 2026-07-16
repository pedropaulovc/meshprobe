"""Standalone, evaluator-truth-isolated calibration agent.

The harness copies this file into each public episode input and runs it with the
standard library only.  It deliberately imports no MeshProbe or generator code.
Its purpose is to prove that a pinned corpus is answerable through the public
protocol and that every mandatory oracle can be satisfied; it is not presented
as a learned-model result.
"""

from __future__ import annotations

import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


class ProtocolError(RuntimeError):
    """The evaluation broker rejected or malformed a public tool response."""


class ProtocolClient:
    def __init__(self) -> None:
        self.sequence = 0

    def call(self, operation: str, **arguments: object) -> Any:
        self.sequence += 1
        command = {
            "request_id": f"reference-{self.sequence:03d}",
            "op": operation,
            **arguments,
        }
        self._send({"type": "tool_call", "command": command})
        line = sys.stdin.readline()
        if not line:
            raise ProtocolError("evaluation broker closed before returning a tool result")
        reply = json.loads(line)
        if not isinstance(reply, dict) or reply.get("type") != "tool_result":
            raise ProtocolError("evaluation broker returned an invalid tool envelope")
        if not reply.get("ok"):
            error = reply.get("error")
            raise ProtocolError(f"public tool rejected {operation}: {error}")
        response = reply.get("response")
        if not isinstance(response, dict) or "result" not in response:
            raise ProtocolError("evaluation broker omitted the command result")
        return response["result"]

    @staticmethod
    def submit(submission: JsonObject) -> None:
        ProtocolClient._send({"type": "submission", "submission": submission})

    @staticmethod
    def _send(payload: JsonObject) -> None:
        print(json.dumps(payload, separators=(",", ":")), flush=True)


def _quoted_names(prompt: str) -> list[str]:
    return re.findall(r"'([^']+)'", prompt)


def _center(component: JsonObject) -> tuple[float, float, float]:
    transform = component["world_transform"]
    return float(transform[3]), float(transform[7]), float(transform[11])


def _bounds_size(component: JsonObject, key: str = "local_bounds") -> tuple[float, float, float]:
    bounds = component[key]
    return tuple(
        float(high) - float(low)
        for low, high in zip(bounds["minimum_mm"], bounds["maximum_mm"], strict=True)
    )  # type: ignore[return-value]


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sub(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _unit(vector: tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(_dot(vector, vector))
    if length == 0:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(value / length for value in vector)


def _axis(component: JsonObject, index: int) -> tuple[float, float, float]:
    transform = component["world_transform"]
    return _unit(
        (
            float(transform[index]),
            float(transform[4 + index]),
            float(transform[8 + index]),
        )
    )  # type: ignore[return-value]


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-3, abs_tol=1e-3)


def _materials(component: JsonObject) -> set[str]:
    return set(component["materials"]["names"])


def _select_procedural_roles(
    manifest: JsonObject,
    target_matches: list[JsonObject],
) -> dict[str, JsonObject]:
    components = manifest["components"]
    if not isinstance(components, list):
        raise ValueError("scene manifest has no component list")
    by_id = {component["id"]: component for component in components}
    intended = [match for match in target_matches if match.get("child_ids")]
    target = intended[0] if len(intended) == 1 else target_matches[0]
    housing_id = target["parent_id"]
    siblings = [component for component in components if component.get("parent_id") == housing_id]
    children = [component for component in components if component.get("parent_id") == target["id"]]
    if len(children) != 1:
        raise ValueError("target does not have exactly one retaining-clip child")
    clip = children[0]

    grandchildren = [
        component
        for component in components
        if component.get("parent_id") in {sibling["id"] for sibling in siblings}
        and component.get("parent_id") != target["id"]
    ]
    if len(grandchildren) != 1:
        raise ValueError("assembly does not expose one isolated surface-detail child")
    arrow_component = grandchildren[0]

    cover_candidates = [
        component
        for component in siblings
        if _materials(component) & {"dark-polymer", "clear-shell"}
    ]
    if not cover_candidates:
        raise ValueError("assembly has no cover candidate")
    cover = max(cover_candidates, key=lambda item: math.prod(_bounds_size(item)))

    equal_geometry: dict[tuple[float, float, float], list[JsonObject]] = {}
    for component in siblings:
        dimensions = sorted(_bounds_size(component))
        key = (
            round(dimensions[0], 3),
            round(dimensions[1], 3),
            round(dimensions[2], 3),
        )
        equal_geometry.setdefault(key, []).append(component)
    gap_pairs = [
        group
        for group in equal_geometry.values()
        if len(group) == 2 and all("dark-polymer" in _materials(item) for item in group)
    ]
    if len(gap_pairs) != 1:
        raise ValueError("assembly does not expose one matching clearance-block pair")
    gap_left, gap_right = sorted(gap_pairs[0], key=lambda item: _center(item))

    steel = [component for component in siblings if "brushed-steel" in _materials(component)]
    shafts: list[JsonObject] = []
    rings: list[JsonObject] = []
    for component in steel:
        small, middle, large = sorted(_bounds_size(component))
        if _same(small, middle) and large > 2 * small:
            shafts.append(component)
        if _same(middle, large) and large > 2 * small:
            rings.append(component)
    if len(shafts) != 2 or len(rings) != 2:
        raise ValueError("assembly shaft or ring geometry is not uniquely classifiable")
    contacted_shaft = min(shafts, key=lambda item: _distance(_center(item), _center(target)))
    camera_position = tuple(
        float(value) for value in manifest["imported_camera"]["pose"]["position_mm"]
    )
    nearer_ring = min(rings, key=lambda item: _distance(_center(item), camera_position))

    return {
        "target": by_id[target["id"]],
        "clip": clip,
        "cover": cover,
        "arrow": arrow_component,
        "gap_left": gap_left,
        "gap_right": gap_right,
        "contacted_shaft": contacted_shaft,
        "nearer_ring": nearer_ring,
    }


def _clip_side(target: JsonObject, clip: JsonObject) -> str:
    transform = target["world_transform"]
    local_x_world = (float(transform[0]), float(transform[4]), float(transform[8]))
    offset = _sub(_center(clip), _center(target))
    return "positive" if _dot(local_x_world, offset) > 0 else "negative"


def _oriented_half_extent(component: JsonObject, direction: tuple[float, ...]) -> float:
    transform = component["world_transform"]
    columns = (
        (float(transform[0]), float(transform[4]), float(transform[8])),
        (float(transform[1]), float(transform[5]), float(transform[9])),
        (float(transform[2]), float(transform[6]), float(transform[10])),
    )
    half_sizes = tuple(value / 2 for value in _bounds_size(component))
    return sum(
        abs(_dot(column, direction)) * half
        for column, half in zip(columns, half_sizes, strict=True)
    )


def _clearance(left: JsonObject, right: JsonObject) -> float:
    delta = _sub(_center(right), _center(left))
    direction = _unit(delta)
    gap = (
        _distance(_center(left), _center(right))
        - _oriented_half_extent(left, direction)
        - _oriented_half_extent(right, direction)
    )
    return round(gap, 3)


def _glb_chunks(path: Path) -> tuple[JsonObject, bytes]:
    payload = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or total_length != len(payload):
        raise ValueError("assigned model is not a valid GLB 2.0 file")
    offset = 12
    document: JsonObject | None = None
    binary = b""
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            parsed = json.loads(chunk.rstrip(b" \x00"))
            if not isinstance(parsed, dict):
                raise ValueError("GLB JSON chunk is not an object")
            document = parsed
        if chunk_type == 0x004E4942:
            binary = chunk
    if document is None or not binary:
        raise ValueError("GLB omits its JSON or binary chunk")
    return document, binary


def _component_vertices(path: Path, component_name: str) -> list[tuple[float, float, float]]:
    document, binary = _glb_chunks(path)
    nodes = document["nodes"]
    matching = [node for node in nodes if node.get("name") == component_name]
    if len(matching) != 1:
        raise ValueError("component name does not identify one GLB node")
    primitive = document["meshes"][matching[0]["mesh"]]["primitives"][0]
    accessor = document["accessors"][primitive["attributes"]["POSITION"]]
    if accessor["componentType"] != 5126 or accessor["type"] != "VEC3":
        raise ValueError("component positions are not float VEC3 values")
    view = document["bufferViews"][accessor["bufferView"]]
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", 12))
    return [
        struct.unpack_from("<fff", binary, start + index * stride)
        for index in range(int(accessor["count"]))
    ]


def infer_arrow_direction(path: Path, component_name: str) -> str:
    points = {(round(x, 6), round(y, 6)) for x, y, _z in _component_vertices(path, component_name)}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    if x_span > y_span:
        minimum_count = sum(_same(x, min(xs)) for x in xs)
        maximum_count = sum(_same(x, max(xs)) for x in xs)
        return "left" if minimum_count < maximum_count else "right"
    minimum_count = sum(_same(y, min(ys)) for y in ys)
    maximum_count = sum(_same(y, max(ys)) for y in ys)
    return "down" if minimum_count < maximum_count else "up"


def _projection(mode: str, *, focal_length: float = 85, scale: float = 1) -> JsonObject:
    if mode == "perspective":
        return {
            "mode": "perspective",
            "focal_length_mm": focal_length,
            "sensor_width_mm": 36,
            "near_clip_mm": 1,
            "far_clip_mm": 500_000,
        }
    return {
        "mode": "orthographic",
        "scale_mm": max(scale, 1),
        "near_clip_mm": 1,
        "far_clip_mm": 500_000,
    }


def _orbit(
    client: ProtocolClient,
    component_ids: list[str],
    target: tuple[float, float, float],
    *,
    projection: JsonObject,
    distance: float,
    azimuth: float,
    elevation: float,
) -> None:
    client.call(
        "view.orbit",
        target_mm=list(target),
        azimuth_degrees=azimuth,
        elevation_degrees=elevation,
        roll_degrees=0,
        distance_mm=max(distance, 1),
        projection=projection,
        focus_component_ids=component_ids,
        aspect_ratio=1,
    )


def _orbit_direction(
    client: ProtocolClient,
    component_ids: list[str],
    target: tuple[float, float, float],
    direction: tuple[float, ...],
    *,
    projection: JsonObject,
    distance: float,
) -> None:
    unit = _unit(direction)
    _orbit(
        client,
        component_ids,
        target,
        projection=projection,
        distance=distance,
        azimuth=math.degrees(math.atan2(unit[1], unit[0])),
        elevation=math.degrees(math.asin(max(-1.0, min(1.0, unit[2])))),
    )


def _render(client: ProtocolClient, name: str) -> str:
    result = client.call(
        "render.image",
        output_path=name,
        width=192,
        height=192,
        samples=1,
        engine="eevee",
    )
    return str(result["color"]["path"])


def _set_preliminary_camera(client: ProtocolClient, manifest: JsonObject, target_id: str) -> None:
    camera = json.loads(json.dumps(manifest["imported_camera"]))
    position = camera["pose"]["position_mm"]
    position[0] = float(position[0]) + 137.0
    camera["projection"] = _projection("perspective", focal_length=70)
    client.call("view.set", camera=camera, focus_component_ids=[target_id], aspect_ratio=1)


def _run_curated(
    client: ProtocolClient,
    episode: JsonObject,
    manifest: JsonObject,
    required: set[str],
) -> tuple[JsonObject, list[str]]:
    names = _quoted_names(episode["prompt"])
    roles: dict[str, JsonObject] = {}
    for name in names:
        matches = client.call(
            "component.find",
            selector={"kind": "exact_name", "pattern": name},
        )
        if len(matches) != 1:
            raise ValueError(f"curated role name is not unique: {name}")
        role = name.split("__", 1)[0]
        roles[role] = matches[0]
    if "component.inspect" in required:
        for role in ("target", "reference", "occluder"):
            client.call("component.inspect", component_id=roles[role]["id"])

    values = {
        f"{role}_{suffix}": component[key]
        for role, component in roles.items()
        for suffix, key in (("component_id", "id"), ("path", "path"))
    }
    evidence: list[str] = []
    if "render.image" not in required:
        return values, evidence

    target = roles["target"]
    target_center = _center(target)
    extent = max(_bounds_size(target, "world_bounds"))
    if "view.set" in required:
        _set_preliminary_camera(client, manifest, target["id"])
    client.call("component.display", component_ids=[target["id"]], mode="isolated")
    client.call("component.mark", component_ids=[target["id"]], mode="highlighted")
    client.call("illumination.set", illumination={"preset": "neutral_studio"})
    _orbit(
        client,
        [target["id"]],
        target_center,
        projection=_projection("perspective"),
        distance=extent * 3,
        azimuth=315,
        elevation=25,
    )
    evidence.append(_render(client, "context.png"))
    client.call("illumination.set", illumination={"preset": "raking_left"})
    _orbit(
        client,
        [target["id"]],
        target_center,
        projection=_projection("orthographic", scale=extent * 1.5),
        distance=extent * 5,
        azimuth=315,
        elevation=25,
    )
    evidence.append(_render(client, "orthographic-rake.png"))
    if "render.contact_sheet" in required:
        sheet = client.call(
            "render.contact_sheet",
            output_path="contact-sheet.png",
            recipe="focused_3x3",
            focus_component_ids=[target["id"]],
            panel_width=128,
            panel_height=128,
            samples=1,
            engine="eevee",
            visibility_threshold=0.2,
            occluder_budget=3,
        )
        evidence.append(str(sheet["sheet"]["path"]))
    return values, evidence


def _run_procedural(
    client: ProtocolClient,
    episode: JsonObject,
    manifest: JsonObject,
    required: set[str],
) -> tuple[JsonObject, JsonObject, list[str]]:
    names = _quoted_names(episode["prompt"])
    if not names:
        raise ValueError("procedural prompt does not identify its target name")
    target_matches = client.call(
        "component.find",
        selector={"kind": "exact_name", "pattern": names[0]},
    )
    roles = _select_procedural_roles(manifest, target_matches)
    target = roles["target"]
    if "component.inspect" in required:
        for match in target_matches:
            client.call("component.inspect", component_id=match["id"])

    fields = set(episode["answer_schema"]["properties"]["values"]["properties"])
    values: JsonObject = {
        "target_component_id": target["id"],
        "target_component_path": target["path"],
    }
    if "contacted_shaft_component_id" in fields:
        values.update(
            {
                "contacted_shaft_component_id": roles["contacted_shaft"]["id"],
                "clip_side": _clip_side(target, roles["clip"]),
                "arrow_direction": infer_arrow_direction(
                    Path(episode["model_path"]), roles["arrow"]["display_name"]
                ),
                "nearer_ring_component_id": roles["nearer_ring"]["id"],
            }
        )
    if "clearance_mm" in fields:
        values["clearance_mm"] = _clearance(roles["gap_left"], roles["gap_right"])

    answer: JsonObject = {"status": "answered", "values": values}
    prompt = episode["prompt"]
    if "every conflicting stable component ID" in prompt:
        answer = {
            "status": "indeterminate",
            "values": {},
            "conflicting_component_ids": [match["id"] for match in target_matches],
        }
    if "arrow material" in prompt:
        answer = {
            "status": "indeterminate",
            "values": {},
            "missing_capabilities": ["arrow material"],
        }

    evidence: list[str] = []
    if "render.image" not in required:
        return values, answer, evidence
    if "view.set" in required:
        _set_preliminary_camera(client, manifest, target["id"])

    target_extent = max(_bounds_size(target, "world_bounds"))
    client.call(
        "component.display",
        component_ids=[
            target["id"],
            roles["clip"]["id"],
            roles["contacted_shaft"]["id"],
        ],
        mode="isolated",
    )
    client.call("component.mark", component_ids=[target["id"]], mode="highlighted")
    client.call("illumination.set", illumination={"preset": "neutral_studio"})
    context_direction = tuple(
        horizontal - normal + 0.65 * vertical
        for horizontal, normal, vertical in zip(
            _axis(target, 0), _axis(target, 1), _axis(target, 2), strict=True
        )
    )
    _orbit_direction(
        client,
        [target["id"]],
        _center(target),
        context_direction,
        projection=_projection("perspective"),
        distance=max(target_extent * 3, 1_000),
    )
    evidence.append(_render(client, "context.png"))

    arrow = roles["arrow"]
    arrow_extent = max(_bounds_size(arrow, "world_bounds"))
    client.call("component.display", component_ids=[arrow["id"]], mode="isolated")
    client.call("illumination.set", illumination={"preset": "raking_left"})
    arrow_normal = _axis(arrow, 1)
    arrow_horizontal = _axis(arrow, 0)
    arrow_vertical = _axis(arrow, 2)
    left_direction = tuple(
        normal + 2 * horizontal + 0.27 * vertical
        for normal, horizontal, vertical in zip(
            arrow_normal, arrow_horizontal, arrow_vertical, strict=True
        )
    )
    _orbit_direction(
        client,
        [arrow["id"]],
        _center(arrow),
        left_direction,
        projection=_projection("orthographic", scale=arrow_extent * 1.5),
        distance=max(arrow_extent * 5, 1_000),
    )
    evidence.append(_render(client, "surface-left.png"))
    client.call("illumination.set", illumination={"preset": "raking_right"})
    right_direction = tuple(
        normal - 2 * horizontal + 0.27 * vertical
        for normal, horizontal, vertical in zip(
            arrow_normal, arrow_horizontal, arrow_vertical, strict=True
        )
    )
    _orbit_direction(
        client,
        [arrow["id"]],
        _center(arrow),
        right_direction,
        projection=_projection("orthographic", scale=arrow_extent * 1.5),
        distance=max(arrow_extent * 5, 1_000),
    )
    evidence.append(_render(client, "surface-right.png"))

    gaps = [roles["gap_left"], roles["gap_right"]]
    gap_center = tuple(sum(_center(component)[axis] for component in gaps) / 2 for axis in range(3))
    gap_extent = max(
        high - low
        for axis in range(3)
        for low, high in (
            (
                min(component["world_bounds"]["minimum_mm"][axis] for component in gaps),
                max(component["world_bounds"]["maximum_mm"][axis] for component in gaps),
            ),
        )
    )
    client.call(
        "component.display",
        component_ids=[component["id"] for component in gaps],
        mode="isolated",
    )
    client.call("illumination.set", illumination={"preset": "backlit"})
    _orbit(
        client,
        [component["id"] for component in gaps],
        gap_center,  # type: ignore[arg-type]
        projection=_projection("orthographic", scale=gap_extent * 1.5),
        distance=max(gap_extent * 5, 1_000),
        azimuth=270,
        elevation=15,
    )
    evidence.append(_render(client, "gap-backlit.png"))

    if "render.contact_sheet" in required:
        client.call("session.reset")
        sheet = client.call(
            "render.contact_sheet",
            output_path="contact-sheet.png",
            recipe="focused_3x3",
            focus_component_ids=[target["id"]],
            panel_width=128,
            panel_height=128,
            samples=1,
            engine="eevee",
            visibility_threshold=0.2,
            occluder_budget=3,
        )
        evidence.append(str(sheet["sheet"]["path"]))
    return values, answer, evidence


def main() -> None:
    envelope = json.loads(sys.stdin.readline())
    if not isinstance(envelope, dict) or envelope.get("type") != "episode":
        raise ProtocolError("missing JSONL episode envelope")
    client = ProtocolClient()
    required = set(envelope.get("required_operations", ()))
    manifest = client.call("scene.open", source_path=envelope["model_path"])
    client.call("session.snapshot")

    answer_properties = envelope["answer_schema"]["properties"]["values"]["properties"]
    curated = "target_path" in answer_properties
    if curated:
        values, evidence = _run_curated(client, envelope, manifest, required)
        answer: JsonObject = {"status": "answered", "values": values}
    else:
        _values, answer, evidence = _run_procedural(client, envelope, manifest, required)

    if "session.reset" in required:
        client.call("session.reset")
    submission = {
        "episode_id": envelope["episode_id"],
        "answer": answer,
        "evidence_manifest_paths": evidence,
    }
    client.submit(submission)


if __name__ == "__main__":
    main()
