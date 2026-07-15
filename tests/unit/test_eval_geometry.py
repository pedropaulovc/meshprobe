from __future__ import annotations

import json
import struct

import pytest

from meshprobe.evals.geometry import (
    CameraSpec,
    ComponentSpec,
    GlbScene,
    MaterialSpec,
    box,
    cylinder,
    look_at_quaternion,
)


def glb_json(payload: bytes) -> dict[str, object]:
    chunk_length, chunk_type = struct.unpack_from("<II", payload, 12)
    assert chunk_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + chunk_length])


def sample_scene() -> GlbScene:
    steel = MaterialSpec("steel", (0.4, 0.45, 0.5, 1), metallic=0.7)
    scene = GlbScene(camera=CameraSpec("inspection", (6, -6, 5), (0, 0, 0), focal_length_mm=85))
    scene.add(ComponentSpec("base", "assembly", box((4, 3, 0.4)), steel))
    scene.add(
        ComponentSpec(
            "shaft",
            "shaft",
            cylinder(0.3, 2),
            steel,
            parent_role="base",
            translation_m=(1, 0, 1),
        )
    )
    return scene


def test_glb_writer_is_byte_deterministic_and_embeds_geometry() -> None:
    first = sample_scene().to_glb()
    second = sample_scene().to_glb()

    assert first == second
    assert first[:4] == b"glTF"
    document = glb_json(first)
    assert len(document["meshes"]) == 2  # type: ignore[arg-type]
    assert document["buffers"][0]["byteLength"] > 0  # type: ignore[index]


def test_component_paths_follow_source_hierarchy() -> None:
    assert sample_scene().component_paths() == {
        "base": "assembly",
        "shaft": "assembly/shaft",
    }


def test_role_order_must_keep_parents_before_children() -> None:
    with pytest.raises(ValueError, match="parents before children"):
        sample_scene().to_glb(role_order=("shaft", "base"))


def test_look_at_quaternion_is_normalized() -> None:
    quaternion = look_at_quaternion((6, -6, 5), (0, 0, 0))
    assert sum(value * value for value in quaternion) == pytest.approx(1.0)
