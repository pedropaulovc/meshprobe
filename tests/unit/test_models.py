from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from meshprobe.models import (
    AreaLight,
    Bounds,
    CustomIllumination,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
    SceneManifest,
)


def test_pose_normalizes_quaternion() -> None:
    pose = Pose(position_mm=(1, 2, 3), orientation_xyzw=(0, 0, 0, 4))
    assert pose.orientation_xyzw == (0, 0, 0, 1)


def test_pose_rejects_zero_quaternion() -> None:
    with pytest.raises(ValidationError, match="non-zero magnitude"):
        Pose(position_mm=(1, 2, 3), orientation_xyzw=(0, 0, 0, 0))


def test_bounds_reject_inverted_axis() -> None:
    with pytest.raises(ValidationError, match="minimum_mm"):
        Bounds(minimum_mm=(2, 0, 0), maximum_mm=(1, 1, 1))


def test_perspective_reports_field_of_view() -> None:
    projection = PerspectiveProjection(focal_length_mm=50, sensor_width_mm=36)
    assert projection.horizontal_fov_degrees == pytest.approx(39.5978, rel=1e-4)
    assert projection.vertical_fov_degrees(16 / 9) == pytest.approx(22.8952, rel=1e-4)


@pytest.mark.parametrize("projection", [PerspectiveProjection, OrthographicProjection])
def test_projection_rejects_inverted_clip_range(projection) -> None:  # type: ignore[no-untyped-def]
    kwargs = {"scale_mm": 100} if projection is OrthographicProjection else {}
    with pytest.raises(ValidationError, match="far_clip_mm"):
        projection(near_clip_mm=100, far_clip_mm=10, **kwargs)


def test_custom_illumination_requires_one_color_source() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        AreaLight(
            id="key",
            position_mm=(1, 2, 3),
            orientation_xyzw=(0, 0, 0, 1),
            power_w=100,
            size_mm=50,
        )


def test_custom_illumination_rejects_duplicate_ids() -> None:
    light = AreaLight(
        id="key",
        position_mm=(1, 2, 3),
        orientation_xyzw=(0, 0, 0, 1),
        power_w=100,
        size_mm=50,
        color_temperature_k=5200,
    )
    with pytest.raises(ValidationError, match="unique"):
        CustomIllumination(
            background_rgb=(0, 0, 0),
            ambient_strength=0,
            lights=(light, light),
        )


def test_normalized_quaternion_has_unit_length() -> None:
    pose = Pose(position_mm=(0, 0, 0), orientation_xyzw=(1, -2, 3, -4))
    assert math.sqrt(sum(value**2 for value in pose.orientation_xyzw)) == pytest.approx(1)


def test_normalized_quaternion_handles_largest_finite_values() -> None:
    pose = Pose(position_mm=(0, 0, 0), orientation_xyzw=(1e308, -1e308, 1e308, -1e308))
    assert pose.orientation_xyzw == pytest.approx((0.5, -0.5, 0.5, -0.5))


def test_light_quaternion_is_normalized_and_zero_is_rejected() -> None:
    light = AreaLight(
        id="key",
        position_mm=(1, 2, 3),
        orientation_xyzw=(0, 0, 0, 4),
        power_w=100,
        size_mm=50,
        color_temperature_k=5200,
    )
    assert light.orientation_xyzw == (0, 0, 0, 1)
    with pytest.raises(ValidationError, match="non-zero magnitude"):
        AreaLight(
            id="key",
            position_mm=(1, 2, 3),
            orientation_xyzw=(0, 0, 0, 0),
            power_w=100,
            size_mm=50,
            color_temperature_k=5200,
        )


def test_scene_rejects_unmirrored_parent_link(scene_manifest: SceneManifest) -> None:
    payload = scene_manifest.model_dump(mode="json")
    child_id = payload["components"][1]["id"]
    payload["components"][0]["child_ids"].remove(child_id)
    with pytest.raises(ValidationError, match="does not list child"):
        SceneManifest.model_validate(payload)


def test_scene_rejects_component_cycle(scene_manifest: SceneManifest) -> None:
    payload = scene_manifest.model_dump(mode="json")
    root_id = payload["components"][0]["id"]
    idler_id = payload["components"][2]["id"]
    payload["components"][0]["parent_id"] = idler_id
    payload["components"][2]["child_ids"].append(root_id)
    with pytest.raises(ValidationError, match="contains a cycle"):
        SceneManifest.model_validate(payload)


def test_scene_rejects_component_id_not_derived_from_source_and_path(
    scene_manifest: SceneManifest,
) -> None:
    payload = scene_manifest.model_dump(mode="json")
    old_id = payload["components"][0]["id"]
    replacement = "cmp_000000000000000000000000"
    payload["components"][0]["id"] = replacement
    for component in payload["components"]:
        if component["parent_id"] == old_id:
            component["parent_id"] = replacement
    with pytest.raises(ValidationError, match="stable derivation"):
        SceneManifest.model_validate(payload)


def test_scene_rejects_duplicate_child_ids(scene_manifest: SceneManifest) -> None:
    payload = scene_manifest.model_dump(mode="json")
    payload["components"][0]["child_ids"].append(payload["components"][0]["child_ids"][0])
    with pytest.raises(ValidationError, match="child ids must be unique"):
        SceneManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("units", "meter", "millimeter"),
        ("source_format", "fbx", "Input should be"),
    ],
)
def test_scene_rejects_noncanonical_source_metadata(
    scene_manifest: SceneManifest, field: str, value: str, message: str
) -> None:
    payload = scene_manifest.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        SceneManifest.model_validate(payload)


def test_scene_rejects_warning_for_unknown_component(scene_manifest: SceneManifest) -> None:
    payload = scene_manifest.model_dump(mode="json")
    payload["warnings"].append(
        {"code": "import.partial", "message": "missing detail", "component_ids": ["missing"]}
    )
    with pytest.raises(ValidationError, match="warnings reference unknown"):
        SceneManifest.model_validate(payload)
