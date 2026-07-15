from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from meshprobe.identity import stable_component_id
from meshprobe.models import (
    AreaLight,
    Bounds,
    CustomIllumination,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
    RenderManifest,
    SceneManifest,
    SensorFit,
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
    assert projection.horizontal_fov_degrees(16 / 9) == pytest.approx(39.5978, rel=1e-4)
    assert projection.vertical_fov_degrees(16 / 9) == pytest.approx(22.8952, rel=1e-4)


def test_perspective_respects_vertical_and_automatic_sensor_fit() -> None:
    vertical = PerspectiveProjection(
        focal_length_mm=50,
        sensor_width_mm=36,
        sensor_height_mm=24,
        sensor_fit=SensorFit.VERTICAL,
    )
    automatic = vertical.model_copy(update={"sensor_fit": SensorFit.AUTO})

    assert vertical.horizontal_fov_degrees(16 / 9) == pytest.approx(46.2127, rel=1e-4)
    assert vertical.vertical_fov_degrees(16 / 9) == pytest.approx(26.9915, rel=1e-4)
    assert automatic.effective_sensor_fit(16 / 9) is SensorFit.HORIZONTAL
    assert automatic.effective_sensor_fit(9 / 16) is SensorFit.VERTICAL


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


def test_custom_illumination_rejects_zero_effective_output() -> None:
    black_light = AreaLight(
        id="black",
        position_mm=(1, 2, 3),
        orientation_xyzw=(0, 0, 0, 1),
        power_w=100,
        size_mm=50,
        linear_rgb=(0, 0, 0),
    )
    with pytest.raises(ValidationError, match="non-zero light output"):
        CustomIllumination(
            background_rgb=(0, 0, 0),
            ambient_strength=0,
            lights=(black_light,),
        )


def test_custom_illumination_accepts_effective_background() -> None:
    black_light = AreaLight(
        id="black",
        position_mm=(1, 2, 3),
        orientation_xyzw=(0, 0, 0, 1),
        power_w=100,
        size_mm=50,
        linear_rgb=(0, 0, 0),
    )
    illumination = CustomIllumination(
        background_rgb=(0.1, 0, 0),
        ambient_strength=0.5,
        lights=(black_light,),
    )
    assert illumination.ambient_strength == 0.5


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


def test_scene_rejects_child_path_outside_parent(scene_manifest: SceneManifest) -> None:
    payload = scene_manifest.model_dump(mode="json")
    child = payload["components"][1]
    old_id = child["id"]
    child["path"] = "elsewhere/clip"
    child["id"] = stable_component_id(scene_manifest.source_sha256, child["path"])
    payload["components"][0]["child_ids"] = [
        child["id"] if component_id == old_id else component_id
        for component_id in payload["components"][0]["child_ids"]
    ]
    with pytest.raises(ValidationError, match="path must be under parent path"):
        SceneManifest.model_validate(payload)


def test_scene_rejects_root_bounds_that_exclude_component(
    scene_manifest: SceneManifest,
) -> None:
    payload = scene_manifest.model_dump(mode="json")
    payload["root_bounds"] = {"minimum_mm": [-5, -5, -5], "maximum_mm": [5, 5, 5]}
    with pytest.raises(ValidationError, match="root_bounds must enclose"):
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


def test_render_manifest_rejects_unordered_luminance() -> None:
    with pytest.raises(ValidationError, match="luminance values must be ordered"):
        RenderManifest.model_validate(
            {
                "source_sha256": "a" * 64,
                "state_sha256": "b" * 64,
                "width": 512,
                "height": 512,
                "samples": 16,
                "engine": "eevee",
                "color": {
                    "path": "render.png",
                    "media_type": "image/png",
                    "sha256": "c" * 64,
                    "bytes": 1,
                },
                "luminance": {
                    "minimum": 0.1,
                    "median": 0.9,
                    "maximum": 0.5,
                    "crushed_fraction": 0,
                    "clipped_fraction": 0,
                },
            }
        )


def test_evaluator_component_colors_must_be_unique() -> None:
    artifact = {
        "path": "pass.png",
        "media_type": "image/png",
        "sha256": "c" * 64,
        "bytes": 1,
    }
    payload = {
        "source_sha256": "a" * 64,
        "state_sha256": "b" * 64,
        "width": 512,
        "height": 512,
        "samples": 16,
        "engine": "eevee",
        "color": artifact,
        "evaluator": {
            "multilayer": {**artifact, "media_type": "image/x-exr"},
            "component_ids": artifact,
            "highlighted": artifact,
            "component_colors": {"one": [1, 2, 3], "two": [1, 2, 3]},
        },
        "luminance": {
            "minimum": 0.1,
            "median": 0.2,
            "maximum": 0.5,
            "crushed_fraction": 0,
            "clipped_fraction": 0,
        },
    }
    with pytest.raises(ValidationError, match="colors must be unique"):
        RenderManifest.model_validate(payload)
