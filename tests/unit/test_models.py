from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from meshprobe.identity import stable_component_id
from meshprobe.models import (
    AreaLight,
    Bounds,
    ComponentVisualState,
    ContactSheetManifest,
    CustomIllumination,
    DepthOfField,
    EnvironmentMap,
    MarkMode,
    OccluderRemovalStep,
    OcclusionEvidence,
    OrthographicProjection,
    OrthonormalBasis,
    PerspectiveProjection,
    PointLight,
    Pose,
    PresetIllumination,
    RenderManifest,
    SceneManifest,
    SensorFit,
    ShadedEdgesStyle,
    SpotLight,
    SunLight,
    VisibleBackgroundMode,
)


def test_contact_sheet_contract_versions_camera_attribution_schema() -> None:
    schema_version = ContactSheetManifest.model_json_schema()["properties"]["schema_version"]

    assert schema_version["const"] == 5
    assert schema_version["default"] == 5
    with pytest.raises(ValidationError, match="Input should be 5"):
        ContactSheetManifest.model_validate({"schema_version": 4})


def test_occlusion_schema_distinguishes_ray_ranking_from_pixel_visibility() -> None:
    properties = OcclusionEvidence.model_json_schema()["properties"]

    assert "only to rank blocking components" in properties["sample_count"]["description"]
    assert "attributed camera" in properties["visibility_width_px"]["description"]
    assert "attributed camera" in properties["visibility_height_px"]["description"]
    assert "visibility denominator" in properties["isolated_pixel_count"]["description"]
    assert "divided by isolated" in properties["visible_fraction_before"]["description"]


def occlusion_evidence(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "camera": render_session()["camera"],
        "camera_source": "generated_focus_context",
        "visibility_width_px": 256,
        "visibility_height_px": 8,
        "visibility_threshold": 0.9,
        "removal_budget": 1,
        "sample_count": 9,
        "visible_pixel_count_before": 2,
        "visible_pixel_count_after": 8,
        "isolated_pixel_count": 10,
        "visible_fraction_before": 0.2,
        "visible_fraction_after": 0.8,
        "steps": (
            OccluderRemovalStep(
                component_id="blocker",
                display_name="blocker",
                component_path="/blocker",
                ray_hit_count=4,
                visible_pixel_count_after=8,
                visible_fraction_after=0.8,
            ),
        ),
        "stop_reason": "budget_exhausted",
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"visible_fraction_before": 0.3}, "before removal must match"),
        ({"visible_pixel_count_after": 11, "visible_fraction_after": 1.0}, "cannot exceed"),
        ({"visible_pixel_count_after": 1, "visible_fraction_after": 0.1}, "cannot reduce"),
        ({"visible_pixel_count_after": 7, "visible_fraction_after": 0.7}, "last removal step"),
        (
            {
                "steps": (),
                "visible_pixel_count_after": 8,
                "visible_fraction_after": 0.8,
            },
            "without a removal step",
        ),
    ],
)
def test_occlusion_evidence_rejects_inconsistent_visibility(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        OcclusionEvidence.model_validate(occlusion_evidence(**updates))


def render_session(state_sha256: str = "b" * 64) -> dict[str, object]:
    return {
        "camera": {
            "pose": {
                "position_mm": [100, 100, 100],
                "orientation_xyzw": [0, 0, 0, 1],
            },
            "projection": {"mode": "perspective", "focal_length_mm": 50},
        },
        "camera_diagnostics": {
            "aspect_ratio": 1,
            "horizontal_fov_degrees": 40,
            "vertical_fov_degrees": 40,
            "right": [1, 0, 0],
            "up": [0, 1, 0],
            "forward": [0, 0, -1],
            "frustum_corners_mm": [[0, 0, 0]] * 8,
            "target_depth_mm": 100,
            "projected_bounds": {},
        },
        "illumination": {"preset": "neutral_studio"},
        "components": {},
        "state_sha256": state_sha256,
    }


def test_pose_normalizes_quaternion() -> None:
    pose = Pose(position_mm=(1, 2, 3), orientation_xyzw=(0, 0, 0, 4))
    assert pose.orientation_xyzw == (0, 0, 0, 1)


def test_pose_rejects_zero_quaternion() -> None:
    with pytest.raises(ValidationError, match="non-zero magnitude"):
        Pose(position_mm=(1, 2, 3), orientation_xyzw=(0, 0, 0, 0))


@pytest.mark.parametrize("frame", ["camera", "component"])
def test_camera_pose_rejects_unimplemented_frames(frame: str) -> None:
    with pytest.raises(ValidationError, match=r"source.*world"):
        Pose(
            position_mm=(1, 2, 3),
            orientation_xyzw=(0, 0, 0, 1),
            frame=frame,  # type: ignore[arg-type]
        )


def test_bounds_reject_inverted_axis() -> None:
    with pytest.raises(ValidationError, match="minimum_mm"):
        Bounds(minimum_mm=(2, 0, 0), maximum_mm=(1, 1, 1))


def test_component_mark_color_is_canonical_and_requires_a_mark() -> None:
    marked = ComponentVisualState(mark=MarkMode.HIGHLIGHTED, mark_color="#FF00aA")

    assert marked.mark_color == "#ff00aa"
    with pytest.raises(ValidationError, match="requires a visible mark"):
        ComponentVisualState(mark_color="#ff00aa")


def test_perspective_reports_field_of_view() -> None:
    projection = PerspectiveProjection(focal_length_mm=50, sensor_width_mm=36)
    assert projection.horizontal_fov_degrees(16 / 9) == pytest.approx(39.5978, rel=1e-4)
    assert projection.vertical_fov_degrees(16 / 9) == pytest.approx(22.8952, rel=1e-4)


def test_perspective_depth_of_field_requires_one_focus_source() -> None:
    projection = PerspectiveProjection.model_validate(
        {
            "focal_length_mm": 20,
            "depth_of_field": {
                "mode": "enabled",
                "aperture_fstop": 3.8,
                "focus": {"component_id": "cmp_focus"},
            },
        }
    )
    assert projection.depth_of_field.aperture_fstop == 3.8
    assert projection.depth_of_field.focus is not None
    assert projection.depth_of_field.focus.component_id == "cmp_focus"

    exact = DepthOfField(mode="enabled", aperture_fstop=3.8, focus_distance_mm=750)
    assert exact.focus_distance_mm == 750
    with pytest.raises(ValidationError, match="exactly one focus"):
        DepthOfField(mode="enabled", aperture_fstop=3.8)
    with pytest.raises(ValidationError, match="cannot declare"):
        DepthOfField(mode="disabled", focus_distance_mm=100)


def test_camera_basis_must_be_orthonormal_and_right_handed() -> None:
    basis = OrthonormalBasis(x=(0, 1, 0), y=(-1, 0, 0), z=(0, 0, 1))
    assert basis.x == (0, 1, 0)

    with pytest.raises(ValidationError, match="unit length"):
        OrthonormalBasis(x=(2, 0, 0))
    with pytest.raises(ValidationError, match="perpendicular"):
        OrthonormalBasis(x=(1, 0, 0), y=(1, 0, 0))
    with pytest.raises(ValidationError, match="right-handed"):
        OrthonormalBasis(z=(0, 0, -1))


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


def test_light_and_environment_intensities_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        AreaLight(
            id="area",
            position_mm=(1, 2, 3),
            orientation_xyzw=(0, 0, 0, 1),
            power_w=0,
            size_mm=50,
            color_temperature_k=5200,
        )
    with pytest.raises(ValidationError, match="greater than 0"):
        PointLight(
            id="point",
            position_mm=(1, 2, 3),
            power_w=0,
            color_temperature_k=5200,
        )
    with pytest.raises(ValidationError, match="greater than 0"):
        SpotLight(
            id="spot",
            position_mm=(1, 2, 3),
            orientation_xyzw=(0, 0, 0, 1),
            power_w=0,
            spot_size_degrees=45,
            color_temperature_k=5200,
        )
    with pytest.raises(ValidationError, match="greater than 0"):
        SunLight(
            id="sun",
            orientation_xyzw=(0, 0, 0, 1),
            strength=0,
            color_temperature_k=5200,
        )
    with pytest.raises(ValidationError, match="greater than 0"):
        EnvironmentMap(path="studio.exr", sha256="a" * 64, strength=0)


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


def test_custom_illumination_rejects_camera_only_background() -> None:
    with pytest.raises(ValidationError, match="non-zero light output"):
        CustomIllumination(
            background_rgb=(0.1, 0, 0),
            background_strength=0.5,
            ambient_rgb=(0, 0, 0),
            ambient_strength=0,
        )


def test_custom_illumination_derives_omitted_split_fields() -> None:
    illumination = CustomIllumination(
        background_rgb=(0.03, 0.03, 0.04),
        ambient_strength=0.18,
    )

    assert illumination.background_strength == 0.18
    assert illumination.ambient_rgb == (0.03, 0.03, 0.04)
    assert illumination.visible_background_mode is VisibleBackgroundMode.COLOR


def test_custom_illumination_records_background_and_ambient_separately() -> None:
    illumination = CustomIllumination(
        background_rgb=(1, 1, 1),
        background_strength=1,
        ambient_rgb=(0.1, 0.2, 0.3),
        ambient_strength=0.15,
    )

    assert illumination.model_dump(mode="json") == {
        "preset": "custom",
        "background_rgb": [1.0, 1.0, 1.0],
        "background_strength": 1.0,
        "ambient_rgb": [0.1, 0.2, 0.3],
        "ambient_strength": 0.15,
        "background_srgb": None,
        "lights": [],
        "environment_map": None,
        "visible_background_mode": "color",
    }


def test_preset_illumination_accepts_display_referred_background() -> None:
    illumination = PresetIllumination(preset="neutral_studio", background_srgb=(1, 1, 1))

    assert illumination.background_srgb == (1, 1, 1)
    assert illumination.background_rgb is None
    assert illumination.background_strength is None


def test_preset_illumination_rejects_out_of_range_display_background() -> None:
    with pytest.raises(ValidationError):
        PresetIllumination(preset="neutral_studio", background_srgb=(1.5, 0, 0))


def test_custom_illumination_accepts_content_addressed_environment_only() -> None:
    illumination = CustomIllumination(
        background_rgb=(0, 0, 0),
        ambient_strength=0,
        environment_map=EnvironmentMap(
            path="studio.exr",
            sha256="a" * 64,
            strength=1.5,
            rotation_degrees=45,
        ),
    )

    assert illumination.lights == ()
    assert illumination.environment_map is not None
    assert illumination.environment_map.projection == "equirectangular"
    assert illumination.visible_background_mode is VisibleBackgroundMode.ENVIRONMENT


def test_custom_illumination_accepts_explicit_color_background_with_environment() -> None:
    illumination = CustomIllumination(
        background_rgb=(0.1, 0.2, 0.3),
        background_strength=0.5,
        ambient_strength=0,
        environment_map=EnvironmentMap(path="studio.exr", sha256="a" * 64),
        visible_background_mode=VisibleBackgroundMode.COLOR,
    )

    assert illumination.visible_background_mode is VisibleBackgroundMode.COLOR


def test_custom_illumination_rejects_environment_background_without_map() -> None:
    with pytest.raises(ValidationError, match="requires an environment map"):
        CustomIllumination(
            background_rgb=(0.1, 0.2, 0.3),
            ambient_strength=0.5,
            visible_background_mode=VisibleBackgroundMode.ENVIRONMENT,
        )


def test_custom_illumination_rejects_no_lights_background_or_environment() -> None:
    with pytest.raises(ValidationError, match="non-zero light output"):
        CustomIllumination(
            background_rgb=(0, 0, 0),
            ambient_strength=0,
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


def test_scene_reports_source_and_world_coordinate_frames(
    scene_manifest: SceneManifest,
) -> None:
    frames = scene_manifest.coordinate_frames

    assert scene_manifest.schema_version == 2
    assert frames.source.name == "gltf_y_up"
    assert frames.source.up_axis == "+y"
    assert frames.source.handedness == "right"
    assert frames.world.name == "meshprobe_z_up"
    assert frames.source_to_world == (
        1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
    )
    assert scene_manifest.root_bounds.frame == "world"
    assert scene_manifest.components[0].world_transform_frame == "world"
    assert scene_manifest.imported_camera.pose.frame == "world"


def test_scene_manifest_v2_requires_coordinate_frames(scene_manifest: SceneManifest) -> None:
    payload = scene_manifest.model_dump(mode="json")
    payload["schema_version"] = 1
    with pytest.raises(ValidationError, match="Input should be 2"):
        SceneManifest.model_validate(payload)

    payload["schema_version"] = 2
    del payload["coordinate_frames"]
    with pytest.raises(ValidationError, match="Field required"):
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
                "device": "graphics_hardware",
                "graphics_policy": "software_allowed",
                "graphics": {
                    "vendor": "Test",
                    "renderer": "Test GPU",
                    "version": "1.0",
                    "backend": "OPENGL",
                    "blender_device_type": "NVIDIA",
                    "device_class": "hardware",
                },
                "blender_version": "5.2.0",
                "session": render_session(),
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


def test_shaded_edges_style_rejects_duplicate_or_empty_edge_types() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        ShadedEdgesStyle(edge_types=())
    with pytest.raises(ValidationError, match="must be unique"):
        ShadedEdgesStyle(edge_types=("crease", "crease"))


def test_evidence_manifest_schema_tracks_camera_operation_contract() -> None:
    render_schema = RenderManifest.model_json_schema()["properties"]["schema_version"]
    sheet_schema = ContactSheetManifest.model_json_schema()["properties"]["schema_version"]

    assert render_schema["const"] == 2
    assert sheet_schema["const"] == 5


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
        "device": "graphics_hardware",
        "graphics_policy": "software_allowed",
        "graphics": {
            "vendor": "Test",
            "renderer": "Test GPU",
            "version": "1.0",
            "backend": "OPENGL",
            "blender_device_type": "NVIDIA",
            "device_class": "hardware",
        },
        "blender_version": "5.2.0",
        "session": render_session(),
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
