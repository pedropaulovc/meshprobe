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
