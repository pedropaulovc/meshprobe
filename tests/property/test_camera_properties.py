from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from meshprobe.camera import (
    _normalize,
    _quaternion_from_matrix,
    camera_diagnostics,
    orbit_angles_from_orientation,
    orbit_camera,
)
from meshprobe.models import (
    Camera,
    CameraPoseFrame,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
)

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=0.1, max_value=10_000, allow_nan=False, allow_infinity=False)


@given(st.tuples(finite, finite, finite, finite))
def test_pose_quaternion_is_always_normalized(
    quaternion: tuple[float, float, float, float],
) -> None:
    assume(math.sqrt(sum(value * value for value in quaternion)) > 1e-9)
    pose = Pose(position_mm=(0, 0, 0), orientation_xyzw=quaternion)
    norm = math.sqrt(sum(value * value for value in pose.orientation_xyzw))
    assert math.isclose(norm, 1.0, abs_tol=1e-12)


@given(focal_length=positive, sensor_width=positive)
def test_perspective_fov_is_bounded(focal_length: float, sensor_width: float) -> None:
    projection = PerspectiveProjection(
        focal_length_mm=focal_length,
        sensor_width_mm=sensor_width,
    )
    assert 0 < projection.horizontal_fov_degrees(16 / 9) < 180


def test_orbit_camera_handles_vertical_view_and_roll() -> None:
    camera = orbit_camera(
        target_mm=(0.0, 0.0, 0.0),
        azimuth_degrees=45.0,
        elevation_degrees=90.0,
        roll_degrees=37.0,
        distance_mm=500.0,
        projection=PerspectiveProjection(focal_length_mm=50.0),
    )

    norm = math.sqrt(sum(value * value for value in camera.pose.orientation_xyzw))
    assert math.isclose(norm, 1.0, abs_tol=1e-12)


@given(
    azimuth=st.floats(min_value=-179.0, max_value=179.0),
    elevation=st.floats(min_value=-89.0, max_value=89.0),
)
def test_orbit_angles_round_trip_through_orientation(azimuth: float, elevation: float) -> None:
    camera = orbit_camera(
        target_mm=(12.0, -3.0, 40.0),
        azimuth_degrees=azimuth,
        elevation_degrees=elevation,
        roll_degrees=0.0,
        distance_mm=500.0,
        projection=PerspectiveProjection(focal_length_mm=50.0),
    )

    recovered_azimuth, recovered_elevation = orbit_angles_from_orientation(
        camera.pose.orientation_xyzw
    )

    assert recovered_azimuth == pytest.approx(azimuth, abs=1e-6)
    assert recovered_elevation == pytest.approx(elevation, abs=1e-6)


def test_orbit_angles_are_target_and_distance_independent() -> None:
    projection = PerspectiveProjection(focal_length_mm=50.0)
    near = orbit_camera(
        target_mm=(0.0, 0.0, 0.0),
        azimuth_degrees=35.0,
        elevation_degrees=20.0,
        roll_degrees=0.0,
        distance_mm=100.0,
        projection=projection,
    )
    far = orbit_camera(
        target_mm=(500.0, -200.0, 80.0),
        azimuth_degrees=35.0,
        elevation_degrees=20.0,
        roll_degrees=0.0,
        distance_mm=9_000.0,
        projection=projection,
    )

    assert orbit_angles_from_orientation(near.pose.orientation_xyzw) == pytest.approx(
        orbit_angles_from_orientation(far.pose.orientation_xyzw)
    )


def test_camera_math_covers_x_dominant_rotation_and_zero_vector() -> None:
    quaternion = _quaternion_from_matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    assert quaternion == (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="degenerate"):
        _normalize((0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "projection",
    [
        PerspectiveProjection(focal_length_mm=85, near_clip_mm=1, far_clip_mm=1_000),
        OrthographicProjection(scale_mm=200, near_clip_mm=1, far_clip_mm=1_000),
    ],
)
def test_camera_diagnostics_return_basis_frustum_and_target_depth(
    projection: PerspectiveProjection | OrthographicProjection,
) -> None:
    camera = orbit_camera(
        target_mm=(0, 0, 0),
        azimuth_degrees=0,
        elevation_degrees=0,
        roll_degrees=0,
        distance_mm=500,
        projection=projection,
    )

    diagnostics = camera_diagnostics(camera, target_mm=(0, 0, 0), aspect_ratio=16 / 9)

    assert diagnostics.right == pytest.approx((0, 1, 0))
    assert diagnostics.up == pytest.approx((0, 0, 1))
    assert diagnostics.forward == pytest.approx((-1, 0, 0))
    assert diagnostics.target_depth_mm == pytest.approx(500)
    assert len(diagnostics.frustum_corners_mm) == 8
    if isinstance(projection, PerspectiveProjection):
        assert diagnostics.horizontal_fov_degrees == pytest.approx(
            projection.horizontal_fov_degrees(16 / 9)
        )
        assert diagnostics.vertical_fov_degrees == pytest.approx(
            projection.vertical_fov_degrees(16 / 9)
        )
    else:
        assert diagnostics.horizontal_fov_degrees is None
        assert diagnostics.vertical_fov_degrees is None
    near_depths = [500 - corner[0] for corner in diagnostics.frustum_corners_mm[:4]]
    far_depths = [500 - corner[0] for corner in diagnostics.frustum_corners_mm[4:]]
    assert near_depths == pytest.approx([1] * 4)
    assert far_depths == pytest.approx([1_000] * 4)


def test_camera_diagnostics_reject_source_frame_without_a_scene_transform() -> None:
    camera = Camera(
        pose=Pose(
            position_mm=(1, 2, 3),
            orientation_xyzw=(0, 0, 0, 1),
            frame=CameraPoseFrame.SOURCE,
        ),
        projection=PerspectiveProjection(),
    )

    with pytest.raises(ValueError, match="world-frame pose"):
        camera_diagnostics(camera, target_mm=(0, 0, 0))
