from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from meshprobe.camera import (
    _axis_angle,
    _multiply_quaternions,
    _normalize,
    _quaternion_from_matrix,
    _rotate_vector,
    camera_diagnostics,
    camera_local_axis_world,
    camera_local_point_world,
    orbit_angles_from_orientation,
    orbit_camera,
)
from meshprobe.models import (
    Camera,
    CameraPoseFrame,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
    Quaternion,
    Vec3,
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


def _local_change(orientation: Quaternion, world_axis: Vec3, angle: float) -> Quaternion:
    world_rotation = _axis_angle(world_axis, angle)
    rotated = _multiply_quaternions(world_rotation, orientation)
    conjugate = (-orientation[0], -orientation[1], -orientation[2], orientation[3])
    return _multiply_quaternions(conjugate, rotated)


def test_camera_frame_rotation_is_direction_independent() -> None:
    projection = PerspectiveProjection(focal_length_mm=50.0)
    facing_front = orbit_camera(
        target_mm=(0.0, 0.0, 0.0),
        azimuth_degrees=0.0,
        elevation_degrees=10.0,
        roll_degrees=0.0,
        distance_mm=500.0,
        projection=projection,
    ).pose.orientation_xyzw
    facing_back = orbit_camera(
        target_mm=(0.0, 0.0, 0.0),
        azimuth_degrees=180.0,
        elevation_degrees=10.0,
        roll_degrees=0.0,
        distance_mm=500.0,
        projection=projection,
    ).pose.orientation_xyzw

    local_right = (1.0, 0.0, 0.0)
    angle = math.radians(15.0)
    probes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    # A camera-frame tilt induces the SAME change in the camera's own basis no matter
    # which way it faces, and that change is exactly the local axis-angle rotation.
    expected = _axis_angle(local_right, angle)
    for orientation in (facing_front, facing_back):
        axis_world = camera_local_axis_world(orientation, local_right)
        change = _local_change(orientation, axis_world, angle)
        for probe in probes:
            assert _rotate_vector(change, probe) == pytest.approx(
                _rotate_vector(expected, probe), abs=1e-9
            )

    # Contrast: a world-frame rotation about the same nominal axis is NOT invariant —
    # front and back facings tilt oppositely. This is the exact pitfall #65 removes.
    world_axis = (1.0, 0.0, 0.0)
    front_change = _local_change(facing_front, world_axis, angle)
    back_change = _local_change(facing_back, world_axis, angle)
    assert any(
        _rotate_vector(front_change, probe)
        != pytest.approx(_rotate_vector(back_change, probe), abs=1e-6)
        for probe in probes
    )


def test_camera_frame_pivot_resolves_through_the_pose() -> None:
    camera = orbit_camera(
        target_mm=(10.0, -5.0, 30.0),
        azimuth_degrees=48.0,
        elevation_degrees=15.0,
        roll_degrees=0.0,
        distance_mm=400.0,
        projection=PerspectiveProjection(focal_length_mm=50.0),
    )
    position = camera.pose.position_mm
    orientation = camera.pose.orientation_xyzw

    # A zero camera-local pivot is the camera itself — the zero-radius tilt case.
    assert camera_local_point_world(orientation, position, (0.0, 0.0, 0.0)) == pytest.approx(
        position
    )

    # A non-zero camera-local pivot is offset by the orientation-rotated point, never the
    # raw world coordinates (the bug: pivoting at world origin unless target is (0,0,0)).
    local_point = (0.0, 0.0, -400.0)
    resolved = camera_local_point_world(orientation, position, local_point)
    expected = tuple(
        origin + offset
        for origin, offset in zip(position, _rotate_vector(orientation, local_point), strict=True)
    )
    assert resolved == pytest.approx(expected)
    assert resolved != pytest.approx(local_point)


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


@pytest.mark.parametrize(
    ("aspect_ratio", "expected_half_width", "expected_half_height"),
    [
        (16 / 9, 100.0, 56.25),
        (1 / 4, 25.0, 100.0),
    ],
)
def test_orthographic_camera_diagnostics_match_blender_auto_sensor_fit(
    aspect_ratio: float,
    expected_half_width: float,
    expected_half_height: float,
) -> None:
    camera = orbit_camera(
        target_mm=(0, 0, 0),
        azimuth_degrees=0,
        elevation_degrees=0,
        roll_degrees=0,
        distance_mm=500,
        projection=OrthographicProjection(scale_mm=200, near_clip_mm=1, far_clip_mm=1_000),
    )

    diagnostics = camera_diagnostics(camera, target_mm=(0, 0, 0), aspect_ratio=aspect_ratio)
    near_corners = diagnostics.frustum_corners_mm[:4]

    for corner in near_corners:
        assert abs(corner[1]) == pytest.approx(expected_half_width)
        assert abs(corner[2]) == pytest.approx(expected_half_height)


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
