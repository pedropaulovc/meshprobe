from __future__ import annotations

import math

from hypothesis import assume, given
from hypothesis import strategies as st

from meshprobe.models import PerspectiveProjection, Pose

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=0.1, max_value=10_000, allow_nan=False, allow_infinity=False)


@given(st.tuples(finite, finite, finite, finite))
def test_pose_quaternion_is_always_normalized(quaternion: tuple[float, ...]) -> None:
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
    assert 0 < projection.horizontal_fov_degrees < 180
