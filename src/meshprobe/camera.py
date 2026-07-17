"""Renderer-independent camera geometry."""

from __future__ import annotations

import math
from typing import cast

from meshprobe.models import (
    Camera,
    CameraDiagnostics,
    CameraPoseFrame,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
    Projection,
    Quaternion,
    Vec3,
)

_EPSILON = 1e-12

# Fraction of the frame's limiting axis the model should occupy when the default
# camera auto-frames the scene on open. 0.9 leaves a ~5% margin on every side —
# comfortably clear of the edge, yet far tighter than an untouched source camera.
DEFAULT_FRAME_FILL = 0.9


def bounds_fit_distance_mm(
    *,
    minimum_mm: Vec3,
    maximum_mm: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    horizontal_half_tan: float,
    vertical_half_tan: float,
    frame_fill: float = DEFAULT_FRAME_FILL,
) -> float:
    """Smallest pinhole-camera distance that frames an axis-aligned box tightly.

    Returns the distance, in millimetres, to place the camera along ``-forward``
    from the box centre so every one of the box's eight corners stays within
    ``frame_fill`` of the frame's limiting axis. ``forward``/``right``/``up`` are
    the camera basis (unit) vectors; the half-tangents are ``tan(fov/2)`` for the
    horizontal and vertical fields of view. A larger ``frame_fill`` frames the box
    more tightly; the limiting axis reaches exactly ``frame_fill`` and the other
    axis stays inside it, so the box never clips.
    """

    if not 0.0 < frame_fill <= 1.0:
        raise ValueError("frame_fill must be within (0, 1]")
    if horizontal_half_tan <= 0.0 or vertical_half_tan <= 0.0:
        raise ValueError("half-angle tangents must be positive")
    center = cast(
        Vec3, tuple((low + high) / 2 for low, high in zip(minimum_mm, maximum_mm, strict=True))
    )
    distance = 0.0
    for corner in _box_corners(minimum_mm, maximum_mm):
        offset = cast(
            Vec3, tuple(value - origin for value, origin in zip(corner, center, strict=True))
        )
        depth = _dot(offset, forward)
        horizontal = abs(_dot(offset, right)) / (horizontal_half_tan * frame_fill)
        vertical = abs(_dot(offset, up)) / (vertical_half_tan * frame_fill)
        distance = max(distance, horizontal - depth, vertical - depth)
    return distance


def _box_corners(minimum_mm: Vec3, maximum_mm: Vec3) -> list[Vec3]:
    return [
        (x, y, z)
        for x in (minimum_mm[0], maximum_mm[0])
        for y in (minimum_mm[1], maximum_mm[1])
        for z in (minimum_mm[2], maximum_mm[2])
    ]


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def orbit_camera(
    *,
    target_mm: Vec3,
    azimuth_degrees: float,
    elevation_degrees: float,
    roll_degrees: float,
    distance_mm: float,
    projection: Projection,
) -> Camera:
    """Convert spherical orbit coordinates to an absolute Blender-style pose."""

    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    offset = (
        distance_mm * math.cos(elevation) * math.cos(azimuth),
        distance_mm * math.cos(elevation) * math.sin(azimuth),
        distance_mm * math.sin(elevation),
    )
    position = cast(
        Vec3,
        tuple(target + delta for target, delta in zip(target_mm, offset, strict=True)),
    )
    forward = _normalize(
        cast(
            Vec3,
            tuple(target - origin for target, origin in zip(target_mm, position, strict=True)),
        )
    )
    orientation = _look_orientation(forward)
    if roll_degrees:
        roll = _axis_angle(forward, math.radians(roll_degrees))
        orientation = _multiply_quaternions(roll, orientation)
    return Camera(
        pose=Pose(position_mm=position, orientation_xyzw=orientation),
        projection=projection,
    )


def orbit_angles_from_orientation(orientation_xyzw: Quaternion) -> tuple[float, float]:
    """Recover the (azimuth, elevation) an orbit would need to reproduce this view.

    The inverse of ``orbit_camera``'s spherical placement: a camera looking along
    ``forward`` sits on the orbit sphere at ``-forward``, so the returned angles fed
    back to ``view-orbit`` (with the same target and distance) reproduce the current
    viewing direction. Both are target-independent, so a caller recovers orbit
    continuity from the stored orientation alone without decomposing the quaternion by
    hand. Elevation is degrees above the world XY plane; azimuth is degrees from +X
    toward +Y about +Z, matching ``view-orbit``. Looking straight up or down leaves
    azimuth undetermined, and it degrades to 0.
    """

    forward = _rotate_vector(orientation_xyzw, (0.0, 0.0, -1.0))
    offset_x, offset_y, offset_z = (-forward[0], -forward[1], -forward[2])
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, offset_z))))
    azimuth = math.degrees(math.atan2(offset_y, offset_x))
    return azimuth, elevation


def camera_local_axis_world(orientation_xyzw: Quaternion, axis: Vec3) -> Vec3:
    """World direction of a camera-local axis for a camera at the given orientation.

    Rotating around this axis (the ``--frame camera`` path) is direction-independent:
    because the axis is carried by the camera's own orientation, the rotation it induces
    in the camera's local basis is identical no matter which way the camera currently
    faces, so a calibrated sign never flips when the camera crosses to the far side of
    the model. Mirrors the renderer's ``rotate_camera`` camera-frame branch.
    """

    return _normalize(_rotate_vector(orientation_xyzw, axis))


def camera_local_point_world(orientation_xyzw: Quaternion, position_mm: Vec3, point: Vec3) -> Vec3:
    """World position of a camera-local point for a camera at the given pose.

    The ``--frame camera`` rotation pivot is expressed in camera-local coordinates, so a
    zero point pivots at the camera itself. Mirrors the renderer's ``rotate_camera``
    camera-frame target transform.
    """

    rotated = _rotate_vector(orientation_xyzw, point)
    return cast(
        Vec3,
        tuple(origin + offset for origin, offset in zip(position_mm, rotated, strict=True)),
    )


def camera_diagnostics(
    camera: Camera,
    *,
    target_mm: Vec3,
    aspect_ratio: float = 1.0,
) -> CameraDiagnostics:
    """Describe camera basis and world-space frustum without a renderer."""

    if camera.pose.frame is not CameraPoseFrame.WORLD:
        raise ValueError("camera diagnostics require a world-frame pose")
    right = _rotate_vector(camera.pose.orientation_xyzw, (1.0, 0.0, 0.0))
    up = _rotate_vector(camera.pose.orientation_xyzw, (0.0, 1.0, 0.0))
    forward = _rotate_vector(camera.pose.orientation_xyzw, (0.0, 0.0, -1.0))
    projection = camera.projection
    near = projection.near_clip_mm
    far = projection.far_clip_mm
    orthographic_half_extents: tuple[float, float] | None = None
    perspective_projection: Projection | None = projection
    if isinstance(projection, OrthographicProjection):
        orthographic_half_extents = _orthographic_half_extents(projection.scale_mm, aspect_ratio)
        perspective_projection = None
    frustum: list[Vec3] = []
    for depth in (near, far):
        if orthographic_half_extents is not None:
            half_width, half_height = orthographic_half_extents
        if orthographic_half_extents is None:
            assert isinstance(perspective_projection, PerspectiveProjection)
            half_width = depth * math.tan(
                math.radians(perspective_projection.horizontal_fov_degrees(aspect_ratio)) / 2
            )
            half_height = depth * math.tan(
                math.radians(perspective_projection.vertical_fov_degrees(aspect_ratio)) / 2
            )
        for vertical in (-half_height, half_height):
            for horizontal in (-half_width, half_width):
                frustum.append(
                    cast(
                        Vec3,
                        tuple(
                            origin
                            + forward_axis * depth
                            + right_axis * horizontal
                            + up_axis * vertical
                            for origin, forward_axis, right_axis, up_axis in zip(
                                camera.pose.position_mm,
                                forward,
                                right,
                                up,
                                strict=True,
                            )
                        ),
                    )
                )
    target_offset = cast(
        Vec3,
        tuple(
            target - origin
            for target, origin in zip(target_mm, camera.pose.position_mm, strict=True)
        ),
    )
    return CameraDiagnostics(
        aspect_ratio=aspect_ratio,
        horizontal_fov_degrees=(
            None
            if isinstance(projection, OrthographicProjection)
            else projection.horizontal_fov_degrees(aspect_ratio)
        ),
        vertical_fov_degrees=(
            None
            if isinstance(projection, OrthographicProjection)
            else projection.vertical_fov_degrees(aspect_ratio)
        ),
        right=right,
        up=up,
        forward=forward,
        frustum_corners_mm=cast(
            tuple[Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3], tuple(frustum)
        ),
        target_depth_mm=sum(
            offset * forward_axis
            for offset, forward_axis in zip(target_offset, forward, strict=True)
        ),
        projected_bounds={},
    )


def _orthographic_half_extents(scale_mm: float, aspect_ratio: float) -> tuple[float, float]:
    """Return Blender AUTO sensor-fit orthographic half extents (width, height)."""

    half_scale = scale_mm / 2
    if aspect_ratio >= 1:
        return half_scale, half_scale / aspect_ratio
    return half_scale * aspect_ratio, half_scale


def _look_orientation(forward: Vec3) -> Quaternion:
    world_up: Vec3 = (0.0, 0.0, 1.0)
    right = _cross(forward, world_up)
    if _length(right) <= _EPSILON:
        world_up = (0.0, 1.0, 0.0)
        right = _cross(forward, world_up)
    right = _normalize(right)
    local_up = _normalize(_cross(right, forward))
    local_back = cast(Vec3, tuple(-value for value in forward))
    matrix = (
        (right[0], local_up[0], local_back[0]),
        (right[1], local_up[1], local_back[1]),
        (right[2], local_up[2], local_back[2]),
    )
    return _quaternion_from_matrix(matrix)


def _quaternion_from_matrix(matrix: tuple[Vec3, Vec3, Vec3]) -> Quaternion:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, scale / 4.0)
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (scale / 4.0, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m01 + m10) / scale, scale / 4.0, (m12 + m21) / scale, (m02 - m20) / scale)
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m02 + m20) / scale, (m12 + m21) / scale, scale / 4.0, (m10 - m01) / scale)


def _axis_angle(axis: Vec3, angle: float) -> Quaternion:
    half_angle = angle / 2.0
    sine = math.sin(half_angle)
    return (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(half_angle))


def _multiply_quaternions(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate_vector(quaternion: Quaternion, vector: Vec3) -> Vec3:
    x, y, z, w = quaternion
    q_vector = (x, y, z)
    twice_cross = cast(Vec3, tuple(2 * value for value in _cross(q_vector, vector)))
    nested_cross = _cross(q_vector, twice_cross)
    return cast(
        Vec3,
        tuple(
            original + w * first_cross + second_cross
            for original, first_cross, second_cross in zip(
                vector,
                twice_cross,
                nested_cross,
                strict=True,
            )
        ),
    )


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _length(value: Vec3) -> float:
    return math.sqrt(sum(component * component for component in value))


def _normalize(value: Vec3) -> Vec3:
    length = _length(value)
    if length <= _EPSILON:
        raise ValueError("camera direction is degenerate")
    return cast(Vec3, tuple(component / length for component in value))
