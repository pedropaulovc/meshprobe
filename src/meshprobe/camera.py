"""Renderer-independent camera geometry."""

from __future__ import annotations

import math
from typing import cast

from meshprobe.models import (
    Camera,
    CameraDiagnostics,
    OrthographicProjection,
    Pose,
    Projection,
    Quaternion,
    Vec3,
)

_EPSILON = 1e-12


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


def camera_diagnostics(
    camera: Camera,
    *,
    target_mm: Vec3,
    aspect_ratio: float = 1.0,
) -> CameraDiagnostics:
    """Describe camera basis and world-space frustum without a renderer."""

    right = _rotate_vector(camera.pose.orientation_xyzw, (1.0, 0.0, 0.0))
    up = _rotate_vector(camera.pose.orientation_xyzw, (0.0, 1.0, 0.0))
    forward = _rotate_vector(camera.pose.orientation_xyzw, (0.0, 0.0, -1.0))
    projection = camera.projection
    near = projection.near_clip_mm
    far = projection.far_clip_mm
    frustum: list[Vec3] = []
    for depth in (near, far):
        if isinstance(projection, OrthographicProjection):
            half_height = projection.scale_mm / 2
            half_width = half_height * aspect_ratio
        else:
            half_width = depth * math.tan(
                math.radians(projection.horizontal_fov_degrees(aspect_ratio)) / 2
            )
            half_height = depth * math.tan(
                math.radians(projection.vertical_fov_degrees(aspect_ratio)) / 2
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
