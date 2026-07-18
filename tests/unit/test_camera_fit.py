from __future__ import annotations

import math
from typing import Any

import pytest

from meshprobe.camera import DEFAULT_FRAME_FILL, bounds_fit_distance_mm

# Camera looking down -Z with the conventional Blender basis.
FORWARD = (0.0, 0.0, -1.0)
RIGHT = (1.0, 0.0, 0.0)
UP = (0.0, 1.0, 0.0)


def test_fit_distance_frames_limiting_axis_exactly_at_full_fill() -> None:
    # A box 200 x 100 x 400 (mm) centred at the origin, viewed down -Z with a
    # 90-degree field of view (tan(45deg) = 1) on both axes. The nearest face
    # sits at z = +200, so the width (half = 100) governs the fit.
    distance = bounds_fit_distance_mm(
        minimum_mm=(-100.0, -50.0, -200.0),
        maximum_mm=(100.0, 50.0, 200.0),
        forward=FORWARD,
        right=RIGHT,
        up=UP,
        horizontal_half_tan=1.0,
        vertical_half_tan=1.0,
        frame_fill=1.0,
    )
    # Nearest face depth from camera = distance - 200; width fills the frame when
    # 100 / (distance - 200) == 1  =>  distance == 300.
    assert distance == pytest.approx(300.0)

    # At that distance the nearest corner touches the frame edge exactly and the
    # far corner stays comfortably inside it (no clipping).
    near_depth = distance - 200.0
    far_depth = distance + 200.0
    assert 100.0 / near_depth == pytest.approx(1.0)
    assert 100.0 / far_depth < 1.0


def test_smaller_fill_pushes_the_camera_back_for_margin() -> None:
    kwargs: dict[str, Any] = dict(
        minimum_mm=(-100.0, -50.0, -200.0),
        maximum_mm=(100.0, 50.0, 200.0),
        forward=FORWARD,
        right=RIGHT,
        up=UP,
        horizontal_half_tan=1.0,
        vertical_half_tan=1.0,
    )
    full = bounds_fit_distance_mm(**kwargs, frame_fill=1.0)
    padded = bounds_fit_distance_mm(**kwargs, frame_fill=0.9)
    # 90% fill leaves a margin, so the camera must sit farther back than a 100% fit.
    assert padded > full
    # Width governs: near-face depth = 100 / (htan * fill) => distance = that + 200.
    assert padded == pytest.approx(100.0 / 0.9 + 200.0)


def test_fit_is_far_tighter_than_a_typical_source_camera() -> None:
    # Reproduces the shape of the reported regression: a tall, thin assembly
    # (628 mm in Y, 212 mm across) with a source camera parked far away down -Z,
    # so the model fills a small vertical slice of the frame.
    minimum_mm = (-106.0, -314.0, -106.0)
    maximum_mm = (106.0, 314.0, 106.0)
    half_tan = math.tan(math.radians(39.6) / 2)  # ~50 mm lens on a 36 mm sensor
    # A source camera 6000 mm away frames the 628 mm-tall model to a small strip.
    source_distance = 6000.0
    source_fill = 314.0 / (half_tan * (source_distance - 106.0))

    fitted = bounds_fit_distance_mm(
        minimum_mm=minimum_mm,
        maximum_mm=maximum_mm,
        forward=(0.0, 0.0, -1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        horizontal_half_tan=half_tan,
        vertical_half_tan=half_tan,
        frame_fill=DEFAULT_FRAME_FILL,
    )
    # The height (half = 314) governs; the nearest face is at z = +106, so
    # fitted = 314 / (half_tan * fill) + 106.
    near_depth = fitted - 106.0
    fitted_fill = 314.0 / (half_tan * near_depth)
    assert fitted_fill == pytest.approx(DEFAULT_FRAME_FILL)
    # The default frame fills the frame many times more tightly than the source.
    assert fitted_fill > source_fill * 4


def test_fit_leaves_a_clearance_margin_for_wide_fov_diagonal_fits() -> None:
    # A cube centred at the origin, viewed along its body diagonal by a very wide
    # lens. The nearest corner sits almost exactly on the view axis (zero
    # horizontal/vertical offset), so the per-corner screen-space fit alone
    # converges to (but does not exceed) that corner's own depth, leaving no
    # margin between the camera and the nearest surface.
    half_extent = 500.0
    minimum_mm = (-half_extent, -half_extent, -half_extent)
    maximum_mm = (half_extent, half_extent, half_extent)
    forward = (-1 / math.sqrt(3), -1 / math.sqrt(3), -1 / math.sqrt(3))
    right = (-1 / math.sqrt(2), 1 / math.sqrt(2), 0.0)
    up = (-1 / math.sqrt(6), -1 / math.sqrt(6), 2 / math.sqrt(6))

    distance = bounds_fit_distance_mm(
        minimum_mm=minimum_mm,
        maximum_mm=maximum_mm,
        forward=forward,
        right=right,
        up=up,
        horizontal_half_tan=3.6,
        vertical_half_tan=2.4,
        frame_fill=DEFAULT_FRAME_FILL,
    )

    nearest_corner_depth = half_extent * sum(forward)  # the (500, 500, 500) corner
    # The nearest corner stays strictly in front of the camera by at least 1 mm,
    # instead of landing exactly on it (distance == -nearest_corner_depth).
    assert distance > -nearest_corner_depth + 0.5
