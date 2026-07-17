from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from meshprobe.protocol import (
    COMMAND_ADAPTER,
    ComponentFindCommand,
    ComponentOcclusionCommand,
    RenderContactSheetCommand,
    RenderImageCommand,
    ViewMoveCommand,
    ViewRotateCommand,
    command_json_schema,
    command_result_json_schema,
    parse_command_json,
)


def test_find_command_round_trip() -> None:
    command = parse_command_json(
        json.dumps(
            {
                "request_id": "req-1",
                "op": "component.find",
                "selector": {"kind": "glob", "pattern": "assembly/**/idler*"},
            }
        )
    )
    assert isinstance(command, ComponentFindCommand)
    assert command.selector.pattern.endswith("idler*")


def test_unknown_operation_fails() -> None:
    with pytest.raises(ValidationError):
        parse_command_json('{"request_id":"req-1","op":"geometry.delete"}')


def test_schema_contains_all_public_operations() -> None:
    schema = command_json_schema()
    encoded = json.dumps(schema)
    for operation in (
        "scene.open",
        "session.snapshot",
        "component.find",
        "component.inspect",
        "component.occlusion",
        "view.set",
        "view.orbit",
        "view.frame",
        "view.move",
        "view.rotate",
        "illumination.set",
        "component.display",
        "component.mark",
        "render.image",
        "render.contact_sheet",
        "session.reset",
    ):
        assert operation in encoded
    assert schema["$defs"]["CameraPoseFrame"]["enum"] == ["source", "world"]
    assert schema["$defs"]["Pose"]["properties"]["frame"]["$ref"] == ("#/$defs/CameraPoseFrame")


def test_result_schema_covers_every_command_operation() -> None:
    """Every op in the Command union must have a result schema entry, or a client
    reading `meshprobe schema --kind results` silently has no schema for it."""
    result_schema = command_result_json_schema()
    for operation in (
        "scene.open",
        "session.snapshot",
        "component.find",
        "component.inspect",
        "component.occlusion",
        "view.set",
        "view.orbit",
        "view.frame",
        "view.move",
        "view.rotate",
        "illumination.set",
        "component.display",
        "component.mark",
        "render.image",
        "render.contact_sheet",
        "session.reset",
    ):
        assert operation in result_schema, f"missing result schema for {operation}"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "request_id": "invalid-view-set-frame",
            "op": "view.set",
            "camera": {
                "pose": {
                    "position_mm": [0, 0, 100],
                    "orientation_xyzw": [0, 0, 0, 1],
                    "frame": "camera",
                },
                "projection": {"mode": "perspective"},
            },
        },
        {
            "request_id": "invalid-contact-sheet-frame",
            "op": "render.contact_sheet",
            "output_path": "sheet.png",
            "recipe": "custom_3x3",
            "focus_component_ids": ["cmp_target"],
            "panels": [
                {
                    "caption": "invalid camera frame",
                    "camera": {
                        "pose": {
                            "position_mm": [0, 0, 100],
                            "orientation_xyzw": [0, 0, 0, 1],
                            "frame": "component",
                        },
                        "projection": {"mode": "perspective"},
                    },
                }
            ]
            * 9,
        },
    ],
)
def test_command_cameras_only_accept_source_or_world_frames(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match=r"source.*world"):
        COMMAND_ADAPTER.validate_python(payload)


def test_occlusion_command_validates_sampling_dimensions() -> None:
    command = COMMAND_ADAPTER.validate_python(
        {
            "request_id": "occlusion",
            "op": "component.occlusion",
            "component_ids": ["target"],
            "max_samples_per_component": 512,
        }
    )

    assert isinstance(command, ComponentOcclusionCommand)
    assert command.max_samples_per_component == 512
    with pytest.raises(ValidationError):
        COMMAND_ADAPTER.validate_python(
            {
                "request_id": "occlusion",
                "op": "component.occlusion",
                "component_ids": ["target"],
                "max_samples_per_component": 0,
            }
        )


@pytest.mark.parametrize("field", ["azimuth_degrees", "elevation_degrees", "roll_degrees"])
def test_orbit_rejects_nonfinite_angles(field: str) -> None:
    payload = {
        "request_id": "req-orbit",
        "op": "view.orbit",
        "target_mm": [0, 0, 0],
        "azimuth_degrees": 30,
        "elevation_degrees": 20,
        "roll_degrees": 0,
        "distance_mm": 100,
        "projection": {"mode": "perspective"},
    }
    payload[field] = math.inf
    with pytest.raises(ValidationError):
        COMMAND_ADAPTER.validate_python(payload)


def test_move_combines_world_and_camera_deltas() -> None:
    command = COMMAND_ADAPTER.validate_python(
        {
            "request_id": "move",
            "op": "view.move",
            "world_delta_mm": [0, 0, 100],
            "camera_delta_mm": [-25, 0, 50],
        }
    )

    assert isinstance(command, ViewMoveCommand)
    assert command.world_delta_mm == (0, 0, 100)
    assert command.camera_delta_mm == (-25, 0, 50)
    with pytest.raises(ValidationError, match="non-zero"):
        COMMAND_ADAPTER.validate_python({"request_id": "stationary", "op": "view.move"})


def test_rotate_accepts_source_axis_and_preserves_projection_by_default() -> None:
    command = COMMAND_ADAPTER.validate_python(
        {
            "request_id": "rotate-source-y",
            "op": "view.rotate",
            "target_mm": [0, 10, 20],
            "axis": "y",
            "degrees": 145,
            "frame": "source",
            "basis": {"x": [0, 1, 0], "y": [-1, 0, 0], "z": [0, 0, 1]},
        }
    )

    assert isinstance(command, ViewRotateCommand)
    assert command.frame == "source"
    assert command.axis == "y"
    assert command.basis.x == (0, 1, 0)
    assert command.projection is None


def test_rotate_accepts_camera_frame_but_not_component() -> None:
    base = {
        "request_id": "rotate-camera",
        "op": "view.rotate",
        "target_mm": [0, 0, 0],
        "axis": "x",
        "degrees": 15,
    }

    command = COMMAND_ADAPTER.validate_python({**base, "frame": "camera"})
    assert isinstance(command, ViewRotateCommand)
    assert command.frame == "camera"

    with pytest.raises(ValidationError):
        COMMAND_ADAPTER.validate_python({**base, "frame": "component"})


def test_render_command_bounds_engine_and_samples() -> None:
    command = COMMAND_ADAPTER.validate_python(
        {
            "request_id": "render",
            "op": "render.image",
            "output_path": "evidence.png",
            "width": 2048,
            "height": 1024,
            "samples": 128,
            "engine": "cycles",
        }
    )
    assert isinstance(command, RenderImageCommand)
    assert command.samples == 128
    assert command.engine == "cycles"

    with pytest.raises(ValidationError, match="Extra inputs"):
        COMMAND_ADAPTER.validate_python(
            {
                "request_id": "private-pass",
                "op": "render.image",
                "output_path": "evidence.png",
                "evaluator_output_dir": "private",
            }
        )

    with pytest.raises(ValidationError):
        COMMAND_ADAPTER.validate_python(
            {
                "request_id": "oversized",
                "op": "render.image",
                "output_path": "evidence.png",
                "width": 20_000,
            }
        )


def test_custom_contact_sheet_requires_nine_declared_panels() -> None:
    panel = {
        "caption": "inspection",
        "orbit": {
            "azimuth_degrees": 45,
            "elevation_degrees": 20,
            "distance_mm": 1_000,
            "projection": {"mode": "orthographic", "scale_mm": 500},
        },
        "illumination": {"preset": "raking_left"},
    }
    command = COMMAND_ADAPTER.validate_python(
        {
            "request_id": "custom-sheet",
            "op": "render.contact_sheet",
            "recipe": "custom_3x3",
            "focus_component_ids": ["cmp_target"],
            "output_path": "sheet.png",
            "panels": [panel] * 9,
        }
    )

    assert isinstance(command, RenderContactSheetCommand)
    assert len(command.panels) == 9
    with pytest.raises(ValidationError, match="exactly nine"):
        COMMAND_ADAPTER.validate_python(
            {
                "request_id": "short-sheet",
                "op": "render.contact_sheet",
                "recipe": "custom_3x3",
                "focus_component_ids": ["cmp_target"],
                "output_path": "sheet.png",
                "panels": [panel] * 8,
            }
        )
    with pytest.raises(ValidationError, match="does not accept custom"):
        COMMAND_ADAPTER.validate_python(
            {
                "request_id": "focused-sheet",
                "op": "render.contact_sheet",
                "focus_component_ids": ["cmp_target"],
                "output_path": "sheet.png",
                "panels": [panel],
            }
        )


def test_dolly_zoom_requires_perspective_reference() -> None:
    with pytest.raises(ValidationError, match="reference focal length and distance"):
        COMMAND_ADAPTER.validate_python(
            {
                "request_id": "dolly-sheet",
                "op": "render.contact_sheet",
                "recipe": "custom_3x3",
                "focus_component_ids": ["cmp_target"],
                "output_path": "sheet.png",
                "panels": [
                    {
                        "caption": "dolly",
                        "experiment": "dolly_zoom",
                        "orbit": {
                            "azimuth_degrees": 45,
                            "elevation_degrees": 20,
                            "distance_mm": 1_000,
                            "projection": {"mode": "perspective", "focal_length_mm": 85},
                        },
                    }
                ]
                * 9,
            }
        )
