from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from meshprobe.protocol import (
    COMMAND_ADAPTER,
    ComponentFindCommand,
    RenderImageCommand,
    command_json_schema,
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
    encoded = json.dumps(command_json_schema())
    for operation in (
        "scene.open",
        "scene.describe",
        "component.find",
        "component.inspect",
        "view.set",
        "view.orbit",
        "illumination.set",
        "component.display",
        "component.mark",
        "render.image",
        "render.contact_sheet",
        "session.reset",
    ):
        assert operation in encoded


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
