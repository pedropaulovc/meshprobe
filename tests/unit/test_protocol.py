from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from meshprobe.protocol import ComponentFindCommand, command_json_schema, parse_command_json


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
