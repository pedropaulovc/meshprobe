from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest

from meshprobe.controller import BlenderController
from meshprobe.protocol import RenderImageCommand, SceneDescribeCommand, SceneOpenCommand
from meshprobe.service import MeshProbeService


def controller_mock() -> BlenderController:
    return create_autospec(BlenderController, instance=True)


def test_service_requires_open_as_first_command() -> None:
    controller = controller_mock()
    service = MeshProbeService(controller=controller)

    with pytest.raises(ValueError, match=r"scene\.open must be the first"):
        service.execute(SceneDescribeCommand(request_id="describe", op="scene.describe"))

    controller.start.assert_not_called()  # type: ignore[attr-defined]


def test_service_starts_once_and_returns_json_envelopes() -> None:
    controller = controller_mock()
    controller.execute.side_effect = [  # type: ignore[attr-defined]
        {"source_sha256": "a" * 64},
        {"scene": {"components": []}, "session": {"state_sha256": "b" * 64}},
    ]
    service = MeshProbeService(controller=controller)

    opened = service.execute(
        SceneOpenCommand(request_id="open", op="scene.open", source_path="assembly.glb")
    )
    described = service.execute(SceneDescribeCommand(request_id="describe", op="scene.describe"))

    controller.start.assert_called_once_with()  # type: ignore[attr-defined]
    assert opened.model_dump(mode="json") == {
        "request_id": "open",
        "op": "scene.open",
        "result": {"source_sha256": "a" * 64},
    }
    assert described.request_id == "describe"
    assert described.result == {
        "scene": {"components": []},
        "session": {"state_sha256": "b" * 64},
    }


def test_service_closes_controller_and_requires_a_new_open() -> None:
    controller = controller_mock()
    controller.execute.return_value = {}  # type: ignore[attr-defined]
    service = MeshProbeService(controller=controller)
    service.execute(
        SceneOpenCommand(request_id="open", op="scene.open", source_path="assembly.glb")
    )

    service.close()

    controller.close.assert_called_once_with()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match=r"scene\.open must be the first"):
        service.execute(SceneDescribeCommand(request_id="describe", op="scene.describe"))


def test_evaluation_execution_routes_private_render_outputs(tmp_path: Path) -> None:
    controller = controller_mock()
    controller.execute.return_value = {}  # type: ignore[attr-defined]
    controller.render_image.return_value = {"state_sha256": "a" * 64}  # type: ignore[attr-defined]
    service = MeshProbeService(controller=controller)
    service.execute(SceneOpenCommand(request_id="open", op="scene.open", source_path="fixture.glb"))
    command = RenderImageCommand(
        request_id="render",
        op="render.image",
        output_path=str(tmp_path / "render.png"),
    )

    response = service.execute_for_evaluation(
        command,
        evaluator_output_dir=str(tmp_path / "private"),
    )

    assert response.request_id == "render"
    controller.render_image.assert_called_once_with(  # type: ignore[attr-defined]
        command,
        evaluator_output_dir=str(tmp_path / "private"),
    )
