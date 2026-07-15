from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import create_autospec

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from meshprobe import cli
from meshprobe.controller import BlenderController
from meshprobe.mcp_server import create_server
from meshprobe.protocol import SceneOpenCommand, command_json_schema
from meshprobe.service import MeshProbeService

runner = CliRunner()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def service_with_result(result: object) -> MeshProbeService:
    controller = create_autospec(BlenderController, instance=True)
    controller.execute.return_value = result
    return MeshProbeService(controller=controller)


@pytest.mark.anyio
async def test_mcp_tool_schema_contains_the_public_command_contract() -> None:
    service = service_with_result({})
    server = create_server(service)
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 1
    tool_schema = listed.tools[0].inputSchema
    mcp_command = tool_schema["$defs"]["Command"]
    public_command = command_json_schema()
    assert mcp_command["discriminator"] == public_command["discriminator"]
    assert mcp_command["oneOf"] == public_command["oneOf"]


@pytest.mark.anyio
async def test_cli_and_mcp_return_the_same_service_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = {"source_sha256": "a" * 64, "components": []}
    command = SceneOpenCommand(
        request_id="open-contract",
        op="scene.open",
        source_path="assembly.glb",
    )

    cli_service = service_with_result(result)
    monkeypatch.setattr(cli, "MeshProbeService", lambda **_: cli_service)
    commands = tmp_path / "commands.jsonl"
    commands.write_text(command.model_dump_json() + "\n", encoding="utf-8")
    cli_result = runner.invoke(cli.app, ["run", str(commands)])
    assert cli_result.exit_code == 0
    cli_envelope = json.loads(cli_result.stdout)["results"][0]

    mcp_service = service_with_result(result)
    server = create_server(mcp_service)
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        mcp_result = await session.call_tool(
            "meshprobe",
            {"command": command.model_dump(mode="json")},
        )

    assert not mcp_result.isError
    assert cast(dict[str, object], mcp_result.structuredContent) == cli_envelope


@pytest.mark.anyio
async def test_mcp_rejects_an_operation_before_scene_open() -> None:
    service = service_with_result({})
    server = create_server(service)
    async with create_connected_server_and_client_session(
        server, raise_exceptions=False
    ) as session:
        result = await session.call_tool(
            "meshprobe",
            {"command": {"request_id": "describe", "op": "scene.describe"}},
        )

    assert result.isError
    assert "scene.open must be the first" in result.content[0].text  # type: ignore[union-attr]
