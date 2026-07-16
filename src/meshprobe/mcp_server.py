"""MCP stdio adapter over the shared MeshProbe application service."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from meshprobe.client import MeshProbeClient
from meshprobe.protocol import Command
from meshprobe.service import CommandResponse, MeshProbeService
from meshprobe.workspace import OperationReceipt


def create_server(service: MeshProbeService | None = None) -> FastMCP:
    """Create an MCP server, optionally around an injected service for tests."""

    active_service = service
    client = MeshProbeClient(Path(os.environ.get("MESHPROBE_WORKSPACE", ".")))
    session = os.environ.get("MESHPROBE_SESSION", "default")

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if active_service is not None:
                active_service.close()

    server = FastMCP(
        "MeshProbe",
        instructions=(
            "Inspect 3D models without modifying source geometry. Operations use the named "
            "durable session selected by MESHPROBE_SESSION. Compact state is stored under "
            ".meshprobe for rg, jq, and yq queries."
        ),
        json_response=True,
        lifespan=lifespan,
    )

    @server.tool(name="meshprobe")
    def execute(command: Command) -> CommandResponse | OperationReceipt:
        """Execute one typed read-only 3D inspection operation."""

        if active_service is not None:
            return active_service.execute(command)
        return client.execute(session, command)

    return server


mcp = create_server()


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
