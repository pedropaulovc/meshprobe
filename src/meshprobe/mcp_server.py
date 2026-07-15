"""MCP stdio adapter over the shared MeshProbe application service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from meshprobe.protocol import Command
from meshprobe.service import CommandResponse, MeshProbeService


def create_server(service: MeshProbeService | None = None) -> FastMCP:
    """Create an MCP server, optionally around an injected service for tests."""

    active_service = service or MeshProbeService()

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            active_service.close()

    server = FastMCP(
        "MeshProbe",
        instructions=(
            "Inspect 3D models without modifying source geometry. Call scene.open first, "
            "then use the returned stable component IDs for display, marking, and evidence."
        ),
        json_response=True,
        lifespan=lifespan,
    )

    @server.tool(name="meshprobe")
    def execute(command: Command) -> CommandResponse:
        """Execute one typed read-only 3D inspection operation."""

        return active_service.execute(command)

    return server


mcp = create_server()


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
