"""Authenticated loopback daemon for persistent MeshProbe sessions."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socketserver
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from meshprobe.protocol import COMMAND_ADAPTER
from meshprobe.workspace import SessionManager, atomic_json


class DaemonServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, root: Path, token: str, blender: str | None) -> None:
        super().__init__(("127.0.0.1", 0), RequestHandler)
        self.root = root
        self.token = token
        self.manager = SessionManager(root, blender=blender)


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        if not isinstance(server, DaemonServer):
            return
        line = self.rfile.readline(16 * 1024 * 1024)
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            if not secrets.compare_digest(str(request.get("token", "")), server.token):
                raise PermissionError("invalid daemon token")
            result = dispatch(server, request)
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {
                "ok": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        self.wfile.write((json.dumps(response, separators=(",", ":")) + "\n").encode())


def dispatch(server: DaemonServer, request: dict[str, Any]) -> object:
    action = request.get("action")
    session = str(request.get("session", "default"))
    if action == "ping":
        return {"pid": os.getpid(), "protocol_version": 1}
    if action == "execute":
        command = COMMAND_ADAPTER.validate_python(request.get("command"))
        return server.manager.execute(session, command).model_dump(mode="json")
    if action == "resolve":
        return {"component_id": server.manager.resolve_component(session, str(request["value"]))}
    if action == "list":
        return {"sessions": server.manager.list_sessions()}
    if action == "close":
        return server.manager.close(session).model_dump(mode="json")
    if action == "kill":
        return server.manager.kill(session).model_dump(mode="json")
    if action in {"close_all", "kill_all"}:
        force = action == "kill_all"
        receipts = server.manager.kill_all() if force else server.manager.close_all()
        server.manager.shutdown(force=force)
        threading.Thread(target=server.shutdown, daemon=True).start()
        return {"sessions": [item.model_dump(mode="json") for item in receipts]}
    raise ValueError(f"unsupported daemon action: {action}")


def serve(root: Path, *, blender: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    server = DaemonServer(root, token, blender)
    metadata = {
        "schema_version": 1,
        "host": "127.0.0.1",
        "port": server.server_address[1],
        "pid": os.getpid(),
        "token": token,
    }
    atomic_json(root / "daemon.json", metadata)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        server.manager.shutdown()
        with suppress(FileNotFoundError):
            (root / "daemon.json").unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--blender")
    arguments = parser.parse_args()
    try:
        serve(arguments.workspace.resolve(), blender=arguments.blender)
    except Exception as error:
        print(f"meshprobe daemon failed: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
