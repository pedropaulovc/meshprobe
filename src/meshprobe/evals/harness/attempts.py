"""Common auditable artifacts for every evaluation agent attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meshprobe.workspace import atomic_text


@dataclass(frozen=True)
class AttemptFiles:
    root: Path
    prompt: Path
    stream: Path
    stderr: Path


def prepare_attempt_files(
    root: Path,
    prompt: str,
    *,
    reset_streams: bool = False,
) -> AttemptFiles:
    """Persist the exact prompt and materialize live provider/protocol logs."""

    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    files = AttemptFiles(
        root=resolved,
        prompt=resolved / "prompt.txt",
        stream=resolved / "stream.jsonl",
        stderr=resolved / "stderr.log",
    )
    atomic_text(files.prompt, prompt)
    for path in (files.stream, files.stderr):
        if reset_streams:
            path.write_bytes(b"")
            continue
        path.touch(exist_ok=True)
    return files
