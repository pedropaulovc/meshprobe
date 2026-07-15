"""Stable identities derived from immutable scene inputs."""

from __future__ import annotations

import hashlib


def stable_component_id(source_sha256: str, instance_path: str) -> str:
    """Derive an ID from immutable source identity and the hierarchy path."""

    digest = hashlib.sha256(f"{source_sha256}\0{instance_path}".encode()).hexdigest()
    return f"cmp_{digest[:24]}"
