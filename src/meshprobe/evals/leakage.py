"""Leakage guards separating agent-visible episodes from evaluator ground truth."""

from __future__ import annotations

import json
from pathlib import Path

from meshprobe.evals.schemas import EpisodeGroundTruth, EpisodeSpec

_PRIVATE_KEYS = {
    "answer",
    "component_roles",
    "component_paths",
    "state_requirements",
    "evidence_requirements",
    "forbidden_public_tokens",
}


class LeakageError(ValueError):
    """Public episode content reveals evaluator-owned information."""


def validate_public_episode(spec: EpisodeSpec, truth: EpisodeGroundTruth) -> None:
    if spec.episode_id != truth.episode_id or spec.model_sha256 != truth.model_sha256:
        raise LeakageError("public episode and private ground truth do not identify the same case")
    payload = spec.model_dump(mode="json")
    leaked_keys = _find_keys(payload, _PRIVATE_KEYS)
    if leaked_keys:
        raise LeakageError(f"public episode contains private keys: {sorted(leaked_keys)}")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    leaked_tokens = [token for token in truth.forbidden_public_tokens if token in serialized]
    if leaked_tokens:
        raise LeakageError(f"public episode contains evaluator tokens: {sorted(leaked_tokens)}")
    if spec.episode_id in spec.prompt or spec.model_sha256 in spec.prompt:
        raise LeakageError("prompt contains an opaque evaluator identifier")


def validate_public_directory(public_root: Path, private_root: Path) -> None:
    public = public_root.resolve(strict=True)
    private = private_root.resolve(strict=True)
    if private.is_relative_to(public):
        raise LeakageError("private evaluator directory must not be under the public directory")
    forbidden_fragments = {"ground_truth", "evaluator", "private"}
    for path in public.rglob("*"):
        if not path.is_file():
            continue
        lowered_parts = {part.lower() for part in path.relative_to(public).parts}
        if lowered_parts & forbidden_fragments:
            raise LeakageError(f"public artifact uses a private path name: {path}")


def scan_error_message(message: str, truth: EpisodeGroundTruth) -> None:
    leaked = [token for token in truth.forbidden_public_tokens if token in message]
    if leaked:
        raise LeakageError(f"tool error leaked evaluator tokens: {sorted(leaked)}")


def _find_keys(value: object, forbidden: set[str]) -> set[str]:
    if isinstance(value, dict):
        found = {str(key) for key in value if str(key) in forbidden}
        for nested in value.values():
            found.update(_find_keys(nested, forbidden))
        return found
    if isinstance(value, list):
        list_found: set[str] = set()
        for nested in value:
            list_found.update(_find_keys(nested, forbidden))
        return list_found
    return set()
