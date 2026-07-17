"""One-way evaluation corpus migration with identity-preserving audits."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from meshprobe.evals.factory import CorpusBuild, validate_corpus
from meshprobe.evals.schemas import EVALUATED_OPERATIONS, TaskFamily
from meshprobe.sources import sha256_file


@dataclass(frozen=True)
class MigrationAudit:
    source_version: str
    destination_version: str
    models: int
    episodes: int
    model_hashes_unchanged: bool
    episode_ids_unchanged: bool
    truth_payloads_unchanged: bool
    full_investigations_valid: bool


def migrate_corpus_v2(
    source: Path,
    output_root: Path,
    *,
    corpus_version: str,
    opaque_family: str | None = None,
) -> tuple[CorpusBuild, MigrationAudit]:
    """Hard-link immutable models and rewrite only versioned JSON contracts."""

    source = source.expanduser().resolve(strict=True)
    source_manifest = _read_json(source / "public" / "manifest.json")
    if source_manifest.get("schema_version") != 1:
        raise ValueError("eval migration requires a schema version 1 source corpus")
    resolved_output = output_root.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    destination = resolved_output / corpus_version
    if destination.exists():
        build = validate_corpus(destination)
        return build, audit_migration(source, destination)
    staging = resolved_output / f".{corpus_version}.migrating"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging, copy_function=_link_or_copy)

    old_opaque = _opaque_family(source_manifest)
    for path in sorted((staging / "public" / "episodes").glob("*.json")):
        payload = _migrated_payload(_read_json(path))
        _write_json(path, payload)
    for path in sorted((staging / "private" / "ground_truth").glob("*.json")):
        payload = _migrated_payload(_read_json(path))
        if opaque_family is not None and payload.get("generator_family") == old_opaque:
            payload["generator_family"] = opaque_family
        _write_json(path, payload)

    manifest = _migrated_payload(source_manifest)
    manifest["corpus_version"] = corpus_version
    families = list(manifest["generator_families"])
    if opaque_family is not None and old_opaque is not None:
        families = [opaque_family if family == old_opaque else family for family in families]
    manifest["generator_families"] = families
    manifest["episode_sha256"] = {
        episode_id: sha256_file(staging / "public" / "episodes" / f"{episode_id}.json")
        for episode_id in manifest["episodes"]
    }
    _write_json(staging / "public" / "manifest.json", manifest)
    os.replace(staging, destination)
    audit = audit_migration(source, destination)
    build = validate_corpus(destination)
    return build, audit


def audit_migration(source: Path, destination: Path) -> MigrationAudit:
    source_manifest = _read_json(source / "public" / "manifest.json")
    destination_manifest = _read_json(destination / "public" / "manifest.json")
    source_ids = tuple(source_manifest["episodes"])
    destination_ids = tuple(destination_manifest["episodes"])
    if source_manifest["model_sha256"] != destination_manifest["model_sha256"]:
        raise RuntimeError("migration changed model identities or hashes")
    if source_ids != destination_ids:
        raise RuntimeError("migration changed episode identities or order")

    truth_unchanged = True
    full_investigations_valid = True
    for episode_id in source_ids:
        old_spec = _read_json(source / "public" / "episodes" / f"{episode_id}.json")
        new_spec = _read_json(destination / "public" / "episodes" / f"{episode_id}.json")
        _assert_allowed_payload_change(old_spec, new_spec, private=False)
        old_truth = _read_json(source / "private" / "ground_truth" / f"{episode_id}.json")
        new_truth = _read_json(destination / "private" / "ground_truth" / f"{episode_id}.json")
        _assert_allowed_payload_change(old_truth, new_truth, private=True)
        if _normalized_payload(old_truth, private=True) != _normalized_payload(
            new_truth, private=True
        ):
            truth_unchanged = False
        if new_spec.get("family") == TaskFamily.FULL_INVESTIGATION.value:
            full_investigations_valid &= set(new_spec["required_operations"]) == {
                operation.value for operation in EVALUATED_OPERATIONS
            }
    if not truth_unchanged:
        raise RuntimeError("migration changed evaluator truth beyond the declared rename")
    if not full_investigations_valid:
        raise RuntimeError("migration produced an invalid full-investigation operation set")
    return MigrationAudit(
        source_version=str(source_manifest["corpus_version"]),
        destination_version=str(destination_manifest["corpus_version"]),
        models=len(source_manifest["model_sha256"]),
        episodes=len(source_ids),
        model_hashes_unchanged=True,
        episode_ids_unchanged=True,
        truth_payloads_unchanged=True,
        full_investigations_valid=True,
    )


def codify_opaque_family(corpus_root: Path, *, name: str = "opaque_family_v7") -> None:
    """Apply the held-out-family construction used by private-v6 as a production step."""

    manifest_path = corpus_root / "public" / "manifest.json"
    manifest = _read_json(manifest_path)
    episode_root = corpus_root / "public" / "episodes"
    truth_root = corpus_root / "private" / "ground_truth"
    first_episode = str(manifest["episodes"][0])
    first_spec = _read_json(episode_root / f"{first_episode}.json")
    for episode_id in manifest["episodes"]:
        spec = _read_json(episode_root / f"{episode_id}.json")
        if spec["model_file"] != first_spec["model_file"]:
            continue
        truth_path = truth_root / f"{episode_id}.json"
        truth = _read_json(truth_path)
        truth["generator_family"] = name
        _write_json(truth_path, truth)
    families = [
        family
        for family in manifest["generator_families"]
        if not family.startswith("opaque_family_")
    ]
    manifest["generator_families"] = [*families, name]
    _write_json(manifest_path, manifest)


def restore_schema_v1_source(
    corpus_root: Path,
    *,
    corpus_version: str,
    opaque_family: str | None = None,
) -> None:
    """Restore a local v1 corpus whose JSON was linked by an early migrator build."""

    episode_root = corpus_root / "public" / "episodes"
    truth_root = corpus_root / "private" / "ground_truth"
    for path in sorted(episode_root.glob("*.json")):
        payload = _read_json(path)
        payload["schema_version"] = 1
        _rename_operation_v1(payload)
        _remove_view_rotate(payload)
        _write_json(path, payload)
    for path in sorted(truth_root.glob("*.json")):
        payload = _read_json(path)
        payload["schema_version"] = 1
        _rename_operation_v1(payload)
        _remove_view_rotate(payload)
        if opaque_family is not None and str(payload.get("generator_family", "")).startswith(
            "opaque_family_"
        ):
            payload["generator_family"] = opaque_family
        _write_json(path, payload)
    manifest_path = corpus_root / "public" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["schema_version"] = 1
    manifest["corpus_version"] = corpus_version
    if opaque_family is not None:
        manifest["generator_families"] = [
            opaque_family if str(family).startswith("opaque_family_") else family
            for family in manifest["generator_families"]
        ]
    manifest["episode_sha256"] = {
        episode_id: sha256_file(episode_root / f"{episode_id}.json")
        for episode_id in manifest["episodes"]
    }
    _write_json(manifest_path, manifest)


def _assert_allowed_payload_change(
    old: dict[str, Any], new: dict[str, Any], *, private: bool
) -> None:
    expected = _normalized_payload(old, private=private)
    observed = _normalized_payload(new, private=private)
    if expected != observed:
        raise RuntimeError("migration changed an episode beyond schema and operation renames")


def _normalized_payload(payload: dict[str, Any], *, private: bool) -> dict[str, Any]:
    normalized = cast(dict[str, Any], json.loads(json.dumps(payload)))
    normalized["schema_version"] = 2
    _rename_operation(normalized)
    _upgrade_full_operation_contract(normalized)
    if private and str(normalized.get("generator_family", "")).startswith("opaque_family_"):
        normalized["generator_family"] = "opaque_family"
    return normalized


def _migrated_payload(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = cast(dict[str, Any], json.loads(json.dumps(payload)))
    migrated["schema_version"] = 2
    _rename_operation(migrated)
    _upgrade_full_operation_contract(migrated)
    return migrated


def _upgrade_full_operation_contract(payload: dict[str, Any]) -> None:
    required = payload.get("required_operations")
    if not isinstance(required, list):
        return
    legacy_operations = {
        operation.value for operation in EVALUATED_OPERATIONS if operation.value != "view.rotate"
    }
    if set(required) != legacy_operations:
        return
    payload["required_operations"] = [operation.value for operation in EVALUATED_OPERATIONS]


def _remove_view_rotate(payload: dict[str, Any]) -> None:
    required = payload.get("required_operations")
    if not isinstance(required, list):
        return
    payload["required_operations"] = [
        operation for operation in required if operation != "view.rotate"
    ]


def _rename_operation(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if child == "scene.describe":
                value[key] = "session.snapshot"
                continue
            _rename_operation(child)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            if child == "scene.describe":
                value[index] = "session.snapshot"
                continue
            _rename_operation(child)


def _rename_operation_v1(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if child == "session.snapshot":
                value[key] = "scene.describe"
                continue
            _rename_operation_v1(child)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            if child == "session.snapshot":
                value[index] = "scene.describe"
                continue
            _rename_operation_v1(child)


def _opaque_family(manifest: dict[str, Any]) -> str | None:
    return next(
        (
            str(family)
            for family in manifest.get("generator_families", [])
            if str(family).startswith("opaque_family_")
        ),
        None,
    )


def _link_or_copy(source: str, destination: str) -> str:
    if not source.endswith(".glb"):
        shutil.copy2(source, destination)
        return destination
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
