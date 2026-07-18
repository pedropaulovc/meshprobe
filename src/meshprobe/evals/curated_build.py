"""Checkpointed publication of controlled curated assembly variants."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from meshprobe.evals.curated import CuratedCatalog, topology_sha256
from meshprobe.evals.generators import MetamorphicVariant
from meshprobe.sources import sha256_file, snapshot_source


@dataclass(frozen=True)
class CuratedBuild:
    root: Path
    model_sha256: dict[str, str]
    model_count: int


def build_curated_variants(
    catalog: CuratedCatalog,
    sources: dict[str, Path],
    output_root: Path,
    *,
    version: str = "curated-v1",
    blender: str = "blender",
) -> CuratedBuild:
    _validate_curated_sources(catalog, sources)
    destination = output_root.expanduser().resolve() / version
    builder = Path(__file__).parents[1] / "blender" / "curated_builder.py"
    catalog_hash = _catalog_hash(catalog)
    builder_hash = sha256_file(builder)
    if destination.exists():
        return validate_curated_build(destination, catalog_hash, builder_hash)
    resolved_blender = shutil.which(blender)
    if resolved_blender is None:
        raise RuntimeError(
            f"Blender executable not found: {blender} (pass --blender /path/to/blender to override)"
        )
    snapshots = {source_id: snapshot_source(path) for source_id, path in sources.items()}
    staging = output_root.expanduser().resolve() / f".{version}.building"
    build_id = _build_id(version, catalog_hash, builder_hash, sources)
    _prepare_staging(staging, build_id)
    models = staging / "public" / "models"
    roles = staging / "private" / "roles"
    models.mkdir(parents=True, exist_ok=True)
    roles.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, object]] = []
    for assembly in catalog.assemblies:
        for variant in MetamorphicVariant:
            stem = f"{assembly.assembly_id}--{variant}"
            jobs.append(
                {
                    "key": stem,
                    "variant": variant,
                    "output": str(models / f"{stem}.glb"),
                    "metadata": str(roles / f"{stem}.json"),
                    "placements": [
                        {
                            "path": str(sources[placement.source_id]),
                            "role": placement.role,
                            "position_mm": placement.position_mm,
                        }
                        for placement in assembly.placements
                    ],
                }
            )
    jobs_path = staging / "jobs.json"
    _write_json(jobs_path, {"build_id": build_id, "jobs": jobs})
    checkpoint = staging / "checkpoint.json"
    completed = subprocess.run(
        (
            resolved_blender,
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python",
            str(builder),
            "--",
            str(jobs_path),
            str(checkpoint),
        ),
        capture_output=True,
        text=True,
        timeout=7_200,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "curated Blender build failed\n"
            + completed.stdout[-20_000:]
            + completed.stderr[-20_000:]
        )
    for source_id, path in sources.items():
        if snapshot_source(path) != snapshots[source_id]:
            raise RuntimeError(f"curated source changed during assembly build: {source_id}")
    expected = len(catalog.assemblies) * len(MetamorphicVariant)
    model_paths = sorted(models.glob("*.glb"))
    role_paths = sorted(roles.glob("*.json"))
    if len(model_paths) != expected or len(role_paths) != expected:
        raise RuntimeError("curated assembly publication count is incomplete")
    model_hashes = {path.name: sha256_file(path) for path in model_paths}
    _write_json(
        staging / "public" / "manifest.json",
        {
            "schema_version": 1,
            "version": version,
            "catalog_sha256": catalog_hash,
            "builder_sha256": builder_hash,
            "variants": [str(variant) for variant in MetamorphicVariant],
            "model_sha256": model_hashes,
        },
    )
    jobs_path.unlink()
    checkpoint.unlink()
    (staging / "build.json").unlink()
    os.replace(staging, destination)
    return validate_curated_build(destination, catalog_hash, builder_hash)


def _validate_curated_sources(catalog: CuratedCatalog, sources: dict[str, Path]) -> None:
    expected = {source.source_id: source for source in catalog.sources}
    if sources.keys() != expected.keys():
        missing = sorted(expected.keys() - sources.keys())
        extra = sorted(sources.keys() - expected.keys())
        raise ValueError(f"curated source set mismatch: missing={missing}, extra={extra}")
    content_hashes: dict[Path, str] = {}
    topology_hashes: dict[Path, str] = {}
    for source_id, declared in expected.items():
        path = sources[source_id].expanduser().resolve(strict=True)
        content = content_hashes.setdefault(path, sha256_file(path))
        if content != declared.source_sha256:
            raise RuntimeError(f"curated source hash mismatch: {source_id}")
        topology = topology_hashes.setdefault(path, topology_sha256(path))
        if topology != declared.topology_sha256:
            raise RuntimeError(f"curated topology hash mismatch: {source_id}")


def validate_curated_build(root: Path, catalog_hash: str, builder_hash: str) -> CuratedBuild:
    manifest_path = root / "public" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("catalog_sha256") != catalog_hash:
        raise RuntimeError("curated build belongs to a different catalog")
    if manifest.get("builder_sha256") != builder_hash:
        raise RuntimeError("curated build belongs to a different Blender builder")
    if manifest.get("variants") != [str(variant) for variant in MetamorphicVariant]:
        raise RuntimeError("curated build belongs to a different metamorphic variant set")
    declared = manifest.get("model_sha256")
    if not isinstance(declared, dict):
        raise RuntimeError("curated build manifest has no model hash map")
    actual = {
        path.name: sha256_file(path) for path in sorted((root / "public" / "models").glob("*.glb"))
    }
    if declared != actual:
        raise RuntimeError("curated model files do not match their manifest")
    roles = {path.stem for path in (root / "private" / "roles").glob("*.json")}
    if roles != {Path(name).stem for name in actual}:
        raise RuntimeError("curated role maps do not exactly cover models")
    return CuratedBuild(root=root, model_sha256=actual, model_count=len(actual))


def _catalog_hash(catalog: CuratedCatalog) -> str:
    return hashlib.sha256(catalog.model_dump_json(exclude_none=False).encode("utf-8")).hexdigest()


def _build_id(
    version: str,
    catalog_hash: str,
    builder_hash: str,
    sources: dict[str, Path],
) -> str:
    payload = {
        "version": version,
        "catalog_sha256": catalog_hash,
        "builder_sha256": builder_hash,
        "variants": [str(variant) for variant in MetamorphicVariant],
        "sources": {source_id: sha256_file(path) for source_id, path in sorted(sources.items())},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_staging(staging: Path, build_id: str) -> None:
    identity_path = staging / "build.json"
    identity: object = None
    if identity_path.is_file():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            identity = None
    if staging.exists() and identity != {"build_id": build_id}:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _write_json(identity_path, {"build_id": build_id})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)
