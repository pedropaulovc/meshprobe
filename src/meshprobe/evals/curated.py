"""Licensed curated-asset ingestion and split-safe topology fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from meshprobe.evals.schemas import Sha256
from meshprobe.sources import sha256_file

_ALLOWED_LICENSES = {"Apache-2.0", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0", "MIT"}
_COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class CuratedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CuratedSource(CuratedModel):
    source_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]+$")]
    name: str
    download_url: str
    source_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    source_sha256: Sha256
    topology_sha256: Sha256
    license_spdx: str
    license_url: str
    attribution: str


class CuratedPlacement(CuratedModel):
    source_id: str
    role: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]+$")]
    position_mm: tuple[float, float, float]


class CuratedAssemblyRecipe(CuratedModel):
    assembly_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]+$")]
    description: str
    placements: tuple[CuratedPlacement, ...] = Field(min_length=3)


class CuratedCatalog(CuratedModel):
    schema_version: int = 1
    catalog_version: str
    sources: tuple[CuratedSource, ...] = Field(min_length=20)
    assemblies: tuple[CuratedAssemblyRecipe, ...] = Field(min_length=20)


def load_catalog(path: Path) -> CuratedCatalog:
    catalog = CuratedCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    source_ids = {source.source_id for source in catalog.sources}
    if len(source_ids) != len(catalog.sources):
        raise ValueError("curated source IDs must be unique")
    assembly_ids = {assembly.assembly_id for assembly in catalog.assemblies}
    if len(assembly_ids) != len(catalog.assemblies):
        raise ValueError("curated assembly IDs must be unique")
    for source in catalog.sources:
        _validate_source_policy(source)
    for assembly in catalog.assemblies:
        placement_ids = [placement.source_id for placement in assembly.placements]
        unknown = set(placement_ids) - source_ids
        if unknown:
            raise ValueError(
                f"curated assembly {assembly.assembly_id} has unknown sources: {sorted(unknown)}"
            )
        if len(set(placement.role for placement in assembly.placements)) != len(
            assembly.placements
        ):
            raise ValueError(f"curated assembly {assembly.assembly_id} roles must be unique")
    return catalog


def ingest_curated_sources(
    catalog: CuratedCatalog,
    cache_root: Path,
    *,
    workers: int = 8,
) -> dict[str, Path]:
    """Download, hash, and topology-check curated GLBs into a content cache."""

    if workers < 1 or workers > 32:
        raise ValueError("curated download workers must be between 1 and 32")
    cache = cache_root.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    def ingest(source: CuratedSource) -> tuple[str, Path]:
        destination = cache / f"{source.source_sha256}.glb"
        if not destination.exists():
            _download_atomic(source.download_url, destination)
        if sha256_file(destination) != source.source_sha256:
            raise RuntimeError(f"curated source hash mismatch: {source.source_id}")
        if topology_sha256(destination) != source.topology_sha256:
            raise RuntimeError(f"curated topology hash mismatch: {source.source_id}")
        return source.source_id, destination

    unique_by_hash: dict[str, CuratedSource] = {}
    for source in catalog.sources:
        representative = unique_by_hash.setdefault(source.source_sha256, source)
        if source.topology_sha256 != representative.topology_sha256:
            raise ValueError("identical curated bytes declare different topology hashes")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="curated-download") as pool:
        cached_by_id = dict(pool.map(ingest, unique_by_hash.values()))
    cached_by_hash = {
        source.source_sha256: cached_by_id[source.source_id] for source in unique_by_hash.values()
    }
    return {source.source_id: cached_by_hash[source.source_sha256] for source in catalog.sources}


def topology_sha256(path: Path) -> str:
    """Fingerprint mesh connectivity and scale-invariant triangle proportions."""

    document, binary = _glb_payload(path)
    meshes = document.get("meshes", [])
    if not isinstance(meshes, list) or not meshes:
        raise ValueError("curated GLB contains no meshes")
    primitive_hashes: list[str] = []
    for mesh in meshes:
        if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list):
            raise ValueError("curated GLB mesh primitives are malformed")
        for primitive in mesh["primitives"]:
            primitive_hashes.append(_primitive_fingerprint(document, binary, primitive))
    nodes = document.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("curated GLB nodes must be an array")
    mesh_instances = sorted(
        int(node["mesh"])
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("mesh"), int)
    )
    child_degrees = sorted(
        len(node.get("children", []))
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("children", []), list)
    )
    payload = {
        "primitive_hashes": sorted(primitive_hashes),
        "mesh_instances": mesh_instances,
        "child_degrees": child_degrees,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_source_policy(source: CuratedSource) -> None:
    if source.license_spdx not in _ALLOWED_LICENSES:
        raise ValueError(
            f"curated source {source.source_id} uses unsupported license {source.license_spdx}"
        )
    parsed = urlsplit(source.download_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"curated source {source.source_id} must use a plain HTTPS URL")
    if source.source_commit not in parsed.path or "/main/" in parsed.path:
        raise ValueError(f"curated source {source.source_id} URL is not commit-pinned")
    license_url = urlsplit(source.license_url)
    if license_url.scheme != "https" or not license_url.netloc:
        raise ValueError(f"curated source {source.source_id} license URL must use HTTPS")


def _download_atomic(url: str, destination: Path) -> None:
    pending = destination.with_name(f".{destination.name}.pending-{os.getpid()}")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "meshprobe/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response, pending.open("wb") as out:
                if response.status != 200:
                    raise RuntimeError(f"curated download returned HTTP {response.status}: {url}")
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            os.replace(pending, destination)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            pending.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(0.2 * (2**attempt))
    raise RuntimeError(f"curated download failed after 3 attempts: {url}: {last_error}")


def _glb_payload(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise ValueError("curated source is not a GLB container")
    version, declared_length = struct.unpack_from("<II", payload, 4)
    if version != 2 or declared_length != len(payload):
        raise ValueError("curated GLB header is invalid")
    document: dict[str, object] | None = None
    binary = b""
    offset = 12
    while offset + 8 <= len(payload):
        length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + length]
        offset += length
        if len(chunk) != length:
            raise ValueError("curated GLB chunk is truncated")
        if chunk_type == 0x4E4F534A:
            decoded = json.loads(chunk.rstrip(b"\x00 \t\r\n"))
            if not isinstance(decoded, dict):
                raise ValueError("curated GLB JSON chunk must be an object")
            document = decoded
        if chunk_type == 0x004E4942:
            binary = chunk
    if document is None or not binary:
        raise ValueError("curated GLB requires JSON and binary chunks")
    return document, binary


def _primitive_fingerprint(document: dict[str, object], binary: bytes, primitive: object) -> str:
    if not isinstance(primitive, dict):
        raise ValueError("curated GLB primitive must be an object")
    attributes = primitive.get("attributes")
    if not isinstance(attributes, dict) or not isinstance(attributes.get("POSITION"), int):
        raise ValueError("curated GLB primitive has no POSITION accessor")
    positions = _accessor_values(document, binary, attributes["POSITION"])
    if not positions or any(len(position) < 3 for position in positions):
        raise ValueError("curated GLB POSITION accessor is empty or malformed")
    indices = list(range(len(positions)))
    if isinstance(primitive.get("indices"), int):
        raw_indices = _accessor_values(document, binary, primitive["indices"])
        indices = [int(value[0]) for value in raw_indices]
    mode = int(primitive.get("mode", 4))
    triangles = _triangles(indices, mode)
    edge_lengths: list[tuple[float, float, float]] = []
    for triangle in triangles:
        points = [positions[index] for index in triangle]
        lengths = sorted(
            math.dist(points[left][:3], points[right][:3])
            for left, right in ((0, 1), (1, 2), (2, 0))
        )
        edge_lengths.append((lengths[0], lengths[1], lengths[2]))
    nonzero = sorted(length for lengths in edge_lengths for length in lengths if length > 0)
    scale = nonzero[len(nonzero) // 2] if nonzero else 1.0
    normalized = sorted(
        tuple(round(length / scale, 6) for length in lengths) for lengths in edge_lengths
    )
    connectivity = sorted(tuple(sorted(triangle)) for triangle in triangles)
    payload = {
        "mode": mode,
        "vertices": len(positions),
        "indices": connectivity,
        "edge_ratios": normalized,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _accessor_values(
    document: dict[str, object], binary: bytes, accessor_index: int
) -> list[tuple[float | int, ...]]:
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise ValueError("curated GLB accessors or bufferViews are malformed")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict) or not isinstance(accessor.get("bufferView"), int):
        raise ValueError("curated GLB sparse or missing bufferView accessors are unsupported")
    view = views[accessor["bufferView"]]
    if not isinstance(view, dict):
        raise ValueError("curated GLB bufferView must be an object")
    component_type = accessor.get("componentType")
    value_type = accessor.get("type")
    count = accessor.get("count")
    if component_type not in _COMPONENT_FORMATS or value_type not in _TYPE_COMPONENTS:
        raise ValueError("curated GLB accessor component or value type is unsupported")
    if not isinstance(count, int) or count < 1:
        raise ValueError("curated GLB accessor count must be positive")
    code, width = _COMPONENT_FORMATS[component_type]
    components = _TYPE_COMPONENTS[value_type]
    packed_width = width * components
    stride = int(view.get("byteStride", packed_width))
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    unpacker = struct.Struct("<" + code * components)
    values = []
    for index in range(count):
        start = offset + index * stride
        end = start + packed_width
        if end > len(binary):
            raise ValueError("curated GLB accessor exceeds its binary buffer")
        values.append(unpacker.unpack(binary[start:end]))
    return values


def _triangles(indices: list[int], mode: int) -> list[tuple[int, int, int]]:
    if mode == 4:
        return [
            (indices[index], indices[index + 1], indices[index + 2])
            for index in range(0, len(indices) - 2, 3)
        ]
    if mode == 5:
        return [
            (indices[index], indices[index + 1], indices[index + 2])
            for index in range(len(indices) - 2)
        ]
    if mode == 6:
        return [
            (indices[0], indices[index], indices[index + 1]) for index in range(1, len(indices) - 1)
        ]
    return []
