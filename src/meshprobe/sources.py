"""Immutable source-asset discovery and content identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class SourceAsset:
    logical_path: str
    path: Path
    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class SourceSnapshot:
    assets: tuple[SourceAsset, ...]
    sha256: str


def snapshot_source(source_path: Path) -> SourceSnapshot:
    """Discover every imported glTF asset and fingerprint its content and metadata."""

    source = source_path.expanduser().resolve(strict=True)
    dependencies = _gltf_dependencies(source) if source.suffix.lower() == ".gltf" else ()
    logical_assets = ((".", source), *dependencies)
    assets = tuple(_snapshot_asset(logical_path, path) for logical_path, path in logical_assets)
    if len(assets) == 1:
        return SourceSnapshot(assets=assets, sha256=assets[0].sha256)

    digest = hashlib.sha256(b"meshprobe-source-bundle-v1\0")
    for asset in assets:
        digest.update(asset.logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(asset.sha256))
    return SourceSnapshot(assets=assets, sha256=digest.hexdigest())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_asset(logical_path: str, path: Path) -> SourceAsset:
    if not path.is_file():
        raise ValueError(f"source asset is not a file: {logical_path}")
    stat = path.stat()
    return SourceAsset(
        logical_path=logical_path,
        path=path,
        sha256=sha256_file(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _gltf_dependencies(source: Path) -> tuple[tuple[str, Path], ...]:
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid glTF JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("glTF document must be a JSON object")

    dependencies: list[tuple[str, Path]] = []
    for collection_name in ("buffers", "images"):
        collection = document.get(collection_name, [])
        if not isinstance(collection, list):
            raise ValueError(f"glTF {collection_name} must be an array")
        for index, entry in enumerate(collection):
            if not isinstance(entry, dict) or "uri" not in entry:
                continue
            uri = entry["uri"]
            if not isinstance(uri, str):
                raise ValueError(f"glTF {collection_name}[{index}].uri must be a string")
            if uri.startswith("data:"):
                continue
            parsed = urlsplit(uri)
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError(f"external glTF URI is not a local relative path: {uri}")
            decoded_path = unquote(parsed.path)
            if not decoded_path:
                raise ValueError(f"glTF {collection_name}[{index}].uri is empty")
            dependency = (source.parent / decoded_path).resolve(strict=True)
            logical_path = f"{collection_name}/{index}/{uri}"
            dependencies.append((logical_path, dependency))
    return tuple(dependencies)
