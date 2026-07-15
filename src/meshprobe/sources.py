"""Immutable source-asset discovery and content identity."""

from __future__ import annotations

import hashlib
import json
import shlex
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
    """Discover every imported sidecar and fingerprint its content and metadata."""

    source = source_path.expanduser().resolve(strict=True)
    dependencies: tuple[tuple[str, Path], ...] = ()
    if source.suffix.lower() == ".gltf":
        dependencies = _gltf_dependencies(source)
    if source.suffix.lower() == ".obj":
        dependencies = _obj_dependencies(source)
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


def _obj_dependencies(source: Path) -> tuple[tuple[str, Path], ...]:
    material_paths: list[tuple[str, Path]] = []
    for line_number, tokens in _tokenized_lines(source):
        if not tokens or tokens[0].lower() != "mtllib":
            continue
        if len(tokens) == 1:
            raise ValueError(f"OBJ line {line_number} mtllib has no path")
        for reference in tokens[1:]:
            material_paths.append(
                (f"materials/{line_number}/{reference}", _local_reference(source, reference))
            )

    dependencies = list(material_paths)
    for material_index, (_, material_path) in enumerate(material_paths):
        for line_number, tokens in _tokenized_lines(material_path):
            if not tokens or tokens[0].lower() not in _MTL_TEXTURE_DIRECTIVES:
                continue
            reference = _mtl_texture_reference(tokens[1:], material_path, line_number)
            dependencies.append(
                (
                    f"textures/{material_index}/{line_number}/{reference}",
                    _local_reference(material_path, reference),
                )
            )

    unique: dict[Path, tuple[str, Path]] = {}
    for logical_path, path in dependencies:
        unique.setdefault(path, (logical_path, path))
    return tuple(unique.values())


_MTL_TEXTURE_DIRECTIVES = {
    "bump",
    "decal",
    "disp",
    "map_bump",
    "map_d",
    "map_ka",
    "map_kd",
    "map_ke",
    "map_ks",
    "map_ns",
    "norm",
    "refl",
}

_MTL_FIXED_OPTION_ARITY = {
    "-blendu": 1,
    "-blendv": 1,
    "-boost": 1,
    "-bm": 1,
    "-cc": 1,
    "-clamp": 1,
    "-imfchan": 1,
    "-mm": 2,
    "-texres": 1,
    "-type": 1,
}


def _tokenized_lines(path: Path) -> tuple[tuple[int, list[str]], ...]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"source sidecar is not UTF-8 text: {path}") from error
    parsed: list[tuple[int, list[str]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as error:
            raise ValueError(f"invalid quoted path at {path}:{line_number}: {error}") from error
        parsed.append((line_number, tokens))
    return tuple(parsed)


def _mtl_texture_reference(tokens: list[str], path: Path, line_number: int) -> str:
    index = 0
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index].lower()
        index += 1
        if option in {"-o", "-s", "-t"}:
            consumed = 0
            while index < len(tokens) and consumed < 3:
                try:
                    float(tokens[index])
                except ValueError:
                    break
                index += 1
                consumed += 1
            if consumed == 0:
                raise ValueError(
                    f"MTL option {option} has no numeric value at {path}:{line_number}"
                )
            continue
        arity = _MTL_FIXED_OPTION_ARITY.get(option)
        if arity is None:
            raise ValueError(f"unsupported MTL texture option {option} at {path}:{line_number}")
        if index + arity > len(tokens):
            raise ValueError(f"MTL option {option} is incomplete at {path}:{line_number}")
        index += arity
    reference = " ".join(tokens[index:])
    if not reference:
        raise ValueError(f"MTL texture directive has no path at {path}:{line_number}")
    return reference


def _local_reference(source: Path, reference: str) -> Path:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"external asset URI is not a local relative path: {reference}")
    decoded_path = unquote(parsed.path)
    if not decoded_path or Path(decoded_path).is_absolute():
        raise ValueError(f"external asset URI is not a local relative path: {reference}")
    return (source.parent / decoded_path).resolve(strict=True)
