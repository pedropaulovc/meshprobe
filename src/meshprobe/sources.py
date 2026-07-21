"""Immutable source-asset discovery and content identity."""

from __future__ import annotations

import hashlib
import json
import shlex
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
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


class _InvalidGlb(ValueError):
    """The binary container cannot be inspected for external dependencies."""


def snapshot_source(
    source_path: Path,
    *,
    check_deadline: Callable[[], object] | None = None,
) -> SourceSnapshot:
    """Discover every imported sidecar and fingerprint its content and metadata."""

    _check_deadline(check_deadline)
    source = source_path.expanduser().resolve(strict=True)
    dependencies: tuple[tuple[str, Path], ...] = ()
    if source.suffix.lower() in {".glb", ".gltf"}:
        dependencies = _gltf_dependencies(source, check_deadline)
    if source.suffix.lower() == ".obj":
        dependencies = _obj_dependencies(source, check_deadline)
    logical_assets = ((".", source), *dependencies)
    assets = tuple(
        _snapshot_asset(logical_path, path, check_deadline) for logical_path, path in logical_assets
    )
    if len(assets) == 1:
        return SourceSnapshot(assets=assets, sha256=assets[0].sha256)

    digest = hashlib.sha256(b"meshprobe-source-bundle-v1\0")
    for asset in assets:
        _check_deadline(check_deadline)
        digest.update(asset.logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(asset.sha256))
    return SourceSnapshot(assets=assets, sha256=digest.hexdigest())


def sha256_file(
    path: Path,
    *,
    check_deadline: Callable[[], object] | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            _check_deadline(check_deadline)
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    _check_deadline(check_deadline)
    return digest.hexdigest()


def _snapshot_asset(
    logical_path: str,
    path: Path,
    check_deadline: Callable[[], object] | None,
) -> SourceAsset:
    _check_deadline(check_deadline)
    if not path.is_file():
        raise ValueError(f"source asset is not a file: {logical_path}")
    stat = path.stat()
    return SourceAsset(
        logical_path=logical_path,
        path=path,
        sha256=sha256_file(path, check_deadline=check_deadline),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _check_deadline(check_deadline: Callable[[], object] | None) -> None:
    if check_deadline is not None:
        check_deadline()


def _gltf_dependencies(
    source: Path,
    check_deadline: Callable[[], object] | None,
) -> tuple[tuple[str, Path], ...]:
    try:
        document = _gltf_document(source, check_deadline)
    except _InvalidGlb:
        return ()

    dependencies: list[tuple[str, Path]] = []
    for collection_name in ("buffers", "images"):
        _check_deadline(check_deadline)
        collection = document.get(collection_name, [])
        if not isinstance(collection, list):
            raise ValueError(f"glTF {collection_name} must be an array")
        for index, entry in enumerate(collection):
            _check_deadline(check_deadline)
            if not isinstance(entry, dict) or "uri" not in entry:
                continue
            uri = entry["uri"]
            if not isinstance(uri, str):
                raise ValueError(f"glTF {collection_name}[{index}].uri must be a string")
            if uri.startswith("data:"):
                continue
            if not uri:
                raise ValueError(f"glTF {collection_name}[{index}].uri is empty")
            dependency = _local_reference(source, uri)
            logical_path = f"{collection_name}/{index}/{uri}"
            dependencies.append((logical_path, dependency))
    return tuple(dependencies)


def _gltf_document(
    source: Path,
    check_deadline: Callable[[], object] | None,
) -> dict[str, object]:
    if source.suffix.lower() == ".glb":
        return _glb_document(source, check_deadline)
    try:
        with source.open(encoding="utf-8") as stream:
            _check_deadline(check_deadline)
            document = json.load(stream)
            _check_deadline(check_deadline)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid glTF JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("glTF document must be a JSON object")
    return document


def _glb_document(
    source: Path,
    check_deadline: Callable[[], object] | None,
) -> dict[str, object]:
    with source.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"glTF":
            raise _InvalidGlb("invalid GLB header")
        version, declared_length = struct.unpack_from("<II", header, 4)
        if version != 2 or declared_length != source.stat().st_size:
            raise _InvalidGlb("invalid GLB version or declared length")
        offset = 12
        while offset + 8 <= declared_length:
            _check_deadline(check_deadline)
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise _InvalidGlb("GLB chunk header is truncated")
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            offset += 8
            chunk_end = offset + chunk_length
            if chunk_end > declared_length:
                raise _InvalidGlb("GLB chunk exceeds the declared file length")
            if chunk_type != 0x4E4F534A:
                stream.seek(chunk_length, 1)
                offset = chunk_end
                continue
            chunk = stream.read(chunk_length)
            offset = chunk_end
            _check_deadline(check_deadline)
            try:
                document = json.loads(chunk.rstrip(b"\x00 \t\r\n").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise _InvalidGlb(f"invalid GLB JSON: {error}") from error
            if not isinstance(document, dict):
                raise _InvalidGlb("GLB JSON document must be an object")
            return document
    raise _InvalidGlb("GLB has no JSON chunk")


def _obj_dependencies(
    source: Path,
    check_deadline: Callable[[], object] | None,
) -> tuple[tuple[str, Path], ...]:
    material_paths: list[tuple[str, Path]] = []
    for line_number, tokens in _tokenized_lines(source, check_deadline):
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
        for line_number, tokens in _tokenized_lines(material_path, check_deadline):
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


def _tokenized_lines(
    path: Path,
    check_deadline: Callable[[], object] | None,
) -> Iterator[tuple[int, list[str]]]:
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                _check_deadline(check_deadline)
                try:
                    tokens = shlex.split(line, comments=True, posix=True)
                except ValueError as error:
                    raise ValueError(
                        f"invalid quoted path at {path}:{line_number}: {error}"
                    ) from error
                yield line_number, tokens
    except UnicodeDecodeError as error:
        raise ValueError(f"source sidecar is not UTF-8 text: {path}") from error


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
    if not decoded_path or any(
        path_type(decoded_path).is_absolute()
        for path_type in (Path, PurePosixPath, PureWindowsPath)
    ):
        raise ValueError(f"external asset URI is not a local relative path: {reference}")
    bundle_root = source.parent.resolve()
    resolved = (bundle_root / decoded_path).resolve(strict=True)
    if not resolved.is_relative_to(bundle_root):
        raise ValueError(f"external asset URI escapes the source bundle: {reference}")
    return resolved
