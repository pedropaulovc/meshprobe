"""Deterministic, atomic storage for derived MeshProbe artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from meshprobe.sources import sha256_file

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

_MANIFEST_NAME = "manifest.json"
_PAYLOAD_DIRECTORY = "payload"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactOutput:
    """One immutable file in a cache entry."""

    relative_path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ArtifactCacheEntry:
    """A validated, completely published cache entry."""

    key: str
    source_sha256: str
    importer_version: str
    normalization_parameters: dict[str, JsonValue]
    output_sha256: str
    outputs: tuple[ArtifactOutput, ...]
    path: Path

    def output_path(self, relative_path: str) -> Path:
        """Return a declared output path and reject undeclared names."""

        declared = {output.relative_path for output in self.outputs}
        normalized = _normalized_relative_path(relative_path)
        if normalized not in declared:
            raise KeyError(f"cache entry has no output named {normalized!r}")
        return self.path / _PAYLOAD_DIRECTORY / Path(*PurePosixPath(normalized).parts)


class ArtifactCache:
    """Content-addressed cache whose incomplete writes are never visible."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self._entries = self.root / "entries"
        self._staging = self.root / ".staging"
        self._entries.mkdir(parents=True, exist_ok=True)
        self._staging.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(
        source_sha256: str,
        importer_version: str,
        normalization_parameters: Mapping[str, JsonValue],
    ) -> str:
        """Derive an entry key from every input that can change normalization."""

        normalized = _normalized_parameters(normalization_parameters)
        _validate_sha256(source_sha256, "source_sha256")
        _validate_importer_version(importer_version)
        material = {
            "schema_version": _SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "importer_version": importer_version,
            "normalization_parameters": normalized,
        }
        return hashlib.sha256(_canonical_json(material)).hexdigest()

    def lookup(
        self,
        source_sha256: str,
        importer_version: str,
        normalization_parameters: Mapping[str, JsonValue],
    ) -> ArtifactCacheEntry | None:
        """Return a verified entry or ``None``; staging data is never consulted."""

        key = self.key_for(source_sha256, importer_version, normalization_parameters)
        entry_path = self._entry_path(key)
        if not entry_path.exists():
            return None
        entry = self._validate_entry(entry_path)
        if entry.key != key:
            raise ValueError(f"cache manifest key mismatch at {entry_path}")
        return entry

    def publish(
        self,
        source_sha256: str,
        importer_version: str,
        normalization_parameters: Mapping[str, JsonValue],
        write_outputs: Callable[[Path], None],
    ) -> ArtifactCacheEntry:
        """Create and atomically publish one cache entry.

        ``write_outputs`` receives an empty payload directory on the cache volume.
        It may create files and directories but not symbolic links. If another
        process publishes the same key first, its validated entry wins.
        """

        parameters = _normalized_parameters(normalization_parameters)
        key = self.key_for(source_sha256, importer_version, parameters)
        existing = self.lookup(source_sha256, importer_version, parameters)
        if existing is not None:
            return existing

        staging_path = self._staging / f"{key}-{uuid.uuid4().hex}"
        payload_path = staging_path / _PAYLOAD_DIRECTORY
        payload_path.mkdir(parents=True)
        try:
            write_outputs(payload_path)
            outputs = _inventory_payload(payload_path)
            if not outputs:
                raise ValueError("cache publication produced no output files")
            output_sha256 = _outputs_digest(outputs)
            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "key": key,
                "source_sha256": source_sha256,
                "importer_version": importer_version,
                "normalization_parameters": parameters,
                "output_sha256": output_sha256,
                "outputs": [
                    {
                        "relative_path": output.relative_path,
                        "sha256": output.sha256,
                        "bytes": output.bytes,
                    }
                    for output in outputs
                ],
            }
            (staging_path / _MANIFEST_NAME).write_bytes(_canonical_json(manifest) + b"\n")
            self._validate_entry(staging_path)

            destination = self._entry_path(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(staging_path, destination)
            except OSError as error:
                if not destination.exists():
                    raise
                winner = self._validate_entry(destination)
                if winner.key != key:
                    raise ValueError(
                        f"concurrent cache publication produced wrong key {winner.key}"
                    ) from error
            return self._validate_entry(destination)
        finally:
            shutil.rmtree(staging_path, ignore_errors=True)

    def evict(self, key: str) -> bool:
        """Remove one complete entry and return whether it existed."""

        _validate_sha256(key, "cache key")
        entry_path = self._entry_path(key)
        if not entry_path.exists():
            return False
        shutil.rmtree(entry_path)
        return True

    def cleanup_staging(self) -> int:
        """Remove abandoned unpublished directories and return their count."""

        removed = 0
        for candidate in self._staging.iterdir():
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            removed += 1
        return removed

    def _entry_path(self, key: str) -> Path:
        _validate_sha256(key, "cache key")
        return self._entries / key[:2] / key

    def _validate_entry(self, entry_path: Path) -> ArtifactCacheEntry:
        manifest_path = entry_path / _MANIFEST_NAME
        payload_path = entry_path / _PAYLOAD_DIRECTORY
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid cache manifest at {manifest_path}: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"cache manifest must be an object at {manifest_path}")
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported cache manifest schema at {manifest_path}")

        key = _required_string(raw, "key")
        source_sha256 = _required_string(raw, "source_sha256")
        importer_version = _required_string(raw, "importer_version")
        output_sha256 = _required_string(raw, "output_sha256")
        _validate_sha256(key, "cache manifest key")
        _validate_sha256(source_sha256, "cache manifest source_sha256")
        _validate_sha256(output_sha256, "cache manifest output_sha256")
        _validate_importer_version(importer_version)

        parameters_raw = raw.get("normalization_parameters")
        if not isinstance(parameters_raw, dict):
            raise ValueError("cache manifest normalization_parameters must be an object")
        parameters = _normalized_parameters(cast(dict[str, JsonValue], parameters_raw))

        outputs_raw = raw.get("outputs")
        if not isinstance(outputs_raw, list) or not outputs_raw:
            raise ValueError("cache manifest outputs must be a non-empty array")
        outputs: list[ArtifactOutput] = []
        for item in outputs_raw:
            if not isinstance(item, dict):
                raise ValueError("cache manifest output must be an object")
            relative_path = _normalized_relative_path(_required_string(item, "relative_path"))
            sha256 = _required_string(item, "sha256")
            size = item.get("bytes")
            _validate_sha256(sha256, "cache output sha256")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError("cache output bytes must be a non-negative integer")
            outputs.append(ArtifactOutput(relative_path, sha256, size))
        if len({output.relative_path for output in outputs}) != len(outputs):
            raise ValueError("cache manifest output paths must be unique")
        declared = tuple(sorted(outputs, key=lambda output: output.relative_path))
        actual = _inventory_payload(payload_path)
        if actual != declared:
            raise ValueError(f"cache payload does not match manifest at {entry_path}")
        if _outputs_digest(declared) != output_sha256:
            raise ValueError(f"cache output digest mismatch at {entry_path}")
        expected_key = self.key_for(source_sha256, importer_version, parameters)
        if expected_key != key:
            raise ValueError(f"cache input digest mismatch at {entry_path}")

        return ArtifactCacheEntry(
            key=key,
            source_sha256=source_sha256,
            importer_version=importer_version,
            normalization_parameters=parameters,
            output_sha256=output_sha256,
            outputs=declared,
            path=entry_path,
        )


def _inventory_payload(payload_path: Path) -> tuple[ArtifactOutput, ...]:
    if not payload_path.is_dir() or payload_path.is_symlink():
        raise ValueError("cache payload must be a real directory")
    outputs: list[ArtifactOutput] = []
    for path in sorted(payload_path.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"cache payload cannot contain symbolic links: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"cache payload contains unsupported entry: {path}")
        relative_path = path.relative_to(payload_path).as_posix()
        outputs.append(
            ArtifactOutput(
                relative_path=_normalized_relative_path(relative_path),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
        )
    return tuple(outputs)


def _outputs_digest(outputs: tuple[ArtifactOutput, ...]) -> str:
    records = [
        {
            "relative_path": output.relative_path,
            "sha256": output.sha256,
            "bytes": output.bytes,
        }
        for output in outputs
    ]
    return hashlib.sha256(_canonical_json(records)).hexdigest()


def _normalized_parameters(parameters: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    if any(not isinstance(key, str) for key in parameters):
        raise ValueError("normalization parameter keys must be strings")
    try:
        encoded = _canonical_json(dict(parameters))
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"normalization parameters must be finite JSON values: {error}") from error
    if not isinstance(normalized, dict):
        raise ValueError("normalization parameters must be an object")
    return cast(dict[str, JsonValue], normalized)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _normalized_relative_path(relative_path: str) -> str:
    path = PurePosixPath(relative_path.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"cache output path must be a normalized relative path: {relative_path!r}")
    return path.as_posix()


def _required_string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"cache manifest {key} must be a non-empty string")
    return item


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _validate_importer_version(importer_version: str) -> None:
    if not importer_version or len(importer_version) > 256:
        raise ValueError("importer_version must contain between 1 and 256 characters")
