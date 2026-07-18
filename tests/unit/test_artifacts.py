from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from meshprobe.artifacts import ArtifactCache, JsonValue

SOURCE_HASH = "a" * 64
IMPORTER_VERSION = "blender-5.2.0+meshprobe-worker-deadbeef"
PARAMETERS: dict[str, JsonValue] = {
    "format": "glb",
    "units": "millimeter",
    "merge_vertices": False,
}


def write_geometry(payload: Path, content: bytes = b"normalized geometry") -> None:
    (payload / "geometry").mkdir(parents=True)
    (payload / "geometry" / "scene.glb").write_bytes(content)


def test_cache_key_covers_source_importer_and_canonical_parameters(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    original = cache.key_for(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS)

    assert original == cache.key_for(
        SOURCE_HASH,
        IMPORTER_VERSION,
        {"merge_vertices": False, "units": "millimeter", "format": "glb"},
    )
    assert original != cache.key_for("b" * 64, IMPORTER_VERSION, PARAMETERS)
    assert original != cache.key_for(SOURCE_HASH, "other-importer", PARAMETERS)
    assert original != cache.key_for(SOURCE_HASH, IMPORTER_VERSION, {**PARAMETERS, "axis": "Z"})


def test_publish_is_verified_and_addressable_only_as_a_complete_entry(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)

    entry = cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, write_geometry)

    assert entry == cache.lookup(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS)
    assert entry.output_path("geometry/scene.glb").read_bytes() == b"normalized geometry"
    manifest = json.loads((entry.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == SOURCE_HASH
    assert manifest["importer_version"] == IMPORTER_VERSION
    assert manifest["normalization_parameters"] == PARAMETERS
    assert manifest["output_sha256"] == entry.output_sha256
    assert len(manifest["outputs"]) == 1


def test_failed_publication_leaves_no_visible_entry(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)

    def fail_after_write(payload: Path) -> None:
        write_geometry(payload)
        raise RuntimeError("normalizer crashed")

    with pytest.raises(RuntimeError, match="normalizer crashed"):
        cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, fail_after_write)

    assert cache.lookup(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS) is None
    assert not tuple((tmp_path / ".staging").iterdir())


def test_abandoned_staging_directory_is_invisible_and_cleanable(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    abandoned = tmp_path / ".staging" / "interrupted-process"
    write_geometry(abandoned / "payload")
    (abandoned / "manifest.json").write_text("{}", encoding="utf-8")

    assert cache.lookup(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS) is None
    assert cache.cleanup_staging() == 1
    assert not abandoned.exists()


def test_tampered_payload_is_rejected(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    entry = cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, write_geometry)
    entry.output_path("geometry/scene.glb").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="payload does not match manifest"):
        cache.lookup(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS)


def test_evict_removes_the_complete_entry(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    entry = cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, write_geometry)

    assert cache.evict(entry.key)
    assert not entry.path.exists()
    assert cache.lookup(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS) is None
    assert not cache.evict(entry.key)


def test_concurrent_publishers_converge_on_one_valid_entry(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = tuple(
            pool.map(
                lambda _: cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, write_geometry),
                range(16),
            )
        )

    assert len({entry.path for entry in entries}) == 1
    assert len({entry.output_sha256 for entry in entries}) == 1
    assert cache.lookup(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS) == entries[0]


@pytest.mark.parametrize(
    "parameters",
    [
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": object()},
    ],
)
def test_cache_key_rejects_non_json_or_non_finite_parameters(
    tmp_path: Path, parameters: dict[str, object]
) -> None:
    cache = ArtifactCache(tmp_path)
    with pytest.raises(ValueError, match="finite JSON"):
        cache.key_for(SOURCE_HASH, IMPORTER_VERSION, parameters)  # type: ignore[arg-type]


def test_publication_rejects_symlinks_and_empty_payloads(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)

    with pytest.raises(ValueError, match="no output files"):
        cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, lambda payload: None)

    target = tmp_path / "outside.glb"
    target.write_bytes(b"outside")

    def write_link(payload: Path) -> None:
        (payload / "scene.glb").symlink_to(target)

    with pytest.raises(ValueError, match="symbolic links"):
        cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, write_link)


def test_output_paths_must_be_declared_and_normalized(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    entry = cache.publish(SOURCE_HASH, IMPORTER_VERSION, PARAMETERS, write_geometry)

    with pytest.raises(KeyError, match="no output"):
        entry.output_path("other.glb")
    with pytest.raises(ValueError, match="normalized relative"):
        entry.output_path("../geometry/scene.glb")
