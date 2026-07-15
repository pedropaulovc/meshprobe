from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshprobe.sources import sha256_file, snapshot_source


def write_gltf(tmp_path: Path, document: object) -> Path:
    source = tmp_path / "scene.gltf"
    source.write_text(json.dumps(document), encoding="utf-8")
    return source


def test_embedded_gltf_keeps_single_file_content_hash(tmp_path: Path) -> None:
    source = write_gltf(
        tmp_path,
        {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": "data:application/octet-stream;base64,AA==", "byteLength": 1}],
            "images": [{"bufferView": 0, "mimeType": "image/png"}],
        },
    )

    snapshot = snapshot_source(source)

    assert len(snapshot.assets) == 1
    assert snapshot.sha256 == sha256_file(source)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "JSON object"),
        ({"buffers": {}}, "buffers must be an array"),
        ({"images": [{"uri": 42}]}, "uri must be a string"),
        ({"buffers": [{"uri": "https://example.com/mesh.bin"}]}, "local relative path"),
        ({"buffers": [{"uri": ""}]}, "uri is empty"),
    ],
)
def test_gltf_dependency_validation_is_explicit(
    tmp_path: Path, document: object, message: str
) -> None:
    source = write_gltf(tmp_path, document)
    with pytest.raises(ValueError, match=message):
        snapshot_source(source)


def test_snapshot_rejects_missing_external_asset(tmp_path: Path) -> None:
    source = write_gltf(tmp_path, {"buffers": [{"uri": "missing.bin"}]})
    with pytest.raises(FileNotFoundError):
        snapshot_source(source)
