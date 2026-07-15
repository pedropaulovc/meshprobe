from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from meshprobe.evals.curated import (
    CuratedAssemblyRecipe,
    CuratedCatalog,
    CuratedPlacement,
    CuratedSource,
    _download_atomic,
    _glb_payload,
    _triangles,
    ingest_curated_sources,
    load_catalog,
    topology_sha256,
)
from meshprobe.evals.curated_build import (
    CuratedBuild,
    _prepare_staging,
    build_curated_variants,
)
from meshprobe.evals.curated_tasks import (
    _prepare_staging as prepare_task_staging,
)
from meshprobe.evals.curated_tasks import (
    _read_checkpoint as read_task_checkpoint,
)
from meshprobe.evals.curated_tasks import (
    build_curated_corpus,
    curated_task_generator_sha256,
    generate_curated_episodes,
    resolve_curated_truth,
)
from meshprobe.evals.generators import (
    GeneratorFamily,
    MetamorphicVariant,
    build_model,
    publish_model,
)
from meshprobe.evals.schemas import Operation
from meshprobe.identity import stable_component_id
from meshprobe.models import SceneManifest
from meshprobe.sources import sha256_file


def source_record(path: Path, index: int) -> CuratedSource:
    commit = "a" * 40
    return CuratedSource(
        source_id=f"source-{index:02d}",
        name=f"Source {index}",
        download_url=f"https://assets.example/{commit}/source-{index}.glb",
        source_commit=commit,
        source_sha256=sha256_file(path),
        topology_sha256=topology_sha256(path),
        license_spdx="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="Public domain test fixture",
    )


def catalog_for(path: Path) -> CuratedCatalog:
    sources = tuple(source_record(path, index) for index in range(20))
    assemblies = tuple(
        CuratedAssemblyRecipe(
            assembly_id=f"assembly-{index:02d}",
            description="Three-source qualification fixture",
            placements=tuple(
                CuratedPlacement(
                    source_id=sources[(index + offset) % len(sources)].source_id,
                    role=f"component-{offset}",
                    position_mm=(float(offset * 100), 0, 0),
                )
                for offset in range(3)
            ),
        )
        for index in range(20)
    )
    return CuratedCatalog(
        catalog_version="test-v1",
        sources=sources,
        assemblies=assemblies,
    )


def test_topology_hash_is_invariant_to_rename_and_rigid_variants(tmp_path: Path) -> None:
    base = publish_model(
        build_model(GeneratorFamily.HIDDEN_CLIP, 0, variant=MetamorphicVariant.MATERIAL),
        tmp_path,
    )
    renamed = publish_model(
        build_model(GeneratorFamily.HIDDEN_CLIP, 0, variant=MetamorphicVariant.RENAME),
        tmp_path / "renamed",
    )
    rigid = publish_model(
        build_model(GeneratorFamily.HIDDEN_CLIP, 0, variant=MetamorphicVariant.RIGID_TRANSFORM),
        tmp_path / "rigid",
    )

    assert topology_sha256(base.path) == topology_sha256(renamed.path)
    assert topology_sha256(base.path) == topology_sha256(rigid.path)


def test_curated_ingestion_checkpoints_verified_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "model")
    catalog = catalog_for(model.path)

    def copy_download(url: str, destination: Path) -> None:
        del url
        shutil.copy2(model.path, destination)

    monkeypatch.setattr("meshprobe.evals.curated._download_atomic", copy_download)
    ingested = ingest_curated_sources(catalog, tmp_path / "cache", workers=4)

    assert len(ingested) == 20
    assert len(set(ingested.values())) == 1
    assert next(iter(ingested.values())).name == f"{sha256_file(model.path)}.glb"


def test_catalog_loader_rejects_unpinned_or_unlicensed_sources(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "model")
    catalog = catalog_for(model.path)
    path = tmp_path / "catalog.json"
    invalid_source = catalog.sources[0].model_copy(
        update={
            "download_url": "https://assets.example/main/source.glb",
            "license_spdx": "NOASSERTION",
        }
    )
    path.write_text(
        catalog.model_copy(
            update={"sources": (invalid_source, *catalog.sources[1:])}
        ).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported license"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"download_url": "http://assets.example/model.glb"}, "plain HTTPS"),
        (
            {"download_url": "https://assets.example/main/" + "a" * 40 + "/model.glb"},
            "not commit-pinned",
        ),
        ({"license_url": "file:///license"}, "license URL"),
    ],
)
def test_catalog_loader_enforces_each_source_policy(
    tmp_path: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "model")
    catalog = catalog_for(model.path)
    path = tmp_path / "catalog.json"
    invalid = catalog.sources[0].model_copy(update=updates)
    path.write_text(
        catalog.model_copy(update={"sources": (invalid, *catalog.sources[1:])}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_catalog(path)


def test_catalog_loader_rejects_duplicate_and_dangling_recipe_entries(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "model")
    catalog = catalog_for(model.path)
    path = tmp_path / "catalog.json"
    duplicate_source = catalog.model_copy(
        update={"sources": (catalog.sources[0], catalog.sources[0], *catalog.sources[2:])}
    )
    path.write_text(duplicate_source.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="source IDs"):
        load_catalog(path)

    duplicate_assembly = catalog.model_copy(
        update={
            "assemblies": (
                catalog.assemblies[0],
                catalog.assemblies[0],
                *catalog.assemblies[2:],
            )
        }
    )
    path.write_text(duplicate_assembly.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="assembly IDs"):
        load_catalog(path)

    dangling = catalog.assemblies[0].model_copy(
        update={
            "placements": (
                catalog.assemblies[0]
                .placements[0]
                .model_copy(update={"source_id": "missing-source"}),
                *catalog.assemblies[0].placements[1:],
            )
        }
    )
    path.write_text(
        catalog.model_copy(
            update={"assemblies": (dangling, *catalog.assemblies[1:])}
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown sources"):
        load_catalog(path)

    repeated_role = catalog.assemblies[0].model_copy(
        update={
            "placements": (
                catalog.assemblies[0].placements[0],
                catalog.assemblies[0].placements[1].model_copy(update={"role": "component-0"}),
                catalog.assemblies[0].placements[2],
            )
        }
    )
    path.write_text(
        catalog.model_copy(
            update={"assemblies": (repeated_role, *catalog.assemblies[1:])}
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="roles must be unique"):
        load_catalog(path)


def test_curated_ingestion_rejects_worker_and_content_mismatches(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "model")
    catalog = catalog_for(model.path)
    with pytest.raises(ValueError, match="workers"):
        ingest_curated_sources(catalog, tmp_path / "cache", workers=0)

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{catalog.sources[0].source_sha256}.glb").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        ingest_curated_sources(catalog, cache)

    inconsistent = catalog.sources[0].model_copy(update={"topology_sha256": "0" * 64})
    with pytest.raises(ValueError, match="different topology"):
        ingest_curated_sources(
            catalog.model_copy(update={"sources": (inconsistent, *catalog.sources[1:])}),
            tmp_path / "other-cache",
        )


def test_atomic_download_and_glb_container_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 200

        def __init__(self) -> None:
            self.chunks = [b"payload", b""]

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, size: int) -> bytes:
            del size
            return self.chunks.pop(0)

    monkeypatch.setattr(
        "meshprobe.evals.curated.urllib.request.urlopen", lambda *args, **kwargs: Response()
    )
    downloaded = tmp_path / "download.glb"
    _download_atomic("https://assets.example/model.glb", downloaded)
    assert downloaded.read_bytes() == b"payload"

    malformed = tmp_path / "malformed.glb"
    malformed.write_bytes(b"short")
    with pytest.raises(ValueError, match="not a GLB"):
        _glb_payload(malformed)
    json_chunk = b"{}  "
    payload = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(json_chunk))
    payload += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    malformed.write_bytes(payload)
    with pytest.raises(ValueError, match="JSON and binary"):
        _glb_payload(malformed)
    assert _triangles([0, 1, 2, 3], 4) == [(0, 1, 2)]
    assert _triangles([0, 1, 2, 3], 5) == [(0, 1, 2), (1, 2, 3)]
    assert _triangles([0, 1, 2, 3], 6) == [(0, 1, 2), (0, 2, 3)]
    assert _triangles([0, 1, 2], 1) == []


def test_curated_build_discards_staging_from_a_different_input(tmp_path: Path) -> None:
    staging = tmp_path / ".curated-v1.building"
    _prepare_staging(staging, "a" * 64)
    stale = staging / "public" / "models" / "stale.glb"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    _prepare_staging(staging, "b" * 64)

    assert not stale.exists()
    assert (staging / "build.json").read_text() == '{"build_id":"' + "b" * 64 + '"}\n'


def test_curated_tasks_resolve_private_roles_and_force_full_operation_use(
    tmp_path: Path,
    scene_manifest: SceneManifest,
) -> None:
    model = tmp_path / "curated-model.glb"
    model.write_bytes(b"fixture")
    truth = resolve_curated_truth(
        model,
        scene_manifest,
        {
            "model_file": model.name,
            "roles": {
                "target": ["idler"],
                "reference": ["assembly"],
                "occluder": ["retaining clip"],
            },
        },
    )

    episodes = generate_curated_episodes(truth)

    assert len(episodes) == 3
    assert set(episodes[-1].spec.required_operations) == set(Operation)
    assert episodes[-1].spec.model_source == "curated"
    assert episodes[-1].ground_truth.state_requirements[-1].predicate == "reset_to_imported"
    assert all(
        component_id not in episode.spec.prompt
        for episode in episodes
        for component_id in truth.component_ids.values()
    )
    for episode in episodes:
        value_schema = episode.spec.answer_schema["properties"]["values"]
        assert isinstance(value_schema, dict)
        properties = value_schema["properties"]
        assert isinstance(properties, dict)
        assert set(episode.ground_truth.answer.values) == set(properties)


def test_curated_variant_build_is_atomic_resumable_and_hash_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "source")
    catalog = catalog_for(source.path)
    sources = {item.source_id: source.path for item in catalog.sources}

    def fake_blender(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        jobs_path = Path(command[-2])
        checkpoint = Path(command[-1])
        payload = json.loads(jobs_path.read_text(encoding="utf-8"))
        completed = []
        for job in payload["jobs"]:
            output = Path(job["output"])
            metadata = Path(job["metadata"])
            output.write_bytes(f"model:{job['key']}".encode())
            metadata.write_text(
                json.dumps(
                    {
                        "model_file": output.name,
                        "roles": {"target": ["target"]},
                        "variant": job["variant"],
                    }
                ),
                encoding="utf-8",
            )
            completed.append(job["key"])
        checkpoint.write_text(
            json.dumps({"build_id": payload["build_id"], "completed": completed}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("meshprobe.evals.curated_build.shutil.which", lambda _: "/usr/bin/blender")
    monkeypatch.setattr("meshprobe.evals.curated_build.subprocess.run", fake_blender)

    build = build_curated_variants(catalog, sources, tmp_path / "builds")

    assert build.model_count == 160
    assert build_curated_variants(catalog, sources, tmp_path / "builds") == build
    manifest_path = build.root / "public" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["variants"] == [str(variant) for variant in MetamorphicVariant]
    manifest["variants"] = manifest["variants"][:-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="variant set"):
        build_curated_variants(catalog, sources, tmp_path / "builds")
    manifest["variants"] = [str(variant) for variant in MetamorphicVariant]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    model = next((build.root / "public" / "models").glob("*.glb"))
    model.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="do not match"):
        build_curated_variants(catalog, sources, tmp_path / "builds")


def test_curated_variant_build_surfaces_blender_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "source")
    catalog = catalog_for(source.path)
    sources = {item.source_id: source.path for item in catalog.sources}
    monkeypatch.setattr("meshprobe.evals.curated_build.shutil.which", lambda _: "/usr/bin/blender")
    monkeypatch.setattr(
        "meshprobe.evals.curated_build.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, "stdout", "stderr"),
    )

    with pytest.raises(RuntimeError, match="curated Blender build failed"):
        build_curated_variants(catalog, sources, tmp_path / "builds")


def test_curated_task_corpus_publishes_with_a_persistent_manifest_worker(
    tmp_path: Path,
    scene_manifest: SceneManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "variant-build"
    models = build_root / "public" / "models"
    roles = build_root / "private" / "roles"
    models.mkdir(parents=True)
    roles.mkdir(parents=True)
    model = models / "curated-model.glb"
    model.write_bytes(b"curated model")
    source_sha256 = sha256_file(model)
    manifest = _manifest_with_source(scene_manifest, source_sha256)
    (build_root / "public" / "manifest.json").write_text("{}\n", encoding="utf-8")
    (roles / "curated-model.json").write_text(
        json.dumps(
            {
                "model_file": model.name,
                "roles": {
                    "target": ["idler"],
                    "reference": ["assembly"],
                    "occluder": ["retaining clip"],
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeController:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def __enter__(self) -> FakeController:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def open_scene(self, path: Path) -> SceneManifest:
            assert path.name == model.name
            return manifest

    monkeypatch.setattr("meshprobe.evals.curated_tasks.BlenderController", FakeController)
    source_build = CuratedBuild(
        root=build_root,
        model_sha256={model.name: source_sha256},
        model_count=1,
    )

    corpus = build_curated_corpus(source_build, tmp_path / "corpora")

    assert corpus.model_count == 1
    assert corpus.episode_count == 3
    assert corpus.full_investigation_count == 1
    assert build_curated_corpus(source_build, tmp_path / "corpora") == corpus


def test_curated_task_role_and_checkpoint_validation_errors(
    tmp_path: Path,
    scene_manifest: SceneManifest,
) -> None:
    model = tmp_path / "model.glb"
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="malformed"):
        resolve_curated_truth(model, scene_manifest, [])
    with pytest.raises(ValueError, match="different model"):
        resolve_curated_truth(model, scene_manifest, {"model_file": "other.glb", "roles": {}})
    with pytest.raises(ValueError, match="no target"):
        resolve_curated_truth(model, scene_manifest, {"model_file": model.name, "roles": {}})
    with pytest.raises(ValueError, match="absent from manifest"):
        resolve_curated_truth(
            model,
            scene_manifest,
            {
                "model_file": model.name,
                "roles": {
                    "target": ["missing"],
                    "reference": ["assembly"],
                    "occluder": ["retaining clip"],
                },
            },
        )

    staging = tmp_path / ".tasks.building"
    staging.mkdir()
    (staging / "checkpoint.json").write_text("not-json", encoding="utf-8")
    stale = staging / "stale"
    stale.write_text("stale", encoding="utf-8")
    prepare_task_staging(staging, "a" * 64)
    assert not stale.exists()
    checkpoint = staging / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"generator_sha256": "b" * 64, "completed": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different input"):
        read_task_checkpoint(staging, "a" * 64)
    checkpoint.write_text(
        json.dumps({"generator_sha256": "a" * 64, "completed": "invalid"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid completed-model"):
        read_task_checkpoint(staging, "a" * 64)


def test_curated_task_hash_includes_blender_importer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_root = tmp_path / "build"
    (build_root / "public").mkdir(parents=True)
    (build_root / "public" / "manifest.json").write_text("{}\n", encoding="utf-8")
    build = CuratedBuild(root=build_root, model_sha256={}, model_count=0)
    hashed: list[Path] = []

    def record(path: Path) -> str:
        hashed.append(path)
        return "a" * 64

    monkeypatch.setattr("meshprobe.evals.curated_tasks.sha256_file", record)

    curated_task_generator_sha256(build)

    assert any(path.name == "worker.py" for path in hashed)


def test_curated_task_hash_is_path_independent_and_includes_private_roles(
    tmp_path: Path,
) -> None:
    builds: list[CuratedBuild] = []
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "public").mkdir(parents=True)
        (root / "private" / "roles").mkdir(parents=True)
        (root / "public" / "manifest.json").write_text("{}\n", encoding="utf-8")
        (root / "private" / "roles" / "model.json").write_text(
            '{"target":"idler"}\n', encoding="utf-8"
        )
        builds.append(CuratedBuild(root=root, model_sha256={}, model_count=0))

    first_hash = curated_task_generator_sha256(builds[0])
    assert curated_task_generator_sha256(builds[1]) == first_hash

    (builds[1].root / "private" / "roles" / "model.json").write_text(
        '{"target":"shaft"}\n', encoding="utf-8"
    )
    assert curated_task_generator_sha256(builds[1]) != first_hash


def _manifest_with_source(manifest: SceneManifest, source_sha256: str) -> SceneManifest:
    payload = manifest.model_dump(mode="json")
    old_to_new = {
        component["id"]: stable_component_id(source_sha256, component["path"])
        for component in payload["components"]
    }
    payload["source_sha256"] = source_sha256
    for component in payload["components"]:
        component["id"] = old_to_new[component["id"]]
        if component["parent_id"] is not None:
            component["parent_id"] = old_to_new[component["parent_id"]]
        component["child_ids"] = [old_to_new[child_id] for child_id in component["child_ids"]]
    return SceneManifest.model_validate(payload)
