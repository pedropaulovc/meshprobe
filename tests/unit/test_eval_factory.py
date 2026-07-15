from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshprobe.evals.factory import (
    _prepare_staging,
    _read_checkpoint,
    _recipe_sha256,
    _validate_operation_class_coverage,
    build_corpus,
    generator_source_sha256,
    merge_corpora,
    validate_corpus,
)
from meshprobe.evals.generators import (
    GeneratorFamily,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.leakage import (
    LeakageError,
    scan_error_message,
    validate_public_directory,
    validate_public_episode,
)
from meshprobe.evals.schemas import ModelSource


def test_factory_atomically_publishes_checkpointed_public_and_private_trees(
    tmp_path: Path,
) -> None:
    build = build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP, GeneratorFamily.STAMPED_ARROW),
        seeds=(0, 1),
    )

    assert build.model_count == 4
    assert build.episode_count == 16
    assert build.full_investigation_count == 4
    assert build.full_investigation_count / build.episode_count == 0.25
    assert not (build.root / "checkpoint.json").exists()
    assert len(tuple((build.root / "public" / "models").glob("*.glb"))) == 4
    assert len(tuple((build.root / "private" / "ground_truth").glob("*.json"))) == 16
    assert (
        build_corpus(
            tmp_path,
            corpus_version="test-v1",
            families=(GeneratorFamily.HIDDEN_CLIP, GeneratorFamily.STAMPED_ARROW),
            seeds=(0, 1),
        )
        == build
    )


def test_factory_rejects_existing_version_with_different_recipe(tmp_path: Path) -> None:
    build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )

    with pytest.raises(RuntimeError, match="different generator families"):
        build_corpus(
            tmp_path,
            corpus_version="test-v1",
            families=(GeneratorFamily.STAMPED_ARROW,),
            seeds=(0,),
        )

    with pytest.raises(RuntimeError, match="different generator seeds"):
        build_corpus(
            tmp_path,
            corpus_version="test-v1",
            families=(GeneratorFamily.HIDDEN_CLIP,),
            seeds=(1,),
        )


@pytest.mark.parametrize(
    ("families", "seeds", "message"),
    [
        ((), (0,), "at least one family"),
        ((GeneratorFamily.HIDDEN_CLIP,), (), "at least one family"),
        (
            (GeneratorFamily.HIDDEN_CLIP, GeneratorFamily.HIDDEN_CLIP),
            (0,),
            "families must be unique",
        ),
        ((GeneratorFamily.HIDDEN_CLIP,), (0, 0), "seeds must be unique"),
        ((GeneratorFamily.HIDDEN_CLIP,), (-1,), "seeds must be unique"),
    ],
)
def test_factory_rejects_invalid_generation_recipes(
    tmp_path: Path,
    families: tuple[GeneratorFamily, ...],
    seeds: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_corpus(tmp_path, families=families, seeds=seeds)


def test_factory_merges_validated_corpora_without_rewriting_artifacts(tmp_path: Path) -> None:
    first = build_corpus(
        tmp_path / "inputs",
        corpus_version="first",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    second = build_corpus(
        tmp_path / "inputs",
        corpus_version="second",
        families=(GeneratorFamily.STAMPED_ARROW,),
        seeds=(1,),
    )

    merged = merge_corpora(
        (first.root, second.root),
        tmp_path / "combined",
        corpus_version="qualification-v1",
    )

    assert merged.model_count == first.model_count + second.model_count
    assert merged.episode_count == first.episode_count + second.episode_count
    for name, digest in first.manifest.model_sha256.items():
        assert merged.manifest.model_sha256[name] == digest
    assert (
        merge_corpora(
            (first.root, second.root),
            tmp_path / "combined",
            corpus_version="qualification-v1",
        )
        == merged
    )


def test_factory_rejects_existing_version_from_stale_generator(tmp_path: Path) -> None:
    build = build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    manifest_path = build.root / "public" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="different generator"):
        build_corpus(
            tmp_path,
            corpus_version="test-v1",
            families=(GeneratorFamily.HIDDEN_CLIP,),
            seeds=(0,),
        )


def test_validation_rejects_tampered_model(tmp_path: Path) -> None:
    build = build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    model_path = next((build.root / "public" / "models").glob("*.glb"))
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="published model hash mismatch"):
        validate_corpus(build.root)


def test_validation_rejects_unmanifested_episode(tmp_path: Path) -> None:
    build = build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    episodes = build.root / "public" / "episodes"
    (episodes / "unmanifested.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not exactly match"):
        validate_corpus(build.root)


def test_validation_rejects_missing_private_truth(tmp_path: Path) -> None:
    build = build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    next((build.root / "private" / "ground_truth").glob("*.json")).unlink()

    with pytest.raises(RuntimeError, match="private ground-truth files"):
        validate_corpus(build.root)


def test_validation_rejects_tampered_episode_and_unreferenced_model(tmp_path: Path) -> None:
    build = build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    episode = next((build.root / "public" / "episodes").glob("*.json"))
    episode.write_text(episode.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="episode hash mismatch"):
        validate_corpus(build.root)

    rebuilt = build_corpus(
        tmp_path,
        corpus_version="test-v2",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    (rebuilt.root / "public" / "models" / "unreferenced.glb").write_bytes(b"model")
    with pytest.raises(RuntimeError, match="not referenced exactly"):
        validate_corpus(rebuilt.root)


def test_checkpoint_validation_and_recovery_paths(tmp_path: Path) -> None:
    generator_hash = generator_source_sha256()
    staging = tmp_path / ".corpus.building"
    staging.mkdir()
    _prepare_staging(staging, generator_hash)
    checkpoint = staging / "checkpoint.json"
    assert checkpoint.is_file()

    checkpoint.write_text(
        json.dumps({"generator_sha256": generator_hash, "completed": "invalid"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid completed-model"):
        _read_checkpoint(staging, generator_hash)

    checkpoint.write_text(
        json.dumps({"generator_sha256": "0" * 64, "completed": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different generator revision"):
        _read_checkpoint(staging, generator_hash)
    _prepare_staging(staging, generator_hash)
    assert _read_checkpoint(staging, generator_hash) == set()


def test_completed_checkpoint_cannot_publish_missing_artifacts(tmp_path: Path) -> None:
    staging = tmp_path / ".broken-v1.building"
    staging.mkdir()
    checkpoint = {
        "generator_sha256": generator_source_sha256(),
        "recipe_sha256": _recipe_sha256(
            generator_source_sha256(),
            (GeneratorFamily.HIDDEN_CLIP,),
            (0,),
            ModelSource.PROCEDURAL,
        ),
        "completed": [f"{GeneratorFamily.HIDDEN_CLIP}:0"],
    }
    (staging / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RuntimeError, match="publication count"):
        build_corpus(
            tmp_path,
            corpus_version="broken-v1",
            families=(GeneratorFamily.HIDDEN_CLIP,),
            seeds=(0,),
        )


def test_operation_coverage_validator_rejects_incomplete_classes(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    path = tmp_path / "episode.json"
    path.write_text(generate_episodes(model)[0].spec.model_dump_json(), encoding="utf-8")

    with pytest.raises(RuntimeError, match="operation class coverage"):
        _validate_operation_class_coverage([path])


def test_leakage_guards_reject_mismatches_paths_and_errors(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    episode = generate_episodes(model)[0]
    wrong_truth = episode.ground_truth.model_copy(update={"episode_id": "different_case"})
    with pytest.raises(LeakageError, match="same case"):
        validate_public_episode(episode.spec, wrong_truth)

    wrong_operations = episode.ground_truth.model_copy(
        update={"required_operations": (episode.ground_truth.required_operations[0],)}
    )
    with pytest.raises(LeakageError, match="required operations"):
        validate_public_episode(episode.spec, wrong_operations)

    prompt_leak = episode.spec.model_copy(
        update={"prompt": episode.spec.prompt + " " + episode.spec.episode_id}
    )
    with pytest.raises(LeakageError, match="opaque evaluator identifier"):
        validate_public_episode(prompt_leak, episode.ground_truth)

    with pytest.raises(LeakageError, match="tool error leaked"):
        scan_error_message(model.component_ids["idler"], episode.ground_truth)

    public = tmp_path / "served"
    public.mkdir()
    private = public / "private"
    private.mkdir()
    with pytest.raises(LeakageError, match="must not be under"):
        validate_public_directory(public, private)

    external_private = tmp_path / "evaluator-owned"
    external_private.mkdir()
    forbidden = public / "ground_truth"
    forbidden.write_text("leak", encoding="utf-8")
    with pytest.raises(LeakageError, match="private path name"):
        validate_public_directory(public, external_private)


def test_public_episode_rejects_private_component_id_leak(tmp_path: Path) -> None:
    published = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    episode = generate_episodes(published)[0]
    leaked = episode.spec.model_copy(
        update={"prompt": episode.spec.prompt + " " + published.component_ids["idler"]}
    )

    with pytest.raises(LeakageError, match="evaluator tokens"):
        validate_public_episode(leaked, episode.ground_truth)


def test_public_manifest_contains_no_ground_truth_fields(tmp_path: Path) -> None:
    build = build_corpus(
        tmp_path,
        corpus_version="test-v1",
        families=(GeneratorFamily.CLEARANCE_SLOT,),
        seeds=(0,),
    )
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (build.root / "public" / "episodes").glob("*.json")
    )
    for private_key in (
        '"answer"',
        '"component_roles"',
        '"component_paths"',
        '"state_requirements"',
        '"evidence_requirements"',
    ):
        assert private_key not in serialized
    assert json.loads((build.root / "public" / "manifest.json").read_text())["episodes"]
