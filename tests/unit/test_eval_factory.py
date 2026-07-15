from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import (
    GeneratorFamily,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.leakage import LeakageError, validate_public_episode


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
    assert build_corpus(tmp_path, corpus_version="test-v1") == build


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
