from __future__ import annotations

from pathlib import Path

import pytest

from meshprobe.evals.ergonomics import select_paired_episodes
from meshprobe.evals.factory import build_corpus, validate_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.migration import migrate_corpus_v2, restore_schema_v1_source
from meshprobe.evals.schemas import Difficulty, EpisodeSpec


def test_v2_migration_preserves_model_and_episode_identity(tmp_path: Path) -> None:
    source = build_corpus(
        tmp_path,
        corpus_version="procedural-v5",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    restore_schema_v1_source(source.root, corpus_version="procedural-v5")
    with pytest.raises(ValueError, match="migrate the corpus to schema version 2"):
        validate_corpus(source.root)

    migrated, audit = migrate_corpus_v2(
        source.root,
        tmp_path,
        corpus_version="procedural-v6",
    )

    assert migrated.model_count == 1
    assert migrated.episode_count == 4
    assert audit.source_version == "procedural-v5"
    assert audit.destination_version == "procedural-v6"
    assert audit.model_hashes_unchanged
    assert audit.episode_ids_unchanged
    assert audit.truth_payloads_unchanged


def test_ergonomics_selection_pairs_basic_and_intermediate_episodes(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path,
        corpus_version="selection-v2",
        families=(GeneratorFamily.HIDDEN_CLIP, GeneratorFamily.STAMPED_ARROW),
        seeds=(0,),
    )
    selected = select_paired_episodes(corpus.root, per_difficulty=2)
    specs = [
        EpisodeSpec.model_validate_json(
            (corpus.root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        for episode_id in selected
    ]

    assert len(selected) == 4
    assert [spec.difficulty for spec in specs].count(Difficulty.BASIC) == 2
    assert [spec.difficulty for spec in specs].count(Difficulty.INTERMEDIATE) == 2
