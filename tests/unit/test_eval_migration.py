from __future__ import annotations

from pathlib import Path

import pytest

from meshprobe.evals.ergonomics import ergonomics_report_markdown, select_paired_episodes
from meshprobe.evals.factory import build_corpus, validate_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.migration import (
    codify_opaque_family,
    migrate_corpus_v2,
    restore_schema_v1_source,
)
from meshprobe.evals.schemas import Difficulty, EpisodeGroundTruth, EpisodeSpec, ModelSource


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

    existing, repeated_audit = migrate_corpus_v2(
        source.root,
        tmp_path,
        corpus_version="procedural-v6",
    )
    assert existing.root == migrated.root
    assert repeated_audit == audit


def test_private_opaque_family_construction_is_a_production_step(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path,
        corpus_version="private-v7",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(32, 33),
        model_source=ModelSource.PRIVATE,
    )

    codify_opaque_family(corpus.root)
    validated = validate_corpus(corpus.root)
    truth = EpisodeGroundTruth.model_validate_json(
        next((corpus.root / "private" / "ground_truth").glob("*.json")).read_text(encoding="utf-8")
    )

    assert validated.manifest.generator_families == ("hidden_clip", "opaque_family_v7")
    assert truth.generator_family == "opaque_family_v7"


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


def test_ergonomics_markdown_is_explicitly_diagnostic() -> None:
    report = {
        "summary": {
            "claude-opus": {
                "attempts": 2,
                "answer_passes": 1,
                "evidence_passes": 2,
                "input_tokens": 100,
                "output_tokens": 20,
                "reasoning_output_tokens": 0,
                "meshprobe_operations": 5,
                "invalid_calls": 1,
                "elapsed_seconds": 12.5,
            }
        }
    }

    markdown = ergonomics_report_markdown(report)

    assert "| claude-opus | 2 | 1 | 2 | 100 | 20 | 0 | 5 | 1 | 12.5 |" in markdown
    assert "not a release qualification result" in markdown
    assert "Hidden chain of thought was not collected" in markdown
