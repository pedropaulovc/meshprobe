from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily, build_model
from meshprobe.evals.harness.adapters import ReferenceAgentAdapter
from meshprobe.evals.harness.runner import run_episode
from meshprobe.evals.schemas import Difficulty, EpisodeSpec

pytestmark = pytest.mark.skipif(
    shutil.which("blender") is None or (os.name != "nt" and shutil.which("bwrap") is None),
    reason="Blender and the platform sandbox are required",
)


def test_reference_agent_passes_full_stack_rigid_and_rescaled_episodes(
    tmp_path: Path,
) -> None:
    corpus = build_corpus(
        tmp_path / "corpora",
        corpus_version="reference-integration-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=range(3),
    )
    target_names = {
        seed: build_model(GeneratorFamily.HIDDEN_CLIP, seed).target_name for seed in range(3)
    }
    selected: list[str] = []
    for episode_id in corpus.manifest.episodes:
        spec = EpisodeSpec.model_validate_json(
            (corpus.root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        full_stack_base = (
            target_names[0] in spec.prompt and spec.difficulty is Difficulty.FULL_STACK
        )
        transformed_intermediate = (
            any(target_names[seed] in spec.prompt for seed in (1, 2))
            and spec.difficulty is Difficulty.INTERMEDIATE
        )
        if full_stack_base or transformed_intermediate:
            selected.append(episode_id)

    assert len(selected) == 3
    for episode_id in selected:
        run = run_episode(
            corpus_root=corpus.root,
            episode_id=episode_id,
            adapter=ReferenceAgentAdapter(),
            output_root=tmp_path / "runs",
            validated_corpus=corpus,
        )

        assert run.adapter.protocol_error is None
        assert run.report.passed, {
            gate.gate: gate.details for gate in run.report.gates if gate.status == "fail"
        }
