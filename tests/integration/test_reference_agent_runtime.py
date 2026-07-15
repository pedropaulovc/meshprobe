from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily, build_model
from meshprobe.evals.harness.adapters import ReferenceAgentAdapter
from meshprobe.evals.harness.suite import run_tier
from meshprobe.evals.schemas import (
    CorpusTier,
    Difficulty,
    EpisodeReport,
    EpisodeSpec,
    PassThresholds,
    TaskFamily,
    TierManifest,
)
from meshprobe.evals.tiers import current_runtime_pin
from meshprobe.sources import sha256_file

pytestmark = pytest.mark.skipif(
    shutil.which("blender") is None or (os.name != "nt" and shutil.which("bwrap") is None),
    reason="Blender and the platform sandbox are required",
)


def test_reference_agent_passes_transforms_and_rescaled_raking_evidence(
    tmp_path: Path,
) -> None:
    transformed_seeds = (0, 1, 18)
    corpus = build_corpus(
        tmp_path / "corpora",
        corpus_version="reference-integration-v1",
        families=(GeneratorFamily.HIDDEN_CLIP, GeneratorFamily.STAMPED_ARROW),
        seeds=transformed_seeds,
    )
    target_names = {
        seed: build_model(GeneratorFamily.HIDDEN_CLIP, seed).target_name
        for seed in transformed_seeds
    }
    stamped_arrow_name = build_model(GeneratorFamily.STAMPED_ARROW, 18).target_name
    selected: list[str] = []
    specs: dict[str, EpisodeSpec] = {}
    for episode_id in corpus.manifest.episodes:
        spec = EpisodeSpec.model_validate_json(
            (corpus.root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        full_stack_base = (
            target_names[0] in spec.prompt and spec.difficulty is Difficulty.FULL_STACK
        )
        transformed_intermediate = (
            any(target_names[seed] in spec.prompt for seed in (1, 18))
            and spec.difficulty is Difficulty.INTERMEDIATE
        )
        rescaled_raking_evidence = (
            stamped_arrow_name in spec.prompt
            and spec.family is TaskFamily.SURFACE_DETAIL
        )
        if full_stack_base or transformed_intermediate or rescaled_raking_evidence:
            selected.append(episode_id)
            specs[episode_id] = spec

    assert len(selected) == 4
    runtime = current_runtime_pin()
    tier = TierManifest(
        tier=CorpusTier.SMOKE,
        corpus_version=corpus.manifest.corpus_version,
        corpus_manifest_sha256=sha256_file(corpus.root / "public" / "manifest.json"),
        generator_sha256=corpus.manifest.generator_sha256,
        runtime=runtime,
        thresholds=PassThresholds(overall=1, full_stack=1, per_operation=1),
        episodes=tuple(selected),
        episode_sha256={
            episode_id: corpus.manifest.episode_sha256[episode_id] for episode_id in selected
        },
        model_sha256={
            spec.model_file: corpus.manifest.model_sha256[spec.model_file]
            for spec in specs.values()
        },
    )
    tier_path = tmp_path / "smoke.json"
    tier_path.write_text(tier.model_dump_json(), encoding="utf-8")

    result = run_tier(
        corpus_root=corpus.root,
        tier_manifest_path=tier_path,
        adapter=ReferenceAgentAdapter(),
        output_root=tmp_path / "runs",
        runtime_provider=lambda: runtime,
        workers=2,
    )

    assert result.completed == 4
    failures: dict[str, dict[str, object]] = {}
    for episode_id in selected:
        report = EpisodeReport.model_validate_json(
            (result.root / "episodes" / episode_id / "evaluator" / "report.json").read_text(
                encoding="utf-8"
            )
        )
        if report.passed:
            continue
        adapter = json.loads(
            (result.root / "episodes" / episode_id / "evaluator" / "adapter.json").read_text(
                encoding="utf-8"
            )
        )
        failures[episode_id] = {
            "adapter": adapter,
            "gates": {gate.gate: gate.details for gate in report.gates if gate.status == "fail"},
        }

    if failures:
        raise AssertionError(json.dumps(failures, indent=2, sort_keys=True))
    assert result.report.passed_thresholds, result.report.threshold_failures
