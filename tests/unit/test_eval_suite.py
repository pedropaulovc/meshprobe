from __future__ import annotations

from pathlib import Path

from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.harness.adapters import AdapterRun
from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.harness.suite import run_tier
from meshprobe.evals.schemas import (
    CorpusTier,
    EpisodeGroundTruth,
    EpisodeSpec,
    EpisodeSubmission,
    PassThresholds,
    RuntimePin,
    TierManifest,
)
from meshprobe.protocol import Command
from meshprobe.service import CommandResponse
from meshprobe.sources import sha256_file


class NoopService:
    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        del command, evaluator_output_dir
        raise AssertionError("answer-only adapter must not execute tools")

    def close(self) -> None:
        pass


class AnswerAdapter:
    command = ("test-answer-agent",)

    def __init__(self, answers: dict[str, EpisodeSubmission]) -> None:
        self.answers = answers
        self.calls = 0

    def run(
        self,
        *,
        spec: EpisodeSpec,
        broker: EvaluationBroker,
        input_root: Path,
        artifact_root: Path,
    ) -> AdapterRun:
        del broker, input_root, artifact_root
        self.calls += 1
        return AdapterRun(
            submission=self.answers[spec.episode_id],
            returncode=0,
            stderr="",
            elapsed_seconds=0.01,
            timed_out=False,
            protocol_error=None,
        )


def test_tier_runner_checkpoints_reports_and_resumes(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path / "corpora",
        corpus_version="test-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    answers: dict[str, EpisodeSubmission] = {}
    for episode_id in corpus.manifest.episodes:
        truth = EpisodeGroundTruth.model_validate_json(
            (corpus.root / "private" / "ground_truth" / f"{episode_id}.json").read_text(
                encoding="utf-8"
            )
        )
        answers[episode_id] = EpisodeSubmission(
            episode_id=episode_id,
            answer=truth.answer,
        )
    adapter = AnswerAdapter(answers)
    tier = TierManifest(
        tier=CorpusTier.SMOKE,
        corpus_version=corpus.manifest.corpus_version,
        corpus_manifest_sha256=sha256_file(corpus.root / "public" / "manifest.json"),
        generator_sha256=corpus.manifest.generator_sha256,
        runtime=RuntimePin(
            meshprobe_version="test",
            blender_version="test",
            importer_sha256="a" * 64,
            render_engines=("eevee",),
        ),
        thresholds=PassThresholds(overall=0, full_stack=0, per_operation=0),
        episodes=corpus.manifest.episodes,
        episode_sha256=corpus.manifest.episode_sha256,
        model_sha256=corpus.manifest.model_sha256,
    )
    tier_path = tmp_path / "smoke.json"
    tier_path.write_text(tier.model_dump_json(), encoding="utf-8")

    first = run_tier(
        corpus_root=corpus.root,
        tier_manifest_path=tier_path,
        adapter=adapter,
        output_root=tmp_path / "runs",
        service_factory=NoopService,
        runtime_provider=lambda: tier.runtime,
    )
    second = run_tier(
        corpus_root=corpus.root,
        tier_manifest_path=tier_path,
        adapter=adapter,
        output_root=tmp_path / "runs",
        service_factory=NoopService,
        runtime_provider=lambda: tier.runtime,
    )

    assert first.completed == 4
    assert first.report_path.is_file()
    assert second == first
    assert adapter.calls == 4
