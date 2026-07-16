from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pydantic import JsonValue

from meshprobe.evals.factory import build_corpus
from meshprobe.evals.generators import GeneratorFamily
from meshprobe.evals.harness.adapters import AdapterRun
from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.harness.runner import run_episode
from meshprobe.evals.schemas import (
    EpisodeGroundTruth,
    EpisodeSpec,
    EpisodeSubmission,
    StructuredAnswer,
    TaskFamily,
)
from meshprobe.protocol import (
    Command,
    ComponentFindCommand,
    ComponentInspectCommand,
    SceneOpenCommand,
    SessionSnapshotCommand,
)
from meshprobe.selectors import ComponentSelector, SelectorKind
from meshprobe.service import CommandResponse


class RunnerService:
    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.closed = False

    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        del evaluator_output_dir
        operation = command.op
        request_id = command.request_id
        result: JsonValue = {"source_sha256": "a" * 64}
        if operation == "session.snapshot":
            result = {"session": {"state_sha256": "1" * 64}}
        if operation == "component.find":
            result = [{"id": self.target_id}]
        if operation == "component.inspect":
            result = {"id": self.target_id}
        return CommandResponse(request_id=request_id, op=operation, result=result)

    def close(self) -> None:
        self.closed = True


class PassingAdapter:
    def __init__(self, answer: StructuredAnswer, target_id: str) -> None:
        self.answer = answer
        self.target_id = target_id

    def run(
        self,
        *,
        spec: EpisodeSpec,
        broker: EvaluationBroker,
        input_root: Path,
        artifact_root: Path,
    ) -> AdapterRun:
        del input_root, artifact_root
        broker.execute(
            SceneOpenCommand(
                request_id="open",
                op="scene.open",
                source_path=broker.visible_model_path,
            )
        )
        broker.execute(SessionSnapshotCommand(request_id="describe", op="session.snapshot"))
        broker.execute(
            ComponentFindCommand(
                request_id="find",
                op="component.find",
                selector=ComponentSelector(kind=SelectorKind.GLOB, pattern="**"),
            )
        )
        broker.execute(
            ComponentInspectCommand(
                request_id="inspect",
                op="component.inspect",
                component_id=self.target_id,
            )
        )
        return AdapterRun(
            submission=EpisodeSubmission(episode_id=spec.episode_id, answer=self.answer),
            returncode=0,
            stderr="",
            elapsed_seconds=0.2,
            timed_out=False,
            protocol_error=None,
        )


class NonzeroAdapter(PassingAdapter):
    def run(
        self,
        *,
        spec: EpisodeSpec,
        broker: EvaluationBroker,
        input_root: Path,
        artifact_root: Path,
    ) -> AdapterRun:
        run = super().run(
            spec=spec,
            broker=broker,
            input_root=input_root,
            artifact_root=artifact_root,
        )
        return replace(run, returncode=7)


def test_runner_materializes_isolated_episode_scores_and_publishes_report(tmp_path: Path) -> None:
    corpus = build_corpus(
        tmp_path / "corpus",
        corpus_version="runner-v1",
        families=(GeneratorFamily.HIDDEN_CLIP,),
        seeds=(0,),
    )
    episode_id = next(
        candidate
        for candidate in corpus.manifest.episodes
        if EpisodeSpec.model_validate_json(
            (corpus.root / "public" / "episodes" / f"{candidate}.json").read_text(encoding="utf-8")
        ).family
        is TaskFamily.COMPONENT_DISCOVERY
    )
    truth_path = corpus.root / "private" / "ground_truth" / f"{episode_id}.json"
    truth = EpisodeGroundTruth.model_validate_json(truth_path.read_text(encoding="utf-8"))
    service = RunnerService(truth.component_roles["idler"])

    run = run_episode(
        corpus_root=corpus.root,
        episode_id=episode_id,
        adapter=PassingAdapter(truth.answer, truth.component_roles["idler"]),
        output_root=tmp_path / "runs",
        service_factory=lambda: service,
    )

    assert run.report.passed
    assert run.report.gates[0].gate == "adapter"
    assert run.report.gates[0].status == "pass"
    assert run.report_path.is_file()
    assert run.trace_path.read_text(encoding="utf-8").count("\n") == 4
    assert service.closed
    assert not (run.run_root / "agent-input" / "ground_truth").exists()
    agent_run = run.run_root / "evaluator" / "agent-run"
    assert (agent_run / "prompt.txt").read_text(encoding="utf-8") == (
        run.run_root / "agent-input" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert (agent_run / "stream.jsonl").is_file()
    assert (agent_run / "stderr.log").is_file()

    failed = run_episode(
        corpus_root=corpus.root,
        episode_id=episode_id,
        adapter=NonzeroAdapter(truth.answer, truth.component_roles["idler"]),
        output_root=tmp_path / "nonzero-runs",
        service_factory=lambda: RunnerService(truth.component_roles["idler"]),
    )
    assert failed.report.gates[0].gate == "adapter"
    assert failed.report.gates[0].status == "fail"
