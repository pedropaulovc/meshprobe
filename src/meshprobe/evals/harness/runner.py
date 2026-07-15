"""End-to-end execution and scoring of one isolated qualification episode."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from meshprobe.evals.factory import CorpusBuild, validate_corpus
from meshprobe.evals.harness.adapters import AdapterRun
from meshprobe.evals.harness.broker import EvaluationBroker, EvaluationService
from meshprobe.evals.oracles import OracleInputs, score_episode
from meshprobe.evals.schemas import (
    AnswerStatus,
    EpisodeGroundTruth,
    EpisodeReport,
    EpisodeSpec,
    EpisodeSubmission,
    GateResult,
    GateStatus,
    StructuredAnswer,
    TraceEvent,
    TraceStatus,
)
from meshprobe.models import ContactSheetManifest, RenderManifest
from meshprobe.service import MeshProbeService
from meshprobe.sources import snapshot_source


class AgentAdapter(Protocol):
    def run(
        self,
        *,
        spec: EpisodeSpec,
        broker: EvaluationBroker,
        input_root: Path,
        artifact_root: Path,
    ) -> AdapterRun: ...


class EvaluationSession(EvaluationService, Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class EpisodeRun:
    report: EpisodeReport
    adapter: AdapterRun
    run_root: Path
    trace_path: Path
    report_path: Path


def run_episode(
    *,
    corpus_root: Path,
    episode_id: str,
    adapter: AgentAdapter,
    output_root: Path,
    service_factory: Callable[[], EvaluationSession] = MeshProbeService,
    validated_corpus: CorpusBuild | None = None,
) -> EpisodeRun:
    """Materialize one episode, run its agent, and score private evidence."""

    corpus_path = corpus_root.expanduser().resolve(strict=True)
    corpus = validated_corpus or validate_corpus(corpus_path)
    if corpus.root != corpus_path:
        raise ValueError("validated corpus does not match the requested corpus root")
    if episode_id not in corpus.manifest.episodes:
        raise ValueError(f"episode is not present in corpus manifest: {episode_id}")
    public_episode = corpus.root / "public" / "episodes" / f"{episode_id}.json"
    private_truth = corpus.root / "private" / "ground_truth" / f"{episode_id}.json"
    spec = EpisodeSpec.model_validate_json(public_episode.read_text(encoding="utf-8"))
    truth = EpisodeGroundTruth.model_validate_json(private_truth.read_text(encoding="utf-8"))
    run_root = output_root.expanduser().resolve() / episode_id
    if run_root.exists():
        raise FileExistsError(f"episode run already exists: {run_root}")
    input_root = run_root / "agent-input"
    artifact_root = run_root / "agent-artifacts"
    evaluator_root = run_root / "evaluator"
    for path in (input_root, artifact_root, evaluator_root):
        path.mkdir(parents=True)
    source_model = corpus.root / "public" / "models" / spec.model_file
    assigned_model = input_root / spec.model_file
    shutil.copy2(source_model, assigned_model)
    shutil.copy2(public_episode, input_root / "episode.json")
    (input_root / "prompt.txt").write_text(spec.prompt + "\n", encoding="utf-8")
    source_before = snapshot_source(assigned_model)
    trace_path = evaluator_root / "trace.jsonl"
    service = service_factory()
    try:
        broker = EvaluationBroker(
            service=service,
            model_path=assigned_model,
            artifact_root=artifact_root,
            evaluator_root=evaluator_root / "passes",
            trace_path=trace_path,
            budgets=spec.budgets,
        )
        adapter_run = adapter.run(
            spec=spec,
            broker=broker,
            input_root=input_root,
            artifact_root=artifact_root,
        )
    finally:
        service.close()
    source_after = snapshot_source(assigned_model)
    submission = adapter_run.submission or EpisodeSubmission(
        episode_id=spec.episode_id,
        answer=StructuredAnswer(status=AnswerStatus.ANSWERED, values={}),
    )
    renders, contact_sheets = _render_results(broker.events)
    report = score_episode(
        OracleInputs(
            spec=spec,
            truth=truth,
            submission=submission,
            trace=broker.events,
            source_before=source_before,
            source_after=source_after,
            artifact_root=artifact_root,
            renders=renders,
            contact_sheets=contact_sheets,
            public_errors=broker.public_errors,
            wall_seconds=adapter_run.elapsed_seconds,
        )
    )
    adapter_gate = _adapter_gate(adapter_run)
    report = report.model_copy(update={"gates": (adapter_gate, *report.gates)})
    report_path = evaluator_root / "report.json"
    _write_json(report_path, report.model_dump(mode="json"))
    _write_json(
        evaluator_root / "adapter.json",
        {
            "returncode": adapter_run.returncode,
            "stderr": adapter_run.stderr,
            "elapsed_seconds": adapter_run.elapsed_seconds,
            "timed_out": adapter_run.timed_out,
            "protocol_error": adapter_run.protocol_error,
        },
    )
    return EpisodeRun(
        report=report,
        adapter=adapter_run,
        run_root=run_root,
        trace_path=trace_path,
        report_path=report_path,
    )


def _render_results(
    events: tuple[TraceEvent, ...],
) -> tuple[tuple[RenderManifest, ...], tuple[ContactSheetManifest, ...]]:
    renders: list[RenderManifest] = []
    sheets: list[ContactSheetManifest] = []
    for item in events:
        if item.status is not TraceStatus.ACCEPTED:
            continue
        if item.operation == "render.image":
            renders.append(RenderManifest.model_validate(item.result))
        if item.operation == "render.contact_sheet":
            sheets.append(ContactSheetManifest.model_validate(item.result))
    return tuple(renders), tuple(sheets)


def _adapter_gate(run: AdapterRun) -> GateResult:
    if (
        run.submission is not None
        and run.protocol_error is None
        and not run.timed_out
        and run.returncode == 0
    ):
        return GateResult(
            gate="adapter",
            status=GateStatus.PASS,
            message="agent protocol completed with a structured submission",
        )
    return GateResult(
        gate="adapter",
        status=GateStatus.FAIL,
        message="agent did not complete the evaluation protocol",
        details={
            "returncode": run.returncode,
            "timed_out": run.timed_out,
            "protocol_error": run.protocol_error,
        },
    )


def _write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(serialized, encoding="utf-8")
    os.replace(pending, path)
