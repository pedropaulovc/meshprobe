"""Checkpointed execution of an exact pinned qualification tier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from meshprobe.evals.factory import validate_corpus
from meshprobe.evals.harness.runner import AgentAdapter, EvaluationSession, run_episode
from meshprobe.evals.reports import (
    QualificationReport,
    ScoredEpisode,
    aggregate_reports,
    publish_report,
    report_markdown,
)
from meshprobe.evals.schemas import EpisodeGroundTruth, EpisodeReport, EpisodeSpec, RuntimePin
from meshprobe.evals.tiers import current_runtime_pin, validate_runtime_pin, validate_tier_manifest
from meshprobe.service import MeshProbeService
from meshprobe.sources import sha256_file


@dataclass(frozen=True)
class TierRun:
    root: Path
    completed: int
    report: QualificationReport
    report_path: Path


def run_tier(
    *,
    corpus_root: Path,
    tier_manifest_path: Path,
    adapter: AgentAdapter,
    output_root: Path,
    service_factory: Callable[[], EvaluationSession] = MeshProbeService,
    runtime_provider: Callable[[], RuntimePin] = current_runtime_pin,
) -> TierRun:
    """Run every pinned episode exactly once and resume from validated reports."""

    corpus_path = corpus_root.expanduser().resolve(strict=True)
    tier_path = tier_manifest_path.expanduser().resolve(strict=True)
    corpus = validate_corpus(corpus_path)
    tier = validate_tier_manifest(tier_path, corpus_path)
    validate_runtime_pin(tier.runtime, runtime_provider())
    run_root = output_root.expanduser().resolve() / str(tier.tier)
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_root / "checkpoint.json"
    identity = _run_identity(corpus.root / "public" / "manifest.json", tier_path, adapter)
    completed = _read_checkpoint(checkpoint_path, identity)
    runs_root = run_root / "episodes"
    runs_root.mkdir(exist_ok=True)

    for episode_id in tier.episodes:
        if episode_id in completed:
            report_path = runs_root / episode_id / "evaluator" / "report.json"
            EpisodeReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            continue
        episode_root = runs_root / episode_id
        report_path = episode_root / "evaluator" / "report.json"
        if report_path.is_file():
            EpisodeReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            completed.add(episode_id)
            _write_checkpoint(checkpoint_path, identity, completed)
            continue
        if episode_root.exists():
            shutil.rmtree(episode_root)
        run_episode(
            corpus_root=corpus.root,
            episode_id=episode_id,
            adapter=adapter,
            output_root=runs_root,
            validated_corpus=corpus,
            service_factory=service_factory,
        )
        completed.add(episode_id)
        _write_checkpoint(checkpoint_path, identity, completed)

    cases = tuple(
        _load_scored_episode(corpus.root, runs_root, episode_id) for episode_id in tier.episodes
    )
    report = aggregate_reports(cases, thresholds=tier.thresholds)
    report_path = run_root / "qualification-report.json"
    publish_report(report_path, report)
    markdown = run_root / "qualification-report.md"
    _write_text(markdown, report_markdown(report))
    return TierRun(
        root=run_root,
        completed=len(completed),
        report=report,
        report_path=report_path,
    )


def _run_identity(corpus_manifest: Path, tier_manifest: Path, adapter: AgentAdapter) -> str:
    command = getattr(adapter, "command", None)
    identity_paths = list(getattr(adapter, "identity_paths", ()))
    if isinstance(command, tuple):
        identity_paths.extend(_command_files(command))
    payload = {
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "tier_manifest_sha256": sha256_file(tier_manifest),
        "adapter_type": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "adapter_command": list(command) if isinstance(command, tuple) else repr(adapter),
        "agent_files": {
            str(path.expanduser().resolve()): sha256_file(path.expanduser().resolve(strict=True))
            for path in sorted(set(identity_paths))
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _command_files(command: tuple[str, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, token in enumerate(command):
        candidate = Path(token).expanduser()
        if index == 0 and not candidate.is_file():
            resolved = shutil.which(token)
            if resolved is not None:
                candidate = Path(resolved)
        if candidate.is_file():
            paths.append(candidate)
    return tuple(paths)


def _read_checkpoint(path: Path, identity: str) -> set[str]:
    if not path.is_file():
        _write_json(path, {"identity_sha256": identity, "completed": []})
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity_sha256") != identity:
        raise RuntimeError("suite output belongs to a different corpus, tier, or agent command")
    completed = payload.get("completed")
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise RuntimeError("suite checkpoint has an invalid completed-episode list")
    return set(completed)


def _write_checkpoint(path: Path, identity: str, completed: set[str]) -> None:
    _write_json(path, {"identity_sha256": identity, "completed": sorted(completed)})


def _load_scored_episode(
    corpus_root: Path,
    runs_root: Path,
    episode_id: str,
) -> ScoredEpisode:
    spec = EpisodeSpec.model_validate_json(
        (corpus_root / "public" / "episodes" / f"{episode_id}.json").read_text(encoding="utf-8")
    )
    truth = EpisodeGroundTruth.model_validate_json(
        (corpus_root / "private" / "ground_truth" / f"{episode_id}.json").read_text(
            encoding="utf-8"
        )
    )
    report = EpisodeReport.model_validate_json(
        (runs_root / episode_id / "evaluator" / "report.json").read_text(encoding="utf-8")
    )
    return ScoredEpisode(spec=spec, truth=truth, report=report)


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _write_text(path: Path, payload: str) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(payload, encoding="utf-8")
    os.replace(pending, path)
