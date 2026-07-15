"""Qualification report aggregation across every required evaluation slice."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from meshprobe.evals.schemas import (
    CorpusTier,
    EpisodeGroundTruth,
    EpisodeReport,
    EpisodeSpec,
    EvidenceKind,
    GateStatus,
    PassThresholds,
    RuntimePin,
    Sha256,
    StatePredicate,
    TaskFamily,
)


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SliceMetrics(ReportModel):
    episodes: int = Field(ge=0)
    passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)


class QualificationProvenance(ReportModel):
    tier: CorpusTier
    corpus_version: str
    corpus_manifest_sha256: Sha256
    tier_manifest_sha256: Sha256
    runtime: RuntimePin
    thresholds: PassThresholds
    adapter: str
    adapter_identity_sha256: Sha256


class QualificationReport(ReportModel):
    schema_version: int = 1
    provenance: QualificationProvenance
    overall: SliceMetrics
    full_stack: SliceMetrics
    by_operation: dict[str, SliceMetrics]
    by_task_family: dict[str, SliceMetrics]
    by_difficulty: dict[str, SliceMetrics]
    by_projection: dict[str, SliceMetrics]
    by_focal_length_band: dict[str, SliceMetrics]
    by_illumination: dict[str, SliceMetrics]
    by_model_source: dict[str, SliceMetrics]
    failure_classes: dict[str, int]
    threshold_failures: tuple[str, ...]

    @property
    def passed_thresholds(self) -> bool:
        return not self.threshold_failures


@dataclass(frozen=True)
class ScoredEpisode:
    spec: EpisodeSpec
    truth: EpisodeGroundTruth
    report: EpisodeReport


def aggregate_reports(
    episodes: Iterable[ScoredEpisode],
    *,
    thresholds: PassThresholds,
    provenance: QualificationProvenance,
) -> QualificationReport:
    cases = tuple(episodes)
    if not cases:
        raise ValueError("qualification report requires at least one scored episode")
    _validate_case_ids(cases)
    overall = _metrics(cases)
    full_stack = _metrics(
        case for case in cases if case.spec.family is TaskFamily.FULL_INVESTIGATION
    )
    by_operation = _split(
        cases,
        lambda case: (str(operation) for operation in case.spec.required_operations),
    )
    by_task_family = _split(cases, lambda case: (str(case.spec.family),))
    by_difficulty = _split(cases, lambda case: (str(case.spec.difficulty),))
    by_projection = _split(cases, _projection_keys)
    by_focal = _split(cases, _focal_keys)
    by_illumination = _split(cases, _illumination_keys)
    by_source = _split(cases, lambda case: (str(case.spec.model_source),))
    failures = Counter(
        gate.gate for case in cases for gate in case.report.gates if gate.status is GateStatus.FAIL
    )
    threshold_failures: list[str] = []
    if overall.pass_rate < thresholds.overall:
        threshold_failures.append("overall")
    if full_stack.pass_rate < thresholds.full_stack:
        threshold_failures.append("full_stack")
    threshold_failures.extend(
        f"operation:{operation}"
        for operation, metrics in by_operation.items()
        if metrics.pass_rate < thresholds.per_operation
    )
    return QualificationReport(
        provenance=provenance,
        overall=overall,
        full_stack=full_stack,
        by_operation=by_operation,
        by_task_family=by_task_family,
        by_difficulty=by_difficulty,
        by_projection=by_projection,
        by_focal_length_band=by_focal,
        by_illumination=by_illumination,
        by_model_source=by_source,
        failure_classes=dict(sorted(failures.items())),
        threshold_failures=tuple(threshold_failures),
    )


def publish_report(path: Path, report: QualificationReport) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = report.model_dump_json(indent=2) + "\n"
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(serialized, encoding="utf-8")
    os.replace(pending, path)


def report_markdown(report: QualificationReport) -> str:
    lines = [
        "# MeshProbe qualification report",
        "",
        f"Tier: `{report.provenance.tier}`",
        f"Corpus: `{report.provenance.corpus_version}`",
        f"Tier manifest: `{report.provenance.tier_manifest_sha256}`",
        f"Adapter: `{report.provenance.adapter}` (`{report.provenance.adapter_identity_sha256}`)",
        f"Runtime: MeshProbe {report.provenance.runtime.meshprobe_version}, "
        f"Blender {report.provenance.runtime.blender_version}",
        "",
        f"Overall: {report.overall.passed}/{report.overall.episodes} "
        f"({report.overall.pass_rate:.1%})",
        f"Full stack: {report.full_stack.passed}/{report.full_stack.episodes} "
        f"({report.full_stack.pass_rate:.1%})",
        "",
        "| Operation | Passed | Episodes | Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| `{operation}` | {metrics.passed} | {metrics.episodes} | {metrics.pass_rate:.1%} |"
        for operation, metrics in report.by_operation.items()
    )
    if report.threshold_failures:
        lines.extend(("", "Threshold failures: " + ", ".join(report.threshold_failures)))
    return "\n".join(lines) + "\n"


def _validate_case_ids(cases: tuple[ScoredEpisode, ...]) -> None:
    ids: set[str] = set()
    for case in cases:
        identifiers = {
            case.spec.episode_id,
            case.truth.episode_id,
            case.report.episode_id,
        }
        if len(identifiers) != 1:
            raise ValueError("scored episode inputs identify different cases")
        if case.spec.episode_id in ids:
            raise ValueError(f"duplicate scored episode: {case.spec.episode_id}")
        ids.add(case.spec.episode_id)


def _metrics(cases: Iterable[ScoredEpisode]) -> SliceMetrics:
    selected = tuple(cases)
    passed = sum(case.report.passed for case in selected)
    return SliceMetrics(
        episodes=len(selected),
        passed=passed,
        pass_rate=passed / len(selected) if selected else 0,
    )


def _split(
    cases: tuple[ScoredEpisode, ...],
    keys: Callable[[ScoredEpisode], Iterable[str]],
) -> dict[str, SliceMetrics]:
    grouped: defaultdict[str, list[ScoredEpisode]] = defaultdict(list)
    for case in cases:
        case_keys = tuple(dict.fromkeys(keys(case))) or ("not_required",)
        for key in case_keys:
            grouped[key].append(case)
    return {key: _metrics(grouped[key]) for key in sorted(grouped)}


def _projection_keys(case: ScoredEpisode) -> Iterable[str]:
    for state_requirement in case.truth.state_requirements:
        if state_requirement.predicate is StatePredicate.PROJECTION_MODE:
            yield str(state_requirement.expected)
    for evidence_requirement in case.truth.evidence_requirements:
        if evidence_requirement.kind is EvidenceKind.PROJECTION:
            yield str(evidence_requirement.expected)


def _focal_keys(case: ScoredEpisode) -> Iterable[str]:
    values: list[float] = []
    for state_requirement in case.truth.state_requirements:
        if state_requirement.predicate is StatePredicate.FOCAL_LENGTH_MM and isinstance(
            state_requirement.expected, (int, float)
        ):
            values.append(float(state_requirement.expected))
    for evidence_requirement in case.truth.evidence_requirements:
        if evidence_requirement.kind is EvidenceKind.FOCAL_LENGTH and isinstance(
            evidence_requirement.expected, (int, float)
        ):
            values.append(float(evidence_requirement.expected))
    yield from (_focal_band(value) for value in values)


def _illumination_keys(case: ScoredEpisode) -> Iterable[str]:
    for state_requirement in case.truth.state_requirements:
        if state_requirement.predicate is StatePredicate.ILLUMINATION_PRESET:
            yield str(state_requirement.expected)
    for evidence_requirement in case.truth.evidence_requirements:
        if evidence_requirement.kind is EvidenceKind.ILLUMINATION:
            yield str(evidence_requirement.expected)


def _focal_band(value: float) -> str:
    if value < 35:
        return "wide_below_35mm"
    if value < 70:
        return "normal_35_to_69mm"
    if value < 135:
        return "telephoto_70_to_134mm"
    return "long_135mm_and_above"
