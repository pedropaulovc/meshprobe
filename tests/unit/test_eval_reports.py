from __future__ import annotations

from pathlib import Path

from meshprobe.evals.generators import (
    GeneratedEpisode,
    GeneratorFamily,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.reports import (
    QualificationProvenance,
    ScoredEpisode,
    aggregate_reports,
    publish_report,
    report_markdown,
)
from meshprobe.evals.schemas import (
    CorpusTier,
    EpisodeReport,
    GateResult,
    GateStatus,
    PassThresholds,
    RuntimePin,
)


def report_for(episode_id: str, *, passed: bool, template: GeneratedEpisode) -> EpisodeReport:
    spec = template.spec
    return EpisodeReport(
        episode_id=episode_id,
        family=spec.family,
        episode_class=spec.episode_class,
        difficulty=spec.difficulty,
        gates=(
            GateResult(
                gate="answer",
                status=GateStatus.PASS if passed else GateStatus.FAIL,
                message="scored",
            ),
        ),
        tool_calls=1,
        renders=0,
        total_pixels=0,
        output_bytes=0,
        wall_seconds=1,
    )


def provenance(thresholds: PassThresholds) -> QualificationProvenance:
    return QualificationProvenance(
        tier=CorpusTier.SMOKE,
        corpus_version="test-v1",
        corpus_manifest_sha256="1" * 64,
        tier_manifest_sha256="2" * 64,
        runtime=RuntimePin(
            meshprobe_version="0.1.0.dev0",
            blender_version="5.2.0 LTS",
            importer_sha256="3" * 64,
            render_engines=("eevee", "cycles"),
        ),
        thresholds=thresholds,
        adapter="tests.ReferenceAgent",
        adapter_identity_sha256="4" * 64,
        workers=3,
    )


def test_reports_split_every_required_dimension_and_enforce_thresholds(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    generated = generate_episodes(model)
    discovery = generated[0]
    full = generated[3]
    cases = (
        ScoredEpisode(
            spec=discovery.spec,
            truth=discovery.ground_truth,
            report=report_for(discovery.spec.episode_id, passed=True, template=discovery),
        ),
        ScoredEpisode(
            spec=full.spec,
            truth=full.ground_truth,
            report=report_for(full.spec.episode_id, passed=False, template=full),
        ),
    )

    thresholds = PassThresholds(overall=0.5, full_stack=0.5, per_operation=0.5)
    report = aggregate_reports(cases, thresholds=thresholds, provenance=provenance(thresholds))

    assert report.overall.pass_rate == 0.5
    assert report.full_stack.pass_rate == 0
    assert report.by_operation["scene.open"].episodes == 2
    assert "perspective" in report.by_projection
    assert "telephoto_70_to_134mm" in report.by_focal_length_band
    assert "raking_left" in report.by_illumination
    assert "procedural" in report.by_model_source
    assert report.failure_classes == {"answer": 1}
    assert report.provenance.tier is CorpusTier.SMOKE
    assert report.provenance.adapter_identity_sha256 == "4" * 64
    assert report.provenance.workers == 3
    assert "full_stack" in report.threshold_failures
    assert "operation:render.image" in report.threshold_failures

    output = tmp_path / "reports" / "qualification.json"
    publish_report(output, report)
    assert output.is_file()
    markdown = report_markdown(report)
    assert "MeshProbe qualification report" in markdown
    assert "`scene.open`" in markdown
