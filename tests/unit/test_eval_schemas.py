from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from meshprobe.evals.schemas import (
    EVALUATED_OPERATIONS,
    AnswerStatus,
    CorpusManifest,
    CorpusTier,
    Difficulty,
    EpisodeClass,
    EpisodeGroundTruth,
    EpisodeSpec,
    EvidenceKind,
    EvidenceRequirement,
    ModelSource,
    Operation,
    PassThresholds,
    RuntimePin,
    StatePredicate,
    StateRequirement,
    StructuredAnswer,
    TaskFamily,
    TierManifest,
    TraceEvent,
    TraceStatus,
)


def test_full_investigation_requires_every_evaluated_operation() -> None:
    assert Operation.VIEW_ROTATE in EVALUATED_OPERATIONS

    with pytest.raises(ValidationError, match="every evaluated operation"):
        EpisodeSpec(
            episode_id="episode_0001",
            model_file="fixture.glb",
            model_sha256="a" * 64,
            family=TaskFamily.FULL_INVESTIGATION,
            episode_class=EpisodeClass.POSITIVE,
            difficulty=Difficulty.FULL_STACK,
            model_source=ModelSource.PROCEDURAL,
            prompt="Inspect the complete assembly and submit all requested evidence.",
            answer_schema={"type": "object"},
            required_operations=(Operation.SCENE_OPEN, Operation.SESSION_SNAPSHOT),
        )


def test_indeterminate_answer_requires_an_explanation() -> None:
    with pytest.raises(ValidationError, match="explain"):
        StructuredAnswer(status=AnswerStatus.INDETERMINATE)


def test_ground_truth_rejects_mismatched_role_maps() -> None:
    with pytest.raises(ValidationError, match="same roles"):
        EpisodeGroundTruth(
            episode_id="episode_0001",
            model_sha256="a" * 64,
            generator_family="fixture",
            answer=StructuredAnswer(status=AnswerStatus.ANSWERED, values={"side": "positive"}),
            component_roles={"target": "cmp_abc"},
            component_paths={},
            required_operations=(Operation.SCENE_OPEN,),
        )


def test_answer_evidence_and_state_require_context() -> None:
    with pytest.raises(ValidationError, match="answered results"):
        StructuredAnswer(
            status=AnswerStatus.ANSWERED,
            conflicting_component_ids=("component",),
        )
    with pytest.raises(ValidationError, match="minimum"):
        EvidenceRequirement(kind=EvidenceKind.PROJECTION, minimum=2, maximum=1)
    with pytest.raises(ValidationError, match="component_role"):
        EvidenceRequirement(kind=EvidenceKind.TARGET_VISIBLE)
    with pytest.raises(ValidationError, match="component_role"):
        EvidenceRequirement(kind=EvidenceKind.TARGET_SCREEN_SPAN, minimum=0.2)
    with pytest.raises(ValidationError, match="component_role"):
        StateRequirement(predicate=StatePredicate.COMPONENT_DISPLAY, expected="hidden")


def test_episode_and_ground_truth_reject_ambiguous_operation_contracts() -> None:
    fields: dict[str, Any] = {
        "episode_id": "episode_0001",
        "model_file": "fixture.glb",
        "model_sha256": "a" * 64,
        "family": TaskFamily.COMPONENT_DISCOVERY,
        "episode_class": EpisodeClass.POSITIVE,
        "difficulty": Difficulty.BASIC,
        "model_source": ModelSource.PROCEDURAL,
        "prompt": "Inspect the assigned model and report the component identity.",
        "answer_schema": {"type": "object"},
    }
    with pytest.raises(ValidationError, match="must be unique"):
        EpisodeSpec(
            **fields,
            required_operations=(Operation.SCENE_OPEN, Operation.SCENE_OPEN),
        )
    with pytest.raises(ValidationError, match="first required"):
        EpisodeSpec(**fields, required_operations=(Operation.SESSION_SNAPSHOT,))

    truth_fields: dict[str, Any] = {
        "episode_id": "episode_0001",
        "model_sha256": "a" * 64,
        "generator_family": "fixture",
        "answer": StructuredAnswer(status=AnswerStatus.ANSWERED),
        "component_roles": {"target": "component"},
        "component_paths": {"target": "assembly/target"},
    }
    with pytest.raises(ValidationError, match="unknown component roles"):
        EpisodeGroundTruth(
            **truth_fields,
            state_requirements=(
                StateRequirement(
                    predicate=StatePredicate.COMPONENT_MARK,
                    component_role="missing",
                    expected="highlighted",
                ),
            ),
            required_operations=(Operation.SCENE_OPEN,),
        )
    with pytest.raises(ValidationError, match="must be unique"):
        EpisodeGroundTruth(
            **truth_fields,
            required_operations=(Operation.SCENE_OPEN, Operation.SCENE_OPEN),
        )


def test_trace_status_requires_consistent_error_fields() -> None:
    fields: dict[str, Any] = {
        "sequence": 1,
        "request_id": "request",
        "operation": Operation.SCENE_OPEN,
        "arguments": {},
        "started_monotonic": 1.0,
        "elapsed_seconds": 0.1,
    }
    with pytest.raises(ValidationError, match="accepted trace"):
        TraceEvent(**fields, status=TraceStatus.ACCEPTED, error_code="error")
    with pytest.raises(ValidationError, match="failed trace"):
        TraceEvent(**fields, status=TraceStatus.REJECTED)


def test_corpus_and_tier_manifests_reject_duplicate_or_incomplete_membership() -> None:
    base = {
        "corpus_version": "test-v1",
        "tier": CorpusTier.RELEASE,
        "generator_sha256": "a" * 64,
        "model_sources": (ModelSource.PROCEDURAL,),
        "generator_families": ("family",),
        "generator_seeds": (0,),
        "model_sha256": {"model.glb": "b" * 64},
        "episodes": ("episode_0001",),
        "episode_sha256": {"episode_0001": "c" * 64},
    }
    for update, message in (
        ({"model_sources": (ModelSource.PROCEDURAL, ModelSource.PROCEDURAL)}, "sources"),
        ({"generator_families": ("family", "family")}, "families"),
        ({"generator_seeds": (0, 0)}, "seeds"),
        ({"episodes": ("episode_0001", "episode_0001")}, "episodes"),
        ({"episode_sha256": {"other_ep": "c" * 64}}, "exactly cover"),
    ):
        with pytest.raises(ValidationError, match=message):
            CorpusManifest(**(base | update))

    runtime = RuntimePin(
        meshprobe_version="test",
        meshprobe_sha256="b" * 64,
        blender_version="test",
        importer_sha256="d" * 64,
        render_engines=("eevee",),
    )
    tier: dict[str, Any] = {
        "tier": CorpusTier.SMOKE,
        "corpus_version": "test-v1",
        "corpus_manifest_sha256": "e" * 64,
        "generator_sha256": "a" * 64,
        "runtime": runtime,
        "thresholds": PassThresholds(overall=0, full_stack=0, per_operation=0),
        "model_sha256": {"model.glb": "b" * 64},
    }
    with pytest.raises(ValidationError, match="unique"):
        TierManifest(
            **tier,
            episodes=("episode_0001", "episode_0001"),
            episode_sha256={"episode_0001": "c" * 64},
        )
    with pytest.raises(ValidationError, match="exactly cover"):
        TierManifest(
            **tier,
            episodes=("episode_0001",),
            episode_sha256={"other_ep": "c" * 64},
        )
