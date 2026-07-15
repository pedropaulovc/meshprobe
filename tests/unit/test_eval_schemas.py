from __future__ import annotations

import pytest
from pydantic import ValidationError

from meshprobe.evals.schemas import (
    AnswerStatus,
    Difficulty,
    EpisodeClass,
    EpisodeGroundTruth,
    EpisodeSpec,
    Operation,
    StructuredAnswer,
    TaskFamily,
)


def test_full_investigation_requires_every_public_operation() -> None:
    with pytest.raises(ValidationError, match="every public operation"):
        EpisodeSpec(
            episode_id="episode_0001",
            model_file="fixture.glb",
            model_sha256="a" * 64,
            family=TaskFamily.FULL_INVESTIGATION,
            episode_class=EpisodeClass.POSITIVE,
            difficulty=Difficulty.FULL_STACK,
            prompt="Inspect the complete assembly and submit all requested evidence.",
            answer_schema={"type": "object"},
            required_operations=(Operation.SCENE_OPEN, Operation.SCENE_DESCRIBE),
        )


def test_indeterminate_answer_requires_an_explanation() -> None:
    with pytest.raises(ValidationError, match="explain"):
        StructuredAnswer(status=AnswerStatus.INDETERMINATE)


def test_ground_truth_rejects_mismatched_role_maps() -> None:
    with pytest.raises(ValidationError, match="same roles"):
        EpisodeGroundTruth(
            episode_id="episode_0001",
            model_sha256="a" * 64,
            answer=StructuredAnswer(status=AnswerStatus.ANSWERED, values={"side": "positive"}),
            component_roles={"target": "cmp_abc"},
            component_paths={},
            required_operations=(Operation.SCENE_OPEN,),
        )
