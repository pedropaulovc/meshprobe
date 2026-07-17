from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

import meshprobe.evals.oracles as oracles
from meshprobe.camera import camera_diagnostics
from meshprobe.evals.generators import (
    GeneratorFamily,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.oracles import OracleInputs, score_episode
from meshprobe.evals.schemas import (
    AnswerStatus,
    EpisodeSubmission,
    EvidenceKind,
    GateStatus,
    Operation,
    StatePredicate,
    StateRequirement,
    TraceEvent,
    TraceStatus,
)
from meshprobe.models import (
    Camera,
    ComponentVisualState,
    DisplayMode,
    EvaluatorPasses,
    IlluminationPreset,
    ImageArtifact,
    LuminanceSummary,
    MarkMode,
    OrthographicProjection,
    PerspectiveProjection,
    Pose,
    PresetIllumination,
    RenderEngine,
    RenderManifest,
    SessionSnapshot,
)
from meshprobe.sources import sha256_file, snapshot_source


def accepted_event(
    sequence: int,
    operation: str,
    *,
    arguments: object,
    result: object,
    before: str | None = None,
    after: str | None = None,
) -> TraceEvent:
    return TraceEvent.model_validate(
        {
            "sequence": sequence,
            "request_id": f"request-{sequence}",
            "operation": operation,
            "status": TraceStatus.ACCEPTED,
            "arguments": arguments,
            "result": result,
            "state_before_sha256": before,
            "state_after_sha256": after,
            "started_monotonic": float(sequence),
            "elapsed_seconds": 0.1,
        }
    )


@pytest.mark.parametrize(
    ("case", "transitions", "expected", "final_state", "saw_transition"),
    [
        (
            "initial None",
            [(Operation.VIEW_ORBIT, None, "1" * 64)],
            True,
            "1" * 64,
            True,
        ),
        (
            "compact first",
            [(Operation.VIEW_SET, "0" * 64, "1" * 64)],
            True,
            "1" * 64,
            True,
        ),
        (
            "reset",
            [
                (Operation.VIEW_SET, "0" * 64, "1" * 64),
                (Operation.SESSION_RESET, "1" * 64, "0" * 64),
            ],
            True,
            "0" * 64,
            True,
        ),
        (
            "repeated same hash",
            [
                (Operation.VIEW_SET, "0" * 64, "1" * 64),
                (Operation.ILLUMINATION_SET, "1" * 64, "1" * 64),
            ],
            True,
            "1" * 64,
            True,
        ),
        (
            "branch mismatch",
            [
                (Operation.VIEW_SET, "0" * 64, "1" * 64),
                (Operation.ILLUMINATION_SET, "9" * 64, "2" * 64),
            ],
            False,
            "1" * 64,
            True,
        ),
        (
            "no-op target",
            [
                (Operation.VIEW_SET, "0" * 64, "1" * 64),
                (Operation.COMPONENT_MARK, "1" * 64, "1" * 64),
            ],
            True,
            "1" * 64,
            True,
        ),
        (
            "view.rotate",
            [(Operation.VIEW_ROTATE, "0" * 64, "1" * 64)],
            True,
            "1" * 64,
            True,
        ),
    ],
)
def test_compact_state_replay_transition_matrix(
    case: str,
    transitions: list[tuple[Operation, str | None, str]],
    expected: bool,
    final_state: str,
    saw_transition: bool,
) -> None:
    del case
    replay = oracles._CompactStateReplay()
    result = True
    for sequence, (operation, before, after) in enumerate(transitions, start=1):
        event_result: dict[str, object] = {"state_sha256": after}
        if operation is Operation.SESSION_RESET:
            event_result["reset"] = True
        event = accepted_event(
            sequence,
            operation,
            arguments={},
            result=event_result,
            before=before,
            after=after,
        )
        if replay.apply(event):
            continue
        result = False
        break

    assert result is expected
    assert replay.current_state_sha256 == final_state
    assert replay.saw_transition is saw_transition


def test_compact_state_chain_requires_every_grouped_predicate() -> None:
    state_before = "0" * 64
    state_after = "1" * 64
    missing_camera = accepted_event(
        1,
        Operation.VIEW_ROTATE,
        arguments={},
        result={"state_sha256": state_after},
        before=state_before,
        after=state_after,
    )
    requirement = StateRequirement(
        predicate=StatePredicate.PROJECTION_MODE,
        expected="orthographic",
    )

    assert not oracles._transition_chain_satisfies_state_group(
        [missing_camera], missing_camera, [requirement], {}
    )

    complete = missing_camera.model_copy(
        update={
            "result": {
                "state_sha256": state_after,
                "camera": {"projection": {"mode": "orthographic"}},
            }
        }
    )
    assert oracles._transition_chain_satisfies_state_group([complete], complete, [requirement], {})


def test_compact_state_chain_rejects_unproven_earlier_delta() -> None:
    camera_state = "1" * 64
    rendered_state = "2" * 64
    camera = accepted_event(
        1,
        Operation.VIEW_SET,
        arguments={},
        result={
            "state_sha256": "f" * 64,
            "camera": {"projection": {"mode": "orthographic"}},
        },
        before=camera_state,
        after=camera_state,
    )
    illumination = accepted_event(
        2,
        Operation.ILLUMINATION_SET,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "illumination": {"preset": "neutral_studio"},
        },
        before=camera_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
    ]

    assert not oracles._transition_chain_satisfies_state_group(
        [camera, illumination], illumination, requirements, {}
    )

    proven_camera = camera.model_copy(
        update={
            "result": {
                "state_sha256": camera_state,
                "camera": {"projection": {"mode": "orthographic"}},
            }
        }
    )
    assert oracles._transition_chain_satisfies_state_group(
        [proven_camera, illumination], illumination, requirements, {}
    )


def test_compact_state_chain_rejects_discontinuous_hashes() -> None:
    camera_state = "1" * 64
    rendered_state = "2" * 64
    camera = accepted_event(
        1,
        Operation.VIEW_ROTATE,
        arguments={},
        result={
            "state_sha256": camera_state,
            "camera": {"projection": {"mode": "orthographic"}},
        },
        before="0" * 64,
        after=camera_state,
    )
    illumination = accepted_event(
        2,
        Operation.ILLUMINATION_SET,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "illumination": {"preset": "neutral_studio"},
        },
        before="9" * 64,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
    ]

    assert not oracles._transition_chain_satisfies_state_group(
        [camera, illumination], illumination, requirements, {}
    )

    continuous = illumination.model_copy(update={"state_before_sha256": camera_state})
    assert oracles._transition_chain_satisfies_state_group(
        [camera, continuous], continuous, requirements, {}
    )


def test_compact_state_chain_accepts_hash_proven_noop_target() -> None:
    rendered_state = "1" * 64
    camera = accepted_event(
        1,
        Operation.VIEW_ORBIT,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "camera": {"projection": {"mode": "orthographic"}},
        },
        before="0" * 64,
        after=rendered_state,
    )
    illumination = accepted_event(
        2,
        Operation.ILLUMINATION_SET,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "illumination": {"preset": "neutral_studio"},
        },
        before=rendered_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
    ]

    assert oracles._transition_chain_satisfies_state_group(
        [camera, illumination], illumination, requirements, {}
    )


def test_compact_state_chain_restores_imported_state_on_reset() -> None:
    imported_state = "0" * 64
    changed_state = "1" * 64
    rendered_state = "2" * 64
    target = "cmp_target"
    snapshot = accepted_event(
        1,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": imported_state,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {target: {"display": "shown", "mark": "unmarked"}},
            }
        },
        before=imported_state,
        after=imported_state,
    )
    changed = accepted_event(
        2,
        Operation.VIEW_SET,
        arguments={},
        result={
            "state_sha256": changed_state,
            "camera": {"projection": {"mode": "perspective"}},
        },
        before=imported_state,
        after=changed_state,
    )
    reset = accepted_event(
        3,
        Operation.SESSION_RESET,
        arguments={},
        result={"reset": True, "state_sha256": imported_state},
        before=changed_state,
        after=imported_state,
    )
    marked = accepted_event(
        4,
        Operation.COMPONENT_MARK,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "components": {target: {"display": "shown", "mark": "highlighted"}},
        },
        before=imported_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_MARK,
            component_role="target",
            expected="highlighted",
        ),
    ]

    assert oracles._transition_chain_satisfies_state_group(
        [snapshot, changed, reset, marked],
        marked,
        requirements,
        {"target": target},
    )


def test_compact_state_chain_starts_from_proven_session_snapshot() -> None:
    imported_state = "0" * 64
    rendered_state = "1" * 64
    target = "cmp_target"
    snapshot = accepted_event(
        1,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": imported_state,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {target: {"display": "shown", "mark": "unmarked"}},
            }
        },
        before=imported_state,
        after=imported_state,
    )
    marked = accepted_event(
        2,
        Operation.COMPONENT_MARK,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "components": {target: {"display": "shown", "mark": "highlighted"}},
        },
        before=imported_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_MARK,
            component_role="target",
            expected="highlighted",
        ),
    ]

    assert oracles._transition_chain_satisfies_state_group(
        [snapshot, marked], marked, requirements, {"target": target}
    )


@pytest.mark.parametrize(
    ("before", "after", "embedded", "expected"),
    [
        pytest.param(None, "1" * 64, "1" * 64, True, id="initial-bootstrap"),
        pytest.param("1" * 64, "1" * 64, "1" * 64, True, id="known-noop"),
        pytest.param(None, None, "1" * 64, False, id="missing-after"),
        pytest.param("1" * 64, None, "1" * 64, False, id="lost-after"),
        pytest.param("1" * 64, "2" * 64, "2" * 64, False, id="known-mutation"),
        pytest.param(None, "1" * 64, "2" * 64, False, id="bootstrap-payload-mismatch"),
        pytest.param("1" * 64, "1" * 64, "2" * 64, False, id="noop-payload-mismatch"),
    ],
)
def test_replayable_session_snapshot_boundary_matrix(
    before: str | None,
    after: str | None,
    embedded: str,
    expected: bool,
) -> None:
    snapshot = accepted_event(
        1,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": embedded,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {},
            }
        },
        before=before,
        after=after,
    )

    assert (oracles._replayable_session_state(snapshot) is not None) is expected


def test_compact_state_chain_starts_from_initial_bootstrap_snapshot() -> None:
    imported_state = "0" * 64
    rendered_state = "1" * 64
    target = "cmp_target"
    opened = accepted_event(
        1,
        Operation.SCENE_OPEN,
        arguments={},
        result={"source_sha256": "f" * 64},
        before=None,
        after=None,
    )
    snapshot = accepted_event(
        2,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": imported_state,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {target: {"display": "shown", "mark": "unmarked"}},
            }
        },
        before=None,
        after=imported_state,
    )
    marked = accepted_event(
        3,
        Operation.COMPONENT_MARK,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "components": {target: {"display": "shown", "mark": "highlighted"}},
        },
        before=imported_state,
        after=rendered_state,
    )
    rendered = accepted_event(
        4,
        Operation.RENDER_IMAGE,
        arguments={},
        result={"state_sha256": rendered_state},
        before=rendered_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_MARK,
            component_role="target",
            expected="highlighted",
        ),
    ]

    assert oracles._transition_chain_satisfies_state_group(
        [opened, snapshot, marked, rendered],
        marked,
        requirements,
        {"target": target},
    )


def test_compact_state_replay_does_not_reapply_bootstrap_after_transition() -> None:
    imported_state = "0" * 64
    changed_state = "1" * 64
    initial = accepted_event(
        1,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": imported_state,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {},
            }
        },
        before=None,
        after=imported_state,
    )
    changed = accepted_event(
        2,
        Operation.COMPONENT_MARK,
        arguments={},
        result={"state_sha256": changed_state, "components": {}},
        before=imported_state,
        after=changed_state,
    )
    late_bootstrap = accepted_event(
        3,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": changed_state,
                "camera": {"projection": {"mode": "perspective"}},
                "illumination": {"preset": "raking_left"},
                "components": {},
            }
        },
        before=None,
        after=changed_state,
    )
    replay = oracles._CompactStateReplay()

    replay.observe_snapshot(initial)
    assert replay.apply(changed)
    replay.observe_snapshot(late_bootstrap)

    assert replay.state["camera"] == {"projection": {"mode": "orthographic"}}
    assert replay.state["illumination"] == {"preset": "neutral_studio"}


def test_compact_state_chain_allows_first_transition_without_before_hash() -> None:
    camera_state = "1" * 64
    target = "cmp_target"
    camera = accepted_event(
        1,
        Operation.VIEW_ORBIT,
        arguments={},
        result={
            "state_sha256": camera_state,
            "camera": {"projection": {"mode": "orthographic"}},
        },
        before=None,
        after=camera_state,
    )
    late_snapshot = accepted_event(
        2,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": camera_state,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {target: {"display": "shown", "mark": "unmarked"}},
            }
        },
        before=camera_state,
        after=camera_state,
    )
    illumination = accepted_event(
        3,
        Operation.ILLUMINATION_SET,
        arguments={},
        result={
            "state_sha256": camera_state,
            "illumination": {"preset": "neutral_studio"},
        },
        before=camera_state,
        after=camera_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_DISPLAY,
            component_role="target",
            expected="shown",
        ),
    ]

    assert oracles._transition_chain_satisfies_state_group(
        [camera, late_snapshot, illumination],
        illumination,
        requirements,
        {"target": target},
    )


def test_compact_state_chain_allows_reset_without_cached_snapshot() -> None:
    changed_state = "9" * 64
    imported_state = "0" * 64
    camera_state = "1" * 64
    rendered_state = "2" * 64
    reset = accepted_event(
        1,
        Operation.SESSION_RESET,
        arguments={},
        result={"reset": True, "state_sha256": imported_state},
        before=changed_state,
        after=imported_state,
    )
    camera = accepted_event(
        2,
        Operation.VIEW_SET,
        arguments={},
        result={
            "state_sha256": camera_state,
            "camera": {"projection": {"mode": "orthographic"}},
        },
        before=imported_state,
        after=camera_state,
    )
    illumination = accepted_event(
        3,
        Operation.ILLUMINATION_SET,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "illumination": {"preset": "neutral_studio"},
        },
        before=camera_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
    ]

    assert oracles._transition_chain_satisfies_state_group(
        [reset, camera, illumination], illumination, requirements, {}
    )


def test_compact_state_chain_refreshes_from_snapshot_after_target() -> None:
    rendered_state = "1" * 64
    target = "cmp_target"
    camera = accepted_event(
        1,
        Operation.VIEW_ORBIT,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "camera": {"projection": {"mode": "orthographic"}},
        },
        before=None,
        after=rendered_state,
    )
    snapshot = accepted_event(
        2,
        Operation.SESSION_SNAPSHOT,
        arguments={},
        result={
            "session": {
                "state_sha256": rendered_state,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {target: {"display": "shown", "mark": "unmarked"}},
            }
        },
        before=rendered_state,
        after=rendered_state,
    )
    render = accepted_event(
        3,
        Operation.RENDER_IMAGE,
        arguments={},
        result={"state_sha256": rendered_state},
        before=rendered_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_DISPLAY,
            component_role="target",
            expected="shown",
        ),
    ]

    assert oracles._transition_chain_satisfies_state_group(
        [camera, snapshot, render], camera, requirements, {"target": target}
    )


def test_compact_state_chain_does_not_seed_from_render_manifest() -> None:
    camera_state = "1" * 64
    rendered_state = "2" * 64
    target = "cmp_target"
    camera = accepted_event(
        1,
        Operation.VIEW_ORBIT,
        arguments={},
        result={
            "state_sha256": camera_state,
            "camera": {"projection": {"mode": "orthographic"}},
        },
        before="0" * 64,
        after=camera_state,
    )
    render = accepted_event(
        2,
        Operation.RENDER_IMAGE,
        arguments={},
        result={
            "state_sha256": camera_state,
            "session": {
                "state_sha256": camera_state,
                "camera": {"projection": {"mode": "orthographic"}},
                "illumination": {"preset": "neutral_studio"},
                "components": {target: {"display": "shown", "mark": "unmarked"}},
            },
        },
        before=camera_state,
        after=camera_state,
    )
    marked = accepted_event(
        3,
        Operation.COMPONENT_MARK,
        arguments={},
        result={
            "state_sha256": rendered_state,
            "components": {target: {"display": "shown", "mark": "highlighted"}},
        },
        before=camera_state,
        after=rendered_state,
    )
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_MARK,
            component_role="target",
            expected="highlighted",
        ),
    ]

    assert not oracles._transition_chain_satisfies_state_group(
        [camera, render, marked], marked, requirements, {"target": target}
    )


def discovery_inputs(tmp_path: Path) -> OracleInputs:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    episode = generate_episodes(model)[0]
    target = model.component_ids["idler"]
    state = "a" * 64
    trace = (
        accepted_event(1, "scene.open", arguments={}, result={}),
        accepted_event(
            2,
            "session.snapshot",
            arguments={},
            result={"session": {"state_sha256": state}},
            before=state,
            after=state,
        ),
        accepted_event(
            3,
            "component.find",
            arguments={"selector": {"kind": "exact_name", "pattern": model.generated.target_name}},
            result=[{"id": target}],
            before=state,
            after=state,
        ),
        accepted_event(
            4,
            "component.inspect",
            arguments={"component_id": target},
            result={"id": target},
            before=state,
            after=state,
        ),
    )
    snapshot = snapshot_source(model.path)
    submission = EpisodeSubmission(
        episode_id=episode.spec.episode_id,
        answer=episode.ground_truth.answer,
    )
    return OracleInputs(
        spec=episode.spec,
        truth=episode.ground_truth,
        submission=submission,
        trace=trace,
        source_before=snapshot,
        source_after=snapshot,
    )


def test_discovery_episode_passes_answer_coverage_and_read_only_gates(tmp_path: Path) -> None:
    report = score_episode(discovery_inputs(tmp_path))
    by_gate = {gate.gate: gate for gate in report.gates}

    assert report.passed
    assert by_gate["answer"].status is GateStatus.PASS
    assert by_gate["coverage"].status is GateStatus.PASS
    assert by_gate["read_only"].status is GateStatus.PASS
    assert by_gate["evidence"].status is GateStatus.NOT_APPLICABLE


def test_coverage_rejects_a_find_call_that_missed_the_target(tmp_path: Path) -> None:
    inputs = discovery_inputs(tmp_path)
    missed = inputs.trace[2].model_copy(update={"result": []})
    report = score_episode(replace(inputs, trace=(*inputs.trace[:2], missed, inputs.trace[3])))

    assert next(gate for gate in report.gates if gate.gate == "coverage").status is GateStatus.FAIL


def test_coverage_uses_curated_target_role(tmp_path: Path) -> None:
    inputs = discovery_inputs(tmp_path)
    target = inputs.truth.component_roles["idler"]
    curated_truth = inputs.truth.model_copy(
        update={"component_roles": {"target": target, "reference": "reference"}}
    )
    missed = inputs.trace[2].model_copy(update={"result": []})
    report = score_episode(
        replace(
            inputs,
            truth=curated_truth,
            trace=(*inputs.trace[:2], missed, inputs.trace[3]),
        )
    )

    assert next(gate for gate in report.gates if gate.gate == "coverage").status is GateStatus.FAIL


def test_coverage_counts_first_observed_state_transition(tmp_path: Path) -> None:
    inputs = discovery_inputs(tmp_path)
    truth = inputs.truth.model_copy(
        update={"required_operations": (*inputs.truth.required_operations, Operation.VIEW_SET)}
    )
    first_state = accepted_event(
        5,
        "view.set",
        arguments={},
        result={"state_sha256": "b" * 64},
        before=None,
        after="b" * 64,
    )
    report = score_episode(replace(inputs, truth=truth, trace=(*inputs.trace, first_state)))

    assert next(gate for gate in report.gates if gate.gate == "coverage").status is GateStatus.PASS


def test_answer_gate_rejects_plausible_but_wrong_values(tmp_path: Path) -> None:
    inputs = discovery_inputs(tmp_path)
    wrong = inputs.submission.model_copy(
        update={
            "answer": inputs.submission.answer.model_copy(
                update={"status": AnswerStatus.ANSWERED, "values": {"contacted_shaft": 99}}
            )
        }
    )
    report = score_episode(replace(inputs, submission=wrong))

    assert next(gate for gate in report.gates if gate.gate == "answer").status is GateStatus.FAIL


def test_read_only_gate_compares_bundle_metadata_and_hashes(tmp_path: Path) -> None:
    inputs = discovery_inputs(tmp_path)
    changed = inputs.source_after.assets[0]
    modified_snapshot = inputs.source_after.__class__(
        assets=(changed.__class__(**{**changed.__dict__, "mtime_ns": changed.mtime_ns + 1}),),
        sha256=inputs.source_after.sha256,
    )
    report = score_episode(replace(inputs, source_after=modified_snapshot))

    assert next(gate for gate in report.gates if gate.gate == "read_only").status is GateStatus.FAIL


def image_artifact(path: Path, media_type: str = "image/png") -> ImageArtifact:
    return ImageArtifact.model_validate(
        {
            "path": str(path),
            "media_type": media_type,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    )


def evidence_render(
    tmp_path: Path,
    *,
    name: str,
    source_sha256: str,
    component_ids: dict[str, str],
    projection: PerspectiveProjection | OrthographicProjection,
    illumination: IlluminationPreset,
    visible_roles: set[str],
    isolated_role: str | None = None,
) -> RenderManifest:
    color_path = tmp_path / f"{name}.png"
    mask_path = tmp_path / f"{name}.components.png"
    highlight_path = tmp_path / f"{name}.highlighted.png"
    exr_path = tmp_path / f"{name}.exr"
    color = Image.new("RGB", (64, 64), (20, 20, 20))
    mask = Image.new("RGB", (64, 64), (0, 0, 0))
    highlighted = Image.new("RGB", (64, 64), (0, 0, 0))
    if "idler" in visible_roles:
        for y in range(16, 48):
            for x in range(16, 48):
                color.putpixel((x, y), (40 + (x - 16) * 6, 60, 80))
                mask.putpixel((x, y), (240, 30, 20))
                highlighted.putpixel((x, y), (255, 255, 255))
    if "arrow" in visible_roles:
        for y in range(4, 12):
            for x in range(4, 12):
                color.putpixel((x, y), (20 + (x - 4) * 25, 30, 40))
                mask.putpixel((x, y), (20, 220, 40))
    for role, x_range, mask_color in (
        ("gap_left", range(4, 12), (30, 80, 230)),
        ("gap_right", range(20, 28), (230, 180, 30)),
    ):
        if role not in visible_roles:
            continue
        for y in range(50, 58):
            for x in x_range:
                color.putpixel((x, y), mask_color)
                mask.putpixel((x, y), mask_color)
    color.save(color_path)
    mask.save(mask_path)
    highlighted.save(highlight_path)
    exr_path.write_bytes(b"private evaluator pass")
    state_hash = hashlib.sha256(name.encode()).hexdigest()
    camera = Camera(
        pose=Pose(position_mm=(3_000, -4_000, 2_000), orientation_xyzw=(0, 0, 0, 1)),
        projection=projection,
    )
    session = SessionSnapshot(
        camera=camera,
        camera_diagnostics=camera_diagnostics(camera, target_mm=(0, 0, 0)),
        illumination=PresetIllumination(preset=illumination),
        components={
            component_id: ComponentVisualState(
                display=(
                    DisplayMode.ISOLATED
                    if role == isolated_role
                    else DisplayMode.HIDDEN
                    if isolated_role is not None or role == "cover"
                    else DisplayMode.SHOWN
                ),
                mark=(
                    MarkMode.HIGHLIGHTED
                    if role == "idler" and "idler" in visible_roles
                    else MarkMode.UNMARKED
                ),
            )
            for role, component_id in component_ids.items()
        },
        state_sha256=state_hash,
    )
    return RenderManifest(
        source_sha256=source_sha256,
        state_sha256=state_hash,
        width=64,
        height=64,
        samples=1,
        engine=RenderEngine.EEVEE,
        device="graphics_hardware",
        graphics_policy="software_allowed",
        graphics={
            "vendor": "Test",
            "renderer": "Test GPU",
            "version": "1.0",
            "backend": "OPENGL",
            "blender_device_type": "NVIDIA",
            "device_class": "hardware",
        },
        blender_version="test",
        session=session,
        color=image_artifact(color_path),
        evaluator=EvaluatorPasses(
            multilayer=image_artifact(exr_path, "image/x-exr"),
            component_ids=image_artifact(mask_path),
            highlighted=image_artifact(highlight_path),
            component_colors={
                component_ids["idler"]: (240, 30, 20),
                component_ids["arrow"]: (20, 220, 40),
                component_ids["gap_left"]: (30, 80, 230),
                component_ids["gap_right"]: (230, 180, 30),
            },
        ),
        luminance=LuminanceSummary(
            minimum=0.1,
            median=0.3,
            maximum=0.8,
            crushed_fraction=0,
            clipped_fraction=0,
        ),
    )


def test_target_screen_span_rejects_a_target_that_is_too_small(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    render = evidence_render(
        tmp_path,
        name="target-span",
        source_sha256=model.sha256,
        component_ids=model.component_ids,
        projection=PerspectiveProjection(focal_length_mm=85),
        illumination=IlluminationPreset.NEUTRAL_STUDIO,
        visible_roles={"idler"},
    )
    target = model.component_ids["idler"]

    assert oracles._component_screen_span(render, target) == 0.5
    assert oracles._evidence_satisfied(
        EvidenceKind.TARGET_SCREEN_SPAN,
        None,
        0.5,
        None,
        target,
        (render,),
        (),
    )
    assert not oracles._evidence_satisfied(
        EvidenceKind.TARGET_SCREEN_SPAN,
        None,
        0.6,
        None,
        target,
        (render,),
        (),
    )


def break_state_predicate(
    payload: object,
    predicate: StatePredicate,
    component_id: str | None,
) -> object:
    assert isinstance(payload, dict)
    broken = deepcopy(payload)
    if predicate is StatePredicate.PROJECTION_MODE:
        if "camera" not in broken:
            return payload
        broken["camera"]["projection"]["mode"] = "invalidated"
    elif predicate is StatePredicate.FOCAL_LENGTH_MM:
        if "camera" not in broken:
            return payload
        broken["camera"]["projection"]["focal_length_mm"] = 17.0
    elif predicate is StatePredicate.ILLUMINATION_PRESET:
        if "illumination" not in broken:
            return payload
        broken["illumination"]["preset"] = "invalidated"
    elif predicate is StatePredicate.COMPONENT_DISPLAY:
        assert component_id is not None
        if component_id not in broken.get("components", {}):
            return payload
        broken["components"][component_id]["display"] = "shown"
    elif predicate is StatePredicate.COMPONENT_MARK:
        assert component_id is not None
        if component_id not in broken.get("components", {}):
            return payload
        broken["components"][component_id]["mark"] = "unmarked"
    else:
        raise AssertionError(predicate)
    return broken


def test_private_masks_and_render_state_satisfy_evidence_gate(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    episode = generate_episodes(model)[1]
    initial_hash = "0" * 64
    perspective = evidence_render(
        tmp_path,
        name="perspective",
        source_sha256=model.sha256,
        component_ids=model.component_ids,
        projection=PerspectiveProjection(focal_length_mm=85),
        illumination=IlluminationPreset.NEUTRAL_STUDIO,
        visible_roles={"idler"},
    )
    raking_left = evidence_render(
        tmp_path,
        name="raking-left",
        source_sha256=model.sha256,
        component_ids=model.component_ids,
        projection=OrthographicProjection(scale_mm=5_000),
        illumination=IlluminationPreset.RAKING_LEFT,
        visible_roles={"arrow"},
        isolated_role="arrow",
    )
    raking_right = evidence_render(
        tmp_path,
        name="raking-right",
        source_sha256=model.sha256,
        component_ids=model.component_ids,
        projection=OrthographicProjection(scale_mm=5_000),
        illumination=IlluminationPreset.RAKING_RIGHT,
        visible_roles={"arrow"},
        isolated_role="arrow",
    )
    backlit = evidence_render(
        tmp_path,
        name="backlit",
        source_sha256=model.sha256,
        component_ids=model.component_ids,
        projection=OrthographicProjection(scale_mm=5_000),
        illumination=IlluminationPreset.BACKLIT,
        visible_roles={"gap_left", "gap_right"},
    )
    target = model.component_ids["idler"]
    trace = (
        accepted_event(1, "scene.open", arguments={}, result={}),
        accepted_event(
            2,
            "session.snapshot",
            arguments={},
            result={"session": {"state_sha256": initial_hash}},
            before=initial_hash,
            after=initial_hash,
        ),
        accepted_event(3, "component.find", arguments={}, result=[{"id": target}]),
        accepted_event(
            4, "component.inspect", arguments={"component_id": target}, result={"id": target}
        ),
        accepted_event(
            5,
            "component.display",
            arguments={"component_ids": [model.component_ids["cover"]], "mode": "hidden"},
            result={
                "components": {
                    model.component_ids["cover"]: {"display": "hidden", "mark": "unmarked"}
                },
                "state_sha256": "1" * 64,
            },
            before=initial_hash,
            after="1" * 64,
        ),
        accepted_event(
            6,
            "component.mark",
            arguments={"component_ids": [target], "mode": "highlighted"},
            result={
                "components": {target: {"display": "shown", "mark": "highlighted"}},
                "state_sha256": "2" * 64,
            },
            before="1" * 64,
            after="2" * 64,
        ),
        accepted_event(
            7,
            "illumination.set",
            arguments={},
            result={
                "illumination": perspective.session.illumination.model_dump(mode="json"),
                "state_sha256": "3" * 64,
            },
            before="2" * 64,
            after="3" * 64,
        ),
        accepted_event(
            8,
            "view.orbit",
            arguments={},
            result={
                "camera": perspective.session.camera.model_dump(mode="json"),
                "state_sha256": perspective.state_sha256,
            },
            before="3" * 64,
            after=perspective.state_sha256,
        ),
        accepted_event(
            9,
            "view.set",
            arguments={},
            result={
                "camera": raking_left.session.camera.model_dump(mode="json"),
                "state_sha256": "4" * 64,
            },
            before=perspective.state_sha256,
            after="4" * 64,
        ),
        accepted_event(
            10,
            "component.display",
            arguments={"component_ids": [model.component_ids["arrow"]], "mode": "isolated"},
            result={
                "components": {
                    model.component_ids["arrow"]: {"display": "isolated", "mark": "unmarked"}
                },
                "state_sha256": "5" * 64,
            },
            before="4" * 64,
            after="5" * 64,
        ),
        accepted_event(
            11,
            "illumination.set",
            arguments={},
            result={
                "illumination": raking_left.session.illumination.model_dump(mode="json"),
                "state_sha256": raking_left.state_sha256,
            },
            before="5" * 64,
            after=raking_left.state_sha256,
        ),
        accepted_event(
            12,
            "illumination.set",
            arguments={},
            result={
                "illumination": raking_right.session.illumination.model_dump(mode="json"),
                "state_sha256": raking_right.state_sha256,
            },
            before=raking_left.state_sha256,
            after=raking_right.state_sha256,
        ),
        accepted_event(
            13,
            "illumination.set",
            arguments={},
            result={
                "illumination": backlit.session.illumination.model_dump(mode="json"),
                "state_sha256": backlit.state_sha256,
            },
            before=raking_right.state_sha256,
            after=backlit.state_sha256,
        ),
        accepted_event(
            14,
            "render.image",
            arguments={},
            result=perspective.model_dump(mode="json"),
            before=perspective.state_sha256,
            after=perspective.state_sha256,
        ),
        accepted_event(
            15,
            "render.image",
            arguments={},
            result=raking_left.model_dump(mode="json"),
            before=raking_left.state_sha256,
            after=raking_left.state_sha256,
        ),
        accepted_event(
            16,
            "render.image",
            arguments={},
            result=raking_right.model_dump(mode="json"),
            before=raking_right.state_sha256,
            after=raking_right.state_sha256,
        ),
        accepted_event(
            17,
            "render.image",
            arguments={},
            result=backlit.model_dump(mode="json"),
            before=backlit.state_sha256,
            after=backlit.state_sha256,
        ),
    )
    snapshot = snapshot_source(model.path)
    submission = EpisodeSubmission(
        episode_id=episode.spec.episode_id,
        answer=episode.ground_truth.answer,
        evidence_manifest_paths=tuple(
            render.color.path for render in (perspective, raking_left, raking_right, backlit)
        ),
    )
    report = score_episode(
        OracleInputs(
            spec=episode.spec,
            truth=episode.ground_truth,
            submission=submission,
            trace=trace,
            source_before=snapshot,
            source_after=snapshot,
            artifact_root=tmp_path,
            renders=(perspective, raking_left, raking_right, backlit),
        )
    )

    assert next(gate for gate in report.gates if gate.gate == "state").status is GateStatus.PASS
    assert next(gate for gate in report.gates if gate.gate == "evidence").status is GateStatus.PASS
    missing_submission = score_episode(
        replace(
            OracleInputs(
                spec=episode.spec,
                truth=episode.ground_truth,
                submission=submission.model_copy(update={"evidence_manifest_paths": ()}),
                trace=trace,
                source_before=snapshot,
                source_after=snapshot,
                artifact_root=tmp_path,
                renders=(perspective, raking_left, raking_right, backlit),
            )
        )
    )
    missing_evidence = next(gate for gate in missing_submission.gates if gate.gate == "evidence")
    assert missing_evidence.status is GateStatus.FAIL
    detached_trace = tuple(
        event.model_copy(update={"state_after_sha256": "f" * 64})
        if event.operation is not Operation.RENDER_IMAGE
        and event.state_after_sha256 == perspective.state_sha256
        else event
        for event in trace
    )
    detached_report = score_episode(
        OracleInputs(
            spec=episode.spec,
            truth=episode.ground_truth,
            submission=submission,
            trace=detached_trace,
            source_before=snapshot,
            source_after=snapshot,
            artifact_root=tmp_path,
            renders=(perspective, raking_left, raking_right, backlit),
        )
    )
    detached_state = next(gate for gate in detached_report.gates if gate.gate == "state")
    assert detached_state.status is GateStatus.FAIL

    group_state_hashes = {
        "context_85": perspective.state_sha256,
        "surface_left": raking_left.state_sha256,
        "surface_right": raking_right.state_sha256,
        "gap_backlit": backlit.state_sha256,
    }
    for group_name, state_hash in group_state_hashes.items():
        reduced_trace = tuple(
            event
            for event in trace
            if not (
                isinstance(event.result, dict) and event.result.get("state_sha256") == state_hash
            )
        )
        reduced_report = score_episode(
            OracleInputs(
                spec=episode.spec,
                truth=episode.ground_truth,
                submission=submission,
                trace=reduced_trace,
                source_before=snapshot,
                source_after=snapshot,
                artifact_root=tmp_path,
                renders=(perspective, raking_left, raking_right, backlit),
            )
        )
        state_gate = next(gate for gate in reduced_report.gates if gate.gate == "state")
        assert state_gate.status is GateStatus.FAIL, group_name

    for requirement in episode.ground_truth.state_requirements:
        if requirement.state_group is None:
            continue
        state_hash = group_state_hashes[requirement.state_group]
        component_id = (
            model.component_ids[requirement.component_role]
            if requirement.component_role is not None
            else None
        )
        broken_trace = tuple(
            event.model_copy(
                update={
                    "result": break_state_predicate(
                        event.result,
                        requirement.predicate,
                        component_id,
                    )
                }
            )
            if isinstance(event.result, dict)
            and event.result.get("state_sha256") == state_hash
            and event.operation is not Operation.RENDER_IMAGE
            else event
            for event in trace
        )
        if broken_trace == trace:
            continue
        broken_report = score_episode(
            OracleInputs(
                spec=episode.spec,
                truth=episode.ground_truth,
                submission=submission,
                trace=broken_trace,
                source_before=snapshot,
                source_after=snapshot,
                artifact_root=tmp_path,
                renders=(perspective, raking_left, raking_right, backlit),
            )
        )
        state_gate = next(gate for gate in broken_report.gates if gate.gate == "state")
        assert state_gate.status is GateStatus.FAIL, requirement

    perspective_raking_session = perspective.session.model_copy(
        update={
            "illumination": PresetIllumination(preset=IlluminationPreset.RAKING_LEFT),
            "state_sha256": "a" * 64,
        }
    )
    perspective_raking = perspective.model_copy(
        update={
            "state_sha256": "a" * 64,
            "session": perspective_raking_session,
        }
    )
    neutral_orthographic_session = raking_left.session.model_copy(
        update={
            "illumination": PresetIllumination(preset=IlluminationPreset.NEUTRAL_STUDIO),
            "state_sha256": "b" * 64,
        }
    )
    neutral_orthographic = raking_left.model_copy(
        update={
            "state_sha256": "b" * 64,
            "session": neutral_orthographic_session,
        }
    )
    split_report = score_episode(
        replace(
            OracleInputs(
                spec=episode.spec,
                truth=episode.ground_truth,
                submission=submission,
                trace=trace,
                source_before=snapshot,
                source_after=snapshot,
                artifact_root=tmp_path,
                renders=(
                    perspective_raking,
                    neutral_orthographic,
                    raking_left,
                    raking_right,
                    backlit,
                ),
            )
        )
    )
    split_evidence = next(gate for gate in split_report.gates if gate.gate == "evidence")
    assert split_evidence.status is GateStatus.FAIL

    for removed_render in (perspective, raking_left, raking_right, backlit):
        reduced = tuple(
            render
            for render in (perspective, raking_left, raking_right, backlit)
            if render is not removed_render
        )
        reduced_report = score_episode(
            replace(
                OracleInputs(
                    spec=episode.spec,
                    truth=episode.ground_truth,
                    submission=submission,
                    trace=trace,
                    source_before=snapshot,
                    source_after=snapshot,
                    artifact_root=tmp_path,
                    renders=reduced,
                )
            )
        )
        evidence_gate = next(gate for gate in reduced_report.gates if gate.gate == "evidence")
        assert evidence_gate.status is GateStatus.FAIL

    Path(perspective.color.path).write_bytes(b"agent replacement")
    tampered_report = score_episode(
        OracleInputs(
            spec=episode.spec,
            truth=episode.ground_truth,
            submission=submission,
            trace=trace,
            source_before=snapshot,
            source_after=snapshot,
            artifact_root=tmp_path,
            renders=(perspective, raking_left, raking_right, backlit),
        )
    )
    tampered_evidence = next(gate for gate in tampered_report.gates if gate.gate == "evidence")
    assert tampered_evidence.status is GateStatus.FAIL


def test_full_investigation_coverage_fails_when_any_operation_is_removed(
    tmp_path: Path,
) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    episode = generate_episodes(model)[3]
    target = model.component_ids["idler"]
    events: list[TraceEvent] = []
    state = "0" * 64
    for sequence, operation in enumerate(Operation, start=1):
        arguments: object = {}
        result: object = {}
        before = state
        after = state
        if operation is Operation.SESSION_SNAPSHOT:
            result = {"session": {"state_sha256": state}}
        if operation is Operation.COMPONENT_FIND:
            result = [{"id": target}]
        if operation is Operation.COMPONENT_INSPECT:
            arguments = {"component_id": target}
            result = {"id": target}
        if operation in {
            Operation.VIEW_SET,
            Operation.VIEW_ORBIT,
            Operation.VIEW_MOVE,
            Operation.VIEW_ROTATE,
            Operation.ILLUMINATION_SET,
            Operation.COMPONENT_DISPLAY,
            Operation.COMPONENT_MARK,
        }:
            after = f"{sequence:x}" * 64
            state = after
        if operation is Operation.SESSION_RESET:
            after = "0" * 64
            state = after
            result = {"state_sha256": state}
        events.append(
            accepted_event(
                sequence,
                operation,
                arguments=arguments,
                result=result,
                before=before,
                after=after,
            )
        )
    snapshot = snapshot_source(model.path)
    inputs = OracleInputs(
        spec=episode.spec,
        truth=episode.ground_truth,
        submission=EpisodeSubmission(
            episode_id=episode.spec.episode_id,
            answer=episode.ground_truth.answer,
        ),
        trace=tuple(events),
        source_before=snapshot,
        source_after=snapshot,
    )
    baseline = score_episode(inputs)
    assert (
        next(gate for gate in baseline.gates if gate.gate == "coverage").status is GateStatus.PASS
    )

    for operation in Operation:
        reduced = tuple(event for event in events if event.operation is not operation)
        report = score_episode(replace(inputs, trace=reduced))
        coverage = next(gate for gate in report.gates if gate.gate == "coverage")
        assert coverage.status is GateStatus.FAIL, operation
