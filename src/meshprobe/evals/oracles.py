"""Operation-aware answer, state, evidence, safety, and budget gates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import cast

from PIL import Image
from pydantic import JsonValue, TypeAdapter

from meshprobe.evals.leakage import LeakageError, scan_error_message
from meshprobe.evals.schemas import (
    EpisodeGroundTruth,
    EpisodeReport,
    EpisodeSpec,
    EpisodeSubmission,
    EvidenceKind,
    EvidenceRequirement,
    GateResult,
    GateStatus,
    Operation,
    StatePredicate,
    StateRequirement,
    StructuredAnswer,
    TraceEvent,
    TraceStatus,
)
from meshprobe.models import ContactSheetManifest, RenderManifest
from meshprobe.sources import SourceSnapshot, sha256_file


@dataclass(frozen=True)
class OracleInputs:
    spec: EpisodeSpec
    truth: EpisodeGroundTruth
    submission: EpisodeSubmission
    trace: tuple[TraceEvent, ...]
    source_before: SourceSnapshot
    source_after: SourceSnapshot
    artifact_root: Path | None = None
    renders: tuple[RenderManifest, ...] = ()
    contact_sheets: tuple[ContactSheetManifest, ...] = ()
    public_errors: tuple[str, ...] = ()
    wall_seconds: float | None = None


@dataclass(frozen=True)
class EpisodeMetrics:
    tool_calls: int
    renders: int
    total_pixels: int
    output_bytes: int
    wall_seconds: float


def score_episode(inputs: OracleInputs) -> EpisodeReport:
    metrics = _metrics(inputs)
    gates = (
        _answer_gate(inputs),
        _state_gate(inputs),
        _evidence_gate(inputs),
        _coverage_gate(inputs),
        _read_only_gate(inputs),
        _reset_gate(inputs),
        _leakage_gate(inputs),
        _budget_gate(inputs, metrics),
    )
    return EpisodeReport(
        episode_id=inputs.spec.episode_id,
        family=inputs.spec.family,
        episode_class=inputs.spec.episode_class,
        difficulty=inputs.spec.difficulty,
        gates=gates,
        tool_calls=metrics.tool_calls,
        renders=metrics.renders,
        total_pixels=metrics.total_pixels,
        output_bytes=metrics.output_bytes,
        wall_seconds=metrics.wall_seconds,
    )


def _answer_gate(inputs: OracleInputs) -> GateResult:
    if inputs.submission.episode_id != inputs.truth.episode_id:
        return _fail("answer", "submission identifies a different episode")
    expected = inputs.truth.answer
    actual = inputs.submission.answer
    differences = _answer_differences(expected, actual)
    if differences:
        return _fail(
            "answer", "structured answer does not match ground truth", differences=differences
        )
    return _pass("answer", "structured answer matches ground truth")


def _answer_differences(expected: StructuredAnswer, actual: StructuredAnswer) -> list[str]:
    differences: list[str] = []
    if expected.status is not actual.status:
        differences.append(f"status expected {expected.status}, received {actual.status}")
    _compare_json(expected.values, actual.values, "values", differences)
    if set(expected.conflicting_component_ids) != set(actual.conflicting_component_ids):
        differences.append("conflicting_component_ids differ")
    if set(expected.missing_capabilities) != set(actual.missing_capabilities):
        differences.append("missing_capabilities differ")
    return differences


def _compare_json(
    expected: JsonValue, actual: JsonValue, path: str, differences: list[str]
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            differences.append(f"{path} keys differ")
            return
        for key in expected:
            _compare_json(expected[key], actual[key], f"{path}.{key}", differences)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(f"{path} lengths differ")
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _compare_json(expected_item, actual_item, f"{path}[{index}]", differences)
        return
    expected_number = _as_number(expected)
    actual_number = _as_number(actual)
    if expected_number is not None and actual_number is not None:
        if not math.isclose(expected_number, actual_number, rel_tol=1e-5, abs_tol=1e-4):
            differences.append(f"{path} expected {expected}, received {actual}")
        return
    if expected != actual:
        differences.append(f"{path} expected {expected!r}, received {actual!r}")


def _state_gate(inputs: OracleInputs) -> GateResult:
    failures: list[str] = []
    accepted = [event for event in inputs.trace if event.status is TraceStatus.ACCEPTED]
    rendered_state_hashes = {
        event.state_after_sha256
        for event in accepted
        if event.operation is Operation.RENDER_IMAGE
        and event.state_after_sha256 is not None
        and isinstance(event.result, dict)
        and event.result.get("state_sha256") == event.state_after_sha256
    }
    renders_by_state = {render.state_sha256: render for render in _all_renders(inputs)}
    ungrouped = [
        requirement
        for requirement in inputs.truth.state_requirements
        if requirement.state_group is None
    ]
    grouped: dict[str, list[StateRequirement]] = {}
    for requirement in inputs.truth.state_requirements:
        if requirement.state_group is not None:
            grouped.setdefault(requirement.state_group, []).append(requirement)
    for requirement in ungrouped:
        component_id = (
            inputs.truth.component_roles[requirement.component_role]
            if requirement.component_role is not None
            else None
        )
        if not any(
            _event_satisfies_state(event, requirement.predicate, requirement.expected, component_id)
            for event in accepted
        ):
            failures.append(
                f"no accepted event proves {requirement.predicate}={requirement.expected}"
            )
    for group_name, requirements in grouped.items():
        if any(
            _transition_chain_satisfies_state_group(
                accepted,
                event,
                requirements,
                inputs.truth.component_roles,
            )
            and event.state_after_sha256 in rendered_state_hashes
            and _render_satisfies_state_group(
                renders_by_state[event.state_after_sha256],
                requirements,
                inputs.truth.component_roles,
            )
            for event in accepted
            if event.operation is not Operation.RENDER_IMAGE
            and event.state_after_sha256 in renders_by_state
        ):
            continue
        failures.append(
            "no accepted state transition and render prove the same snapshot "
            f"for group {group_name}"
        )
    if failures:
        return _fail("state", "required scene transitions were not proven", failures=failures)
    return _pass("state", "every required scene transition is present in accepted results")


def _transition_chain_satisfies_state_group(
    events: list[TraceEvent],
    target: TraceEvent,
    requirements: list[StateRequirement],
    component_roles: dict[str, str],
) -> bool:
    replay = _CompactStateReplay()
    target_reached = False
    for event in events:
        replay.observe_snapshot(event)
        if target_reached:
            if event.operation is Operation.RENDER_IMAGE:
                break
            if event.operation in _COMPACT_STATE_OPERATIONS:
                break
            continue
        if event.operation not in _COMPACT_STATE_OPERATIONS:
            continue
        if not replay.apply(event):
            return False
        target_reached = event is target
    if not target_reached:
        return False
    snapshot = TraceEvent.model_construct(operation=target.operation, result=replay.state)
    return replay.saw_transition and all(
        _state_delta_contains_requirement(
            replay.state,
            requirement.predicate,
            (
                component_roles[requirement.component_role]
                if requirement.component_role is not None
                else None
            ),
        )
        and _event_satisfies_state(
            snapshot,
            requirement.predicate,
            requirement.expected,
            (
                component_roles[requirement.component_role]
                if requirement.component_role is not None
                else None
            ),
        )
        for requirement in requirements
    )


_COMPACT_STATE_OPERATIONS = frozenset(
    {
        Operation.VIEW_SET,
        Operation.VIEW_ORBIT,
        Operation.VIEW_MOVE,
        Operation.VIEW_ROTATE,
        Operation.ILLUMINATION_SET,
        Operation.COMPONENT_DISPLAY,
        Operation.COMPONENT_MARK,
        Operation.SESSION_RESET,
    }
)


@dataclass
class _CompactStateReplay:
    state: dict[str, JsonValue] = field(default_factory=lambda: {"components": {}})
    current_state_sha256: str | None = None
    chain_started: bool = False
    saw_transition: bool = False
    known_snapshots: dict[str, dict[str, JsonValue]] = field(default_factory=dict)

    def observe_snapshot(self, event: TraceEvent) -> None:
        snapshot = _replayable_session_state(event)
        if snapshot is None or event.state_after_sha256 is None:
            return
        if event.state_before_sha256 is None and (self.chain_started or self.known_snapshots):
            return
        self.known_snapshots[event.state_after_sha256] = snapshot
        if self.chain_started and event.state_after_sha256 == self.current_state_sha256:
            self.state = deepcopy(snapshot)

    def apply(self, event: TraceEvent) -> bool:
        if not isinstance(event.result, dict):
            return False
        if not self.chain_started:
            self.chain_started = True
            self.current_state_sha256 = event.state_before_sha256
            if self.current_state_sha256 is not None:
                initial = self.known_snapshots.get(self.current_state_sha256)
                if initial is not None:
                    self.state = deepcopy(initial)
        elif event.state_before_sha256 != self.current_state_sha256:
            return False
        if not _event_proves_state_hash(event):
            return False
        assert event.state_after_sha256 is not None
        self.current_state_sha256 = event.state_after_sha256
        self.saw_transition |= event.state_before_sha256 != event.state_after_sha256
        if event.operation is Operation.SESSION_RESET:
            reset = self.known_snapshots.get(event.state_after_sha256, {"components": {}})
            self.state = deepcopy(reset)
        for key in ("camera", "illumination"):
            value = event.result.get(key)
            if isinstance(value, dict):
                self.state[key] = deepcopy(value)
        components = event.result.get("components")
        if isinstance(components, dict):
            accumulated = self.state.setdefault("components", {})
            if isinstance(accumulated, dict):
                accumulated.update(deepcopy(components))
        return True


def _replayable_session_state(event: TraceEvent) -> dict[str, JsonValue] | None:
    if event.operation is not Operation.SESSION_SNAPSHOT:
        return None
    if not isinstance(event.result, dict):
        return None
    session = event.result.get("session")
    if (
        event.state_after_sha256 is None
        or (
            event.state_before_sha256 is not None
            and event.state_before_sha256 != event.state_after_sha256
        )
        or not isinstance(session, dict)
        or session.get("state_sha256") != event.state_after_sha256
    ):
        return None
    camera = session.get("camera")
    illumination = session.get("illumination")
    components = session.get("components")
    if not all(isinstance(value, dict) for value in (camera, illumination, components)):
        return None
    return {
        "camera": deepcopy(camera),
        "illumination": deepcopy(illumination),
        "components": deepcopy(components),
    }


def _event_proves_state_hash(event: TraceEvent) -> bool:
    if event.state_after_sha256 is None or not isinstance(event.result, dict):
        return False
    return event.result.get("state_sha256") == event.state_after_sha256


def _state_delta_contains_requirement(
    state: dict[str, JsonValue],
    predicate: StatePredicate,
    component_id: str | None,
) -> bool:
    if predicate is StatePredicate.RESET_TO_IMPORTED:
        return True
    if predicate in {StatePredicate.PROJECTION_MODE, StatePredicate.FOCAL_LENGTH_MM}:
        return isinstance(state.get("camera"), dict)
    if predicate is StatePredicate.ILLUMINATION_PRESET:
        return isinstance(state.get("illumination"), dict)
    components = state.get("components")
    if component_id is None or not isinstance(components, dict):
        return False
    component = components.get(component_id)
    if not isinstance(component, dict):
        return False
    key = "display" if predicate is StatePredicate.COMPONENT_DISPLAY else "mark"
    return key in component


def _render_satisfies_state_group(
    render: RenderManifest,
    requirements: list[StateRequirement],
    component_roles: dict[str, str],
) -> bool:
    snapshot = TraceEvent.model_construct(
        operation=Operation.RENDER_IMAGE,
        result=render.session.model_dump(mode="json"),
    )
    return all(
        _event_satisfies_state(
            snapshot,
            requirement.predicate,
            requirement.expected,
            (
                component_roles[requirement.component_role]
                if requirement.component_role is not None
                else None
            ),
        )
        for requirement in requirements
    )


def _event_satisfies_state(
    event: TraceEvent,
    predicate: StatePredicate,
    expected: JsonValue,
    component_id: str | None,
) -> bool:
    result = event.result
    if not isinstance(result, dict):
        return False
    if predicate is StatePredicate.RESET_TO_IMPORTED:
        return event.operation is Operation.SESSION_RESET
    camera = result.get("camera")
    if predicate in {StatePredicate.PROJECTION_MODE, StatePredicate.FOCAL_LENGTH_MM}:
        if not isinstance(camera, dict):
            return False
        projection = camera.get("projection")
        if not isinstance(projection, dict):
            return False
        if predicate is StatePredicate.PROJECTION_MODE:
            return projection.get("mode") == expected
        return _numbers_equal(projection.get("focal_length_mm"), expected)
    if predicate is StatePredicate.ILLUMINATION_PRESET:
        illumination = result.get("illumination")
        return isinstance(illumination, dict) and illumination.get("preset") == expected
    components = result.get("components")
    if component_id is None or not isinstance(components, dict):
        return False
    component = components.get(component_id)
    if not isinstance(component, dict):
        return False
    key = "display" if predicate is StatePredicate.COMPONENT_DISPLAY else "mark"
    return component.get(key) == expected


def _evidence_gate(inputs: OracleInputs) -> GateResult:
    if not inputs.truth.evidence_requirements:
        return _na("evidence", "episode declares no rendered evidence requirements")
    renders, contact_sheets, submission_error = _submitted_evidence(inputs)
    if submission_error is not None:
        return _fail("evidence", submission_error)
    failures: list[str] = []
    ungrouped = [
        requirement
        for requirement in inputs.truth.evidence_requirements
        if requirement.render_group is None
    ]
    grouped: dict[str, list[EvidenceRequirement]] = {}
    for requirement in inputs.truth.evidence_requirements:
        if requirement.render_group is not None:
            grouped.setdefault(requirement.render_group, []).append(requirement)
    for requirement in ungrouped:
        component_id = (
            inputs.truth.component_roles[requirement.component_role]
            if requirement.component_role is not None
            else None
        )
        if not _evidence_satisfied(
            requirement.kind,
            requirement.expected,
            requirement.minimum,
            requirement.maximum,
            component_id,
            renders,
            contact_sheets,
        ):
            failures.append(f"unsatisfied evidence requirement: {requirement.kind}")
    for group_name, requirements in grouped.items():
        if any(
            all(
                _evidence_satisfied(
                    requirement.kind,
                    requirement.expected,
                    requirement.minimum,
                    requirement.maximum,
                    (
                        inputs.truth.component_roles[requirement.component_role]
                        if requirement.component_role is not None
                        else None
                    ),
                    (render,),
                    (),
                )
                for requirement in requirements
            )
            for render in renders
        ):
            continue
        failures.append(f"no single render proves evidence group {group_name}")
    if failures:
        return _fail(
            "evidence", "rendered evidence does not prove every requirement", failures=failures
        )
    return _pass("evidence", "private render passes prove every evidence requirement")


def _submitted_evidence(
    inputs: OracleInputs,
) -> tuple[tuple[RenderManifest, ...], tuple[ContactSheetManifest, ...], str | None]:
    if inputs.artifact_root is None:
        return (), (), "evidence artifact root is unavailable"
    if not inputs.submission.evidence_manifest_paths:
        return (), (), "submission declares no evidence manifest paths"
    artifact_root = inputs.artifact_root.expanduser().resolve(strict=True)
    submitted: set[Path] = set()
    try:
        for visible_path in inputs.submission.evidence_manifest_paths:
            submitted.add(_submitted_artifact_path(visible_path, artifact_root))
        render_paths = {
            Path(render.color.path).resolve(strict=True): render for render in inputs.renders
        }
        sheet_paths = {
            Path(sheet.sheet.path).resolve(strict=True): sheet for sheet in inputs.contact_sheets
        }
    except (OSError, ValueError):
        return (), (), "submission contains an invalid evidence manifest path"
    unknown = submitted - render_paths.keys() - sheet_paths.keys()
    if unknown:
        return (), (), "submission references an unknown evidence manifest path"
    renders = tuple(
        render
        for path, render in render_paths.items()
        if path in submitted and _artifact_intact(render.color)
    )
    sheets = tuple(
        sheet
        for path, sheet in sheet_paths.items()
        if path in submitted and _artifact_intact(sheet.sheet)
    )
    if len(renders) + len(sheets) != len(submitted):
        return (), (), "submitted evidence artifact bytes do not match their render manifests"
    return (
        (*renders, *(panel.render for sheet in sheets for panel in sheet.panels)),
        sheets,
        None,
    )


def _submitted_artifact_path(visible_path: str, artifact_root: Path) -> Path:
    candidate = PurePosixPath(visible_path)
    prefix = PurePosixPath("/workspace/artifacts")
    if candidate.is_absolute() and candidate.is_relative_to(prefix):
        relative = candidate.relative_to(prefix)
        resolved = (artifact_root / Path(*relative.parts)).resolve(strict=True)
    else:
        host_candidate = Path(visible_path).expanduser()
        resolved = (
            host_candidate.resolve(strict=True)
            if host_candidate.is_absolute()
            else (artifact_root / host_candidate).resolve(strict=True)
        )
    if not resolved.is_file() or not resolved.is_relative_to(artifact_root):
        raise ValueError("evidence path escapes artifact root")
    return resolved


def _evidence_satisfied(
    kind: EvidenceKind,
    expected: JsonValue,
    minimum: float | None,
    maximum: float | None,
    component_id: str | None,
    renders: tuple[RenderManifest, ...],
    sheets: tuple[ContactSheetManifest, ...],
) -> bool:
    if kind is EvidenceKind.CONTACT_SHEET_PANELS:
        return any(len(sheet.panels) == expected for sheet in sheets)
    if kind is EvidenceKind.DISTINCT_VIEWS:
        distinct = max(
            (len({panel.render.state_sha256 for panel in sheet.panels}) for sheet in sheets),
            default=0,
        )
        return minimum is not None and distinct >= minimum
    if kind is EvidenceKind.PROJECTION:
        return any(render.session.camera.projection.mode == expected for render in renders)
    if kind is EvidenceKind.FOCAL_LENGTH:
        return any(
            getattr(render.session.camera.projection, "focal_length_mm", None) == expected
            for render in renders
        )
    if kind is EvidenceKind.ILLUMINATION:
        return any(render.session.illumination.preset == expected for render in renders)
    if component_id is None:
        return False
    for render in renders:
        if render.evaluator is None:
            continue
        fraction = _component_fraction(render, component_id)
        if kind is EvidenceKind.TARGET_VISIBLE and _within(fraction, minimum, maximum):
            return True
        if kind is EvidenceKind.TARGET_SCREEN_SPAN:
            span = _component_screen_span(render, component_id)
            if _within(span, minimum, maximum):
                return True
        if kind is EvidenceKind.COMPONENT_ABSENT and _within(fraction, minimum, maximum):
            return True
        if kind is EvidenceKind.TARGET_HIGHLIGHTED:
            highlighted = _highlighted_fraction(render, component_id)
            if _within(highlighted, minimum, maximum):
                return True
        if kind is EvidenceKind.TARGET_CONTRAST:
            contrast = _component_contrast(render, component_id)
            if _within(contrast, minimum, maximum):
                return True
    return False


def _coverage_gate(inputs: OracleInputs) -> GateResult:
    accepted = [event for event in inputs.trace if event.status is TraceStatus.ACCEPTED]
    failures: list[str] = []
    for operation in inputs.truth.required_operations:
        events = [event for event in accepted if event.operation is operation]
        if not events:
            failures.append(f"missing accepted {operation}")
            continue
        if operation in {
            Operation.VIEW_SET,
            Operation.VIEW_ORBIT,
            Operation.VIEW_MOVE,
            Operation.VIEW_ROTATE,
            Operation.ILLUMINATION_SET,
            Operation.COMPONENT_DISPLAY,
            Operation.COMPONENT_MARK,
            Operation.SESSION_RESET,
        } and not any(
            event.state_after_sha256 is not None
            and event.state_before_sha256 != event.state_after_sha256
            for event in events
        ):
            failures.append(f"{operation} was accepted but did not change state")
    target_id = inputs.truth.component_roles.get("idler") or inputs.truth.component_roles.get(
        "target"
    )
    if target_id is not None:
        if Operation.COMPONENT_FIND in inputs.truth.required_operations and not any(
            _result_contains_component(event.result, target_id)
            for event in accepted
            if event.operation is Operation.COMPONENT_FIND
        ):
            failures.append("component.find did not return the intended target")
        if Operation.COMPONENT_INSPECT in inputs.truth.required_operations and not any(
            isinstance(event.arguments, dict) and event.arguments.get("component_id") == target_id
            for event in accepted
            if event.operation is Operation.COMPONENT_INSPECT
        ):
            failures.append("component.inspect did not address the intended target")
    if failures:
        return _fail(
            "coverage", "required operations were missing, no-op, or misdirected", failures=failures
        )
    return _pass("coverage", "every required operation succeeded on intended state")


def _read_only_gate(inputs: OracleInputs) -> GateResult:
    if inputs.source_before != inputs.source_after:
        return _fail("read_only", "source asset content or metadata changed during the episode")
    return _pass("read_only", "source asset content and metadata are unchanged")


def _reset_gate(inputs: OracleInputs) -> GateResult:
    required = any(
        requirement.predicate is StatePredicate.RESET_TO_IMPORTED
        for requirement in inputs.truth.state_requirements
    )
    if not required:
        return _na("reset", "episode does not require a final reset")
    describes = [
        event
        for event in inputs.trace
        if event.operation is Operation.SESSION_SNAPSHOT
        and event.status is TraceStatus.ACCEPTED
        and isinstance(event.result, dict)
    ]
    resets = [
        event
        for event in inputs.trace
        if event.operation is Operation.SESSION_RESET
        and event.status is TraceStatus.ACCEPTED
        and isinstance(event.result, dict)
    ]
    if not describes or not resets:
        return _fail("reset", "initial session description or accepted reset is missing")
    describe_result = describes[0].result
    final_reset = resets[-1].result
    if not isinstance(describe_result, dict) or not isinstance(final_reset, dict):
        return _fail("reset", "initial description or final reset result is malformed")
    initial_session = describe_result.get("session")
    if not isinstance(initial_session, dict) or final_reset.get(
        "state_sha256"
    ) != initial_session.get("state_sha256"):
        return _fail("reset", "final reset did not restore the imported session hash")
    if inputs.trace[-1].operation is not Operation.SESSION_RESET:
        return _fail("reset", "session.reset was not the final tool operation")
    return _pass("reset", "final operation restored the imported session state")


def _leakage_gate(inputs: OracleInputs) -> GateResult:
    failures: list[str] = []
    for message in inputs.public_errors:
        try:
            scan_error_message(message, inputs.truth)
        except LeakageError as error:
            failures.append(str(error))
    if failures:
        return _fail(
            "leakage", "public tool errors revealed evaluator information", failures=failures
        )
    return _pass("leakage", "public errors contain no forbidden evaluator tokens")


def _budget_gate(inputs: OracleInputs, metrics: EpisodeMetrics) -> GateResult:
    budget = inputs.spec.budgets
    values = {
        "tool_calls": (metrics.tool_calls, budget.tool_calls),
        "renders": (metrics.renders, budget.renders),
        "total_pixels": (metrics.total_pixels, budget.total_pixels),
        "output_bytes": (metrics.output_bytes, budget.output_bytes),
        "wall_seconds": (metrics.wall_seconds, budget.wall_seconds),
    }
    exceeded = {
        key: {"actual": actual, "limit": limit}
        for key, (actual, limit) in values.items()
        if actual > limit
    }
    if exceeded:
        return _fail("budget", "episode exceeded one or more declared budgets", exceeded=exceeded)
    return _pass("budget", "episode stayed within all declared budgets")


def _metrics(inputs: OracleInputs) -> EpisodeMetrics:
    renders = _all_renders(inputs)
    unique_renders = {render.color.path: render for render in renders}
    artifact_bytes: dict[str, int] = {}
    for render in unique_renders.values():
        artifact_bytes[render.color.path] = render.color.bytes
        if render.evaluator is not None:
            for artifact in (
                render.evaluator.multilayer,
                render.evaluator.component_ids,
                render.evaluator.highlighted,
            ):
                artifact_bytes[artifact.path] = artifact.bytes
    for sheet in inputs.contact_sheets:
        artifact_bytes[sheet.sheet.path] = sheet.sheet.bytes
    starts = [event.started_monotonic for event in inputs.trace]
    ends = [event.started_monotonic + event.elapsed_seconds for event in inputs.trace]
    trace_wall_seconds = max(ends) - min(starts) if starts else 0.0
    wall_seconds = inputs.wall_seconds if inputs.wall_seconds is not None else trace_wall_seconds
    return EpisodeMetrics(
        tool_calls=len(inputs.trace),
        renders=len(unique_renders),
        total_pixels=sum(render.width * render.height for render in unique_renders.values()),
        output_bytes=sum(artifact_bytes.values()),
        wall_seconds=wall_seconds,
    )


def _all_renders(inputs: OracleInputs) -> tuple[RenderManifest, ...]:
    return (
        *inputs.renders,
        *(panel.render for sheet in inputs.contact_sheets for panel in sheet.panels),
    )


def _component_fraction(render: RenderManifest, component_id: str) -> float:
    if render.evaluator is None or component_id not in render.evaluator.component_colors:
        return 0.0
    if not _artifact_intact(render.evaluator.component_ids):
        return 0.0
    expected = render.evaluator.component_colors[component_id]
    count, total, _width_span, _height_span = _component_mask_stats(
        render.evaluator.component_ids.path,
        render.evaluator.component_ids.sha256,
        expected,
    )
    return count / total if total else 0.0


def _component_screen_span(render: RenderManifest, component_id: str) -> float:
    if render.evaluator is None or component_id not in render.evaluator.component_colors:
        return 0.0
    if not _artifact_intact(render.evaluator.component_ids):
        return 0.0
    expected = render.evaluator.component_colors[component_id]
    _count, _total, width_span, height_span = _component_mask_stats(
        render.evaluator.component_ids.path,
        render.evaluator.component_ids.sha256,
        expected,
    )
    return max(width_span, height_span)


@lru_cache(maxsize=512)
def _component_mask_stats(
    path: str,
    sha256: str,
    expected: tuple[int, int, int],
) -> tuple[int, int, float, float]:
    del sha256  # Keep the digest in the cache key so rewritten artifacts cannot reuse results.
    with Image.open(path) as image:
        pixels = image.convert("RGB")
        width, height = pixels.size
        minimum_x = width
        minimum_y = height
        maximum_x = -1
        maximum_y = -1
        count = 0
        data = cast(Iterable[tuple[int, int, int]], pixels.get_flattened_data())
        for index, pixel in enumerate(data):
            if pixel != expected:
                continue
            x = index % width
            y = index // width
            count += 1
            minimum_x = min(minimum_x, x)
            minimum_y = min(minimum_y, y)
            maximum_x = max(maximum_x, x)
            maximum_y = max(maximum_y, y)
    if not count:
        return 0, width * height, 0.0, 0.0
    return (
        count,
        width * height,
        (maximum_x - minimum_x + 1) / width,
        (maximum_y - minimum_y + 1) / height,
    )


def _highlighted_fraction(render: RenderManifest, component_id: str) -> float:
    if render.evaluator is None or component_id not in render.evaluator.component_colors:
        return 0.0
    if not all(
        _artifact_intact(artifact)
        for artifact in (render.evaluator.component_ids, render.evaluator.highlighted)
    ):
        return 0.0
    expected = render.evaluator.component_colors[component_id]
    with (
        Image.open(render.evaluator.component_ids.path) as component_image,
        Image.open(render.evaluator.highlighted.path) as highlight_image,
    ):
        component_pixels = component_image.convert("RGB")
        highlight_pixels = highlight_image.convert("RGB")
        target_count = 0
        highlighted_count = 0
        component_data = cast(Iterable[tuple[int, int, int]], component_pixels.get_flattened_data())
        highlight_data = cast(Iterable[tuple[int, int, int]], highlight_pixels.get_flattened_data())
        for component_pixel, highlight_pixel in zip(component_data, highlight_data, strict=True):
            if component_pixel != expected:
                continue
            target_count += 1
            highlighted_count += highlight_pixel == (255, 255, 255)
        return highlighted_count / target_count if target_count else 0.0


def _component_contrast(render: RenderManifest, component_id: str) -> float:
    if render.evaluator is None or component_id not in render.evaluator.component_colors:
        return 0.0
    if not all(
        _artifact_intact(artifact) for artifact in (render.evaluator.component_ids, render.color)
    ):
        return 0.0
    expected = render.evaluator.component_colors[component_id]
    with (
        Image.open(render.evaluator.component_ids.path) as component_image,
        Image.open(render.color.path) as color_image,
    ):
        mask = component_image.convert("RGB")
        color = color_image.convert("RGB")
        mask_data = cast(Iterable[tuple[int, int, int]], mask.get_flattened_data())
        color_data = cast(Iterable[tuple[int, int, int]], color.get_flattened_data())
        values = [
            0.2126 * pixel[0] / 255 + 0.7152 * pixel[1] / 255 + 0.0722 * pixel[2] / 255
            for mask_pixel, pixel in zip(mask_data, color_data, strict=True)
            if mask_pixel == expected
        ]
    if len(values) < 2:
        return 0.0
    values.sort()
    low = values[int((len(values) - 1) * 0.05)]
    high = values[int((len(values) - 1) * 0.95)]
    return high - low


def _artifact_intact(artifact: object) -> bool:
    path_value = getattr(artifact, "path", None)
    expected_bytes = getattr(artifact, "bytes", None)
    expected_sha256 = getattr(artifact, "sha256", None)
    if not isinstance(path_value, str):
        return False
    path = Path(path_value)
    try:
        return path.stat().st_size == expected_bytes and sha256_file(path) == expected_sha256
    except OSError:
        return False


def _result_contains_component(result: JsonValue, component_id: str) -> bool:
    if isinstance(result, dict):
        return result.get("id") == component_id or any(
            _result_contains_component(value, component_id) for value in result.values()
        )
    if isinstance(result, list):
        return any(_result_contains_component(value, component_id) for value in result)
    return False


def _within(value: float, minimum: float | None, maximum: float | None) -> bool:
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _numbers_equal(left: JsonValue, right: JsonValue) -> bool:
    left_number = _as_number(left)
    right_number = _as_number(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=1e-6, abs_tol=1e-6)
    )


def _as_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _pass(gate: str, message: str) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.PASS, message=message)


def _fail(gate: str, message: str, **details: object) -> GateResult:
    serialized: dict[str, JsonValue] = {
        key: TypeAdapter(JsonValue).validate_python(value) for key, value in details.items()
    }
    return GateResult(gate=gate, status=GateStatus.FAIL, message=message, details=serialized)


def _na(gate: str, message: str) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.NOT_APPLICABLE, message=message)
