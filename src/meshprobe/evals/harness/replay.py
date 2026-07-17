"""Semantic replay checks for accepted evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.schemas import Operation, TraceEvent, TraceStatus
from meshprobe.protocol import COMMAND_ADAPTER


@dataclass(frozen=True)
class ReplayEvent:
    sequence: int
    operation: Operation
    replayed: bool
    accepted: bool
    state_match: bool
    semantic_result_match: bool
    message: str

    @property
    def passed(self) -> bool:
        return not self.replayed or (
            self.accepted and self.state_match and self.semantic_result_match
        )


@dataclass(frozen=True)
class ReplayReport:
    events: tuple[ReplayEvent, ...]

    @property
    def passed(self) -> bool:
        return all(event.passed for event in self.events)


def replay_trace(events: tuple[TraceEvent, ...], broker: EvaluationBroker) -> ReplayReport:
    """Replay accepted calls and compare deterministic state and semantic results."""

    expected_sequences = tuple(range(1, len(events) + 1))
    if tuple(event.sequence for event in events) != expected_sequences:
        raise ValueError("trace event sequence must be contiguous and one-based")
    replayed: list[ReplayEvent] = []
    for event in events:
        if event.status is not TraceStatus.ACCEPTED:
            replayed.append(
                ReplayEvent(
                    sequence=event.sequence,
                    operation=event.operation,
                    replayed=False,
                    accepted=True,
                    state_match=True,
                    semantic_result_match=True,
                    message="rejected or crashed calls are not replayed",
                )
            )
            continue
        payload: dict[str, JsonValue] = {
            "request_id": event.request_id,
            "op": event.operation,
        }
        if not isinstance(event.arguments, dict):
            raise ValueError(f"trace event {event.sequence} arguments must be an object")
        payload.update(event.arguments)
        command = COMMAND_ADAPTER.validate_python(payload)
        reply = broker.execute(command)
        actual_event = broker.events[-1]
        accepted = reply.ok and actual_event.status is TraceStatus.ACCEPTED
        state_match = actual_event.state_after_sha256 == event.state_after_sha256
        result_match = _semantic_result(event.operation, actual_event.result) == _semantic_result(
            event.operation, event.result
        )
        failures: list[str] = []
        if not accepted:
            failures.append("replayed command was rejected")
        if not state_match:
            failures.append("state hash changed")
        if not result_match:
            failures.append("semantic result changed")
        replayed.append(
            ReplayEvent(
                sequence=event.sequence,
                operation=event.operation,
                replayed=True,
                accepted=accepted,
                state_match=state_match,
                semantic_result_match=result_match,
                message="; ".join(failures) or "semantic result and state reproduced",
            )
        )
    return ReplayReport(events=tuple(replayed))


def _semantic_result(operation: Operation, result: JsonValue) -> JsonValue:
    if operation not in {Operation.RENDER_IMAGE, Operation.RENDER_CONTACT_SHEET}:
        return result
    return _without_render_variance(result)


def _without_render_variance(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_without_render_variance(item) for item in value]
    if not isinstance(value, dict):
        return value
    ignored = {
        "color",
        "evaluator",
        "sheet",
        "luminance",
        "foreground",
        "warnings",
        "device",
        "blender_version",
    }
    return {
        key: _without_render_variance(item) for key, item in value.items() if key not in ignored
    }
