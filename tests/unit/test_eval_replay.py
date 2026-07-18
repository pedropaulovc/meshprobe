from __future__ import annotations

from pathlib import Path

import pytest

from meshprobe.evals.harness.broker import EvaluationBroker
from meshprobe.evals.harness.replay import _semantic_result, replay_trace
from meshprobe.evals.schemas import EpisodeBudgets, Operation, TraceEvent, TraceStatus
from meshprobe.protocol import Command, SceneOpenCommand, SessionSnapshotCommand
from meshprobe.service import CommandResponse


class ReplayService:
    def __init__(self, state: str = "1") -> None:
        self.state = state

    def execute_for_evaluation(
        self,
        command: Command,
        *,
        evaluator_output_dir: str,
    ) -> CommandResponse:
        del evaluator_output_dir
        result = {"source_sha256": "a" * 64}
        if command.op == "session.snapshot":
            result = {"session": {"state_sha256": self.state * 64}}
        return CommandResponse(request_id=command.request_id, op=command.op, result=result)


def make_broker(tmp_path: Path, name: str, service: ReplayService) -> EvaluationBroker:
    model = tmp_path / "model.glb"
    if not model.exists():
        model.write_bytes(b"model")
    return EvaluationBroker(
        service=service,
        model_path=model,
        artifact_root=tmp_path / name / "artifacts",
        evaluator_root=tmp_path / name / "private",
        trace_path=tmp_path / name / "trace.jsonl",
        budgets=EpisodeBudgets(),
    )


def recorded_trace(tmp_path: Path) -> tuple[tuple[TraceEvent, ...], EvaluationBroker]:
    broker = make_broker(tmp_path, "record", ReplayService())
    broker.execute(
        SceneOpenCommand(
            request_id="open",
            op="scene.open",
            source_path=broker.visible_model_path,
        )
    )
    broker.execute(SessionSnapshotCommand(request_id="describe", op="session.snapshot"))
    return broker.events, broker


def test_trace_replay_reproduces_semantic_results_and_state(tmp_path: Path) -> None:
    events, _ = recorded_trace(tmp_path)
    replay = make_broker(tmp_path, "replay", ReplayService())

    report = replay_trace(events, replay)

    assert report.passed
    assert all(event.replayed for event in report.events)


def test_trace_replay_reports_state_drift(tmp_path: Path) -> None:
    events, _ = recorded_trace(tmp_path)
    replay = make_broker(tmp_path, "drift", ReplayService(state="2"))

    report = replay_trace(events, replay)

    assert not report.passed
    assert report.events[-1].message == "state hash changed; semantic result changed"


def test_trace_replay_preserves_nonaccepted_events_without_executing(tmp_path: Path) -> None:
    events, _ = recorded_trace(tmp_path)
    rejected = events[0].model_copy(
        update={"status": TraceStatus.REJECTED, "error_code": "broker.test"}
    )
    replay = make_broker(tmp_path, "rejected", ReplayService())

    report = replay_trace((rejected,), replay)

    assert report.passed
    assert not report.events[0].replayed
    assert replay.events == ()


def test_trace_replay_rejects_invalid_sequence_and_arguments(tmp_path: Path) -> None:
    events, _ = recorded_trace(tmp_path)
    replay = make_broker(tmp_path, "invalid", ReplayService())

    with pytest.raises(ValueError, match="contiguous and one-based"):
        replay_trace((events[0].model_copy(update={"sequence": 2}),), replay)
    with pytest.raises(ValueError, match="arguments must be an object"):
        replay_trace((events[0].model_copy(update={"arguments": []}),), replay)


def test_render_replay_ignores_nondeterministic_artifact_fields() -> None:
    result = {
        "state_sha256": "a" * 64,
        "panels": [
            {"caption": "front", "color": {"path": "first.png"}},
            {"caption": "back", "device": "cuda"},
        ],
        "sheet": {"path": "sheet.png"},
    }

    assert _semantic_result(Operation.RENDER_CONTACT_SHEET, result) == {
        "state_sha256": "a" * 64,
        "panels": [{"caption": "front"}, {"caption": "back"}],
        "warnings": [],
    }


def test_render_replay_keeps_warnings_but_ignores_foreground_noise() -> None:
    recorded = {
        "state_sha256": "a" * 64,
        "foreground": {"visible_fraction": 0.5},
        "warnings": ["Rendered frame is effectively empty: ..."],
    }
    replayed_same_signal_noisy_pixels = {
        "state_sha256": "a" * 64,
        "foreground": {"visible_fraction": 0.4},
        "warnings": ["Rendered frame is effectively empty: ..."],
    }
    replayed_regressed_signal = {
        "state_sha256": "a" * 64,
        "foreground": {"visible_fraction": 0.5},
        "warnings": [],
    }

    # Pixel-derived visible_fraction is allowed to jitter across artifact/device variance.
    assert _semantic_result(Operation.RENDER_IMAGE, recorded) == _semantic_result(
        Operation.RENDER_IMAGE, replayed_same_signal_noisy_pixels
    )
    # But the discrete empty-frame warning must actually reproduce.
    assert _semantic_result(Operation.RENDER_IMAGE, recorded) != _semantic_result(
        Operation.RENDER_IMAGE, replayed_regressed_signal
    )


def test_render_replay_treats_a_missing_warnings_key_as_no_warnings() -> None:
    # A trace recorded before `warnings` existed has no such key at all; a fresh
    # render always emits `warnings: []` when nothing is wrong, so the two must
    # compare as reproducing each other rather than "semantic result changed".
    legacy_no_warnings_key = {"state_sha256": "a" * 64}
    fresh_empty_warnings = {"state_sha256": "a" * 64, "warnings": []}

    assert _semantic_result(Operation.RENDER_IMAGE, legacy_no_warnings_key) == _semantic_result(
        Operation.RENDER_IMAGE, fresh_empty_warnings
    )

    # A real regression (a new warning fires) must still be detected.
    fresh_with_warning = {"state_sha256": "a" * 64, "warnings": ["something is wrong"]}
    assert _semantic_result(Operation.RENDER_IMAGE, legacy_no_warnings_key) != _semantic_result(
        Operation.RENDER_IMAGE, fresh_with_warning
    )
