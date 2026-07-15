from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from PIL import Image

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
    GateStatus,
    Operation,
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


def discovery_inputs(tmp_path: Path) -> OracleInputs:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path)
    episode = generate_episodes(model)[0]
    target = model.component_ids["idler"]
    state = "a" * 64
    trace = (
        accepted_event(1, "scene.open", arguments={}, result={}),
        accepted_event(
            2,
            "scene.describe",
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
        device="graphics",
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
            "scene.describe",
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
            "view.orbit",
            arguments={},
            result=perspective.session.model_dump(mode="json"),
            before=initial_hash,
            after=perspective.state_sha256,
        ),
        accepted_event(
            6,
            "illumination.set",
            arguments={},
            result=raking_left.session.model_dump(mode="json"),
            before=perspective.state_sha256,
            after=raking_left.state_sha256,
        ),
        accepted_event(
            7,
            "component.display",
            arguments={"component_ids": [model.component_ids["arrow"]], "mode": "isolated"},
            result=raking_left.session.model_dump(mode="json"),
            before=raking_left.state_sha256,
            after="3" * 64,
        ),
        accepted_event(
            8,
            "component.mark",
            arguments={"component_ids": [target], "mode": "highlighted"},
            result=perspective.session.model_dump(mode="json"),
            before="3" * 64,
            after="4" * 64,
        ),
        accepted_event(
            9,
            "illumination.set",
            arguments={},
            result=raking_right.session.model_dump(mode="json"),
            before="4" * 64,
            after=raking_right.state_sha256,
        ),
        accepted_event(
            10,
            "illumination.set",
            arguments={},
            result=backlit.session.model_dump(mode="json"),
            before=raking_right.state_sha256,
            after=backlit.state_sha256,
        ),
        accepted_event(
            11, "render.image", arguments={}, result=perspective.model_dump(mode="json")
        ),
    )
    snapshot = snapshot_source(model.path)
    report = score_episode(
        OracleInputs(
            spec=episode.spec,
            truth=episode.ground_truth,
            submission=EpisodeSubmission(
                episode_id=episode.spec.episode_id,
                answer=episode.ground_truth.answer,
            ),
            trace=trace,
            source_before=snapshot,
            source_after=snapshot,
            renders=(perspective, raking_left, raking_right, backlit),
        )
    )

    assert next(gate for gate in report.gates if gate.gate == "state").status is GateStatus.PASS
    assert next(gate for gate in report.gates if gate.gate == "evidence").status is GateStatus.PASS
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
                isinstance(event.result, dict)
                and event.result.get("state_sha256") == state_hash
            )
        )
        reduced_report = score_episode(
            OracleInputs(
                spec=episode.spec,
                truth=episode.ground_truth,
                submission=EpisodeSubmission(
                    episode_id=episode.spec.episode_id,
                    answer=episode.ground_truth.answer,
                ),
                trace=reduced_trace,
                source_before=snapshot,
                source_after=snapshot,
                renders=(perspective, raking_left, raking_right, backlit),
            )
        )
        state_gate = next(gate for gate in reduced_report.gates if gate.gate == "state")
        assert state_gate.status is GateStatus.FAIL, group_name

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
                submission=EpisodeSubmission(
                    episode_id=episode.spec.episode_id,
                    answer=episode.ground_truth.answer,
                ),
                trace=trace,
                source_before=snapshot,
                source_after=snapshot,
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
                    submission=EpisodeSubmission(
                        episode_id=episode.spec.episode_id,
                        answer=episode.ground_truth.answer,
                    ),
                    trace=trace,
                    source_before=snapshot,
                    source_after=snapshot,
                    renders=reduced,
                )
            )
        )
        evidence_gate = next(
            gate for gate in reduced_report.gates if gate.gate == "evidence"
        )
        assert evidence_gate.status is GateStatus.FAIL


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
        if operation is Operation.SCENE_DESCRIBE:
            result = {"session": {"state_sha256": state}}
        if operation is Operation.COMPONENT_FIND:
            result = [{"id": target}]
        if operation is Operation.COMPONENT_INSPECT:
            arguments = {"component_id": target}
            result = {"id": target}
        if operation in {
            Operation.VIEW_SET,
            Operation.VIEW_ORBIT,
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
