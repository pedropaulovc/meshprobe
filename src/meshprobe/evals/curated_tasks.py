"""Private-ground-truth tasks over controlled assemblies of licensed models."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from meshprobe.controller import BlenderController
from meshprobe.evals.curated_build import CuratedBuild
from meshprobe.evals.factory import CorpusBuild, validate_corpus
from meshprobe.evals.generators import GeneratedEpisode
from meshprobe.evals.leakage import validate_public_directory, validate_public_episode
from meshprobe.evals.schemas import (
    EVALUATED_OPERATIONS,
    AnswerStatus,
    CorpusManifest,
    CorpusTier,
    Difficulty,
    EpisodeBudgets,
    EpisodeClass,
    EpisodeGroundTruth,
    EpisodeSpec,
    EvidenceKind,
    EvidenceRequirement,
    ModelSource,
    Operation,
    StatePredicate,
    StateRequirement,
    StructuredAnswer,
    TaskFamily,
)
from meshprobe.models import SceneManifest
from meshprobe.sources import sha256_file


@dataclass(frozen=True)
class CuratedModelTruth:
    model_path: Path
    model_sha256: str
    component_names: dict[str, str]
    component_ids: dict[str, str]
    component_paths: dict[str, str]


def build_curated_corpus(
    build: CuratedBuild,
    output_root: Path,
    *,
    corpus_version: str = "curated-tasks-v1",
    blender: str = "blender",
) -> CorpusBuild:
    """Inspect every built variant and atomically publish three tasks per model."""

    destination = output_root.expanduser().resolve() / corpus_version
    generator_sha256 = curated_task_generator_sha256(build)
    if destination.exists():
        existing = validate_corpus(destination)
        if existing.manifest.generator_sha256 != generator_sha256:
            raise RuntimeError("curated corpus version belongs to a different generator input")
        return existing
    staging = output_root.expanduser().resolve() / f".{corpus_version}.building"
    _prepare_staging(staging, generator_sha256)
    public = staging / "public"
    private = staging / "private"
    models = public / "models"
    episodes = public / "episodes"
    ground_truth = private / "ground_truth"
    for path in (models, episodes, ground_truth):
        path.mkdir(parents=True, exist_ok=True)

    completed = _read_checkpoint(staging, generator_sha256)
    source_models = sorted((build.root / "public" / "models").glob("*.glb"))
    with BlenderController(executable=blender, timeout_seconds=180) as controller:
        for source_model in source_models:
            if source_model.name in completed:
                continue
            model_path = models / source_model.name
            _link_or_copy(source_model, model_path)
            manifest = controller.open_scene(model_path)
            role_path = build.root / "private" / "roles" / f"{source_model.stem}.json"
            role_payload = json.loads(role_path.read_text(encoding="utf-8"))
            model_truth = resolve_curated_truth(model_path, manifest, role_payload)
            for episode in generate_curated_episodes(model_truth):
                validate_public_episode(episode.spec, episode.ground_truth)
                _write_json(
                    episodes / f"{episode.spec.episode_id}.json",
                    episode.spec.model_dump(mode="json"),
                )
                _write_json(
                    ground_truth / f"{episode.ground_truth.episode_id}.json",
                    episode.ground_truth.model_dump(mode="json"),
                )
            completed.add(source_model.name)
            _write_checkpoint(staging, generator_sha256, completed)

    episode_paths = sorted(episodes.glob("*.json"))
    model_hashes = {path.name: sha256_file(path) for path in sorted(models.glob("*.glb"))}
    if len(model_hashes) != build.model_count or len(episode_paths) != build.model_count * 3:
        raise RuntimeError("curated task publication is incomplete")
    corpus_manifest = CorpusManifest(
        corpus_version=corpus_version,
        tier=CorpusTier.RELEASE,
        generator_sha256=generator_sha256,
        model_sources=(ModelSource.CURATED,),
        generator_families=("controlled_curated_assemblies",),
        generator_seeds=(0,),
        model_sha256=model_hashes,
        episodes=tuple(path.stem for path in episode_paths),
        episode_sha256={path.stem: sha256_file(path) for path in episode_paths},
    )
    _write_json(public / "manifest.json", corpus_manifest.model_dump(mode="json"))
    validate_public_directory(public, private)
    (staging / "checkpoint.json").unlink()
    os.replace(staging, destination)
    return validate_corpus(destination)


def resolve_curated_truth(
    model_path: Path,
    manifest: SceneManifest,
    role_payload: object,
) -> CuratedModelTruth:
    if not isinstance(role_payload, dict) or not isinstance(role_payload.get("roles"), dict):
        raise ValueError("curated role payload is malformed")
    if role_payload.get("model_file") != model_path.name:
        raise ValueError("curated role payload points to a different model")
    by_name = {component.display_name: component for component in manifest.components}
    component_names: dict[str, str] = {}
    component_ids: dict[str, str] = {}
    component_paths: dict[str, str] = {}
    for role in ("target", "reference", "occluder"):
        declared = role_payload["roles"].get(role)
        if not isinstance(declared, list) or not declared:
            raise ValueError(f"curated role payload has no {role} components")
        preferred = sorted(
            (str(name) for name in declared),
            key=lambda name: ("collider" in name.casefold(), name),
        )[0]
        component = by_name.get(preferred)
        if component is None:
            raise ValueError(f"curated role component is absent from manifest: {preferred}")
        component_names[role] = preferred
        component_ids[role] = component.id
        component_paths[role] = component.path
    return CuratedModelTruth(
        model_path=model_path,
        model_sha256=manifest.source_sha256,
        component_names=component_names,
        component_ids=component_ids,
        component_paths=component_paths,
    )


def generate_curated_episodes(model: CuratedModelTruth) -> tuple[GeneratedEpisode, ...]:
    task_contracts = (
        (
            TaskFamily.IDENTITY_VERIFICATION,
            Difficulty.BASIC,
            _discovery_operations(),
        ),
        (
            TaskFamily.PROJECTION_REASONING,
            Difficulty.HARD,
            _evidence_operations(),
        ),
        (
            TaskFamily.FULL_INVESTIGATION,
            Difficulty.FULL_STACK,
            EVALUATED_OPERATIONS,
        ),
    )
    return tuple(
        _episode(model, index, family, difficulty, operations)
        for index, (family, difficulty, operations) in enumerate(task_contracts)
    )


def curated_task_generator_sha256(build: CuratedBuild) -> str:
    digest = hashlib.sha256(b"meshprobe-curated-task-generator-v1\0")
    importer = Path(__file__).parents[1] / "blender" / "worker.py"
    inputs = [
        ("curated_tasks.py", Path(__file__)),
        ("schemas.py", Path(__file__).with_name("schemas.py")),
        ("leakage.py", Path(__file__).with_name("leakage.py")),
        ("blender/worker.py", importer),
        ("curated-build-manifest.json", build.root / "public" / "manifest.json"),
    ]
    inputs.extend(
        (f"roles/{path.name}", path)
        for path in sorted((build.root / "private" / "roles").glob("*.json"))
    )
    for logical_name, path in inputs:
        digest.update(logical_name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _episode(
    model: CuratedModelTruth,
    index: int,
    family: TaskFamily,
    difficulty: Difficulty,
    operations: tuple[Operation, ...],
) -> GeneratedEpisode:
    digest = hashlib.sha256(f"curated:{model.model_path.name}:{index}".encode()).hexdigest()
    episode_id = f"ep_{digest[:24]}"
    prompt = _prompt(model, index)
    values: dict[str, JsonValue] = {
        f"{role}_component_id": component_id for role, component_id in model.component_ids.items()
    }
    values.update(
        {f"{role}_path": component_path for role, component_path in model.component_paths.items()}
    )
    answer = StructuredAnswer(
        status=AnswerStatus.ANSWERED,
        values=values,
    )
    spec = EpisodeSpec(
        episode_id=episode_id,
        model_file=model.model_path.name,
        model_sha256=model.model_sha256,
        family=family,
        episode_class=(EpisodeClass.POSITIVE if index == 0 else EpisodeClass.ADVERSARIAL),
        difficulty=difficulty,
        model_source=ModelSource.CURATED,
        prompt=prompt,
        answer_schema=_answer_schema(),
        required_operations=operations,
        budgets=EpisodeBudgets(
            tool_calls=50 if index == 0 else 140,
            renders=0 if index == 0 else 28,
            total_pixels=0 if index == 0 else 140_000_000,
            output_bytes=10_000_000 if index == 0 else 3_000_000_000,
            wall_seconds=240 if index == 0 else 1_200,
        ),
    )
    truth = EpisodeGroundTruth(
        episode_id=episode_id,
        model_sha256=model.model_sha256,
        generator_family="controlled_curated_assemblies",
        answer=answer,
        component_roles=model.component_ids,
        component_paths=model.component_paths,
        state_requirements=() if index == 0 else _state_requirements(index),
        evidence_requirements=() if index == 0 else _evidence_requirements(),
        required_operations=operations,
        forbidden_public_tokens=tuple(sorted(model.component_ids.values())),
    )
    return GeneratedEpisode(spec=spec, ground_truth=truth)


def _prompt(model: CuratedModelTruth, index: int) -> str:
    target = model.component_names["target"]
    reference = model.component_names["reference"]
    occluder = model.component_names["occluder"]
    if index == 0:
        return (
            f"Open the assigned model and distinguish {target!r}, {reference!r}, and "
            f"{occluder!r}. Return each stable component ID and complete hierarchy path."
        )
    if index == 1:
        return (
            f"Verify the identity and hierarchy of {target!r} against {reference!r}. Highlight "
            f"the target and hide {occluder!r} in both a neutral 85 mm perspective context "
            "render and an orthographic raking-left render. Submit both images plus focused "
            "contact-sheet evidence, then return all three stable IDs and paths."
        )
    return (
        f"Perform a complete inspection of {target!r}, using {reference!r} as the comparison and "
        f"{occluder!r} as the removable foreground component. Exercise every evaluated MeshProbe "
        "operation, "
        "including a neutral 85 mm perspective context state and an orthographic raking-left "
        "state with the occluder hidden and target highlighted, still renders, a focused 3x3 "
        "contact sheet, and a final session reset. Return all three stable IDs and paths with "
        "the evidence manifests."
    )


def _discovery_operations() -> tuple[Operation, ...]:
    return (
        Operation.SCENE_OPEN,
        Operation.SESSION_SNAPSHOT,
        Operation.COMPONENT_FIND,
        Operation.COMPONENT_INSPECT,
    )


def _evidence_operations() -> tuple[Operation, ...]:
    return (
        Operation.SCENE_OPEN,
        Operation.SESSION_SNAPSHOT,
        Operation.COMPONENT_FIND,
        Operation.COMPONENT_INSPECT,
        Operation.VIEW_SET,
        Operation.VIEW_ORBIT,
        Operation.VIEW_MOVE,
        Operation.VIEW_ROTATE,
        Operation.ILLUMINATION_SET,
        Operation.COMPONENT_DISPLAY,
        Operation.COMPONENT_MARK,
        Operation.RENDER_IMAGE,
        Operation.RENDER_CONTACT_SHEET,
    )


def _state_requirements(index: int) -> tuple[StateRequirement, ...]:
    requirements = [
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="perspective",
            state_group="context_85",
        ),
        StateRequirement(
            predicate=StatePredicate.FOCAL_LENGTH_MM,
            expected=85.0,
            state_group="context_85",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="neutral_studio",
            state_group="context_85",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_DISPLAY,
            component_role="occluder",
            expected="hidden",
            state_group="context_85",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_MARK,
            component_role="target",
            expected="highlighted",
            state_group="context_85",
        ),
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
            state_group="orthographic_rake",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="raking_left",
            state_group="orthographic_rake",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_DISPLAY,
            component_role="occluder",
            expected="hidden",
            state_group="orthographic_rake",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_MARK,
            component_role="target",
            expected="highlighted",
            state_group="orthographic_rake",
        ),
    ]
    if index == 2:
        requirements.append(
            StateRequirement(predicate=StatePredicate.RESET_TO_IMPORTED, expected=True)
        )
    return tuple(requirements)


def _evidence_requirements() -> tuple[EvidenceRequirement, ...]:
    return (
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_VISIBLE,
            component_role="target",
            minimum=0.01,
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_SCREEN_SPAN,
            component_role="target",
            minimum=0.2,
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_HIGHLIGHTED,
            component_role="target",
            minimum=0.005,
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.COMPONENT_ABSENT,
            component_role="occluder",
            maximum=0.0,
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.PROJECTION,
            expected="perspective",
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.FOCAL_LENGTH,
            expected=85.0,
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.ILLUMINATION,
            expected="neutral_studio",
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_VISIBLE,
            component_role="target",
            minimum=0.01,
            render_group="orthographic_rake",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_SCREEN_SPAN,
            component_role="target",
            minimum=0.2,
            render_group="orthographic_rake",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_HIGHLIGHTED,
            component_role="target",
            minimum=0.005,
            render_group="orthographic_rake",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.COMPONENT_ABSENT,
            component_role="occluder",
            maximum=0.0,
            render_group="orthographic_rake",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.PROJECTION,
            expected="orthographic",
            render_group="orthographic_rake",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.ILLUMINATION,
            expected="raking_left",
            render_group="orthographic_rake",
        ),
        EvidenceRequirement(kind=EvidenceKind.CONTACT_SHEET_PANELS, expected=9),
        EvidenceRequirement(kind=EvidenceKind.DISTINCT_VIEWS, minimum=9),
    )


def _answer_schema() -> dict[str, JsonValue]:
    values: dict[str, JsonValue] = {
        f"{role}_{suffix}": {
            "type": "string",
            "description": (
                f"stable component ID for the {role}"
                if suffix == "component_id"
                else f"complete hierarchy path for the {role}"
            ),
        }
        for role in ("target", "reference", "occluder")
        for suffix in ("component_id", "path")
    }
    return {
        "type": "object",
        "required": ["status", "values"],
        "additionalProperties": False,
        "properties": {
            "status": {"enum": ["answered", "indeterminate"]},
            "values": {
                "type": "object",
                "properties": values,
                "required": [key for key in values],
                "additionalProperties": False,
            },
            "conflicting_component_ids": {"type": "array", "items": {"type": "string"}},
            "missing_capabilities": {"type": "array", "items": {"type": "string"}},
        },
    }


def _prepare_staging(staging: Path, generator_sha256: str) -> None:
    checkpoint = staging / "checkpoint.json"
    current: object = None
    if checkpoint.is_file():
        try:
            current = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
    if staging.exists() and (
        not isinstance(current, dict) or current.get("generator_sha256") != generator_sha256
    ):
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        _write_checkpoint(staging, generator_sha256, set())


def _read_checkpoint(staging: Path, generator_sha256: str) -> set[str]:
    payload = json.loads((staging / "checkpoint.json").read_text(encoding="utf-8"))
    if payload.get("generator_sha256") != generator_sha256:
        raise RuntimeError("curated task checkpoint belongs to a different input")
    completed = payload.get("completed")
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise RuntimeError("curated task checkpoint has an invalid completed-model list")
    return set(completed)


def _write_checkpoint(staging: Path, generator_sha256: str, completed: set[str]) -> None:
    _write_json(
        staging / "checkpoint.json",
        {"generator_sha256": generator_sha256, "completed": sorted(completed)},
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"staged curated model has changed: {destination.name}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)
