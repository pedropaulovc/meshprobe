"""Seeded procedural assemblies and operation-forcing qualification tasks."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import JsonValue

from meshprobe.evals.geometry import (
    CameraSpec,
    ComponentSpec,
    GlbScene,
    MaterialSpec,
    PointLightSpec,
    arrow,
    box,
    cylinder,
)
from meshprobe.evals.schemas import (
    EVALUATED_OPERATIONS,
    AnswerStatus,
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
from meshprobe.identity import stable_component_id


class GeneratorFamily(StrEnum):
    HIDDEN_CLIP = "hidden_clip"
    STAMPED_ARROW = "stamped_arrow"
    COAXIAL_DEPTH = "coaxial_depth"
    CLEARANCE_SLOT = "clearance_slot"
    OCCLUDED_FASTENER = "occluded_fastener"
    AXIAL_GAP = "axial_gap"
    MIRRORED_HANDEDNESS = "mirrored_handedness"
    DUPLICATE_NAME = "duplicate_name"
    NESTED_HIERARCHY = "nested_hierarchy"
    REFLECTIVE_CAVITY = "reflective_cavity"
    LOW_CONTRAST_RELIEF = "low_contrast_relief"
    EXTREME_SCALE = "extreme_scale"
    TRANSPARENT_SHELL = "transparent_shell"
    COPLANAR_MARKER = "coplanar_marker"
    MISSING_MATERIAL = "missing_material"
    AMBIGUOUS_TWIN = "ambiguous_twin"


PUBLIC_GENERATOR_FAMILIES = (
    GeneratorFamily.HIDDEN_CLIP,
    GeneratorFamily.STAMPED_ARROW,
    GeneratorFamily.COAXIAL_DEPTH,
    GeneratorFamily.CLEARANCE_SLOT,
    GeneratorFamily.OCCLUDED_FASTENER,
    GeneratorFamily.AXIAL_GAP,
    GeneratorFamily.MIRRORED_HANDEDNESS,
    GeneratorFamily.DUPLICATE_NAME,
    GeneratorFamily.NESTED_HIERARCHY,
    GeneratorFamily.REFLECTIVE_CAVITY,
    GeneratorFamily.LOW_CONTRAST_RELIEF,
    GeneratorFamily.EXTREME_SCALE,
    GeneratorFamily.TRANSPARENT_SHELL,
    GeneratorFamily.COPLANAR_MARKER,
    GeneratorFamily.MISSING_MATERIAL,
    GeneratorFamily.AMBIGUOUS_TWIN,
)


class MetamorphicVariant(StrEnum):
    RENAME = "rename"
    RIGID_TRANSFORM = "rigid_transform"
    RESCALE = "rescale"
    MIRROR = "mirror"
    MATERIAL = "material"
    EXTRA_OCCLUDER = "extra_occluder"
    REORDER_HIERARCHY = "reorder_hierarchy"
    START_STATE = "start_state"


@dataclass(frozen=True)
class GeneratedModel:
    family: GeneratorFamily
    seed: int
    variant: MetamorphicVariant
    model_id: str
    scene: GlbScene
    role_order: tuple[str, ...] | None
    semantic_answers: dict[str, str | float | int]
    target_name: str


@dataclass(frozen=True)
class PublishedModel:
    generated: GeneratedModel
    path: Path
    sha256: str
    component_paths: dict[str, str]
    component_ids: dict[str, str]


@dataclass(frozen=True)
class GeneratedEpisode:
    spec: EpisodeSpec
    ground_truth: EpisodeGroundTruth


FAMILY_TASKS: dict[GeneratorFamily, TaskFamily] = {
    GeneratorFamily.HIDDEN_CLIP: TaskFamily.HIDDEN_GEOMETRY,
    GeneratorFamily.STAMPED_ARROW: TaskFamily.SURFACE_DETAIL,
    GeneratorFamily.COAXIAL_DEPTH: TaskFamily.PROJECTION_REASONING,
    GeneratorFamily.CLEARANCE_SLOT: TaskFamily.SPATIAL_RELATIONSHIP,
    GeneratorFamily.OCCLUDED_FASTENER: TaskFamily.OCCLUSION_REASONING,
    GeneratorFamily.AXIAL_GAP: TaskFamily.SILHOUETTE_GAP,
    GeneratorFamily.MIRRORED_HANDEDNESS: TaskFamily.SPATIAL_RELATIONSHIP,
    GeneratorFamily.DUPLICATE_NAME: TaskFamily.COMPONENT_DISCOVERY,
    GeneratorFamily.NESTED_HIERARCHY: TaskFamily.IDENTITY_VERIFICATION,
    GeneratorFamily.REFLECTIVE_CAVITY: TaskFamily.HIDDEN_GEOMETRY,
    GeneratorFamily.LOW_CONTRAST_RELIEF: TaskFamily.SURFACE_DETAIL,
    GeneratorFamily.EXTREME_SCALE: TaskFamily.LENS_REASONING,
    GeneratorFamily.TRANSPARENT_SHELL: TaskFamily.OCCLUSION_REASONING,
    GeneratorFamily.COPLANAR_MARKER: TaskFamily.SURFACE_DETAIL,
    GeneratorFamily.MISSING_MATERIAL: TaskFamily.NEGATIVE_AMBIGUOUS,
    GeneratorFamily.AMBIGUOUS_TWIN: TaskFamily.NEGATIVE_AMBIGUOUS,
}

_ADJECTIVES = (
    "amber",
    "brisk",
    "cobalt",
    "delta",
    "ember",
    "frost",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kinetic",
    "lunar",
    "mosaic",
    "nickel",
    "onyx",
    "polar",
    "quartz",
    "raven",
    "solar",
    "tundra",
)
_NOUNS = (
    "anchor",
    "bushing",
    "collar",
    "drum",
    "elbow",
    "flange",
    "gimbal",
    "hub",
    "insert",
    "journal",
    "key",
    "link",
    "mount",
    "needle",
    "orbit",
    "plate",
    "rail",
    "sleeve",
    "trunnion",
    "yoke",
)


def build_model(
    family: GeneratorFamily,
    seed: int,
    *,
    variant: MetamorphicVariant | None = None,
) -> GeneratedModel:
    if seed < 0:
        raise ValueError("generator seed must be non-negative")
    rng = random.Random(_seed_int(family, seed))
    selected_variant = variant or tuple(MetamorphicVariant)[seed % len(MetamorphicVariant)]
    model_id = (
        "mdl_" + hashlib.sha256(f"{family}:{seed}:{selected_variant}".encode()).hexdigest()[:24]
    )
    names = _component_names(rng)
    if selected_variant is MetamorphicVariant.RENAME:
        names = _component_names(random.Random(_seed_int(family, seed) ^ 0x5A17C0DE))
    if family in {
        GeneratorFamily.DUPLICATE_NAME,
        GeneratorFamily.AMBIGUOUS_TWIN,
    }:
        names["distractor"] = names["idler"]

    mirror = (
        -1
        if selected_variant is MetamorphicVariant.MIRROR
        or family is GeneratorFamily.MIRRORED_HANDEDNESS
        else 1
    )
    clip_sign = mirror * rng.choice((-1, 1))
    contacted_shaft = rng.choice((1, 2))
    arrow_direction = rng.choice(("left", "right", "up", "down"))
    if mirror < 0:
        horizontal_flip = {"left": "right", "right": "left"}
        arrow_direction = horizontal_flip.get(arrow_direction, arrow_direction)
    clearance_mm = round(rng.uniform(80.0, 420.0), 3)
    near_ring_position = rng.choice(("near_ring", "far_ring"))

    muted = MaterialSpec("muted-alloy", (0.14, 0.16, 0.18, 1), metallic=0.75, roughness=0.18)
    steel = MaterialSpec("brushed-steel", (0.42, 0.48, 0.54, 1), metallic=0.8, roughness=0.32)
    accent = MaterialSpec("inspection-accent", (0.72, 0.16, 0.05, 1), metallic=0.15)
    dark = MaterialSpec("dark-polymer", (0.018, 0.021, 0.025, 1), roughness=0.78)
    glass = MaterialSpec("clear-shell", (0.2, 0.5, 0.8, 0.22), roughness=0.1)
    base_material = muted if family is GeneratorFamily.REFLECTIVE_CAVITY else steel
    detail_material = (
        base_material
        if family
        in {
            GeneratorFamily.STAMPED_ARROW,
            GeneratorFamily.LOW_CONTRAST_RELIEF,
            GeneratorFamily.COPLANAR_MARKER,
        }
        else accent
    )
    cover_material = glass if family is GeneratorFamily.TRANSPARENT_SHELL else dark
    if selected_variant is MetamorphicVariant.MATERIAL:
        base_material = MaterialSpec("variant-ceramic", (0.62, 0.58, 0.48, 1), roughness=0.86)

    camera_projection = (
        "orthographic" if selected_variant is MetamorphicVariant.START_STATE else "perspective"
    )
    camera_position = (9.0, -10.0, 7.0)
    if family is GeneratorFamily.COAXIAL_DEPTH:
        camera_position = (0.0, -11.0, 3.0)
    scene = GlbScene(
        camera=CameraSpec(
            names["camera"],
            camera_position,
            (0.0, 0.0, 1.8),
            projection=camera_projection,
            focal_length_mm=rng.choice((24.0, 50.0, 120.0)),
            orthographic_height_m=8.0,
        )
    )
    if selected_variant is MetamorphicVariant.START_STATE:
        scene.lights.extend(
            (
                PointLightSpec("cold-key", (-5, -4, 8), (0.55, 0.7, 1.0), 1_500),
                PointLightSpec("warm-fill", (5, 1, 4), (1.0, 0.6, 0.35), 550),
            )
        )

    root_translation = (0.0, 0.0, 0.0)
    root_rotation = (0.0, 0.0, 0.0, 1.0)
    root_scale = (1.0, 1.0, 1.0)
    physical_scale = 1.0
    if selected_variant is MetamorphicVariant.RIGID_TRANSFORM:
        angle = rng.uniform(-math.pi, math.pi)
        root_translation = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.5, 2))
        root_rotation = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
    if selected_variant is MetamorphicVariant.RESCALE:
        scale = rng.choice((0.1, 10.0))
        root_scale = (scale, scale, scale)
        physical_scale = scale

    scene.add(
        ComponentSpec(
            "root",
            names["root"],
            box((8.0, 6.0, 0.45)),
            base_material,
            translation_m=root_translation,
            rotation_xyzw=root_rotation,
            scale=root_scale,
        )
    )
    scene.add(
        ComponentSpec(
            "housing", names["housing"], box((6.5, 4.5, 0.3)), base_material, "root", (0, 0, 0.6)
        )
    )
    shaft_x = {1: -1.7 * mirror, 2: 1.7 * mirror}
    scene.add(
        ComponentSpec(
            "shaft_one",
            names["shaft_one"],
            cylinder(0.34, 2.4),
            steel,
            "housing",
            (shaft_x[1], 0.25, 1.35),
        )
    )
    scene.add(
        ComponentSpec(
            "shaft_two",
            names["shaft_two"],
            cylinder(0.34, 2.4),
            steel,
            "housing",
            (shaft_x[2], 0.25, 1.35),
        )
    )
    contacted_x = shaft_x[contacted_shaft]
    inward = -mirror if contacted_shaft == 2 else mirror
    idler_x = contacted_x + inward * 1.08
    scene.add(
        ComponentSpec(
            "idler",
            names["idler"],
            cylinder(0.74, 0.55),
            accent,
            "housing",
            (idler_x, 0.25, 1.35),
        )
    )
    scene.add(
        ComponentSpec(
            "clip",
            names["clip"],
            box((0.12, 0.85, 0.9)),
            steel,
            "idler",
            (clip_sign * 0.84, 0.0, 0.0),
        )
    )
    cover_y = -1.15 if family is not GeneratorFamily.AXIAL_GAP else -0.75
    scene.add(
        ComponentSpec(
            "cover",
            names["cover"],
            box((5.8, 0.32, 2.8)),
            cover_material,
            "housing",
            (0.0, cover_y, 1.45),
        )
    )
    scene.add(
        ComponentSpec(
            "detail_plate",
            names["detail_plate"],
            box((2.4, 1.8, 0.18)),
            base_material,
            "housing",
            (0.0, 1.2, 2.5),
        )
    )
    arrow_depth = 0.002 if family is GeneratorFamily.COPLANAR_MARKER else 0.015
    if family in {GeneratorFamily.STAMPED_ARROW, GeneratorFamily.LOW_CONTRAST_RELIEF}:
        arrow_depth = 0.007
    scene.add(
        ComponentSpec(
            "arrow",
            names["arrow"],
            arrow(arrow_direction, depth=arrow_depth),
            None if family is GeneratorFamily.MISSING_MATERIAL else detail_material,
            "detail_plate",
            (0.0, 0.0, 0.1 + arrow_depth / 2),
        )
    )
    ring_rotation = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))
    ring_positions = {
        near_ring_position: -0.45,
        "far_ring" if near_ring_position == "near_ring" else "near_ring": 0.55,
    }
    for role, radius in (("near_ring", 0.9), ("far_ring", 0.62)):
        scene.add(
            ComponentSpec(
                role,
                names[role],
                cylinder(radius, 0.2),
                steel,
                "housing",
                (0.0, ring_positions[role], 3.1),
                ring_rotation,
            )
        )
    gap_m = clearance_mm / 1_000
    scene.add(
        ComponentSpec(
            "gap_left",
            names["gap_left"],
            box((1.4, 0.4, 1.6)),
            dark,
            "housing",
            (-0.7 - gap_m / 2, 1.4, 1.2),
        )
    )
    scene.add(
        ComponentSpec(
            "gap_right",
            names["gap_right"],
            box((1.4, 0.4, 1.6)),
            dark,
            "housing",
            (0.7 + gap_m / 2, 1.4, 1.2),
        )
    )
    distractor_scale = 0.08 if family is GeneratorFamily.EXTREME_SCALE else 0.42
    scene.add(
        ComponentSpec(
            "distractor",
            names["distractor"],
            cylinder(distractor_scale, distractor_scale * 1.5),
            accent,
            "housing",
            (2.4 * mirror, -0.55, 2.5),
        )
    )
    if (
        selected_variant is MetamorphicVariant.EXTRA_OCCLUDER
        or family is GeneratorFamily.OCCLUDED_FASTENER
    ):
        scene.add(
            ComponentSpec(
                "extra_occluder",
                names["extra_occluder"],
                box((1.4, 0.25, 2.0)),
                dark,
                "housing",
                (idler_x, -1.55, 1.4),
            )
        )

    order_seed = _seed_int(family, seed) ^ 0xA11CE5EED
    if selected_variant is MetamorphicVariant.REORDER_HIERARCHY:
        order_seed ^= 0x5EED0F00D
    role_order = _random_topological_order(scene, random.Random(order_seed))
    ring_world_positions = {
        role: _apply_root_transform(
            (0.0, ring_y, 3.7),
            translation=root_translation,
            rotation_xyzw=root_rotation,
            scale=root_scale,
        )
        for role, ring_y in ring_positions.items()
    }
    nearer_ring_role = min(
        ring_world_positions,
        key=lambda role: math.dist(ring_world_positions[role], camera_position),
    )
    semantic_answers: dict[str, str | float | int] = {
        "contacted_shaft": contacted_shaft,
        "clip_side": "positive" if clip_sign > 0 else "negative",
        "arrow_direction": arrow_direction,
        "near_ring_role": nearer_ring_role,
        "clearance_mm": round(clearance_mm * physical_scale, 3),
        "handedness": "mirrored" if mirror < 0 else "standard",
    }
    return GeneratedModel(
        family=family,
        seed=seed,
        variant=selected_variant,
        model_id=model_id,
        scene=scene,
        role_order=role_order,
        semantic_answers=semantic_answers,
        target_name=names["idler"],
    )


def publish_model(generated: GeneratedModel, models_dir: Path) -> PublishedModel:
    path = models_dir / f"{generated.model_id}.glb"
    sha256 = generated.scene.write_glb(path, role_order=generated.role_order)
    component_paths = generated.scene.component_paths(role_order=generated.role_order)
    component_ids = {
        role: stable_component_id(sha256, component_path)
        for role, component_path in component_paths.items()
    }
    return PublishedModel(
        generated=generated,
        path=path,
        sha256=sha256,
        component_paths=component_paths,
        component_ids=component_ids,
    )


def generate_episodes(
    model: PublishedModel,
    *,
    model_source: ModelSource = ModelSource.PROCEDURAL,
) -> tuple[GeneratedEpisode, ...]:
    tasks = (
        (TaskFamily.COMPONENT_DISCOVERY, _discovery_operations()),
        (FAMILY_TASKS[model.generated.family], _reasoning_operations()),
        (TaskFamily.PROJECTION_REASONING, _evidence_operations()),
        (TaskFamily.FULL_INVESTIGATION, EVALUATED_OPERATIONS),
    )
    return tuple(
        _episode(model, index, family, operations, model_source)
        for index, (family, operations) in enumerate(tasks)
    )


def _episode(
    model: PublishedModel,
    index: int,
    task_family: TaskFamily,
    operations: tuple[Operation, ...],
    model_source: ModelSource,
) -> GeneratedEpisode:
    digest = hashlib.sha256(
        f"{model.generated.family}:{model.generated.seed}:{index}".encode()
    ).hexdigest()
    episode_id = f"ep_{digest[:24]}"
    answers = _answer_values(model, index)
    status = AnswerStatus.ANSWERED
    conflicts: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    if model.generated.family is GeneratorFamily.AMBIGUOUS_TWIN and index > 0:
        status = AnswerStatus.INDETERMINATE
        answers = {}
        conflicts = (model.component_ids["idler"], model.component_ids["distractor"])
    if model.generated.family is GeneratorFamily.MISSING_MATERIAL and index > 0:
        status = AnswerStatus.INDETERMINATE
        answers = {}
        missing = ("arrow material",)

    prompt = _prompt(model, task_family, index)
    answer = StructuredAnswer(
        status=status,
        values=answers,
        conflicting_component_ids=conflicts,
        missing_capabilities=missing,
    )
    spec = EpisodeSpec(
        episode_id=episode_id,
        model_file=model.path.name,
        model_sha256=model.sha256,
        family=task_family,
        episode_class=_episode_class(model, index),
        difficulty=tuple(Difficulty)[index],
        model_source=model_source,
        prompt=prompt,
        answer_schema=_answer_schema(index),
        required_operations=operations,
        budgets=EpisodeBudgets(
            tool_calls=40 if index == 0 else 120,
            renders=0 if index == 0 else 24,
            total_pixels=0 if index == 0 else 120_000_000,
            output_bytes=10_000_000 if index == 0 else 1_500_000_000,
            wall_seconds=180 if index == 0 else 900,
        ),
    )
    state_requirements = () if index == 0 else _state_requirements(index)
    evidence_requirements = () if index == 0 else _evidence_requirements(model, index)
    truth = EpisodeGroundTruth(
        episode_id=episode_id,
        model_sha256=model.sha256,
        generator_family=model.generated.family.value,
        answer=answer,
        component_roles=model.component_ids,
        component_paths=model.component_paths,
        state_requirements=state_requirements,
        evidence_requirements=evidence_requirements,
        required_operations=operations,
        forbidden_public_tokens=tuple(sorted(model.component_ids.values())),
    )
    return GeneratedEpisode(spec=spec, ground_truth=truth)


def _prompt(model: PublishedModel, task_family: TaskFamily, index: int) -> str:
    target = model.generated.target_name
    if index == 0:
        return (
            f"Open the assigned model, locate every component named {target!r}, inspect the exact "
            "hierarchy candidates, and return target_component_id plus target_component_path for "
            "the intended idler."
        )
    qualification = ""
    if model.generated.family is GeneratorFamily.MISSING_MATERIAL:
        qualification = (
            " The conclusion also requires the arrow material. If the capability report proves "
            "that material is absent, return indeterminate and name the missing capability."
        )
    if model.generated.family is GeneratorFamily.AMBIGUOUS_TWIN:
        qualification = (
            " If the named idler cannot be uniquely distinguished, return indeterminate with "
            "every conflicting stable component ID in hierarchy order."
        )
    if index == 1:
        return (
            f"Investigate {target!r}. Determine the numbered shaft it contacts, "
            "the local-X side carrying its retaining clip, the stamped arrow direction, and which "
            "coaxial ring is nearer the starting camera. Submit an 85 mm perspective context "
            "render with the cover hidden and target highlighted, isolated orthographic arrow "
            "renders under both raking directions, and an orthographic backlit gap render. Return "
            "the exact fields declared by the answer schema." + qualification
        )
    if index == 2:
        return (
            f"Produce comparative evidence for {target!r}: an 85 mm perspective context render, an "
            "orthographic clearance view, opposing raking-light detail evidence, "
            "a backlit gap view, "
            "and a focused 3x3 contact sheet. Report every field declared by the answer schema."
            + qualification
        )
    return (
        f"Perform a complete inspection of {target!r}. Exercise every MeshProbe operation, "
        "identify its contacted shaft and clip side, read the stamped arrow, "
        "distinguish the nearer coaxial ring, measure the declared clearance, hide the cover, "
        "highlight the target, submit "
        "the focused contact sheet and supporting renders, then reset the session before answering."
        + qualification
    )


def _answer_values(model: PublishedModel, index: int) -> dict[str, JsonValue]:
    values: dict[str, JsonValue] = {
        "target_component_id": model.component_ids["idler"],
        "target_component_path": model.component_paths["idler"],
    }
    if index == 0:
        return values
    contacted_shaft = int(model.generated.semantic_answers["contacted_shaft"])
    near_ring_role = str(model.generated.semantic_answers["near_ring_role"])
    values.update(
        {
            "contacted_shaft_component_id": model.component_ids[
                "shaft_one" if contacted_shaft == 1 else "shaft_two"
            ],
            "clip_side": model.generated.semantic_answers["clip_side"],
            "arrow_direction": model.generated.semantic_answers["arrow_direction"],
            "nearer_ring_component_id": model.component_ids[near_ring_role],
        }
    )
    if index >= 2:
        values["clearance_mm"] = model.generated.semantic_answers["clearance_mm"]
    return values


def _answer_schema(index: int) -> dict[str, JsonValue]:
    properties: dict[str, JsonValue] = {
        "target_component_id": {"type": "string", "description": "stable component ID"},
        "target_component_path": {"type": "string", "description": "complete hierarchy path"},
    }
    if index > 0:
        properties.update(
            {
                "contacted_shaft_component_id": {
                    "type": "string",
                    "description": "stable ID of the shaft geometrically contacted by the idler",
                },
                "clip_side": {
                    "enum": ["positive", "negative"],
                    "description": "retaining-clip side along the idler parent-local X axis",
                },
                "arrow_direction": {"enum": ["left", "right", "up", "down"]},
                "nearer_ring_component_id": {
                    "type": "string",
                    "description": "stable ID of the coaxial ring nearer the starting camera",
                },
            }
        )
    if index >= 2:
        properties["clearance_mm"] = {
            "type": "number",
            "description": "orthographically measured gap between the clearance blocks in mm",
        }
    required_values: list[JsonValue] = [key for key in properties]
    return {
        "type": "object",
        "required": ["status", "values"],
        "additionalProperties": False,
        "properties": {
            "status": {"enum": ["answered", "indeterminate"]},
            "values": {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            },
            "conflicting_component_ids": {"type": "array", "items": {"type": "string"}},
            "missing_capabilities": {"type": "array", "items": {"type": "string"}},
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "answered"}}},
                "then": {"properties": {"values": {"required": required_values}}},
            }
        ],
    }


def _episode_class(model: PublishedModel, index: int) -> EpisodeClass:
    if (
        model.generated.family
        in {
            GeneratorFamily.AMBIGUOUS_TWIN,
            GeneratorFamily.MISSING_MATERIAL,
        }
        and index > 0
    ):
        return EpisodeClass.NEGATIVE
    if index == 0 or (index == 3 and model.generated.seed % 4 == 0):
        return EpisodeClass.POSITIVE
    return EpisodeClass.ADVERSARIAL


def _discovery_operations() -> tuple[Operation, ...]:
    return (
        Operation.SCENE_OPEN,
        Operation.SESSION_SNAPSHOT,
        Operation.COMPONENT_FIND,
        Operation.COMPONENT_INSPECT,
    )


def _reasoning_operations() -> tuple[Operation, ...]:
    return (
        Operation.SCENE_OPEN,
        Operation.SESSION_SNAPSHOT,
        Operation.COMPONENT_FIND,
        Operation.COMPONENT_INSPECT,
        Operation.VIEW_ORBIT,
        Operation.ILLUMINATION_SET,
        Operation.COMPONENT_DISPLAY,
        Operation.COMPONENT_MARK,
        Operation.RENDER_IMAGE,
    )


def _evidence_operations() -> tuple[Operation, ...]:
    return (
        Operation.SCENE_OPEN,
        Operation.COMPONENT_FIND,
        Operation.VIEW_SET,
        Operation.VIEW_ORBIT,
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
            component_role="cover",
            expected="hidden",
            state_group="context_85",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_MARK,
            component_role="idler",
            expected="highlighted",
            state_group="context_85",
        ),
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
            state_group="surface_left",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="raking_left",
            state_group="surface_left",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_DISPLAY,
            component_role="arrow",
            expected="isolated",
            state_group="surface_left",
        ),
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
            state_group="surface_right",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="raking_right",
            state_group="surface_right",
        ),
        StateRequirement(
            predicate=StatePredicate.COMPONENT_DISPLAY,
            component_role="arrow",
            expected="isolated",
            state_group="surface_right",
        ),
        StateRequirement(
            predicate=StatePredicate.PROJECTION_MODE,
            expected="orthographic",
            state_group="gap_backlit",
        ),
        StateRequirement(
            predicate=StatePredicate.ILLUMINATION_PRESET,
            expected="backlit",
            state_group="gap_backlit",
        ),
    ]
    if index == 3:
        requirements.append(
            StateRequirement(predicate=StatePredicate.RESET_TO_IMPORTED, expected=True)
        )
    return tuple(requirements)


def _evidence_requirements(model: PublishedModel, index: int) -> tuple[EvidenceRequirement, ...]:
    requirements = [
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_VISIBLE,
            component_role="idler",
            minimum=0.02,
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_HIGHLIGHTED,
            component_role="idler",
            minimum=0.01,
            render_group="context_85",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.COMPONENT_ABSENT,
            component_role="cover",
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
            component_role="arrow",
            minimum=0.005,
            render_group="surface_left",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.PROJECTION,
            expected="orthographic",
            render_group="surface_left",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.ILLUMINATION,
            expected="raking_left",
            render_group="surface_left",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_VISIBLE,
            component_role="arrow",
            minimum=0.005,
            render_group="surface_right",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.PROJECTION,
            expected="orthographic",
            render_group="surface_right",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.ILLUMINATION,
            expected="raking_right",
            render_group="surface_right",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_VISIBLE,
            component_role="gap_left",
            minimum=0.001,
            render_group="gap_backlit",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.TARGET_VISIBLE,
            component_role="gap_right",
            minimum=0.001,
            render_group="gap_backlit",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.PROJECTION,
            expected="orthographic",
            render_group="gap_backlit",
        ),
        EvidenceRequirement(
            kind=EvidenceKind.ILLUMINATION,
            expected="backlit",
            render_group="gap_backlit",
        ),
    ]
    if model.generated.family is not GeneratorFamily.MISSING_MATERIAL:
        requirements.extend(
            (
                EvidenceRequirement(
                    kind=EvidenceKind.TARGET_CONTRAST,
                    component_role="arrow",
                    minimum=0.04,
                    render_group="surface_left",
                ),
                EvidenceRequirement(
                    kind=EvidenceKind.TARGET_CONTRAST,
                    component_role="arrow",
                    minimum=0.04,
                    render_group="surface_right",
                ),
            )
        )
    if index >= 2:
        requirements.extend(
            (
                EvidenceRequirement(kind=EvidenceKind.CONTACT_SHEET_PANELS, expected=9),
                EvidenceRequirement(kind=EvidenceKind.DISTINCT_VIEWS, minimum=9),
            )
        )
    return tuple(requirements)


def _component_names(rng: random.Random) -> dict[str, str]:
    roles = (
        "root",
        "housing",
        "shaft_one",
        "shaft_two",
        "idler",
        "clip",
        "cover",
        "detail_plate",
        "arrow",
        "near_ring",
        "far_ring",
        "gap_left",
        "gap_right",
        "distractor",
        "extra_occluder",
        "camera",
    )
    pairs = [(adjective, noun) for adjective in _ADJECTIVES for noun in _NOUNS]
    rng.shuffle(pairs)
    suffixes = rng.sample(range(10, 100), len(roles))
    return {
        role: f"{adjective}-{noun}-{suffix:02d}"
        for role, (adjective, noun), suffix in zip(
            roles,
            pairs[: len(roles)],
            suffixes,
            strict=True,
        )
    }


def _random_topological_order(scene: GlbScene, rng: random.Random) -> tuple[str, ...]:
    pending = {component.role: component.parent_role for component in scene.components}
    emitted: list[str] = []
    while pending:
        ready = [role for role, parent in pending.items() if parent is None or parent in emitted]
        if not ready:
            raise ValueError("component hierarchy contains a cycle")
        rng.shuffle(ready)
        for role in ready:
            emitted.append(role)
            del pending[role]
    return tuple(emitted)


def _apply_root_transform(
    point: tuple[float, float, float],
    *,
    translation: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    scale: tuple[float, float, float],
) -> tuple[float, float, float]:
    scaled = tuple(value * factor for value, factor in zip(point, scale, strict=True))
    x, y, z, w = rotation_xyzw
    q_vector = (x, y, z)
    cross = (
        q_vector[1] * scaled[2] - q_vector[2] * scaled[1],
        q_vector[2] * scaled[0] - q_vector[0] * scaled[2],
        q_vector[0] * scaled[1] - q_vector[1] * scaled[0],
    )
    twice_cross = tuple(2 * value for value in cross)
    nested = (
        q_vector[1] * twice_cross[2] - q_vector[2] * twice_cross[1],
        q_vector[2] * twice_cross[0] - q_vector[0] * twice_cross[2],
        q_vector[0] * twice_cross[1] - q_vector[1] * twice_cross[0],
    )
    rotated = tuple(
        original + w * first + second
        for original, first, second in zip(scaled, twice_cross, nested, strict=True)
    )
    return tuple(offset + value for offset, value in zip(translation, rotated, strict=True))  # type: ignore[return-value]


def _seed_int(family: GeneratorFamily, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"meshprobe:{family}:{seed}".encode()).digest()[:8], "big")
