from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from meshprobe.evals.generators import (
    PUBLIC_GENERATOR_FAMILIES,
    GeneratorFamily,
    MetamorphicVariant,
    build_model,
    generate_episodes,
    publish_model,
)
from meshprobe.evals.schemas import (
    EVALUATED_OPERATIONS,
    AnswerStatus,
    EpisodeClass,
    EvidenceKind,
    Operation,
    TaskFamily,
)
from meshprobe.evals.variants import relation_for, verify_relation


def test_public_generator_surface_contains_only_released_families() -> None:
    assert len(PUBLIC_GENERATOR_FAMILIES) == 16
    assert set(PUBLIC_GENERATOR_FAMILIES) == set(GeneratorFamily)


def test_seeded_model_and_episodes_are_deterministic(tmp_path: Path) -> None:
    first = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 9), tmp_path / "first")
    second = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 9), tmp_path / "second")

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert generate_episodes(first) == generate_episodes(second)


def test_component_names_and_node_order_do_not_encode_roles() -> None:
    models = [build_model(GeneratorFamily.HIDDEN_CLIP, seed) for seed in range(12)]
    role_suffixes: dict[str, set[str]] = {}
    for model in models:
        assert model.role_order is not None
        assert set(model.role_order) == {component.role for component in model.scene.components}
        names = {component.role: component.name for component in model.scene.components}
        for role, name in names.items():
            role_suffixes.setdefault(role, set()).add(name.rsplit("-", 1)[-1])

    assert all(
        len(suffixes) > 1 for role, suffixes in role_suffixes.items() if role != "extra_occluder"
    )
    assert len({model.role_order for model in models}) > 1


@pytest.mark.parametrize("family", tuple(GeneratorFamily))
def test_every_family_generates_four_operation_aware_episodes(
    tmp_path: Path, family: GeneratorFamily
) -> None:
    model = publish_model(build_model(family, 3), tmp_path)
    episodes = generate_episodes(model)

    assert len(episodes) == 4
    assert episodes[0].spec.family is TaskFamily.COMPONENT_DISCOVERY
    assert episodes[-1].spec.family is TaskFamily.FULL_INVESTIGATION
    assert set(episodes[-1].spec.required_operations) == set(EVALUATED_OPERATIONS)
    assert episodes[-1].ground_truth.state_requirements[-1].predicate == "reset_to_imported"
    for episode in episodes[1:]:
        assert {
            requirement.state_group
            for requirement in episode.ground_truth.state_requirements
            if requirement.state_group is not None
        } == {"context_85", "surface_left", "surface_right", "gap_backlit"}
        assert {
            requirement.render_group
            for requirement in episode.ground_truth.evidence_requirements
            if requirement.render_group is not None
        } == {"context_85", "surface_left", "surface_right", "gap_backlit"}


def test_seed_rotation_covers_every_metamorphic_variant() -> None:
    variants = {
        build_model(GeneratorFamily.CLEARANCE_SLOT, seed).variant
        for seed in range(len(MetamorphicVariant))
    }
    assert variants == set(MetamorphicVariant)


def test_missing_material_does_not_demand_impossible_surface_contrast(tmp_path: Path) -> None:
    missing_episodes = generate_episodes(
        publish_model(build_model(GeneratorFamily.MISSING_MATERIAL, 0), tmp_path / "missing")
    )
    missing = missing_episodes[1]
    positive = generate_episodes(
        publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 0), tmp_path / "positive")
    )[1]

    assert EvidenceKind.TARGET_CONTRAST not in {
        requirement.kind for requirement in missing.ground_truth.evidence_requirements
    }
    assert (
        sum(
            requirement.kind is EvidenceKind.TARGET_CONTRAST
            for requirement in positive.ground_truth.evidence_requirements
        )
        == 2
    )
    assert all(
        episode.spec.episode_class is EpisodeClass.NEGATIVE
        and episode.ground_truth.answer.status is AnswerStatus.INDETERMINATE
        and episode.ground_truth.answer.missing_capabilities == ("arrow material",)
        for episode in missing_episodes[1:]
    )


def test_ambiguous_twin_requires_indeterminate_answer_in_every_inspection(
    tmp_path: Path,
) -> None:
    episodes = generate_episodes(
        publish_model(build_model(GeneratorFamily.AMBIGUOUS_TWIN, 0), tmp_path)
    )

    assert all(
        episode.spec.episode_class is EpisodeClass.NEGATIVE
        and episode.ground_truth.answer.status is AnswerStatus.INDETERMINATE
        and len(episode.ground_truth.answer.conflicting_component_ids) == 2
        for episode in episodes[1:]
    )


@pytest.mark.parametrize("variant", tuple(MetamorphicVariant))
def test_metamorphic_variants_obey_declared_answer_relations(
    variant: MetamorphicVariant,
) -> None:
    base = build_model(
        GeneratorFamily.HIDDEN_CLIP,
        12,
        variant=MetamorphicVariant.MATERIAL,
    )
    transformed = build_model(GeneratorFamily.HIDDEN_CLIP, 12, variant=variant)
    verify_relation(base, transformed)


def test_rigid_transform_recomputes_view_dependent_nearer_ring() -> None:
    base = build_model(
        GeneratorFamily.HIDDEN_CLIP,
        12,
        variant=MetamorphicVariant.MATERIAL,
    )
    transformed = build_model(
        GeneratorFamily.HIDDEN_CLIP,
        12,
        variant=MetamorphicVariant.RIGID_TRANSFORM,
    )

    assert base.semantic_answers["near_ring_role"] != transformed.semantic_answers["near_ring_role"]
    assert relation_for(MetamorphicVariant.RIGID_TRANSFORM).transformed_fields == (
        "near_ring_role",
    )


def test_metamorphic_verifier_rejects_each_broken_relation() -> None:
    base = build_model(
        GeneratorFamily.HIDDEN_CLIP,
        12,
        variant=MetamorphicVariant.MATERIAL,
    )
    with pytest.raises(ValueError, match="share a family"):
        verify_relation(base, build_model(GeneratorFamily.STAMPED_ARROW, 12))

    invariant = replace(base, semantic_answers=base.semantic_answers | {"contacted_shaft": 99})
    with pytest.raises(ValueError, match="invariant fields"):
        verify_relation(base, invariant)

    mirror = build_model(
        GeneratorFamily.HIDDEN_CLIP,
        12,
        variant=MetamorphicVariant.MIRROR,
    )
    for field, value, message in (
        ("clip_side", base.semantic_answers["clip_side"], "clip_side"),
        ("arrow_direction", "diagonal", "arrow_direction"),
        ("handedness", base.semantic_answers["handedness"], "handedness"),
    ):
        broken = replace(
            mirror,
            semantic_answers=mirror.semantic_answers | {field: value},
        )
        with pytest.raises(ValueError, match=message):
            verify_relation(base, broken)

    rescale = build_model(
        GeneratorFamily.HIDDEN_CLIP,
        12,
        variant=MetamorphicVariant.RESCALE,
    )
    broken_scale = replace(
        rescale,
        semantic_answers=rescale.semantic_answers
        | {"clearance_mm": base.semantic_answers["clearance_mm"]},
    )
    with pytest.raises(ValueError, match="unexpected ratio"):
        verify_relation(base, broken_scale)


def test_private_component_ids_do_not_appear_in_public_prompt(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.NESTED_HIERARCHY, 4), tmp_path)
    for episode in generate_episodes(model):
        assert all(
            component_id not in episode.spec.prompt for component_id in model.component_ids.values()
        )


def test_public_answer_schema_names_every_scored_field(tmp_path: Path) -> None:
    model = publish_model(build_model(GeneratorFamily.HIDDEN_CLIP, 4), tmp_path)
    episodes = generate_episodes(model)

    discovery = episodes[0]
    assert set(discovery.ground_truth.answer.values) == {
        "target_component_id",
        "target_component_path",
    }
    for episode in episodes:
        value_schema = episode.spec.answer_schema["properties"]["values"]
        assert isinstance(value_schema, dict)
        properties = value_schema["properties"]
        assert isinstance(properties, dict)
        if episode.ground_truth.answer.status is AnswerStatus.ANSWERED:
            assert set(episode.ground_truth.answer.values) == set(properties)


def test_every_operation_has_positive_negative_and_adversarial_episodes(
    tmp_path: Path,
) -> None:
    coverage = {operation: set() for operation in EVALUATED_OPERATIONS}
    for family in GeneratorFamily:
        for seed in (0, 1):
            model = publish_model(build_model(family, seed), tmp_path)
            for episode in generate_episodes(model):
                for operation in episode.spec.required_operations:
                    coverage[operation].add(episode.spec.episode_class)

    assert all(classes == set(EpisodeClass) for classes in coverage.values())
    assert coverage[Operation.COMPONENT_OCCLUSION] == set(EpisodeClass)
