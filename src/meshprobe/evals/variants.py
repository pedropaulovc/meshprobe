"""Metamorphic answer relations for controlled procedural variants."""

from __future__ import annotations

from dataclasses import dataclass

from meshprobe.evals.generators import GeneratedModel, MetamorphicVariant


@dataclass(frozen=True)
class MetamorphicRelation:
    invariant_fields: tuple[str, ...]
    transformed_fields: tuple[str, ...] = ()


_GEOMETRIC_FIELDS = (
    "contacted_shaft",
    "clip_side",
    "arrow_direction",
    "near_ring_role",
    "clearance_mm",
    "handedness",
)


def relation_for(variant: MetamorphicVariant) -> MetamorphicRelation:
    if variant is MetamorphicVariant.MIRROR:
        return MetamorphicRelation(
            invariant_fields=("contacted_shaft", "near_ring_role", "clearance_mm"),
            transformed_fields=("clip_side", "arrow_direction", "handedness"),
        )
    if variant is MetamorphicVariant.RESCALE:
        return MetamorphicRelation(
            invariant_fields=(
                "contacted_shaft",
                "clip_side",
                "arrow_direction",
                "near_ring_role",
                "handedness",
            ),
            transformed_fields=("clearance_mm",),
        )
    return MetamorphicRelation(invariant_fields=_GEOMETRIC_FIELDS)


def verify_relation(base: GeneratedModel, transformed: GeneratedModel) -> None:
    if base.family is not transformed.family or base.seed != transformed.seed:
        raise ValueError("metamorphic models must share a family and base seed")
    relation = relation_for(transformed.variant)
    mismatches = [
        field
        for field in relation.invariant_fields
        if base.semantic_answers[field] != transformed.semantic_answers[field]
    ]
    if mismatches:
        raise ValueError(f"metamorphic invariant fields changed: {mismatches}")
    if transformed.variant is MetamorphicVariant.MIRROR:
        _verify_mirror(base, transformed)
    if transformed.variant is MetamorphicVariant.RESCALE:
        _verify_rescale(base, transformed)


def _verify_mirror(base: GeneratedModel, transformed: GeneratedModel) -> None:
    expected_side = "negative" if base.semantic_answers["clip_side"] == "positive" else "positive"
    if transformed.semantic_answers["clip_side"] != expected_side:
        raise ValueError("mirror variant did not flip clip_side")
    expected_direction = {"left": "right", "right": "left"}.get(
        str(base.semantic_answers["arrow_direction"]),
        base.semantic_answers["arrow_direction"],
    )
    if transformed.semantic_answers["arrow_direction"] != expected_direction:
        raise ValueError("mirror variant did not transform arrow_direction")
    if transformed.semantic_answers["handedness"] == base.semantic_answers["handedness"]:
        raise ValueError("mirror variant did not change handedness")


def _verify_rescale(base: GeneratedModel, transformed: GeneratedModel) -> None:
    base_clearance = float(base.semantic_answers["clearance_mm"])
    transformed_clearance = float(transformed.semantic_answers["clearance_mm"])
    ratio = transformed_clearance / base_clearance
    if not math_is_close_to_scale(ratio):
        raise ValueError(f"rescale variant changed clearance by unexpected ratio {ratio}")


def math_is_close_to_scale(ratio: float) -> bool:
    return abs(ratio - 0.1) < 1e-5 or abs(ratio - 10.0) < 1e-5
