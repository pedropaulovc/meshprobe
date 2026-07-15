"""Public and evaluator-private contracts for MeshProbe qualification episodes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
OpaqueId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{7,127}$")]


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Operation(StrEnum):
    SCENE_OPEN = "scene.open"
    SCENE_DESCRIBE = "scene.describe"
    COMPONENT_FIND = "component.find"
    COMPONENT_INSPECT = "component.inspect"
    VIEW_SET = "view.set"
    VIEW_ORBIT = "view.orbit"
    ILLUMINATION_SET = "illumination.set"
    COMPONENT_DISPLAY = "component.display"
    COMPONENT_MARK = "component.mark"
    RENDER_IMAGE = "render.image"
    RENDER_CONTACT_SHEET = "render.contact_sheet"
    SESSION_RESET = "session.reset"


class TaskFamily(StrEnum):
    COMPONENT_DISCOVERY = "component_discovery"
    SPATIAL_RELATIONSHIP = "spatial_relationship"
    IDENTITY_VERIFICATION = "identity_verification"
    HIDDEN_GEOMETRY = "hidden_geometry"
    OCCLUSION_REASONING = "occlusion_reasoning"
    PROJECTION_REASONING = "projection_reasoning"
    LENS_REASONING = "lens_reasoning"
    SURFACE_DETAIL = "surface_detail_reasoning"
    SILHOUETTE_GAP = "silhouette_gap_reasoning"
    FULL_INVESTIGATION = "full_investigation"
    NEGATIVE_AMBIGUOUS = "negative_ambiguous"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INDETERMINATE = "indeterminate"


class StructuredAnswer(EvalModel):
    status: AnswerStatus
    values: dict[str, JsonValue] = {}
    conflicting_component_ids: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_indeterminate_context(self) -> Self:
        if self.status is AnswerStatus.ANSWERED:
            if self.conflicting_component_ids or self.missing_capabilities:
                raise ValueError(
                    "answered results cannot declare conflicts or missing capabilities"
                )
            return self
        if not self.conflicting_component_ids and not self.missing_capabilities:
            raise ValueError("indeterminate results must explain a conflict or missing capability")
        return self


class EpisodeBudgets(EvalModel):
    tool_calls: Annotated[int, Field(ge=1, le=500)] = 80
    renders: Annotated[int, Field(ge=0, le=100)] = 16
    total_pixels: Annotated[int, Field(ge=0, le=2_000_000_000)] = 80_000_000
    output_bytes: Annotated[int, Field(ge=0, le=20_000_000_000)] = 1_000_000_000
    wall_seconds: Annotated[float, Field(gt=0, le=7_200, allow_inf_nan=False)] = 600


class EvidenceKind(StrEnum):
    TARGET_VISIBLE = "target_visible"
    TARGET_HIGHLIGHTED = "target_highlighted"
    COMPONENT_ABSENT = "component_absent"
    PROJECTION = "projection"
    FOCAL_LENGTH = "focal_length"
    ILLUMINATION = "illumination"
    TARGET_CONTRAST = "target_contrast"
    CONTACT_SHEET_PANELS = "contact_sheet_panels"
    DISTINCT_VIEWS = "distinct_views"


class EvidenceRequirement(EvalModel):
    kind: EvidenceKind
    component_role: str | None = None
    panel_caption: str | None = None
    expected: JsonValue = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("evidence minimum must not exceed maximum")
        component_kinds = {
            EvidenceKind.TARGET_VISIBLE,
            EvidenceKind.TARGET_HIGHLIGHTED,
            EvidenceKind.COMPONENT_ABSENT,
            EvidenceKind.TARGET_CONTRAST,
        }
        if self.kind in component_kinds and self.component_role is None:
            raise ValueError(f"{self.kind} evidence requires component_role")
        return self


class StatePredicate(StrEnum):
    PROJECTION_MODE = "projection_mode"
    FOCAL_LENGTH_MM = "focal_length_mm"
    ILLUMINATION_PRESET = "illumination_preset"
    COMPONENT_DISPLAY = "component_display"
    COMPONENT_MARK = "component_mark"
    RESET_TO_IMPORTED = "reset_to_imported"


class StateRequirement(EvalModel):
    predicate: StatePredicate
    expected: JsonValue
    component_role: str | None = None

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if (
            self.predicate
            in {
                StatePredicate.COMPONENT_DISPLAY,
                StatePredicate.COMPONENT_MARK,
            }
            and self.component_role is None
        ):
            raise ValueError(f"{self.predicate} requires component_role")
        return self


class EpisodeSpec(EvalModel):
    schema_version: Literal[1] = 1
    episode_id: OpaqueId
    model_file: Annotated[str, StringConstraints(pattern=r"^[a-z0-9_-]+\.glb$")]
    model_sha256: Sha256
    family: TaskFamily
    prompt: Annotated[str, StringConstraints(min_length=20, max_length=8_000)]
    answer_schema: dict[str, JsonValue]
    required_operations: tuple[Operation, ...] = Field(min_length=1)
    budgets: EpisodeBudgets = EpisodeBudgets()

    @model_validator(mode="after")
    def validate_operation_contract(self) -> Self:
        if len(set(self.required_operations)) != len(self.required_operations):
            raise ValueError("required operations must be unique")
        if self.required_operations[0] is not Operation.SCENE_OPEN:
            raise ValueError("scene.open must be the first required operation")
        if self.family is TaskFamily.FULL_INVESTIGATION and set(self.required_operations) != set(
            Operation
        ):
            raise ValueError("full investigations must require every public operation")
        return self


class EpisodeGroundTruth(EvalModel):
    schema_version: Literal[1] = 1
    episode_id: OpaqueId
    model_sha256: Sha256
    answer: StructuredAnswer
    component_roles: dict[str, str]
    component_paths: dict[str, str]
    state_requirements: tuple[StateRequirement, ...] = ()
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    required_operations: tuple[Operation, ...] = Field(min_length=1)
    forbidden_public_tokens: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_roles_and_operations(self) -> Self:
        if self.component_roles.keys() != self.component_paths.keys():
            raise ValueError("component role IDs and paths must cover the same roles")
        state_roles = {
            requirement.component_role
            for requirement in self.state_requirements
            if requirement.component_role is not None
        }
        evidence_roles = {
            requirement.component_role
            for requirement in self.evidence_requirements
            if requirement.component_role is not None
        }
        referenced_roles = state_roles | evidence_roles
        unknown = referenced_roles - self.component_roles.keys()
        if unknown:
            raise ValueError(f"requirements reference unknown component roles: {sorted(unknown)}")
        if len(set(self.required_operations)) != len(self.required_operations):
            raise ValueError("required operations must be unique")
        return self


class TraceStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CRASHED = "crashed"


class TraceEvent(EvalModel):
    sequence: Annotated[int, Field(ge=1)]
    request_id: str
    operation: Operation
    status: TraceStatus
    arguments: JsonValue
    result: JsonValue = None
    error_code: str | None = None
    state_before_sha256: Sha256 | None = None
    state_after_sha256: Sha256 | None = None
    started_monotonic: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    elapsed_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_status_fields(self) -> Self:
        if self.status is TraceStatus.ACCEPTED and self.error_code is not None:
            raise ValueError("accepted trace events cannot have an error code")
        if self.status is not TraceStatus.ACCEPTED and self.error_code is None:
            raise ValueError("failed trace events require an error code")
        return self


class EpisodeSubmission(EvalModel):
    episode_id: OpaqueId
    answer: StructuredAnswer
    evidence_manifest_paths: tuple[str, ...] = ()


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class GateResult(EvalModel):
    gate: str
    status: GateStatus
    message: str
    details: dict[str, JsonValue] = {}


class EpisodeReport(EvalModel):
    schema_version: Literal[1] = 1
    episode_id: OpaqueId
    family: TaskFamily
    gates: tuple[GateResult, ...]
    tool_calls: Annotated[int, Field(ge=0)]
    renders: Annotated[int, Field(ge=0)]
    total_pixels: Annotated[int, Field(ge=0)]
    output_bytes: Annotated[int, Field(ge=0)]
    wall_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @property
    def passed(self) -> bool:
        return all(gate.status is not GateStatus.FAIL for gate in self.gates)


class CorpusTier(StrEnum):
    SMOKE = "smoke"
    PULL_REQUEST = "pull_request"
    NIGHTLY = "nightly"
    RELEASE = "release"
    PRIVATE = "private"


class CorpusManifest(EvalModel):
    schema_version: Literal[1] = 1
    corpus_version: str
    tier: CorpusTier
    generator_sha256: Sha256
    episodes: tuple[OpaqueId, ...] = Field(min_length=1)
    episode_sha256: dict[OpaqueId, Sha256]

    @model_validator(mode="after")
    def validate_episode_hashes(self) -> Self:
        if len(set(self.episodes)) != len(self.episodes):
            raise ValueError("tier episodes must be unique")
        if set(self.episodes) != self.episode_sha256.keys():
            raise ValueError("episode_sha256 must exactly cover tier episodes")
        return self
