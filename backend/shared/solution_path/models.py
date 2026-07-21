"""Typed durable records for the shared solution-path engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_ROUTE_STEPS = 32
MAX_MAIN_ROUTE_CHARS = 8_000
MAX_STEP_TITLE_CHARS = 500
MAX_STEP_DESCRIPTION_CHARS = 8_000
MAX_RATIONALE_CHARS = 8_000
MAX_FEEDBACK_CHARS = 8_000
MAX_METADATA_JSON_CHARS = 8_000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


class StepOrdering(str, Enum):
    ORDERED = "ordered"
    UNORDERED = "unordered"


class RouteStep(BaseModel):
    step_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=MAX_STEP_TITLE_CHARS)
    description: str = Field(default="", max_length=MAX_STEP_DESCRIPTION_CHARS)
    status: Literal["pending", "active", "complete", "blocked"] = "pending"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def metadata_is_bounded_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        import json

        try:
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("route step metadata must be JSON serializable") from exc
        if len(encoded) > MAX_METADATA_JSON_CHARS:
            raise ValueError(
                f"route step metadata exceeds {MAX_METADATA_JSON_CHARS} characters"
            )
        return value


class SolutionRoute(BaseModel):
    ordering: StepOrdering = StepOrdering.ORDERED
    steps: list[RouteStep] = Field(default_factory=list, max_length=MAX_ROUTE_STEPS)

    @field_validator("ordering", mode="before")
    @classmethod
    def normalize_parallel_ordering(cls, value: Any) -> Any:
        if isinstance(value, str) and value.lower() in {"parallel", "mixed"}:
            return StepOrdering.UNORDERED
        return value

    @model_validator(mode="after")
    def unique_step_ids(self) -> "SolutionRoute":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("route step_id values must be unique")
        return self


class SolutionPlan(BaseModel):
    run_id: str
    revision: int = Field(default=1, ge=1)
    main_route: str = Field(default="", max_length=MAX_MAIN_ROUTE_CHARS)
    route: SolutionRoute
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ProposalStatus(str, Enum):
    QUEUED = "queued"
    REVIEWING = "reviewing"
    FOLLOWUP = "followup"
    APPROVED = "approved"
    REJECTED = "rejected"
    USER_REPAIR_REQUIRED = "user_repair_required"


class RepairReason(str, Enum):
    CONTEXT_OVERFLOW = "context_overflow"
    MISSING_API_KEY = "missing_api_key"
    PROVIDER_CONFIGURATION = "provider_configuration"
    PRIVACY_POLICY = "privacy_policy"
    INVALID_REVIEW = "invalid_review"
    UNKNOWN_REVIEWER_FAILURE = "unknown_reviewer_failure"


class PlanProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    base_revision: int = Field(ge=0)
    main_route: str = Field(default="", max_length=MAX_MAIN_ROUTE_CHARS)
    route: SolutionRoute
    rationale: str = Field(default="", max_length=MAX_RATIONALE_CHARS)
    status: ProposalStatus = ProposalStatus.QUEUED
    review_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    feedback: str = Field(default="", max_length=MAX_FEEDBACK_CHARS)
    proposer_role: str = Field(default="validator", max_length=128)
    source_task_id: str | None = Field(default=None, max_length=256)
    source_phase: str | None = Field(default=None, max_length=128)
    source_decision: Literal["accept", "reject", "mixed"] | None = None
    last_error_type: str | None = None
    repair_reason: RepairReason | None = None
    repair_detail: str = Field(default="", max_length=MAX_FEEDBACK_CHARS)
    repair_generation: int | None = Field(default=None, ge=1)
    next_retry_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RouteEdit(BaseModel):
    """One checked edit to a proposal's working route."""

    operation: Literal["add", "update", "delete", "check"]
    step_id: str | None = None
    step: RouteStep | None = None
    after_step_id: str | None = None
    expected_title: str | None = None

    @model_validator(mode="after")
    def validate_edit_shape(self) -> "RouteEdit":
        if self.operation == "add" and self.step is None:
            raise ValueError("add edits require step")
        if self.operation in {"update", "delete", "check"} and not self.step_id:
            raise ValueError(f"{self.operation} edits require step_id")
        if self.operation == "update" and self.step is None:
            raise ValueError("update edits require step")
        return self


class ReviewDecision(BaseModel):
    decision: Literal["approve", "reject", "followup"]
    reasoning: str = Field(default="", max_length=MAX_FEEDBACK_CHARS)
    followup_route: SolutionRoute | None = None
    main_route: str | None = None
    edits: list[RouteEdit] = Field(default_factory=list, max_length=MAX_ROUTE_STEPS)
    more_edits: bool = False

    @model_validator(mode="after")
    def followup_has_route(self) -> "ReviewDecision":
        if (
            self.decision == "followup"
            and self.followup_route is None
            and not self.edits
            and self.main_route is None
        ):
            raise ValueError("followup decisions require a route, main_route, or edits")
        if self.more_edits and self.decision == "reject":
            raise ValueError("rejected decisions cannot request more edits")
        return self


class AuditRecord(BaseModel):
    sequence: int = Field(ge=1)
    event: str
    proposal_id: str | None = None
    plan_revision: int | None = None
    detail: str = ""
    actor: str = "solution_path_engine"
    source_task_id: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class DurableSolutionPathState(BaseModel):
    schema_version: int = 2
    run_id: str
    workflow_mode: str = ""
    user_prompt: str = ""
    prompt_hash: str = ""
    lifecycle_generation: int = Field(default=1, ge=1)
    acceptance_count: int = Field(default=0, ge=0)
    plan: SolutionPlan | None = None
    proposals: list[PlanProposal] = Field(default_factory=list)
    audit: list[AuditRecord] = Field(default_factory=list)
    next_audit_sequence: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def normalize_prompt_provenance(self) -> "DurableSolutionPathState":
        if self.user_prompt and not self.prompt_hash:
            self.prompt_hash = prompt_fingerprint(self.user_prompt)
        return self
