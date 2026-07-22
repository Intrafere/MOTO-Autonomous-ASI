"""Shared durable solution-path planning primitives."""

from .engine import MIN_ACCEPTANCES, ProposalReviewer, SolutionPathEngine
from .models import (
    AuditRecord,
    DurableSolutionPathState,
    PlanProposal,
    ProposalStatus,
    RepairReason,
    ReviewDecision,
    RouteEdit,
    RouteStep,
    SolutionPlan,
    SolutionRoute,
    StepOrdering,
)
from .integration import (
    advisory_plan_block,
    enqueue_optional_update,
    note_acceptances,
    validator_instruction,
    with_budgeted_solver_plan,
    with_solver_plan,
    with_validator_hook,
)
from .registry import (
    SolutionPathManagerRegistry,
    solution_path_registry,
    stable_solution_path_run_id,
)
from .reviewer import build_review_prompt, compact_review_prompt, review_with_json_retry

__all__ = [
    "MIN_ACCEPTANCES",
    "AuditRecord",
    "DurableSolutionPathState",
    "PlanProposal",
    "ProposalReviewer",
    "ProposalStatus",
    "RepairReason",
    "ReviewDecision",
    "RouteEdit",
    "RouteStep",
    "SolutionPathEngine",
    "SolutionPlan",
    "SolutionRoute",
    "StepOrdering",
    "SolutionPathManagerRegistry",
    "advisory_plan_block",
    "enqueue_optional_update",
    "note_acceptances",
    "validator_instruction",
    "solution_path_registry",
    "stable_solution_path_run_id",
    "with_budgeted_solver_plan",
    "with_solver_plan",
    "with_validator_hook",
    "build_review_prompt",
    "compact_review_prompt",
    "review_with_json_retry",
]
