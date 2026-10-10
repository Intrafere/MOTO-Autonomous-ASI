"""Test-only inventory of production provider/API-error Live Activity sources.

Every call that publishes a provider/API-error activity event, every call into
the shared retry-activity seam, and every explicitly known dynamic forwarder of
those events must match one explicit policy below. A policy labels the site
``attributed`` (the line names mode, role, model, provider and host) or
``exempt`` with a reason (for example a non-model workflow-loop retry). The
``attributed`` label is verified syntactically, so a site cannot claim route
identity it does not build. Payload contents are proven by focused tests, not
by this ledger. The overlay stays removable: production never imports it.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator


# Test-only copy of the closed production event set. A drift test pins it to
# backend.shared.provider_route_identity.PROVIDER_ERROR_ACTIVITY_EVENTS.
PROVIDER_ERROR_ACTIVITY_EVENTS = frozenset({
    "provider_retry_cooldown",
    "openai_codex_oauth_error",
    "sakana_fugu_error",
    "oauth_provider_error",
    "oauth_provider_usage_limited",
    "provider_usage_limit_resumed",
    "hung_connection_alert",
    "openrouter_rate_limit",
    "openrouter_privacy_error",
    "account_credits_exhausted",
    "boost_credits_exhausted",
    "openrouter_fallback",
    "openrouter_fallback_failed",
    "free_model_rotated",
    "free_model_auto_selector_used",
    "free_models_exhausted",
    "provider_repair_required",
    "provider_role_disabled",
    "autonomous_proof_provider_paused",
    "autonomous_proof_provider_resumed",
    "proof_run_provider_paused",
    "proof_run_provider_resumed",
    "proof_retry_scheduled",
    "proof_retry_started",
    "proof_prune_provider_paused",
    "proof_prune_repair_required",
    "assistant_proof_pack_failed",
    "solution_path_proposal_user_repair_required",
    "leanoj_model_call_failed",
    "leanoj_provider_paused",
    "leanoj_provider_resumed",
    "proof_competition_unavailable",
    "proof_competitor_unavailable",
    "auto_research_stopped",
})

# Calls that publish provider_retry_cooldown activity through the shared seam.
RETRY_SEAM_CALLS = frozenset({
    "emit_and_wait_for_retry",
    "assured_retry_wait",
    "_broadcast_provider_retry_cooldown",
})
RETRY_SEAM_EVENT = "provider_retry_cooldown"
# Shared identity composers from backend.shared.provider_route_identity plus the
# APIClientManager role-identity entry point.
IDENTITY_HELPERS = frozenset({
    "route_identity_fields",
    "route_identity_from_error",
    "route_identity_for_role",
    "merge_route_identity",
    "with_route_identity",
})
# The central APIClientManager seam composes identity for every retry it publishes.
IDENTITY_SEAM_PATH = "backend/shared/api_client_manager.py"
IDENTITY_SEAM_SCOPES = frozenset({
    "APIClientManager._broadcast_provider_retry_cooldown",
    "APIClientManager.emit_and_wait_for_retry",
    "APIClientManager.wait_for_retryable_provider_error",
    "APIClientManager.wait_for_provider_cooldown",
})
DYNAMIC_EVENT = "<dynamic>"
EVENT_PARAMETER_NAMES = frozenset({"event", "event_type", "event_name"})
PLACEHOLDER_PROVIDER_VALUES = frozenset({"", "unknown", "configured_role", "provider", "chromadb"})


class ActivityClassification(str, Enum):
    ATTRIBUTED = "attributed"
    EXEMPT = "exempt"
    UNCLASSIFIED = "unclassified"


class SiteKind(str, Enum):
    EVENT_LITERAL = "event_literal"
    RETRY_SEAM = "retry_seam"
    DYNAMIC_FORWARDER = "dynamic_forwarder"


@dataclass(frozen=True)
class ActivitySelector:
    path: str
    scope: str
    call: str
    event: str
    occurrence: int


@dataclass(frozen=True)
class ActivityPolicy:
    selector: ActivitySelector
    classification: ActivityClassification
    rationale: str


@dataclass(frozen=True)
class ActivitySite:
    selector: ActivitySelector
    line: int
    kind: SiteKind
    classification: ActivityClassification
    rationale: str
    call_evidence: tuple[str, ...]
    scope_evidence: tuple[str, ...]

    @property
    def evidence(self) -> tuple[str, ...]:
        return self.call_evidence + self.scope_evidence

    @property
    def label_mismatch(self) -> str | None:
        if self.classification is ActivityClassification.ATTRIBUTED and not self.evidence:
            return "labelled attributed but builds no route identity"
        if self.classification is ActivityClassification.EXEMPT and self.call_evidence:
            return "labelled exempt but the call itself carries route identity"
        return None


@dataclass(frozen=True)
class ProviderErrorActivityInventory:
    sites: tuple[ActivitySite, ...]
    stale_policies: tuple[ActivityPolicy, ...]

    @property
    def unclassified(self) -> tuple[ActivitySite, ...]:
        return tuple(site for site in self.sites if site.classification is ActivityClassification.UNCLASSIFIED)

    @property
    def attributed(self) -> tuple[ActivitySite, ...]:
        return tuple(site for site in self.sites if site.classification is ActivityClassification.ATTRIBUTED)

    @property
    def exempt(self) -> tuple[ActivitySite, ...]:
        return tuple(site for site in self.sites if site.classification is ActivityClassification.EXEMPT)

    @property
    def mismatched(self) -> tuple[ActivitySite, ...]:
        return tuple(site for site in self.sites if site.label_mismatch)

    @property
    def complete(self) -> bool:
        return not self.unclassified and not self.stale_policies

    @property
    def route_attributed(self) -> bool:
        return self.complete and not self.mismatched


def _production_source_paths(repo_root: Path) -> tuple[str, ...]:
    roots = (
        repo_root / "backend" / "aggregator",
        repo_root / "backend" / "api",
        repo_root / "backend" / "autonomous",
        repo_root / "backend" / "compiler",
        repo_root / "backend" / "leanoj",
        repo_root / "backend" / "shared",
    )
    return tuple(
        path.relative_to(repo_root).as_posix()
        for root in roots
        for path in sorted(root.rglob("*.py"))
    )


def _policy(
    path: str,
    scope: str,
    call: str,
    event: str,
    classification: ActivityClassification,
    rationale: str,
    *,
    occurrence: int = 1,
) -> ActivityPolicy:
    return ActivityPolicy(
        selector=ActivitySelector(path, scope, call, event, occurrence),
        classification=classification,
        rationale=rationale,
    )


def _policies(
    path: str,
    scope: str,
    call: str,
    event: str,
    count: int,
    classification: ActivityClassification,
    rationale: str,
) -> tuple[ActivityPolicy, ...]:
    return tuple(
        _policy(path, scope, call, event, classification, rationale, occurrence=occurrence)
        for occurrence in range(1, count + 1)
    )


ATTRIBUTED = ActivityClassification.ATTRIBUTED
EXEMPT = ActivityClassification.EXEMPT

_SUBMITTER = "backend/aggregator/agents/submitter.py"
_VALIDATOR = "backend/aggregator/agents/validator.py"
_AGGREGATOR = "backend/aggregator/core/coordinator.py"
_RAG = "backend/aggregator/core/rag_manager.py"
_PROOFS_ROUTE = "backend/api/routes/proofs.py"
_TITLE_SELECTOR = "backend/autonomous/agents/paper_title_selector.py"
_AUTONOMOUS = "backend/autonomous/core/autonomous_coordinator.py"
_PRUNING = "backend/autonomous/core/proof_pruning_coordinator.py"
_PROOF_RUNS = "backend/autonomous/core/proof_run_manager.py"
_PROOF_STAGE = "backend/autonomous/core/proof_verification_stage.py"
_RIGOR = "backend/compiler/agents/high_param_submitter.py"
_COMPILER = "backend/compiler/core/compiler_coordinator.py"
_LEANOJ = "backend/leanoj/core/leanoj_coordinator.py"
_MANAGER = "backend/shared/api_client_manager.py"
_ASSISTANT = "backend/shared/proof_search/assistant_coordinator.py"
_RECOVERY_RUNTIME = "backend/shared/provider_recovery/runtime.py"
_SOLUTION_PATH = "backend/shared/solution_path/engine.py"

_EMIT = "api_client_manager.emit_and_wait_for_retry"
_RETRY = RETRY_SEAM_EVENT

_ERROR_SUPPLIES_ROUTE = "Workflow retry passes the caught error, whose typed route supplies model, provider and host when present."
_FREE_ROTATION_ROUTE = "Free-model exhaustion names the route the typed error carries; otherwise the role's primary model and pinned host when no rotation leg can have run, else the free-rotation route with auto host."
_FREE_EXHAUSTED_ERROR_ROUTE = "Free-model exhaustion names mode and role; model and host come from the typed error when it carries them."
_ROLE_ROUTE = "Payload merges route_identity_for_role for the failing role and its typed error."
_ADAPTER_ROUTE = "Provider adapter retry payload names its model and provider; the manager seam adds role and host."
_NON_MODEL_LOOP = "Non-model workflow-loop retry with no typed provider error in scope; names mode and role only."
_PASS_THROUGH = "Pass-through forwarder; route identity is verified at each inventoried producer it forwards."

ACTIVITY_POLICIES: tuple[ActivityPolicy, ...] = (
    _policy(_SUBMITTER, "SubmitterAgent._run_loop", _EMIT, _RETRY, ATTRIBUTED, _FREE_ROTATION_ROUTE),
    _policy(_SUBMITTER, "SubmitterAgent._run_loop", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE, occurrence=2),
    _policy(_SUBMITTER, "SubmitterAgent._generate_submission", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE),
    _policy(_VALIDATOR, "ValidatorAgent._assess_quality", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE),
    _policy(_AGGREGATOR, "Coordinator._validator_loop", "_broadcast", "free_models_exhausted", ATTRIBUTED, _FREE_ROTATION_ROUTE),
    _policy(_AGGREGATOR, "Coordinator._validator_loop", _EMIT, _RETRY, ATTRIBUTED, _FREE_ROTATION_ROUTE),
    _policy(_AGGREGATOR, "Coordinator._validator_loop", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE, occurrence=2),
    _policy(_AGGREGATOR, "Coordinator._single_model_workflow", "_broadcast", "free_models_exhausted", ATTRIBUTED, _FREE_ROTATION_ROUTE),
    _policy(_AGGREGATOR, "Coordinator._single_model_workflow", _EMIT, _RETRY, ATTRIBUTED, _FREE_ROTATION_ROUTE),
    _policy(_AGGREGATOR, "Coordinator._single_model_workflow", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE, occurrence=2),
    *(
        _policy(_AGGREGATOR, scope, call, event, ATTRIBUTED, "Repair payload names the error's model and provider and merges the role's route identity.")
        for scope, event in (
            ("Coordinator._handle_submitter_provider_repair_required", "provider_role_disabled"),
            ("Coordinator._handle_provider_repair_required", "provider_repair_required"),
        )
        for call in ("_broadcast", "_add_persisted_event")
    ),
    _policy(_RAG, "RAGManager._vector_search", _EMIT, _RETRY, EXEMPT, "ChromaDB index-race storage retry; no model route exists."),
    _policy(_PROOFS_ROUTE, "_run_manual_proof_check.execute_round", "broadcast_fn", "proof_run_provider_resumed", ATTRIBUTED, "Resume line reuses the identity-bearing retry payload and prefixes it with with_route_identity."),
    _policy(_PROOFS_ROUTE, "_run_manual_proof_check.execute_round.broadcast_retry_activity", "broadcast_fn", "proof_retry_scheduled|proof_retry_started", ATTRIBUTED, "Forwards the manager seam's identity-bearing retry payload; the started line is prefixed with with_route_identity."),
    _policy(_PROOFS_ROUTE, "_run_manual_proof_check.execute_round", _EMIT, _RETRY, ATTRIBUTED, "Transient manual proof retry merges _manual_proof_route_identity and passes the error."),
    _policy(_PROOFS_ROUTE, "_broadcast_manual_aggregator_proof_event", "websocket.broadcast_event", DYNAMIC_EVENT, EXEMPT, _PASS_THROUGH),
    _policy(_PROOFS_ROUTE, "_broadcast_manual_aggregator_proof_event", "event_log.add_event", DYNAMIC_EVENT, ATTRIBUTED, "Persisted manual Aggregator proof lines are composed by _manual_aggregator_proof_event_message, which prefixes provider events with route identity."),
    _policy(_TITLE_SELECTOR, "PaperTitleSelectorAgent.select_title", _EMIT, _RETRY, ATTRIBUTED, "Failed title generation passes the last generation error, whose typed route supplies identity when present."),
    _policy(_TITLE_SELECTOR, "PaperTitleSelectorAgent.select_title", _EMIT, _RETRY, EXEMPT, "Semantic title rejection retry; the validator decided, no provider error occurred.", occurrence=2),
    *(
        _policy(_AUTONOMOUS, "AutonomousCoordinator._run_proof_verification.run_single_proof_round", "_broadcast", event, ATTRIBUTED, "Credit pause/resume payload merges route_identity_for_role for the paused proof role and error.", occurrence=occurrence)
        for event in ("autonomous_proof_provider_paused", "autonomous_proof_provider_resumed")
        for occurrence in (1, 2, 3)
    ),
    _policy(_AUTONOMOUS, "AutonomousCoordinator._broadcast_stopped_once", "asyncio.to_thread", "auto_research_stopped", ATTRIBUTED, "Model-error stop record names the terminal event's effective model and provider plus its typed route identity."),
    _policy(_AUTONOMOUS, "AutonomousCoordinator._broadcast_stopped_once", "_broadcast", "auto_research_stopped", ATTRIBUTED, "The model-error stop broadcast merges the attributed stored record; lifecycle-only stops are not provider-error activity."),
    *_policies(_AUTONOMOUS, "AutonomousCoordinator.start", _EMIT, _RETRY, 8, EXEMPT, "Paper compilation retry after a failed child workflow; child provider errors publish their own attributed activity."),
    _policy(_AUTONOMOUS, "AutonomousCoordinator.start", "_broadcast", "free_models_exhausted", ATTRIBUTED, _FREE_EXHAUSTED_ERROR_ROUTE),
    _policy(_AUTONOMOUS, "AutonomousCoordinator.start", _EMIT, _RETRY, ATTRIBUTED, _FREE_EXHAUSTED_ERROR_ROUTE, occurrence=9),
    *_policies(_AUTONOMOUS, "AutonomousCoordinator._resume_research_loop_after_tier3", _EMIT, _RETRY, 2, EXEMPT, "Paper compilation retry after a failed child workflow; child provider errors publish their own attributed activity."),
    _policy(_AUTONOMOUS, "AutonomousCoordinator._resume_research_loop_after_tier3", "_broadcast", "free_models_exhausted", ATTRIBUTED, _FREE_EXHAUSTED_ERROR_ROUTE),
    _policy(_AUTONOMOUS, "AutonomousCoordinator._resume_research_loop_after_tier3", _EMIT, _RETRY, ATTRIBUTED, _FREE_EXHAUSTED_ERROR_ROUTE, occurrence=3),
    *(
        _policy(_AUTONOMOUS, scope, "_mark_terminal_stop", "provider_repair_required", EXEMPT, "Terminal-reason literal, not a publication; the stop forwards the child Aggregator fatal payload attributed at Coordinator._handle_provider_repair_required.")
        for scope in (
            "AutonomousCoordinator._topic_exploration_phase",
            "AutonomousCoordinator._brainstorm_aggregation_loop",
            "AutonomousCoordinator._paper_title_exploration_phase",
        )
    ),
    _policy(_AUTONOMOUS, "AutonomousCoordinator._topic_selection_loop", _EMIT, _RETRY, EXEMPT, _NON_MODEL_LOOP),
    _policy(_AUTONOMOUS, "AutonomousCoordinator._brainstorm_continuation_decision", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE),
    _policy(_AUTONOMOUS, "AutonomousCoordinator._run_completed_paper_proof_checks", "_broadcast", "proof_retry_scheduled", EXEMPT, "Deferred Lean proof-candidate retry for a completed paper; no provider/API error is involved."),
    _policy(_PRUNING, "ProofPruningCoordinator._broadcast", "broadcast_fn", DYNAMIC_EVENT, EXEMPT, _PASS_THROUGH),
    _policy(_PRUNING, "ProofPruningCoordinator._handle_failure", "_broadcast", "proof_prune_provider_paused", ATTRIBUTED, "Pause payload spreads _failure_route_identity for the failing pruning role."),
    _policy(_PRUNING, "ProofPruningCoordinator._handle_failure", _EMIT, _RETRY, ATTRIBUTED, "Pruning retry merges _failure_route_identity and passes the error."),
    _policy(_PRUNING, "ProofPruningCoordinator._handle_failure", "_broadcast", "proof_prune_repair_required", ATTRIBUTED, "Repair payload spreads _failure_route_identity for the failing pruning role."),
    _policy(_PROOF_RUNS, "ProofRunManager._emit", "control.event_callback", DYNAMIC_EVENT, EXEMPT, _PASS_THROUGH),
    _policy(_PROOF_RUNS, "ProofRunManager.provider_paused", "_emit", "proof_run_provider_paused", ATTRIBUTED, "Pause event carries the caller's route identity and a with_route_identity message."),
    _policy(_PROOF_RUNS, "ProofRunManager.resumed", "_emit", "proof_run_provider_resumed", ATTRIBUTED, "Resume after a manual proof credit pause must name the paused route like provider_paused."),
    _policy(_PROOF_STAGE, "ProofVerificationStage._broadcast", "broadcast_fn", DYNAMIC_EVENT, EXEMPT, _PASS_THROUGH),
    _policy(_PROOF_STAGE, "ProofVerificationStage.run.observe_competition", "_broadcast", "proof_competitor_unavailable", ATTRIBUTED, "Competitor payload merges competitor_route_identity for the secondary route."),
    _policy(_PROOF_STAGE, "ProofVerificationStage.run", "_broadcast", "proof_retry_started", EXEMPT, "Deferred Lean proof-candidate retry round started; no provider/API error is involved."),
    _policy(_PROOF_STAGE, "ProofVerificationStage.run", "broadcast_candidate_list_interrupted", "provider_repair_required", ATTRIBUTED, "Error-kind literal; the candidate-list interruption merges the passed error's route identity."),
    *_policies(_RIGOR, "HighParamSubmitter._step_formalize", "_broadcast", "proof_competition_unavailable", 2, ATTRIBUTED, "Competition payload merges route_identity_for_role for the unavailable route."),
    _policy(_COMPILER, "CompilerCoordinator._main_workflow", "_broadcast", "free_models_exhausted", ATTRIBUTED, _FREE_EXHAUSTED_ERROR_ROUTE),
    _policy(_COMPILER, "CompilerCoordinator._main_workflow", _EMIT, _RETRY, ATTRIBUTED, _FREE_EXHAUSTED_ERROR_ROUTE),
    _policy(_COMPILER, "CompilerCoordinator._main_workflow", "normalize_activity_payload", "provider_repair_required", ATTRIBUTED, _ROLE_ROUTE),
    _policy(_COMPILER, "CompilerCoordinator._main_workflow", "_broadcast", "provider_repair_required", ATTRIBUTED, _ROLE_ROUTE),
    _policy(_COMPILER, "CompilerCoordinator._outline_creation_loop", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE),
    _policy(_COMPILER, "CompilerCoordinator._outline_creation_loop", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE, occurrence=2),
    _policy(_COMPILER, "CompilerCoordinator._outline_creation_loop", _EMIT, _RETRY, EXEMPT, "Empty outline submission retry; " + _NON_MODEL_LOOP, occurrence=3),
    _policy(_COMPILER, "CompilerCoordinator._initial_paper_loop", _EMIT, _RETRY, EXEMPT, "Empty construction submission retry; " + _NON_MODEL_LOOP),
    *(
        _policy(_COMPILER, "CompilerCoordinator._initial_paper_loop", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE, occurrence=occurrence)
        for occurrence in (2, 3)
    ),
    _policy(_COMPILER, "CompilerCoordinator._run_critique_aggregation", _EMIT, _RETRY, EXEMPT, "Empty critique submission retry; " + _NON_MODEL_LOOP),
    _policy(_COMPILER, "CompilerCoordinator._run_critique_aggregation", _EMIT, _RETRY, ATTRIBUTED, _ERROR_SUPPLIES_ROUTE, occurrence=2),
    *(
        _policy(_LEANOJ, scope, "_persist_and_broadcast", event, ATTRIBUTED, "Pause/resume payload is built by _route_identity_payload for the paused role and error.")
        for scope in ("LeanOJCoordinator.start", "LeanOJCoordinator._pause_for_provider_credits")
        for event in ("leanoj_provider_paused", "leanoj_provider_resumed")
    ),
    *(
        _policy(_LEANOJ, scope, _EMIT, _RETRY, ATTRIBUTED, "LeanOJ submitter retry merges _route_identity_payload and passes the error.")
        for scope in ("LeanOJCoordinator._topic_submitter_loop", "LeanOJCoordinator._brainstorm_submitter_loop")
    ),
    *_policies(_LEANOJ, "LeanOJCoordinator._call_json", "_broadcast", "leanoj_model_call_failed", 4, ATTRIBUTED, "Model-call failure payload is built by _route_identity_payload for the calling role."),
    _policy(_LEANOJ, "LeanOJCoordinator._call_json", _EMIT, _RETRY, ATTRIBUTED, "Invalid-JSON retry merges route_identity_for_role, which names an active LM Studio fallback, and passes the error."),
    *(
        _policy(_MANAGER, "APIClientManager._broadcast_provider_retry_cooldown", call, _RETRY, ATTRIBUTED, "Central identity seam composes route identity before publishing.")
        for call in ("canonical_recovery_event", "asyncio.to_thread", "_broadcast")
    ),
    _policy(_MANAGER, "APIClientManager._provider_retry_callback.callback", "emit_and_wait_for_retry", _RETRY, ATTRIBUTED, "Merges the call's route kind, task and configured route before the identity seam."),
    _policy(_MANAGER, "APIClientManager.emit_and_wait_for_retry", "_broadcast_provider_retry_cooldown", _RETRY, ATTRIBUTED, "Central identity seam merges the typed error's route identity."),
    _policy(_MANAGER, "APIClientManager.wait_for_retryable_provider_error", "emit_and_wait_for_retry", _RETRY, ATTRIBUTED, "Central identity seam resolves the error's route before the activity callback."),
    _policy(_MANAGER, "APIClientManager.wait_for_provider_cooldown", "emit_and_wait_for_retry", _RETRY, ATTRIBUTED, "Central identity seam resolves the cooldown route before publishing."),
    *(
        _policy(_MANAGER, scope, call, event, ATTRIBUTED, "Unrecoverable provider error payload merges route_identity_for_role for the role, model and error.")
        for scope, event in (
            ("APIClientManager._broadcast_unrecoverable_codex_error", "openai_codex_oauth_error"),
            ("APIClientManager._broadcast_unrecoverable_sakana_fugu_error", "sakana_fugu_error"),
            ("APIClientManager._broadcast_unrecoverable_xai_grok_error", "oauth_provider_error"),
        )
        for call in ("asyncio.to_thread", "_broadcast")
    ),
    *(
        _policy(_MANAGER, "APIClientManager._confirm_provider_usage_limit_resumed", call, "provider_usage_limit_resumed", ATTRIBUTED, "Resume payload merges route_identity_for_role for the recovered role.")
        for call in ("asyncio.to_thread", "_broadcast")
    ),
    *(
        _policy(_MANAGER, "APIClientManager._broadcast_provider_usage_limit", call, "oauth_provider_usage_limited", ATTRIBUTED, "Every caller passes a cooldown payload built by _mark_provider_cooldown or _cooldown_payload_for_role.")
        for call in ("asyncio.to_thread", "_broadcast")
    ),
    _policy(_MANAGER, "APIClientManager._with_hung_connection_watchdog._watchdog", "_broadcast", "hung_connection_alert", ATTRIBUTED, "Hung-call alert merges route_identity_for_role for the watched call."),
    *(
        _policy(_MANAGER, "APIClientManager._generate_completion_once_routed", "_broadcast", event, ATTRIBUTED, "Routing event names its model and provider and merges route_identity_for_role.", occurrence=occurrence)
        for event, occurrence in (
            ("openrouter_rate_limit", 1),
            ("openrouter_privacy_error", 1),
            ("boost_credits_exhausted", 1),
            ("account_credits_exhausted", 1),
            ("openrouter_rate_limit", 2),
            ("openrouter_privacy_error", 2),
            ("openrouter_fallback_failed", 1),
            ("openrouter_fallback", 1),
        )
    ),
    *(
        _policy(_MANAGER, "APIClientManager._try_free_model_rotation", "_broadcast", event, ATTRIBUTED, "Rotation event names the free model and merges route_identity_for_role with its rotation route kind.")
        for event in ("free_model_rotated", "free_model_auto_selector_used")
    ),
    *_policies("backend/shared/lm_studio_client.py", "LMStudioClient._execute_completion_request", "assured_retry_wait", _RETRY, 2, ATTRIBUTED, _ADAPTER_ROUTE),
    _policy("backend/shared/lm_studio_client.py", "LMStudioClient._get_embeddings_with_retry", "assured_retry_wait", _RETRY, ATTRIBUTED, _ADAPTER_ROUTE),
    *_policies("backend/shared/openai_codex_client.py", "OpenAICodexClient._post_with_retry", "assured_retry_wait", _RETRY, 2, ATTRIBUTED, _ADAPTER_ROUTE),
    *_policies("backend/shared/openai_codex_client.py", "OpenAICodexClient._stream_completion_with_retry", "assured_retry_wait", _RETRY, 2, ATTRIBUTED, _ADAPTER_ROUTE),
    _policy("backend/shared/openai_codex_client.py", "OpenAICodexClient.generate_completion", "assured_retry_wait", _RETRY, ATTRIBUTED, _ADAPTER_ROUTE),
    _policy("backend/shared/openrouter_client.py", "OpenRouterClient._execute_completion_request.wait_before_retry", "assured_retry_wait", _RETRY, ATTRIBUTED, "OpenRouter retry payload names model, provider, configured and upstream host and request id."),
    _policy(_ASSISTANT, "AssistantProofSearchCoordinator._select_and_publish_assistant_pack", "_broadcast_event", "assistant_proof_pack_failed", ATTRIBUTED, "Assistant failure payload merges _assistant_failure_route_identity for the Assistant role."),
    _policy(_RECOVERY_RUNTIME, "assured_retry_wait", _EMIT, _RETRY, EXEMPT, _PASS_THROUGH),
    *_policies("backend/shared/sakana_fugu_client.py", "SakanaFuguClient._request_with_retry", "assured_retry_wait", _RETRY, 2, ATTRIBUTED, _ADAPTER_ROUTE),
    *_policies(_SOLUTION_PATH, "SolutionPathEngine._review_loop", "_broadcast", "solution_path_proposal_user_repair_required", 3, ATTRIBUTED, "Reviewer repair payload merges _reviewer_route_identity for the Main Submitter 1 review route."),
    *_policies("backend/shared/xai_grok_client.py", "XAIGrokClient._post_with_retry", "assured_retry_wait", _RETRY, 2, ATTRIBUTED, _ADAPTER_ROUTE),
)


def _dotted_name(node: ast.expr) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return "<expr>"
    name = ".".join(reversed(parts))
    return name[len("self."):] if name.startswith("self.") else name


def _final_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _literal_events(node: ast.expr) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,) if node.value in PROVIDER_ERROR_ACTIVITY_EVENTS else ()
    if isinstance(node, ast.IfExp):
        return _literal_events(node.body) + _literal_events(node.orelse)
    return ()


def _call_arguments(node: ast.Call) -> Iterator[ast.expr]:
    yield from node.args
    for keyword in node.keywords:
        yield keyword.value


def _call_events(node: ast.Call) -> tuple[str, ...]:
    events: list[str] = []
    for argument in _call_arguments(node):
        events.extend(_literal_events(argument))
    return tuple(sorted(set(events)))


def _function_nodes(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _own_nodes(function: ast.AST) -> Iterator[ast.AST]:
    """Yield nodes of one function body without descending into nested functions."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _is_site_call(node: ast.Call) -> bool:
    return _final_name(node.func) in RETRY_SEAM_CALLS or bool(_call_events(node))


def _calls_any(node: ast.AST, names: frozenset[str] | set[str]) -> bool:
    return any(
        isinstance(child, ast.Call) and _final_name(child.func) in names
        for child in ast.walk(node)
    )


def _returns_identity(function: ast.AST, composers: set[str]) -> bool:
    """Return whether a function returns a value built by an identity composer."""
    bound: set[str] = set()
    for node in _own_nodes(function):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            targets, value = [node.target], node.value
        if value is not None and _calls_any(value, composers):
            bound.update(target.id for target in targets if isinstance(target, ast.Name))
    for node in _own_nodes(function):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if _calls_any(node.value, composers):
            return True
        if any(isinstance(child, ast.Name) and child.id in bound for child in ast.walk(node.value)):
            return True
    return False


def _local_identity_helpers(tree: ast.AST) -> frozenset[str]:
    """Return module functions that return composed identity without publishing activity.

    A module-local wrapper such as ``_with_provider_route_identity`` counts as an
    identity helper when it returns a value built by a shared identity helper
    (directly or through another such wrapper) and it publishes no inventoried
    activity itself. Workflow functions that merely call identity code somewhere
    in their body do not lend that evidence to their callers.
    """
    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    publishers: set[str] = set()
    for function in _function_nodes(tree):
        functions.setdefault(function.name, []).append(function)
        if any(isinstance(node, ast.Call) and _is_site_call(node) for node in _own_nodes(function)):
            publishers.add(function.name)
    helpers: set[str] = set()
    changed = True
    while changed:
        changed = False
        composers = set(IDENTITY_HELPERS) | helpers
        for name, definitions in functions.items():
            if name in helpers or name in publishers or name in RETRY_SEAM_CALLS or name in IDENTITY_HELPERS:
                continue
            if any(_returns_identity(definition, composers) for definition in definitions):
                helpers.add(name)
                changed = True
    return frozenset(helpers)


def _dict_keys(node: ast.Dict) -> dict[str, ast.expr]:
    return {
        key.value: value
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _names_concrete_route(node: ast.Dict) -> bool:
    """Return whether a dict literal names a model and a non-placeholder provider."""
    keys = _dict_keys(node)
    model = keys.get("model")
    provider = keys.get("provider")
    if model is None or provider is None:
        return False
    if isinstance(model, ast.Constant) and not str(model.value or "").strip():
        return False
    if isinstance(provider, ast.Constant) and str(provider.value or "").strip().lower() in PLACEHOLDER_PROVIDER_VALUES:
        return False
    return True


def _parameter_names(function: ast.AST) -> list[str]:
    arguments = function.args
    return [parameter.arg for parameter in arguments.posonlyargs + arguments.args + arguments.kwonlyargs]


def _caller_argument(call: ast.Call, function: ast.AST, name: str) -> ast.expr | None:
    """Return the expression one call passes for a named parameter, when explicit."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    positional = [parameter.arg for parameter in function.args.posonlyargs + function.args.args]
    if name not in positional:
        return None
    index = positional.index(name)
    if positional and positional[0] in {"self", "cls"} and isinstance(call.func, ast.Attribute):
        index -= 1
    if 0 <= index < len(call.args) and not isinstance(call.args[index], ast.Starred):
        return call.args[index]
    return None


class _ModuleIndex:
    """Per-module lookups for syntactic route-identity evidence."""

    max_depth = 4

    def __init__(self, tree: ast.AST) -> None:
        self.local_helpers = _local_identity_helpers(tree)
        self.composers = frozenset(IDENTITY_HELPERS | self.local_helpers)
        self.callers: dict[str, list[tuple[ast.Call, ast.AST]]] = {}
        self._bindings: dict[int, dict[str, list[ast.expr]]] = {}
        for function in _function_nodes(tree):
            for node in _own_nodes(function):
                if isinstance(node, ast.Call):
                    self.callers.setdefault(_final_name(node.func), []).append((node, function))

    def bindings(self, function: ast.AST) -> dict[str, list[ast.expr]]:
        """Values assigned to, or merged via ``name.update(...)`` into, each local name."""
        cached = self._bindings.get(id(function))
        if cached is not None:
            return cached
        bindings: dict[str, list[ast.expr]] = {}
        for node in _own_nodes(function):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bindings.setdefault(target.id, []).append(node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
                bindings.setdefault(node.target.id, []).append(node.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
            ):
                bindings.setdefault(node.func.value.id, []).extend(node.args)
        self._bindings[id(function)] = bindings
        return bindings

    def names_route(self, expr: ast.expr, function: ast.AST | None, depth: int = 0) -> bool:
        """Whether an argument is, or is built from, a dict naming a concrete route."""
        if depth > self.max_depth:
            return False
        if any(isinstance(node, ast.Dict) and _names_concrete_route(node) for node in ast.walk(expr)):
            return True
        if isinstance(expr, ast.Name) and function is not None:
            return any(
                self.names_route(value, function, depth + 1)
                for value in self.bindings(function).get(expr.id, ())
            )
        return False

    def carries_identity(self, expr: ast.expr, function: ast.AST, depth: int = 0) -> bool:
        """Whether an expression is built by an identity composer, here or at every caller."""
        if depth > self.max_depth:
            return False
        if _calls_any(expr, self.composers):
            return True
        if not isinstance(expr, ast.Name):
            return False
        if any(
            self.carries_identity(value, function, depth + 1)
            for value in self.bindings(function).get(expr.id, ())
        ):
            return True
        if expr.id in _parameter_names(function):
            return self.callers_supply_identity(function, expr.id, depth + 1)
        return False

    def callers_supply_identity(self, function: ast.AST, parameter: str, depth: int = 0) -> bool:
        """Whether every in-module caller passes an identity-bearing value for a parameter."""
        callers = [
            (call, caller)
            for call, caller in self.callers.get(function.name, ())
            if caller is not function
        ]
        if not callers or depth > self.max_depth:
            return False
        for call, caller in callers:
            argument = _caller_argument(call, function, parameter)
            if argument is None or not self.carries_identity(argument, caller, depth + 1):
                return False
        return True

    def forwarded_parameter(self, expr: ast.expr, function: ast.AST, depth: int = 0) -> str | None:
        """Return the parameter a payload argument forwards, directly or via local copies."""
        if not isinstance(expr, ast.Name) or depth > self.max_depth:
            return None
        if expr.id in _parameter_names(function):
            return expr.id
        for value in self.bindings(function).get(expr.id, ()):
            for node in ast.walk(value):
                if isinstance(node, ast.Name) and node.id != expr.id:
                    parameter = self.forwarded_parameter(node, function, depth + 1)
                    if parameter:
                        return parameter
        return None

    def call_evidence(self, node: ast.Call, function: ast.AST | None) -> tuple[str, ...]:
        evidence: list[str] = []
        if any(keyword.arg == "error" for keyword in node.keywords):
            evidence.append("error_keyword")
        arguments = tuple(_call_arguments(node))
        if any(self.names_route(argument, function) for argument in arguments):
            evidence.append("payload_model_provider")
        if function is not None:
            for argument in arguments:
                parameter = self.forwarded_parameter(argument, function)
                if parameter and self.callers_supply_identity(function, parameter):
                    evidence.append(f"caller_identity:{function.name}")
                    break
        return tuple(evidence)

    def scope_evidence(self, path: str, scope: str, function: ast.AST | None) -> tuple[str, ...]:
        evidence: list[str] = []
        if path == IDENTITY_SEAM_PATH and any(
            scope == seam or scope.startswith(f"{seam}.") for seam in IDENTITY_SEAM_SCOPES
        ):
            evidence.append("identity_seam")
        if function is not None:
            called = {
                _final_name(node.func)
                for node in _own_nodes(function)
                if isinstance(node, ast.Call)
            }
            evidence.extend(f"identity_helper:{name}" for name in sorted(called & IDENTITY_HELPERS))
            evidence.extend(
                f"local_identity_helper:{name}" for name in sorted(called & self.local_helpers)
            )
        return tuple(evidence)


def _forwards_event_parameter(node: ast.Call, function: ast.AST | None) -> bool:
    if function is None or not node.args or not isinstance(node.args[0], ast.Name):
        return False
    name = node.args[0].id
    return name in EVENT_PARAMETER_NAMES and name in _parameter_names(function)


def _names_provider_error_event(function: ast.AST | None) -> bool:
    """Return whether a function branches on provider-error events outside call arguments."""
    if function is None:
        return False
    call_arguments: set[int] = set()
    for node in _own_nodes(function):
        if isinstance(node, ast.Call):
            for argument in _call_arguments(node):
                for nested in ast.walk(argument):
                    call_arguments.add(id(nested))
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in PROVIDER_ERROR_ACTIVITY_EVENTS
        and id(node) not in call_arguments
        for node in _own_nodes(function)
    )


@dataclass(frozen=True)
class _DiscoveredCall:
    selector: ActivitySelector
    line: int
    kind: SiteKind
    call_evidence: tuple[str, ...]
    scope_evidence: tuple[str, ...]
    forwarder_candidate: bool


class _ScopedCallVisitor(ast.NodeVisitor):
    def __init__(self, path: str, index: _ModuleIndex) -> None:
        self.path = path
        self.index = index
        self.scope: list[str] = []
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.counters: dict[tuple[str, str, str], int] = {}
        self.found: list[_DiscoveredCall] = []

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.functions.append(node)
        self._visit_scope(node, node.name)
        self.functions.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _record(self, node: ast.Call, kind: SiteKind, event: str, *, forwarder_candidate: bool = False) -> None:
        scope = ".".join(self.scope) or "<module>"
        call = _dotted_name(node.func)
        key = (scope, call, event)
        self.counters[key] = self.counters.get(key, 0) + 1
        function = self.functions[-1] if self.functions else None
        self.found.append(
            _DiscoveredCall(
                selector=ActivitySelector(self.path, scope, call, event, self.counters[key]),
                line=node.lineno,
                kind=kind,
                call_evidence=self.index.call_evidence(node, function),
                scope_evidence=self.index.scope_evidence(self.path, scope, function),
                forwarder_candidate=forwarder_candidate,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        function = self.functions[-1] if self.functions else None
        events = _call_events(node)
        if _final_name(node.func) in RETRY_SEAM_CALLS:
            self._record(node, SiteKind.RETRY_SEAM, RETRY_SEAM_EVENT)
        elif events:
            self._record(node, SiteKind.EVENT_LITERAL, "|".join(events))
        elif _forwards_event_parameter(node, function):
            self._record(
                node,
                SiteKind.DYNAMIC_FORWARDER,
                DYNAMIC_EVENT,
                forwarder_candidate=_names_provider_error_event(function),
            )
        self.generic_visit(node)


@lru_cache(maxsize=None)
def _discover(repo_root: Path, relative_path: str) -> tuple[_DiscoveredCall, ...]:
    # Cached per process: policy variants re-classify the same unchanged sources.
    source_path = repo_root / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"), filename=str(source_path))
    visitor = _ScopedCallVisitor(relative_path, _ModuleIndex(tree))
    visitor.visit(tree)
    return tuple(visitor.found)


def build_provider_error_activity_inventory(
    repo_root: Path,
    *,
    policies: Iterable[ActivityPolicy] = ACTIVITY_POLICIES,
) -> ProviderErrorActivityInventory:
    policies = tuple(policies)
    policy_by_selector = {policy.selector: policy for policy in policies}
    sites: list[ActivitySite] = []
    seen: set[ActivitySelector] = set()
    for relative_path in _production_source_paths(repo_root):
        for discovered in _discover(repo_root, relative_path):
            policy = policy_by_selector.get(discovered.selector)
            if discovered.kind is SiteKind.DYNAMIC_FORWARDER and policy is None and not discovered.forwarder_candidate:
                # Generic event plumbing that never names a provider-error event.
                continue
            seen.add(discovered.selector)
            sites.append(
                ActivitySite(
                    selector=discovered.selector,
                    line=discovered.line,
                    kind=discovered.kind,
                    classification=(
                        policy.classification
                        if policy is not None
                        else ActivityClassification.UNCLASSIFIED
                    ),
                    rationale=(
                        policy.rationale
                        if policy is not None
                        else "No explicit attributed/exempt policy covers this provider-error activity source."
                    ),
                    call_evidence=discovered.call_evidence,
                    scope_evidence=discovered.scope_evidence,
                )
            )
    stale = tuple(policy for policy in policies if policy.selector not in seen)
    return ProviderErrorActivityInventory(tuple(sites), stale)


def format_activity_site(site: ActivitySite) -> str:
    selector = site.selector
    detail = site.label_mismatch or ", ".join(site.evidence) or "no identity evidence"
    return (
        f"{selector.path}:{site.line} {selector.scope} "
        f"{selector.call}[{selector.event}]#{selector.occurrence} "
        f"<{site.kind.value}> [{site.classification.value}] {detail}"
    )
