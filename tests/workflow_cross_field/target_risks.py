from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tests.workflow_harness.coverage_metadata import (
    AdapterType,
    InteractionCoverage,
    ResultStatus,
)
from tests.workflow_harness.invariant_catalog import FIRST_BUILD_FIELDS, INVARIANTS_BY_ID


TARGET_RISK_SCHEMA_VERSION = 1
RiskPriority = Literal["critical", "high", "medium"]
SupportStatus = Literal["passed", "failed", "skipped", "blocked", "uncovered"]
PRIORITY_WEIGHT: dict[RiskPriority, int] = {"critical": 3, "high": 2, "medium": 1}


@dataclass(frozen=True)
class TargetRisk:
    risk_id: str
    title: str
    priority: RiskPriority
    required_fields: tuple[str, ...]
    supporting_invariants: tuple[str, ...]
    required_adapters: tuple[AdapterType, ...]
    description: str


@dataclass(frozen=True)
class AdapterSupport:
    adapter: AdapterType
    status: SupportStatus
    scenario_ids: tuple[str, ...]
    result_statuses: tuple[ResultStatus, ...]
    missing_invariants: tuple[str, ...]


@dataclass(frozen=True)
class TargetRiskAnalysis:
    risk: TargetRisk
    rank: int
    gap_score: int
    support: tuple[AdapterSupport, ...]
    existing_scenarios: tuple[str, ...]
    missing_isolated_artifacts: tuple[str, ...]


TARGET_RISKS: tuple[TargetRisk, ...] = (
    TargetRisk(
        "risk.pruned_paper_context_leak",
        "Pruned paper remains in model context",
        "high",
        ("workflow_filesystem_state", "prompt_context"),
        ("state.pruned_papers_excluded_from_context",),
        ("model", "real_coordinator"),
        "A pruned paper remains available to model or RAG context after removal.",
    ),
    TargetRisk(
        "risk.stale_child_phase_takeover",
        "Stale child output overrides parent phase",
        "critical",
        ("runtime_exclusivity", "workflow_filesystem_state", "websocket_api_contracts"),
        ("runtime.parent_action_fences_child_outputs",),
        ("model", "real_coordinator"),
        "Late child output changes state after a parent or user action takes ownership.",
    ),
    TargetRisk(
        "risk.disabled_proof_runtime_executes",
        "Disabled proof runtime executes tools",
        "critical",
        ("proof_runtime_gating", "allowed_outputs"),
        ("proof_runtime.no_lean_when_disabled", "proof_runtime.no_smt_when_disabled"),
        ("model", "real_coordinator"),
        "Lean or SMT runs despite disabled proof-runtime controls.",
    ),
    TargetRisk(
        "risk.proofs_only_enters_paper",
        "Proofs-only run enters paper phases",
        "critical",
        ("allowed_outputs", "proof_runtime_gating", "workflow_filesystem_state"),
        ("outputs.at_least_one_output_enabled", "outputs.proofs_only_no_paper_phase"),
        ("model", "real_coordinator"),
        "Allowed-output routing silently writes papers or strands a proof-only handoff.",
    ),
    TargetRisk(
        "risk.provider_pause_loses_checkpoint",
        "Provider pause loses resumable work",
        "critical",
        ("provider_pause_resume", "workflow_filesystem_state", "proof_runtime_gating"),
        (
            "provider.pause_preserves_checkpoint",
            "provider.stop_resume_preserves_pause",
            "provider.reset_wakes_without_corrupting_checkpoint",
        ),
        ("model", "real_coordinator"),
        "Credit pause, stop, or reset loses or corrupts the durable proof cursor.",
    ),
    TargetRisk(
        "risk.assistant_memory_crosses_boundary",
        "Assistant memory crosses role boundary",
        "high",
        ("assistant_memory", "prompt_context"),
        (
            "assistant.no_validator_injection",
            "assistant.disable_clears_live_pack",
            "prompt.validator_excludes_assistant_memory",
        ),
        ("model", "real_coordinator"),
        "Assistant proof memory reaches validators or remains live after disable.",
    ),
    TargetRisk(
        "risk.proof_scope_leak",
        "Proof records leak across scopes",
        "critical",
        ("proof_scope_isolation", "workflow_filesystem_state", "websocket_api_contracts"),
        (
            "proof_scope.manual_not_in_autonomous_current",
            "proof_scope.autonomous_events_not_manual",
            "proof_scope.manual_clear_archives_active",
        ),
        ("model", "real_coordinator"),
        "Manual, autonomous, archived, or event proof state is exposed in the wrong scope.",
    ),
    TargetRisk(
        "risk.clear_destroys_history",
        "Clear destroys history or leaves active state",
        "high",
        ("workflow_filesystem_state", "proof_scope_isolation", "assistant_memory"),
        ("state.clear_removes_active_preserves_history", "proof_scope.manual_clear_archives_active"),
        ("model", "real_coordinator"),
        "Clear/reset fails to remove active state or removes preserved run history.",
    ),
    TargetRisk(
        "risk.generated_proofs_reenter_prompts",
        "Generated proofs reenter source prompts",
        "high",
        ("prompt_context", "proof_scope_isolation", "workflow_filesystem_state"),
        (
            "prompt.generated_appendices_stripped",
            "proof_scope.generated_appendices_stripped_from_prompts",
            "prompt.proof_source_context_required",
        ),
        ("model", "real_coordinator"),
        "Generated appendices duplicate into source context or replace mandatory source text.",
    ),
    TargetRisk(
        "risk.hosted_desktop_boundary",
        "Hosted mode invokes desktop-only behavior",
        "critical",
        (
            "proof_runtime_gating",
            "websocket_api_contracts",
            "runtime_exclusivity",
            "allowed_outputs",
        ),
        (
            "proof_runtime.hosted_proof_settings_unavailable",
            "api.hosted_desktop_only_routes_unavailable",
        ),
        ("model", "real_route"),
        "Hosted mode reaches proof settings or another desktop-only route or behavior.",
    ),
    TargetRisk(
        "risk.context_overflow_activity_loses_identity",
        "Context overflow loses route identity or lifecycle semantics",
        "critical",
        (
            "websocket_api_contracts",
            "workflow_filesystem_state",
            "provider_pause_resume",
            "proof_runtime_gating",
        ),
        (
            "events.context_overflow_route_identity",
            "events.context_overflow_persists_across_reload",
            "events.context_overflow_terminal_stop_once",
            "events.proof_context_overflow_nonfatal",
        ),
        ("model", "real_coordinator"),
        (
            "Overflow activity loses configured/effective route attribution, disappears on reload, "
            "duplicates terminal stop activity, or turns a scoped proof overflow into a workflow stop."
        ),
    ),
    TargetRisk(
        "risk.direct_source_duplicated_by_rag",
        "Direct source is duplicated or displaced by RAG",
        "high",
        ("prompt_context", "workflow_filesystem_state", "proof_runtime_gating"),
        (
            "prompt.direct_sources_excluded_from_rag",
            "prompt.mandatory_source_overflow_fails_visible",
        ),
        ("model", "real_coordinator"),
        (
            "A directly injected source reappears as supplemental evidence, or optional retrieval "
            "silently replaces mandatory source text when the configured context is too small."
        ),
    ),
)


def validate_target_risks(risks: tuple[TargetRisk, ...] = TARGET_RISKS) -> None:
    ids = [risk.risk_id for risk in risks]
    if len(ids) != len(set(ids)):
        raise AssertionError("Target-risk ids must be unique.")
    for risk in risks:
        if not risk.risk_id.startswith("risk.") or not risk.title or not risk.description:
            raise AssertionError(f"Target risk {risk.risk_id!r} has incomplete identity.")
        if not risk.required_fields or not risk.supporting_invariants or not risk.required_adapters:
            raise AssertionError(f"{risk.risk_id} has incomplete support requirements.")
        unknown_fields = set(risk.required_fields) - FIRST_BUILD_FIELDS
        unknown_invariants = set(risk.supporting_invariants) - set(INVARIANTS_BY_ID)
        if unknown_fields:
            raise AssertionError(f"{risk.risk_id} references unknown fields {sorted(unknown_fields)!r}.")
        if unknown_invariants:
            raise AssertionError(
                f"{risk.risk_id} references unknown invariants {sorted(unknown_invariants)!r}."
            )
        recognized_fields: set[str] = set()
        for invariant_id in risk.supporting_invariants:
            invariant = INVARIANTS_BY_ID[invariant_id]
            recognized_fields.add(invariant.field)
            recognized_fields.update(invariant.crossed_fields)
        if not set(risk.required_fields).issubset(recognized_fields):
            raise AssertionError(f"{risk.risk_id} has fields disconnected from its invariants.")


def analyze_target_risks(
    records: tuple[InteractionCoverage, ...] | list[InteractionCoverage],
    risks: tuple[TargetRisk, ...] = TARGET_RISKS,
) -> tuple[TargetRiskAnalysis, ...]:
    validate_target_risks(risks)
    provisional: list[tuple[TargetRisk, int, tuple[AdapterSupport, ...]]] = []
    for risk in risks:
        support = tuple(_adapter_support(risk, adapter, records) for adapter in risk.required_adapters)
        gap_score = PRIORITY_WEIGHT[risk.priority] * sum(
            len(item.missing_invariants) + (1 if item.status != "passed" else 0)
            for item in support
        )
        provisional.append((risk, gap_score, support))

    ordered = sorted(provisional, key=lambda item: (-item[1], item[0].risk_id))
    analyses: list[TargetRiskAnalysis] = []
    for rank, (risk, gap_score, support) in enumerate(ordered, start=1):
        scenarios = tuple(sorted({item for entry in support for item in entry.scenario_ids}))
        missing = tuple(
            f"{entry.adapter}:{invariant_id}"
            for entry in support
            for invariant_id in entry.missing_invariants
        )
        analyses.append(
            TargetRiskAnalysis(risk, rank, gap_score, support, scenarios, missing)
        )
    return tuple(analyses)


def _adapter_support(
    risk: TargetRisk,
    adapter: AdapterType,
    records: tuple[InteractionCoverage, ...] | list[InteractionCoverage],
) -> AdapterSupport:
    matches = [
        record
        for record in records
        if record.adapter == adapter
        and set(record.invariants).intersection(risk.supporting_invariants)
    ]
    covered = {
        invariant_id
        for record in matches
        if record.result == "passed"
        for invariant_id in record.invariants
        if invariant_id in risk.supporting_invariants
    }
    missing = tuple(sorted(set(risk.supporting_invariants) - covered))
    statuses = tuple(sorted({record.result for record in matches}))
    status = _combined_status(statuses, missing)
    return AdapterSupport(
        adapter=adapter,
        status=status,
        scenario_ids=tuple(sorted({record.scenario_id for record in matches})),
        result_statuses=statuses,
        missing_invariants=missing,
    )


def _combined_status(
    statuses: tuple[ResultStatus, ...], missing_invariants: tuple[str, ...]
) -> SupportStatus:
    if "failed" in statuses:
        return "failed"
    if "blocked" in statuses:
        return "blocked"
    if "skipped" in statuses:
        return "skipped"
    if statuses and not missing_invariants:
        return "passed"
    return "uncovered"
