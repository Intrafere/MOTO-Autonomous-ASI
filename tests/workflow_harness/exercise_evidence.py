from __future__ import annotations

from collections.abc import Iterable

from .model import WorkflowModel


KNOWN_EXERCISE_TOKENS = frozenset(
    {
        "assistant_disable",
        "assistant_pack",
        "context_overflow",
        "manual_archive",
        "manual_aggregator_accept",
        "manual_aggregator_clear",
        "manual_proof",
        "leanoj_clear",
        "leanoj_draft",
        "leanoj_force",
        "leanoj_skip",
        "paper_checkpoint",
        "no_outputs_rejected",
        "papers_only",
        "prompt_context",
        "rag_source_exclusion",
        "mandatory_source_overflow",
        "provider_pause",
        "registered_proof_event",
        "reset_credit",
        "scoped_event",
        "start_blocked",
        "stop_resume",
    }
)


def observed_exercise_tokens(model: WorkflowModel) -> frozenset[str]:
    event_types = {event.event_type for event in model.events}
    replay = model.replay
    observed: set[str] = set()

    if model.provider.pause_count >= 1 and "provider_paused" in event_types:
        observed.add("provider_pause")
    stop_indexes = [index for index, action in enumerate(replay) if action == "stop()"]
    resume_indexes = [
        index for index, action in enumerate(replay) if action.startswith("resume(")
    ]
    if (
        stop_indexes
        and resume_indexes
        and any(stop_index < resume_index for stop_index in stop_indexes for resume_index in resume_indexes)
        and "stopped" in event_types
        and "resumed" in event_types
    ):
        observed.add("stop_resume")
    paused_event_indexes = [
        index
        for index, event in enumerate(model.events)
        if event.event_type == "provider_paused"
    ]
    resumed_event_indexes = [
        index
        for index, event in enumerate(model.events)
        if event.event_type == "provider_resumed"
    ]
    if (
        not model.provider.credit_exhausted
        and paused_event_indexes
        and resumed_event_indexes
        and any(
            paused_index < resumed_index
            for paused_index in paused_event_indexes
            for resumed_index in resumed_event_indexes
        )
        and model.checkpoint.get("paused") is False
        and model.checkpoint.get("phase")
    ):
        observed.add("reset_credit")
    if any(action.startswith("refresh_assistant_pack(") for action in replay):
        observed.add("assistant_pack")
    if (
        any("toggle_session_history_memory(enabled=False)" in action for action in replay)
        and not model.assistant.enabled
        and not model.assistant.live_pack
    ):
        observed.add("assistant_disable")
    if (
        any(action.startswith("prepare_prompt_context(") for action in replay)
        and {
            "user_prompt_direct_injected",
            "validator_assistant_memory_excluded",
            "proof_source_context_present",
            "generated_appendices_stripped",
        }.issubset(model.exercise_observations)
    ):
        observed.add("prompt_context")
    if "manual-proof-1" in model.manual_proofs_active.union(model.manual_proofs_archived):
        observed.add("manual_proof")
    if "manual-proof-1" in model.manual_proofs_archived and not model.manual_proofs_active:
        observed.add("manual_archive")
    if model.manual_aggregator_submissions or model.manual_aggregator_history:
        observed.add("manual_aggregator_accept")
    if model.manual_aggregator_history and not model.manual_aggregator_submissions:
        observed.add("manual_aggregator_clear")
    if model.leanoj_draft_written and (
        model.leanoj_master_proof or model.leanoj_draft_preserved_on_resume or model.leanoj_cleared
    ):
        observed.add("leanoj_draft")
    if model.leanoj_skip_count:
        observed.add("leanoj_skip")
    if model.leanoj_force_count:
        observed.add("leanoj_force")
    if model.leanoj_cleared and not model.leanoj_master_proof:
        observed.add("leanoj_clear")
    if model.autonomous_paper_checkpoint_count:
        observed.add("paper_checkpoint")

    explicitly_scoped_events = [
        event
        for event in model.events
        if event.event_type in {"proof_progress", "proof_verified"}
    ]
    if explicitly_scoped_events and all(
        "scope" in event.payload and "phase" in event.payload
        for event in explicitly_scoped_events
    ):
        observed.add("scoped_event")
    if "registered-proof-1" in model.autonomous_proofs and any(
        event.event_type == "proof_verified"
        and event.payload.get("proof_id") == "registered-proof-1"
        for event in model.events
    ):
        observed.add("registered_proof_event")
    if any(
        event.event_type == "start_rejected"
        and event.payload.get("reason") == "no_allowed_outputs"
        for event in model.events
    ):
        observed.add("no_outputs_rejected")
    if (
        any(
            action.startswith("start_autonomous(")
            and "papers=True" in action
            and "proofs=False" in action
            for action in replay
        )
        and any(action.startswith("complete_brainstorm(") for action in replay)
        and not model.allow_mathematical_proofs
        and model.allow_research_papers
        and model.lean.invocations == 0
        and model.smt.invocations == 0
    ):
        observed.add("papers_only")
    if "start_blocked" in event_types:
        observed.add("start_blocked")
    if (
        "context_overflow_error" in event_types
        and "proof_context_overflow" in event_types
        and model.terminal_stop_events == 1
    ):
        observed.add("context_overflow")
    if (
        any(action.startswith("verify_rag_source_exclusion(") for action in replay)
        and "direct_sources_excluded_from_rag" in model.exercise_observations
    ):
        observed.add("rag_source_exclusion")
    if (
        any(action.startswith("reject_mandatory_source_overflow(") for action in replay)
        and "mandatory_source_overflow_rejected_visible" in model.exercise_observations
    ):
        observed.add("mandatory_source_overflow")

    return frozenset(observed)


def assert_exercise_tokens(
    model: WorkflowModel,
    required: Iterable[str],
    *,
    scenario_id: str,
) -> frozenset[str]:
    required_tokens = frozenset(required)
    unknown = required_tokens - KNOWN_EXERCISE_TOKENS
    if unknown:
        raise AssertionError(
            f"{scenario_id} has unknown must_exercise tokens: {sorted(unknown)!r}."
        )

    observed = observed_exercise_tokens(model)
    missing = required_tokens - observed
    if missing:
        raise AssertionError(
            f"{scenario_id} did not exercise required behavior {sorted(missing)!r}; "
            f"observed {sorted(observed)!r}."
        )
    return observed
