from __future__ import annotations

from collections.abc import Callable

from .invariants import assert_all_invariants
from .model import WorkflowModel


WorkflowAction = Callable[[WorkflowModel], None]


def run_actions(model: WorkflowModel, actions: list[WorkflowAction]) -> WorkflowModel:
    for action in actions:
        action(model)
        assert_all_invariants(model)
    return model


def start_autonomous_proofs_only(model: WorkflowModel) -> None:
    model.start_autonomous(proofs=True, papers=False, lean_enabled=True)


def start_autonomous_papers_and_proofs(model: WorkflowModel) -> None:
    model.start_autonomous(proofs=True, papers=True, lean_enabled=True)


def start_autonomous_papers_only(model: WorkflowModel) -> None:
    model.start_autonomous(proofs=False, papers=True, lean_enabled=False)


def start_autonomous_no_outputs(model: WorkflowModel) -> None:
    model.start_autonomous_with_no_outputs()


def start_manual_compiler(model: WorkflowModel) -> None:
    model.start_manual_compiler()


def start_manual_aggregator(model: WorkflowModel) -> None:
    model.start_manual_aggregator()


def accept_manual_aggregator_submission(model: WorkflowModel) -> None:
    model.accept_manual_aggregator_submission()


def start_leanoj(model: WorkflowModel) -> None:
    model.start_leanoj()


def edit_leanoj_master_proof(model: WorkflowModel) -> None:
    model.edit_leanoj_master_proof()


def skip_leanoj_brainstorm(model: WorkflowModel) -> None:
    model.skip_leanoj_brainstorm()


def force_leanoj_brainstorm(model: WorkflowModel) -> None:
    model.force_leanoj_brainstorm()


def enter_autonomous_paper_checkpoint(model: WorkflowModel) -> None:
    model.enter_autonomous_paper_checkpoint()


def complete_autonomous_paper_checkpoint(model: WorkflowModel) -> None:
    model.complete_autonomous_paper_checkpoint()


def attempt_conflict_during_autonomous_paper_checkpoint(model: WorkflowModel) -> None:
    model.start_manual_compiler()


def enable_manual_lean(model: WorkflowModel) -> None:
    model.record("enable_manual_lean")
    model.lean.enabled = True


def complete_topic_exploration(model: WorkflowModel) -> None:
    model.complete_topic_exploration()


def complete_brainstorm(model: WorkflowModel) -> None:
    model.complete_brainstorm()


def attempt_disabled_lean_checkpoint(model: WorkflowModel) -> None:
    model.attempt_disabled_lean_checkpoint()


def start_child_task(model: WorkflowModel) -> None:
    model.start_child_task()


def stale_child_output_arrives_after_parent_action(model: WorkflowModel) -> None:
    model.stale_child_output_arrives_after_parent_action()


def force_paper_writing(model: WorkflowModel) -> None:
    model.force_paper_writing()


def complete_paper(model: WorkflowModel) -> None:
    model.complete_paper()


def run_manual_proof_check(model: WorkflowModel) -> None:
    model.run_manual_proof_check()


def simulate_credit_exhaustion(model: WorkflowModel) -> None:
    model.simulate_credit_exhaustion()


def run_smt_hint_generation(model: WorkflowModel) -> None:
    model.run_smt_hint_generation()


def simulate_provider_output_truncation(model: WorkflowModel) -> None:
    model.simulate_provider_output_truncation()


def try_hosted_desktop_only_route(model: WorkflowModel) -> None:
    model.try_hosted_desktop_only_route()


def reset_credit(model: WorkflowModel) -> None:
    model.reset_credit()


def stop(model: WorkflowModel) -> None:
    model.stop()


def resume(model: WorkflowModel) -> None:
    model.resume()


def clear(model: WorkflowModel) -> None:
    model.clear()


def disable_session_history(model: WorkflowModel) -> None:
    model.toggle_session_history_memory(False)


def enable_session_history(model: WorkflowModel) -> None:
    model.toggle_session_history_memory(True)


def refresh_assistant_pack(model: WorkflowModel) -> None:
    model.refresh_assistant_pack("prior-proof-1", "prior-proof-2")


def assistant_retrieve_non_blocking(model: WorkflowModel) -> None:
    model.assistant_retrieve_non_blocking()


def assistant_stagnant_pack_backoff(model: WorkflowModel) -> None:
    model.assistant_stagnant_pack_backoff()


def complete_brainstorm_with_generated_paper(model: WorkflowModel) -> None:
    model.complete_brainstorm_with_generated_paper()


def resume_completed_brainstorm(model: WorkflowModel) -> None:
    model.resume_completed_brainstorm()


def prune_paper(model: WorkflowModel) -> None:
    model.prune_paper("paper-1")


def add_pruned_paper_to_model_context(model: WorkflowModel) -> None:
    model.add_paper_to_model_context("paper-1")


def prepare_prompt_context(model: WorkflowModel) -> None:
    model.prepare_prompt_context()


def prepare_validator_prompt_with_live_assistant(model: WorkflowModel) -> None:
    model.prepare_validator_prompt_with_live_assistant()


def resolve_runtime_path(model: WorkflowModel) -> None:
    model.resolve_runtime_path()


def verify_rag_source_exclusion(model: WorkflowModel) -> None:
    model.verify_rag_source_exclusion()


def reject_mandatory_source_overflow(model: WorkflowModel) -> None:
    model.reject_mandatory_source_overflow()


def emit_frontend_scoped_event(model: WorkflowModel) -> None:
    model.emit_frontend_scoped_event()


def emit_context_overflow_contract_events(model: WorkflowModel) -> None:
    model.emit_context_overflow_contract_events()


def emit_registered_proof_verified(model: WorkflowModel) -> None:
    model.emit_registered_proof_verified()

