from __future__ import annotations

from .invariant_catalog import assert_all_catalog_invariants, assert_invariant
from .model import WorkflowModel


def assert_observed_invariant(model: WorkflowModel, invariant_id: str) -> None:
    """Assert one real-adapter invariant only after the test marks it observed."""
    observed = getattr(model, "observed_invariants", None)
    if observed is not None and invariant_id not in observed:
        raise AssertionError(f"Invariant {invariant_id!r} was not exercised by this observation.")
    assert_invariant(model, invariant_id)


def assert_single_active_workflow(model: WorkflowModel) -> None:
    assert_invariant(model, "runtime.single_active_workflow")


def assert_lean_disabled_means_no_invocations(model: WorkflowModel) -> None:
    assert_invariant(model, "proof_runtime.no_lean_when_disabled")


def assert_proofs_only_never_enters_paper_phase(model: WorkflowModel) -> None:
    assert_invariant(model, "outputs.proofs_only_no_paper_phase")


def assert_provider_pause_preserves_checkpoint(model: WorkflowModel) -> None:
    assert_invariant(model, "provider.pause_preserves_checkpoint")


def assert_assistant_not_injected_into_validators(model: WorkflowModel) -> None:
    assert_invariant(model, "assistant.no_validator_injection")


def assert_disabled_assistant_has_no_live_pack(model: WorkflowModel) -> None:
    assert_invariant(model, "assistant.disable_clears_live_pack")


def assert_manual_and_autonomous_proofs_are_isolated(model: WorkflowModel) -> None:
    assert_invariant(model, "proof_scope.manual_not_in_autonomous_current")


def assert_clear_archives_manual_proofs(model: WorkflowModel) -> None:
    assert_invariant(model, "proof_scope.manual_clear_archives_active")


def assert_all_invariants(model: WorkflowModel) -> None:
    assert_all_catalog_invariants(model)

