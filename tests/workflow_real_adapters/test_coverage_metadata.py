from __future__ import annotations

from tests.workflow_harness.coverage_metadata import assert_coverage_records_valid
from tests.workflow_harness.invariant_catalog import INVARIANTS_BY_ID, invariant_ids_for_adapter
from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE


def test_real_adapter_coverage_metadata_is_valid():
    assert_coverage_records_valid(REAL_ADAPTER_COVERAGE)


def test_passed_real_coverage_links_exact_pytest_nodes_and_claimed_invariants():
    for record in REAL_ADAPTER_COVERAGE:
        if record.result == "blocked":
            assert record.runner is None
            assert record.test_selectors == ()
            assert record.asserted_invariants == ()
            continue
        assert record.runner == "pytest"
        assert record.test_selectors
        assert set(record.asserted_invariants) == set(record.invariants)


def test_first_build_real_adapter_coverage_advertises_landed_scenarios():
    covered_by_real_route = {
        invariant_id
        for record in REAL_ADAPTER_COVERAGE
        if record.adapter == "real_route" and record.result != "blocked"
        for invariant_id in record.invariants
    }
    covered_by_real_coordinator = {
        invariant_id
        for record in REAL_ADAPTER_COVERAGE
        if record.adapter == "real_coordinator" and record.result != "blocked"
        for invariant_id in record.invariants
    }

    assert "runtime.single_active_workflow" in covered_by_real_route
    assert {
        "outputs.proofs_only_no_paper_phase",
        "provider.pause_preserves_checkpoint",
        "proof_scope.manual_not_in_autonomous_current",
    }.issubset(covered_by_real_coordinator)
    assert covered_by_real_route.issubset(invariant_ids_for_adapter("real_route"))
    assert covered_by_real_coordinator.issubset(invariant_ids_for_adapter("real_coordinator"))


def test_plan2_real_adapter_gaps_are_explicitly_model_only_or_future_work():
    real_passed = {
        invariant_id
        for record in REAL_ADAPTER_COVERAGE
        if record.result == "passed"
        for invariant_id in record.invariants
    }
    model_only_or_future = set(INVARIANTS_BY_ID) - real_passed

    assert model_only_or_future
    assert {
        "runtime.parent_action_fences_child_outputs",
        "proof_runtime.no_smt_when_disabled",
        "prompt.proof_source_context_required",
    }.issubset(model_only_or_future)
