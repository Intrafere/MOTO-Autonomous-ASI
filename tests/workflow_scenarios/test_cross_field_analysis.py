from __future__ import annotations

import pytest

from tests.workflow_harness.actions import run_actions
from tests.workflow_harness.coverage_metadata import assert_coverage_records_valid
from tests.workflow_harness.cross_field_scenarios import (
    CROSS_FIELD_SCENARIOS,
    first_build_cross_field_coverage,
    validate_cross_field_scenario,
)
from tests.workflow_harness.exercise_evidence import assert_exercise_tokens
from tests.workflow_harness.invariants import assert_all_invariants
from tests.workflow_harness.model import WorkflowModel, WorkflowPhase


TEST_FILE = "tests/workflow_scenarios/test_cross_field_analysis.py"
HIGH_PRIORITY_FIELDS = {
    "allowed_outputs",
    "provider_pause_resume",
    "proof_runtime_gating",
    "assistant_memory",
    "proof_scope_isolation",
    "workflow_filesystem_state",
    "websocket_api_contracts",
    "prompt_context",
}


def test_cross_field_scenario_artifacts_are_valid():
    scenario_ids = [scenario.scenario_id for scenario in CROSS_FIELD_SCENARIOS]
    assert len(scenario_ids) == len(set(scenario_ids))

    for scenario in CROSS_FIELD_SCENARIOS:
        validate_cross_field_scenario(scenario)


@pytest.mark.parametrize("scenario", CROSS_FIELD_SCENARIOS, ids=lambda scenario: scenario.scenario_id)
def test_model_cross_field_scenarios_preserve_invariants_and_exercise_fields(tmp_path, scenario):
    model = WorkflowModel(runtime_root=tmp_path / scenario.scenario_id)

    run_actions(model, list(scenario.actions))

    assert_all_invariants(model)
    assert_exercise_tokens(model, scenario.must_exercise, scenario_id=scenario.scenario_id)


def test_first_build_cross_field_coverage_metadata_is_valid():
    records = first_build_cross_field_coverage(TEST_FILE)

    assert_coverage_records_valid(records)


def test_first_build_cross_field_coverage_map_includes_high_priority_fields():
    records = first_build_cross_field_coverage(TEST_FILE)
    covered_fields = {field for record in records for field in record.fields}
    covered_invariants = {invariant for record in records for invariant in record.invariants}

    assert HIGH_PRIORITY_FIELDS.issubset(covered_fields)
    assert {
        "outputs.proofs_only_no_paper_phase",
        "provider.pause_preserves_checkpoint",
        "assistant.no_validator_injection",
        "proof_scope.manual_clear_archives_active",
        "events.frontend_events_include_scope_phase",
        "proof_runtime.no_lean_when_disabled",
    }.issubset(covered_invariants)


def test_first_build_cross_field_scenarios_are_inspectable_without_runtime_dependencies():
    for scenario in CROSS_FIELD_SCENARIOS:
        assert scenario.adapter == "model"
        assert scenario.expected_result == "passed"
        assert not scenario.notes
        assert all(callable(action) for action in scenario.actions)


def test_proofs_only_cross_field_scenario_does_not_land_in_paper_phase(tmp_path):
    scenario = next(
        item
        for item in CROSS_FIELD_SCENARIOS
        if item.scenario_id == "model_allowed_outputs_provider_pause_stop_resume"
    )
    model = WorkflowModel(runtime_root=tmp_path / scenario.scenario_id)

    run_actions(model, list(scenario.actions))

    assert model.phase is WorkflowPhase.TIER1_AGGREGATION
    assert model.allow_mathematical_proofs is True
    assert model.allow_research_papers is False
