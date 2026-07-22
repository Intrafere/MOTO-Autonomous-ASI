from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflow_cross_field.target_risks import (
    TARGET_RISKS,
    analyze_target_risks,
    validate_target_risks,
)
from tests.workflow_harness.coverage_metadata import InteractionCoverage


def _record(
    scenario_id: str,
    invariant: str,
    *,
    adapter: str = "model",
    result: str = "passed",
) -> InteractionCoverage:
    return InteractionCoverage(
        scenario_id=scenario_id,
        fields=("runtime_exclusivity", "workflow_filesystem_state"),
        invariants=(invariant,),
        adapter=adapter,
        result=result,
        test_file="tests/workflow_cross_field/test_target_risks.py",
        diagnostics={"reason": "fixture"} if result != "passed" else None,
        evidence=("fixture",) if result == "passed" else (),
    )


def test_catalog_has_exactly_twelve_unique_recognized_risks():
    validate_target_risks()
    assert len(TARGET_RISKS) == 12
    risk_ids = {risk.risk_id for risk in TARGET_RISKS}
    assert len(risk_ids) == 12
    assert "risk.pruned_paper_context_leak" in risk_ids
    assert "risk.hosted_desktop_boundary" in risk_ids
    assert "risk.context_overflow_activity_loses_identity" in risk_ids
    assert "risk.direct_source_duplicated_by_rag" in risk_ids
    assert "risk.concurrent_workflow_ownership" not in risk_ids
    assert "risk.proof_event_precedes_registration" not in risk_ids


@pytest.mark.parametrize(
    "change, message",
    [
        ({"risk_id": ""}, "incomplete identity"),
        ({"required_fields": ("unknown",)}, "unknown fields"),
        ({"supporting_invariants": ("unknown",)}, "unknown invariants"),
        ({"required_fields": ("assistant_memory",)}, "disconnected"),
    ],
)
def test_catalog_validation_rejects_unrecognized_or_disconnected_values(change, message):
    risk = replace(TARGET_RISKS[0], **change)
    with pytest.raises(AssertionError, match=message):
        validate_target_risks((risk,))


def test_support_preserves_adapter_and_result_status_distinctions():
    risk = next(item for item in TARGET_RISKS if item.risk_id == "risk.provider_pause_loses_checkpoint")
    records = (
        _record("model_pass", risk.supporting_invariants[0]),
        _record(
            "coordinator_block",
            risk.supporting_invariants[0],
            adapter="real_coordinator",
            result="blocked",
        ),
    )
    analysis = analyze_target_risks(records, (risk,))[0]
    by_adapter = {item.adapter: item for item in analysis.support}
    assert by_adapter["model"].status == "uncovered"
    assert by_adapter["model"].result_statuses == ("passed",)
    assert by_adapter["real_coordinator"].status == "blocked"
    assert by_adapter["real_coordinator"].result_statuses == ("blocked",)
    assert analysis.missing_isolated_artifacts


def test_complete_passed_support_removes_adapter_gap():
    risk = TARGET_RISKS[0]
    records = tuple(
        _record(f"{adapter}_{index}", invariant, adapter=adapter)
        for adapter in risk.required_adapters
        for index, invariant in enumerate(risk.supporting_invariants)
    )
    analysis = analyze_target_risks(records, (risk,))[0]
    assert all(item.status == "passed" for item in analysis.support)
    assert analysis.missing_isolated_artifacts == ()
    assert analysis.gap_score == 0


def test_ranking_is_deterministic_with_risk_id_tiebreak():
    first = analyze_target_risks((), TARGET_RISKS)
    second = analyze_target_risks((), tuple(reversed(TARGET_RISKS)))
    assert [item.risk.risk_id for item in first] == [item.risk.risk_id for item in second]
    assert [(item.rank, item.gap_score) for item in first] == [
        (item.rank, item.gap_score) for item in second
    ]
