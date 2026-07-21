from __future__ import annotations

import json

import pytest

from tests.workflow_cross_field.support_graph import (
    SUPPORT_GRAPH_SCHEMA_VERSION,
    build_support_graph,
    render_support_graph_json,
    render_support_graph_markdown,
    validate_support_graph,
)
from tests.workflow_harness.coverage_metadata import InteractionCoverage
from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE


def test_support_graph_is_deterministic_across_record_order():
    forward_json = render_support_graph_json(REAL_ADAPTER_COVERAGE)
    reverse_json = render_support_graph_json(reversed(REAL_ADAPTER_COVERAGE))
    forward_markdown = render_support_graph_markdown(REAL_ADAPTER_COVERAGE)
    reverse_markdown = render_support_graph_markdown(reversed(REAL_ADAPTER_COVERAGE))

    assert forward_json == reverse_json
    assert forward_markdown == reverse_markdown
    assert json.loads(forward_json)["schema_version"] == SUPPORT_GRAPH_SCHEMA_VERSION


def test_support_graph_separates_declared_and_observed_basis():
    graph = build_support_graph(REAL_ADAPTER_COVERAGE)

    hosted = [
        edge
        for edge in graph["edges"]
        if edge["invariant_id"] == "proof_runtime.hosted_proof_settings_unavailable"
        and edge["adapter"] == "real_route"
    ]
    parent_fence = [
        edge
        for edge in graph["edges"]
        if edge["invariant_id"] == "runtime.parent_action_fences_child_outputs"
        and edge["adapter"] == "real_coordinator"
    ]

    assert any(edge["observed"] and edge["status"] == "passed" for edge in hosted)
    assert any(
        edge["observed"]
        and not edge["declared"]
        and edge["status"] == "blocked"
        and edge["reason"]
        for edge in parent_fence
    )


def test_support_graph_emits_structured_required_adapter_gaps():
    graph = build_support_graph(())

    assert graph["gaps"]
    assert all(gap["status"] == "uncovered" for gap in graph["gaps"])
    assert all(gap["gap_id"] and gap["reason"] for gap in graph["gaps"])


def test_support_graph_preserves_failed_skipped_and_blocked_statuses():
    records = tuple(
        InteractionCoverage(
            scenario_id=f"status-{status}",
            fields=("proof_runtime_gating",),
            invariants=("proof_runtime.no_lean_when_disabled",),
            adapter="model",
            result=status,
            test_file="tests/workflow_cross_field/test_support_graph.py",
            diagnostics={"reason": f"{status} reason"} if status != "passed" else None,
            evidence=("observed",) if status == "passed" else (),
        )
        for status in ("passed", "failed", "skipped", "blocked")
    )

    graph = build_support_graph(records)
    statuses = {
        edge["status"]
        for edge in graph["edges"]
        if edge["scenario_id"] and edge["scenario_id"].startswith("status-")
    }

    assert statuses == {"passed", "failed", "skipped", "blocked"}


def test_support_graph_validation_rejects_false_observation():
    graph = build_support_graph(REAL_ADAPTER_COVERAGE)
    graph["edges"][0]["observed"] = True
    graph["edges"][0]["scenario_id"] = None

    with pytest.raises(AssertionError, match="Observed support edge"):
        validate_support_graph(graph)
