from __future__ import annotations

import ast
from dataclasses import replace
import re
from pathlib import Path

import pytest

from tests.workflow_harness.coverage_metadata import (
    assert_coverage_metadata_valid,
    assert_coverage_records_valid,
)
from tests.workflow_harness.invariant_catalog import INVARIANTS_BY_ID, invariant_ids_for_adapter
from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE


def test_real_adapter_coverage_metadata_is_valid():
    assert_coverage_records_valid(REAL_ADAPTER_COVERAGE)


def test_vitest_coverage_requires_exact_file_suite_and_test_selector():
    vitest_record = next(record for record in REAL_ADAPTER_COVERAGE if record.runner == "vitest")

    with pytest.raises(AssertionError, match="non-exact Vitest selector"):
        assert_coverage_metadata_valid(
            replace(
                vitest_record,
                test_selectors=("frontend/src/utils/activityPersistence.test.js",),
            )
        )


def test_passed_real_coverage_links_exact_runner_nodes_and_claimed_invariants():
    for record in REAL_ADAPTER_COVERAGE:
        if record.result == "blocked":
            assert record.runner is None
            assert record.test_selectors == ()
            assert record.asserted_invariants == ()
            continue
        assert record.runner in {"pytest", "vitest"}
        assert record.test_selectors
        assert set(record.asserted_invariants) == set(record.invariants)
        assert record.real_components
        assert record.observed_transition
        assert record.adapter in {"focused_production", "real_route", "real_coordinator"}
        for component in record.real_components:
            assert Path(component).is_file(), component
        for selector in record.test_selectors:
            parts = selector.split("::")
            if record.runner == "vitest":
                assert len(parts) == 3
                source = Path(parts[0]).read_text(encoding="utf-8-sig")
                assert re.search(rf"\bdescribe\(\s*['\"]{re.escape(parts[1])}['\"]", source)
                assert re.search(rf"\b(?:test|it)\(\s*['\"]{re.escape(parts[2])}['\"]", source)
                continue
            module = ast.parse(Path(parts[0]).read_text(encoding="utf-8-sig"))
            owner = module.body
            if len(parts) == 3:
                class_node = next(
                    node
                    for node in module.body
                    if isinstance(node, ast.ClassDef) and node.name == parts[1]
                )
                owner = class_node.body
            assert any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == parts[-1]
                for node in owner
            ), selector


def test_blocked_real_coverage_names_unobservable_dependencies():
    for record in REAL_ADAPTER_COVERAGE:
        if record.result == "blocked":
            assert record.blocked_dependencies
            assert record.diagnostics and record.diagnostics.get("reason")


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
