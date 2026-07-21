from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from tests.workflow_harness.real_adapters.maturity_registry import (
    AdapterMaturity,
    REAL_ADAPTER_MATURITY_REGISTRY,
    RepeatContract,
    descriptors_for,
    validate_maturity_registry,
)
from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DEEP_IDS = {
    "real_actual_route_start_conflict_matrix_no_side_effects",
    "real_shared_start_guard_representative_race",
    "real_proof_scope_matrix_isolated_under_temp_root",
    "real_direct_source_rag_exclusion_matrix",
    "real_model_visible_prompts_strip_generated_proof_appendices",
    "real_rigor_mandatory_source_overflow_fails_before_model_or_rag",
}
EXPECTED_BLOCKED_IDS = {
    "real_parent_action_fencing_unavailable_without_production_seam",
    "real_provider_stop_reset_checkpoint_unavailable_without_wait_seam",
    "real_leanoj_full_final_loop_not_safely_bounded",
}


def test_build_f_registry_describes_every_build_b_through_e_real_scenario():
    validate_maturity_registry(workspace_root=WORKSPACE_ROOT)

    assert {item.scenario_id for item in REAL_ADAPTER_MATURITY_REGISTRY} == {
        item.scenario_id for item in REAL_ADAPTER_COVERAGE
    }
    assert descriptors_for(AdapterMaturity.NORMAL)
    assert descriptors_for(AdapterMaturity.DEEP)


def test_build_f_maturity_matches_coverage_result_and_exact_partitions():
    blocked = descriptors_for(AdapterMaturity.BLOCKED)
    deep = descriptors_for(AdapterMaturity.DEEP)
    normal = descriptors_for(AdapterMaturity.NORMAL)
    coverage_by_id = {item.scenario_id: item for item in REAL_ADAPTER_COVERAGE}

    assert {item.scenario_id for item in blocked} == EXPECTED_BLOCKED_IDS
    assert {item.scenario_id for item in deep} == EXPECTED_DEEP_IDS
    partitions = [
        {item.scenario_id for item in maturity}
        for maturity in (normal, deep, blocked)
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    )
    assert set().union(*partitions) == set(coverage_by_id)
    assert all(coverage_by_id[item.scenario_id].result == "blocked" for item in blocked)
    assert all(
        coverage_by_id[item.scenario_id].result == "passed"
        for item in (*normal, *deep)
    )
    assert all(not item.executable for item in blocked)
    assert all(not item.repeat_safe for item in blocked)
    assert all(item.blocked_reason for item in blocked)


def test_build_f_deep_promotions_are_repeat_safe_bounded_and_cleanup_guarded():
    deep = descriptors_for(AdapterMaturity.DEEP)

    assert len(deep) >= 6
    for descriptor in deep:
        assert descriptor.executable
        assert descriptor.repeat_safe
        assert 0 < descriptor.timeout_seconds <= 180
        assert descriptor.cleanup.uses_temporary_roots
        assert descriptor.cleanup.removes_runtime_state
        assert descriptor.cleanup.forbids_workspace_escape
        assert descriptor.pytest_args()[0] == "-q"
        assert descriptor.repeat_contract in {
            RepeatContract.NORMALIZED_PROCESS_RESULT,
            RepeatContract.NORMALIZED_PROCESS_AND_RUNTIME_STATE,
        }
        assert descriptor.repeat_contract_reason


def test_build_f_all_executable_selectors_are_exact_collectable_node_ids(tmp_path):
    executable = [
        descriptor
        for descriptor in REAL_ADAPTER_MATURITY_REGISTRY
        if descriptor.executable
    ]
    selectors = [
        selector
        for descriptor in executable
        for selector in descriptor.test_selectors
    ]

    assert selectors
    assert len(selectors) == len(set(selectors))
    assert all(selector.count("::") == 1 for selector in selectors)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(tmp_path / "collect"),
            *selectors,
        ],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        f"selector collection failed\nSTDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )
    collected = {
        line.strip().replace("\\", "/").split("[", 1)[0]
        for line in completed.stdout.splitlines()
        if "::test_" in line
    }
    assert collected == set(selectors)
