from __future__ import annotations

import pytest

from tests.workflow_harness.exercise_evidence import assert_exercise_tokens
from tests.workflow_harness.recombinatory_generator import (
    DEFAULT_TARGETS,
    run_generated_scenario,
)
from tests.workflow_harness.model import WorkflowModel


SEEDS = (7, 41)


@pytest.mark.parametrize("target", DEFAULT_TARGETS, ids=lambda target: target.target_id)
@pytest.mark.parametrize("seed", SEEDS)
def test_seeded_workflow_action_recombination_preserves_invariants_and_coverage(
    tmp_path, target, seed
):
    model = WorkflowModel(runtime_root=tmp_path / target.target_id / f"seed-{seed}")

    result = run_generated_scenario(model, target, seed=seed)

    assert result.action_count <= target.max_steps
    assert result.fields == target.fields
    assert set(target.fields).issubset(result.observed_fields)
    assert result.invariants == target.invariants
    assert set(target.invariants) == result.exercised_invariants
    assert_exercise_tokens(model, target.must_exercise, scenario_id=target.target_id)
