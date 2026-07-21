from __future__ import annotations

import os

import pytest

from tests.workflow_harness.exercise_evidence import assert_exercise_tokens
from tests.workflow_harness.model import WorkflowModel
from tests.workflow_harness.recombinatory_generator import (
    DEFAULT_TARGETS,
    run_generated_scenario,
)


pytestmark = pytest.mark.skipif(
    os.getenv("MOTO_WORKFLOW_DEEP_TESTS") != "1",
    reason="set MOTO_WORKFLOW_DEEP_TESTS=1 to run deep workflow tests",
)

DEEP_SEEDS = (3, 7, 41, 53, 73, 89, 149, 211, 307, 401)


@pytest.mark.parametrize("target", DEFAULT_TARGETS, ids=lambda target: target.target_id)
@pytest.mark.parametrize("seed", DEEP_SEEDS)
def test_deep_seed_matrix_preserves_invariants_coverage_and_replay(
    tmp_path, target, seed
) -> None:
    first_model = WorkflowModel(
        runtime_root=tmp_path / target.target_id / f"seed-{seed}" / "first"
    )
    first = run_generated_scenario(
        first_model,
        target,
        seed=seed,
    )
    replayed = run_generated_scenario(
        WorkflowModel(runtime_root=tmp_path / target.target_id / f"seed-{seed}" / "replay"),
        target,
        seed=seed,
    )

    assert first.action_count <= target.max_steps
    assert first.fields == target.fields
    assert set(target.fields).issubset(first.observed_fields)
    assert first.invariants == target.invariants
    assert set(target.invariants) == first.exercised_invariants
    assert_exercise_tokens(
        first_model,
        target.must_exercise,
        scenario_id=target.target_id,
    )
    assert first.action_ids == replayed.action_ids
    assert first.replay == replayed.replay
