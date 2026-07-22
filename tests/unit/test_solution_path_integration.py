from types import SimpleNamespace

import pytest

from backend.shared.solution_path.integration import (
    advisory_plan_block,
    enqueue_optional_update,
    validator_instruction,
    with_budgeted_solver_plan,
    with_validator_hook,
)
from backend.shared.solution_path.models import (
    SolutionPlan,
    SolutionRoute,
)


class _Manager:
    def __init__(self, *, active=True, plan=None):
        self.active = active
        self.state = SimpleNamespace(plan=plan)
        self.proposals = []

    async def propose(
        self,
        route,
        *,
        lifecycle_generation=0,
        rationale="",
        main_route="",
        proposer_role="validator",
        source_task_id=None,
        source_phase=None,
        source_decision=None,
    ):
        self.proposals.append(
            (
                route,
                rationale,
                main_route,
                proposer_role,
                source_task_id,
                source_phase,
                source_decision,
            )
        )
        return self.proposals[-1]


def test_plan_is_absent_before_activation():
    plan = SolutionPlan(run_id="run", route=SolutionRoute(steps=[]))
    assert advisory_plan_block(_Manager(active=False, plan=plan)) == ""
    prompt = "ordinary prompt"
    assert with_validator_hook(prompt, _Manager(active=False, plan=plan)) == prompt
    assert "solution_path_update" not in with_validator_hook(
        prompt, _Manager(active=False, plan=plan)
    )


def test_budgeted_solver_plan_is_optional_when_context_is_full():
    plan = SolutionPlan(
        run_id="run",
        main_route="Use the direct route",
        route=SolutionRoute(steps=[]),
    )
    manager = _Manager(active=True, plan=plan)
    prompt = "base prompt"
    assert with_budgeted_solver_plan(prompt, manager, 1) == prompt
    assert "[ADVISORY PROGRESSIVE SOLUTION PATH]" in with_budgeted_solver_plan(
        prompt, manager, 1000
    )


def test_validator_instruction_requests_one_optional_object():
    instruction = validator_instruction()
    assert "`solution_path_update`" in instruction
    assert "propose the initial path" in instruction
    assert "absence of an existing path is not a reason" in instruction
    assert "material revisions or meaningful progress" in instruction
    assert "Omit the entire extension otherwise" in instruction
    assert '"ordering":"ordered|parallel|mixed"' in instruction
    assert "sole permitted extension" in instruction
    batch_instruction = validator_instruction(batch=True)
    assert "top level beside `decisions`" in batch_instruction
    assert "never inside an individual decision" in batch_instruction


@pytest.mark.asyncio
async def test_optional_update_is_not_queued_before_activation():
    manager = _Manager(active=False)
    assert await enqueue_optional_update(
        {
            "solution_path_update": {
                "route": {
                    "ordering": "ordered",
                    "steps": [{"title": "Too early"}],
                }
            }
        },
        manager,
    ) is None
    assert manager.proposals == []


@pytest.mark.asyncio
async def test_malformed_optional_update_does_not_raise_or_enqueue():
    manager = _Manager()
    assert await enqueue_optional_update(
        {"decision": "accept", "solution_path_update": {"route": "bad"}},
        manager,
    ) is None
    assert manager.proposals == []


@pytest.mark.asyncio
async def test_valid_optional_update_is_durably_enqueued():
    manager = _Manager()
    proposal = await enqueue_optional_update(
            {
                "decision": "accept",
                "solution_path_update": {
                    "main_route": "Resolve the central obstruction",
                    "route": {
                        "ordering": "ordered",
                        "steps": [{"title": "Resolve the obstruction"}],
                    },
                    "rationale": "Materially sharper route",
                },
            },
        manager,
        proposer_role="agg_val",
        source_task_id="agg_val_001",
        source_phase="submission_validation",
        source_decision="accept",
    )
    assert proposal is not None
    assert manager.proposals[0][1] == "Materially sharper route"
    assert manager.proposals[0][2] == "Resolve the central obstruction"
    assert manager.proposals[0][4:] == (
        "agg_val_001",
        "submission_validation",
        "accept",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ["parallel", "mixed"])
async def test_parallel_optional_update_orderings_are_normalized(ordering):
    manager = _Manager()
    proposal = await enqueue_optional_update(
        {
            "solution_path_update": {
                "main_route": "Pursue independent workstreams",
                "route": {
                    "ordering": ordering,
                    "steps": [{"title": "Independent workstream"}],
                },
            }
        },
        manager,
    )

    assert proposal is not None
    assert manager.proposals[0][0].ordering.value == "unordered"


@pytest.mark.asyncio
async def test_initial_update_requires_nonempty_main_route():
    manager = _Manager()
    assert await enqueue_optional_update(
        {
            "solution_path_update": {
                "route": {
                    "ordering": "ordered",
                    "steps": [{"title": "First route"}],
                }
            }
        },
        manager,
    ) is None
    assert manager.proposals == []


@pytest.mark.asyncio
async def test_later_update_omission_preserves_main_route():
    plan = SolutionPlan(
        run_id="run",
        main_route="Preserve this route",
        route=SolutionRoute(steps=[]),
    )
    manager = _Manager(plan=plan)
    await enqueue_optional_update(
        {
            "solution_path_update": {
                "route": {
                    "ordering": "ordered",
                    "steps": [{"title": "Refined step"}],
                }
            }
        },
        manager,
    )
    assert manager.proposals[0][2] == "Preserve this route"


def test_validator_and_solver_injection_use_only_active_canonical_plan():
    from backend.shared.solution_path.integration import (
        with_solver_plan,
        with_validator_hook,
    )

    plan = SolutionPlan(
        run_id="run",
        main_route="Attack the obstruction",
        route=SolutionRoute(steps=[]),
    )
    inactive = _Manager(active=False, plan=plan)
    active = _Manager(active=True, plan=plan)

    assert with_validator_hook("validator", inactive) == "validator"
    assert with_solver_plan("solver", inactive) == "solver"
    validator_prompt = with_validator_hook("validator", active)
    solver_prompt = with_solver_plan("solver", active)
    assert "Attack the obstruction" in validator_prompt
    assert "solution_path_update" in validator_prompt
    assert "Attack the obstruction" in solver_prompt
    assert "solution_path_update" not in solver_prompt
