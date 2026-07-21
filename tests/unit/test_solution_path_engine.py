import asyncio

import pytest

from backend.shared.solution_path import (
    ReviewDecision,
    RouteEdit,
    RouteStep,
    SolutionPathEngine,
    SolutionRoute,
    StepOrdering,
)


def route(title: str, ordering: StepOrdering = StepOrdering.ORDERED) -> SolutionRoute:
    return SolutionRoute(ordering=ordering, steps=[RouteStep(title=title)])


@pytest.mark.asyncio
async def test_inactive_until_five_then_persists_one_plan(tmp_path):
    calls = []

    async def reviewer(proposal, plan):
        calls.append((proposal.proposal_id, plan.revision if plan else 0))
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-a", reviewer)
    await engine.propose(route("first"))
    await engine.set_acceptance_count(4)
    await asyncio.sleep(0)
    assert calls == []
    assert engine.state.plan is None

    await engine.set_acceptance_count(5)
    await engine.wait_idle()
    assert engine.state.plan.route.steps[0].title == "first"
    assert engine.state.plan.revision == 1

    restored = SolutionPathEngine(tmp_path, "run-a", reviewer)
    assert restored.state.plan.revision == 1
    assert restored.state.plan.route.ordering == StepOrdering.ORDERED


@pytest.mark.asyncio
async def test_acceptance_count_is_monotonic_and_activation_emits_once(tmp_path, monkeypatch):
    events = []

    async def reviewer(proposal, plan):
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-monotonic", reviewer)

    async def capture(event, payload):
        events.append(event)

    monkeypatch.setattr(engine, "_broadcast", capture)
    await engine.set_acceptance_count(4)
    await engine.set_acceptance_count(3)
    assert engine.state.acceptance_count == 4

    await asyncio.gather(
        engine.set_acceptance_count(5),
        engine.set_acceptance_count(7),
        engine.set_acceptance_count(6),
    )
    assert engine.state.acceptance_count == 7
    assert events.count("solution_path_activated") == 1


@pytest.mark.asyncio
async def test_serial_followup_and_unordered_route(tmp_path):
    active = 0
    max_active = 0
    seen = {}

    async def reviewer(proposal, plan):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        count = seen.get(proposal.proposal_id, 0)
        seen[proposal.proposal_id] = count + 1
        if count == 0 and proposal.route.steps[0].title == "follow":
            return ReviewDecision(
                decision="followup",
                reasoning="revise",
                followup_route=route("revised", StepOrdering.UNORDERED),
            )
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-b", reviewer)
    await engine.set_acceptance_count(5)
    await engine.propose(route("follow"))
    await engine.propose(route("second"))
    await engine.wait_idle()
    assert max_active == 1
    assert engine.state.plan.revision == 2
    assert engine.state.proposals[0].review_count == 2
    assert engine.state.proposals[0].route.ordering == StepOrdering.UNORDERED
    assert [p.status.value for p in engine.state.proposals] == ["approved", "approved"]


@pytest.mark.asyncio
async def test_stale_revision_is_re_reviewed(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-c", reviewer)
    await engine.set_acceptance_count(5)
    await engine.propose(route("stale"))
    await entered.wait()
    # Simulate a concurrent durable plan revision while review is outside the lock.
    engine._state.plan = engine._state.plan or __import__(
        "backend.shared.solution_path", fromlist=["SolutionPlan"]
    ).SolutionPlan(run_id="run-c", route=route("concurrent"))
    release.set()
    await engine.wait_idle()
    assert calls == 2
    assert engine.state.plan.revision == 2
    assert any(a.event == "proposal_stale_requeued" for a in engine.state.audit)


@pytest.mark.asyncio
async def test_stop_preserves_and_clear_deletes(tmp_path):
    entered = asyncio.Event()

    async def reviewer(proposal, plan):
        entered.set()
        await asyncio.Event().wait()

    engine = SolutionPathEngine(tmp_path, "run-d", reviewer)
    await engine.set_acceptance_count(5)
    await engine.propose(route("pending"))
    await entered.wait()
    await engine.stop()
    state_path = tmp_path / "run-d" / "solution_path_state.json"
    assert state_path.exists()
    assert engine.state.proposals[0].status.value == "queued"

    await engine.clear()
    assert not state_path.exists()


@pytest.mark.asyncio
async def test_scoped_lifecycle_queue_review_and_update_events(tmp_path, monkeypatch):
    events = []

    async def reviewer(proposal, plan):
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(
        tmp_path,
        "run-events",
        reviewer,
        workflow_mode="autonomous",
    )

    async def capture(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(engine, "_broadcast", capture)
    await engine.set_acceptance_count(5)
    await engine.propose(route("event route"))
    await engine.wait_idle()

    names = [event for event, _ in events]
    assert names == [
        "solution_path_activated",
        "solution_path_proposal_queued",
        "solution_path_proposal_reviewing",
        "solution_path_updated",
    ]
    for _, payload in events:
        assert set(payload) == {
            "workflow_mode",
            "run_id",
            "acceptance_count",
            "prompt_hash",
            "lifecycle_generation",
            "revision",
            "proposal_id",
            "proposal_status",
            "base_revision",
            "decision",
            "source_task_id",
            "source_phase",
            "source_decision",
            "queued_proposals",
            "reviewing_proposals",
            "repair_required_proposals",
            "repair_reason",
            "message",
            "detail",
        }
        assert payload["workflow_mode"] == "autonomous"
        assert payload["run_id"] == "run-events"
        assert payload["acceptance_count"] == 5
        assert payload["prompt_hash"] is None
        assert isinstance(payload["revision"], int)
        assert isinstance(payload["queued_proposals"], int)
        assert isinstance(payload["reviewing_proposals"], int)
        assert isinstance(payload["message"], str)
    assert events[0][1]["proposal_id"] is None
    assert events[1][1]["proposal_status"] == "queued"
    assert events[2][1]["proposal_status"] == "reviewing"
    assert events[3][1]["proposal_status"] == "approved"


@pytest.mark.asyncio
async def test_stop_start_resumes_same_run_and_pending_proposal(tmp_path):
    first_entered = asyncio.Event()
    resumed_calls = 0

    async def blocked_reviewer(proposal, plan):
        first_entered.set()
        await asyncio.Event().wait()

    engine = SolutionPathEngine(
        tmp_path,
        "run-resume",
        blocked_reviewer,
        workflow_mode="autonomous",
        user_prompt="Find a route",
    )
    await engine.set_acceptance_count(5)
    await engine.propose(route("pending"))
    await first_entered.wait()
    await engine.stop()

    async def resumed_reviewer(proposal, plan):
        nonlocal resumed_calls
        resumed_calls += 1
        return ReviewDecision(decision="approve")

    restored = SolutionPathEngine(
        tmp_path,
        "run-resume",
        resumed_reviewer,
        workflow_mode="autonomous",
        user_prompt="Find a route",
    )
    await restored.start()
    await restored.wait_idle()
    assert resumed_calls == 1
    assert restored.state.plan.route.steps[0].title == "pending"
    assert restored.state.proposals[0].status.value == "approved"


@pytest.mark.asyncio
async def test_run_ownership_rejects_prompt_or_mode_mismatch(tmp_path):
    async def reviewer(proposal, plan):
        return ReviewDecision(decision="approve")

    owner = SolutionPathEngine(
        tmp_path,
        "owned-run",
        reviewer,
        workflow_mode="manual",
        user_prompt="Original prompt",
    )
    await owner.set_acceptance_count(1)

    with pytest.raises(ValueError, match="workflow_mode mismatch"):
        SolutionPathEngine(
            tmp_path,
            "owned-run",
            reviewer,
            workflow_mode="leanoj",
            user_prompt="Original prompt",
        )
    with pytest.raises(ValueError, match="user_prompt mismatch"):
        SolutionPathEngine(
            tmp_path,
            "owned-run",
            reviewer,
            workflow_mode="manual",
            user_prompt="Different prompt",
        )


@pytest.mark.asyncio
async def test_retry_is_automatic_and_does_not_drop_proposal(tmp_path):
    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary transport failure")
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(
        tmp_path,
        "run-retry",
        reviewer,
        retry_base_seconds=0.01,
        retry_max_seconds=0.01,
    )
    await engine.set_acceptance_count(5)
    await engine.propose(route("retry me"))
    await engine.wait_idle()

    assert calls == 2
    assert engine.state.plan.route.steps[0].title == "retry me"
    assert engine.state.proposals[0].failure_count == 1
    assert any("retry_scheduled" in item.event for item in engine.state.audit)


@pytest.mark.asyncio
async def test_followup_supports_main_route_and_checked_partial_edits(tmp_path):
    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        original = proposal.route.steps[0]
        if calls == 1:
            return ReviewDecision(
                decision="followup",
                reasoning="sharpen then review again",
                main_route="Sharper overall route",
                edits=[
                    RouteEdit(
                        operation="check",
                        step_id=original.step_id,
                        expected_title=original.title,
                    ),
                    RouteEdit(
                        operation="update",
                        step_id=original.step_id,
                        step=__import__(
                            "backend.shared.solution_path", fromlist=["RouteStep"]
                        ).RouteStep(title="Sharpened step"),
                    ),
                    RouteEdit(
                        operation="add",
                        after_step_id=original.step_id,
                        step=__import__(
                            "backend.shared.solution_path", fromlist=["RouteStep"]
                        ).RouteStep(title="New step"),
                    ),
                ],
                more_edits=True,
            )
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-edits", reviewer)
    await engine.set_acceptance_count(5)
    await engine.propose(route("Original"), main_route="Original route")
    await engine.wait_idle()

    assert calls == 2
    assert engine.state.plan.main_route == "Sharper overall route"
    assert [step.title for step in engine.state.plan.route.steps] == [
        "Sharpened step",
        "New step",
    ]


@pytest.mark.asyncio
async def test_audit_and_terminal_proposal_history_are_bounded(tmp_path):
    async def reviewer(proposal, plan):
        return ReviewDecision(decision="reject")

    engine = SolutionPathEngine(tmp_path, "run-bounds", reviewer)
    await engine.set_acceptance_count(5)
    for index in range(270):
        await engine.propose(route(f"proposal-{index}"))
        await engine.wait_idle()

    assert len(engine.state.proposals) <= 100
    assert len(engine.state.audit) <= 500
    sequences = [record.sequence for record in engine.state.audit]
    assert sequences == sorted(sequences)
    assert sequences[-1] > len(sequences)


@pytest.mark.asyncio
async def test_followups_continue_past_old_hidden_cap(tmp_path):
    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        if calls <= 26:
            return ReviewDecision(
                decision="followup",
                reasoning="continue material revision",
                main_route=f"revision-{calls}",
            )
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-long-followup", reviewer)
    await engine.set_acceptance_count(5)
    await engine.propose(route("Long revision"))
    await engine.wait_idle()

    assert calls == 27
    assert engine.state.proposals[0].status.value == "approved"
    assert engine.state.plan.main_route == "revision-26"


@pytest.mark.asyncio
async def test_context_overflow_becomes_terminal_user_repair(tmp_path):
    async def reviewer(proposal, plan):
        raise ValueError(
            "solution-path reviewer mandatory context exceeds the configured input budget"
        )

    engine = SolutionPathEngine(tmp_path, "run-overflow", reviewer)
    await engine.set_acceptance_count(5)
    await engine.propose(route("Needs room"))
    await engine.wait_idle()

    proposal = engine.state.proposals[0]
    assert proposal.status.value == "user_repair_required"
    assert proposal.next_retry_at is None
    assert proposal.failure_count == 0
    assert proposal.repair_reason.value == "context_overflow"


@pytest.mark.asyncio
async def test_failed_later_edit_rolls_back_route_and_main_route(tmp_path):
    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        if calls < 3:
            original = proposal.route.steps[0]
            return ReviewDecision(
                decision="approve",
                main_route="must not leak",
                followup_route=SolutionRoute(
                    steps=[
                        RouteStep(
                            step_id=original.step_id,
                            title="must not leak either",
                        )
                    ]
                ),
                edits=[
                    RouteEdit(
                        operation="update",
                        step_id=proposal.route.steps[0].step_id,
                        step=RouteStep(title="partial mutation"),
                    ),
                    RouteEdit(
                        operation="delete",
                        step_id="missing-later-step",
                    ),
                ],
            )
        return ReviewDecision(decision="reject", reasoning="stop")

    engine = SolutionPathEngine(
        tmp_path,
        "run-transaction",
        reviewer,
        retry_base_seconds=0.01,
        retry_max_seconds=0.01,
    )
    await engine.set_acceptance_count(5)
    await engine.propose(route("original"), main_route="original route")
    await engine.wait_idle()

    proposal = engine.state.proposals[0]
    assert proposal.main_route == "original route"
    assert [step.title for step in proposal.route.steps] == ["original"]
    assert proposal.status.value == "rejected"


@pytest.mark.asyncio
async def test_hard_failure_requires_explicit_generation_fenced_resume(
    tmp_path, monkeypatch
):
    events = []
    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("No OpenRouter API key is available")
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-hard-repair", reviewer)

    async def capture(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(engine, "_broadcast", capture)
    await engine.set_acceptance_count(5)
    await engine.propose(route("repair me"))
    await engine.wait_idle()

    blocked = engine.state.proposals[0]
    assert calls == 1
    assert blocked.status.value == "user_repair_required"
    assert blocked.repair_reason.value == "missing_api_key"

    await engine.stop()
    await engine.start()
    await engine.wait_idle()
    assert calls == 1

    with pytest.raises(ValueError, match="stale"):
        await engine.resume_proposal(
            blocked.proposal_id,
            lifecycle_generation=engine.state.lifecycle_generation - 1,
        )
    await engine.resume_proposal(
        blocked.proposal_id,
        lifecycle_generation=engine.state.lifecycle_generation,
    )
    await engine.wait_idle()
    assert calls == 2
    assert engine.state.proposals[0].status.value == "approved"
    assert "solution_path_proposal_resumed" in [event for event, _ in events]


@pytest.mark.asyncio
async def test_stale_detached_proposal_cannot_resurrect_cleared_state(tmp_path):
    async def reviewer(proposal, plan):
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-clear-fence", reviewer)
    await engine.set_acceptance_count(5)
    captured_generation = engine.state.lifecycle_generation
    await engine.clear()

    with pytest.raises(ValueError, match="stale|stopped"):
        await engine.propose(
            route("stale"),
            lifecycle_generation=captured_generation,
        )
    assert not (tmp_path / "run-clear-fence" / "solution_path_state.json").exists()

    restored = SolutionPathEngine(tmp_path, "run-clear-fence", reviewer)
    assert restored.state.lifecycle_generation > captured_generation
    assert restored.state.proposals == []


@pytest.mark.asyncio
async def test_persist_failure_rolls_back_in_memory_mutation(tmp_path, monkeypatch):
    async def reviewer(proposal, plan):
        return ReviewDecision(decision="approve")

    engine = SolutionPathEngine(tmp_path, "run-persist-rollback", reviewer)
    original = engine.state

    def fail_write(payload):
        raise OSError("disk unavailable")

    monkeypatch.setattr(engine, "_write_payload", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        await engine.set_acceptance_count(5)

    assert engine.state == original


@pytest.mark.asyncio
async def test_unknown_failures_are_bounded_before_user_repair(tmp_path):
    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        raise RuntimeError("opaque reviewer failure")

    engine = SolutionPathEngine(
        tmp_path,
        "run-bounded-failure",
        reviewer,
        retry_base_seconds=0.01,
        retry_max_seconds=0.01,
    )
    await engine.set_acceptance_count(5)
    await engine.propose(route("bounded"))
    await engine.wait_idle()

    proposal = engine.state.proposals[0]
    assert calls == 3
    assert proposal.status.value == "user_repair_required"
    assert proposal.repair_reason.value == "unknown_reviewer_failure"


def test_route_structure_and_metadata_are_bounded():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SolutionRoute(steps=[RouteStep(title=str(index)) for index in range(33)])
    with pytest.raises(ValidationError):
        RouteStep(title="x", metadata={"payload": "y" * 9000})
