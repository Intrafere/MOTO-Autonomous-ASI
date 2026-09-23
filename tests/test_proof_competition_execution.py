import asyncio
from types import SimpleNamespace

import pytest

from backend.autonomous.core.proof_competition import ProofCompetition
from backend.shared.models import ProofCandidate
from backend.shared.api_client_manager import RetryableProviderError


def outcome(candidate, success=False):
    return SimpleNamespace(candidate=candidate, success=success, theorem_name="t",
                           lean_code="code" if success else "", attempts=[], context_overflow_payload={})


def driver(**kwargs):
    return ProofCompetition(config={"enabled": True, "secondaries": [{"model_id": "secondary"}]},
                            scope="autonomous", identity={"run_id": "r"}, batch_size=1, **kwargs)


@pytest.mark.asyncio
async def test_order_isolation_and_cached_success():
    candidate = ProofCandidate(theorem_id="a", statement="A")
    seen = []
    competition = driver()
    async def execute(index, route, state, save):
        seen.append((index, state["attempts"]))
        await save({"attempts": [{"private": index}]})
        return outcome(candidate, bool(index))
    classify = lambda value, index: "success" if value.success else "exhausted"
    restore = lambda record: outcome(candidate, record["outcome"]["success"])
    result = await competition.run_candidate(candidate, execute=execute, classify=classify, restore=restore)
    assert result.success and seen == [(0, []), (1, [])]
    resumed = driver(state=competition.state)
    assert (await resumed.run_candidate(candidate, execute=execute, classify=classify, restore=restore)).success
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_primary_capacity_released_while_secondary_waits():
    entered, release = asyncio.Event(), asyncio.Event()
    competition = driver()
    finished = []
    async def execute(index, route, state, save):
        candidate = ProofCandidate(theorem_id=state["theorem_id"], statement="A")
        if index:
            entered.set()
            await release.wait()
        return outcome(candidate, bool(index))
    async def run(name):
        candidate = ProofCandidate(theorem_id=name, statement="A")
        return await competition.run_candidate(candidate, execute=execute,
            classify=lambda result, index: "success" if result.success else "exhausted",
            restore=lambda record: None, primary_finished=lambda: finished.append(name))
    first = asyncio.create_task(run("a"))
    await entered.wait()
    second = asyncio.create_task(run("b"))
    await asyncio.sleep(0)
    assert finished == ["a", "b"]
    release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), RuntimeError("LEAN workspace broken"), OSError("disk failure"), PermissionError("permission denied")])
async def test_secondary_does_not_swallow_infrastructure_or_cancel(error):
    competition = driver()
    candidate = ProofCandidate(theorem_id="a", statement="A")
    async def execute(index, route, state, save):
        if index:
            raise error
        return outcome(candidate)
    with pytest.raises(type(error)):
        await competition.run_candidate(candidate, execute=execute,
            classify=lambda value, index: "exhausted", restore=lambda record: None)


@pytest.mark.asyncio
async def test_secondary_provider_failure_is_not_attempt_loss():
    competition = driver()
    candidate = ProofCandidate(theorem_id="a", statement="A")
    async def execute(index, route, state, save):
        if index:
            raise RetryableProviderError(provider="test", provider_label="Test", role_id="s",
                model="s", reason="unavailable", message="not ready")
        return outcome(candidate)
    result = await competition.run_candidate(candidate, execute=execute,
        classify=lambda value, index: "exhausted", restore=lambda record: None)
    assert not result.success
    secondary = next(value for value in competition.state.values() if value["competitor_index"] == 1)
    assert secondary["status"] == "unavailable" and secondary["attempts"] == []


@pytest.mark.asyncio
async def test_secondary_full_cohort_drains_before_next_admission():
    competition = ProofCompetition(config={"enabled": True, "secondaries": [{"model_id": "s"}]},
        scope="autonomous", identity={}, batch_size=2)
    releases = {name: asyncio.Event() for name in "abc"}
    entered = {name: asyncio.Event() for name in "abc"}
    primary_finished = []

    async def execute(index, route, state, save):
        name = state["theorem_id"]
        if index:
            entered[name].set()
            await releases[name].wait()
        return outcome(ProofCandidate(theorem_id=name, statement=name), bool(index))

    async def run(name):
        return await competition.run_candidate(ProofCandidate(theorem_id=name, statement=name),
            execute=execute, classify=lambda value, index: "success" if value.success else "exhausted",
            restore=lambda record: None, primary_finished=lambda: primary_finished.append(name))

    tasks = [asyncio.create_task(run(name)) for name in "abc"]
    try:
        await asyncio.wait_for(entered["b"].wait(), 1)
        releases["a"].set()
        await asyncio.wait_for(tasks[0], 1)
        assert primary_finished == list("abc")
        assert not entered["c"].is_set(), "Completed slots must not refill inside a cohort"
        releases["b"].set()
        await asyncio.wait_for(entered["c"].wait(), 1)
        releases["c"].set()
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_prework_is_queued_until_explicit_model_attempt_start():
    events = []
    async def observe(event):
        events.append(event)
    competition = driver(observer=observe)
    candidate = ProofCandidate(theorem_id="a", statement="A")
    async def execute(index, route, state, save):
        assert events[-1]["status"] == "eligible"
        assert not events[-1].get("attempt_started")
        await save({"status": "running", "attempt_started": True})
        return outcome(candidate, True)
    await competition.run_candidate(candidate, execute=execute,
        classify=lambda value, index: "success", restore=lambda record: None)
    assert [event["status"] for event in events] == ["eligible", "running", "success"]
    assert events[-1]["attempt_started"] is True


@pytest.mark.asyncio
async def test_success_survives_primary_route_revision_change():
    candidate = ProofCandidate(theorem_id="a", statement="A")
    competition = driver()
    async def execute(index, route, state, save):
        return outcome(candidate, True)
    classify = lambda value, index: "success"
    restore = lambda record: outcome(candidate, True)
    await competition.run_candidate(candidate, execute=execute, classify=classify, restore=restore)
    resumed = ProofCompetition(config={"enabled": True, "secondaries": [{"model_id": "changed"}]},
        scope="autonomous", identity={"run_id": "r", "primary_route": "changed"},
        batch_size=1, state=competition.state)
    async def forbidden(*args):
        raise AssertionError("Accepted Lean artifact must not be regenerated")
    assert (await resumed.run_candidate(candidate, execute=forbidden, classify=classify, restore=restore)).success


@pytest.mark.asyncio
async def test_callback_failure_never_becomes_secondary_unavailability():
    candidate = ProofCandidate(theorem_id="a", statement="A")
    competition = driver()
    async def execute(index, route, state, save):
        if index:
            async def broken_checkpoint():
                raise RetryableProviderError(provider="test", provider_label="Test", role_id="s",
                    model="s", reason="unavailable", message="not ready")
            competition.checkpoint = broken_checkpoint
            await save({"attempt_started": True})
        return outcome(candidate)
    with pytest.raises(RetryableProviderError):
        await competition.run_candidate(candidate, execute=execute,
            classify=lambda value, index: "exhausted", restore=lambda record: None)
    assert not competition.unavailable


def test_scope_gate():
    competition = ProofCompetition(config={"enabled": True, "secondaries": [{"model_id": "s"}]},
                                   scope="manual", identity={}, batch_size=1)
    assert not competition.enabled
