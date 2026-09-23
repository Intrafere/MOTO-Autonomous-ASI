"""Real formalizer contract handling and route-local checkpoint regressions."""
import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest

from backend.autonomous.agents.proof_formalization_agent import ProofFormalizationAgent
from backend.autonomous.core.proof_competition import ProofCompetition, fingerprint
from backend.shared.lean4_client import Lean4Result
from backend.shared.models import ProofCandidate
from types import SimpleNamespace


MODULE = "backend.autonomous.agents.proof_formalization_agent"


def response(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["{}", '{"theorem_name":"t"}', '{"lean_code":""}',
    '{"lean_code":"  "}', '{"lean_code":null}', '{"lean_code":42}'])
@pytest.mark.parametrize("recover", [False, True])
async def test_strict_real_agent_recovers_invalid_lean_contract(content, recover):
    agent = ProofFormalizationAgent("model", 8000, 1000, "secondary", strict_execution_errors=True)
    candidate = ProofCandidate(theorem_id="t", statement="True")
    valid = response('{"lean_code":"import Mathlib\\ntheorem t : True := by trivial"}')
    generate = AsyncMock(side_effect=[response(content), valid] if recover else [response(content)] * 3)
    lean = AsyncMock()
    lean.check_proof.return_value = Lean4Result(success=True)
    callback = AsyncMock()
    with (patch(f"{MODULE}.api_client_manager.generate_completion", generate),
          patch(f"{MODULE}.get_lean4_client", return_value=lean),
          patch(f"{MODULE}._latest_assistant_pack_for_lean_attempts", return_value=("", "", []))):
        success, _, _, attempts = await agent.prove_candidate(
            "Prove True", "paper", candidate, "Complete source", max_attempts=3,
            attempt_callback=callback)
    assert success is recover
    assert generate.await_count == (2 if recover else 3)
    assert len(attempts) == 1
    assert attempts[0].attempt == 1
    if recover:
        lean.check_proof.assert_awaited_once()
    else:
        lean.check_proof.assert_not_awaited()
        assert attempts[0].failure_kind == "malformed_output"
        assert not attempts[0].lean_was_run
    callback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [OSError("disk failure"), RuntimeError("Lean infrastructure failure"),
    ValueError("unrelated infrastructure value error"), asyncio.CancelledError()])
async def test_strict_real_agent_preserves_lean_failure_authority(error):
    agent = ProofFormalizationAgent("model", 8000, 1000, "secondary", strict_execution_errors=True)
    lean = AsyncMock()
    lean.check_proof.side_effect = error
    with (patch(f"{MODULE}.api_client_manager.generate_completion", AsyncMock(return_value=response(
              '{"lean_code":"theorem t : True := by trivial"}'))),
          patch(f"{MODULE}.get_lean4_client", return_value=lean)):
        with pytest.raises(type(error)):
            await agent._run_full_script_attempt(
                user_research_prompt="Prove True", source_type="paper",
                theorem_candidate=ProofCandidate(theorem_id="t", statement="True"),
                prior_attempts=[], source_excerpt="True", source_content="Complete source", attempt_number=1)


def failed_outcome(candidate):
    return SimpleNamespace(candidate=candidate, success=False, theorem_name="", lean_code="",
                           attempts=[], context_overflow_payload={})


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
async def test_primary_change_preserves_secondary_private_attempts(legacy, exhausted):
    candidate = ProofCandidate(theorem_id="t", statement="True")
    config = {"enabled": True, "secondaries": [{"model_id": "secondary"}]}
    old_identity = {"run_id": "r", "source": "s", "primary_route": {"model_id": "old"}}
    initial = ProofCompetition(config=config, scope="autonomous", identity=old_identity, batch_size=1)
    private_attempts = [{"attempt": i, "lean_code": f"private-{i}"} for i in range(1, 6 if exhausted else 3)]

    async def initial_execute(index, route, state, save):
        if index:
            await save({"attempts": private_attempts, "attempt_started": True, "status": "running"})
            if not exhausted:
                raise asyncio.CancelledError()
        return failed_outcome(candidate)

    kwargs = {"classify": lambda value, index: "exhausted", "restore": lambda record: failed_outcome(candidate)}
    if exhausted:
        await initial.run_candidate(candidate, execute=initial_execute, **kwargs)
    else:
        with pytest.raises(asyncio.CancelledError):
            await initial.run_candidate(candidate, execute=initial_execute, **kwargs)
    state = deepcopy(initial.state)
    secondary_key = next(key for key, value in state.items() if value["competitor_index"] == 1)
    if legacy:
        record = state.pop(secondary_key)
        old_key = fingerprint([old_identity, candidate.theorem_id, candidate.statement,
                               candidate.formal_sketch, 1, record["route_revision"]])
        state[old_key] = record
    resumed = ProofCompetition(config=config, scope="autonomous", batch_size=1, state=state,
        identity={**old_identity, "primary_route": {"model_id": "new"}})
    seen = []
    restored = []

    async def execute(index, route, record, save):
        seen.append(index)
        if index:
            assert record["attempts"] == private_attempts
            assert record["attempt_started"] is True
        else:
            assert record["attempts"] == []
        return failed_outcome(candidate)

    def restore(record):
        restored.append(record)
        return failed_outcome(candidate)

    await resumed.run_candidate(candidate, execute=execute, classify=kwargs["classify"], restore=restore)
    assert seen == ([0] if exhausted else [0, 1])
    if exhausted:
        assert restored[0]["attempts"] == private_attempts
    else:
        assert resumed.state[secondary_key]["attempts"] == private_attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["secondary", "scope", "candidate"])
async def test_secondary_history_does_not_cross_execution_identity(change):
    config = {"enabled": True, "secondaries": [{"model_id": "secondary"}]}
    candidate = ProofCandidate(theorem_id="t", statement="True")
    identity = {"run_id": "r", "primary_route": "primary"}
    initial = ProofCompetition(config=config, scope="autonomous", identity=identity, batch_size=1)

    async def execute(index, route, record, save):
        await save({"attempts": [{"private": index}]})
        return failed_outcome(candidate)

    kwargs = {"classify": lambda value, index: "exhausted", "restore": lambda record: failed_outcome(candidate)}
    await initial.run_candidate(candidate, execute=execute, **kwargs)
    if change == "secondary":
        config["secondaries"][0]["model_id"] = "different"
    elif change == "scope":
        identity["run_id"] = "different"
    else:
        candidate = ProofCandidate(theorem_id="t", statement="False")
    resumed = ProofCompetition(config=config, scope="autonomous", identity=identity, batch_size=1,
                               state=initial.state)
    seen = []

    async def fresh(index, route, record, save):
        if index:
            assert record["attempts"] == []
            seen.append(index)
        return failed_outcome(candidate)

    await resumed.run_candidate(candidate, execute=fresh, **kwargs)
    assert seen == [1]
