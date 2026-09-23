import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.shared.models import ProofCandidate, ProofAttemptFeedback
from tests.unit.test_proof_competition_wiring import config


@pytest.mark.asyncio
async def test_secondary_resume_keeps_private_budget_and_accepted_artifact(monkeypatch, tmp_path):
    import backend.compiler.agents.high_param_submitter as module
    import backend.autonomous.agents.proof_formalization_agent as formalization
    from backend.compiler.memory.proof_competition_state import CompilerCompetitionState

    monkeypatch.setattr(module.paper_memory, "get_paper", AsyncMock(return_value="full source"))
    monkeypatch.setattr(module, "validate_full_lean_proof_integrity", AsyncMock(return_value=SimpleNamespace(valid=True)))
    monkeypatch.setattr(module.system_config, "compiler_high_param_context_window", 12000)
    monkeypatch.setattr(module.system_config, "compiler_high_param_max_output_tokens", 2000)
    monkeypatch.setattr(module.api_client_manager, "configure_role", lambda *args: None)
    calls = []
    interrupt = True

    class Formalizer:
        def __init__(self, **kwargs):
            self.model = kwargs["model_id"]

        async def prove_candidate(self, **kwargs):
            nonlocal interrupt
            prior = kwargs.get("prior_attempts", [])
            calls.append((self.model, kwargs["max_attempts"], len(prior)))
            assert all(item.error_output == self.model for item in prior)
            feedback = list(prior)
            for number in range(len(prior) + 1, 6):
                await kwargs["attempt_start_callback"](number, "full_script")
                item = ProofAttemptFeedback(attempt=number, theorem_id="t", success=self.model == "secondary" and number == 2,
                    lean_code="theorem won : True := by trivial", error_output=self.model)
                feedback.append(item)
                await kwargs["attempt_callback"](item)
                if self.model == "secondary" and interrupt:
                    interrupt = False
                    raise asyncio.CancelledError()
                if item.success:
                    return True, "won", item.lean_code, feedback
            return False, "", "", feedback

    monkeypatch.setattr(formalization, "ProofFormalizationAgent", Formalizer)

    async def submitter():
        instance = module.HighParamSubmitter("primary", "goal", proof_competition=config())
        instance.proof_database = SimpleNamespace(record_failed_candidate=AsyncMock())
        instance._broadcast = AsyncMock()
        instance._competition_state = CompilerCompetitionState(tmp_path, "run", "paper")
        await instance._competition_state.load()
        return instance

    candidate = ProofCandidate(theorem_id="t", statement="statement", formal_sketch="original")
    first = await submitter()
    with pytest.raises(asyncio.CancelledError):
        await first._step_formalize(candidate, candidate.statement)
    second = await submitter()
    result = await second._step_formalize(candidate, candidate.statement)
    assert result[1] == "theorem won : True := by trivial"
    assert calls == [("primary", 5, 0), ("secondary", 5, 0), ("secondary", 4, 1)]
    third = await submitter()
    recovered = await third._step_formalize(candidate, candidate.statement)
    assert recovered[1] == result[1]
    assert len(calls) == 3
    assert [entry.attempt for entry in recovered[2]] == [1, 2]
    # Provider edits cannot erase an already Lean-accepted artifact.
    third.proof_competition.secondaries = []
    third.model_name = "changed primary"
    recovered_again = await third._step_formalize(candidate, candidate.statement)
    assert recovered_again[1] == result[1]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_compiler_state_scopes_run_source_and_commits_atomically(tmp_path):
    from backend.compiler.memory.proof_competition_state import CompilerCompetitionState
    state = CompilerCompetitionState(tmp_path, "run", "paper")
    state.data = {"competitors": {"1:route": {"attempts": []}}}
    await state.save()
    same = CompilerCompetitionState(tmp_path, "run", "paper")
    assert await same.load() == state.data
    assert await CompilerCompetitionState(tmp_path, "other", "paper").load() == {}
    assert await CompilerCompetitionState(tmp_path, "run", "other").load() == {}
    assert not list(state.path.parent.glob("*.tmp"))
