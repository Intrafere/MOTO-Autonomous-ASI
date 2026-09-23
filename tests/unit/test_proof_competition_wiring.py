from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.shared.models import ProofCompetitionConfig, ProofRoleRuntimeConfig, ProofRuntimeConfigSnapshot, ProofCandidate, ProofAttemptFeedback


def config():
    return ProofCompetitionConfig(enabled=True, secondaries=[ProofRoleRuntimeConfig(
        provider="openrouter", model_id="secondary", context_window=12000, max_output_tokens=2000,
    )])


def test_legacy_snapshot_defaults_off_and_explicit_budgets_required():
    snapshot = ProofRuntimeConfigSnapshot(brainstorm={}, paper={}, validator={})
    assert not snapshot.proof_competition.enabled
    with pytest.raises(ValidationError):
        ProofRoleRuntimeConfig(provider="openrouter", model_id="secondary")
    with pytest.raises(ValidationError):
        ProofRoleRuntimeConfig(provider="openrouter", model_id="secondary", context_window=10, max_output_tokens=10)
    assert ProofRuntimeConfigSnapshot.model_validate(snapshot.model_dump()).proof_competition.secondaries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_success,enabled,expected", [(True, True, ["primary"]), (False, False, ["primary"]), (False, True, ["primary", "secondary"])])
async def test_compiler_fallback_is_five_full_attempts_and_isolated(monkeypatch, primary_success, enabled, expected):
    import backend.compiler.agents.high_param_submitter as module
    import backend.autonomous.agents.proof_formalization_agent as formalization
    calls = []
    routes = []

    class Formalizer:
        def __init__(self, **kwargs):
            self.model = kwargs["model_id"]

        async def prove_candidate(self, **kwargs):
            calls.append(self.model)
            assert kwargs["max_attempts"] == 5
            assert not kwargs.get("prior_attempts")
            assert kwargs["theorem_candidate"].formal_sketch == "original"
            kwargs["theorem_candidate"].formal_sketch = "private mutation"
            success = primary_success or self.model == "secondary"
            feedback = [ProofAttemptFeedback(attempt=i, theorem_id="t", success=success, error_output="" if success else "Lean failed") for i in range(1, 6)]
            return success, "proved", "theorem proved : True := by trivial", feedback

    monkeypatch.setattr(formalization, "ProofFormalizationAgent", Formalizer)
    monkeypatch.setattr(module.api_client_manager, "configure_role", lambda role, route: routes.append((role, route)))
    monkeypatch.setattr(module.paper_memory, "get_paper", AsyncMock(return_value="complete source"))
    monkeypatch.setattr(module, "validate_full_lean_proof_integrity", AsyncMock(return_value=SimpleNamespace(valid=True)))
    monkeypatch.setattr(module.system_config, "compiler_high_param_context_window", 12000)
    monkeypatch.setattr(module.system_config, "compiler_high_param_max_output_tokens", 2000)
    competition = config()
    competition.enabled = enabled
    submitter = module.HighParamSubmitter("primary", "goal", proof_competition=competition)
    submitter.proof_database = SimpleNamespace(record_failed_candidate=AsyncMock())
    submitter._broadcast = AsyncMock()
    candidate = ProofCandidate(theorem_id="t", statement="statement", formal_sketch="original")
    await submitter._step_formalize(candidate, "statement")
    assert calls == expected
    assert bool(routes) == (len(expected) == 2)


@pytest.mark.asyncio
async def test_history_snapshot_uses_original_session_not_active(monkeypatch, tmp_path):
    import json
    from backend.api.routes import proofs
    from backend.shared.models import ProofCheckRequest
    saved = ProofRuntimeConfigSnapshot(brainstorm={}, paper={}, validator={}, proof_competition=config())
    (tmp_path / "session_metadata.json").write_text(json.dumps({"proof_runtime_config": saved.model_dump(mode="json")}), encoding="utf-8")
    monkeypatch.setattr(proofs, "_history_session_dir", lambda session_id: tmp_path)
    monkeypatch.setattr(proofs.autonomous_coordinator, "get_proof_runtime_config", lambda: pytest.fail("active session borrowed"))
    snapshot = await proofs._get_runtime_snapshot(ProofCheckRequest(source_type="paper", source_id="original:paper_1"))
    assert snapshot.proof_competition.enabled
    assert snapshot.proof_competition.secondaries[0].model_id == "secondary"
