from types import SimpleNamespace

import pytest

from backend.autonomous.core import proof_verification_stage as stage_module
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.shared.api_client_manager import RetryableProviderError
from backend.shared.config import system_config
from backend.shared.models import ProofAttemptFeedback, ProofCandidate, SmtHint


def _candidate() -> ProofCandidate:
    return ProofCandidate(
        theorem_id="t1",
        statement="For n : Nat, n + 0 = n",
        formal_sketch="Arithmetic equality over Nat",
    )


@pytest.mark.asyncio
async def test_disabled_smt_makes_no_translation_or_solver_call(monkeypatch) -> None:
    stage = ProofVerificationStage()
    agent = SimpleNamespace()

    async def forbidden_translate(**_kwargs):
        raise AssertionError("translation must not run while SMT is disabled")

    agent.translate_candidate_to_smt = forbidden_translate
    previous = system_config.smt_enabled
    system_config.smt_enabled = False
    try:
        hint = await stage._run_smt_check(
            user_prompt="Prove it",
            source_type="brainstorm",
            source_id="source",
            base_event={},
            candidate=_candidate(),
            proof_label="Proof A",
            source_content="source",
            source_title="title",
            identification_agent=agent,
            broadcast_fn=None,
        )
    finally:
        system_config.smt_enabled = previous

    assert hint is None


@pytest.mark.asyncio
@pytest.mark.parametrize("solver_result", ("unsat", "sat", "unknown", "invalid"))
async def test_only_unsat_suggests_lean_tactics(monkeypatch, solver_result) -> None:
    stage = ProofVerificationStage()

    async def translate(**_kwargs):
        return "(assert false)"

    async def check_smt2(_smtlib, timeout):
        return SimpleNamespace(
            result=solver_result, stdout=solver_result, stderr="", success=True
        )

    monkeypatch.setattr(
        stage_module, "get_smt_client", lambda: SimpleNamespace(check_smt2=check_smt2)
    )
    previous = system_config.smt_enabled
    system_config.smt_enabled = True
    try:
        hint = await stage._run_smt_check(
            user_prompt="Prove it",
            source_type="brainstorm",
            source_id="source",
            base_event={},
            candidate=_candidate(),
            proof_label="Proof A",
            source_content="source",
            source_title="title",
            identification_agent=SimpleNamespace(
                translate_candidate_to_smt=translate
            ),
            broadcast_fn=None,
        )
    finally:
        system_config.smt_enabled = previous

    assert hint is not None
    assert bool(hint.suggested_tactics) is (solver_result == "unsat")
    assert hint.result == (solver_result if solver_result != "invalid" else "unknown")


def test_smt_assistance_metadata_requires_successful_first_attempt_tactic_use() -> None:
    hint = SmtHint(result="unsat", suggested_tactics=["omega"])
    rejected = ProofAttemptFeedback(
        attempt=1,
        theorem_id="t1",
        lean_code="by omega",
        success=False,
    )
    accepted_without_hint = ProofAttemptFeedback(
        attempt=1,
        theorem_id="t1",
        lean_code="by trivial",
        success=True,
    )
    accepted_with_hint = ProofAttemptFeedback(
        attempt=1,
        theorem_id="t1",
        lean_code="by omega",
        success=True,
    )

    assert not ProofVerificationStage._first_attempt_used_smt_hint([rejected], hint)
    assert not ProofVerificationStage._first_attempt_used_smt_hint(
        [accepted_without_hint], hint
    )
    assert ProofVerificationStage._first_attempt_used_smt_hint(
        [accepted_with_hint], hint
    )
    assert not ProofVerificationStage._first_attempt_used_smt_hint(
        [accepted_with_hint],
        SmtHint(result="sat", suggested_tactics=["omega"]),
    )


@pytest.mark.asyncio
async def test_transient_smt_translation_failure_propagates_to_checkpoint_owner(
    monkeypatch,
) -> None:
    stage = ProofVerificationStage()
    monkeypatch.setattr(
        ProofVerificationStage,
        "_is_smt_amenable",
        staticmethod(lambda _candidate: True),
    )
    monkeypatch.setattr(
        stage_module.system_config,
        "smt_enabled",
        True,
    )

    async def fail_translation(**_kwargs):
        raise RetryableProviderError(
            provider="openrouter",
            provider_label="OpenRouter",
            role_id="proof",
            model="test-model",
            reason="temporary_disconnect",
            message="temporary provider disconnect",
        )

    assert stage_module.system_config.smt_enabled is True
    assert stage._is_smt_amenable(_candidate()) is True
    with pytest.raises(RetryableProviderError):
        await stage._run_smt_check(
            user_prompt="Prove it",
            source_type="brainstorm",
            source_id="source",
            base_event={},
            candidate=ProofCandidate(
                theorem_id="t2",
                statement="n : Nat satisfies n + 0 = n",
                formal_sketch="Arithmetic equality",
            ),
            proof_label="Proof A",
            source_content="source",
            source_title="title",
            identification_agent=SimpleNamespace(
                translate_candidate_to_smt=fail_translation
            ),
            broadcast_fn=None,
        )
