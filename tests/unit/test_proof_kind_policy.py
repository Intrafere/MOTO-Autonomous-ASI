"""Single-shot and master proof kinds chosen under Allowed Outputs."""
from unittest import mock

import pytest
from fastapi import HTTPException

from backend.api.routes import proofs as proofs_routes
from backend.autonomous.prompts.proof_prompts import (
    build_proof_candidate_list_validation_prompt,
    build_proof_identification_prompt,
)
from backend.shared import runtime_settings
from backend.shared.config import system_config
from backend.shared.models import ProofCandidate, ProofSettingsUpdateRequest
from backend.shared.proof_kind_policy import ProofKindPolicy

MASTER_OPTION = "MASTER PROOF OPTION"
MASTER_ONLY = 'every selected target is proved as a master proof'
INDIVIDUAL_BUDGET = "each selected target receives 5 Lean 4 attempts"
MASTER_REVIEW = 'An entry whose proposal_kind is "master_proof"'


@pytest.fixture
def restore_proof_kinds():
    saved = (system_config.proof_single_shot_enabled, system_config.proof_master_proofs_enabled)
    yield
    system_config.proof_single_shot_enabled, system_config.proof_master_proofs_enabled = saved


def _identification_prompt(**kinds):
    return build_proof_identification_prompt(
        user_prompt="Prove the conjecture.",
        source_type="brainstorm",
        source_id="b1",
        source_content="source",
        **kinds,
    )


def _candidate(theorem_id, kind):
    return ProofCandidate(theorem_id=theorem_id, statement=f"claim {theorem_id}", proposal_kind=kind)


def test_both_kinds_offer_single_shot_budget_and_master_option():
    prompt = _identification_prompt()
    assert INDIVIDUAL_BUDGET in prompt
    assert MASTER_OPTION in prompt
    assert MASTER_ONLY not in prompt
    assert '"proposal_kind": "individual"' in prompt


def test_single_shot_only_removes_every_master_proof_mention():
    prompt = _identification_prompt(allow_master_proofs=False)
    assert INDIVIDUAL_BUDGET in prompt
    assert "master" not in prompt.lower()
    assert "proposal_kind" not in prompt


def test_master_only_removes_single_shot_budget():
    prompt = _identification_prompt(allow_single_shot_proofs=False)
    assert MASTER_ONLY in prompt
    assert INDIVIDUAL_BUDGET not in prompt
    assert MASTER_OPTION not in prompt
    assert '"proposal_kind": "master_proof"' in prompt
    assert '"individual"' not in prompt


def test_identification_prompt_requires_an_allowed_kind():
    with pytest.raises(ValueError):
        _identification_prompt(allow_single_shot_proofs=False, allow_master_proofs=False)


def test_validation_prompt_drops_master_review_guidance_when_master_proofs_are_off():
    kwargs = dict(user_prompt="p", source_type="brainstorm", source_id="b1", source_title="", candidates_json="[]")
    assert MASTER_REVIEW in build_proof_candidate_list_validation_prompt(**kwargs)
    without = build_proof_candidate_list_validation_prompt(**kwargs, allow_master_proofs=False)
    assert "master" not in without.lower()


def test_policy_routes_candidates_to_allowed_kinds():
    candidates = [_candidate("a", "individual"), _candidate("b", "master_proof")]
    assert [c.proposal_kind for c in ProofKindPolicy().normalize(candidates)] == ["individual", "master_proof"]
    single = ProofKindPolicy(single_shot=True, master=False).normalize(candidates)
    assert [c.proposal_kind for c in single] == ["individual", "individual"]
    master = ProofKindPolicy(single_shot=False, master=True).normalize(candidates)
    assert [c.proposal_kind for c in master] == ["master_proof", "master_proof"]
    # Inputs are never mutated in place.
    assert candidates[1].proposal_kind == "master_proof"


def test_policy_rejects_no_kinds_and_config_falls_back_to_both(restore_proof_kinds):
    with pytest.raises(ValueError):
        ProofKindPolicy(single_shot=False, master=False)
    system_config.proof_single_shot_enabled = False
    system_config.proof_master_proofs_enabled = False
    assert ProofKindPolicy.from_config() == ProofKindPolicy()


@pytest.mark.asyncio
async def test_settings_route_updates_kinds_and_rejects_turning_both_off(monkeypatch, restore_proof_kinds):
    monkeypatch.setattr(system_config, "generic_mode", False)
    monkeypatch.setattr(proofs_routes, "save_proof_runtime_settings", lambda: None)
    monkeypatch.setattr(proofs_routes, "reconcile_lean4_runtime_state", lambda: None)

    async def fake_status():
        return {
            "single": system_config.proof_single_shot_enabled,
            "master": system_config.proof_master_proofs_enabled,
        }

    monkeypatch.setattr(proofs_routes, "get_proofs_status", fake_status)
    monkeypatch.setattr(system_config, "lean4_enabled", False)
    # The route writes these on every save; restore them after the test.
    monkeypatch.setattr(system_config, "lean4_user_enabled", system_config.lean4_user_enabled)
    monkeypatch.setattr(system_config, "lean4_proof_timeout", system_config.lean4_proof_timeout)
    system_config.proof_single_shot_enabled = True
    system_config.proof_master_proofs_enabled = True

    status = await proofs_routes.update_proof_settings(
        ProofSettingsUpdateRequest(enabled=False, master_proofs_enabled=False)
    )
    assert status == {"single": True, "master": False}

    with pytest.raises(HTTPException) as excinfo:
        await proofs_routes.update_proof_settings(
            ProofSettingsUpdateRequest(enabled=False, single_shot_proofs_enabled=False)
        )
    assert excinfo.value.status_code == 400
    assert system_config.proof_single_shot_enabled is True
    assert system_config.proof_master_proofs_enabled is False

    # Omitted fields keep the current choice.
    status = await proofs_routes.update_proof_settings(ProofSettingsUpdateRequest(enabled=False))
    assert status == {"single": True, "master": False}


def test_proof_kinds_persist_and_a_file_cannot_turn_both_off(restore_proof_kinds, monkeypatch):
    monkeypatch.setattr(system_config, "generic_mode", False)
    monkeypatch.setattr(system_config, "lean4_enabled", system_config.lean4_enabled)
    system_config.proof_single_shot_enabled = False
    system_config.proof_master_proofs_enabled = True
    saved = runtime_settings._proof_settings_from_config()
    assert saved["proof_single_shot_enabled"] is False
    assert saved["proof_master_proofs_enabled"] is True

    payload = {"proof_settings": {"proof_single_shot_enabled": False, "proof_master_proofs_enabled": False}}
    with mock.patch.object(runtime_settings, "_read_settings", return_value=payload):
        runtime_settings.apply_persisted_runtime_settings()
    assert system_config.proof_single_shot_enabled is True
    assert system_config.proof_master_proofs_enabled is True


@pytest.mark.asyncio
async def test_identification_agent_offers_and_returns_only_allowed_kinds(restore_proof_kinds):
    from backend.autonomous.agents.proof_identification_agent import ProofIdentificationAgent

    agent = ProofIdentificationAgent(
        model_id="m", context_window=100000, max_output_tokens=1000, role_id="r"
    )
    system_config.proof_single_shot_enabled = True
    system_config.proof_master_proofs_enabled = False
    seen = {}

    async def fake_identify(**kwargs):
        seen["kinds"] = kwargs["proof_kinds"]
        return True, [_candidate("a", "master_proof")]

    with mock.patch.object(agent, "_identify_candidates", side_effect=fake_identify):
        has_candidates, candidates = await agent.identify_candidates(
            user_research_prompt="p", source_type="brainstorm", source_id="b1", source_content="s"
        )

    assert seen["kinds"] == ProofKindPolicy(single_shot=True, master=False)
    assert has_candidates is True
    assert [c.proposal_kind for c in candidates] == ["individual"]
