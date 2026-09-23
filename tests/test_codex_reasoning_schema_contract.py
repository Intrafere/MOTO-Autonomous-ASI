"""Provider-aware reasoning contracts without provider access or runtime stores."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.shared import models


ROLE_MODELS = [
    models.ModelConfig, models.SubmitterConfig, models.ProofRoleConfigSnapshot,
    models.ProofRoleRuntimeConfig, models.LeanOJRoleConfig,
]
OTHER_PROVIDERS = ["lm_studio", "openrouter", "xai_grok_oauth", "sakana_fugu"]
LEGACY_EFFORTS = ["auto", "xhigh", "high", "medium", "low", "minimal", "none"]


def role(provider="openai_codex_oauth", effort="max"):
    return dict(provider=provider, model_id="model", openrouter_reasoning_effort=effort,
                context_window=10000, max_output_tokens=1000)


@pytest.mark.parametrize("model", ROLE_MODELS)
@pytest.mark.parametrize("provider", ["openai_codex_oauth", *OTHER_PROVIDERS])
def test_role_max_is_codex_only(model, provider):
    data = role(provider)
    if model is models.SubmitterConfig:
        data["submitter_id"] = 1
    if provider == "openai_codex_oauth":
        instance = model.model_validate(data)
        assert instance.openrouter_reasoning_effort == "max"
        assert model.model_validate(instance.model_dump()).openrouter_reasoning_effort == "max"
    else:
        with pytest.raises(ValidationError, match="only by openai_codex_oauth"):
            model.model_validate(data)


STARTS = [
    (models.AggregatorStartRequest, dict(user_prompt="goal", submitter_configs=[], validator_model="v")),
    (models.CompilerStartRequest, dict(compiler_prompt="goal", validator_model="v", writer_model="w", high_param_model="p")),
    (models.AutonomousResearchStartRequest, dict(user_research_prompt="goal", submitter_configs=[], validator_model="v")),
    (models.CritiqueRequest, {}),
]
FLAT_ROLES = [(model, data, name) for model, data in STARTS
              for name in model.model_fields if name.endswith("openrouter_reasoning_effort")]


@pytest.mark.parametrize("model,data,field", FLAT_ROLES)
@pytest.mark.parametrize("provider", ["openai_codex_oauth", *OTHER_PROVIDERS])
def test_every_flat_start_role_checks_its_own_provider(model, data, field, provider):
    provider_field = field.removesuffix("openrouter_reasoning_effort") + "provider"
    payload = {**data, provider_field: provider, field: "max"}
    if provider == "openai_codex_oauth":
        assert getattr(model.model_validate(payload), field) == "max"
    else:
        with pytest.raises(ValidationError):
            model.model_validate(payload)


@pytest.mark.parametrize("model,data", STARTS[:3])
def test_nested_submitter_max_validation(model, data):
    if "submitter_configs" not in model.model_fields:
        return
    for provider in ["openai_codex_oauth", *OTHER_PROVIDERS]:
        payload = {**data, "submitter_configs": [{**role(provider), "submitter_id": 1}]}
        if provider == "openai_codex_oauth":
            assert model.model_validate(payload).submitter_configs[0].openrouter_reasoning_effort == "max"
        else:
            with pytest.raises(ValidationError):
                model.model_validate(payload)


@pytest.mark.parametrize("effort", LEGACY_EFFORTS)
def test_openrouter_and_boost_keep_every_legacy_value(effort):
    assert models.ModelConfig(**role("openrouter", effort)).openrouter_reasoning_effort == effort
    assert models.BoostConfig(boost_reasoning_effort=effort).boost_reasoning_effort == effort


@pytest.mark.parametrize("effort", ["maximum", "highest", "unsupported", "MAX"])
def test_schema_rejects_noncontract_values(effort):
    with pytest.raises(ValidationError):
        models.ModelConfig(**role(effort=effort))


def test_missing_provider_does_not_authorize_max():
    payload = role()
    payload.pop("provider")
    with pytest.raises(ValidationError):
        models.ModelConfig(**payload)


def test_boost_rejects_max_and_schema_retains_enum():
    with pytest.raises(ValidationError):
        models.BoostConfig(boost_reasoning_effort="max")
    assert models.BoostConfig.model_json_schema()["properties"]["boost_reasoning_effort"]["enum"] == LEGACY_EFFORTS
    assert "max" in models.ModelConfig.model_json_schema()["properties"]["openrouter_reasoning_effort"]["enum"]


@pytest.mark.parametrize("model,data", STARTS[1:3])
def test_legacy_writer_alias_resolves_provider_before_reasoning(model, data):
    payload = {**data, "high_context_provider": "openai_codex_oauth",
               "high_context_openrouter_reasoning_effort": "max"}
    result = model.model_validate(payload)
    assert result.writer_provider == "openai_codex_oauth"
    assert result.writer_openrouter_reasoning_effort == "max"
    payload["writer_provider"] = "openrouter"
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("slot", ["brainstorm", "paper", "validator", "assistant"])
def test_proof_snapshot_nested_roles(slot):
    payload = {name: role() for name in ["brainstorm", "paper", "validator", "assistant"]}
    assert getattr(models.ProofRuntimeConfigSnapshot(**payload), slot).openrouter_reasoning_effort == "max"
    payload[slot] = role("openrouter")
    with pytest.raises(ValidationError):
        models.ProofRuntimeConfigSnapshot(**payload)


@pytest.mark.parametrize("slot", ["topic_generator", "topic_validator", "brainstorm_validator", "path_decider", "final_solver", "assistant", "brainstorm_submitters"])
def test_leanoj_start_nested_roles(slot):
    payload = {name: role() for name in ["topic_generator", "topic_validator", "brainstorm_validator", "path_decider", "final_solver", "assistant"]}
    payload.update(user_prompt="goal", lean_template="template", brainstorm_submitters=[role()])
    models.LeanOJStartRequest(**payload)
    payload[slot] = [role("openrouter")] if slot == "brainstorm_submitters" else role("openrouter")
    with pytest.raises(ValidationError):
        models.LeanOJStartRequest(**payload)


def test_competitor_nested_snapshot_and_start():
    competition = dict(enabled=True, secondaries=[role()])
    snapshot = models.ProofRuntimeConfigSnapshot(brainstorm=role(), paper=role(), validator=role(), proof_competition=competition)
    assert snapshot.proof_competition.secondaries[0].openrouter_reasoning_effort == "max"
    models.AutonomousResearchStartRequest(**STARTS[2][1], proof_competition=competition)
    competition["secondaries"] = [role("openrouter")]
    with pytest.raises(ValidationError):
        models.AutonomousResearchStartRequest(**STARTS[2][1], proof_competition=competition)


def test_contract_version_matches_manifest():
    from backend.shared.build_info import _DEFAULT_BUILD_INFO
    manifest = json.loads((Path(__file__).resolve().parents[1] / "moto-update-manifest.json").read_text())
    assert manifest["api_contract_version"] == _DEFAULT_BUILD_INFO["api_contract_version"] == "build6-v95"
