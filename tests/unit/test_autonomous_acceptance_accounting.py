from types import SimpleNamespace
import importlib

import pytest

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.shared.models import BrainstormMetadata, SubmitterConfig


def test_legacy_brainstorm_metadata_uses_retained_count_as_cumulative_fallback():
    metadata = BrainstormMetadata(
        topic_id="topic_001",
        topic_prompt="test",
        submission_count=7,
    )

    assert metadata.submission_count == 7
    assert metadata.total_acceptances == 7


def test_brainstorm_metadata_keeps_retained_and_cumulative_counts_separate():
    metadata = BrainstormMetadata(
        topic_id="topic_001",
        topic_prompt="test",
        submission_count=5,
        total_acceptances=9,
    )

    assert metadata.submission_count == 5
    assert metadata.total_acceptances == 9


@pytest.mark.asyncio
async def test_solution_path_reviewer_is_configured_before_registry_acquire(monkeypatch):
    import backend.shared.solution_path as solution_path
    coordinator_module = importlib.import_module(
        "backend.autonomous.core.autonomous_coordinator"
    )

    coordinator = AutonomousCoordinator()
    primary = SubmitterConfig(
        submitter_id=1,
        provider="xai_grok_oauth",
        model_id="grok-test",
        openrouter_provider="host-test",
        openrouter_reasoning_effort="high",
        lm_studio_fallback_id="fallback-test",
        context_window=32000,
        max_output_tokens=4000,
        supercharge_enabled=True,
    )
    coordinator._submitter_configs = [primary]
    coordinator._base_user_research_prompt = "test prompt"
    coordinator._solution_path_acceptance_count = 2

    configured = {}

    def configure_role(role_id, config):
        configured[role_id] = config

    monkeypatch.setattr(
        coordinator_module.api_client_manager,
        "configure_role",
        configure_role,
    )
    monkeypatch.setattr(
        coordinator_module.brainstorm_memory,
        "get_all_brainstorms",
        lambda: _async_value([]),
    )

    manager = SimpleNamespace(
        state=SimpleNamespace(acceptance_count=3),
        set_acceptance_count=lambda count: _async_value(None),
    )

    async def acquire(*args, **kwargs):
        assert "autonomous_solution_path_reviewer" in configured
        return manager

    monkeypatch.setattr(solution_path.solution_path_registry, "acquire", acquire)
    await coordinator._initialize_solution_path_manager()

    config = configured["autonomous_solution_path_reviewer"]
    assert config.provider == primary.provider
    assert config.model_id == primary.model_id
    assert config.openrouter_provider == primary.openrouter_provider
    assert config.openrouter_reasoning_effort == primary.openrouter_reasoning_effort
    assert config.lm_studio_fallback_id == primary.lm_studio_fallback_id
    assert config.context_window == primary.context_window
    assert config.max_output_tokens == primary.max_output_tokens
    assert config.supercharge_enabled is True


async def _async_value(value):
    return value
