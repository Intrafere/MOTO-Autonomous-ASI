from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.leanoj.core.leanoj_coordinator import (
    LeanOJCoordinator,
    _solution_path_call_kind,
)
from backend.shared.models import LeanOJRoleConfig, LeanOJStartRequest


def test_leanoj_semantic_validator_classification_is_explicit():
    assert _solution_path_call_kind("leanoj_topic_val", "leanoj_topic_validator") == "validator"
    assert _solution_path_call_kind("leanoj_final_review", "leanoj_final_solver") == "validator"
    assert (
        _solution_path_call_kind(
            "leanoj_master_proof_edit_val", "leanoj_master_proof_edit_validator"
        )
        == "validator"
    )


def test_leanoj_solver_and_excluded_classification():
    assert _solution_path_call_kind("leanoj_final", "leanoj_final_solver") == "solver"
    assert (
        _solution_path_call_kind(
            "leanoj_brainstorm", "leanoj_brainstorm_submitter_10"
        )
        == "solver"
    )
    assert _solution_path_call_kind("proof_lean", "leanoj_proof_novelty") == "excluded"
    assert _solution_path_call_kind("tool_call", "leanoj_brainstorm_validator") == "excluded"


@pytest.mark.asyncio
async def test_leanoj_solution_path_uses_dedicated_main_submitter_role(
    monkeypatch, tmp_path
):
    import backend.shared.solution_path as solution_path
    from backend.leanoj.core import leanoj_coordinator as module

    primary = LeanOJRoleConfig(
        model_id="main-model",
        context_window=4096,
        max_output_tokens=512,
    )
    secondary = LeanOJRoleConfig(
        model_id="secondary-model",
        context_window=8192,
        max_output_tokens=1024,
    )
    request = LeanOJStartRequest(
        user_prompt="Solve the template.",
        lean_template="theorem target : True := by sorry",
        topic_generator=primary,
        topic_validator=primary,
        brainstorm_submitters=[primary, secondary],
        brainstorm_validator=primary,
        final_solver=primary,
    )
    coordinator = LeanOJCoordinator()
    coordinator._state.session_id = "run"
    configured = {}
    monkeypatch.setattr(
        coordinator,
        "_configure_role",
        lambda role_id, config: configured.setdefault(role_id, config),
    )
    monkeypatch.setattr(module.system_config, "data_dir", tmp_path)
    manager = SimpleNamespace(
        set_acceptance_count=AsyncMock(),
    )

    async def acquire(*_args, **_kwargs):
        assert configured["leanoj_solution_path_reviewer"] is primary
        return manager

    monkeypatch.setattr(solution_path.solution_path_registry, "acquire", acquire)
    await coordinator._initialize_solution_path_manager(request)

    assert configured["leanoj_solution_path_reviewer"].model_id == "main-model"
