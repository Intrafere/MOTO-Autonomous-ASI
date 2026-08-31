from types import SimpleNamespace

import pytest

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.compiler.core import compiler_coordinator as compiler_module
from backend.compiler.core.compiler_coordinator import CompilerCoordinator


def test_tier2_compiler_prompt_directly_includes_original_objective():
    coordinator = AutonomousCoordinator()
    coordinator._base_user_research_prompt = "Build a safe orbital transfer controller."
    coordinator._user_research_prompt = "proof-framed derivative"

    prompt = coordinator._get_effective_compiler_prompt("Validated Transfer Policies")

    assert "Original User Research Objective:" in prompt
    assert "Build a safe orbital transfer controller." in prompt
    assert "Paper Title and Instructions:" in prompt
    assert "Validated Transfer Policies" in prompt


def test_tier3_prompt_does_not_use_tier2_compiler_prompt(monkeypatch):
    coordinator = AutonomousCoordinator()
    coordinator._base_user_research_prompt = "original objective"

    monkeypatch.setattr(
        coordinator,
        "_get_effective_compiler_prompt",
        lambda _title: pytest.fail("Tier 3 must not use Tier 2 brainstorm-paper context"),
    )

    prompt = coordinator._build_tier3_compiler_prompt(
        paper_title="Final Answer",
        findings_summary="Supported finding.",
    )

    assert "Supported finding." in prompt
    assert "Original User Research Objective:" not in prompt


def test_parent_requires_authoritative_child_completion_not_abstract_heading():
    coordinator = AutonomousCoordinator()
    coordinator._paper_compiler = SimpleNamespace(autonomous_paper_complete=False)
    paper = "# Abstract\nPremature heading."

    assert coordinator._has_abstract(paper)
    assert not coordinator._child_compiler_confirmed_completion(paper)

    coordinator._paper_compiler.autonomous_paper_complete = True
    assert coordinator._child_compiler_confirmed_completion(paper)


@pytest.mark.asyncio
async def test_child_completion_requires_full_structure_and_explicit_signal(monkeypatch):
    coordinator = CompilerCoordinator()
    coordinator.enable_autonomous_mode()
    coordinator.autonomous_section_phase = "abstract"

    paper = (
        "# Abstract\nResult.\n\n"
        "# I. Introduction\nContext.\n\n"
        "# II. Method\nBody.\n\n"
        "# III. Conclusion\nConclusion.\n"
    )
    monkeypatch.setattr(compiler_module.paper_memory, "get_paper", _async_value(paper))
    monkeypatch.setattr(compiler_module.paper_memory, "get_word_count", _async_value(8))
    monkeypatch.setattr(coordinator, "_broadcast", _async_callable())

    assert not await coordinator._check_phase_transition(section_complete=False)
    assert not coordinator.autonomous_paper_complete
    assert await coordinator._check_phase_transition(section_complete=True)
    assert coordinator.autonomous_paper_complete


@pytest.mark.asyncio
async def test_child_completion_rejects_missing_body(monkeypatch):
    coordinator = CompilerCoordinator()
    coordinator.enable_autonomous_mode()
    coordinator.autonomous_section_phase = "abstract"

    paper = (
        "# Abstract\nResult.\n\n"
        "# I. Introduction\nContext.\n\n"
        "# III. Conclusion\nConclusion.\n"
    )
    monkeypatch.setattr(compiler_module.paper_memory, "get_paper", _async_value(paper))
    monkeypatch.setattr(compiler_module.paper_memory, "get_word_count", _async_value(6))

    assert not await coordinator._check_phase_transition(section_complete=True)
    assert not coordinator.autonomous_paper_complete


def _async_value(value):
    async def getter(*_args, **_kwargs):
        return value

    return getter


def _async_callable():
    async def call(*_args, **_kwargs):
        return None

    return call
