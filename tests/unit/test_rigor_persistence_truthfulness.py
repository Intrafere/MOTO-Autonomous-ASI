from types import SimpleNamespace

import pytest

from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log
from backend.compiler.memory.paper_memory import paper_memory


def _lean_result():
    return SimpleNamespace(
        proof_id="proof-1",
        theorem_statement="theorem saved : True := by trivial",
        lean_code="theorem saved : True := by trivial",
        is_novel=True,
        theorem_name="saved",
        novelty_tier="novel",
        placement_preference="appendix_only",
        metadata={"placement_preference": "appendix_only"},
        initial_placement_submission=None,
    )


@pytest.mark.asyncio
async def test_appendix_failure_after_repair_is_not_reported_as_rigor_success(
    monkeypatch,
) -> None:
    coordinator = CompilerCoordinator()
    events = []
    append_results = iter((False, False))

    async def fake_append(_entry):
        return next(append_results)

    async def fake_repair():
        return None

    async def fake_broadcast(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(paper_memory, "append_to_theorems_appendix", fake_append)
    monkeypatch.setattr(paper_memory, "ensure_markers_intact", fake_repair)
    monkeypatch.setattr(coordinator, "_broadcast", fake_broadcast)

    result = await coordinator._place_or_appendix_fallback(_lean_result())

    assert result is False
    assert coordinator.rigor_acceptances == 0
    assert not any(event == "compiler_acceptance" for event, _ in events)
    assert not any(event == "paper_updated" for event, _ in events)
    assert any(
        payload.get("placement_outcome") == "appendix_persistence_failed"
        for event, payload in events
        if event == "compiler_rejection"
    )


@pytest.mark.asyncio
async def test_appendix_success_after_marker_repair_is_reported_once(
    monkeypatch,
) -> None:
    coordinator = CompilerCoordinator()
    events = []
    acceptance_ids = []
    append_results = iter((False, True))

    async def fake_append(_entry):
        return next(append_results)

    async def fake_repair():
        return None

    async def fake_word_count():
        return 42

    async def fake_broadcast(event, payload):
        events.append((event, payload))

    async def fake_add_acceptance(submission_id, _mode, _preview):
        acceptance_ids.append(submission_id)

    monkeypatch.setattr(paper_memory, "append_to_theorems_appendix", fake_append)
    monkeypatch.setattr(paper_memory, "ensure_markers_intact", fake_repair)
    monkeypatch.setattr(paper_memory, "get_word_count", fake_word_count)
    monkeypatch.setattr(coordinator, "_broadcast", fake_broadcast)
    monkeypatch.setattr(
        compiler_rejection_log, "add_acceptance", fake_add_acceptance
    )

    result = await coordinator._place_or_appendix_fallback(_lean_result())

    assert result is True
    assert coordinator.rigor_acceptances == 1
    assert sum(event == "compiler_acceptance" for event, _ in events) == 1
    assert sum(event == "paper_updated" for event, _ in events) == 1
    assert acceptance_ids == ["rigor_appendix_requested_proof-1"]
