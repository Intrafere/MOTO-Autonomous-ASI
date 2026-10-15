from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.routes import proofs as routes
from backend.autonomous.core import proof_run_manager as manager_module
from backend.autonomous.core.proof_run_manager import (
    NO_DISCOVERY_LIMIT_TERMINAL_REASON,
    ProofRunManager,
)
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.config import system_config
from backend.shared.models import (
    ProofCheckRequest,
    ProofCandidate,
    ProofCandidateListValidation,
    ProofCandidateNoveltyDecision,
    ProofRoleConfigSnapshot,
    ProofRunStopRequest,
    ProofRuntimeConfigSnapshot,
    ProofStageResult,
)
from tests.unit.test_autonomous_proof_rounds import (
    _FakeProofStage,
    _configured_coordinator,
)


async def _proof_loop_environment(monkeypatch, tmp_path):
    database = ProofDatabase()
    database.set_base_dir(tmp_path / "proofs")
    await database.initialize()
    manager = ProofRunManager()
    source = routes.ProofSourceAdapter(
        source_type="paper",
        source_id="paper-loop",
        source_title="Loop paper",
        source_content="Complete source.",
        user_prompt="Find prompt-relevant proofs.",
        canonical_user_prompt="Find prompt-relevant proofs.",
        proof_database=database,
        proof_store_id="manual:active",
        scope="manual",
        writable=True,
        append_to_source=False,
        append_proof_callback=None,
    )
    role = ProofRoleConfigSnapshot(
        model_id="fake-model",
        context_window=8000,
        max_output_tokens=1000,
    )
    snapshot = ProofRuntimeConfigSnapshot(
        brainstorm=role,
        paper=role,
        validator=role,
    )
    events: list[tuple[str, dict]] = []
    sleep_acquires: list[object] = []
    sleep_releases: list[object] = []

    async def broadcast(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    monkeypatch.setattr(routes, "proof_run_manager", manager)
    monkeypatch.setattr(routes, "_get_runtime_snapshot", AsyncMock(return_value=snapshot))
    monkeypatch.setattr(
        routes,
        "_resolve_proof_source_adapter",
        AsyncMock(return_value=source),
    )
    monkeypatch.setattr(routes, "_refresh_manual_assistant_memory", AsyncMock())
    monkeypatch.setattr(
        routes,
        "_configure_manual_roles",
        lambda kind, config, **kwargs: getattr(
            config, "brainstorm" if kind == "brainstorm" else "paper"
        ),
    )
    monkeypatch.setattr(
        routes,
        "ProofPruningCoordinator",
        lambda **kwargs: SimpleNamespace(
            restore=AsyncMock(),
            drain=AsyncMock(),
            notify_proof_registered=AsyncMock(),
            notify_context_pressure=AsyncMock(),
            route_config_fingerprint=lambda _snapshot: "test-route",
        ),
    )
    monkeypatch.setattr(routes.websocket, "broadcast_event", broadcast)
    monkeypatch.setattr(
        manager_module.sleep_inhibitor,
        "acquire",
        lambda owner: sleep_acquires.append(owner),
    )
    monkeypatch.setattr(
        manager_module.sleep_inhibitor,
        "release",
        lambda owner: sleep_releases.append(owner),
    )
    monkeypatch.setattr(
        routes,
        "get_lean4_client",
        lambda: SimpleNamespace(start_workspace_bootstrap=lambda: None),
    )
    monkeypatch.setattr(system_config, "lean4_enabled", True)
    return manager, events, sleep_acquires, sleep_releases


@pytest.mark.asyncio
async def test_real_continuous_route_completes_after_ten_empty_rounds_and_cleans_up(
    monkeypatch,
    tmp_path,
):
    manager, events, sleep_acquires, sleep_releases = await _proof_loop_environment(
        monkeypatch, tmp_path
    )
    calls: list[int] = []

    async def empty_stage(**kwargs):
        calls.append(kwargs["proof_round_index"])
        return ProofStageResult(
            source_type="paper",
            source_id="paper-loop",
            total_candidates=0,
            results=[],
        )

    monkeypatch.setattr(
        routes,
        "autonomous_coordinator",
        SimpleNamespace(
            _proof_verification_stage=SimpleNamespace(run_manual=empty_stage)
        ),
    )
    queued = await routes.run_manual_proof_check(
        ProofCheckRequest(
            source_type="paper",
            source_id="paper-loop",
            run_mode="loop_with_pruning",
        )
    )
    control = manager._runs[queued.proof_run_id]
    await asyncio.wait_for(control.task, timeout=10)

    assert calls == list(range(1, 11))
    assert control.snapshot.status == "completed"
    assert control.snapshot.terminal_reason == NO_DISCOVERY_LIMIT_TERMINAL_REASON
    assert control.snapshot.consecutive_no_discovery_rounds == 10
    assert control.snapshot.cleanup_completed
    assert len(sleep_acquires) == len(sleep_releases) == 1
    assert sleep_acquires[0] == sleep_releases[0]
    assert len(
        [event for event, _ in events if event == "proof_run_round_started"]
    ) == 10
    complete_events = [
        payload
        for event, payload in events
        if event == "proof_run_round_complete"
    ]
    assert len(complete_events) == 10
    assert [event["candidate_count"] for event in complete_events] == [0] * 10
    assert [
        event["consecutive_no_discovery_rounds"] for event in complete_events
    ] == list(range(1, 11))
    assert [
        event["no_discovery_round_limit"] for event in complete_events
    ] == [10] * 10
    assert [
        event["next_round_automatic"] for event in complete_events
    ] == ([True] * 9) + [False]
    assert complete_events[-2]["next_round_automatic"] is True
    assert complete_events[-1]["next_round_automatic"] is False
    terminal_events = [
        payload for event, payload in events if event == "proof_run_terminal"
    ]
    assert len(terminal_events) == 1
    assert (
        terminal_events[0]["terminal_reason"]
        == NO_DISCOVERY_LIMIT_TERMINAL_REASON
    )
    assert terminal_events[0]["consecutive_no_discovery_rounds"] == 10
    assert terminal_events[0]["no_discovery_round_limit"] == 10
    assert await ProofRunManager().get(queued.proof_run_id) is None


@pytest.mark.asyncio
async def test_rejected_list_empty_regeneration_counts_ten_completed_rounds(
    monkeypatch,
    tmp_path,
):
    manager, events, _sleep_acquires, _sleep_releases = (
        await _proof_loop_environment(monkeypatch, tmp_path)
    )
    identification_calls = 0

    async def identify(*_args, **_kwargs):
        nonlocal identification_calls
        identification_calls += 1
        if identification_calls % 2 == 0:
            return False, []
        round_index = (identification_calls + 1) // 2
        return True, [
            ProofCandidate(
                theorem_id=f"rejected-round-{round_index}",
                statement=f"Rejected theorem for round {round_index}.",
            )
        ]

    async def reject(_self, *, candidates, **_kwargs):
        return ProofCandidateListValidation(
            results=[
                ProofCandidateNoveltyDecision(
                    theorem_id=candidate.theorem_id,
                    decision="reject_not_novel",
                    reasoning="This target does not satisfy the novelty gate.",
                )
                for candidate in candidates
            ],
            feedback="Search for a genuinely novel prompt-directed target.",
        )

    monkeypatch.setattr(
        routes,
        "autonomous_coordinator",
        SimpleNamespace(_proof_verification_stage=ProofVerificationStage()),
    )
    with patch(
        "backend.autonomous.core.proof_verification_stage."
        "ProofIdentificationAgent.identify_candidates",
        new=identify,
    ), patch(
        "backend.autonomous.core.proof_verification_stage."
        "ProofCandidateListValidator.validate",
        new=reject,
    ):
        queued = await routes.run_manual_proof_check(
            ProofCheckRequest(
                source_type="paper",
                source_id="paper-loop",
                run_mode="loop_with_pruning",
            )
        )
        control = manager._runs[queued.proof_run_id]
        await asyncio.wait_for(control.task, timeout=10)

    assert identification_calls == 20
    assert control.snapshot.status == "completed"
    assert control.snapshot.terminal_reason == NO_DISCOVERY_LIMIT_TERMINAL_REASON
    assert control.snapshot.consecutive_no_discovery_rounds == 10
    assert len(
        [event for event, _payload in events if event == "proof_candidate_list_review_rejected"]
    ) == 10
    assert len(
        [event for event, _payload in events if event == "proof_candidate_list_regeneration_started"]
    ) == 10
    assert len(
        [event for event, _payload in events if event == "proof_check_no_candidates"]
    ) == 10
    complete_events = [
        payload
        for event, payload in events
        if event == "proof_run_round_complete"
    ]
    assert [
        event["consecutive_no_discovery_rounds"] for event in complete_events
    ] == list(range(1, 11))
    assert complete_events[-1]["next_round_automatic"] is False


@pytest.mark.asyncio
async def test_real_continuous_stop_route_is_generation_fenced_and_cleans_up(
    monkeypatch,
    tmp_path,
):
    manager, events, sleep_acquires, sleep_releases = await _proof_loop_environment(
        monkeypatch, tmp_path
    )
    round_two_started = asyncio.Event()
    hold_round_two = asyncio.Event()
    calls: list[int] = []

    async def blocking_stage(**kwargs):
        round_index = kwargs["proof_round_index"]
        calls.append(round_index)
        if round_index == 2:
            round_two_started.set()
            await hold_round_two.wait()
        return ProofStageResult(
            source_type="paper",
            source_id="paper-loop",
            total_candidates=0,
            results=[],
        )

    monkeypatch.setattr(
        routes,
        "autonomous_coordinator",
        SimpleNamespace(
            _proof_verification_stage=SimpleNamespace(run_manual=blocking_stage)
        ),
    )
    queued = await routes.run_manual_proof_check(
        ProofCheckRequest(
            source_type="paper",
            source_id="paper-loop",
            run_mode="loop_with_pruning",
        )
    )
    control = manager._runs[queued.proof_run_id]
    await asyncio.wait_for(round_two_started.wait(), timeout=5)

    with pytest.raises(Exception) as stale:
        await routes.stop_proof_run(
            queued.proof_run_id,
            ProofRunStopRequest(expected_lifecycle_generation=99),
        )
    assert getattr(stale.value, "status_code", None) == 409

    stopping = await routes.stop_proof_run(
        queued.proof_run_id,
        ProofRunStopRequest(
            expected_lifecycle_generation=queued.lifecycle_generation
        ),
    )
    assert stopping.status in {"stopping", "stopped"}
    await asyncio.gather(control.task, return_exceptions=True)
    hold_round_two.set()

    assert calls == [1, 2]
    assert control.snapshot.status == "stopped"
    assert control.snapshot.cleanup_completed
    assert len(sleep_acquires) == len(sleep_releases) == 1
    assert sleep_acquires[0] == sleep_releases[0]
    assert len([event for event, _ in events if event == "proof_run_terminal"]) == 1


@pytest.mark.asyncio
async def test_real_autonomous_automatic_round_policy_matrix():
    previous = system_config.lean4_enabled
    system_config.lean4_enabled = True
    try:
        proofs_only_zero = _FakeProofStage([0, 1, 0, 1])
        proofs_only = _configured_coordinator(
            proofs_only_zero,
            allow_research_papers=False,
        )
        await proofs_only._run_proof_verification(
            content="Brainstorm source.",
            source_type="brainstorm",
            source_id="topic-zero",
            source_title="Proofs-only brainstorm",
        )
        assert len(proofs_only_zero.calls) == 4
        assert proofs_only_zero.calls[-1]["trigger"] == "automatic_round_4"

        proofs_only_cap = _FakeProofStage([1, 1, 1, 1, 1])
        proofs_only = _configured_coordinator(
            proofs_only_cap,
            allow_research_papers=False,
        )
        await proofs_only._run_proof_verification(
            content="Brainstorm source.",
            source_type="brainstorm",
            source_id="topic-cap",
            source_title="Proofs-only brainstorm",
        )
        assert len(proofs_only_cap.calls) == 4
        assert proofs_only_cap.calls[-1]["trigger"] == "automatic_round_4"

        combined_brainstorm = _FakeProofStage([0, 1, 1])
        combined = _configured_coordinator(
            combined_brainstorm,
            allow_research_papers=True,
        )
        await combined._run_proof_verification(
            content="Brainstorm source.",
            source_type="brainstorm",
            source_id="topic-combined",
            source_title="Combined brainstorm",
        )
        assert len(combined_brainstorm.calls) == 2
        assert all(
            call["proof_max_rounds"] == 2 for call in combined_brainstorm.calls
        )

        completed_paper = _FakeProofStage([1, 1])
        combined = _configured_coordinator(
            completed_paper,
            allow_research_papers=True,
        )
        await combined._run_proof_verification(
            content="Paper source.",
            source_type="paper",
            source_id="paper-complete",
            source_title="Completed paper",
        )
        assert len(completed_paper.calls) == 1
        assert completed_paper.calls[0]["proof_max_rounds"] == 1
    finally:
        system_config.lean4_enabled = previous
