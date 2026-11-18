"""Proof-round memory: what discovery sees about earlier rounds' attempts."""

import json
from types import SimpleNamespace

import pytest

import backend.autonomous.agents.proof_identification_agent as identification_module
from backend.autonomous.agents.proof_identification_agent import (
    ProofIdentificationAgent,
    ProofIdentificationContextError,
)
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.prompts.proof_prompts import build_proof_identification_prompt
from backend.shared.models import ProofAttemptResult, ProofStageResult
from backend.shared.prompt_feedback_budget import (
    fit_prompt_with_feedback_streams,
    shed_oldest_feedback,
)
from backend.shared.provider_errors import (
    ProviderContextLengthError,
    ProviderRouteIdentity,
)
from backend.shared.utils import count_tokens


def test_streams_shed_first_stream_to_newest_before_touching_second(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.prompt_feedback_budget.count_tokens",
        lambda text: len(text),
    )

    def build(streams):
        memory, rejections = streams
        return "|".join(memory) + "#" + "|".join(rejections)

    full = (("m1", "m2", "m3"), ("r1", "r2"))
    result = fit_prompt_with_feedback_streams(
        full,
        build_prompt=build,
        available_tokens=len(build((("m3",), ("r1", "r2")))),
    )
    assert result.retained_streams == (("m3",), ("r1", "r2"))
    assert result.fits

    result = fit_prompt_with_feedback_streams(
        full, build_prompt=build, available_tokens=1
    )
    assert result.retained_streams == (("m3",), ("r2",))
    assert not result.fits

    result = fit_prompt_with_feedback_streams(
        (tuple(f"m{index}" for index in range(8)), ()),
        build_prompt=build,
        available_tokens=10_000,
    )
    assert result.retained_streams[0] == ("m3", "m4", "m5", "m6", "m7")


def test_shed_oldest_feedback_stops_at_each_newest_entry():
    assert shed_oldest_feedback((("a", "b"), ("c", "d"))) == (("b",), ("c", "d"))
    assert shed_oldest_feedback((("b",), ("c", "d"))) == (("b",), ("d",))
    assert shed_oldest_feedback((("b",), ("d",))) is None
    assert shed_oldest_feedback(((), ())) is None


def test_follow_up_prompt_explains_round_memory_and_novelty_ratings():
    entry = (
        "Round 2: 1/1 candidates verified, 0 novel.\n"
        "- Verified in Lean 4, novelty rating not_novel: Hypergeometric miss bound"
    )
    prompt = build_proof_identification_prompt(
        user_prompt="Solve the problem.",
        source_type="brainstorm",
        source_id="topic_001",
        source_content="Source content.",
        proof_round_index=3,
        proof_max_rounds=0,
        prior_round_results=entry,
    )

    assert entry in prompt
    assert "up to the five most recent rounds that attempted proofs" in prompt
    assert "rated not_novel was judged a standard, known, or routine result" in prompt


def _memory_entry(index: int) -> str:
    return (
        f"Round {index}: 1/1 candidates verified, 0 novel.\n"
        f"- Verified in Lean 4, novelty rating not_novel: MEMORY_{index} "
        + "detail " * 400
    )


def _rejection_entry(index: int) -> str:
    return f"Attempt {index}: REJECTION_{index} " + "feedback " * 400


class _FakeApiClientManager:
    def __init__(self, *, reject_first: int = 0):
        self.prompts: list[str] = []
        self.reject_first = reject_first

    async def prewarm_assistant_memory_context(self, **_kwargs):
        return ""

    async def generate_completion(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        self.prompts.append(prompt)
        if len(self.prompts) <= self.reject_first:
            raise ProviderContextLengthError(
                "provider context limit",
                route=ProviderRouteIdentity(provider="openrouter", model="model"),
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"has_provable_theorems": False, "theorems": []}
                        )
                    }
                }
            ]
        }


def _prompt_tokens(memory, rejections) -> int:
    return count_tokens(
        build_proof_identification_prompt(
            user_prompt="Solve the prompt.",
            source_type="brainstorm",
            source_id="topic_001",
            source_content="Source content.",
            proof_round_index=7,
            proof_max_rounds=0,
            prior_round_results="\n\n".join(memory),
            candidate_list_rejection_feedback="\n\n".join(rejections),
        )
    )


async def _identify(monkeypatch, *, available_tokens, fake):
    monkeypatch.setattr(identification_module, "api_client_manager", fake)
    monkeypatch.setattr(
        identification_module,
        "rag_config",
        SimpleNamespace(
            get_available_input_tokens=lambda *_args: available_tokens,
            context_buffer_tokens=0,
        ),
    )
    agent = ProofIdentificationAgent(
        model_id="model",
        context_window=200_000,
        max_output_tokens=1000,
        role_id="autonomous_proof_identification_test",
    )
    rejections = tuple(_rejection_entry(index) for index in range(1, 4))
    return await agent.identify_candidates(
        user_research_prompt="Solve the prompt.",
        source_type="brainstorm",
        source_id="topic_001",
        source_content="Source content.",
        proof_round_index=7,
        proof_max_rounds=0,
        prior_round_results=tuple(_memory_entry(index) for index in range(1, 7)),
        candidate_list_rejection_feedback_entries=rejections,
    )


@pytest.mark.asyncio
async def test_discovery_drops_oldest_round_memory_first(monkeypatch):
    rejections = [_rejection_entry(index) for index in range(1, 4)]
    budget = _prompt_tokens([_memory_entry(5), _memory_entry(6)], rejections)
    fake = _FakeApiClientManager()

    await _identify(monkeypatch, available_tokens=budget, fake=fake)

    prompt = fake.prompts[0]
    assert "MEMORY_1" not in prompt  # beyond the newest five
    assert "MEMORY_4" not in prompt
    assert "MEMORY_5" in prompt and "MEMORY_6" in prompt
    assert all(f"REJECTION_{index}" in prompt for index in range(1, 4))


@pytest.mark.asyncio
async def test_discovery_keeps_newest_entries_then_raises_overflow(monkeypatch):
    budget = _prompt_tokens([_memory_entry(6)], [_rejection_entry(3)]) - 1

    with pytest.raises(ProofIdentificationContextError):
        await _identify(monkeypatch, available_tokens=budget, fake=_FakeApiClientManager())


@pytest.mark.asyncio
async def test_provider_context_rejection_sheds_round_memory_first(monkeypatch):
    fake = _FakeApiClientManager(reject_first=4)

    await _identify(monkeypatch, available_tokens=1_000_000, fake=fake)

    assert len(fake.prompts) == 5
    final = fake.prompts[-1]
    assert "MEMORY_5" not in final and "MEMORY_6" in final
    assert all(f"REJECTION_{index}" in final for index in range(1, 4))


def test_autonomous_restore_rebuilds_memory_from_checkpoint():
    coordinator = AutonomousCoordinator()
    finished = ProofStageResult(
        source_type="brainstorm",
        source_id="topic_001",
        total_candidates=1,
        verified_count=1,
        results=[
            ProofAttemptResult(
                theorem_id="thm_1",
                theorem_statement="Round three target",
                success=True,
                novelty_tier="not_novel",
            )
        ],
    )
    checkpoint = {
        "trigger": "brainstorm_complete_round_3",
        "proof_round_index": 3,
        "status": "complete",
        "prior_round_results": ["Round 1: earlier", "Round 2: earlier"],
        "source_type": "brainstorm",
        "source_id": "topic_001",
        "total_candidates": 1,
        "verified_count": 1,
        "results": [item.model_dump(mode="json") for item in finished.results],
    }

    memory = coordinator._restored_proof_round_memory(
        checkpoint, base_trigger="brainstorm_complete"
    )
    assert memory[:2] == ("Round 1: earlier", "Round 2: earlier")
    assert "novelty rating not_novel: Round three target" in memory[2]

    running = {**checkpoint, "status": "running"}
    assert coordinator._restored_proof_round_memory(
        running, base_trigger="brainstorm_complete"
    ) == ("Round 1: earlier", "Round 2: earlier")
    legacy = {**running, "prior_round_results": "Round 1: legacy text"}
    assert coordinator._restored_proof_round_memory(
        legacy, base_trigger="brainstorm_complete"
    ) == ("Round 1: legacy text",)
    assert coordinator._restored_proof_round_memory(
        checkpoint, base_trigger="paper_complete"
    ) == ()
    assert coordinator._restored_proof_round_memory(None, base_trigger="x") == ()
