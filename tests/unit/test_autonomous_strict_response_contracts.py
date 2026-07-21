import json

import pytest

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator  # noqa: F401
from backend.autonomous.agents.final_answer.answer_format_selector import (
    AnswerFormatSelector,
)
from backend.autonomous.agents.final_answer.volume_organizer import VolumeOrganizer
from backend.autonomous.agents.paper_title_selector import PaperTitleSelectorAgent
from backend.autonomous.agents.topic_selector import TopicSelectorAgent
from backend.shared.api_client_manager import api_client_manager
from backend.shared.models import CertaintyAssessment


def _response(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


@pytest.fixture
def completion_stub(monkeypatch):
    async def install(payload: dict) -> None:
        async def fake_generate_completion(**_kwargs):
            return _response(payload)

        async def fake_prewarm(**_kwargs):
            return None

        monkeypatch.setattr(
            api_client_manager, "generate_completion", fake_generate_completion
        )
        monkeypatch.setattr(
            api_client_manager, "prewarm_assistant_memory_context", fake_prewarm
        )

    return install


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        {"reasoning": "Missing format"},
        {"answer_format": "brief", "reasoning": "Invalid enum"},
        {"answer_format": "short_form", "reasoning": "   "},
    ),
)
async def test_answer_format_rejects_missing_invalid_or_blank_required_fields(
    completion_stub, payload
) -> None:
    await completion_stub(payload)
    selector = AnswerFormatSelector("submitter", "validator", 8192, 1024)

    result = await selector._generate_selection(
        "Exact objective",
        CertaintyAssessment(
            certainty_level="partial_answer",
            known_certainties_summary="Some support",
            reasoning="Evidence is incomplete",
        ),
        [{"paper_id": "p1", "title": "Paper"}],
    )

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        {
            "volume_title": "Volume",
            "chapters": [],
            "outline_complete": True,
            "reasoning": "No chapters",
        },
        {
            "volume_title": "Volume",
            "chapters": [
                {
                    "chapter_type": "existing_paper",
                    "paper_id": "p1",
                    "title": "Body",
                    "order": 1,
                }
            ],
            "outline_complete": "true",
            "reasoning": "Wrong boolean type",
        },
        {
            "volume_title": "Volume",
            "chapters": [
                {
                    "chapter_type": "introduction",
                    "title": "Introduction",
                    "order": 1,
                },
                {
                    "chapter_type": "existing_paper",
                    "paper_id": "p1",
                    "title": "Body",
                    "order": 2,
                },
            ],
            "outline_complete": True,
            "reasoning": "Missing conclusion",
        },
    ),
)
async def test_volume_organizer_rejects_incomplete_or_repaired_contracts(
    completion_stub, payload
) -> None:
    await completion_stub(payload)
    organizer = VolumeOrganizer("submitter", "validator", 8192, 1024)

    result = await organizer._generate_organization(
        "Exact objective",
        CertaintyAssessment(
            certainty_level="partial_answer",
            known_certainties_summary="Some support",
            reasoning="Evidence is incomplete",
        ),
        [{"paper_id": "p1", "title": "Paper"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_volume_organizer_does_not_force_completion_at_iteration_limit(
    monkeypatch,
) -> None:
    organizer = VolumeOrganizer("submitter", "validator", 8192, 1024)
    organizer.MAX_ITERATIONS = 1

    async def fake_generate(*_args, **_kwargs):
        from backend.shared.models import VolumeChapter, VolumeOrganization

        return VolumeOrganization(
            volume_title="Incomplete",
            chapters=[
                VolumeChapter(
                    chapter_type="introduction", title="Introduction", order=1
                ),
                VolumeChapter(
                    chapter_type="conclusion", title="Conclusion", order=2
                ),
            ],
            outline_complete=False,
            revision_reasoning="Needs another pass",
        )

    async def fake_validate(*_args, **_kwargs):
        return True, "Valid but incomplete"

    monkeypatch.setattr(organizer, "_generate_organization", fake_generate)
    monkeypatch.setattr(organizer, "_validate_organization", fake_validate)

    result = await organizer.organize_volume(
        "Exact objective",
        CertaintyAssessment(
            certainty_level="partial_answer",
            known_certainties_summary="Some support",
            reasoning="Evidence is incomplete",
        ),
        [{"paper_id": "p1", "title": "Paper"}],
    )

    assert result is None


@pytest.mark.asyncio
async def test_topic_selector_requires_non_empty_reasoning(completion_stub) -> None:
    await completion_stub(
        {"action": "new_topic", "topic_prompt": "Direct route", "reasoning": ""}
    )
    selector = TopicSelectorAgent("model", 8192, 1024)

    result = await selector.select_topic("Exact objective", [], [], "Candidate")

    assert result is None


@pytest.mark.asyncio
async def test_paper_title_requires_non_empty_reasoning(completion_stub) -> None:
    await completion_stub({"paper_title": "Strong title", "reasoning": ""})
    selector = PaperTitleSelectorAgent("model", "validator", 8192, 1024)

    result = await selector._generate_title(
        "Exact objective", "Topic", "Summary", [], [], "", ""
    )

    assert result is None
