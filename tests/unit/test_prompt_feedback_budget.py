import pytest

from backend.shared.prompt_feedback_budget import (
    fit_prompt_with_feedback,
    fit_prompt_with_feedback_async,
)
from backend.aggregator.memory.local_training import LocalTrainingMemory
from backend.compiler.memory.compiler_rejection_log import CompilerRejectionLog
from backend.compiler.memory.outline_memory import OutlineMemory
from backend.compiler.prompts.outline_prompts import build_outline_create_prompt
from backend.compiler.prompts.review_prompts import build_review_prompt


def test_keeps_largest_newest_whole_entry_suffix(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.prompt_feedback_budget.count_tokens",
        lambda text: len(text),
    )
    entries = ["old", "middle", "new"]

    result = fit_prompt_with_feedback(
        entries,
        build_prompt=lambda selected: "base:" + "|".join(selected),
        available_tokens=len("base:middle|new"),
    )

    assert result.retained_entries == ("middle", "new")
    assert result.removed_entries == 1
    assert entries == ["old", "middle", "new"]
    assert result.fits


def test_counts_complete_rebuilt_prompt(monkeypatch):
    seen = []

    def fake_count(text):
        seen.append(text)
        return len(text)

    monkeypatch.setattr("backend.shared.prompt_feedback_budget.count_tokens", fake_count)
    result = fit_prompt_with_feedback(
        ["one", "two"],
        build_prompt=lambda selected: f"mandatory::{','.join(selected)}::schema",
        available_tokens=len("mandatory::two::schema"),
    )

    assert seen == [
        "mandatory::one,two::schema",
        "mandatory::two::schema",
    ]
    assert result.prompt == "mandatory::two::schema"


def test_newest_entry_is_retained_when_it_cannot_fit(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.prompt_feedback_budget.count_tokens",
        lambda text: len(text),
    )

    result = fit_prompt_with_feedback(
        ["old", "oversized-newest"],
        build_prompt=lambda selected: "mandatory:" + "|".join(selected),
        available_tokens=1,
    )

    assert result.retained_entries == ("oversized-newest",)
    assert result.removed_entries == 1
    assert not result.fits


def test_initial_projection_is_capped_at_five(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.prompt_feedback_budget.count_tokens",
        lambda text: len(text),
    )

    result = fit_prompt_with_feedback(
        range(8),
        build_prompt=lambda selected: ",".join(map(str, selected)),
        available_tokens=100,
    )

    assert result.retained_entries == (3, 4, 5, 6, 7)


def test_no_feedback_builds_prompt_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "backend.shared.prompt_feedback_budget.count_tokens",
        lambda text: len(text),
    )

    result = fit_prompt_with_feedback(
        [],
        build_prompt=lambda selected: calls.append(selected) or "base",
        available_tokens=4,
    )

    assert calls == [()]
    assert result.retained_entries == ()
    assert result.fits


@pytest.mark.asyncio
async def test_async_fitter_preserves_whole_newest_suffix(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.prompt_feedback_budget.count_tokens",
        lambda text: len(text),
    )

    async def build(selected):
        return f"base:{'|'.join(selected)}"

    result = await fit_prompt_with_feedback_async(
        ["old", "middle", "new"],
        build_prompt=build,
        available_tokens=len("base:middle|new"),
    )

    assert result.retained_entries == ("middle", "new")


def test_local_rejection_renderer_does_not_mutate_or_split_entries():
    entries = (
        {"validator_summary": "first reason", "submission_preview": "first body"},
        {"validator_summary": "new reason", "submission_preview": "new body"},
    )

    rendered = LocalTrainingMemory.render_rejections(entries[1:])

    assert "[REJECTION 1]" in rendered
    assert "new reason" in rendered
    assert "first reason" not in rendered
    assert entries[1]["submission_preview"] == "new body"


def test_compiler_model_projection_defaults_to_newest_five():
    memory = CompilerRejectionLog()
    memory.rejections = [{"text": str(index)} for index in range(8)]

    import asyncio

    snapshot = asyncio.run(memory.get_rejection_entries())

    assert [entry["text"] for entry in snapshot] == ["3", "4", "5", "6", "7"]
    snapshot[0]["text"] = "changed"
    assert memory.rejections[3]["text"] == "3"


def test_outline_renderer_keeps_accepted_outline_when_comments_are_shed():
    rendered = OutlineMemory.render_creation_feedback(
        "I. Accepted outline",
        ("newest validator comment",),
    )

    assert "I. Accepted outline" in rendered
    assert "newest validator comment" in rendered


@pytest.mark.asyncio
async def test_compiler_prompt_builders_use_only_caller_selected_feedback():
    outline_prompt = await build_outline_create_prompt(
        "goal",
        "evidence",
        rejection_history="selected rejection",
        creation_feedback="selected outline feedback",
    )
    review_prompt = await build_review_prompt(
        "goal",
        "paper",
        "outline",
        rejection_history="selected review rejection",
    )

    assert "selected rejection" in outline_prompt
    assert "selected outline feedback" in outline_prompt
    assert "selected review rejection" in review_prompt


@pytest.mark.asyncio
async def test_outline_budget_sheds_comments_but_keeps_accepted_outline(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.prompt_feedback_budget.count_tokens",
        lambda text: len(text),
    )
    accepted = "MANDATORY ACCEPTED OUTLINE"
    comments = ("old comment", "new comment")

    async def build(selected):
        return OutlineMemory.render_creation_feedback(accepted, selected)

    newest_only = await build((comments[-1],))
    result = await fit_prompt_with_feedback_async(
        comments,
        build_prompt=build,
        available_tokens=len(newest_only),
    )

    assert result.retained_entries == ("new comment",)
    assert accepted in result.prompt
    assert "old comment" not in result.prompt
