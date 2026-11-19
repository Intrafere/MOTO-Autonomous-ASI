"""Paper critiques get one bounded contract retry and fail visibly, never silently."""
from __future__ import annotations

import json

import pytest

from backend.shared.api_client_manager import api_client_manager
from backend.shared.critique_prompts import (
    CRITIQUE_RESPONSE_SHAPE,
    critique_contract_failure_detail,
    generate_validated_paper_critique,
    parse_critique_response,
)
from backend.shared.model_output_contract import ModelOutputContractError


def _critique(**overrides):
    data = {
        "novelty_rating": 7,
        "novelty_feedback": "Novel framing.",
        "correctness_rating": 6,
        "correctness_feedback": "One step needs a citation.",
        "impact_rating": 5,
        "impact_feedback": "Moderate impact.",
        "full_critique": "Careful but incremental.",
    }
    data.update(overrides)
    return data


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def calls(monkeypatch):
    recorded = {"completions": [], "normalized": []}

    async def report(role_id, repairs, **kwargs):
        if repairs:
            recorded["normalized"].append((role_id, list(repairs), kwargs))

    monkeypatch.setattr(
        "backend.shared.model_output_contract.report_model_output_normalizations", report
    )
    return recorded


def _script(monkeypatch, recorded, replies):
    queue = list(replies)

    async def generate_completion(**kwargs):
        recorded["completions"].append(kwargs)
        return _completion(queue.pop(0))

    monkeypatch.setattr(api_client_manager, "generate_completion", generate_completion)


async def _request():
    return await generate_validated_paper_critique(
        prompt="Critique this paper.",
        task_id="critique_1",
        model="gemma",
        context_window=32000,
        max_output_tokens=4000,
    )


@pytest.mark.asyncio
async def test_valid_critique_makes_one_call(monkeypatch, calls):
    _script(monkeypatch, calls, [json.dumps(_critique())])
    data = await _request()
    assert data["novelty_rating"] == 7
    assert len(calls["completions"]) == 1
    assert calls["normalized"] == []


@pytest.mark.asyncio
async def test_bad_critique_gets_one_retry_naming_the_field(monkeypatch, calls):
    _script(monkeypatch, calls, [json.dumps(_critique(novelty_rating="7")), json.dumps(_critique())])
    data = await _request()
    assert data["novelty_rating"] == 7
    assert len(calls["completions"]) == 2
    retry = calls["completions"][1]
    assert retry["task_id"] == "critique_1_retry"
    retry_text = retry["messages"][-1]["content"]
    assert "Your previous critique response could not be used." in retry_text
    assert "novelty_rating" in retry_text
    assert CRITIQUE_RESPONSE_SHAPE in retry_text


@pytest.mark.asyncio
async def test_second_failure_raises_typed_error(monkeypatch, calls):
    truncated = json.dumps(_critique())[:-40]
    _script(monkeypatch, calls, [truncated, ""])
    with pytest.raises(ModelOutputContractError) as excinfo:
        await _request()
    assert isinstance(excinfo.value.__cause__, ModelOutputContractError)
    assert len(calls["completions"]) == 2
    detail = critique_contract_failure_detail(excinfo.value)
    assert "after one retry" in detail and "No critique was saved" in detail


@pytest.mark.asyncio
async def test_applied_relabel_is_published_once_accepted(monkeypatch, calls):
    reply = json.dumps(_critique()).replace('"novelty_feedback"', '"novelty_feedbak"')
    _script(monkeypatch, calls, [reply])
    data = await _request()
    assert data["novelty_feedback"] == "Novel framing."
    [(role_id, repairs, kwargs)] = calls["normalized"]
    assert role_id == "paper_critic"
    assert repairs[0].to_key == "novelty_feedback"
    assert kwargs["task_id"] == "critique_1"


def test_parser_without_a_repairs_list_is_strict():
    reply = json.dumps(_critique()).replace('"novelty_feedback"', '"novelty_feedbak"')
    with pytest.raises(ModelOutputContractError):
        parse_critique_response(reply)
    trailing = json.dumps(_critique())[:-1] + ",}"
    with pytest.raises(ModelOutputContractError):
        parse_critique_response(trailing)
    repairs: list = []
    assert parse_critique_response(trailing, repairs=repairs)["impact_rating"] == 5
    assert [repair.kind for repair in repairs] == ["trailing_comma"]
