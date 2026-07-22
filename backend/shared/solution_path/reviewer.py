"""Shared production reviewer contract for Progressive Solution Path proposals."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from backend.shared.json_parser import (
    parse_json,
    sanitize_model_output_for_retry_context,
)
from backend.shared.utils import count_tokens

from .models import PlanProposal, ReviewDecision, SolutionPlan


CompletionCall = Callable[[list[dict[str, str]]], Awaitable[Any]]


def build_review_prompt(
    *,
    user_prompt: str,
    proposal: PlanProposal,
    current_plan: SolutionPlan | None,
    extra_context: str = "",
) -> str:
    return (
        "You are production Main Submitter 1 reviewing a validator-proposed "
        "Progressive Solution Path. Preserve the user's exact objective. Approve only "
        "a material improvement. You may reject it, approve it as written, or perform "
        "checked follow-up edits. Follow-up edits may replace `main_route`, replace the "
        "whole `followup_route`, or use `edits` with add/update/delete/check operations "
        "over stable step_id values. Use `more_edits=true` only when another review pass "
        "is genuinely required; otherwise finish with approve or reject. Never treat "
        "the plan as evidence.\n\n"
        "Return one JSON object only:\n"
        '{"decision":"approve|reject|followup","reasoning":"...",'
        '"main_route":"optional replacement","followup_route":null,'
        '"edits":[{"operation":"add|update|delete|check","step_id":"...",'
        '"step":null,"after_step_id":null,"expected_title":null}],'
        '"more_edits":false}\n\n'
        f"USER PROMPT:\n{user_prompt}\n\n"
        f"{extra_context}\n\n"
        "CURRENT PLAN:\n"
        f"{current_plan.model_dump_json(indent=2) if current_plan else 'None'}\n\n"
        "PROPOSAL:\n"
        f"{proposal.model_dump_json(indent=2)}"
    )


def compact_review_prompt(
    *,
    user_prompt: str,
    proposal: PlanProposal,
    current_plan: SolutionPlan | None,
    extra_context: str = "",
) -> str:
    """Drop durable bookkeeping fields while retaining the complete bounded route."""
    proposal_payload = {
        "proposal_id": proposal.proposal_id,
        "base_revision": proposal.base_revision,
        "main_route": proposal.main_route,
        "route": proposal.route.model_dump(mode="json"),
        "rationale": proposal.rationale,
        "feedback": proposal.feedback,
        "source_phase": proposal.source_phase,
        "source_decision": proposal.source_decision,
    }
    plan_payload = (
        {
            "revision": current_plan.revision,
            "main_route": current_plan.main_route,
            "route": current_plan.route.model_dump(mode="json"),
        }
        if current_plan
        else None
    )
    return (
        "Review this bounded Progressive Solution Path update as Main Submitter 1. "
        "Preserve the user's objective. Return one JSON object matching "
        '{"decision":"approve|reject|followup","reasoning":"...",'
        '"main_route":null,"followup_route":null,"edits":[],"more_edits":false}. '
        "Use followup only when another pass is genuinely needed.\n\n"
        f"USER PROMPT:\n{user_prompt}\n\n"
        f"{extra_context}\n\n"
        f"CURRENT PLAN:\n{json.dumps(plan_payload, ensure_ascii=False)}\n\n"
        f"PROPOSAL:\n{json.dumps(proposal_payload, ensure_ascii=False)}"
    )


async def review_with_json_retry(
    *,
    prompt: str,
    call_completion: CompletionCall,
    extract_text: Callable[[Any], str],
    context_window: int,
    max_output_tokens: int,
    compact_prompt: str | None = None,
) -> ReviewDecision:
    """Run one bounded sanitized JSON repair without replaying raw model output."""

    max_input = max(1, context_window - max_output_tokens)
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > max_input:
        if compact_prompt is not None and count_tokens(compact_prompt) <= max_input:
            prompt = compact_prompt
            prompt_tokens = count_tokens(prompt)
        else:
            raise ValueError(
                "solution-path reviewer mandatory context exceeds the configured "
                f"input budget ({prompt_tokens} > {max_input} tokens); automatic "
                "bookkeeping compaction could not fit the complete bounded route"
            )
    messages = [{"role": "user", "content": prompt}]
    last_error = "invalid JSON response"
    for attempt in range(2):
        response = await call_completion(messages)
        raw = extract_text(response)
        try:
            parsed = parse_json(raw)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            if not isinstance(parsed, dict):
                raise ValueError("review response must be a JSON object")
            return ReviewDecision.model_validate(parsed)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt:
                break
            safe = sanitize_model_output_for_retry_context(raw, max_chars=1500)
            repair = (
                "Your prior response did not satisfy the required JSON schema. "
                f"Validation error: {last_error}. Return one corrected JSON object only."
            )
            retry_messages = [
                messages[0],
                {"role": "assistant", "content": safe},
                {"role": "user", "content": repair},
            ]
            if sum(count_tokens(item["content"]) for item in retry_messages) > max_input:
                retry_messages = [messages[0], {"role": "user", "content": repair}]
            messages = retry_messages
    raise ValueError(f"solution-path reviewer JSON retry failed: {last_error}")
