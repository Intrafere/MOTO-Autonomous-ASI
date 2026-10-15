"""Pure helpers for fitting rolling feedback into model prompts.

Durable feedback stores remain authoritative.  These helpers operate on copied,
model-visible projections only. Whole entries are never split; only the tool
diagnostics inside an entry (Lean output) are bounded.
"""

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Iterable, Tuple, TypeVar

from backend.shared.utils import count_tokens


T = TypeVar("T")

# Lean can print every hypothesis of every unsolved goal, so one failed attempt's
# raw output can exceed a whole model context window. Prompts show a bounded,
# priority-ordered prefix of each attempt's output; durable feedback keeps it all.
LEAN_FEEDBACK_PROMPT_CHAR_LIMIT = 30_000
LEAN_GOAL_STATES_PROMPT_CHAR_LIMIT = 10_000
_LEAN_ERROR_LINE_RE = re.compile(r"\berror(?:\([^)]*\))?:")


def _leading_lines(text: str, limit: int) -> tuple[str, list[str]]:
    """Return the leading whole lines within ``limit`` chars and the omitted lines.

    A first line longer than ``limit`` is cut so some diagnostics always remain.
    """
    lines = text.splitlines()
    kept: list[str] = []
    used = 0
    for index, line in enumerate(lines):
        cost = len(line) + (1 if kept else 0)
        if used + cost > limit:
            if not kept:
                return line[:limit], lines[index:]
            return "\n".join(kept), lines[index:]
        kept.append(line)
        used += cost
    return "\n".join(kept), []


def _omission_note(omitted: list[str]) -> str:
    error_lines = sum(1 for line in omitted if _LEAN_ERROR_LINE_RE.search(line))
    detail = f", including {error_lines} error line(s)" if error_lines else ""
    return f"[Shortened to fit the prompt: {len(omitted)} more line(s) omitted{detail}.]"


def project_lean_feedback(error_text: str, goal_states: str) -> tuple[str, str]:
    """Bound one attempt's Lean feedback and goal states for a model prompt.

    Goal-state lines already shown in the feedback are not repeated. Returns
    ``(feedback, goal_states)``; an empty goal-state result means none remain.
    """
    feedback, omitted = _leading_lines(error_text or "", LEAN_FEEDBACK_PROMPT_CHAR_LIMIT)
    if omitted:
        feedback = f"{feedback}\n{_omission_note(omitted)}"
    shown = {line.strip() for line in feedback.splitlines() if line.strip()}
    remaining = "\n".join(
        line
        for line in (goal_states or "").splitlines()
        if line.strip() and line.strip() not in shown
    )
    goals, goals_omitted = _leading_lines(remaining, LEAN_GOAL_STATES_PROMPT_CHAR_LIMIT)
    if goals_omitted:
        goals = f"{goals}\n{_omission_note(goals_omitted)}"
    return feedback, goals


@dataclass(frozen=True)
class FeedbackFitResult(Generic[T]):
    """The largest newest feedback suffix whose complete prompt was evaluated."""

    prompt: str
    retained_entries: Tuple[T, ...]
    removed_entries: int
    prompt_tokens: int
    available_tokens: int

    @property
    def fits(self) -> bool:
        return self.prompt_tokens <= self.available_tokens


def fit_prompt_with_feedback(
    entries: Iterable[T],
    *,
    build_prompt: Callable[[Tuple[T, ...]], str],
    available_tokens: int,
    max_entries: int = 5,
) -> FeedbackFitResult[T]:
    """Keep the largest newest whole-entry suffix that fits the full prompt.

    When feedback exists, the newest entry is always retained even if the
    resulting prompt remains too large.  The caller then uses its normal typed
    context-overflow path.
    """

    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")
    if available_tokens < 0:
        raise ValueError("available_tokens must be non-negative")

    all_entries = tuple(entries)
    retained = all_entries[-max_entries:]
    initial_count = len(retained)

    if not retained:
        prompt = build_prompt(())
        return FeedbackFitResult(
            prompt=prompt,
            retained_entries=(),
            removed_entries=0,
            prompt_tokens=count_tokens(prompt),
            available_tokens=available_tokens,
        )

    while True:
        prompt = build_prompt(retained)
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens <= available_tokens or len(retained) == 1:
            return FeedbackFitResult(
                prompt=prompt,
                retained_entries=retained,
                removed_entries=initial_count - len(retained),
                prompt_tokens=prompt_tokens,
                available_tokens=available_tokens,
            )
        retained = retained[1:]


async def fit_prompt_with_feedback_async(
    entries: Iterable[T],
    *,
    build_prompt: Callable[[Tuple[T, ...]], Awaitable[str]],
    available_tokens: int,
    max_entries: int = 5,
) -> FeedbackFitResult[T]:
    """Async equivalent for prompt builders that perform context allocation."""

    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")
    if available_tokens < 0:
        raise ValueError("available_tokens must be non-negative")

    all_entries = tuple(entries)
    retained = all_entries[-max_entries:]
    initial_count = len(retained)

    if not retained:
        prompt = await build_prompt(())
        return FeedbackFitResult(
            prompt=prompt,
            retained_entries=(),
            removed_entries=0,
            prompt_tokens=count_tokens(prompt),
            available_tokens=available_tokens,
        )

    while True:
        prompt = await build_prompt(retained)
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens <= available_tokens or len(retained) == 1:
            return FeedbackFitResult(
                prompt=prompt,
                retained_entries=retained,
                removed_entries=initial_count - len(retained),
                prompt_tokens=prompt_tokens,
                available_tokens=available_tokens,
            )
        retained = retained[1:]
