"""Pure helpers for fitting rolling feedback into model prompts.

Durable feedback stores remain authoritative.  These helpers operate on copied,
model-visible projections only and never split an individual feedback entry.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Iterable, Tuple, TypeVar

from backend.shared.utils import count_tokens


T = TypeVar("T")


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
