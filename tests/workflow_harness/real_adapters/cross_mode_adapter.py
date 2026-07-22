"""Test-only helpers for serialized cross-mode start races."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import HTTPException


StartCall = Callable[[], Awaitable[dict]]


@dataclass(frozen=True)
class StartOutcome:
    name: str
    started: bool
    status_code: int | None = None
    detail: str = ""


async def race_starts(calls: dict[str, StartCall]) -> tuple[StartOutcome, ...]:
    """Run route starts concurrently and normalize their observable outcomes."""

    async def run(name: str, call: StartCall) -> StartOutcome:
        try:
            await call()
            return StartOutcome(name=name, started=True)
        except HTTPException as exc:
            return StartOutcome(
                name=name,
                started=False,
                status_code=exc.status_code,
                detail=str(exc.detail),
            )

    outcomes = await asyncio.gather(*(run(name, call) for name, call in calls.items()))
    return tuple(outcomes)


def assert_single_race_winner(outcomes: tuple[StartOutcome, ...]) -> StartOutcome:
    winners = [outcome for outcome in outcomes if outcome.started]
    if len(winners) != 1:
        raise AssertionError(f"Expected one start winner, observed {outcomes!r}")
    losers = [outcome for outcome in outcomes if not outcome.started]
    if any(outcome.status_code != 400 for outcome in losers):
        raise AssertionError(f"Expected conflict HTTP 400 for all losers, observed {outcomes!r}")
    return winners[0]
