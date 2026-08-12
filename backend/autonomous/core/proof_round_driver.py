"""Reusable policy-driven orchestration for proof-verification rounds.

The driver deliberately treats one proof round as an opaque callback. Candidate
identification, Lean execution, Phase A/B processing, registration, checkpoints,
and provider recovery remain owned by ``ProofVerificationStage`` and its caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol, Tuple


RoundResult = Any
RoundExecutor = Callable[
    [int, str, str, str],
    Awaitable[Tuple[str, Optional[RoundResult]]],
]
ReserveSource = Callable[[str, str], Awaitable[str]]
ReleaseSource = Callable[[str, str, str], Awaitable[None]]
ShouldStop = Callable[[], bool]


class ProofRoundPolicy(Protocol):
    """Policy contract consumed by :class:`ProofRoundDriver`."""

    @property
    def max_rounds(self) -> Optional[int]:
        """Maximum rounds, or ``None`` for an explicitly continuous policy."""

    @property
    def holds_source_reservation(self) -> bool:
        """Whether one token-owned source reservation spans all rounds."""

    def trigger_for_round(self, base_trigger: str, round_index: int) -> str:
        """Return the durable checkpoint trigger for one round."""


@dataclass(frozen=True)
class OneRoundPolicy:
    """Exactly one round, used by paper+proof, retry, and manual wrappers."""

    max_rounds: Optional[int] = 1
    holds_source_reservation: bool = False

    def trigger_for_round(self, base_trigger: str, round_index: int) -> str:
        return base_trigger


@dataclass(frozen=True)
class AutomaticMultiRoundPolicy:
    """Bounded automatic rounds with distinct follow-up checkpoint triggers."""

    max_rounds: Optional[int] = 4
    holds_source_reservation: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds is None or self.max_rounds < 2:
            raise ValueError("AutomaticMultiRoundPolicy requires at least two rounds")

    def trigger_for_round(self, base_trigger: str, round_index: int) -> str:
        if round_index <= 1:
            return base_trigger
        return f"{base_trigger}_round_{round_index}"


@dataclass(frozen=True)
class ContinuousPruningPolicy:
    """Explicit unbounded policy for operator-controlled continuous proof runs."""

    max_rounds: Optional[int] = None
    holds_source_reservation: bool = True

    def trigger_for_round(self, base_trigger: str, round_index: int) -> str:
        if round_index <= 1:
            return base_trigger
        return f"{base_trigger}_round_{round_index}"


def summarize_round_result(round_index: int, proof_result: Any) -> str:
    """Build the bounded prior-round context used by later identification."""

    if proof_result is None:
        return f"Round {round_index}: skipped."
    lines = [
        (
            f"Round {round_index}: {proof_result.verified_count}/"
            f"{proof_result.total_candidates} candidates verified, "
            f"{proof_result.novel_count} novel."
        )
    ]
    for attempt_result in list(getattr(proof_result, "results", []) or [])[:5]:
        status = "verified" if attempt_result.success else "failed"
        lines.append(f"- {status}: {attempt_result.theorem_statement[:220]}")
    return "\n".join(lines)


class ProofRoundDriver:
    """Drive proof rounds while preserving the stage as the unit of execution."""

    def __init__(
        self,
        *,
        policy: ProofRoundPolicy,
        source_type: str,
        source_id: str,
        base_trigger: str,
        execute_round: RoundExecutor,
        should_stop: ShouldStop,
        reserve_source: ReserveSource,
        release_source: ReleaseSource,
        prior_summary_limit: int = 3,
        initial_round_index: int = 1,
    ) -> None:
        if prior_summary_limit < 1:
            raise ValueError("prior_summary_limit must be positive")
        if initial_round_index < 1:
            raise ValueError("initial_round_index must be positive")
        self.policy = policy
        self.source_type = source_type
        self.source_id = source_id
        self.base_trigger = base_trigger
        self.execute_round = execute_round
        self.should_stop = should_stop
        self.reserve_source = reserve_source
        self.release_source = release_source
        self.prior_summary_limit = prior_summary_limit
        self.initial_round_index = initial_round_index

    async def run(self) -> str:
        """Execute rounds and map per-round outcomes to the parent status."""

        reservation_token = ""
        if self.policy.holds_source_reservation:
            reservation_token = await self.reserve_source(self.source_type, self.source_id)

        prior_round_summaries: list[str] = []
        round_index = self.initial_round_index
        try:
            while self.policy.max_rounds is None or round_index <= self.policy.max_rounds:
                if self.should_stop():
                    return "stopped"
                round_trigger = self.policy.trigger_for_round(
                    self.base_trigger,
                    round_index,
                )
                prior_round_results = "\n".join(
                    prior_round_summaries[-self.prior_summary_limit :]
                )
                round_status, proof_result = await self.execute_round(
                    round_index,
                    round_trigger,
                    prior_round_results,
                    reservation_token,
                )
                if round_status == "retry_same_round":
                    continue
                if round_status == "continue_reset":
                    prior_round_summaries.clear()
                    round_index += 1
                    continue
                if round_status == "completed_reset":
                    prior_round_summaries.clear()
                if round_status == "fatal_stop":
                    return "fatal_stop"
                if round_status == "stopped":
                    return "stopped"
                if round_status == "no_candidates_skipped":
                    return "complete"
                if round_status == "deferred":
                    return "complete"
                if proof_result is None:
                    round_index += 1
                    continue
                if self.should_stop():
                    return "stopped"
                if getattr(proof_result, "had_error", False):
                    return "error_preserved"
                prior_round_summaries.append(
                    summarize_round_result(round_index, proof_result)
                )
                if getattr(proof_result, "total_candidates", 0) == 0:
                    return "complete"
                round_index += 1
            return "complete"
        finally:
            if reservation_token:
                await self.release_source(
                    self.source_type,
                    self.source_id,
                    reservation_token,
                )
