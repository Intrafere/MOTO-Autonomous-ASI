"""Reusable policy-driven orchestration for proof-verification rounds.

The driver deliberately treats one proof round as an opaque callback. Candidate
identification, Lean execution, Phase A/B processing, registration, checkpoints,
and provider recovery remain owned by ``ProofVerificationStage`` and its caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional, Protocol, Tuple, Union


# Discovery sees at most this many of the newest round-memory entries.
ROUND_MEMORY_MAX_ENTRIES = 5

RoundResult = Any
RoundMemory = Tuple[str, ...]
RoundExecutor = Callable[
    [int, str, RoundMemory, str],
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

    @property
    def continues_after_empty_round(self) -> bool:
        """Whether a valid no-candidate round advances instead of completing."""

    def trigger_for_round(self, base_trigger: str, round_index: int) -> str:
        """Return the durable checkpoint trigger for one round."""


@dataclass(frozen=True)
class OneRoundPolicy:
    """Exactly one round, used by paper+proof, retry, and manual wrappers."""

    max_rounds: Optional[int] = 1
    holds_source_reservation: bool = False
    continues_after_empty_round: bool = False

    def trigger_for_round(self, base_trigger: str, round_index: int) -> str:
        return base_trigger


@dataclass(frozen=True)
class AutomaticMultiRoundPolicy:
    """Bounded automatic rounds that always run every configured round."""

    max_rounds: Optional[int] = 4
    holds_source_reservation: bool = True
    continues_after_empty_round: bool = True

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
    continues_after_empty_round: bool = False

    def trigger_for_round(self, base_trigger: str, round_index: int) -> str:
        if round_index <= 1:
            return base_trigger
        return f"{base_trigger}_round_{round_index}"


def normalize_round_memory(value: Union[str, Iterable[str], None]) -> RoundMemory:
    """Return round-memory entries, oldest first, from a list or a legacy string."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    return tuple(str(entry).strip() for entry in value if str(entry or "").strip())


def summarize_round_result(round_index: int, proof_result: Any) -> str:
    """Describe one round's proof attempts: each result and, if verified, its novelty."""

    if proof_result is None:
        return f"Round {round_index}: skipped."
    lines = [
        (
            f"Round {round_index}: {proof_result.verified_count}/"
            f"{proof_result.total_candidates} candidates verified, "
            f"{proof_result.novel_count} novel."
        )
    ]
    for attempt_result in list(getattr(proof_result, "results", []) or []):
        statement = " ".join(str(attempt_result.theorem_statement or "").split())
        if attempt_result.success:
            tier = str(getattr(attempt_result, "novelty_tier", "") or "").strip()
            rating = tier or (
                "novel (tier not recorded)"
                if getattr(attempt_result, "novel", False)
                else "not_novel"
            )
            lines.append(f"- Verified in Lean 4, novelty rating {rating}: {statement}")
        else:
            lines.append(f"- Failed to get a Lean 4 proof verified: {statement}")
    return "\n".join(lines)


def round_memory_entry(round_index: int, proof_result: Any) -> str:
    """Return the round-memory entry for a round, or "" when nothing was attempted."""

    if proof_result is None or not list(getattr(proof_result, "results", []) or []):
        return ""
    return summarize_round_result(round_index, proof_result)


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
        initial_round_memory: Iterable[str] = (),
        initial_round_index: int = 1,
    ) -> None:
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
        self.initial_round_memory = normalize_round_memory(initial_round_memory)
        self.initial_round_index = initial_round_index

    async def run(self) -> str:
        """Execute rounds and map per-round outcomes to the parent status."""

        reservation_token = ""
        if self.policy.holds_source_reservation:
            reservation_token = await self.reserve_source(self.source_type, self.source_id)

        # Round memory survives empty rounds and source resets: it records what
        # was already attempted and how each verified proof was rated.
        round_memory: list[str] = list(self.initial_round_memory)
        round_index = self.initial_round_index
        try:
            while self.policy.max_rounds is None or round_index <= self.policy.max_rounds:
                if self.should_stop():
                    return "stopped"
                round_trigger = self.policy.trigger_for_round(
                    self.base_trigger,
                    round_index,
                )
                prior_round_results = tuple(round_memory[-ROUND_MEMORY_MAX_ENTRIES:])
                round_status, proof_result = await self.execute_round(
                    round_index,
                    round_trigger,
                    prior_round_results,
                    reservation_token,
                )
                if round_status == "retry_same_round":
                    continue
                if round_status == "continue_empty":
                    round_index += 1
                    continue
                if round_status == "fatal_stop":
                    return "fatal_stop"
                if round_status == "stopped":
                    return "stopped"
                if round_status == "no_candidates_skipped":
                    if self.policy.continues_after_empty_round:
                        round_index += 1
                        continue
                    return "complete"
                if proof_result is None:
                    round_index += 1
                    continue
                if self.should_stop():
                    return "stopped"
                if getattr(proof_result, "had_error", False):
                    return "error_preserved"
                entry = round_memory_entry(round_index, proof_result)
                if entry:
                    round_memory.append(entry)
                    del round_memory[:-ROUND_MEMORY_MAX_ENTRIES]
                if (
                    getattr(proof_result, "total_candidates", 0) == 0
                    and not self.policy.continues_after_empty_round
                ):
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
