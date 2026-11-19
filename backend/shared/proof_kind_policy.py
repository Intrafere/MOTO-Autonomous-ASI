"""Which kinds of proof the shared proof stage may propose.

Single-shot proofs get the individual Lean attempt budget; master proofs are
solved as a scaffold of ``sorry`` obligations. The user picks either or both
under Allowed Outputs; the choice is a durable runtime setting that applies
from the next proof discovery round. Compiler rigor and LeanOJ do not use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from backend.shared.config import system_config
from backend.shared.models import ProofCandidate


@dataclass(frozen=True)
class ProofKindPolicy:
    single_shot: bool = True
    master: bool = True

    def __post_init__(self) -> None:
        if not (self.single_shot or self.master):
            raise ValueError("At least one proof kind must stay enabled.")

    @classmethod
    def from_config(cls) -> "ProofKindPolicy":
        single_shot = bool(getattr(system_config, "proof_single_shot_enabled", True))
        master = bool(getattr(system_config, "proof_master_proofs_enabled", True))
        if not (single_shot or master):
            return cls()
        return cls(single_shot=single_shot, master=master)

    def normalize(self, candidates: Iterable[ProofCandidate]) -> List[ProofCandidate]:
        """Route each candidate to an allowed kind, keeping allowed kinds unchanged."""
        normalized: List[ProofCandidate] = []
        for candidate in candidates:
            kind = candidate.proposal_kind
            if kind == "master_proof" and not self.master:
                kind = "individual"
            elif kind != "master_proof" and not self.single_shot:
                kind = "master_proof"
            normalized.append(
                candidate
                if kind == candidate.proposal_kind
                else candidate.model_copy(update={"proposal_kind": kind})
            )
        return normalized
