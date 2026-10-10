"""Frozen MOTO-local source-selection policy for future SyntheticLib proof trade.

This policy only decides which canonical occurrence *categories* MOTO would ever
consider offering. It never makes a proof eligible: the SyntheticLib rank
mapping and frozen service contract remain separate permanent blockers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from backend.shared.models import ProofRecord


SOURCE_SELECTION_POLICY_VERSION = "moto-trade-source-selection-v2"

CanonicalStoreRole = Literal[
    "active_manual",
    "archived_manual",
    "autonomous_session",
    "autonomous_legacy",
]

SourceCategory = Literal[
    "active_manual",
    "archived_manual",
    "manual",
    "autonomous",
    "autonomous_legacy",
    "leanoj_final",
    "leanoj_subproof",
    "unsupported_store",
    "unresolved",
]

_ROLE_STORE_KINDS: dict[str, str] = {
    "active_manual": "manual",
    "archived_manual": "manual",
    "autonomous_session": "autonomous",
    "autonomous_legacy": "autonomous_legacy",
}
_LEANOJ_SOURCE_TYPES = {"leanoj_final", "leanoj_subproof"}
DUPLICATE_OVERLAY_TIER = "duplicate_novel"


@dataclass(frozen=True)
class SourceSelectionDecision:
    """Category and local inclusion verdict for one canonical occurrence."""

    category: SourceCategory
    selected: bool
    reason_codes: tuple[str, ...]


def evaluate_source_selection(
    record: ProofRecord,
    *,
    store_role: Optional[CanonicalStoreRole] = None,
) -> SourceSelectionDecision:
    """Apply the frozen v1 policy to one validated canonical record.

    Included categories: active/archived manual, Autonomous session and legacy
    stores, LeanOJ final proofs and subproofs, including live-context-pruned
    occurrences. Excluded: nonstandard artifact purposes, ``not_novel`` local
    ranks, cross-run exact-duplicate overlay occurrences (which resolve to their
    original canonical occurrence), unsupported stores, and occurrences whose
    category cannot be resolved from persisted identity.
    """
    reasons: list[str] = []
    locator = record.canonical_locator
    category: SourceCategory
    if locator is None:
        category = "unresolved"
        reasons.append("source_category_unresolved")
    else:
        store_kind = locator.store_kind
        if store_role is not None and _ROLE_STORE_KINDS.get(store_role) != store_kind:
            reasons.append("source_store_role_mismatch")
        if store_kind == "shared_legacy":
            category = "unsupported_store"
            reasons.append("source_store_kind_unsupported")
        elif record.source_type in _LEANOJ_SOURCE_TYPES:
            category = record.source_type  # type: ignore[assignment]
            if store_kind == "manual":
                reasons.append("source_store_role_mismatch")
        elif store_kind == "manual":
            if store_role in ("active_manual", "archived_manual"):
                category = store_role
            else:
                category = "manual"
        elif store_kind == "autonomous":
            category = "autonomous"
        else:
            category = "autonomous_legacy"

    if record.artifact_purpose != "verified_occurrence":
        reasons.append("source_artifact_purpose_excluded")

    # v2: a legacy-port occurrence carries its stored rank in the legacy-port
    # receipt because released builds never wrote a local novelty receipt.
    receipt = record.local_novelty_assessment_receipt
    legacy_receipt = record.legacy_port_receipt
    if receipt is not None:
        local_rank = receipt.source_local_rank
    elif legacy_receipt is not None:
        local_rank = legacy_receipt.source_local_rank
    else:
        local_rank = ""
    if not local_rank:
        reasons.append("source_local_rank_unavailable")
    elif local_rank == "not_novel":
        reasons.append("source_local_rank_not_novel")

    if (
        record.novelty_tier == DUPLICATE_OVERLAY_TIER
        or record.exact_duplicate_proof_id
        or record.exact_duplicate_run_id
        or (
            receipt is not None
            and receipt.registration_state == "cross_run_exact_duplicate_overlay"
        )
    ):
        reasons.append("source_duplicate_overlay_occurrence")

    ordered = tuple(sorted(set(reasons)))
    return SourceSelectionDecision(
        category=category,
        selected=not ordered,
        reason_codes=ordered,
    )
