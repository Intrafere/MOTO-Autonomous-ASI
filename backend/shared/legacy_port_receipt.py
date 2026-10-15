"""Canonical legacy-port receipt helpers for proofs stored by released builds.

Released MOTO builds stored verified proofs before compile receipts, client
identity, or local novelty receipts existed. SyntheticLib ports those proofs
without a compile receipt, so the explicit migration marks each one with this
digest-bound receipt instead of inventing proof-time evidence.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, get_args

from backend.shared.models import LegacyPortReceipt, LocalNoveltyTier, ProofRecord

LEGACY_PORT_RECEIPT_VERSION = "moto-legacy-port-receipt-v1"

_LOCAL_RANKS = frozenset(get_args(LocalNoveltyTier))


def legacy_port_receipt_digest(payload: dict) -> str:
    normalized = dict(payload)
    normalized.pop("receipt_digest", None)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def legacy_source_rank(record: ProofRecord) -> Optional[str]:
    """Return the stored local rank, or None when it is not a known local rank.

    The independent tier wins because `duplicate_novel` is a display overlay,
    never a local rank.
    """
    for value in (record.independent_novelty_tier, record.novelty_tier):
        rank = str(value or "").strip().lower()
        if rank in _LOCAL_RANKS:
            return rank
    return None


def finalize_legacy_port_receipt(
    *,
    record: ProofRecord,
    source_local_rank: str,
    migrated_at: Optional[datetime] = None,
) -> LegacyPortReceipt:
    """Bind the minted identity and stored rank of a migrated legacy proof."""
    if record.canonical_locator is None or not record.canonical_locator_digest:
        raise ValueError("Canonical locator identity is required for a legacy-port receipt")
    payload = {
        "schema_version": LEGACY_PORT_RECEIPT_VERSION,
        "receipt_digest": "0" * 64,
        "migrated_at": migrated_at or datetime.now(timezone.utc),
        "prior_verification_evidence_status": "legacy_unrecorded",
        "source_local_rank": source_local_rank,
        "canonical_identity_version": record.canonical_identity_version,
        "canonical_theorem_statement_hash": record.canonical_theorem_statement_hash,
        "canonical_lean_code_hash": record.canonical_lean_code_hash,
        "canonical_locator_digest": record.canonical_locator_digest,
    }
    normalized = LegacyPortReceipt.model_validate(payload).model_dump(mode="json")
    normalized["receipt_digest"] = legacy_port_receipt_digest(normalized)
    return LegacyPortReceipt.model_validate(normalized)
