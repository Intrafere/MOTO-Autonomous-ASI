"""Canonical local novelty-assessment receipt helpers."""
from __future__ import annotations

import hashlib
import json

from backend.shared.models import (
    LocalNoveltyAssessmentProvenance,
    LocalNoveltyAssessmentReceipt,
    ProofRecord,
)

LOCAL_NOVELTY_RECEIPT_VERSION = "moto-local-novelty-receipt-v1"


def local_novelty_receipt_digest(payload: dict) -> str:
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


def finalize_local_novelty_receipt(
    *,
    record: ProofRecord,
    provenance: LocalNoveltyAssessmentProvenance,
) -> LocalNoveltyAssessmentReceipt:
    """Bind event-time novelty provenance to a qualified canonical occurrence."""
    if record.canonical_locator is None or not record.canonical_locator_digest:
        raise ValueError("Canonical locator identity is required for novelty receipt")
    group_ids = list(
        dict.fromkeys(
            [
                *provenance.group_proof_ids,
                *(
                    [record.proof_id]
                    if provenance.assessment_scope == "group"
                    else []
                ),
            ]
        )
    )
    group_claim_sha256 = (
        hashlib.sha256(provenance.group_claim.encode("utf-8")).hexdigest()
        if provenance.assessment_scope == "group"
        else ""
    )
    duplicate = bool(
        record.exact_duplicate_proof_id and record.exact_duplicate_run_id
    )
    payload = {
        "schema_version": LOCAL_NOVELTY_RECEIPT_VERSION,
        "receipt_digest": "0" * 64,
        "local_rank_schema_version": provenance.local_rank_schema_version,
        "assessment_method_version": provenance.assessment_method_version,
        "assessment_origin": provenance.assessment_origin,
        "source_local_rank": provenance.source_local_rank,
        "assessment_scope": provenance.assessment_scope,
        "group_claim_sha256": group_claim_sha256,
        "group_proof_ids": group_ids,
        "registration_state": (
            "cross_run_exact_duplicate_overlay"
            if duplicate
            else "ordinary_occurrence"
        ),
        "effective_display_tier": record.novelty_tier,
        "assessed_at": provenance.assessed_at,
        "canonical_identity_version": record.canonical_identity_version,
        "canonical_theorem_statement_hash": (
            record.canonical_theorem_statement_hash
        ),
        "canonical_lean_code_hash": record.canonical_lean_code_hash,
        "canonical_locator_digest": record.canonical_locator_digest,
        "exact_duplicate_proof_id": (
            record.exact_duplicate_proof_id if duplicate else ""
        ),
        "exact_duplicate_run_id": (
            record.exact_duplicate_run_id if duplicate else ""
        ),
    }
    normalized = LocalNoveltyAssessmentReceipt.model_validate(payload)
    canonical_payload = normalized.model_dump(mode="json")
    canonical_payload["receipt_digest"] = local_novelty_receipt_digest(
        canonical_payload
    )
    return LocalNoveltyAssessmentReceipt.model_validate(canonical_payload)
