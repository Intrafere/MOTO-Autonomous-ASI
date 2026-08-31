"""Pure deterministic evidence projection for semantic proof pruning."""
from __future__ import annotations

import hashlib
import json
from backend.shared.models import ProofRecord
from backend.shared.utils import count_tokens

EVIDENCE_POLICY_VERSION = "proof-pruning-semantic-evidence-v1"


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def dependency_fingerprint(record: ProofRecord) -> str:
    dependencies = sorted(
        (str(item.kind), str(item.name or ""), str(item.source_ref or ""))
        for item in record.dependencies
    )
    return _digest(
        {
            "status": record.dependency_extraction_status,
            "dependencies": dependencies,
        }
    )


def descriptor_fingerprint(record: ProofRecord) -> str:
    identity = {
        "theorem_hash": record.canonical_theorem_statement_hash,
        "lean_hash": record.canonical_lean_code_hash,
    }
    return _digest(
        {
            "proof_id": record.proof_id,
            **identity,
            "dependency_fingerprint": dependency_fingerprint(record),
            "source_type": record.source_type,
            "source_id": record.source_id,
            "theorem_name": record.theorem_name,
            "theorem_statement": record.theorem_statement,
            "formal_sketch": record.formal_sketch,
            "novelty_tier": record.novelty_tier,
            "novelty_reasoning": record.novelty_reasoning,
            "independent_novelty_tier": record.independent_novelty_tier,
            "independent_novelty_reasoning": record.independent_novelty_reasoning,
            "source_title": record.source_title,
            "verification_notes": record.verification_notes,
            "lean_code": record.lean_code,
        }
    )


def role_in_objective(record: ProofRecord) -> str:
    parts = [
        str(record.source_title or "").strip(),
        str(record.formal_sketch or "").strip(),
        str(record.verification_notes or "").strip(),
        str(record.novelty_reasoning or "").strip(),
    ]
    return "\n".join(part for part in parts if part)[:2000]


def estimated_context_tokens(record: ProofRecord) -> int:
    return count_tokens(
        "\n".join(
            (
                record.theorem_name,
                record.theorem_statement,
                record.formal_sketch,
                record.novelty_reasoning,
                record.lean_code,
            )
        )
    )


def evidence_fingerprint(
    *,
    whole_set: list[dict],
    descriptor_fingerprints: list[str],
) -> str:
    return _digest(
        {
            "policy": EVIDENCE_POLICY_VERSION,
            "whole_set": whole_set,
            "descriptors": descriptor_fingerprints,
        }
    )
