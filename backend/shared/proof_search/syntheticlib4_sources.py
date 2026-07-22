"""SyntheticLib4 fixture/snapshot normalization for unified proof search."""
from __future__ import annotations

import hashlib
from typing import Any

from backend.shared.proof_search.models import UnifiedProofSearchRecord
from backend.shared.proof_identity import canonical_proof_identity
from backend.shared.syntheticlib4_client import SyntheticLib4Client, syntheticlib4_client


def normalize_syntheticlib4_record(
    record: dict[str, Any],
    *,
    release_id: str,
    channel: str = "stable",
) -> UnifiedProofSearchRecord:
    """Convert one SyntheticLib4 proof record into the shared search model."""
    fingerprint = str(record.get("fingerprint", "")).strip()
    theorem_statement = str(record.get("theorem_statement", "")).strip()
    lean_code = str(record.get("lean_code", "") or "")
    statement_hash = str(record.get("theorem_statement_hash", "")).strip() or _sha256_text(
        theorem_statement
    )
    lean_hash = str(record.get("lean_code_hash", "")).strip() or (
        _sha256_text(lean_code) if lean_code else ""
    )
    module = str(record.get("module", "") or "")
    source_path = str(record.get("source_path", "") or "")
    internal_identity = canonical_proof_identity(theorem_statement, lean_code)

    return UnifiedProofSearchRecord(
        search_id=f"syntheticlib4:{fingerprint}",
        corpus="syntheticlib4",
        corpus_scope=channel,
        source_kind="verified_proof",
        proof_id=fingerprint,
        external_fingerprint=fingerprint,
        release_id=release_id,
        source_type="syntheticlib4_snapshot",
        source_id=release_id,
        source_title=source_path or module or "SyntheticLib4",
        display_title=str(record.get("display_title", "") or record.get("theorem_name", "")),
        theorem_name=str(record.get("theorem_name", "") or ""),
        theorem_statement=theorem_statement,
        informal_statement=str(record.get("informal_statement", "") or ""),
        proof_description=str(record.get("proof_description", "") or ""),
        formal_sketch=str(record.get("proof_description", "") or ""),
        lean_code=lean_code,
        lean_code_hash=lean_hash,
        theorem_statement_hash=statement_hash,
        canonical_identity_version=internal_identity.version,
        canonical_lean_code_hash=internal_identity.lean_code_hash if lean_code else "",
        canonical_theorem_statement_hash=internal_identity.theorem_statement_hash,
        imports=_string_list(record.get("imports")),
        dependency_names=_string_list(record.get("dependency_names")),
        topic_tags=_string_list(record.get("topic_tags")),
        domain_tags=_string_list(record.get("domain_tags")),
        module=module,
        source_path=source_path,
        novelty_tier=str(record.get("novelty_rank", "") or ""),
        novelty_reasoning=(
            f"SyntheticLib4 novelty confidence: {record.get('novelty_confidence', 'unknown')}"
        ),
        verified=True,
        created_at=str(record.get("created_at", "") or ""),
        canonical_uri=f"syntheticlib4://{release_id}/{fingerprint}",
        metadata={
            "validation_record_id": record.get("validation_record_id", ""),
            "line_range": record.get("line_range", {}),
            "license_terms_id": record.get("license_terms_id", ""),
            "release_membership": record.get("release_membership", ""),
            "hydration_url": record.get("hydration_url"),
            "canonical_identity_version": internal_identity.version,
            "canonical_theorem_statement_hash": internal_identity.theorem_statement_hash,
            "canonical_lean_code_hash": internal_identity.lean_code_hash if lean_code else "",
        },
    )


def load_syntheticlib4_fixture_records(
    client: SyntheticLib4Client | None = None,
) -> list[UnifiedProofSearchRecord]:
    """Load SyntheticLib4 fixture records for offline development and tests."""
    active_client = client or syntheticlib4_client
    manifest = active_client.get_release_manifest()
    release_id = str(manifest.get("release_id", "") or "fixture")
    channel = str(manifest.get("channel", "") or "stable")
    return [
        normalize_syntheticlib4_record(record, release_id=release_id, channel=channel)
        for record in active_client.load_proof_metadata()
    ]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

