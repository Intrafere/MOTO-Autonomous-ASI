from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.shared.local_novelty_receipt import (
    finalize_local_novelty_receipt,
    local_novelty_receipt_digest,
)
from backend.shared.models import (
    CanonicalProofLocator,
    LocalNoveltyAssessmentProvenance,
    ProofRecord,
)
from backend.shared.proof_identity import canonical_proof_identity
from backend.shared.proof_locator import canonical_locator_digest


LEAN_CODE = "import Mathlib\n\ntheorem receipt_fixture : True := by trivial\n"


def _qualified_record(**updates) -> ProofRecord:
    identity = canonical_proof_identity("True", LEAN_CODE)
    locator = CanonicalProofLocator(
        locator_version="moto-proof-locator-v1",
        installation_id="install_fixture",
        root_binding_id="root_fixture",
        store_kind="manual",
        stable_store_id="store_fixture",
        run_id="run_fixture",
        proof_id="proof_001",
        canonical_identity_version=identity.version,
        canonical_theorem_statement_hash=identity.theorem_statement_hash,
        canonical_lean_code_hash=identity.lean_code_hash,
    )
    values = {
        "proof_id": "proof_001",
        "theorem_statement": "True",
        "source_type": "paper",
        "source_id": "paper_001",
        "run_id": "run_fixture",
        "lean_code": LEAN_CODE,
        "novel": True,
        "novelty_tier": "novel_variant",
        "independent_novelty_tier": "novel_variant",
        "canonical_identity_version": identity.version,
        "canonical_theorem_statement_hash": identity.theorem_statement_hash,
        "canonical_lean_code_hash": identity.lean_code_hash,
        "client_proof_id": "moto_proof_0123456789abcdef0123456789abcdef",
        "canonical_locator": locator,
        "canonical_locator_digest": canonical_locator_digest(locator),
    }
    values.update(updates)
    return ProofRecord(**values)


def test_receipt_digest_is_stable_and_bound_to_canonical_occurrence() -> None:
    record = _qualified_record()
    provenance = LocalNoveltyAssessmentProvenance(
        assessment_origin="validator_assessed",
        source_local_rank="novel_variant",
        assessed_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
    )

    receipt = finalize_local_novelty_receipt(
        record=record,
        provenance=provenance,
    )

    assert receipt.receipt_digest == local_novelty_receipt_digest(
        receipt.model_dump(mode="json")
    )
    assert receipt.canonical_locator_digest == record.canonical_locator_digest
    assert receipt.source_local_rank == "novel_variant"
    assert receipt.registration_state == "ordinary_occurrence"


def test_duplicate_overlay_preserves_independent_local_rank() -> None:
    record = _qualified_record(
        novelty_tier="duplicate_novel",
        exact_duplicate_proof_id="proof_prior",
        exact_duplicate_run_id="run_prior",
    )

    receipt = finalize_local_novelty_receipt(
        record=record,
        provenance=LocalNoveltyAssessmentProvenance(
            assessment_origin="caller_supplied",
            source_local_rank="novel_variant",
        ),
    )

    assert receipt.source_local_rank == "novel_variant"
    assert receipt.effective_display_tier == "duplicate_novel"
    assert receipt.registration_state == "cross_run_exact_duplicate_overlay"


def test_group_receipt_binds_claim_and_complete_member_set() -> None:
    record = _qualified_record(
        novelty_scope="group",
        novelty_group_claim="Together the declarations prove both branches.",
        novelty_group_proof_ids=["proof_prior"],
    )
    receipt = finalize_local_novelty_receipt(
        record=record,
        provenance=LocalNoveltyAssessmentProvenance(
            assessment_origin="validator_assessed",
            source_local_rank="mathematical_discovery",
            assessment_scope="group",
            group_claim="Together the declarations prove both branches.",
            group_proof_ids=["proof_prior"],
        ),
    )

    assert receipt.group_proof_ids == ["proof_prior", "proof_001"]
    assert len(receipt.group_claim_sha256) == 64


def test_local_rank_cannot_be_duplicate_overlay() -> None:
    with pytest.raises(Exception):
        LocalNoveltyAssessmentProvenance(
            assessment_origin="validator_assessed",
            source_local_rank="duplicate_novel",
        )
