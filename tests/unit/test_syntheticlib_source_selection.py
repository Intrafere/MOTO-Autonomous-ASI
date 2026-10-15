from __future__ import annotations

import pytest

from backend.shared.local_novelty_receipt import finalize_local_novelty_receipt
from backend.shared.models import (
    CanonicalProofLocator,
    LocalNoveltyAssessmentProvenance,
    ProofRecord,
)
from backend.shared.proof_identity import canonical_proof_identity
from backend.shared.proof_locator import canonical_locator_digest
from backend.shared.syntheticlib_source_selection import (
    SOURCE_SELECTION_POLICY_VERSION,
    evaluate_source_selection,
)


LEAN_CODE = "import Mathlib\n\ntheorem selection_fixture : True := by trivial\n"


def _record(
    *,
    store_kind: str = "manual",
    source_type: str = "paper",
    local_rank: str | None = "novel_variant",
    with_locator: bool = True,
    **updates,
) -> ProofRecord:
    identity = canonical_proof_identity("True", LEAN_CODE)
    locator = CanonicalProofLocator(
        locator_version="moto-proof-locator-v1",
        installation_id="install_fixture",
        root_binding_id="root_fixture",
        store_kind=store_kind,
        stable_store_id="store_fixture",
        run_id="run_fixture",
        proof_id="proof_001",
        canonical_identity_version=identity.version,
        canonical_theorem_statement_hash=identity.theorem_statement_hash,
        canonical_lean_code_hash=identity.lean_code_hash,
    )
    record = ProofRecord(
        proof_id="proof_001",
        theorem_statement="True",
        source_type=source_type,
        source_id="source_001",
        run_id="run_fixture",
        lean_code=LEAN_CODE,
        novelty_tier=local_rank or "not_novel",
        independent_novelty_tier=local_rank or "",
        canonical_identity_version=identity.version,
        canonical_theorem_statement_hash=identity.theorem_statement_hash,
        canonical_lean_code_hash=identity.lean_code_hash,
        client_proof_id="moto_proof_0123456789abcdef0123456789abcdef",
        canonical_locator=locator if with_locator else None,
        canonical_locator_digest=canonical_locator_digest(locator) if with_locator else "",
    )
    if local_rank is not None and with_locator:
        record = record.model_copy(
            update={
                "local_novelty_assessment_receipt": finalize_local_novelty_receipt(
                    record=record,
                    provenance=LocalNoveltyAssessmentProvenance(
                        assessment_origin="validator_assessed",
                        source_local_rank=local_rank,
                    ),
                )
            }
        )
    return record.model_copy(update=updates) if updates else record


def test_policy_version_is_frozen() -> None:
    assert SOURCE_SELECTION_POLICY_VERSION == "moto-trade-source-selection-v2"


@pytest.mark.parametrize(
    ("store_kind", "source_type", "store_role", "category"),
    [
        ("manual", "paper", None, "manual"),
        ("manual", "paper", "active_manual", "active_manual"),
        ("manual", "brainstorm", "archived_manual", "archived_manual"),
        ("autonomous", "paper", "autonomous_session", "autonomous"),
        ("autonomous", "leanoj_final", "autonomous_session", "leanoj_final"),
        ("autonomous_legacy", "leanoj_subproof", "autonomous_legacy", "leanoj_subproof"),
        ("autonomous_legacy", "brainstorm", None, "autonomous_legacy"),
    ],
)
def test_included_categories(store_kind, source_type, store_role, category) -> None:
    decision = evaluate_source_selection(
        _record(store_kind=store_kind, source_type=source_type),
        store_role=store_role,
    )

    assert decision.category == category
    assert decision.selected is True
    assert decision.reason_codes == ()


@pytest.mark.parametrize(
    ("kwargs", "store_role", "reason"),
    [
        ({"store_kind": "shared_legacy"}, None, "source_store_kind_unsupported"),
        ({"local_rank": "not_novel"}, None, "source_local_rank_not_novel"),
        ({"local_rank": None}, None, "source_local_rank_unavailable"),
        ({"with_locator": False}, None, "source_category_unresolved"),
        ({"artifact_purpose": "standalone_exact_duplicate_emphasis"}, None, "source_artifact_purpose_excluded"),
        ({"novelty_tier": "duplicate_novel"}, None, "source_duplicate_overlay_occurrence"),
        ({"exact_duplicate_proof_id": "proof_x"}, None, "source_duplicate_overlay_occurrence"),
        ({"store_kind": "manual"}, "autonomous_session", "source_store_role_mismatch"),
        ({"store_kind": "manual", "source_type": "leanoj_final"}, None, "source_store_role_mismatch"),
    ],
)
def test_excluded_categories(kwargs, store_role, reason) -> None:
    decision = evaluate_source_selection(_record(**kwargs), store_role=store_role)

    assert decision.selected is False
    assert reason in decision.reason_codes


def test_live_context_pruning_does_not_exclude() -> None:
    record = _record()
    pruned = ProofRecord.model_validate(
        {
            **record.model_dump(),
            "live_context_status": "pruned",
            "live_context_owner_run_id": "run_fixture",
            "live_context_pruned_at": "2026-10-08T00:00:00+00:00",
            "live_context_pruned_by": "automatic_proof_pruning",
            "live_context_prune_reason": "superseded in live context",
        }
    )

    assert evaluate_source_selection(pruned).selected is True
