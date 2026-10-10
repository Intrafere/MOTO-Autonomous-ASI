from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

import backend.shared.syntheticlib_trade_eligibility as eligibility_module
from backend.shared.lean_source_evidence import (
    receipt_digest,
    source_bytes,
    source_sha256,
)
from backend.shared.models import (
    CanonicalProofLocator,
    LeanCompileReceipt,
    LeanImportEvidence,
    LocalNoveltyAssessmentProvenance,
    ProofRecord,
)
from backend.shared.proof_identity import (
    CANONICAL_PROOF_IDENTITY_VERSION,
    canonical_proof_identity,
)
from backend.shared.proof_locator import (
    canonical_locator_digest,
    new_client_proof_id,
)
from backend.shared.local_novelty_receipt import (
    finalize_local_novelty_receipt,
    local_novelty_receipt_digest,
)
from backend.shared.syntheticlib_trade_eligibility import (
    project_canonical_proof_store,
    project_installation_proof_stores,
)


LEAN_CODE = "import Mathlib\n\ntheorem trade_fixture : 1 + 1 = 2 := by norm_num\n"


def _write_valid_store(
    root: Path,
    *,
    proof_id: str = "proof_001",
    store_kind: str = "manual",
    stable_store_id: str = "store_fixture",
    run_id: str = "run_fixture",
    source_type: str = "paper",
    theorem_statement: str = "1 + 1 = 2",
    lean_code: str = LEAN_CODE,
    local_rank: str = "novel_variant",
    created_at: datetime | None = None,
    revision: int = 4,
) -> ProofRecord:
    root.mkdir(parents=True, exist_ok=True)
    LEAN_CODE = lean_code
    identity = canonical_proof_identity(theorem_statement, LEAN_CODE)
    locator = CanonicalProofLocator(
        locator_version="moto-proof-locator-v1",
        installation_id="install_fixture",
        root_binding_id="root_fixture",
        store_kind=store_kind,
        stable_store_id=stable_store_id,
        run_id=run_id,
        proof_id=proof_id,
        canonical_identity_version=CANONICAL_PROOF_IDENTITY_VERSION,
        canonical_theorem_statement_hash=identity.theorem_statement_hash,
        canonical_lean_code_hash=identity.lean_code_hash,
    )
    checked_name = f"proof_{proof_id}_checked.lean"
    receipt_payload = {
        "schema_version": "moto-lean-compile-receipt-v1",
        "receipt_digest": "0" * 64,
        "result": "accepted",
        "checker_mode": "subprocess",
        "verification_profile": "full_proof",
        "verification_profile_version": "moto-lean-verification-v1",
        "source_transformation_version": "moto-lean-source-v1",
        "source_transformations": [],
        "original_source_sha256": source_sha256(LEAN_CODE),
        "original_source_byte_length": len(source_bytes(LEAN_CODE)),
        "checked_source_sha256": source_sha256(LEAN_CODE),
        "checked_source_byte_length": len(source_bytes(LEAN_CODE)),
        "checked_source_artifact": checked_name,
        "lean_version": "Lean 4 fixture",
        "mathlib_revision": "mathlib_fixture",
        "moto_version": "fixture",
        "moto_build_commit": "fixture_commit",
        "moto_api_contract_version": "fixture_contract",
        "timeout_seconds": 60,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "explicit_imports": [
            LeanImportEvidence(
                module_name="Mathlib",
                ordinal=0,
                present_in_original_source=True,
            ).model_dump(mode="json")
        ],
        "migration_kind": "original_verification",
    }
    receipt = LeanCompileReceipt.model_validate(receipt_payload)
    canonical_receipt = receipt.model_dump(mode="json")
    canonical_receipt["receipt_digest"] = receipt_digest(canonical_receipt)
    receipt = LeanCompileReceipt.model_validate(canonical_receipt)
    record = ProofRecord(
        proof_id=proof_id,
        theorem_statement=theorem_statement,
        theorem_name="trade_fixture",
        source_type=source_type,
        source_id="private-source-id",
        source_title="private source title",
        run_id=run_id,
        user_prompt="private user prompt",
        lean_code=LEAN_CODE,
        novel=local_rank != "not_novel",
        novelty_tier=local_rank,
        independent_novelty_tier=local_rank,
        canonical_identity_version=CANONICAL_PROOF_IDENTITY_VERSION,
        canonical_theorem_statement_hash=identity.theorem_statement_hash,
        canonical_lean_code_hash=identity.lean_code_hash,
        client_proof_id=new_client_proof_id(),
        canonical_locator=locator,
        canonical_locator_digest=canonical_locator_digest(locator),
        compile_receipt=receipt,
        verification_evidence_status="complete",
        dependency_extraction_status="complete",
        **({"created_at": created_at} if created_at is not None else {}),
    )
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
    (root / f"proof_{proof_id}.json").write_text(
        json.dumps(record.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    (root / f"proof_{proof_id}_lean.lean").write_text(LEAN_CODE, encoding="utf-8")
    (root / checked_name).write_text(LEAN_CODE, encoding="utf-8")
    (root / "proof_set_revision.json").write_text(
        json.dumps({"proof_set_revision": revision}),
        encoding="utf-8",
    )
    return record


def test_valid_build_2_3_evidence_remains_unconditionally_ineligible(tmp_path: Path) -> None:
    _write_valid_store(tmp_path)

    projection = project_canonical_proof_store(tmp_path)

    assert projection.proof_set_revision == 4
    assert projection.stable_store_id == "store_fixture"
    assert projection.evidence_complete is True
    assert projection.eligible_to_offer is False
    assert projection.source_selection_policy_version == (
        "moto-trade-source-selection-v2"
    )
    assert projection.reason_codes == (
        "frozen_syntheticlib_contract_unpublished",
        "novelty_trade_mapping_unavailable",
    )
    assert len(projection.diagnostics) == 1
    result = projection.diagnostics[0]
    assert result.evidence_complete is True, result.reason_codes
    assert result.source_selected is True
    assert result.source_category == "manual"
    assert result.eligible_to_offer is False
    assert result.reason_codes == (
        "frozen_syntheticlib_contract_unpublished",
        "novelty_trade_mapping_unavailable",
    )


def test_projection_ignores_both_derived_indexes(tmp_path: Path) -> None:
    _write_valid_store(tmp_path)
    baseline = project_canonical_proof_store(tmp_path)
    (tmp_path / "proofs_index.json").write_bytes(b"not json")
    search_dir = tmp_path / "proof_search"
    search_dir.mkdir()
    (search_dir / "proof_search.sqlite").write_bytes(
        b"contradictory derived search data marked accepted and novel"
    )

    projected = project_canonical_proof_store(tmp_path)

    assert projected == baseline


def test_malformed_record_is_visible_and_privacy_safe(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "proof_set_revision.json").write_text(
        '{"proof_set_revision": 1}', encoding="utf-8"
    )
    (tmp_path / "proof_secret-name.json").write_text("{broken", encoding="utf-8")

    projection = project_canonical_proof_store(tmp_path)
    serialized = json.dumps(projection.to_dict())

    assert len(projection.diagnostics) == 1
    assert "canonical_record_invalid" in projection.diagnostics[0].reason_codes
    assert projection.diagnostics[0].eligible_to_offer is False
    assert "secret-name" not in serialized
    assert str(tmp_path) not in serialized


def test_tampered_source_receipt_and_imports_fail_closed(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path)
    tampered = LEAN_CODE.replace("import Mathlib", "import Mathlib.Data.Nat.Basic")
    (tmp_path / f"proof_{record.proof_id}_lean.lean").write_text(
        tampered, encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert result.evidence_complete is False
    assert result.eligible_to_offer is False
    assert "canonical_lean_source_record_mismatch" in result.reason_codes
    assert "compile_receipt_source_binding_mismatch" in result.reason_codes
    assert "compile_receipt_imports_mismatch" in result.reason_codes


def test_executable_placeholder_and_fake_declaration_are_rejected(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path)
    bad_source = (
        "import Mathlib\n"
        "axiom fake : False\n"
        "theorem bad : True := by sorry\n"
        '-- "sorry" and opaque mentioned in a string/comment do not matter\n'
    )
    payload = record.model_dump(mode="json")
    payload["lean_code"] = bad_source
    identity = canonical_proof_identity(payload["theorem_statement"], bad_source)
    payload["canonical_lean_code_hash"] = identity.lean_code_hash
    payload["canonical_locator"]["canonical_lean_code_hash"] = identity.lean_code_hash
    locator = CanonicalProofLocator.model_validate(payload["canonical_locator"])
    payload["canonical_locator_digest"] = canonical_locator_digest(locator)
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (tmp_path / f"proof_{record.proof_id}_lean.lean").write_text(
        bad_source, encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert "lean_placeholder_present" in result.reason_codes
    assert "forbidden_declaration_device" in result.reason_codes
    assert result.eligible_to_offer is False


def test_revision_and_hash_changes_change_projection_fingerprint(tmp_path: Path) -> None:
    _write_valid_store(tmp_path)
    first = project_canonical_proof_store(tmp_path)
    (tmp_path / "proof_set_revision.json").write_text(
        '{"proof_set_revision": 5}', encoding="utf-8"
    )

    second = project_canonical_proof_store(tmp_path)

    assert first.projection_fingerprint != second.projection_fingerprint
    assert second.proof_set_revision == 5


def test_duplicate_client_identity_fails_closed(tmp_path: Path) -> None:
    first = _write_valid_store(tmp_path, proof_id="proof_001")
    second = _write_valid_store(tmp_path, proof_id="proof_002")
    payload = second.model_dump(mode="json")
    payload["client_proof_id"] = first.client_proof_id
    (tmp_path / "proof_proof_002.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    projection = project_canonical_proof_store(tmp_path)

    assert len(projection.diagnostics) == 2
    assert all(
        "duplicate_client_proof_id" in item.reason_codes
        for item in projection.diagnostics
    )
    assert all(item.eligible_to_offer is False for item in projection.diagnostics)


def test_missing_revision_and_incomplete_dependencies_are_reason_coded(
    tmp_path: Path,
) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    payload["dependency_extraction_status"] = "partial"
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (tmp_path / "proof_set_revision.json").unlink()

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert "proof_set_revision_missing" in result.reason_codes
    assert "dependency_evidence_incomplete" in result.reason_codes
    assert result.evidence_complete is False


def test_empty_store_exposes_projection_level_revision_failure(tmp_path: Path) -> None:
    projection = project_canonical_proof_store(tmp_path)

    assert projection.diagnostics == ()
    assert projection.evidence_complete is False
    assert projection.eligible_to_offer is False
    assert "canonical_store_empty" in projection.reason_codes
    assert "proof_set_revision_missing" in projection.reason_codes


def test_missing_store_exposes_projection_level_failure(tmp_path: Path) -> None:
    projection = project_canonical_proof_store(tmp_path / "missing")

    assert projection.diagnostics == ()
    assert projection.evidence_complete is False
    assert "canonical_store_empty" in projection.reason_codes
    assert "canonical_store_missing" in projection.reason_codes
    assert "proof_set_revision_missing" in projection.reason_codes


def test_malformed_client_proof_id_is_rejected(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    payload["client_proof_id"] = "client_local_only"
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert result.evidence_complete is False
    assert "client_proof_id_malformed" in result.reason_codes


def test_orphan_canonical_artifacts_are_visible(tmp_path: Path) -> None:
    (tmp_path / "proof_set_revision.json").write_text(
        '{"proof_set_revision": 1}', encoding="utf-8"
    )
    (tmp_path / "proof_orphan_lean.lean").write_text(LEAN_CODE, encoding="utf-8")
    (tmp_path / "proof_orphan_checked.lean").write_text(LEAN_CODE, encoding="utf-8")

    projection = project_canonical_proof_store(tmp_path)
    reasons = {reason for item in projection.diagnostics for reason in item.reason_codes}

    assert "orphan_canonical_lean_source" in reasons
    assert "orphan_checked_source_artifact" in reasons
    assert projection.evidence_complete is False


def test_revision_change_during_projection_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_valid_store(tmp_path)
    original = eligibility_module._validate_record

    def mutate_revision(**kwargs):
        result = original(**kwargs)
        (tmp_path / "proof_set_revision.json").write_text(
            '{"proof_set_revision": 5}', encoding="utf-8"
        )
        return result

    monkeypatch.setattr(eligibility_module, "_validate_record", mutate_revision)
    projection = project_canonical_proof_store(tmp_path)

    assert projection.evidence_complete is False
    assert "proof_set_revision_changed_during_projection" in projection.reason_codes
    assert all(
        "proof_set_revision_changed_during_projection" in item.reason_codes
        for item in projection.diagnostics
    )


def test_sensitive_record_fields_never_enter_projection(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    secrets = [
        "TOP-SECRET-PROMPT",
        "C:\\Users\\private\\proof.lean",
        "/home/private/proof.lean",
        "provider-secret-model",
    ]
    payload.update(
        {
            "user_prompt": secrets[0],
            "source_id": secrets[1],
            "source_title": secrets[2],
            "novelty_reasoning": secrets[3],
            "verification_notes": "|".join(secrets),
            "dependency_extraction_detail": "|".join(secrets),
            "live_context_status": "pruned",
            "live_context_owner_run_id": "run_fixture",
            "live_context_pruned_by": "user",
            "live_context_prune_reason": "|".join(secrets),
        }
    )
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    serialized = json.dumps(project_canonical_proof_store(tmp_path).to_dict())

    assert all(secret not in serialized for secret in secrets)
    assert str(tmp_path) not in serialized


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="symlinks are not supported on this platform",
)
def test_symlinked_canonical_source_is_rejected(
    tmp_path: Path,
) -> None:
    record = _write_valid_store(tmp_path)
    source_path = tmp_path / f"proof_{record.proof_id}_lean.lean"
    target_path = tmp_path / "outside.lean"
    target_path.write_text(LEAN_CODE, encoding="utf-8")
    source_path.unlink()
    try:
        source_path.symlink_to(target_path)
    except OSError:
        pytest.skip("symlink creation is not permitted")

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert result.evidence_complete is False
    assert "canonical_lean_source_not_regular" in result.reason_codes


def test_dependency_identity_conflicts_fail_closed(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    locator = payload["canonical_locator"]
    digest = payload["canonical_locator_digest"]
    payload["portable_dependencies"] = [
        {
            "kind": "moto",
            "declaration_name": "Shared.dep",
            "canonical_locator": locator,
            "canonical_locator_digest": digest,
            "canonical_theorem_statement_hash": locator[
                "canonical_theorem_statement_hash"
            ],
            "canonical_lean_code_hash": locator["canonical_lean_code_hash"],
        },
        {
            "kind": "moto",
            "declaration_name": "Shared.dep",
            "module_name": "Mathlib.Forbidden",
            "canonical_locator": locator,
            "canonical_locator_digest": "f" * 64,
            "canonical_theorem_statement_hash": locator[
                "canonical_theorem_statement_hash"
            ],
            "canonical_lean_code_hash": locator["canonical_lean_code_hash"],
        },
    ]
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert "dependency_identity_conflict" in result.reason_codes
    assert "moto_dependency_contains_mathlib_identity" in result.reason_codes
    assert result.evidence_complete is False


def test_compile_receipt_transformation_contradiction_fails_closed(
    tmp_path: Path,
) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    payload["compile_receipt"]["source_transformations"] = [
        "normalize_lean_source"
    ]
    payload["compile_receipt"]["receipt_digest"] = receipt_digest(
        payload["compile_receipt"]
    )
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert "compile_receipt_transformations_unexpected" in result.reason_codes
    assert result.evidence_complete is False


def test_legacy_record_without_novelty_receipt_remains_readable_and_ineligible(
    tmp_path: Path,
) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    payload.pop("local_novelty_assessment_receipt", None)
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert "novelty_receipt_missing" in result.reason_codes
    assert "novelty_trade_mapping_unavailable" in result.reason_codes
    assert result.eligible_to_offer is False


def test_tampered_novelty_receipt_fails_closed(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    payload["local_novelty_assessment_receipt"][
        "canonical_lean_code_hash"
    ] = "f" * 64
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert "novelty_receipt_digest_mismatch" in result.reason_codes
    assert "novelty_receipt_canonical_binding_mismatch" in result.reason_codes
    assert result.eligible_to_offer is False


def test_duplicate_novelty_is_registration_overlay_not_local_rank(
    tmp_path: Path,
) -> None:
    record = _write_valid_store(tmp_path)
    payload = record.model_dump(mode="json")
    payload.update(
        {
            "novelty_tier": "duplicate_novel",
            "exact_duplicate_proof_id": "proof_elsewhere",
            "exact_duplicate_run_id": "run_elsewhere",
        }
    )
    receipt = payload["local_novelty_assessment_receipt"]
    receipt.update(
        {
            "registration_state": "cross_run_exact_duplicate_overlay",
            "effective_display_tier": "duplicate_novel",
            "exact_duplicate_proof_id": "proof_elsewhere",
            "exact_duplicate_run_id": "run_elsewhere",
        }
    )
    receipt["receipt_digest"] = local_novelty_receipt_digest(receipt)
    (tmp_path / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert result.evidence_complete is True, result.reason_codes
    assert result.source_selected is False
    assert "source_duplicate_overlay_occurrence" in result.reason_codes
    assert result.eligible_to_offer is False
    assert "novelty_trade_mapping_unavailable" in result.reason_codes


# --- Build 4B source-selection categories and Build 5 multi-store projection ---

OTHER_LEAN = "import Mathlib\n\ntheorem trade_other : 2 + 2 = 4 := by norm_num\n"


def _rewrite(root: Path, record: ProofRecord, **updates) -> None:
    payload = record.model_dump(mode="json")
    payload.update(updates)
    (root / f"proof_{record.proof_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("store_kind", "source_type", "store_role", "expected_category"),
    [
        ("manual", "paper", "active_manual", "active_manual"),
        ("manual", "brainstorm", "archived_manual", "archived_manual"),
        ("autonomous", "brainstorm", "autonomous_session", "autonomous"),
        ("autonomous", "leanoj_final", "autonomous_session", "leanoj_final"),
        ("autonomous_legacy", "leanoj_subproof", "autonomous_legacy", "leanoj_subproof"),
        ("autonomous_legacy", "paper", "autonomous_legacy", "autonomous_legacy"),
    ],
)
def test_included_source_categories_are_selected_but_never_eligible(
    tmp_path: Path,
    store_kind: str,
    source_type: str,
    store_role: str,
    expected_category: str,
) -> None:
    _write_valid_store(tmp_path, store_kind=store_kind, source_type=source_type)

    result = project_canonical_proof_store(tmp_path, store_role=store_role).diagnostics[0]

    assert result.source_category == expected_category
    assert result.source_selected is True, result.reason_codes
    assert result.evidence_complete is True
    assert result.eligible_to_offer is False


def test_live_context_pruned_occurrence_remains_selected(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path)
    _rewrite(
        tmp_path,
        record,
        live_context_status="pruned",
        live_context_owner_run_id="run_fixture",
        live_context_pruned_at=datetime.now(timezone.utc).isoformat(),
        live_context_pruned_by="user",
    )

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert result.source_selected is True, result.reason_codes
    assert result.eligible_to_offer is False


@pytest.mark.parametrize(
    ("kwargs", "updates", "expected_reason"),
    [
        ({}, {"artifact_purpose": "standalone_exact_duplicate_emphasis"}, "source_artifact_purpose_excluded"),
        ({"local_rank": "not_novel"}, {}, "source_local_rank_not_novel"),
        ({"store_kind": "shared_legacy"}, {}, "source_store_kind_unsupported"),
    ],
)
def test_excluded_source_categories_are_reason_coded(
    tmp_path: Path, kwargs: dict, updates: dict, expected_reason: str
) -> None:
    record = _write_valid_store(tmp_path, **kwargs)
    if updates:
        _rewrite(tmp_path, record, **updates)

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert expected_reason in result.reason_codes
    assert result.source_selected is False
    assert result.eligible_to_offer is False


def test_store_role_mismatch_is_excluded(tmp_path: Path) -> None:
    _write_valid_store(tmp_path, store_kind="manual")

    result = project_canonical_proof_store(
        tmp_path, store_role="autonomous_session"
    ).diagnostics[0]

    assert "source_store_role_mismatch" in result.reason_codes
    assert result.source_selected is False


def test_identical_content_resolves_to_earliest_occurrence(tmp_path: Path) -> None:
    early = _write_valid_store(
        tmp_path,
        proof_id="proof_001",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    late = _write_valid_store(
        tmp_path,
        proof_id="proof_002",
        created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    by_client = {
        item.client_proof_id: item
        for item in project_canonical_proof_store(tmp_path).diagnostics
    }

    assert by_client[early.client_proof_id].source_selected is True
    superseded = by_client[late.client_proof_id]
    assert superseded.source_selected is False
    assert superseded.evidence_complete is True
    assert "source_local_duplicate_superseded" in superseded.reason_codes


def _populate_installation(data_root: Path) -> dict[str, ProofRecord]:
    return {
        "active": _write_valid_store(
            data_root / "manual_proofs", stable_store_id="store_active"
        ),
        "archived": _write_valid_store(
            data_root / "manual_proof_runs" / "manual-run-1" / "proofs",
            stable_store_id="store_archived",
            lean_code=OTHER_LEAN,
            theorem_statement="2 + 2 = 4",
        ),
        "session": _write_valid_store(
            data_root / "auto_sessions" / "PRIVATE_PROMPT_session_2026" / "proofs",
            store_kind="autonomous",
            stable_store_id="store_session",
            source_type="leanoj_final",
            lean_code=LEAN_CODE.replace("trade_fixture", "trade_session"),
        ),
        "legacy": _write_valid_store(
            data_root / "proofs",
            store_kind="autonomous_legacy",
            stable_store_id="store_legacy",
            source_type="leanoj_subproof",
            lean_code=LEAN_CODE.replace("trade_fixture", "trade_legacy"),
        ),
    }


def test_installation_projection_covers_every_canonical_store(tmp_path: Path) -> None:
    records = _populate_installation(tmp_path)
    (tmp_path / "manual_proof_runs" / ".pending.tmp" / "proofs").mkdir(parents=True)

    projection = project_installation_proof_stores(tmp_path)

    assert projection.schema_version == "moto-trade-evidence-installation-projection-v1"
    assert projection.eligible_to_offer is False
    assert projection.evidence_complete is True, projection.reason_codes
    assert {store.store_role for store in projection.stores} == {
        "active_manual",
        "archived_manual",
        "autonomous_session",
        "autonomous_legacy",
    }
    assert len(projection.stores) == 4
    assert {item.client_proof_id for item in projection.diagnostics} == {
        record.client_proof_id for record in records.values()
    }
    assert all(item.source_selected for item in projection.diagnostics)
    assert all(item.eligible_to_offer is False for item in projection.diagnostics)
    assert projection.reason_codes == (
        "frozen_syntheticlib_contract_unpublished",
        "novelty_trade_mapping_unavailable",
    )


def test_installation_projection_never_emits_paths_or_session_names(
    tmp_path: Path,
) -> None:
    _populate_installation(tmp_path)

    serialized = json.dumps(project_installation_proof_stores(tmp_path).to_dict())

    assert "PRIVATE_PROMPT" not in serialized
    assert "manual-run-1" not in serialized
    assert str(tmp_path) not in serialized
    assert "private user prompt" not in serialized
    assert "private source title" not in serialized


def test_installation_projection_ignores_derived_indexes(tmp_path: Path) -> None:
    _populate_installation(tmp_path)
    baseline = project_installation_proof_stores(tmp_path)
    (tmp_path / "manual_proofs" / "proofs_index.json").write_bytes(b"not json")
    (tmp_path / "proof_search").mkdir()
    (tmp_path / "proof_search" / "proof_search.sqlite").write_bytes(b"stale index")

    assert project_installation_proof_stores(tmp_path) == baseline


def test_copied_occurrence_across_stores_fails_closed(tmp_path: Path) -> None:
    record = _write_valid_store(tmp_path / "manual_proofs", stable_store_id="store_a")
    copy_root = tmp_path / "manual_proof_runs" / "copied" / "proofs"
    copy_root.mkdir(parents=True)
    for path in (tmp_path / "manual_proofs").iterdir():
        (copy_root / path.name).write_bytes(path.read_bytes())

    projection = project_installation_proof_stores(tmp_path)

    copies = [
        item
        for item in projection.diagnostics
        if item.client_proof_id == record.client_proof_id
    ]
    assert len(copies) == 2
    assert all("cross_store_duplicate_occurrence" in item.reason_codes for item in copies)
    assert all("cross_store_stable_store_id_conflict" in item.reason_codes for item in copies)
    assert all(item.evidence_complete is False for item in copies)
    assert projection.evidence_complete is False
    assert "cross_store_duplicate_occurrence" in projection.reason_codes


def test_identical_content_across_stores_resolves_to_earliest(tmp_path: Path) -> None:
    early = _write_valid_store(
        tmp_path / "proofs",
        store_kind="autonomous_legacy",
        stable_store_id="store_legacy",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    late = _write_valid_store(
        tmp_path / "manual_proofs",
        stable_store_id="store_manual",
        run_id="run_other",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    by_client = {
        item.client_proof_id: item
        for item in project_installation_proof_stores(tmp_path).diagnostics
    }

    assert by_client[early.client_proof_id].source_selected is True
    assert by_client[late.client_proof_id].source_selected is False
    assert (
        "source_local_duplicate_superseded"
        in by_client[late.client_proof_id].reason_codes
    )


def test_empty_installation_is_complete_and_ineligible(tmp_path: Path) -> None:
    projection = project_installation_proof_stores(tmp_path)

    assert projection.stores == ()
    assert projection.diagnostics == ()
    assert projection.evidence_complete is True
    assert projection.eligible_to_offer is False


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="symlinks are not supported on this platform",
)
def test_symlinked_session_store_fails_closed_without_reading(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_valid_store(outside / "proofs", store_kind="autonomous")
    sessions = tmp_path / "data" / "auto_sessions"
    sessions.mkdir(parents=True)
    try:
        (sessions / "linked_session").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not permitted")

    projection = project_installation_proof_stores(tmp_path / "data")

    assert len(projection.stores) == 1
    assert projection.diagnostics == ()
    assert "canonical_store_not_regular" in projection.stores[0].reason_codes
    assert projection.evidence_complete is False


def _rewrite_lean_source(root: Path, record: ProofRecord, source: str) -> None:
    payload = record.model_dump(mode="json")
    payload["lean_code"] = source
    identity = canonical_proof_identity(payload["theorem_statement"], source)
    payload["canonical_lean_code_hash"] = identity.lean_code_hash
    payload["canonical_locator"]["canonical_lean_code_hash"] = identity.lean_code_hash
    locator = CanonicalProofLocator.model_validate(payload["canonical_locator"])
    payload["canonical_locator_digest"] = canonical_locator_digest(locator)
    (root / f"proof_{record.proof_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / f"proof_{record.proof_id}_lean.lean").write_text(source, encoding="utf-8")


@pytest.mark.parametrize(
    ("comment", "kind"),
    [
        (r"-- copied from C:\Users\someone\proofs\x.lean", "absolute_path"),
        ("-- see /home/someone/proofs/x.lean", "absolute_path"),
        ("-- contact someone@example.org", "email"),
        ("-- moto_proof_" + "c" * 32, "occurrence_identifier"),
        ("-- key sk-or-v1-" + "d" * 64, "credential_or_installation_identifier"),
        ("-- built on alicehost", "local_identity"),
    ],
)
def test_lean_source_private_data_fails_closed_without_echo(
    tmp_path: Path, monkeypatch, comment: str, kind: str
) -> None:
    import backend.shared.trade_data_boundary as boundary

    monkeypatch.setattr(boundary, "_local_identity_tokens", lambda: ("alicehost",))
    record = _write_valid_store(tmp_path)
    _rewrite_lean_source(tmp_path, record, LEAN_CODE + comment + "\n")

    result = project_canonical_proof_store(tmp_path).diagnostics[0]

    assert f"lean_source_sensitive_value:{kind}" in result.reason_codes
    assert result.evidence_complete is False
    assert result.eligible_to_offer is False
    rendered = json.dumps(result.to_dict(), ensure_ascii=False)
    sensitive_token = comment.split()[-1]
    assert sensitive_token not in rendered


def test_ordinary_lean_mathematics_is_not_flagged_as_private_data() -> None:
    from backend.shared.trade_data_boundary import trade_sensitive_value_kinds

    source = (
        "import Mathlib\nopen Nat Finset\n\n"
        "/-- sum of the first n odd numbers -/\n"
        "theorem sum_odd (n : ℕ) : ∑ i ∈ range n, (2 * i + 1) = n ^ 2 := by\n"
        "  induction n with\n  | zero => simp\n"
        "  | succ k ih => rw [sum_range_succ, ih]; ring\n"
        "theorem div_le (a b : ℕ) : a / b ≤ a := Nat.div_le_self a b\n"
        "example (x : ℝ) : x ∈ Set.Icc (0:ℝ) 1 → x ≤ 1 := fun h ↦ h.2\n"
        "example : (fun x : ℕ => x - 1) 0 = 0 := rfl\n"
        "theorem z : (7 : ℤ) % 3 = 1 := by decide\n"
        "theorem hex : 0x1f = 31 := rfl\n"
        "example : ∀ ε > 0, ∃ δ > 0, δ < ε := fun ε hε => ⟨ε/2, by positivity, by linarith⟩\n"
        "example (s : Set ℕ) : sᶜᶜ = s := compl_compl s -- trivial\n"
    )
    assert trade_sensitive_value_kinds(source) == ()
