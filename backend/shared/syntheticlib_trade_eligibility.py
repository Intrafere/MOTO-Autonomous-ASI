"""Fail-closed local evidence projection for future SyntheticLib proof trade.

This module deliberately does not implement trade eligibility.  Until the
SyntheticLib rank mapping and frozen SyntheticLib contract exist, every
canonical occurrence remains ineligible.  The projection audits MOTO's persisted
identity, compile, and local novelty evidence and applies the frozen MOTO
source-selection policy across every canonical store, without consulting either
proof-search index.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from backend.shared.lean_proof_integrity import (
    extract_primary_lean_theorem,
    find_introduced_declaration_devices,
    strip_lean_comments_and_strings,
)
from backend.shared.lean_source_evidence import (
    contains_private_path,
    parse_explicit_import_modules,
    receipt_digest,
    source_bytes,
    source_sha256,
)
from backend.shared.legacy_port_receipt import (
    legacy_port_receipt_digest,
    legacy_source_rank,
)
from backend.shared.local_novelty_receipt import local_novelty_receipt_digest
from backend.shared.models import PortableProofDependency, ProofRecord
from backend.shared.proof_identity import (
    CANONICAL_PROOF_IDENTITY_VERSION,
    canonical_proof_identity,
)
from backend.shared.proof_locator import (
    PROOF_LOCATOR_VERSION,
    canonical_locator_digest,
    is_valid_client_proof_id,
)
from backend.shared.trade_data_boundary import trade_sensitive_value_kinds
from backend.shared.syntheticlib_source_selection import (
    SOURCE_SELECTION_POLICY_VERSION,
    CanonicalStoreRole,
    evaluate_source_selection,
)


PROJECTION_SCHEMA_VERSION = "moto-trade-evidence-projection-v2"
INSTALLATION_PROJECTION_SCHEMA_VERSION = (
    "moto-trade-evidence-installation-projection-v1"
)
_RECORD_FILENAME_RE = re.compile(r"^proof_(?!set_revision(?:\.|$))(.+)\.json$")
_LEAN_FILENAME_RE = re.compile(r"^proof_(.+)_lean\.lean$")
_CHECKED_FILENAME_RE = re.compile(r"^proof_(.+)_checked\.lean$")
_PLACEHOLDER_RE = re.compile(r"\b(?:sorry|admit)\b")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_PERMANENT_BLOCKERS = (
    "novelty_trade_mapping_unavailable",
    "frozen_syntheticlib_contract_unpublished",
)
_DUPLICATE_SUPERSEDED_REASON = "source_local_duplicate_superseded"


@dataclass(frozen=True)
class TradeEvidenceDiagnostic:
    """Privacy-safe result for one canonical occurrence or malformed record."""

    record_key: str
    store_key: str
    client_proof_id: str
    canonical_locator_digest: str
    canonical_theorem_statement_hash: str
    canonical_lean_code_hash: str
    created_at: str
    proof_set_revision: int | None
    source_selection_policy_version: str
    source_category: str
    source_selected: bool
    evidence_complete: bool
    eligible_to_offer: bool
    reason_codes: tuple[str, ...]
    # "full": proof-time compile + novelty receipts; "legacy_port": a migrated
    # released-build proof ported without them; "none": neither is valid.
    evidence_class: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


@dataclass(frozen=True)
class TradeEvidenceProjection:
    """Deterministic metadata-only projection for one canonical proof store."""

    schema_version: str
    source_selection_policy_version: str
    proof_set_revision: int | None
    stable_store_id: str
    projection_fingerprint: str
    evidence_complete: bool
    eligible_to_offer: bool
    reason_codes: tuple[str, ...]
    diagnostics: tuple[TradeEvidenceDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_selection_policy_version": self.source_selection_policy_version,
            "proof_set_revision": self.proof_set_revision,
            "stable_store_id": self.stable_store_id,
            "projection_fingerprint": self.projection_fingerprint,
            "evidence_complete": self.evidence_complete,
            "eligible_to_offer": self.eligible_to_offer,
            "reason_codes": list(self.reason_codes),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _ordered_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(reason) for reason in reasons if reason}))


def _safe_record_key(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:20]


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unresolved_diagnostic(
    *,
    record_key: str,
    revision: int | None,
    reasons: Iterable[str],
) -> TradeEvidenceDiagnostic:
    return TradeEvidenceDiagnostic(
        record_key=record_key,
        store_key="",
        client_proof_id="",
        canonical_locator_digest="",
        canonical_theorem_statement_hash="",
        canonical_lean_code_hash="",
        created_at="",
        proof_set_revision=revision,
        source_selection_policy_version=SOURCE_SELECTION_POLICY_VERSION,
        source_category="unresolved",
        source_selected=False,
        evidence_complete=False,
        eligible_to_offer=False,
        reason_codes=_ordered_reasons((*reasons, *_PERMANENT_BLOCKERS)),
    )


def _with_reasons(
    item: TradeEvidenceDiagnostic,
    reasons: Iterable[str],
    *,
    evidence_failure: bool,
) -> TradeEvidenceDiagnostic:
    """Add reasons; any added reason clears selection, evidence failures also completeness."""
    reasons = tuple(reasons)
    if not reasons:
        return item
    return replace(
        item,
        evidence_complete=item.evidence_complete and not evidence_failure,
        source_selected=False,
        reason_codes=_ordered_reasons((*item.reason_codes, *reasons)),
    )


def _resolve_duplicate_content(
    diagnostics: Iterable[TradeEvidenceDiagnostic],
) -> list[TradeEvidenceDiagnostic]:
    """Keep only the earliest selected evidence-complete copy of identical proofs.

    Identical canonical theorem+Lean content may exist as independent ordinary
    occurrences (for example before duplicate overlay detection existed).  The
    deterministic canonical occurrence is the earliest by creation time, then by
    opaque client proof ID; later copies are excluded from selection.
    """
    items = list(diagnostics)
    winners: dict[tuple[str, str], tuple[str, str]] = {}
    for item in items:
        if not (
            item.source_selected
            and item.evidence_complete
            and item.canonical_theorem_statement_hash
            and item.canonical_lean_code_hash
        ):
            continue
        key = (item.canonical_theorem_statement_hash, item.canonical_lean_code_hash)
        rank = (item.created_at, item.client_proof_id)
        if key not in winners or rank < winners[key]:
            winners[key] = rank
    resolved: list[TradeEvidenceDiagnostic] = []
    for item in items:
        key = (item.canonical_theorem_statement_hash, item.canonical_lean_code_hash)
        if (
            item.source_selected
            and key in winners
            and (item.created_at, item.client_proof_id) != winners[key]
        ):
            item = _with_reasons(
                item, (_DUPLICATE_SUPERSEDED_REASON,), evidence_failure=False
            )
        resolved.append(item)
    return resolved


def _load_revision(store_root: Path) -> tuple[int | None, list[str]]:
    path = store_root / "proof_set_revision.json"
    try:
        if path.is_symlink() or not path.is_file():
            if path.exists() or path.is_symlink():
                return None, ["proof_set_revision_not_regular"]
            return None, ["proof_set_revision_missing"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("revision payload is not an object")
        value = payload.get("proof_set_revision")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("revision is not a nonnegative integer")
        return value, []
    except FileNotFoundError:
        return None, ["proof_set_revision_missing"]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None, ["proof_set_revision_invalid"]


def _dependency_has_private_value(dependency: PortableProofDependency) -> bool:
    def values(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from values(child)
        elif isinstance(value, list):
            for child in value:
                yield from values(child)

    return any(
        contains_private_path(value)
        for value in values(dependency.model_dump(mode="json"))
    )


def _validate_dependencies(
    record: ProofRecord,
    reasons: list[str],
    *,
    require_complete: bool = True,
) -> None:
    # Legacy ports never recorded a proof-time Mathlib revision, so their
    # portable dependencies cannot be truthfully completed; any present ones
    # are still validated below.
    if require_complete and record.dependency_extraction_status != "complete":
        reasons.append("dependency_evidence_incomplete")
        return
    dependency_identities: dict[tuple[str, str], tuple[Any, ...]] = {}
    for dependency in record.portable_dependencies:
        if _dependency_has_private_value(dependency):
            reasons.append("dependency_contains_private_path")
        if dependency.kind == "mathlib":
            if not dependency.module_name or not dependency.mathlib_revision:
                reasons.append("mathlib_dependency_incomplete")
            if any(
                (
                    dependency.canonical_locator_digest,
                    dependency.canonical_theorem_statement_hash,
                    dependency.canonical_lean_code_hash,
                )
            ):
                reasons.append("mathlib_dependency_contains_moto_identity")
            identity = (
                dependency.module_name,
                dependency.mathlib_revision,
            )
            key = (dependency.kind, dependency.declaration_name)
            if key in dependency_identities and dependency_identities[key] != identity:
                reasons.append("dependency_identity_conflict")
            dependency_identities[key] = identity
            continue
        if dependency.module_name or dependency.mathlib_revision:
            reasons.append("moto_dependency_contains_mathlib_identity")
        locator = dependency.canonical_locator
        if locator is None:
            reasons.append("moto_dependency_locator_missing")
            continue
        if locator.locator_version != PROOF_LOCATOR_VERSION:
            reasons.append("moto_dependency_locator_unsupported")
        if canonical_locator_digest(locator) != dependency.canonical_locator_digest:
            reasons.append("moto_dependency_locator_digest_mismatch")
        if (
            locator.canonical_theorem_statement_hash
            != dependency.canonical_theorem_statement_hash
            or locator.canonical_lean_code_hash
            != dependency.canonical_lean_code_hash
        ):
            reasons.append("moto_dependency_hash_mismatch")
        identity = (
            dependency.canonical_locator_digest,
            dependency.canonical_theorem_statement_hash,
            dependency.canonical_lean_code_hash,
        )
        key = (dependency.kind, dependency.declaration_name)
        if key in dependency_identities and dependency_identities[key] != identity:
            reasons.append("dependency_identity_conflict")
        dependency_identities[key] = identity


def _validate_record(
    *,
    record: ProofRecord,
    store_root: Path,
    filename_proof_id: str,
    revision: int | None,
    revision_reasons: list[str],
    store_role: CanonicalStoreRole | None,
) -> tuple[TradeEvidenceDiagnostic, str]:
    reasons = list(revision_reasons)
    identity = canonical_proof_identity(record.theorem_statement, record.lean_code)
    locator = record.canonical_locator

    if not record.client_proof_id:
        reasons.append("client_proof_id_missing")
    elif not is_valid_client_proof_id(record.client_proof_id):
        reasons.append("client_proof_id_malformed")
    if record.proof_id != filename_proof_id:
        reasons.append("record_filename_identity_mismatch")
    if record.canonical_identity_version != CANONICAL_PROOF_IDENTITY_VERSION:
        reasons.append("canonical_identity_version_unsupported")
    if (
        record.canonical_theorem_statement_hash != identity.theorem_statement_hash
        or record.canonical_lean_code_hash != identity.lean_code_hash
    ):
        reasons.append("canonical_identity_hash_mismatch")

    stable_store_id = ""
    if locator is None:
        reasons.append("canonical_locator_missing")
    else:
        stable_store_id = locator.stable_store_id
        if locator.locator_version != PROOF_LOCATOR_VERSION:
            reasons.append("canonical_locator_version_unsupported")
        if locator.proof_id != record.proof_id or locator.run_id != record.run_id:
            reasons.append("canonical_locator_occurrence_mismatch")
        if locator.canonical_identity_version != record.canonical_identity_version:
            reasons.append("canonical_locator_identity_version_mismatch")
        if (
            locator.canonical_theorem_statement_hash
            != record.canonical_theorem_statement_hash
            or locator.canonical_lean_code_hash != record.canonical_lean_code_hash
        ):
            reasons.append("canonical_locator_hash_mismatch")
        if canonical_locator_digest(locator) != record.canonical_locator_digest:
            reasons.append("canonical_locator_digest_mismatch")

    legacy_port = False
    legacy_receipt = record.legacy_port_receipt
    if legacy_receipt is not None:
        if (
            record.compile_receipt is not None
            or record.local_novelty_assessment_receipt is not None
            or record.verification_evidence_status != "legacy_unrecorded"
        ):
            reasons.append("legacy_port_receipt_conflict")
        elif (
            legacy_receipt.receipt_digest
            != legacy_port_receipt_digest(legacy_receipt.model_dump(mode="json"))
            or legacy_receipt.canonical_identity_version
            != record.canonical_identity_version
            or legacy_receipt.canonical_theorem_statement_hash
            != record.canonical_theorem_statement_hash
            or legacy_receipt.canonical_lean_code_hash != record.canonical_lean_code_hash
            or legacy_receipt.canonical_locator_digest != record.canonical_locator_digest
            or legacy_receipt.source_local_rank != legacy_source_rank(record)
        ):
            reasons.append("legacy_port_receipt_invalid")
        else:
            legacy_port = True

    novelty_receipt = record.local_novelty_assessment_receipt
    if novelty_receipt is None:
        if not legacy_port:
            reasons.append("novelty_receipt_missing")
    else:
        receipt_payload = novelty_receipt.model_dump(mode="json")
        if novelty_receipt.receipt_digest != local_novelty_receipt_digest(
            receipt_payload
        ):
            reasons.append("novelty_receipt_digest_mismatch")
        if (
            novelty_receipt.canonical_identity_version
            != record.canonical_identity_version
            or novelty_receipt.canonical_theorem_statement_hash
            != record.canonical_theorem_statement_hash
            or novelty_receipt.canonical_lean_code_hash
            != record.canonical_lean_code_hash
            or novelty_receipt.canonical_locator_digest
            != record.canonical_locator_digest
        ):
            reasons.append("novelty_receipt_canonical_binding_mismatch")
        if (
            novelty_receipt.source_local_rank
            != record.independent_novelty_tier
        ):
            reasons.append("novelty_receipt_local_rank_mismatch")
        if novelty_receipt.assessment_scope != record.novelty_scope:
            reasons.append("novelty_receipt_scope_mismatch")
        if novelty_receipt.effective_display_tier != record.novelty_tier:
            reasons.append("novelty_receipt_display_tier_mismatch")
        if novelty_receipt.assessment_scope == "group":
            expected_group_ids = list(
                dict.fromkeys(
                    [*record.novelty_group_proof_ids, record.proof_id]
                )
            )
            expected_claim_hash = hashlib.sha256(
                record.novelty_group_claim.encode("utf-8")
            ).hexdigest()
            if (
                novelty_receipt.group_proof_ids != expected_group_ids
                or novelty_receipt.group_claim_sha256 != expected_claim_hash
            ):
                reasons.append("novelty_receipt_group_binding_mismatch")
        duplicate = bool(
            record.exact_duplicate_proof_id and record.exact_duplicate_run_id
        )
        expected_registration = (
            "cross_run_exact_duplicate_overlay"
            if duplicate
            else "ordinary_occurrence"
        )
        if (
            novelty_receipt.registration_state != expected_registration
            or novelty_receipt.exact_duplicate_proof_id
            != (record.exact_duplicate_proof_id if duplicate else "")
            or novelty_receipt.exact_duplicate_run_id
            != (record.exact_duplicate_run_id if duplicate else "")
        ):
            reasons.append("novelty_receipt_registration_state_mismatch")

    lean_path = store_root / f"proof_{filename_proof_id}_lean.lean"
    checked_path = store_root / f"proof_{filename_proof_id}_checked.lean"
    lean_source: str | None = None
    if lean_path.is_symlink() or (lean_path.exists() and not lean_path.is_file()):
        reasons.append("canonical_lean_source_not_regular")
    elif not lean_path.exists():
        reasons.append("canonical_lean_source_missing")
    else:
        try:
            lean_source = lean_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            reasons.append("canonical_lean_source_unreadable")

    if lean_source is not None:
        if lean_source != record.lean_code:
            reasons.append("canonical_lean_source_record_mismatch")
        masked = strip_lean_comments_and_strings(lean_source)
        if not lean_source.strip():
            reasons.append("canonical_lean_source_empty")
        if _PLACEHOLDER_RE.search(masked):
            reasons.append("lean_placeholder_present")
        if find_introduced_declaration_devices(lean_source):
            reasons.append("forbidden_declaration_device")
        if not extract_primary_lean_theorem(lean_source)[1]:
            reasons.append("lean_theorem_declaration_missing")
        # The proof body is the only content a trade may carry, so it must not
        # embed credentials, private paths, emails, local identity, or trade IDs.
        # The checked-source artifact must equal this text (checked below).
        for kind in trade_sensitive_value_kinds(lean_source):
            reasons.append(f"lean_source_sensitive_value:{kind}")

    receipt = record.compile_receipt
    if not legacy_port:
        if record.verification_evidence_status != "complete":
            reasons.append("verification_evidence_incomplete")
        if receipt is None:
            reasons.append("compile_receipt_missing")
    if receipt is not None:
        receipt_payload = receipt.model_dump(mode="json")
        if receipt.receipt_digest != receipt_digest(receipt_payload):
            reasons.append("compile_receipt_digest_mismatch")
        source_changed = (
            receipt.original_source_sha256 != receipt.checked_source_sha256
            or receipt.original_source_byte_length != receipt.checked_source_byte_length
        )
        if source_changed and not receipt.source_transformations:
            reasons.append("compile_receipt_transformations_missing")
        if not source_changed and receipt.source_transformations:
            reasons.append("compile_receipt_transformations_unexpected")
        if lean_source is not None:
            if (
                receipt.checked_source_sha256 != source_sha256(lean_source)
                or receipt.checked_source_byte_length != len(source_bytes(lean_source))
            ):
                reasons.append("compile_receipt_source_binding_mismatch")
            parsed_imports = parse_explicit_import_modules(lean_source)
            stored_imports = [
                evidence.module_name
                for evidence in sorted(receipt.explicit_imports, key=lambda item: item.ordinal)
            ]
            ordinals = [evidence.ordinal for evidence in receipt.explicit_imports]
            if stored_imports != parsed_imports or ordinals != list(range(len(ordinals))):
                reasons.append("compile_receipt_imports_mismatch")
        expected_artifact = checked_path.name
        if receipt.checked_source_artifact != expected_artifact:
            reasons.append("checked_source_artifact_name_mismatch")
        if checked_path.is_symlink() or (
            checked_path.exists() and not checked_path.is_file()
        ):
            reasons.append("checked_source_artifact_not_regular")
        elif not checked_path.exists():
            reasons.append("checked_source_artifact_missing")
        else:
            try:
                checked_source = checked_path.read_text(encoding="utf-8")
                if lean_source is not None and checked_source != lean_source:
                    reasons.append("checked_source_artifact_mismatch")
            except (OSError, UnicodeError):
                reasons.append("checked_source_artifact_unreadable")

    _validate_dependencies(record, reasons, require_complete=not legacy_port)
    local_reasons = _ordered_reasons(reasons)
    evidence_complete = not local_reasons
    selection = evaluate_source_selection(record, store_role=store_role)
    final_reasons = _ordered_reasons(
        (*local_reasons, *selection.reason_codes, *_PERMANENT_BLOCKERS)
    )
    return (
        TradeEvidenceDiagnostic(
            record_key=_safe_record_key(f"proof_{filename_proof_id}.json"),
            store_key="",
            client_proof_id=record.client_proof_id
            if is_valid_client_proof_id(record.client_proof_id)
            else "",
            canonical_locator_digest=record.canonical_locator_digest
            if _HASH_RE.fullmatch(record.canonical_locator_digest or "")
            else "",
            canonical_theorem_statement_hash=record.canonical_theorem_statement_hash
            if _HASH_RE.fullmatch(record.canonical_theorem_statement_hash or "")
            else "",
            canonical_lean_code_hash=record.canonical_lean_code_hash
            if _HASH_RE.fullmatch(record.canonical_lean_code_hash or "")
            else "",
            created_at=record.created_at.isoformat(),
            proof_set_revision=revision,
            source_selection_policy_version=SOURCE_SELECTION_POLICY_VERSION,
            source_category=selection.category,
            source_selected=evidence_complete and selection.selected,
            evidence_complete=evidence_complete,
            eligible_to_offer=False,
            reason_codes=final_reasons,
            evidence_class=(
                "legacy_port"
                if legacy_port
                else "full"
                if receipt is not None and novelty_receipt is not None
                else "none"
            ),
        ),
        stable_store_id,
    )


def project_canonical_proof_store(
    store_root: Path | str,
    *,
    store_role: CanonicalStoreRole | None = None,
) -> TradeEvidenceProjection:
    """Audit one canonical proof store without using either derived proof index.

    ``store_role`` identifies where the installation enumerator found the store;
    when given, persisted locator store kinds must agree with it.
    """
    root = Path(store_root)
    projection_reasons: list[str] = []
    root_is_unsafe = root.is_symlink() or (root.exists() and not root.is_dir())
    if root_is_unsafe:
        projection_reasons.append("canonical_store_not_regular")
    elif not root.exists():
        projection_reasons.append("canonical_store_missing")

    revision, revision_reasons = (
        (None, []) if root_is_unsafe else _load_revision(root)
    )
    projection_reasons.extend(revision_reasons)
    diagnostics: list[TradeEvidenceDiagnostic] = []
    stable_store_ids: set[str] = set()
    client_ids: dict[str, int] = {}
    locator_digests: dict[str, int] = {}

    try:
        entries = (
            [] if root_is_unsafe else sorted(root.iterdir(), key=lambda path: path.name)
        )
    except OSError:
        entries = []
        projection_reasons.append("canonical_store_unreadable")

    canonical_entries = [
        path
        for path in entries
        if path.name == "proof_set_revision.json"
        or _RECORD_FILENAME_RE.fullmatch(path.name)
        or _LEAN_FILENAME_RE.fullmatch(path.name)
        or _CHECKED_FILENAME_RE.fullmatch(path.name)
    ]
    folded_names: dict[str, list[str]] = {}
    for path in canonical_entries:
        folded_names.setdefault(path.name.casefold(), []).append(path.name)
    if any(len(set(names)) > 1 for names in folded_names.values()):
        projection_reasons.append("canonical_artifact_case_collision")

    record_ids = {
        match.group(1)
        for path in entries
        if (match := _RECORD_FILENAME_RE.fullmatch(path.name)) is not None
    }
    lean_ids = {
        match.group(1)
        for path in entries
        if (match := _LEAN_FILENAME_RE.fullmatch(path.name)) is not None
    }
    checked_ids = {
        match.group(1)
        for path in entries
        if (match := _CHECKED_FILENAME_RE.fullmatch(path.name)) is not None
    }
    if not record_ids:
        projection_reasons.append("canonical_store_empty")

    def append_artifact_diagnostic(path: Path, reason: str) -> None:
        diagnostics.append(
            _unresolved_diagnostic(
                record_key=_safe_record_key(path.name),
                revision=revision,
                reasons=(*projection_reasons, reason),
            )
        )

    for proof_id in sorted(lean_ids - record_ids):
        append_artifact_diagnostic(
            root / f"proof_{proof_id}_lean.lean", "orphan_canonical_lean_source"
        )
    for proof_id in sorted(checked_ids - record_ids):
        append_artifact_diagnostic(
            root / f"proof_{proof_id}_checked.lean", "orphan_checked_source_artifact"
        )

    for path in entries:
        match = _RECORD_FILENAME_RE.fullmatch(path.name)
        if not match:
            continue
        if path.is_symlink() or not path.is_file():
            append_artifact_diagnostic(path, "canonical_record_not_regular")
            continue
        filename_proof_id = match.group(1)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("record payload is not an object")
            record = ProofRecord.model_validate(payload)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            TypeError,
        ):
            append_artifact_diagnostic(path, "canonical_record_invalid")
            continue
        diagnostic, stable_store_id = _validate_record(
            record=record,
            store_root=root,
            filename_proof_id=filename_proof_id,
            revision=revision,
            revision_reasons=list(projection_reasons),
            store_role=store_role,
        )
        diagnostics.append(diagnostic)
        if stable_store_id:
            stable_store_ids.add(stable_store_id)
        if diagnostic.client_proof_id:
            client_ids[diagnostic.client_proof_id] = client_ids.get(
                diagnostic.client_proof_id, 0
            ) + 1
        if diagnostic.canonical_locator_digest:
            locator_digests[diagnostic.canonical_locator_digest] = locator_digests.get(
                diagnostic.canonical_locator_digest, 0
            ) + 1

    store_reasons: list[str] = []
    if len(stable_store_ids) > 1:
        store_reasons.append("conflicting_stable_store_ids")
    duplicate_clients = {value for value, count in client_ids.items() if count > 1}
    duplicate_locators = {
        value for value, count in locator_digests.items() if count > 1
    }
    if store_reasons or duplicate_clients or duplicate_locators:
        diagnostics = [
            _with_reasons(
                item,
                (
                    *store_reasons,
                    *(
                        ("duplicate_client_proof_id",)
                        if item.client_proof_id in duplicate_clients
                        else ()
                    ),
                    *(
                        ("duplicate_canonical_locator",)
                        if item.canonical_locator_digest in duplicate_locators
                        else ()
                    ),
                ),
                evidence_failure=True,
            )
            for item in diagnostics
        ]

    final_revision, final_revision_reasons = (
        (None, []) if root_is_unsafe else _load_revision(root)
    )
    if (
        final_revision != revision
        or _ordered_reasons(final_revision_reasons)
        != _ordered_reasons(revision_reasons)
    ):
        projection_reasons.append("proof_set_revision_changed_during_projection")
        diagnostics = [
            _with_reasons(
                item,
                ("proof_set_revision_changed_during_projection",),
                evidence_failure=True,
            )
            for item in diagnostics
        ]

    diagnostics = _resolve_duplicate_content(diagnostics)
    diagnostics.sort(key=lambda item: item.record_key)
    stable_store_id = next(iter(stable_store_ids)) if len(stable_store_ids) == 1 else ""
    projection_reason_codes = _ordered_reasons(
        (
            *projection_reasons,
            *store_reasons,
            *_PERMANENT_BLOCKERS,
        )
    )
    projection_complete = not _ordered_reasons(
        (*projection_reasons, *store_reasons)
    ) and all(item.evidence_complete for item in diagnostics)
    fingerprint_payload = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "source_selection_policy_version": SOURCE_SELECTION_POLICY_VERSION,
        "proof_set_revision": revision,
        "stable_store_id": stable_store_id,
        "evidence_complete": projection_complete,
        "eligible_to_offer": False,
        "reason_codes": projection_reason_codes,
        "diagnostics": [item.to_dict() for item in diagnostics],
    }
    fingerprint = _fingerprint(fingerprint_payload)
    return TradeEvidenceProjection(
        schema_version=PROJECTION_SCHEMA_VERSION,
        source_selection_policy_version=SOURCE_SELECTION_POLICY_VERSION,
        proof_set_revision=revision,
        stable_store_id=stable_store_id,
        projection_fingerprint=fingerprint,
        evidence_complete=projection_complete,
        eligible_to_offer=False,
        reason_codes=projection_reason_codes,
        diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True)
class CanonicalStoreSummary:
    """Privacy-safe summary of one canonical store; never exposes its path."""

    store_key: str
    store_role: str
    stable_store_id: str
    proof_set_revision: int | None
    projection_fingerprint: str
    record_count: int
    evidence_complete_count: int
    source_selected_count: int
    evidence_complete: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


@dataclass(frozen=True)
class InstallationTradeEvidenceProjection:
    """Metadata-only projection across every canonical proof store in a data root."""

    schema_version: str
    source_selection_policy_version: str
    projection_fingerprint: str
    evidence_complete: bool
    eligible_to_offer: bool
    reason_codes: tuple[str, ...]
    stores: tuple[CanonicalStoreSummary, ...]
    diagnostics: tuple[TradeEvidenceDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_selection_policy_version": self.source_selection_policy_version,
            "projection_fingerprint": self.projection_fingerprint,
            "evidence_complete": self.evidence_complete,
            "eligible_to_offer": self.eligible_to_offer,
            "reason_codes": list(self.reason_codes),
            "stores": [store.to_dict() for store in self.stores],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def canonical_store_key(role: str, relative: str) -> str:
    return hashlib.sha256(f"{role}:{relative}".encode("utf-8")).hexdigest()[:20]


def enumerate_canonical_store_roots(
    data_root: Path,
    auto_sessions_root: Path,
) -> tuple[list[tuple[CanonicalStoreRole, str, Path, bool]], list[str]]:
    """Return (role, private relative label, store path, parent_unsafe) per store.

    The relative label is only hashed into a store key and never emitted.
    """
    stores: list[tuple[CanonicalStoreRole, str, Path, bool]] = []
    container_reasons: list[str] = []

    def add_fixed(role: CanonicalStoreRole, label: str, path: Path) -> None:
        if path.exists() or path.is_symlink():
            stores.append((role, label, path, False))

    def add_container(role: CanonicalStoreRole, label: str, container: Path) -> None:
        if container.is_symlink() or (container.exists() and not container.is_dir()):
            container_reasons.append("canonical_store_container_not_regular")
            return
        if not container.exists():
            return
        try:
            children = sorted(container.iterdir(), key=lambda path: path.name)
        except OSError:
            container_reasons.append("canonical_store_container_unreadable")
            return
        for child in children:
            if child.name.startswith("."):
                continue
            unsafe_parent = child.is_symlink()
            if not unsafe_parent and not child.is_dir():
                continue
            proofs = child / "proofs"
            if unsafe_parent or proofs.exists() or proofs.is_symlink():
                stores.append((role, f"{label}/{child.name}/proofs", proofs, unsafe_parent))

    add_fixed("active_manual", "manual_proofs", data_root / "manual_proofs")
    add_container("archived_manual", "manual_proof_runs", data_root / "manual_proof_runs")
    add_container("autonomous_session", "auto_sessions", auto_sessions_root)
    add_fixed("autonomous_legacy", "proofs", data_root / "proofs")
    return stores, container_reasons


def _unsafe_parent_projection(role: CanonicalStoreRole) -> TradeEvidenceProjection:
    """Fail closed for a store reached through a symlinked parent without reading it."""
    reasons = _ordered_reasons(("canonical_store_not_regular", *_PERMANENT_BLOCKERS))
    payload = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "source_selection_policy_version": SOURCE_SELECTION_POLICY_VERSION,
        "store_role": role,
        "proof_set_revision": None,
        "stable_store_id": "",
        "evidence_complete": False,
        "eligible_to_offer": False,
        "reason_codes": reasons,
        "diagnostics": [],
    }
    return TradeEvidenceProjection(
        schema_version=PROJECTION_SCHEMA_VERSION,
        source_selection_policy_version=SOURCE_SELECTION_POLICY_VERSION,
        proof_set_revision=None,
        stable_store_id="",
        projection_fingerprint=_fingerprint(payload),
        evidence_complete=False,
        eligible_to_offer=False,
        reason_codes=reasons,
        diagnostics=(),
    )


def project_installation_proof_stores(
    data_root: Path | str,
    *,
    auto_sessions_root: Path | str | None = None,
) -> InstallationTradeEvidenceProjection:
    """Audit every canonical proof store under one data root.

    Covers the active manual store, archived manual runs, Autonomous session
    stores (including LeanOJ registrations), and the legacy store. Stores are
    identified only by role, stable store ID, and a short hash of their
    relative location: Autonomous session folder names contain prompt text and
    absolute paths are private, so neither is ever emitted. Derived proof-search
    indexes are never consulted.
    """
    root = Path(data_root)
    sessions_root = (
        Path(auto_sessions_root)
        if auto_sessions_root is not None
        else root / "auto_sessions"
    )
    enumerated, container_reasons = enumerate_canonical_store_roots(root, sessions_root)

    summaries: list[CanonicalStoreSummary] = []
    diagnostics: list[TradeEvidenceDiagnostic] = []
    for role, label, path, unsafe_parent in enumerated:
        store_key = canonical_store_key(role, label)
        projection = (
            _unsafe_parent_projection(role)
            if unsafe_parent
            else project_canonical_proof_store(path, store_role=role)
        )
        store_diagnostics = [
            replace(item, store_key=store_key) for item in projection.diagnostics
        ]
        diagnostics.extend(store_diagnostics)
        summaries.append(
            CanonicalStoreSummary(
                store_key=store_key,
                store_role=role,
                stable_store_id=projection.stable_store_id,
                proof_set_revision=projection.proof_set_revision,
                projection_fingerprint=projection.projection_fingerprint,
                record_count=len(store_diagnostics),
                evidence_complete_count=0,
                source_selected_count=0,
                evidence_complete=projection.evidence_complete,
                reason_codes=projection.reason_codes,
            )
        )

    stores_by_stable_id: dict[str, set[str]] = {}
    for summary in summaries:
        if summary.stable_store_id:
            stores_by_stable_id.setdefault(summary.stable_store_id, set()).add(
                summary.store_key
            )
    conflicting_stores = {
        key for keys in stores_by_stable_id.values() if len(keys) > 1 for key in keys
    }
    client_stores: dict[str, set[str]] = {}
    locator_stores: dict[str, set[str]] = {}
    for item in diagnostics:
        if item.client_proof_id:
            client_stores.setdefault(item.client_proof_id, set()).add(item.store_key)
        if item.canonical_locator_digest:
            locator_stores.setdefault(item.canonical_locator_digest, set()).add(
                item.store_key
            )
    duplicated_clients = {key for key, keys in client_stores.items() if len(keys) > 1}
    duplicated_locators = {key for key, keys in locator_stores.items() if len(keys) > 1}

    diagnostics = [
        _with_reasons(
            item,
            (
                *(
                    ("cross_store_stable_store_id_conflict",)
                    if item.store_key in conflicting_stores
                    else ()
                ),
                *(
                    ("cross_store_duplicate_occurrence",)
                    if item.client_proof_id in duplicated_clients
                    or item.canonical_locator_digest in duplicated_locators
                    else ()
                ),
            ),
            evidence_failure=True,
        )
        for item in diagnostics
    ]
    diagnostics = _resolve_duplicate_content(diagnostics)
    diagnostics.sort(key=lambda item: (item.store_key, item.record_key))

    summaries = [
        replace(
            summary,
            evidence_complete=summary.evidence_complete
            and summary.store_key not in conflicting_stores
            and all(
                item.evidence_complete
                for item in diagnostics
                if item.store_key == summary.store_key
            ),
            evidence_complete_count=sum(
                1
                for item in diagnostics
                if item.store_key == summary.store_key and item.evidence_complete
            ),
            source_selected_count=sum(
                1
                for item in diagnostics
                if item.store_key == summary.store_key and item.source_selected
            ),
            reason_codes=_ordered_reasons(
                (
                    *summary.reason_codes,
                    *(
                        ("cross_store_stable_store_id_conflict",)
                        if summary.store_key in conflicting_stores
                        else ()
                    ),
                )
            ),
        )
        for summary in summaries
    ]
    summaries.sort(key=lambda summary: (summary.store_role, summary.store_key))

    # An empty store is a normal state for an installation, not missing evidence.
    store_level_reasons = {
        reason
        for summary in summaries
        for reason in summary.reason_codes
        if reason not in _PERMANENT_BLOCKERS and reason != "canonical_store_empty"
    }
    duplicate_occurrence_reasons = (
        ("cross_store_duplicate_occurrence",)
        if duplicated_clients or duplicated_locators
        else ()
    )
    installation_reasons = _ordered_reasons(
        (
            *container_reasons,
            *store_level_reasons,
            *duplicate_occurrence_reasons,
            *_PERMANENT_BLOCKERS,
        )
    )
    evidence_complete = (
        not container_reasons
        and not store_level_reasons
        and not duplicate_occurrence_reasons
        and all(item.evidence_complete for item in diagnostics)
    )
    fingerprint = _fingerprint(
        {
            "schema_version": INSTALLATION_PROJECTION_SCHEMA_VERSION,
            "source_selection_policy_version": SOURCE_SELECTION_POLICY_VERSION,
            "evidence_complete": evidence_complete,
            "eligible_to_offer": False,
            "reason_codes": installation_reasons,
            "stores": [summary.to_dict() for summary in summaries],
            "diagnostics": [item.to_dict() for item in diagnostics],
        }
    )
    return InstallationTradeEvidenceProjection(
        schema_version=INSTALLATION_PROJECTION_SCHEMA_VERSION,
        source_selection_policy_version=SOURCE_SELECTION_POLICY_VERSION,
        projection_fingerprint=fingerprint,
        evidence_complete=evidence_complete,
        eligible_to_offer=False,
        reason_codes=installation_reasons,
        stores=tuple(summaries),
        diagnostics=tuple(diagnostics),
    )
