"""
Shared registration for Lean-verified proofs.

Callers that already have Lean-accepted code use this module to classify the
proof with the validator novelty prompt and store the resulting ProofRecord in
the central proof database.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from backend.autonomous.core.proof_novelty import assess_proof_novelty
from backend.autonomous.memory.proof_database import is_prompt_injection_novel_tier
from backend.shared.models import (
    ProofArtifactPurpose,
    ProofAttemptFeedback,
    ProofDependency,
    LeanCompileReceipt,
    LocalNoveltyAssessmentProvenance,
    PortableProofDependency,
    ProofRecord,
)
from backend.shared.proof_identity import (
    CANONICAL_PROOF_IDENTITY_VERSION,
    canonical_proof_identity,
)

logger = logging.getLogger(__name__)

BroadcastFn = Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]


@dataclass
class RegisteredProof:
    """Result of registering or reusing a verified proof record."""

    record: ProofRecord
    duplicate: bool = False


async def _verified_group_candidates_for_run(
    proof_database,
    *,
    run_id: str,
) -> tuple[str, frozenset[str]]:
    """Return compact, current-run verified proof choices for group-scoped ranking."""
    if not run_id:
        return "", frozenset()
    try:
        if hasattr(proof_database, "get_all_proofs_for_live_context"):
            proofs = await proof_database.get_all_proofs_for_live_context(run_id)
        else:
            proofs = await proof_database.get_all_proofs()
    except Exception as exc:
        logger.debug("Unable to load novelty group candidates for %s: %s", run_id, exc)
        return "", frozenset()
    candidates = [
        proof
        for proof in proofs
        if str(getattr(proof, "run_id", "") or "") == run_id
        and str(getattr(proof, "proof_id", "") or "").strip()
    ]
    candidates.sort(key=lambda proof: (str(proof.created_at), proof.proof_id))
    ids = frozenset(proof.proof_id for proof in candidates)
    text = "\n".join(
        (
            f"- proof_id: {proof.proof_id}\n"
            f"  theorem_name: {proof.theorem_name or '[unnamed]'}\n"
            f"  theorem_statement: {proof.theorem_statement}"
        )
        for proof in candidates
    )
    return text, ids


async def _find_existing_proof(
    proof_database,
    *,
    source_type: Optional[str] = None,
    source_id: Optional[str] = None,
    current_run_id: str = "",
    theorem_statement: str,
    lean_code: str,
) -> Optional[ProofRecord]:
    """Return an existing canonical theorem/code match, optionally source-scoped."""
    identity = canonical_proof_identity(theorem_statement, lean_code)
    try:
        candidates: list[ProofRecord] = list(await proof_database.get_all_proofs())
        if hasattr(proof_database, "list_proof_library"):
            for item in await proof_database.list_proof_library(novel_only=False):
                if isinstance(item, dict):
                    try:
                        candidates.append(ProofRecord.model_validate(item))
                    except Exception:
                        continue
        base_dir = getattr(proof_database, "_base_dir", None)
        if base_dir is not None and hasattr(proof_database, "list_proof_library_from_history"):
            history_root = base_dir.parent / "manual_proof_runs"
            for item in await proof_database.list_proof_library_from_history(
                history_root,
                novel_only=False,
            ):
                if isinstance(item, dict):
                    try:
                        candidates.append(ProofRecord.model_validate(item))
                    except Exception:
                        continue

        seen: set[tuple[str, str]] = set()
        for proof in candidates:
            occurrence_key = (proof.run_id, proof.proof_id)
            if occurrence_key in seen:
                continue
            seen.add(occurrence_key)
            if source_type is not None and proof.source_type != source_type:
                continue
            if source_id is not None and proof.source_id != source_id:
                continue
            proof_identity = canonical_proof_identity(
                proof.theorem_statement,
                proof.lean_code,
            )
            if proof_identity.key != identity.key:
                continue
            proof_run_id = str(proof.run_id or f"{proof.source_type}:{proof.source_id}")
            if current_run_id and proof_run_id == current_run_id:
                continue
            return proof
    except Exception as exc:
        logger.debug("Existing proof lookup failed for %s %s: %s", source_type, source_id, exc)
    try:
        from backend.shared.proof_search.models import default_proof_search_corpora
        from backend.shared.proof_search.search_service import proof_search_service

        enabled_internal_corpora = [
            corpus
            for corpus in default_proof_search_corpora()
            if corpus in {"moto", "manual", "leanoj"}
        ]
        matches = await proof_search_service.exact_identity_neighborhood(
            theorem_statement_hashes=[identity.theorem_statement_hash],
            lean_code_hashes=[identity.lean_code_hash],
            corpora=enabled_internal_corpora,
            exclude_run_ids=[current_run_id] if current_run_id else None,
            identity_version=CANONICAL_PROOF_IDENTITY_VERSION,
            limit=1,
        )
        if matches:
            match = matches[0]
            return ProofRecord(
                proof_id=match.proof_id,
                theorem_statement=match.theorem_statement,
                theorem_name=match.theorem_name,
                formal_sketch=match.formal_sketch,
                source_type=match.source_type,
                source_id=match.source_id,
                source_title=match.source_title,
                run_id=match.run_id or match.session_id,
                lean_code=match.lean_code,
                novel=match.novelty_tier != "not_novel",
                novelty_tier=match.novelty_tier or "not_novel",
                novelty_reasoning=match.novelty_reasoning,
            )
    except Exception as exc:
        logger.debug("Archived proof identity lookup failed: %s", exc)
    return None


async def _broadcast_registered_proof(
    *,
    broadcast_fn: BroadcastFn,
    record: ProofRecord,
    base_event: Optional[dict[str, Any]],
    proof_label: str = "",
    retry_origin_source_id: str = "",
) -> None:
    if not broadcast_fn:
        return

    event_payload = {
        **(base_event or {}),
        "proof_id": record.proof_id,
        "theorem_statement": record.theorem_statement,
        "solver": record.solver,
        "is_novel": record.novel,
        "novelty_tier": record.novelty_tier,
        "novelty_reasoning": record.novelty_reasoning,
        "novelty_scope": record.novelty_scope,
        "novelty_group_claim": record.novelty_group_claim,
        "novelty_group_proof_ids": list(record.novelty_group_proof_ids),
        "retry_origin_source_id": retry_origin_source_id,
    }
    if proof_label:
        event_payload["proof_label"] = proof_label

    if record.novel and is_prompt_injection_novel_tier(record.novelty_tier):
        await broadcast_fn("novel_proof_discovered", event_payload)
    else:
        await broadcast_fn("known_proof_verified", event_payload)


async def _broadcast_duplicate_proof(
    *,
    broadcast_fn: BroadcastFn,
    record: ProofRecord,
    base_event: Optional[dict[str, Any]],
    proof_label: str = "",
) -> None:
    if not broadcast_fn:
        return
    event_payload = {
        **(base_event or {}),
        "proof_id": record.proof_id,
        "theorem_statement": record.theorem_statement,
        "solver": record.solver,
        "is_novel": record.novel,
        "novelty_tier": record.novelty_tier,
        "novelty_reasoning": record.novelty_reasoning,
        "novelty_scope": record.novelty_scope,
        "novelty_group_claim": record.novelty_group_claim,
        "novelty_group_proof_ids": list(record.novelty_group_proof_ids),
        "duplicate": True,
    }
    if proof_label:
        event_payload["proof_label"] = proof_label
    await broadcast_fn("proof_registration_duplicate", event_payload)


async def register_verified_lean_proof(
    *,
    proof_database,
    user_prompt: str,
    theorem_statement: str,
    lean_code: str,
    validator_model: str,
    validator_context: int,
    validator_max_tokens: int,
    task_id: str,
    role_id: str,
    source_type: str,
    source_id: str,
    source_title: str = "",
    theorem_id: str = "",
    theorem_name: str = "",
    formal_sketch: str = "",
    solver: str = "Lean 4",
    verification_notes: str = "Lean 4 accepted the submitted proof.",
    attempt_count: int = 0,
    attempts: Optional[list[ProofAttemptFeedback]] = None,
    dependencies: Optional[list[ProofDependency]] = None,
    compile_receipt: Optional[LeanCompileReceipt] = None,
    portable_dependencies: Optional[list[PortableProofDependency]] = None,
    solver_hints: Optional[list[str]] = None,
    broadcast_fn: BroadcastFn = None,
    base_event: Optional[dict[str, Any]] = None,
    proof_label: str = "",
    retry_origin_source_id: str = "",
    run_id: str = "",
    artifact_purpose: ProofArtifactPurpose | None = None,
    ownership_predicate: Optional[Callable[[], bool]] = None,
    novelty_classification: Optional[tuple[str, str]] = None,
) -> RegisteredProof:
    """
    Classify and store Lean-verified proof code using the shared novelty tiers.

    Every Lean-verified occurrence is independently novelty-assessed and stored.
    Historical theorem/code matches are evidence for the validator, not a
    shortcut that can suppress or pre-classify the current-run occurrence.
    """
    if compile_receipt is not None:
        checked_hash = compile_receipt.checked_source_sha256
        from backend.shared.lean_source_evidence import source_sha256
        if checked_hash != source_sha256(lean_code):
            raise ValueError("Compile receipt does not bind the Lean source being registered")

    if novelty_classification is None:
        active_session_id = getattr(
            getattr(proof_database, "_session_manager", None),
            "session_id",
            None,
        )
        provisional_run_id = run_id or active_session_id
        if not provisional_run_id and hasattr(proof_database, "get_or_create_active_run_id"):
            provisional_run_id = await proof_database.get_or_create_active_run_id()
        resolved_run_id = provisional_run_id or f"{source_type}:{source_id}"
        group_candidates, allowed_group_ids = await _verified_group_candidates_for_run(
            proof_database,
            run_id=resolved_run_id,
        )
        if broadcast_fn:
            await broadcast_fn(
                "proof_novelty_assessment_started",
                {
                    **(base_event or {}),
                    "source_type": source_type,
                    "source_id": source_id,
                    "run_id": resolved_run_id,
                    "theorem_id": theorem_id,
                    "theorem_name": theorem_name,
                    "theorem_statement": theorem_statement,
                    "proof_label": proof_label,
                    "lean_verified": True,
                    "verified_group_candidates_available": bool(allowed_group_ids),
                },
            )
        novelty_assessment = await assess_proof_novelty(
            user_prompt=user_prompt,
            theorem_statement=theorem_statement,
            lean_code=lean_code,
            validator_model=validator_model,
            validator_context=validator_context,
            validator_max_tokens=validator_max_tokens,
            existing_novel_proofs="",
            verified_group_candidates=group_candidates,
            allowed_group_proof_ids=allowed_group_ids,
            task_id=task_id,
            role_id=role_id,
        )
        if hasattr(novelty_assessment, "novelty_tier"):
            novelty_tier = novelty_assessment.novelty_tier
            novelty_reasoning = novelty_assessment.reasoning
            novelty_scope = novelty_assessment.novelty_scope
            novelty_group_claim = novelty_assessment.group_claim
            novelty_group_proof_ids = list(novelty_assessment.group_proof_ids)
            assessment_origin = getattr(
                novelty_assessment,
                "assessment_origin",
                "validator_assessed",
            )
        else:
            novelty_tier, novelty_reasoning = novelty_assessment
            novelty_scope = "standalone"
            novelty_group_claim = ""
            novelty_group_proof_ids = []
            assessment_origin = "validator_assessed"
    else:
        novelty_tier, novelty_reasoning = novelty_classification
        novelty_scope = "standalone"
        novelty_group_claim = ""
        novelty_group_proof_ids = []
        assessment_origin = "caller_supplied"
    independent_novelty_tier = novelty_tier
    independent_novelty_reasoning = novelty_reasoning

    active_session_id = getattr(getattr(proof_database, "_session_manager", None), "session_id", None)
    prompt_run_id = run_id or active_session_id
    if not prompt_run_id and hasattr(proof_database, "get_or_create_active_run_id"):
        prompt_run_id = await proof_database.get_or_create_active_run_id()
    resolved_run_id = prompt_run_id or f"{source_type}:{source_id}"
    identity = canonical_proof_identity(theorem_statement, lean_code)
    existing = await _find_existing_proof(
        proof_database,
        theorem_statement=theorem_statement,
        lean_code=lean_code,
        current_run_id=resolved_run_id,
    )
    existing_run_id = (
        str(getattr(existing, "run_id", "") or "")
        or (
            f"{getattr(existing, 'source_type', '')}:{getattr(existing, 'source_id', '')}"
            if existing
            else ""
        )
    )
    is_cross_run_duplicate = bool(existing and existing_run_id != resolved_run_id)
    if (
        is_cross_run_duplicate
        and novelty_tier != "not_novel"
        and novelty_scope == "standalone"
    ):
        novelty_tier = "duplicate_novel"
        novelty_reasoning = independent_novelty_reasoning
    is_novel = novelty_tier != "not_novel"
    resolved_artifact_purpose: ProofArtifactPurpose = (
        artifact_purpose or "verified_occurrence"
    )

    record = ProofRecord(
        proof_id="",
        theorem_id=theorem_id,
        theorem_statement=theorem_statement,
        theorem_name=theorem_name,
        formal_sketch=formal_sketch,
        source_type=source_type,
        source_id=source_id,
        source_title=source_title,
        run_id=resolved_run_id,
        user_prompt=user_prompt,
        solver=solver,
        lean_code=lean_code,
        novel=is_novel,
        novelty_tier=novelty_tier,
        novelty_reasoning=novelty_reasoning,
        novelty_scope=novelty_scope,
        novelty_group_claim=novelty_group_claim,
        novelty_group_proof_ids=novelty_group_proof_ids,
        independent_novelty_tier=independent_novelty_tier,
        independent_novelty_reasoning=independent_novelty_reasoning,
        exact_duplicate_proof_id=existing.proof_id if is_cross_run_duplicate else "",
        exact_duplicate_run_id=existing_run_id if is_cross_run_duplicate else "",
        artifact_purpose=resolved_artifact_purpose,
        canonical_identity_version=identity.version,
        canonical_theorem_statement_hash=identity.theorem_statement_hash,
        canonical_lean_code_hash=identity.lean_code_hash,
        local_novelty_assessment_provenance=LocalNoveltyAssessmentProvenance(
            assessment_origin=assessment_origin,
            source_local_rank=independent_novelty_tier,
            assessment_scope=novelty_scope,
            group_claim=novelty_group_claim,
            group_proof_ids=novelty_group_proof_ids,
            assessed_at=datetime.now(timezone.utc),
        ),
        compile_receipt=compile_receipt,
        verification_evidence_status=("complete" if compile_receipt else "incomplete"),
        portable_dependencies=list(portable_dependencies or []),
        verification_notes=verification_notes,
        attempt_count=attempt_count,
        attempts=list(attempts or []),
        dependencies=list(dependencies or []),
        solver_hints=list(solver_hints or []),
    )
    if ownership_predicate is not None and not ownership_predicate():
        raise RuntimeError("Proof registration ownership was superseded")

    if hasattr(proof_database, "add_proof_occurrence"):
        stored = await proof_database.add_proof_occurrence(record)
        duplicate = False
    elif hasattr(proof_database, "add_proof_if_absent"):
        stored, duplicate = await proof_database.add_proof_if_absent(record)
    else:
        stored = await proof_database.add_proof(record)
        duplicate = stored.proof_id != record.proof_id and record.proof_id != ""

    if stored.novelty_scope == "group" and hasattr(
        proof_database, "ensure_complete_novelty_group"
    ):
        stored = await proof_database.ensure_complete_novelty_group(stored.proof_id)

    if duplicate:
        await _broadcast_duplicate_proof(
            broadcast_fn=broadcast_fn,
            record=stored,
            base_event=base_event,
            proof_label=proof_label,
        )
        return RegisteredProof(record=stored, duplicate=True)

    await _broadcast_registered_proof(
        broadcast_fn=broadcast_fn,
        record=stored,
        base_event=base_event,
        proof_label=proof_label,
        retry_origin_source_id=retry_origin_source_id,
    )
    return RegisteredProof(record=stored, duplicate=False)
