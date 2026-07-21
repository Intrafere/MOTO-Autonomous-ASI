"""
Shared registration for Lean-verified proofs.

Callers that already have Lean-accepted code use this module to classify the
proof with the validator novelty prompt and store the resulting ProofRecord in
the central proof database.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from backend.autonomous.core.proof_novelty import assess_proof_novelty
from backend.autonomous.memory.proof_database import is_prompt_injection_novel_tier
from backend.shared.models import (
    ProofArtifactPurpose,
    ProofAttemptFeedback,
    ProofDependency,
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
    solver_hints: Optional[list[str]] = None,
    broadcast_fn: BroadcastFn = None,
    base_event: Optional[dict[str, Any]] = None,
    proof_label: str = "",
    retry_origin_source_id: str = "",
    run_id: str = "",
    artifact_purpose: ProofArtifactPurpose | None = None,
    ownership_predicate: Optional[Callable[[], bool]] = None,
) -> RegisteredProof:
    """
    Classify and store Lean-verified proof code using the shared novelty tiers.

    Every Lean-verified occurrence is independently novelty-assessed and stored.
    Historical theorem/code matches are evidence for the validator, not a
    shortcut that can suppress or pre-classify the current-run occurrence.
    """
    novelty_tier, novelty_reasoning = await assess_proof_novelty(
        user_prompt=user_prompt,
        theorem_statement=theorem_statement,
        lean_code=lean_code,
        validator_model=validator_model,
        validator_context=validator_context,
        validator_max_tokens=validator_max_tokens,
        existing_novel_proofs="",
        task_id=task_id,
        role_id=role_id,
    )
    independent_novelty_tier = novelty_tier
    independent_novelty_reasoning = novelty_reasoning

    active_session_id = getattr(
        getattr(proof_database, "_session_manager", None),
        "session_id",
        None,
    )
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
    if is_cross_run_duplicate and novelty_tier != "not_novel":
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
        independent_novelty_tier=independent_novelty_tier,
        independent_novelty_reasoning=independent_novelty_reasoning,
        exact_duplicate_proof_id=existing.proof_id if is_cross_run_duplicate else "",
        exact_duplicate_run_id=existing_run_id if is_cross_run_duplicate else "",
        artifact_purpose=resolved_artifact_purpose,
        canonical_identity_version=identity.version,
        canonical_theorem_statement_hash=identity.theorem_statement_hash,
        canonical_lean_code_hash=identity.lean_code_hash,
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
