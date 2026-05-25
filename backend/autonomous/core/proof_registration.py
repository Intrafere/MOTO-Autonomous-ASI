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
from backend.shared.models import ProofAttemptFeedback, ProofDependency, ProofRecord

logger = logging.getLogger(__name__)

BroadcastFn = Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]


@dataclass
class RegisteredProof:
    """Result of registering or reusing a verified proof record."""

    record: ProofRecord
    duplicate: bool = False


def _normalize_for_duplicate_check(value: str) -> str:
    return "\n".join((value or "").strip().splitlines())


async def _find_existing_proof(
    proof_database,
    *,
    source_type: str,
    source_id: str,
    theorem_statement: str,
    lean_code: str,
) -> Optional[ProofRecord]:
    """Return an existing proof for the same source/theorem/code if present."""
    normalized_statement = " ".join((theorem_statement or "").split())
    normalized_code = _normalize_for_duplicate_check(lean_code)
    try:
        for proof in await proof_database.get_all_proofs():
            if proof.source_type != source_type or proof.source_id != source_id:
                continue
            if " ".join((proof.theorem_statement or "").split()) != normalized_statement:
                continue
            if _normalize_for_duplicate_check(proof.lean_code) != normalized_code:
                continue
            return proof
    except Exception as exc:
        logger.debug("Existing proof lookup failed for %s %s: %s", source_type, source_id, exc)
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

    if record.novel:
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
) -> RegisteredProof:
    """
    Classify and store Lean-verified proof code using the shared novelty tiers.

    Duplicate detection is scoped to source type/id, theorem statement, and
    Lean code. When a duplicate is found, the existing record is returned and
    no novelty API call is made.
    """
    existing = await _find_existing_proof(
        proof_database,
        source_type=source_type,
        source_id=source_id,
        theorem_statement=theorem_statement,
        lean_code=lean_code,
    )
    if existing is not None:
        await _broadcast_duplicate_proof(
            broadcast_fn=broadcast_fn,
            record=existing,
            base_event=base_event,
            proof_label=proof_label,
        )
        return RegisteredProof(record=existing, duplicate=True)

    existing_novel_proofs = proof_database.get_novel_proofs_for_injection()
    novelty_tier, novelty_reasoning = await assess_proof_novelty(
        user_prompt=user_prompt,
        theorem_statement=theorem_statement,
        lean_code=lean_code,
        validator_model=validator_model,
        validator_context=validator_context,
        validator_max_tokens=validator_max_tokens,
        existing_novel_proofs=existing_novel_proofs,
        task_id=task_id,
        role_id=role_id,
    )
    is_novel = novelty_tier != "not_novel"

    record = ProofRecord(
        proof_id="",
        theorem_id=theorem_id,
        theorem_statement=theorem_statement,
        theorem_name=theorem_name,
        formal_sketch=formal_sketch,
        source_type=source_type,
        source_id=source_id,
        source_title=source_title,
        solver=solver,
        lean_code=lean_code,
        novel=is_novel,
        novelty_tier=novelty_tier,
        novelty_reasoning=novelty_reasoning,
        verification_notes=verification_notes,
        attempt_count=attempt_count,
        attempts=list(attempts or []),
        dependencies=list(dependencies or []),
        solver_hints=list(solver_hints or []),
    )
    if hasattr(proof_database, "add_proof_if_absent"):
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
