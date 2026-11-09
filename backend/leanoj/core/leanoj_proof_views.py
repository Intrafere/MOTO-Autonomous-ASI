"""Read-only views of saved Proof Solver (LeanOJ) runs and their verified proofs.

The Proof Solver routes and Auto Top-p Translate both build proof views from
the same saved state, so a proof shows the same fields wherever it appears.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.shared.config import system_config

logger = logging.getLogger(__name__)


def _leanoj_sessions_base_dir() -> Path:
    return Path(system_config.data_dir) / "leanoj_sessions"


def _read_leanoj_state_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read LeanOJ state file %s: %s", path, exc)
        return None

    if not isinstance(payload, dict):
        return None
    payload.setdefault("session_id", path.parent.name)
    return payload


def _iter_leanoj_state_payloads() -> list[dict[str, Any]]:
    base_dir = _leanoj_sessions_base_dir()
    if not base_dir.exists():
        return []

    payloads: list[dict[str, Any]] = []
    for state_file in base_dir.glob("*/state.json"):
        if not state_file.is_file():
            continue
        payload = _read_leanoj_state_file(state_file)
        if payload is not None:
            payload["_state_file_mtime"] = state_file.stat().st_mtime
            payloads.append(payload)

    return payloads


def _leanoj_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request_payload = payload.get("request")
    return request_payload if isinstance(request_payload, dict) else {}


def _leanoj_prompt(payload: dict[str, Any]) -> str:
    request_payload = _leanoj_request_payload(payload)
    return (
        str(request_payload.get("user_prompt") or "").strip()
        or str(payload.get("user_prompt") or "").strip()
        or str(payload.get("prompt") or "").strip()
        or str(payload.get("selected_topic") or "").strip()
        or "Proof Solver problem"
    )


def _leanoj_created_at(payload: dict[str, Any], fallback: str = "") -> str:
    return (
        str(payload.get("updated_at") or "").strip()
        or fallback
        or ""
    )


def _leanoj_library_id(session_id: str, proof_id: str) -> str:
    return f"{session_id}:{proof_id}"


def _build_leanoj_final_proof(payload: dict[str, Any]) -> dict[str, Any] | None:
    final_solution = str(payload.get("final_solution") or "").strip()
    if not final_solution:
        return None

    session_id = str(payload.get("session_id") or "latest")
    prompt = _leanoj_prompt(payload)
    request_payload = _leanoj_request_payload(payload)
    proof_id = "final_solution"
    shared_proof_id = str(payload.get("final_proof_id") or "").strip()
    return {
        "library_id": _leanoj_library_id(session_id, proof_id),
        "proof_id": proof_id,
        "shared_proof_id": shared_proof_id,
        "session_id": session_id,
        "run_id": session_id,
        "proof_kind": "final",
        "theorem_name": "Final Proof Solver Submission",
        "theorem_statement": prompt,
        "source_type": "leanoj_final",
        "source_id": session_id,
        "source_title": str(payload.get("selected_topic") or "").strip() or prompt,
        "user_prompt": prompt,
        "lean_template": str(request_payload.get("lean_template") or ""),
        "lean_code": final_solution,
        "solver": "Proof Solver",
        "attempt_count": int(payload.get("final_attempt_count") or 0),
        "verified": True,
        "novel": bool(payload.get("final_novel")),
        "novelty_tier": str(payload.get("final_novelty_tier") or "not_novel"),
        "novelty_reasoning": str(payload.get("final_novelty_reasoning") or ""),
        "novelty_scope": str(payload.get("final_novelty_scope") or "standalone"),
        "novelty_group_claim": str(payload.get("final_novelty_group_claim") or ""),
        "novelty_group_proof_ids": list(
            payload.get("final_novelty_group_proof_ids") or []
        ),
        "created_at": _leanoj_created_at(payload),
        "phase": str(payload.get("phase") or ""),
    }


def _build_leanoj_subproofs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    session_id = str(payload.get("session_id") or "latest")
    prompt = _leanoj_prompt(payload)
    request_payload = _leanoj_request_payload(payload)
    created_at_fallback = _leanoj_created_at(payload)
    subproofs = payload.get("verified_subproofs") or []
    if not isinstance(subproofs, list):
        return []

    proofs: list[dict[str, Any]] = []
    for index, subproof in enumerate(subproofs, start=1):
        if not isinstance(subproof, dict) or subproof.get("verified") is False:
            continue
        lean_code = str(subproof.get("lean_code") or "").strip()
        if not lean_code:
            continue

        proof_id = str(subproof.get("subproof_id") or f"subproof_{index:03d}")
        shared_proof_id = str(subproof.get("proof_id") or "").strip()
        request_text = str(subproof.get("request") or "").strip()
        theorem_or_lemma = str(subproof.get("theorem_or_lemma") or "").strip()
        return_title = theorem_or_lemma or request_text or proof_id
        proofs.append(
            {
                "library_id": _leanoj_library_id(session_id, proof_id),
                "proof_id": proof_id,
                "shared_proof_id": shared_proof_id,
                "session_id": session_id,
                "run_id": session_id,
                "proof_kind": "subproof",
                "theorem_name": return_title,
                "theorem_statement": theorem_or_lemma or request_text or "Verified Proof Solver subproof",
                "source_type": "leanoj_subproof",
                "source_id": session_id,
                "source_title": request_text or prompt,
                "user_prompt": prompt,
                "lean_template": str(request_payload.get("lean_template") or ""),
                "lean_code": lean_code,
                "lean_feedback": str(subproof.get("lean_feedback") or ""),
                "verification_notes": str(subproof.get("lean_feedback") or ""),
                "solver": "Proof Solver",
                "attempt_count": int(subproof.get("attempts_used") or 0),
                "verified": True,
                "novel": bool(subproof.get("novel")),
                "novelty_tier": str(subproof.get("novelty_tier") or "not_novel"),
                "novelty_reasoning": str(subproof.get("novelty_reasoning") or ""),
                "novelty_scope": str(subproof.get("novelty_scope") or "standalone"),
                "novelty_group_claim": str(
                    subproof.get("novelty_group_claim") or ""
                ),
                "novelty_group_proof_ids": list(
                    subproof.get("novelty_group_proof_ids") or []
                ),
                "role": str(subproof.get("role") or ""),
                "created_at": str(subproof.get("created_at") or "") or created_at_fallback,
                "phase": str(payload.get("phase") or ""),
            }
        )
    return proofs


def _extract_leanoj_proofs(payload: dict[str, Any], *, include_subproofs: bool = True) -> list[dict[str, Any]]:
    proofs: list[dict[str, Any]] = []
    final_proof = _build_leanoj_final_proof(payload)
    if final_proof is not None:
        proofs.append(final_proof)
    if include_subproofs:
        proofs.extend(_build_leanoj_subproofs(payload))
    by_shared_id = {
        str(proof.get("shared_proof_id") or ""): proof
        for proof in proofs
        if str(proof.get("shared_proof_id") or "")
    }
    for anchor in proofs:
        if str(anchor.get("novelty_scope") or "standalone") != "group":
            continue
        anchor_id = str(anchor.get("shared_proof_id") or "")
        member_ids = list(dict.fromkeys([
            *list(anchor.get("novelty_group_proof_ids") or []),
            anchor_id,
        ]))
        anchor["novelty_group_proof_ids"] = member_ids
        for member_id in member_ids:
            member = by_shared_id.get(member_id)
            if member is None:
                continue
            backlinks = list(member.get("novelty_group_anchor_proof_ids") or [])
            if anchor_id and anchor_id not in backlinks:
                backlinks.append(anchor_id)
            member["novelty_group_anchor_proof_ids"] = backlinks
    return proofs
