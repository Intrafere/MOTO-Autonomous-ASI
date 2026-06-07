"""Proof Solver API routes backed by the LeanOJ workflow."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from backend.aggregator.core.coordinator import coordinator
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator
from backend.shared.config import system_config
from backend.shared.embedding_readiness import require_embedding_provider_ready
from backend.shared.models import LeanOJStartRequest
from backend.shared.workflow_start_guard import workflow_start_guard

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/leanoj", tags=["leanoj"])


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
    return proofs


def _build_leanoj_session_summary(payload: dict[str, Any], proofs: list[dict[str, Any]]) -> dict[str, Any]:
    session_id = str(payload.get("session_id") or "latest")
    prompt = _leanoj_prompt(payload)
    final_count = sum(1 for proof in proofs if proof.get("proof_kind") == "final")
    subproof_count = sum(1 for proof in proofs if proof.get("proof_kind") == "subproof")
    return {
        "session_id": session_id,
        "user_prompt": prompt,
        "selected_topic": str(payload.get("selected_topic") or ""),
        "created_at": _leanoj_created_at(payload),
        "updated_at": _leanoj_created_at(payload),
        "phase": str(payload.get("phase") or ""),
        "proof_count": len(proofs),
        "final_count": final_count,
        "subproof_count": subproof_count,
        "is_current": session_id == leanoj_coordinator.get_state().session_id,
    }


def _sort_leanoj_proofs(proofs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        proofs,
        key=lambda proof: str(proof.get("created_at") or ""),
        reverse=True,
    )


def _get_start_conflict() -> Optional[str]:
    if leanoj_coordinator.is_active:
        return "Proof Solver is already running"
    if coordinator.is_running:
        return "Cannot start Proof Solver while Aggregator is running. Stop Aggregator first."
    if compiler_coordinator.is_running:
        return "Cannot start Proof Solver while Compiler is running. Stop Compiler first."
    autonomous_state = autonomous_coordinator.get_state()
    if autonomous_state.is_running or autonomous_coordinator.is_active:
        return "Cannot start Proof Solver while Autonomous Research is running. Stop Autonomous Research first."
    return None


def _validate_role_limits(label: str, role_config) -> None:
    try:
        context_window = int(role_config.context_window)
        max_output_tokens = int(role_config.max_output_tokens)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"{label} context window and max output tokens must be configured as positive integers.",
        )
    if context_window <= 0 or max_output_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"{label} context window and max output tokens must be configured as positive integers.",
        )
    if max_output_tokens >= context_window:
        raise HTTPException(
            status_code=400,
            detail=f"{label} max output tokens must be smaller than its context window.",
        )


def _validate_start_role_limits(request: LeanOJStartRequest) -> None:
    _validate_role_limits("Topic generator", request.topic_generator)
    _validate_role_limits("Topic validator", request.topic_validator)
    _validate_role_limits("Brainstorm validator", request.brainstorm_validator)
    _validate_role_limits("Final proof solver", request.final_solver)
    for index, submitter in enumerate(request.brainstorm_submitters, start=1):
        _validate_role_limits(f"Brainstorm submitter {index}", submitter)


@router.post("/start")
async def start_leanoj(request: LeanOJStartRequest):
    """Start a Proof Solver run."""
    try:
        async with workflow_start_guard.reserve():
            conflict = _get_start_conflict()
            if conflict:
                raise HTTPException(status_code=400, detail=conflict)
            if not system_config.lean4_enabled:
                raise HTTPException(status_code=400, detail="Lean 4 is disabled. Enable Lean 4 proof verification before starting Proof Solver.")
            _validate_start_role_limits(request)
            await require_embedding_provider_ready()
            resumed = await leanoj_coordinator.resume_or_initialize(request)
            if not leanoj_coordinator.start_in_background():
                raise HTTPException(status_code=400, detail="Proof Solver is already running")
            return {
                "success": True,
                "message": "Proof Solver resumed" if resumed else "Proof Solver started",
                "resumed": resumed,
                "session_id": leanoj_coordinator.get_state().session_id,
            }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to start Proof Solver")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stop")
async def stop_leanoj():
    """Stop the active Proof Solver run."""
    try:
        await leanoj_coordinator.stop()
        return {
            "success": True,
            "message": "Proof Solver stopped",
            "status": leanoj_coordinator.get_status(),
        }
    except Exception as exc:
        logger.exception("Failed to stop Proof Solver")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/clear")
async def clear_leanoj(confirm: bool = False):
    """Clear saved Proof Solver progress."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required. Use ?confirm=true to clear Proof Solver progress.")
    try:
        await leanoj_coordinator.clear()
        return {
            "success": True,
            "message": "Proof Solver progress cleared",
            "status": leanoj_coordinator.get_status(),
        }
    except Exception as exc:
        logger.exception("Failed to clear Proof Solver progress")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def get_leanoj_status():
    """Return the current Proof Solver state."""
    return leanoj_coordinator.get_status()


@router.get("/master-proof")
async def get_leanoj_master_proof():
    """Return the current Proof Solver master proof draft on demand."""
    return await leanoj_coordinator.get_master_proof_draft()


@router.get("/master-proof/edits")
async def get_leanoj_master_proof_edits(limit: int = 50):
    """Return compact summaries of recent Proof Solver master proof edits."""
    return await leanoj_coordinator.get_master_proof_edit_summaries(limit=limit)


@router.get("/proofs")
async def get_leanoj_proofs():
    """Return verified proofs from the currently loaded LeanOJ run."""
    status = leanoj_coordinator.get_status()
    proofs = _extract_leanoj_proofs(status)
    return {
        "proofs": _sort_leanoj_proofs(proofs),
        "status": status,
        "counts": {
            "total": len(proofs),
            "final": sum(1 for proof in proofs if proof.get("proof_kind") == "final"),
            "subproof": sum(1 for proof in proofs if proof.get("proof_kind") == "subproof"),
        },
    }


@router.get("/library")
async def get_leanoj_library(include_subproofs: bool = True):
    """Return completed Proof Solver proof works across saved sessions."""
    payloads_by_session: dict[str, dict[str, Any]] = {
        str(payload.get("session_id") or ""): payload
        for payload in _iter_leanoj_state_payloads()
        if payload.get("session_id")
    }

    current_status = leanoj_coordinator.get_status()
    current_session_id = str(current_status.get("session_id") or "")
    if current_session_id:
        payloads_by_session[current_session_id] = current_status

    proofs: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for payload in payloads_by_session.values():
        session_proofs = _extract_leanoj_proofs(payload, include_subproofs=include_subproofs)
        if not session_proofs:
            continue
        proofs.extend(session_proofs)
        sessions.append(_build_leanoj_session_summary(payload, session_proofs))

    return {
        "proofs": _sort_leanoj_proofs(proofs),
        "sessions": sorted(
            sessions,
            key=lambda session: str(session.get("updated_at") or ""),
            reverse=True,
        ),
    }


@router.get("/library/{session_id}/{proof_id}")
async def get_leanoj_library_proof(session_id: str, proof_id: str):
    """Return one completed Proof Solver proof work with full Lean source."""
    current_status = leanoj_coordinator.get_status()
    if str(current_status.get("session_id") or "") == session_id:
        for proof in _extract_leanoj_proofs(current_status):
            if proof.get("proof_id") == proof_id:
                return proof

    for payload in _iter_leanoj_state_payloads():
        if str(payload.get("session_id") or "") != session_id:
            continue
        for proof in _extract_leanoj_proofs(payload):
            if proof.get("proof_id") == proof_id:
                return proof
        break

    raise HTTPException(status_code=404, detail="Proof Solver proof work not found")


@router.post("/skip-brainstorm")
async def skip_leanoj_brainstorm():
    """Request immediate exit from Proof Solver brainstorming into final proof solving."""
    if not leanoj_coordinator.is_active:
        raise HTTPException(status_code=400, detail="Proof Solver is not running")
    await leanoj_coordinator.skip_brainstorm()
    return {
        "success": True,
        "message": "Proof Solver brainstorming will be skipped and final proof solving will start",
        "status": leanoj_coordinator.get_status(),
    }


@router.post("/force-brainstorm")
async def force_leanoj_brainstorm():
    """Request a return to recursive Proof Solver brainstorming without clearing proof progress."""
    if not leanoj_coordinator.is_active:
        raise HTTPException(status_code=400, detail="Proof Solver is not running")
    await leanoj_coordinator.force_brainstorm()
    return {
        "success": True,
        "message": "Proof Solver will return to recursive brainstorming with the current proof preserved",
        "status": leanoj_coordinator.get_status(),
    }
