"""
Proof database, Lean 4 status, manual proof checks, and certificate export routes.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.api.routes import websocket
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory, brainstorm_memory
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.proof_database import ProofDatabase, proof_database
from backend.autonomous.memory.research_metadata import research_metadata
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import system_config
from backend.shared.lean4_client import (
    clear_lean4_client,
    close_lean4_client,
    get_lean4_client,
    initialize_lean4_client,
)
from backend.shared.models import (
    ModelConfig,
    ProofCheckRequest,
    ProofRoleConfigSnapshot,
    ProofRuntimeConfigSnapshot,
    ProofSettingsUpdateRequest,
)
from backend.shared.path_safety import resolve_path_within_root
from backend.shared.runtime_settings import RuntimeSettingsError, save_proof_runtime_settings
from backend.shared.smt_client import clear_smt_client, get_smt_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proofs", tags=["proofs"])


def _schedule_lean4_warm_start(client) -> None:
    """Warm the Lean workspace without blocking a settings/status request."""
    async def _warm_start() -> None:
        try:
            await client.warm_start()
        except Exception as exc:  # pragma: no cover - defensive background task
            logger.warning("Lean 4 client warm start failed: %s", exc)

    asyncio.create_task(_warm_start())


def _safe_path_label(path_value: str) -> str:
    """Return a display-safe basename instead of an absolute local path."""
    text = str(path_value or "").strip()
    if not text:
        return ""
    try:
        return Path(text).name or "[configured]"
    except Exception:
        return "[configured]"


async def _get_export_proof_or_404(proof_id: str):
    try:
        proof = await proof_database.get_proof(proof_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Proof not found")
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")
    return proof


async def _get_export_lean_code(proof_id: str) -> str:
    try:
        return await proof_database.get_lean_code(proof_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Proof not found")


def _build_model_config(role: ProofRoleConfigSnapshot) -> ModelConfig:
    return ModelConfig(
        provider=role.provider,
        model_id=role.model_id,
        openrouter_model_id=role.model_id if role.provider == "openrouter" else None,
        openrouter_provider=role.openrouter_provider,
        openrouter_reasoning_effort=role.openrouter_reasoning_effort,
        lm_studio_fallback_id=role.lm_studio_fallback_id,
        context_window=role.context_window,
        max_output_tokens=role.max_output_tokens,
        supercharge_enabled=role.supercharge_enabled,
    )


def _runtime_snapshot_validation_error(snapshot: ProofRuntimeConfigSnapshot) -> Optional[str]:
    roles = {
        "brainstorm": snapshot.brainstorm,
        "paper": snapshot.paper,
        "validator": snapshot.validator,
    }
    for label, role in roles.items():
        if not role.model_id:
            return f"Proof runtime model configuration is missing a model for {label}."
        try:
            context_window = int(role.context_window)
            max_output_tokens = int(role.max_output_tokens)
        except (TypeError, ValueError):
            return (
                f"Proof runtime {label} context window and max output tokens must be "
                "configured as positive integers."
            )
        if context_window <= 0 or max_output_tokens <= 0:
            return (
                f"Proof runtime {label} context window and max output tokens must be "
                "configured as positive integers."
            )
        if max_output_tokens >= context_window:
            return f"Proof runtime {label} max output tokens must be smaller than its context window."
    return None


def _get_request_runtime_snapshot(request: Optional[ProofCheckRequest]) -> Optional[ProofRuntimeConfigSnapshot]:
    if not request or not request.proof_runtime_config:
        return None

    try:
        snapshot = ProofRuntimeConfigSnapshot(**request.proof_runtime_config)
    except Exception as exc:
        logger.error("Manual proof runtime config from request is invalid: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Manual proof runtime model configuration is invalid.",
        )
    validation_error = _runtime_snapshot_validation_error(snapshot)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    return snapshot


async def _get_runtime_snapshot(request: Optional[ProofCheckRequest] = None) -> Optional[ProofRuntimeConfigSnapshot]:
    request_snapshot = _get_request_runtime_snapshot(request)
    if request_snapshot is not None:
        return request_snapshot

    snapshot_dict = autonomous_coordinator.get_proof_runtime_config()
    if not snapshot_dict:
        snapshot_dict = await research_metadata.get_proof_runtime_config()
    if not snapshot_dict:
        return None

    try:
        return ProofRuntimeConfigSnapshot(**snapshot_dict)
    except Exception as exc:
        logger.error("Stored proof runtime config is invalid: %s", exc)
        return None


async def _get_manual_check_status() -> Tuple[bool, str]:
    if not system_config.lean4_enabled:
        return False, "Lean 4 proof checks are disabled."

    snapshot = await _get_runtime_snapshot()
    if snapshot is None:
        return False, "No proof runtime model configuration is available yet. Start autonomous research once before using manual proof checks."

    validation_error = _runtime_snapshot_validation_error(snapshot)
    if validation_error:
        return False, validation_error

    return True, ""


def _configure_manual_roles(source_type: str, snapshot: ProofRuntimeConfigSnapshot) -> ProofRoleConfigSnapshot:
    role_config = snapshot.brainstorm if source_type == "brainstorm" else snapshot.paper
    if not role_config.model_id or not snapshot.validator.model_id:
        raise RuntimeError("Manual proof roles are missing a configured submitter or validator model.")
    suffix = f"manual_{source_type}"
    api_client_manager.configure_role(
        f"autonomous_proof_identification_{suffix}",
        _build_model_config(role_config),
    )
    api_client_manager.configure_role(
        f"autonomous_proof_lemma_search_{suffix}",
        _build_model_config(role_config),
    )
    api_client_manager.configure_role(
        f"autonomous_proof_formalization_{suffix}",
        _build_model_config(role_config),
    )
    api_client_manager.configure_role(
        "autonomous_proof_novelty",
        _build_model_config(snapshot.validator),
    )
    return role_config


async def _prompt_with_verified_proof_context(prompt: str) -> str:
    """Apply proof-library context to a source-specific manual proof prompt."""
    source_prompt = (prompt or "").strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_user_prompt()).strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_base_user_prompt()).strip()
    return proof_database.inject_into_prompt(source_prompt)


def _history_proof_database_for_session(session_id: str) -> Optional[ProofDatabase]:
    """Return a read-only proof database view for a history session."""
    if not session_id:
        return None
    if session_id == "legacy":
        proofs_dir = Path(system_config.data_dir) / "proofs"
    else:
        try:
            session_path = resolve_path_within_root(
                Path(system_config.auto_sessions_base_dir),
                session_id,
            )
        except Exception:
            return None
        proofs_dir = session_path / "proofs"
    if not proofs_dir.exists():
        return None
    history_db = ProofDatabase()
    history_db._base_dir = proofs_dir
    history_db._index_data = None
    return history_db


async def _prompt_with_history_proof_context(prompt: str, session_id: str) -> str:
    """Apply the selected history session's proof context when available."""
    source_prompt = (prompt or "").strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_user_prompt()).strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_base_user_prompt()).strip()

    history_db = _history_proof_database_for_session(session_id)
    if history_db is None:
        return proof_database.inject_into_prompt(source_prompt)
    return history_db.inject_into_prompt(source_prompt)


async def _augment_paper_content_with_source_brainstorms(
    paper_content: str,
    source_brainstorm_ids,
    source_brainstorm_memory=None,
) -> str:
    parts = [f"PAPER CONTENT:\n{(paper_content or '').strip()}"]
    memory = source_brainstorm_memory or brainstorm_memory
    for brainstorm_id in source_brainstorm_ids or []:
        try:
            brainstorm_content = await memory.get_database_content(
                str(brainstorm_id),
                strip_proofs=True,
            )
        except Exception as exc:
            logger.debug("Unable to load source brainstorm %s for manual proof check: %s", brainstorm_id, exc)
            continue
        if brainstorm_content:
            parts.append(
                f"SOURCE BRAINSTORM {brainstorm_id}:\n"
                f"{brainstorm_content.strip()}"
            )
    return "\n\n---\n\n".join(part for part in parts if part.strip())


def _history_brainstorm_memory_for_session(session_id: str) -> Optional[BrainstormMemory]:
    """Return a session-scoped brainstorm reader for manual history proof checks."""
    if session_id == "legacy":
        brainstorms_dir = Path(system_config.auto_brainstorms_dir)
    else:
        try:
            session_path = resolve_path_within_root(
                Path(system_config.auto_sessions_base_dir),
                session_id,
            )
        except Exception:
            return None
        brainstorms_dir = session_path / "brainstorms"

    if not brainstorms_dir.exists():
        return None

    scoped_memory = BrainstormMemory()
    scoped_memory._base_dir = brainstorms_dir
    return scoped_memory


async def _resolve_manual_source(request: ProofCheckRequest) -> Tuple[str, str, str]:
    if request.source_type == "brainstorm":
        metadata = await brainstorm_memory.get_metadata(request.source_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Brainstorm not found")
        content = await brainstorm_memory.get_database_content(
            request.source_id,
            strip_proofs=True,
        )
        if not content:
            raise HTTPException(status_code=404, detail="Brainstorm content not found")
        user_prompt = await _prompt_with_verified_proof_context(await research_metadata.get_user_prompt())
        return content, metadata.topic_prompt, user_prompt

    metadata = await paper_library.get_metadata(request.source_id)
    if metadata is None:
        if ":" not in request.source_id:
            raise HTTPException(status_code=404, detail="Paper not found")
        session_id, paper_id = request.source_id.split(":", 1)
        history_paper = await paper_library.get_history_paper(session_id, paper_id)
        if not history_paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        content = paper_library.strip_verified_proofs_from_content(
            str(history_paper.get("content", "") or "")
        )
        if not content:
            raise HTTPException(status_code=404, detail="Paper content not found")
        source_brainstorm_ids = history_paper.get("source_brainstorm_ids") or []
        history_brainstorm_memory = _history_brainstorm_memory_for_session(session_id)
        if source_brainstorm_ids and history_brainstorm_memory is not None:
            content = await _augment_paper_content_with_source_brainstorms(
                content,
                source_brainstorm_ids,
                source_brainstorm_memory=history_brainstorm_memory,
            )
        user_prompt = await _prompt_with_history_proof_context(
            str(history_paper.get("user_prompt", "") or ""),
            session_id,
        )
        return content, str(history_paper.get("title", "") or paper_id), user_prompt
    content = await paper_library.get_paper_content(
        request.source_id,
        strip_proofs=True,
    )
    if not content:
        raise HTTPException(status_code=404, detail="Paper content not found")
    content = await _augment_paper_content_with_source_brainstorms(
        content,
        metadata.source_brainstorm_ids,
    )
    user_prompt = await _prompt_with_verified_proof_context(await research_metadata.get_user_prompt())
    return content, metadata.title, user_prompt


async def _run_manual_proof_check(request: ProofCheckRequest) -> None:
    source_title = ""
    try:
        source_content, source_title, user_prompt = await _resolve_manual_source(request)
        snapshot = await _get_runtime_snapshot(request)
        if snapshot is None:
            raise RuntimeError("No proof runtime model configuration is available yet.")

        role_config = _configure_manual_roles(request.source_type, snapshot)
        stage = autonomous_coordinator._proof_verification_stage
        await stage.run_manual(
            content=source_content,
            source_type=request.source_type,
            source_id=request.source_id,
            user_prompt=user_prompt,
            submitter_model=role_config.model_id,
            submitter_context=role_config.context_window,
            submitter_max_tokens=role_config.max_output_tokens,
            validator_model=snapshot.validator.model_id,
            validator_context=snapshot.validator.context_window,
            validator_max_tokens=snapshot.validator.max_output_tokens,
            broadcast_fn=websocket.broadcast_event,
            novel_proofs_db=proof_database,
            source_title=source_title,
            source_reserved=True,
        )
    except Exception as exc:
        logger.exception("Manual proof check failed for %s %s", request.source_type, request.source_id)
        await websocket.broadcast_event(
            "proof_check_complete",
            {
                "source_type": request.source_type,
                "source_id": request.source_id,
                "source_title": source_title,
                "trigger": "manual",
                "novel_count": 0,
                "verified_count": 0,
                "total_candidates": 0,
                "message": (
                    "Proof verification encountered an error: "
                    f"{ProofVerificationStage._summarize_error(str(exc), limit=960)}"
                ),
            },
        )
        await ProofVerificationStage.release_source(request.source_type, request.source_id)


@router.get("")
async def list_proofs():
    """Return all verified proofs plus aggregate counts."""
    proofs = await proof_database.get_all_proofs()
    return {
        "proofs": [proof.model_dump(mode="json") for proof in proofs],
        "counts": proof_database.count_proofs(),
    }


@router.get("/novel")
async def list_novel_proofs():
    """Return only novel verified proofs."""
    proofs = await proof_database.get_all_proofs(novel_only=True)
    return {
        "proofs": [proof.model_dump(mode="json") for proof in proofs],
        "counts": proof_database.count_proofs(),
    }


@router.get("/known")
async def list_known_proofs():
    """Return only known (non-novel) verified proofs."""
    proofs = await proof_database.get_all_proofs(novel_only=False)
    return {
        "proofs": [proof.model_dump(mode="json") for proof in proofs],
        "counts": proof_database.count_proofs(),
    }


async def _strip_known_proofs_from_files() -> dict:
    """Utility: strip non-novel proof entries from brainstorm and paper files on disk.

    Iterates all brainstorm and paper files in the current session and removes
    entries marked ``Status: Verified (Known)`` from their proof sections while
    preserving entries marked ``Status: Verified (Novel)``.  Returns a summary
    dict with counts of files modified and proof entries removed.

    This is safe to run mid-session; the proof data is not lost — every proof
    (novel or known) remains in ProofDatabase (the JSON index files).
    """
    import re as _re
    import asyncio as _asyncio

    files_checked = 0
    files_modified = 0
    entries_removed = 0

    def _clean_content(content: str, proof_header: str) -> tuple[str, int]:
        """Return (cleaned_content, removed_count).  Removes Known entries only."""
        if proof_header not in content:
            return content, 0

        before, _, after = content.partition(proof_header)
        # Split the proof section into individual proof blocks
        # Each block starts with "Proof N:" and ends before the next "Proof N:" or EOF
        block_pattern = _re.compile(r'(?=^Proof \d+:)', _re.MULTILINE)
        blocks = _re.split(block_pattern, after)

        kept = []
        removed = 0
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue
            # Remove blocks that are explicitly marked as Known
            if 'Status: Verified (Known)' in block:
                removed += 1
            else:
                kept.append(block)

        if removed == 0:
            return content, 0

        if kept:
            new_after = "\n".join(kept)
            new_content = before + proof_header + "\n\n" + new_after
        else:
            # All proofs in this section were Known — remove the header too
            new_content = before.rstrip()

        return new_content, removed

    # Clean brainstorm files
    brainstorm_paths = list(brainstorm_memory._base_dir.rglob("brainstorm_*.txt")) if hasattr(brainstorm_memory, '_base_dir') else []
    for path in brainstorm_paths:
        try:
            files_checked += 1
            text = path.read_text(encoding="utf-8")
            cleaned, removed = _clean_content(text, "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ===")
            if removed > 0:
                path.write_text(cleaned, encoding="utf-8")
                files_modified += 1
                entries_removed += removed
                logger.info(f"Stripped {removed} known proof(s) from brainstorm file: {path.name}")
        except Exception as exc:
            logger.warning(f"Skipped brainstorm file {path}: {exc}")

    # Clean paper files
    paper_paths = list(paper_library._base_dir.rglob("paper_*.txt")) if hasattr(paper_library, '_base_dir') else []
    for path in paper_paths:
        try:
            files_checked += 1
            text = path.read_text(encoding="utf-8")
            cleaned, removed = _clean_content(text, "=== PROOFS GENERATED FROM THIS PAPER (Lean 4 Verified) ===")
            if removed > 0:
                path.write_text(cleaned, encoding="utf-8")
                files_modified += 1
                entries_removed += removed
                logger.info(f"Stripped {removed} known proof(s) from paper file: {path.name}")
        except Exception as exc:
            logger.warning(f"Skipped paper file {path}: {exc}")

    return {
        "files_checked": files_checked,
        "files_modified": files_modified,
        "entries_removed": entries_removed,
        "message": (
            f"Removed {entries_removed} non-novel proof entries from {files_modified} file(s). "
            "Proof data is retained in ProofDatabase."
        ),
    }


@router.post("/cleanup-known-from-files")
async def cleanup_known_proofs_from_files(confirm: bool = Query(default=False)):
    """One-time cleanup: strip non-novel proof entries from brainstorm/paper files.

    Non-novel proofs are stored in ProofDatabase (no data loss).  This endpoint
    removes their raw Lean 4 code from brainstorm and paper .txt files so that
    compiler and RAG context is no longer polluted by standard known results.

    Requires explicit confirmation because it mutates brainstorm/paper files.
    Novel proof entries are preserved.
    """
    if system_config.generic_mode:
        raise HTTPException(
            status_code=501,
            detail={
                "lean4_enabled": False,
                "message": "Proof file cleanup is unavailable in hosted mode.",
            },
        )
    if not system_config.lean4_enabled:
        raise HTTPException(
            status_code=501,
            detail={
                "lean4_enabled": False,
                "message": "Proof file cleanup is unavailable while Lean 4 is disabled.",
            },
        )
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Pass ?confirm=true to strip known proof entries from brainstorm and paper files.",
        )

    result = await _strip_known_proofs_from_files()
    return result


@router.get("/status")
async def get_proofs_status():
    """Return Lean 4 availability and proof-database status.

    Non-blocking: Lean workspace checks use a short timeout so the
    endpoint always returns quickly even when Lean is unavailable.
    """
    version = ""
    workspace_ready = False
    mathlib_commit = ""
    lsp_active = False
    z3_version = ""
    smt_available = False
    manual_check_ready, manual_check_message = await _get_manual_check_status()
    if system_config.lean4_enabled:
        try:
            client = get_lean4_client()
            version = await asyncio.wait_for(client.get_version(), timeout=5.0)
            workspace_ready = await asyncio.wait_for(client.ensure_workspace(), timeout=5.0)
            mathlib_commit = client.get_mathlib_commit()
            lsp_active = client.is_server_active()
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Lean 4 status check timed out or failed: %s", exc)
        if manual_check_ready:
            version_text = (version or "").strip().lower()
            version_unavailable = (
                not version_text
                or "not found" in version_text
                or "no such file" in version_text
                or "not recognized" in version_text
            )
            if version_unavailable:
                manual_check_ready = False
                manual_check_message = "Lean 4 executable is not available."
            elif not workspace_ready:
                manual_check_ready = False
                manual_check_message = "Lean 4 workspace is not ready yet."

    if system_config.smt_enabled:
        try:
            z3_version = await asyncio.wait_for(get_smt_client().get_version(), timeout=3.0)
            lowered_version = z3_version.lower()
            smt_available = bool(z3_version) and "not found" not in lowered_version and "no such file" not in lowered_version
        except Exception as exc:
            logger.warning("Failed to resolve Z3 status: %s", exc)

    return {
        "lean4_enabled": system_config.lean4_enabled,
        "lean4_lsp_enabled": system_config.lean4_lsp_enabled,
        "lean4_path": _safe_path_label(system_config.lean4_path),
        "lean4_path_configured": bool(system_config.lean4_path),
        "lean4_workspace_dir": _safe_path_label(system_config.lean4_workspace_dir),
        "lean4_workspace_configured": bool(system_config.lean4_workspace_dir),
        "runtime_paths_redacted": True,
        "lean_version": version,
        "lean4_version": version,
        "lean4_proof_timeout": system_config.lean4_proof_timeout,
        "lean4_lsp_idle_timeout": system_config.lean4_lsp_idle_timeout,
        "proof_max_parallel_candidates": system_config.proof_max_parallel_candidates,
        "lsp_available": bool(system_config.lean4_enabled and system_config.lean4_lsp_enabled),
        "lsp_active": lsp_active,
        "workspace_ready": workspace_ready,
        "mathlib_commit": mathlib_commit,
        "smt_enabled": system_config.smt_enabled,
        "smt_available": smt_available,
        "z3_path": _safe_path_label(system_config.z3_path),
        "z3_path_configured": bool(system_config.z3_path),
        "smt_timeout": system_config.smt_timeout,
        "z3_version": z3_version,
        "manual_check_ready": manual_check_ready,
        "manual_check_message": manual_check_message,
        "proof_counts": proof_database.count_proofs(),
    }


@router.post("/settings")
async def update_proof_settings(request: ProofSettingsUpdateRequest):
    """Update runtime Lean 4 proof settings for the current backend process."""
    if system_config.generic_mode:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Lean 4 settings are unavailable in hosted mode."})

    previous_lean_settings = (
        system_config.lean4_enabled,
        system_config.lean4_lsp_enabled,
        system_config.lean4_lsp_idle_timeout,
        system_config.lean4_path,
        system_config.lean4_workspace_dir,
    )
    previous_smt_settings = (
        system_config.smt_enabled,
        system_config.smt_timeout,
    )

    system_config.lean4_enabled = bool(request.enabled)
    system_config.lean4_proof_timeout = int(request.timeout)
    if request.lean4_lsp_enabled is not None:
        system_config.lean4_lsp_enabled = bool(request.lean4_lsp_enabled)
    if request.lean4_lsp_idle_timeout is not None:
        system_config.lean4_lsp_idle_timeout = int(request.lean4_lsp_idle_timeout)
    if request.max_parallel_candidates is not None:
        system_config.proof_max_parallel_candidates = int(request.max_parallel_candidates)
    if request.smt_enabled is not None:
        system_config.smt_enabled = bool(request.smt_enabled)
    if request.smt_timeout is not None:
        system_config.smt_timeout = int(request.smt_timeout)

    lean_settings_changed = previous_lean_settings != (
        system_config.lean4_enabled,
        system_config.lean4_lsp_enabled,
        system_config.lean4_lsp_idle_timeout,
        system_config.lean4_path,
        system_config.lean4_workspace_dir,
    )
    smt_settings_changed = previous_smt_settings != (
        system_config.smt_enabled,
        system_config.smt_timeout,
    )

    if lean_settings_changed:
        await close_lean4_client()
        clear_lean4_client()
        if system_config.lean4_enabled:
            client = initialize_lean4_client()
            _schedule_lean4_warm_start(client)

    if smt_settings_changed:
        clear_smt_client()

    try:
        save_proof_runtime_settings()
    except RuntimeSettingsError as exc:
        logger.error("Failed to persist proof runtime settings: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist proof runtime settings")

    return await get_proofs_status()


@router.post("/check")
async def run_manual_proof_check(request: ProofCheckRequest, background_tasks: BackgroundTasks):
    """Queue a user-triggered proof check for one brainstorm or paper."""
    if not system_config.lean4_enabled:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Lean 4 proof checks are disabled."})

    snapshot = await _get_runtime_snapshot(request)
    if snapshot is None:
        raise HTTPException(
            status_code=409,
            detail="No proof runtime model configuration is available yet. Start autonomous research once before using manual proof checks.",
        )
    selected_role = snapshot.brainstorm if request.source_type == "brainstorm" else snapshot.paper
    if not selected_role.model_id or not snapshot.validator.model_id:
        raise HTTPException(
            status_code=409,
            detail="Proof runtime model configuration is incomplete. Select models for the proof role and validator, then try again.",
        )

    await _resolve_manual_source(request)
    try:
        await ProofVerificationStage.reserve_source(request.source_type, request.source_id)
    except RuntimeError:
        raise HTTPException(status_code=409, detail="A proof verification is already running for that source.")

    background_tasks.add_task(_run_manual_proof_check, request)
    return {
        "queued": True,
        "source_type": request.source_type,
        "source_id": request.source_id,
    }


@router.get("/library")
async def get_proof_library(novel_only: bool = True):
    """Return all proofs across all sessions for the proof library browser."""
    proofs = await proof_database.list_proof_library(novel_only=novel_only)
    novel_count = sum(1 for p in proofs if p.get("novel"))
    return {
        "proofs": proofs,
        "counts": {
            "total": len(proofs) if not novel_only else None,
            "listed": len(proofs),
            "novel": novel_count,
        },
    }


@router.get("/library/{session_id}/{proof_id}")
async def get_library_proof(session_id: str, proof_id: str):
    """Return a single proof from a specific session with full Lean code."""
    proof = await proof_database.get_library_proof(session_id, proof_id)
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")
    return proof


@router.get("/{proof_id}/certificate")
async def get_proof_certificate(proof_id: str):
    """Return a machine-readable proof certificate JSON payload."""
    proof = await _get_export_proof_or_404(proof_id)

    lean_version = ""
    mathlib_commit = ""
    if system_config.lean4_enabled:
        try:
            client = get_lean4_client()
            lean_version = await asyncio.wait_for(client.get_version(), timeout=5.0)
            mathlib_commit = client.get_mathlib_commit()
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Lean 4 certificate metadata lookup timed out or failed: %s", exc)

    lean_code = await _get_export_lean_code(proof_id)
    payload = {
        "proof_id": proof.proof_id,
        "theorem_statement": proof.theorem_statement,
        "theorem_name": proof.theorem_name,
        "lean_code": lean_code,
        "solver": proof.solver or "Lean 4",
        "lean_version": lean_version,
        "mathlib_commit": mathlib_commit,
        "verified_at": proof.created_at.isoformat() if proof.created_at else None,
        "source_type": proof.source_type,
        "source_id": proof.source_id,
        "source_title": proof.source_title,
        "novel": proof.novel,
        "novelty_reasoning": proof.novelty_reasoning,
        "attempt_count": proof.attempt_count,
        "solver_hints": list(proof.solver_hints or []),
        "dependencies": [dependency.model_dump(mode="json") for dependency in (proof.dependencies or [])],
    }
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": f'attachment; filename="{proof_id}_certificate.json"',
        },
    )


@router.get("/{proof_id}/certificate.lean")
async def get_proof_certificate_lean(proof_id: str):
    """Return the raw saved Lean file for a proof."""
    proof = await _get_export_proof_or_404(proof_id)

    lean_code = await _get_export_lean_code(proof_id)
    return PlainTextResponse(
        content=lean_code or proof.lean_code,
        headers={
            "Content-Disposition": f'attachment; filename="{proof_id}.lean"',
        },
    )


@router.get("/{proof_id}/dependencies")
async def get_proof_dependencies(proof_id: str):
    """Return one proof's dependency edges plus reverse MOTO ancestry."""
    if not system_config.lean4_enabled:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Proof dependency data is unavailable while Lean 4 is disabled."})

    proof = await proof_database.get_proof(proof_id)
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")

    dependencies = await proof_database.get_dependencies(proof_id)
    reverse_dependencies = await proof_database.get_proofs_depending_on(proof_id)
    mathlib_reverse_usage = []
    seen_mathlib_names = set()
    for dependency in dependencies:
        if dependency.kind != "mathlib" or not dependency.name or dependency.name in seen_mathlib_names:
            continue
        seen_mathlib_names.add(dependency.name)
        dependents = [
            dependent
            for dependent in await proof_database.get_proofs_using_mathlib(dependency.name)
            if dependent.proof_id != proof.proof_id
        ]
        if not dependents:
            continue
        mathlib_reverse_usage.append(
            {
                "name": dependency.name,
                "source_ref": dependency.source_ref,
                "dependents": [
                    {
                        "proof_id": dependent.proof_id,
                        "theorem_name": dependent.theorem_name,
                        "theorem_statement": dependent.theorem_statement,
                        "source_type": dependent.source_type,
                        "source_id": dependent.source_id,
                    }
                    for dependent in dependents
                ],
            }
        )
    return {
        "proof_id": proof.proof_id,
        "depends_on": [dependency.model_dump(mode="json") for dependency in dependencies],
        "depended_on_by": [
            {
                "proof_id": dependent.proof_id,
                "theorem_name": dependent.theorem_name,
                "theorem_statement": dependent.theorem_statement,
                "source_type": dependent.source_type,
                "source_id": dependent.source_id,
            }
            for dependent in reverse_dependencies
        ],
        "mathlib_depended_on_by": mathlib_reverse_usage,
    }


@router.get("/graph")
async def get_proof_graph():
    """Return the full proof dependency graph in one payload."""
    if not system_config.lean4_enabled:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Proof dependency data is unavailable while Lean 4 is disabled."})

    graph = await proof_database.get_graph()
    return {
        **graph,
        "proof_counts": proof_database.count_proofs(),
    }


@router.get("/mathlib/{lemma_name}/dependents")
async def get_mathlib_dependents(lemma_name: str):
    """Return proofs that depend on one Mathlib declaration."""
    if not system_config.lean4_enabled:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Proof dependency data is unavailable while Lean 4 is disabled."})

    dependents = await proof_database.get_proofs_using_mathlib(lemma_name)
    return {
        "name": lemma_name,
        "dependents": [
            {
                "proof_id": dependent.proof_id,
                "theorem_name": dependent.theorem_name,
                "theorem_statement": dependent.theorem_statement,
                "source_type": dependent.source_type,
                "source_id": dependent.source_id,
            }
            for dependent in dependents
        ],
    }


@router.get("/{proof_id}")
async def get_proof(proof_id: str):
    """Return a single proof record with full Lean code."""
    proof = await proof_database.get_proof(proof_id)
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")
    return proof.model_dump(mode="json")
