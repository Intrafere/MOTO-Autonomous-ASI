"""
Proof database, Lean 4 status, manual proof checks, and certificate export routes.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.api.routes import websocket
from backend.aggregator.core.coordinator import coordinator
from backend.aggregator.memory.event_log import event_log
from backend.aggregator.memory.shared_training import (
    append_proof_to_manual_shared_training,
    load_manual_aggregator_prompt,
    shared_training_memory,
)
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory, brainstorm_memory
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.proof_database import ProofDatabase, manual_proof_database, proof_database
from backend.autonomous.memory.research_metadata import research_metadata
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.compiler.memory.manual_prompt import load_manual_compiler_prompt
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import paper_memory
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
from backend.shared.manual_proof_context import get_manual_proof_context_lock
from backend.shared.path_safety import resolve_path_within_root
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.proof_search.assistant_models import AssistantTargetSnapshot
from backend.shared.runtime_settings import RuntimeSettingsError, save_proof_runtime_settings
from backend.shared.smt_client import clear_smt_client, get_smt_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proofs", tags=["proofs"])

MANUAL_AGGREGATOR_SOURCE_ID = "manual_aggregator"
MANUAL_COMPILER_CURRENT_SOURCE_ID = "manual_compiler_current"
PROOF_SCOPE_AUTONOMOUS = "autonomous"
PROOF_SCOPE_MANUAL = "manual"
_manual_proof_run_lock = asyncio.Lock()
_LEAN_STATUS_STARTING_LOG_INTERVAL_SECONDS = 60.0
_last_lean_status_starting_log_at = 0.0
_ASSISTANT_MANUAL_SOURCE_SUMMARY_CHARS = 8000


def _log_lean_status_starting_up(detail: str) -> None:
    """Avoid noisy startup warnings while Lean is bootstrapping its workspace."""
    global _last_lean_status_starting_log_at
    now = time.monotonic()
    if now - _last_lean_status_starting_log_at < _LEAN_STATUS_STARTING_LOG_INTERVAL_SECONDS:
        return
    _last_lean_status_starting_log_at = now
    logger.info(
        "Lean 4 is still starting up; proof status will become ready after workspace bootstrap completes. %s",
        detail,
    )


def _manual_proof_history_root() -> Path:
    return Path(system_config.data_dir) / "manual_proof_runs"


def _is_non_appending_manual_source(request: ProofCheckRequest) -> bool:
    return (
        (request.source_type == "brainstorm" and request.source_id == MANUAL_AGGREGATOR_SOURCE_ID)
        or (request.source_type == "paper" and request.source_id == MANUAL_COMPILER_CURRENT_SOURCE_ID)
    )


async def _append_manual_aggregator_proof(proof_record) -> bool:
    """Append a manual Aggregator proof to the manual DB, even if another mode moved the singleton path."""
    return await append_proof_to_manual_shared_training(proof_record)


async def _append_manual_compiler_current_proof(proof_record) -> bool:
    """Append a user-triggered proof to the current manual Compiler paper."""
    current_paper = await paper_memory.get_paper()
    if not current_paper.strip():
        return False
    updated_paper = paper_library.attach_verified_proofs_to_content(
        current_paper,
        proof_record,
        "the current manual Compiler paper",
    )
    if updated_paper == current_paper:
        return True
    await paper_memory.update_paper(updated_paper)
    return True


def _manual_append_callback(request: ProofCheckRequest):
    if request.source_type == "brainstorm" and request.source_id == MANUAL_AGGREGATOR_SOURCE_ID:
        return _append_manual_aggregator_proof
    if request.source_type == "paper" and request.source_id == MANUAL_COMPILER_CURRENT_SOURCE_ID:
        return _append_manual_compiler_current_proof
    return None


def _is_manual_aggregator_request(request: ProofCheckRequest) -> bool:
    return request.source_type == "brainstorm" and request.source_id == MANUAL_AGGREGATOR_SOURCE_ID


def _manual_aggregator_proof_event_message(event_type: str, data: dict) -> str:
    target = (
        data.get("theorem_name")
        or data.get("proof_label")
        or data.get("theorem_id")
        or data.get("proof_id")
        or "candidate"
    )

    def _compact(value: object, limit: int = 1200) -> str:
        cleaned = " ".join(str(value or "").split())
        if not cleaned:
            return ""
        return cleaned[:limit] + ("..." if len(cleaned) > limit else "")

    def _lean_response() -> str:
        if data.get("lean_response"):
            return _compact(data.get("lean_response"))
        if data.get("proof_verified") is True:
            return "Lean 4 response: proof verified."
        error = _compact(
            data.get("error_summary") or data.get("error_output") or data.get("reason"),
            limit=960,
        )
        return f"Lean 4 response: {error} - proof not verified." if error else ""

    def _attempt_message(prefix: str) -> str:
        attempt = f", attempt {data.get('attempt')}" if data.get("attempt") else ""
        response = _lean_response()
        base = f"{prefix}: {target}{attempt}"
        return f"{base} - {response}" if response else base

    if event_type == "proof_check_started":
        return "Proof check started for the manual Aggregator database"
    if event_type == "proof_check_no_candidates":
        return "No formal theorem candidates found in the manual Aggregator database"
    if event_type == "proof_check_candidates_found":
        return f"Proof candidates found: {data.get('count') or 0}"
    if event_type == "proof_attempt_started":
        return f"Lean proof attempt started: {target}"
    if event_type == "proof_lean_accepted":
        return f"Lean accepted proof: {target}"
    if event_type == "proof_attempt_failed":
        return _attempt_message("Proof attempt failed")
    if event_type == "proof_attempts_exhausted":
        return _attempt_message("Proof attempts exhausted")
    if event_type == "proof_integrity_rejected":
        return f"Proof integrity rejected: {data.get('reason') or data.get('message') or target}"
    if event_type == "proof_verified":
        return f"Proof verified: {target}"
    if event_type == "known_proof_verified":
        return f"Known proof verified: {target}"
    if event_type == "proof_registration_duplicate":
        return f"Duplicate proof reused: {target}"
    if event_type == "novel_proof_discovered":
        return f"Novel proof discovered: {target}"
    if event_type == "proof_dependency_added":
        return f"Proof dependency added: {target}"
    if event_type == "proof_check_complete":
        return f"Proof check complete: {data.get('verified_count') or 0} verified, {data.get('novel_count') or 0} novel"
    return f"Proof event: {event_type}"


async def _broadcast_manual_aggregator_proof_event(event_type: str, data: dict) -> None:
    """Broadcast and durably log manual Aggregator proof activity."""
    enriched_data = {
        **(data or {}),
        "manual_event_id": f"manual-aggregator-proof-{uuid.uuid4().hex}",
    }
    await websocket.broadcast_event(event_type, enriched_data)
    try:
        await event_log.add_event(
            event_type,
            _manual_aggregator_proof_event_message(event_type, enriched_data),
            enriched_data,
        )
    except Exception as exc:
        logger.warning("Failed to persist manual Aggregator proof event %s: %s", event_type, exc)


def _get_scoped_proof_database(scope: str = PROOF_SCOPE_AUTONOMOUS) -> ProofDatabase:
    normalized = (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower()
    if normalized == PROOF_SCOPE_MANUAL:
        return manual_proof_database
    if normalized != PROOF_SCOPE_AUTONOMOUS:
        raise HTTPException(status_code=400, detail="Proof scope must be 'autonomous' or 'manual'.")
    return proof_database


def _get_request_proof_database(request: ProofCheckRequest) -> ProofDatabase:
    if (
        (request.source_type == "brainstorm" and request.source_id == MANUAL_AGGREGATOR_SOURCE_ID)
        or (request.source_type == "paper" and request.source_id == MANUAL_COMPILER_CURRENT_SOURCE_ID)
    ):
        return manual_proof_database
    return proof_database


def _schedule_lean4_warm_start(client) -> None:
    """Warm the Lean workspace without blocking a settings/status request."""
    async def _warm_start() -> None:
        try:
            await client.warm_start()
        except Exception as exc:  # pragma: no cover - defensive background task
            logger.warning("Lean 4 client warm start failed: %s", exc)

    asyncio.create_task(_warm_start())


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


async def _get_export_proof_or_404(proof_id: str, scoped_proof_database: ProofDatabase = proof_database):
    try:
        proof = await scoped_proof_database.get_proof(proof_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Proof not found")
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")
    return proof


async def _get_export_lean_code(
    proof_id: str,
    scoped_proof_database: ProofDatabase = proof_database,
) -> str:
    try:
        return await scoped_proof_database.get_lean_code(proof_id)
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


def _role_config_from_model_config(config: Optional[ModelConfig]) -> ProofRoleConfigSnapshot:
    if config is None:
        return ProofRoleConfigSnapshot()
    return ProofRoleConfigSnapshot(
        provider=config.provider,
        model_id=config.model_id,
        openrouter_provider=config.openrouter_provider,
        openrouter_reasoning_effort=config.openrouter_reasoning_effort,
        lm_studio_fallback_id=config.lm_studio_fallback_id,
        context_window=config.context_window,
        max_output_tokens=config.max_output_tokens,
        supercharge_enabled=config.supercharge_enabled,
    )


def _get_active_manual_runtime_snapshot(request: ProofCheckRequest) -> Optional[ProofRuntimeConfigSnapshot]:
    """Build proof runtime settings from the active manual mode, never from autonomous presets."""
    if request.source_type == "brainstorm" and request.source_id == MANUAL_AGGREGATOR_SOURCE_ID:
        if not coordinator.submitter_configs or not coordinator.validator_model:
            return None

        first_submitter = coordinator.submitter_configs[0]
        submitter_role = ProofRoleConfigSnapshot(
            provider=first_submitter.provider,
            model_id=first_submitter.model_id,
            openrouter_provider=first_submitter.openrouter_provider,
            openrouter_reasoning_effort=first_submitter.openrouter_reasoning_effort,
            lm_studio_fallback_id=first_submitter.lm_studio_fallback_id,
            context_window=first_submitter.context_window,
            max_output_tokens=first_submitter.max_output_tokens,
            supercharge_enabled=first_submitter.supercharge_enabled,
        )
        validator_role = ProofRoleConfigSnapshot(
            provider=coordinator.validator_provider,
            model_id=coordinator.validator_model,
            openrouter_provider=coordinator.validator_openrouter_provider,
            openrouter_reasoning_effort=coordinator.validator_openrouter_reasoning_effort,
            lm_studio_fallback_id=coordinator.validator_lm_studio_fallback,
            context_window=coordinator.validator_context_window,
            max_output_tokens=coordinator.validator_max_tokens,
            supercharge_enabled=coordinator.validator_supercharge_enabled,
        )
        return ProofRuntimeConfigSnapshot(
            brainstorm=submitter_role,
            paper=submitter_role,
            validator=validator_role,
            assistant=_role_config_from_model_config(
                api_client_manager.get_role_config("aggregator_assistant")
            ),
        )

    if request.source_type == "paper" and request.source_id == MANUAL_COMPILER_CURRENT_SOURCE_ID:
        rigor_submitter = compiler_coordinator.high_param_submitter
        if rigor_submitter is None or not getattr(rigor_submitter, "model_name", "") or not compiler_coordinator.validator_model:
            return None

        paper_role = ProofRoleConfigSnapshot(
            provider=compiler_coordinator.high_param_provider,
            model_id=rigor_submitter.model_name,
            openrouter_provider=compiler_coordinator.high_param_openrouter_provider,
            openrouter_reasoning_effort=compiler_coordinator.high_param_openrouter_reasoning_effort,
            lm_studio_fallback_id=compiler_coordinator.high_param_lm_studio_fallback,
            context_window=system_config.compiler_high_param_context_window,
            max_output_tokens=system_config.compiler_high_param_max_output_tokens,
            supercharge_enabled=compiler_coordinator.high_param_supercharge_enabled,
        )
        validator_role = ProofRoleConfigSnapshot(
            provider=compiler_coordinator.validator_provider,
            model_id=compiler_coordinator.validator_model,
            openrouter_provider=compiler_coordinator.validator_openrouter_provider,
            openrouter_reasoning_effort=compiler_coordinator.validator_openrouter_reasoning_effort,
            lm_studio_fallback_id=compiler_coordinator.validator_lm_studio_fallback,
            context_window=compiler_coordinator.validator_context_window,
            max_output_tokens=compiler_coordinator.validator_max_tokens,
            supercharge_enabled=compiler_coordinator.validator_supercharge_enabled,
        )
        return ProofRuntimeConfigSnapshot(
            brainstorm=paper_role,
            paper=paper_role,
            validator=validator_role,
            assistant=_role_config_from_model_config(
                api_client_manager.get_role_config("compiler_assistant")
            ),
        )

    return None


async def _get_runtime_snapshot(request: Optional[ProofCheckRequest] = None) -> Optional[ProofRuntimeConfigSnapshot]:
    if request and _is_non_appending_manual_source(request):
        request_snapshot = _get_request_runtime_snapshot(request)
        # Active manual sources must not borrow autonomous proof settings.
        # Prefer the backend's live manual runtime so stale browser/localStorage
        # snapshots cannot override the roles that actually produced the source.
        active_manual_snapshot = _get_active_manual_runtime_snapshot(request)
        if active_manual_snapshot is not None:
            return active_manual_snapshot
        return request_snapshot

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
    assistant_config = snapshot.assistant if snapshot.assistant.model_id else snapshot.validator
    api_client_manager.configure_role(
        "manual_proof_assistant",
        _build_model_config(assistant_config),
    )
    return role_config


def _compact_manual_assistant_source(content: str) -> str:
    text = " ".join((content or "").split())
    if len(text) <= _ASSISTANT_MANUAL_SOURCE_SUMMARY_CHARS:
        return text
    return text[:_ASSISTANT_MANUAL_SOURCE_SUMMARY_CHARS].rstrip() + "..."


async def _refresh_manual_assistant_memory(
    *,
    source_type: str,
    source_id: str,
    source_title: str,
    source_content: str,
    user_prompt: str,
) -> None:
    """Run Try-to-Prove Assistant memory even before proof prompt preflight.

    Manual proof discovery may fail during mandatory-source context validation
    before it reaches ``api_client_manager.generate_completion()``, so the
    normal central Assistant injection hook never fires. This preflight refresh
    keeps the user-triggered proof-check button covered by Assistant memory and
    leaves a visible log/event trail.
    """
    if not system_config.agent_conversation_memory_enabled:
        logger.info(
            "Assistant memory preflight skipped for manual proof check %s:%s because Agent Conversation Memory is disabled",
            source_type,
            source_id,
        )
        return

    snapshot = AssistantTargetSnapshot(
        workflow_mode="manual_proof_check",
        target_kind="proof_candidate",
        workflow_phase="manual_try_to_prove",
        active_mode="manual_proof_check",
        user_prompt=user_prompt,
        current_prompt_or_topic=source_title,
        current_submission_or_draft=_compact_manual_assistant_source(source_content),
        writing_goal="User-triggered Try to Prove This proof discovery over the selected source.",
        paper_or_proof_draft_summary=_compact_manual_assistant_source(source_content),
        target_statement=user_prompt or source_title or f"{source_type}:{source_id}",
        formal_sketch=_compact_manual_assistant_source(source_content),
        source_title=source_title,
        source_type=f"manual_{source_type}",
        source_id=source_id,
        source_titles=[source_title] if source_title else [],
        imports=["Mathlib"],
    )
    logger.info(
        "Assistant memory preflight starting for manual proof check %s:%s (%s)",
        source_type,
        source_id,
        source_title or "untitled source",
    )
    pack = await assistant_proof_search_coordinator.refresh_now(snapshot)
    logger.info(
        "Assistant memory preflight complete for manual proof check %s:%s (results=%s)",
        source_type,
        source_id,
        len(pack.results) if pack else 0,
    )


async def _prompt_with_verified_proof_context(
    prompt: str,
    scoped_proof_database: ProofDatabase = proof_database,
) -> str:
    """Apply proof-library context to a source-specific manual proof prompt."""
    source_prompt = (prompt or "").strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_user_prompt()).strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_base_user_prompt()).strip()
    return scoped_proof_database.inject_into_prompt(source_prompt)


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


async def _read_manual_aggregator_content(*, formatted: bool = True, strip_proofs: bool = False) -> str:
    """Read the live/manual Aggregator database without mutating its run state."""
    try:
        manual_path = Path(system_config.shared_training_file)
        if Path(shared_training_memory.file_path) == manual_path:
            content = (
                await shared_training_memory.get_all_content_formatted(strip_proofs=strip_proofs)
                if formatted
                else await shared_training_memory.get_all_content(strip_proofs=strip_proofs)
            )
        else:
            content = ""
    except Exception as exc:
        logger.debug("Unable to read manual Aggregator memory: %s", exc)
        content = ""

    if content.strip():
        return content

    try:
        shared_path = Path(system_config.shared_training_file)
        if shared_path.exists():
            content = await asyncio.to_thread(shared_path.read_text, encoding="utf-8")
            if strip_proofs and "=== PROOFS GENERATED FROM THIS BRAINSTORM" in content:
                content = content.split("=== PROOFS GENERATED FROM THIS BRAINSTORM", 1)[0].rstrip()
            return content
    except Exception as exc:
        logger.debug("Unable to read manual Aggregator file: %s", exc)
    return ""


async def _manual_aggregator_prompt() -> str:
    try:
        prompt = (coordinator.validator.user_prompt if coordinator.validator else "") or ""
    except Exception:
        prompt = ""
    if prompt.strip():
        return prompt
    return await load_manual_aggregator_prompt()


async def _resolve_manual_aggregator_source(
    scoped_proof_database: ProofDatabase = manual_proof_database,
) -> Tuple[str, str, str]:
    content = await _read_manual_aggregator_content(formatted=True, strip_proofs=True)
    if not content.strip():
        raise HTTPException(status_code=404, detail="Manual Aggregator database is empty")
    user_prompt = await _prompt_with_verified_proof_context(
        await _manual_aggregator_prompt(),
        scoped_proof_database,
    )
    return content, "Manual Aggregator Database", user_prompt


async def _resolve_manual_compiler_current_source(
    scoped_proof_database: ProofDatabase = manual_proof_database,
) -> Tuple[str, str, str]:
    paper = paper_library.strip_verified_proofs_from_content(await paper_memory.get_paper())
    if not paper.strip():
        raise HTTPException(status_code=404, detail="Manual Compiler paper content not found")

    outline = await outline_memory.get_outline()
    source_context = await _read_manual_aggregator_content(formatted=False, strip_proofs=True)
    parts = []
    if outline.strip():
        parts.append(f"CURRENT MANUAL COMPILER OUTLINE:\n{outline.strip()}")
    parts.append(f"CURRENT MANUAL COMPILER PAPER:\n{paper.strip()}")
    if source_context.strip():
        parts.append(f"PART 1 AGGREGATOR DATABASE CONTEXT:\n{source_context.strip()}")

    persisted_prompt = compiler_coordinator.user_prompt or await load_manual_compiler_prompt()
    user_prompt = await _prompt_with_verified_proof_context(
        persisted_prompt,
        scoped_proof_database,
    )
    source_title = compiler_coordinator.paper_title or persisted_prompt or "Manual Compiler Paper"
    return "\n\n---\n\n".join(parts), source_title, user_prompt


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


async def _resolve_manual_source(
    request: ProofCheckRequest,
    scoped_proof_database: Optional[ProofDatabase] = None,
) -> Tuple[str, str, str]:
    if scoped_proof_database is None:
        scoped_proof_database = proof_database

    if request.source_type == "brainstorm":
        if request.source_id == MANUAL_AGGREGATOR_SOURCE_ID:
            return await _resolve_manual_aggregator_source(scoped_proof_database)

        metadata = await brainstorm_memory.get_metadata(request.source_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Brainstorm not found")
        content = await brainstorm_memory.get_database_content(
            request.source_id,
            strip_proofs=True,
        )
        if not content:
            raise HTTPException(status_code=404, detail="Brainstorm content not found")
        user_prompt = await _prompt_with_verified_proof_context(
            await research_metadata.get_user_prompt(),
            scoped_proof_database,
        )
        return content, metadata.topic_prompt, user_prompt

    if request.source_id == MANUAL_COMPILER_CURRENT_SOURCE_ID:
        return await _resolve_manual_compiler_current_source(scoped_proof_database)

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
    user_prompt = await _prompt_with_verified_proof_context(
        await research_metadata.get_user_prompt(),
        scoped_proof_database,
    )
    return content, metadata.title, user_prompt


async def _run_manual_proof_check(request: ProofCheckRequest) -> None:
    source_title = ""
    scoped_proof_database = _get_request_proof_database(request)
    try:
        source_content, source_title, user_prompt = await _resolve_manual_source(
            request,
            scoped_proof_database,
        )
        snapshot = await _get_runtime_snapshot(request)
        if snapshot is None:
            if _is_non_appending_manual_source(request):
                raise RuntimeError(
                    "No manual proof runtime model configuration is available for this source. "
                    "Start the manual Aggregator or Single Paper Writer with configured proof roles, "
                    "or retry from a browser session with complete manual role settings."
                )
            raise RuntimeError("No proof runtime model configuration is available yet.")

        async with _manual_proof_run_lock:
            role_config = _configure_manual_roles(request.source_type, snapshot)
            stage = autonomous_coordinator._proof_verification_stage
            broadcast_fn = (
                _broadcast_manual_aggregator_proof_event
                if _is_manual_aggregator_request(request)
                else websocket.broadcast_event
            )
            await _refresh_manual_assistant_memory(
                source_type=request.source_type,
                source_id=request.source_id,
                source_title=source_title,
                source_content=source_content,
                user_prompt=user_prompt,
            )
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
                broadcast_fn=broadcast_fn,
                novel_proofs_db=scoped_proof_database,
                source_title=source_title,
                source_reserved=True,
                append_to_source=not _is_non_appending_manual_source(request),
                append_proof_callback=_manual_append_callback(request),
            )
    except Exception as exc:
        logger.exception("Manual proof check failed for %s %s", request.source_type, request.source_id)
        broadcast_fn = (
            _broadcast_manual_aggregator_proof_event
            if _is_manual_aggregator_request(request)
            else websocket.broadcast_event
        )
        await broadcast_fn(
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
    finally:
        await assistant_proof_search_coordinator.stop_all(
            broadcast=True,
            reason="manual_proof_check_complete",
        )


@router.get("")
async def list_proofs(scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS)):
    """Return all verified proofs plus aggregate counts."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proofs = await scoped_proof_database.get_all_proofs()
    return {
        "proofs": [proof.model_dump(mode="json") for proof in proofs],
        "counts": scoped_proof_database.count_proofs(),
        "scope": (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower(),
    }


@router.get("/novel")
async def list_novel_proofs(scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS)):
    """Return only novel verified proofs."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proofs = await scoped_proof_database.get_all_proofs(novel_only=True)
    return {
        "proofs": [proof.model_dump(mode="json") for proof in proofs],
        "counts": scoped_proof_database.count_proofs(),
        "scope": (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower(),
    }


@router.get("/known")
async def list_known_proofs(scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS)):
    """Return only known (non-novel) verified proofs."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proofs = await scoped_proof_database.get_all_proofs(novel_only=False)
    return {
        "proofs": [proof.model_dump(mode="json") for proof in proofs],
        "counts": scoped_proof_database.count_proofs(),
        "scope": (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower(),
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
    lean_status_starting_up = False
    manual_check_ready, manual_check_message = await _get_manual_check_status()
    if system_config.lean4_enabled:
        try:
            client = get_lean4_client()
            version = await asyncio.wait_for(client.get_version(), timeout=5.0)
            workspace_ready = await asyncio.wait_for(client.ensure_workspace(), timeout=5.0)
            mathlib_commit = client.get_mathlib_commit()
            lsp_active = client.is_server_active()
        except asyncio.TimeoutError:
            lean_status_starting_up = True
            _log_lean_status_starting_up("The latest status check is waiting on startup work.")
        except Exception as exc:
            lean_status_starting_up = True
            _log_lean_status_starting_up(f"Latest status detail: {exc}")
        if manual_check_ready:
            version_text = (version or "").strip().lower()
            version_unavailable = (
                not version_text
                or "not found" in version_text
                or "no such file" in version_text
                or "not recognized" in version_text
            )
            if lean_status_starting_up:
                manual_check_ready = False
                manual_check_message = "Lean 4 is still starting up."
            elif version_unavailable:
                manual_check_ready = False
                manual_check_message = "Lean 4 executable is not available."
            elif not workspace_ready:
                manual_check_ready = False
                manual_check_message = "Lean 4 is still starting up."

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
        "manual_proof_counts": manual_proof_database.count_proofs(),
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
        if _is_non_appending_manual_source(request):
            raise HTTPException(
                status_code=409,
                detail=(
                    "No manual proof runtime model configuration is available for this source. "
                    "Start the manual Aggregator or Single Paper Writer with configured proof roles, "
                    "or retry from a browser session with complete manual role settings."
                ),
            )
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

    async with get_manual_proof_context_lock():
        scoped_proof_database = _get_request_proof_database(request)
        await _resolve_manual_source(request, scoped_proof_database)
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
async def get_proof_library(
    novel_only: bool = True,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return archived proofs for the selected proof-library scope."""
    normalized_scope = (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower()
    if normalized_scope == PROOF_SCOPE_MANUAL:
        proofs = await manual_proof_database.list_proof_library_from_history(
            _manual_proof_history_root(),
            novel_only=novel_only,
        )
    elif normalized_scope == PROOF_SCOPE_AUTONOMOUS:
        proofs = await proof_database.list_proof_library(novel_only=novel_only)
    else:
        raise HTTPException(status_code=400, detail="Proof scope must be 'autonomous' or 'manual'.")
    novel_count = sum(1 for p in proofs if p.get("novel"))
    return {
        "proofs": proofs,
        "counts": {
            "total": len(proofs) if not novel_only else None,
            "listed": len(proofs),
            "novel": novel_count,
        },
        "scope": normalized_scope,
    }


@router.get("/library/{session_id}/{proof_id}")
async def get_library_proof(
    session_id: str,
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return a single archived proof from a specific library scope."""
    normalized_scope = (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower()
    if normalized_scope == PROOF_SCOPE_MANUAL:
        proof = await manual_proof_database.get_library_proof_from_history(
            _manual_proof_history_root(),
            session_id,
            proof_id,
        )
    elif normalized_scope == PROOF_SCOPE_AUTONOMOUS:
        proof = await proof_database.get_library_proof(session_id, proof_id)
    else:
        raise HTTPException(status_code=400, detail="Proof scope must be 'autonomous' or 'manual'.")
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")
    return proof


@router.get("/{proof_id}/certificate")
async def get_proof_certificate(
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return a machine-readable proof certificate JSON payload."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proof = await _get_export_proof_or_404(proof_id, scoped_proof_database)

    lean_version = ""
    mathlib_commit = ""
    if system_config.lean4_enabled:
        try:
            client = get_lean4_client()
            lean_version = await asyncio.wait_for(client.get_version(), timeout=5.0)
            mathlib_commit = client.get_mathlib_commit()
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Lean 4 certificate metadata lookup timed out or failed: %s", exc)

    lean_code = await _get_export_lean_code(proof_id, scoped_proof_database)
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
async def get_proof_certificate_lean(
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return the raw saved Lean file for a proof."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proof = await _get_export_proof_or_404(proof_id, scoped_proof_database)

    lean_code = await _get_export_lean_code(proof_id, scoped_proof_database)
    return PlainTextResponse(
        content=lean_code or proof.lean_code,
        headers={
            "Content-Disposition": f'attachment; filename="{proof_id}.lean"',
        },
    )


@router.get("/{proof_id}/dependencies")
async def get_proof_dependencies(
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return one proof's dependency edges plus reverse MOTO ancestry."""
    if not system_config.lean4_enabled:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Proof dependency data is unavailable while Lean 4 is disabled."})

    scoped_proof_database = _get_scoped_proof_database(scope)
    proof = await scoped_proof_database.get_proof(proof_id)
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")

    dependencies = await scoped_proof_database.get_dependencies(proof_id)
    reverse_dependencies = await scoped_proof_database.get_proofs_depending_on(proof_id)
    mathlib_reverse_usage = []
    seen_mathlib_names = set()
    for dependency in dependencies:
        if dependency.kind != "mathlib" or not dependency.name or dependency.name in seen_mathlib_names:
            continue
        seen_mathlib_names.add(dependency.name)
        dependents = [
            dependent
            for dependent in await scoped_proof_database.get_proofs_using_mathlib(dependency.name)
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
async def get_proof_graph(scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS)):
    """Return the full proof dependency graph in one payload."""
    if not system_config.lean4_enabled:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Proof dependency data is unavailable while Lean 4 is disabled."})

    scoped_proof_database = _get_scoped_proof_database(scope)
    graph = await scoped_proof_database.get_graph()
    return {
        **graph,
        "proof_counts": scoped_proof_database.count_proofs(),
        "scope": (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower(),
    }


@router.get("/mathlib/{lemma_name}/dependents")
async def get_mathlib_dependents(
    lemma_name: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return proofs that depend on one Mathlib declaration."""
    if not system_config.lean4_enabled:
        raise HTTPException(status_code=501, detail={"lean4_enabled": False, "message": "Proof dependency data is unavailable while Lean 4 is disabled."})

    scoped_proof_database = _get_scoped_proof_database(scope)
    dependents = await scoped_proof_database.get_proofs_using_mathlib(lemma_name)
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
async def get_proof(
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return a single proof record with full Lean code."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proof = await scoped_proof_database.get_proof(proof_id)
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")
    return proof.model_dump(mode="json")
