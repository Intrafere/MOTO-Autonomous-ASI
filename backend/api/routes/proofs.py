"""
Proof database, Lean 4 status, manual proof checks, and certificate export routes.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Response
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
from backend.autonomous.core.proof_round_driver import (
    ContinuousPruningPolicy,
    OneRoundPolicy,
    ProofRoundDriver,
    summarize_round_result,
)
from backend.autonomous.core.proof_run_manager import (
    ProofRunControl,
    ProofRunSourceInvalidError,
    proof_run_manager,
)
from backend.autonomous.core.proof_pruning_coordinator import ProofPruningCoordinator
from backend.autonomous.memory.brainstorm_memory import BrainstormMemory, brainstorm_memory
from backend.autonomous.memory.paper_library import PaperLibrary, paper_library
from backend.autonomous.memory.proof_database import ProofDatabase, manual_proof_database, proof_database
from backend.autonomous.memory.proof_database import (
    is_duplicate_novel_tier,
    is_not_novel_tier,
    is_prompt_injection_novel_tier,
    normalize_proof_library_category,
)
from backend.autonomous.memory.research_metadata import research_metadata
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.compiler.memory.manual_prompt import load_manual_compiler_prompt
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import paper_memory
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.config import system_config
from backend.shared.lean4_client import (
    clear_lean4_client,
    close_lean4_client,
    get_lean4_client,
    initialize_lean4_client,
)
from backend.shared.models import (
    CurrentProofListResponse,
    ModelConfig,
    ProofAttemptFeedback,
    ProofCheckRequest,
    ProofCertificateResponse,
    ProofCandidate,
    ProofLiveContextMutationRequest,
    ProofLiveContextMutationResponse,
    ProofLibraryEntry,
    ProofLibraryResponse,
    ProofStageResult,
    ProofRunQueueResponse,
    ProofRunCollectionResponse,
    ProofRunSnapshot,
    ProofRunSourceLookupResponse,
    ProofRunStopRequest,
    ProofRoleConfigSnapshot,
    ProofRuntimeConfigSnapshot,
    ProofSettingsUpdateRequest,
)
from backend.shared.manual_proof_context import get_manual_proof_context_lock
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.path_safety import resolve_path_within_root, validate_single_path_component
from backend.shared.provider_errors import ProviderRepairRequiredError
from backend.shared.provider_pause import is_provider_credit_pause_error
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.proof_search.assistant_models import AssistantTargetSnapshot
from backend.shared.proof_search.search_service import proof_search_service
from backend.shared.runtime_settings import RuntimeSettingsError, save_proof_runtime_settings
from backend.shared.sleep_inhibitor import sleep_inhibitor
from backend.shared.smt_client import clear_smt_client, get_smt_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proofs", tags=["proofs"])

MANUAL_AGGREGATOR_SOURCE_ID = "manual_aggregator"
MANUAL_COMPILER_CURRENT_SOURCE_ID = "manual_compiler_current"
PROOF_SCOPE_AUTONOMOUS = "autonomous"
PROOF_SCOPE_MANUAL = "manual"
ProofLibraryCategory = Literal["novel", "duplicate_novel", "not_novel", "all"]
_manual_proof_run_lock = asyncio.Lock()
_LEAN_STATUS_STARTING_LOG_INTERVAL_SECONDS = 60.0
_last_lean_status_starting_log_at = 0.0
_ASSISTANT_MANUAL_SOURCE_SUMMARY_CHARS = 8000
ProofAppendCallback = Callable[[object], Awaitable[bool]]


@dataclass(frozen=True)
class ProofSourceAdapter:
    """Resolved proof source and every storage behavior bound to its identity."""

    source_type: str
    source_id: str
    source_title: str
    source_content: str
    user_prompt: str
    canonical_user_prompt: str
    proof_database: ProofDatabase
    proof_store_id: str
    scope: str
    writable: bool
    append_to_source: bool
    append_proof_callback: Optional[ProofAppendCallback]
    source_path: Optional[Path] = None

    async def fingerprint(self) -> str:
        """Fingerprint the current physical boundary for future loop re-resolution."""
        parts = [
            self.source_type,
            self.source_id,
            self.proof_store_id,
            str(self.writable),
            hashlib.sha256(self.source_content.encode("utf-8")).hexdigest(),
        ]
        if self.source_path is not None:
            try:
                stat = await asyncio.to_thread(self.source_path.stat)
            except FileNotFoundError:
                parts.append("source:missing")
            else:
                parts.append(f"source:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


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


async def _invalidate_pruned_proof_context(proof_id: str) -> None:
    """Invalidate Assistant state and rebuild the derived human/search index."""
    assistant_proof_search_coordinator.invalidate_live_context_occurrence(proof_id)
    try:
        await proof_search_service.rebuild_index()
    except Exception as exc:
        logger.warning(
            "Proof-search refresh failed after automatic prune %s: %s",
            str(proof_id)[:120],
            str(exc)[:240],
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
            response = _compact(data.get("lean_response"))
            if "timed out after" in response.lower() and "Advanced Settings" not in response:
                response = f"{response} You can change this timeout in Advanced Settings."
            return response
        if data.get("proof_verified") is True:
            return "Lean 4 response: proof verified."
        error = _compact(
            data.get("error_summary") or data.get("error_output") or data.get("reason"),
            limit=1800,
        )
        if error and "timed out after" in error.lower() and "Advanced Settings" not in error:
            error = f"{error} You can change this timeout in Advanced Settings."
        return f"Lean 4 response: {error} - proof not verified." if error else ""

    def _attempt_message(prefix: str) -> str:
        attempt = f", attempt {data.get('attempt')}" if data.get("attempt") else ""
        response = _lean_response()
        base = f"{prefix}: {target}{attempt}"
        return f"{base} - {response}" if response else base

    if event_type == "proof_check_started":
        return "Proof check started for the manual Aggregator database"
    if event_type == "proof_check_no_candidates":
        return (
            "Proof discovery: the model searched for useful novel proof candidates and "
            "found none, so no Lean proof attempts were needed."
        )
    if event_type == "proof_check_candidates_found":
        count = int(data.get("count") or 0)
        subject = "proof candidate" if count == 1 else "proof candidates"
        return f"Proof discovery found {count} {subject}; {count} will be attempted"
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
    round_index = int(data.get("round_index") or data.get("proof_round_index") or 0)
    round_label = f"Round {round_index}" if round_index > 0 else "Proof run"
    if event_type == "proof_run_queued":
        mode = " continuous" if data.get("run_mode") == "loop_with_pruning" else ""
        return f"{round_label} queued in{mode} mode"
    if event_type == "proof_run_round_started":
        return (
            f"{round_label} started. Proof discovery will identify prompt-relevant candidates, "
            "Lean 4 will verify each attempted proof, and accepted proofs may trigger a "
            "non-blocking pruning review."
        )
    if event_type == "proof_run_round_complete":
        candidate_count = data.get("candidate_count")
        if candidate_count == 0:
            detail = (
                "Discovery: the model searched for useful novel proof candidates and "
                "found none, so no Lean proof attempts were needed."
            )
        elif isinstance(candidate_count, int):
            subject = "candidate" if candidate_count == 1 else "candidates"
            detail = f"Discovery found {candidate_count} {subject} for this round."
        else:
            detail = "The round finished its proof discovery and verification work."
        continuation = (
            " The next round will start automatically; the loop continues until you press Stop."
            if data.get("run_mode") == "loop_with_pruning"
            and data.get("next_round_automatic") is not False
            else ""
        )
        return f"{round_label} complete. {detail}{continuation}"
    if event_type == "proof_run_provider_paused":
        return f"{round_label} paused for provider credits"
    if event_type == "proof_run_provider_resumed":
        return f"{round_label} resumed after the provider pause"
    if event_type == "proof_run_terminal":
        reason = data.get("terminal_reason") or "completed"
        return f"Proof run ended: {reason}"
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


def _normalize_proof_response_provenance(proof) -> dict:
    """Normalize legacy provenance at response time without mutating stored records."""
    if hasattr(proof, "model_dump"):
        payload = proof.model_dump(mode="json")
    else:
        payload = dict(proof or {})
    run_id = str(payload.get("run_id") or payload.get("session_id") or "").strip()
    if not run_id:
        source_type = str(payload.get("source_type") or "proof").strip()
        source_id = str(payload.get("source_id") or payload.get("proof_id") or "legacy").strip()
        run_id = f"legacy:{source_type}:{source_id}"
    payload["run_id"] = run_id
    payload["user_prompt"] = str(
        payload.get("user_prompt")
        or payload.get("source_title")
        or payload.get("theorem_statement")
        or ""
    )
    tier = str(payload.get("novelty_tier") or "").strip().lower()
    if not tier:
        tier = "novel_formulation" if bool(payload.get("novel")) else "not_novel"
    payload["novelty_tier"] = tier
    payload["independent_novelty_tier"] = str(
        payload.get("independent_novelty_tier") or tier
    )
    payload["independent_novelty_reasoning"] = str(
        payload.get("independent_novelty_reasoning")
        or payload.get("novelty_reasoning")
        or ""
    )
    payload["live_context_status"] = str(
        payload.get("live_context_status") or "active"
    )
    payload["live_context_owner_run_id"] = str(
        payload.get("live_context_owner_run_id") or ""
    )
    payload["live_context_prune_reason"] = str(
        payload.get("live_context_prune_reason") or ""
    )
    payload["live_context_prune_validator_reasoning"] = str(
        payload.get("live_context_prune_validator_reasoning") or ""
    )
    payload["live_context_prune_trigger_reasons"] = list(
        payload.get("live_context_prune_trigger_reasons") or []
    )
    return payload


def _get_request_proof_database(request: ProofCheckRequest) -> ProofDatabase:
    if (
        (request.source_type == "brainstorm" and request.source_id == MANUAL_AGGREGATOR_SOURCE_ID)
        or (request.source_type == "paper" and request.source_id == MANUAL_COMPILER_CURRENT_SOURCE_ID)
    ):
        return manual_proof_database
    return proof_database


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


async def _get_archived_export(
    session_id: str,
    proof_id: str,
    scope: str,
) -> tuple[dict, str]:
    normalized_scope = (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower()
    if normalized_scope == PROOF_SCOPE_MANUAL:
        payload = await manual_proof_database.get_library_proof_from_history(
            _manual_proof_history_root(), session_id, proof_id
        )
    elif normalized_scope == PROOF_SCOPE_AUTONOMOUS:
        payload = await proof_database.get_library_proof(session_id, proof_id)
    else:
        raise HTTPException(status_code=400, detail="Proof scope must be 'autonomous' or 'manual'.")
    if payload is None:
        raise HTTPException(status_code=404, detail="Proof not found")
    normalized = _normalize_proof_response_provenance(payload)
    return normalized, str(normalized.get("lean_code") or "")


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


def _configure_manual_roles(
    source_type: str,
    snapshot: ProofRuntimeConfigSnapshot,
    *,
    proof_run_id: str = "",
) -> ProofRoleConfigSnapshot:
    from backend.autonomous.agents.proof_pruning_agent import proof_run_role_suffix

    role_config = snapshot.brainstorm if source_type == "brainstorm" else snapshot.paper
    if not role_config.model_id or not snapshot.validator.model_id:
        raise RuntimeError("Manual proof roles are missing a configured submitter or validator model.")
    suffix = f"manual_{source_type}"
    if proof_run_id:
        suffix = f"{suffix}_{proof_run_role_suffix('manual', proof_run_id)}"
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
        f"autonomous_proof_novelty_{suffix}",
        _build_model_config(snapshot.validator),
    )
    proposer_role = snapshot.paper
    api_client_manager.configure_role(
        f"autonomous_proof_prune_proposer_{suffix}",
        _build_model_config(proposer_role),
    )
    api_client_manager.configure_role(
        f"autonomous_proof_prune_validator_{suffix}",
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
    scoped_proof_database: ProofDatabase = manual_proof_database,
) -> None:
    """Schedule Try-to-Prove Assistant memory before proof prompt preflight.

    Manual proof discovery may fail during mandatory-source context validation
    before it reaches ``api_client_manager.generate_completion()``, so the
    normal central Assistant injection hook never fires. This preflight refresh
    keeps the user-triggered proof-check button covered by Assistant memory
    without delaying mandatory proof work.
    """
    if not system_config.agent_conversation_memory_enabled:
        logger.info(
            "Assistant memory preflight skipped for manual proof check %s:%s because Agent Conversation Memory is disabled",
            source_type,
            source_id,
        )
        return

    run_id = await scoped_proof_database.get_or_create_active_run_id()
    snapshot = AssistantTargetSnapshot(
        workflow_mode="manual_proof_check",
        target_kind="proof_candidate",
        workflow_phase="manual_try_to_prove",
        active_mode="manual_proof_check",
        user_prompt=user_prompt,
        current_prompt_or_topic=source_title,
        current_submission_or_draft=_compact_manual_assistant_source(source_content),
        writing_goal="User-triggered Search For More Mathematical Proofs discovery over the selected source.",
        paper_or_proof_draft_summary=_compact_manual_assistant_source(source_content),
        target_statement=user_prompt or source_title or f"{source_type}:{source_id}",
        formal_sketch=_compact_manual_assistant_source(source_content),
        source_title=source_title,
        source_type=f"manual_{source_type}",
        source_id=source_id,
        run_id=run_id,
        source_titles=[source_title] if source_title else [],
        imports=["Mathlib"],
    )
    logger.info(
        "Assistant memory preflight scheduling for manual proof check %s:%s (%s)",
        source_type,
        source_id,
        source_title or "untitled source",
    )
    target_hash = assistant_proof_search_coordinator.submit_target(snapshot)
    logger.info(
        "Assistant memory preflight scheduled for manual proof check %s:%s (target=%s)",
        source_type,
        source_id,
        target_hash[:12],
    )


async def _prompt_with_verified_proof_context(
    prompt: str,
    scoped_proof_database: ProofDatabase = proof_database,
    requesting_run_id: str = "",
) -> str:
    """Apply proof-library context to a source-specific manual proof prompt."""
    source_prompt = (prompt or "").strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_user_prompt()).strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_base_user_prompt()).strip()
    if not requesting_run_id:
        run_id_getter = getattr(
            scoped_proof_database,
            "get_or_create_active_run_id",
            None,
        )
        if run_id_getter is not None:
            requesting_run_id = await run_id_getter()
    try:
        return scoped_proof_database.inject_into_prompt(
            source_prompt,
            requesting_run_id=requesting_run_id,
        )
    except TypeError as exc:
        if "requesting_run_id" not in str(exc):
            raise
        return scoped_proof_database.inject_into_prompt(source_prompt)


def _history_session_dir(session_id: str) -> Optional[Path]:
    """Resolve an existing autonomous history session without creating it."""
    if not session_id:
        return None
    if session_id == "legacy":
        return Path(system_config.data_dir)
    try:
        session_path = resolve_path_within_root(
            Path(system_config.auto_sessions_base_dir),
            session_id,
        )
    except Exception:
        return None
    if not session_path.is_dir():
        return None
    return session_path


def _history_proof_database_for_session(
    session_id: str,
    *,
    require_existing_store: bool = True,
) -> Optional[ProofDatabase]:
    """Return a proof database fixed to one historical session."""
    session_dir = _history_session_dir(session_id)
    if session_dir is None:
        return None
    proofs_dir = session_dir / "proofs"
    if require_existing_store and not proofs_dir.exists():
        return None
    history_db = ProofDatabase()
    history_db.set_base_dir(proofs_dir)
    return history_db


async def _prompt_with_history_proof_context(
    prompt: str,
    session_id: str,
    requesting_run_id: str = "",
    scoped_proof_database: Optional[ProofDatabase] = None,
) -> str:
    """Apply the selected history session's proof context when available."""
    source_prompt = (prompt or "").strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_user_prompt()).strip()
    if not source_prompt:
        source_prompt = (await research_metadata.get_base_user_prompt()).strip()

    history_db = scoped_proof_database or _history_proof_database_for_session(session_id)
    if history_db is None:
        try:
            return proof_database.inject_into_prompt(
                source_prompt,
                requesting_run_id=requesting_run_id,
            )
        except TypeError as exc:
            if "requesting_run_id" not in str(exc):
                raise
            return proof_database.inject_into_prompt(source_prompt)
    return history_db.inject_into_prompt(
        source_prompt,
        requesting_run_id=requesting_run_id,
    )


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


def _proof_store_id(database: ProofDatabase, fallback: str) -> str:
    """Return a stable physical store identity without exposing it publicly."""
    try:
        physical_path = str(database._base_dir.resolve(strict=False))
        digest = hashlib.sha256(physical_path.encode("utf-8")).hexdigest()[:20]
        return f"{fallback}:{digest}"
    except Exception:
        return fallback


def _validate_proof_run_source_id(source_type: str, source_id: str) -> str:
    """Validate source IDs accepted by proof-run lookup.

    Historical autonomous papers use the composite `{session_id}:{paper_id}`
    key. Each side is still a path component, but the separator itself is part
    of the public source identity.
    """
    raw_source_id = str(source_id or "").strip()
    if source_type == "paper" and ":" in raw_source_id:
        session_id, paper_id = raw_source_id.split(":", 1)
        return (
            f"{validate_single_path_component(session_id, 'history session ID')}:"
            f"{validate_single_path_component(paper_id, 'history paper ID')}"
        )
    return validate_single_path_component(raw_source_id, "proof source ID")


async def _resolve_proof_source_adapter(request: ProofCheckRequest) -> ProofSourceAdapter:
    """Resolve source content and bind it to the only store/append target it may use."""
    if request.source_type == "paper" and ":" in request.source_id:
        session_id, paper_id = request.source_id.split(":", 1)
        history_paper = await paper_library.get_history_paper(session_id, paper_id)
        if not history_paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        session_dir = _history_session_dir(session_id)
        papers_dir = paper_library.get_history_papers_dir(session_id)
        if session_dir is None or papers_dir is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        scoped_library = PaperLibrary._build_scoped_library(papers_dir)
        source_path = Path(scoped_library.get_paper_path(paper_id))
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="Paper content not found")

        scoped_database = _history_proof_database_for_session(
            session_id,
            require_existing_store=False,
        )
        if scoped_database is None:
            raise HTTPException(status_code=404, detail="Research session not found")
        await scoped_database.initialize()

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
        canonical_prompt = str(
            history_paper.get("user_prompt", "")
            or history_paper.get("title", "")
            or paper_id
        )
        run_id = await scoped_database.get_or_create_active_run_id()
        user_prompt = await _prompt_with_history_proof_context(
            canonical_prompt,
            session_id,
            requesting_run_id=run_id,
            scoped_proof_database=scoped_database,
        )

        async def _append_history_paper_proof(proof_record) -> bool:
            # Re-check the original source boundary immediately before mutation.
            if not source_path.is_file():
                return False
            return await scoped_library.append_proofs_section(paper_id, proof_record)

        return ProofSourceAdapter(
            source_type=request.source_type,
            source_id=request.source_id,
            source_title=str(history_paper.get("title", "") or paper_id),
            source_content=content,
            user_prompt=user_prompt,
            canonical_user_prompt=canonical_prompt,
            proof_database=scoped_database,
            proof_store_id=_proof_store_id(
                scoped_database,
                f"autonomous:session:{session_id}",
            ),
            scope=PROOF_SCOPE_AUTONOMOUS,
            writable=True,
            append_to_source=False,
            append_proof_callback=_append_history_paper_proof,
            source_path=source_path,
        )

    scoped_database = _get_request_proof_database(request)
    content, title, user_prompt = await _resolve_manual_source(
        request,
        scoped_database,
    )
    if request.source_id == MANUAL_AGGREGATOR_SOURCE_ID:
        canonical_prompt = await _manual_aggregator_prompt()
    elif request.source_id == MANUAL_COMPILER_CURRENT_SOURCE_ID:
        canonical_prompt = compiler_coordinator.user_prompt or await load_manual_compiler_prompt()
    else:
        canonical_prompt = await research_metadata.get_user_prompt()
    is_manual = scoped_database is manual_proof_database
    return ProofSourceAdapter(
        source_type=request.source_type,
        source_id=request.source_id,
        source_title=title,
        source_content=content,
        user_prompt=user_prompt,
        canonical_user_prompt=canonical_prompt,
        proof_database=scoped_database,
        proof_store_id=(
            "manual:active" if is_manual else "autonomous:active"
        ),
        scope=PROOF_SCOPE_MANUAL if is_manual else PROOF_SCOPE_AUTONOMOUS,
        writable=True,
        append_to_source=not _is_non_appending_manual_source(request),
        append_proof_callback=_manual_append_callback(request),
    )


async def _reuse_run_reservation(run_control: ProofRunControl) -> str:
    return run_control.reservation_token


async def _keep_run_reservation() -> None:
    """The manager, not the round driver, owns manual-run cleanup."""


async def _run_manual_proof_check(
    request: ProofCheckRequest,
    run_control: ProofRunControl | object,
    runtime_snapshot: Optional[ProofRuntimeConfigSnapshot] = None,
) -> None:
    source_title = ""
    legacy_sleep_owner = None
    manager_owned_run = isinstance(run_control, ProofRunControl)
    if not isinstance(run_control, ProofRunControl):
        legacy_sleep_owner = run_control
        sleep_inhibitor.acquire(legacy_sleep_owner)
        run_control = ProofRunControl(
            snapshot=ProofRunSnapshot(
                proof_run_id="legacy-manual-proof-check",
                run_mode="one_round",
                scope="manual",
                source_type=request.source_type,
                source_id=request.source_id,
                proof_store_id="legacy-manual",
                run_id="legacy-manual",
                lifecycle_generation=1,
                status="running",
            ),
            sleep_owner=legacy_sleep_owner,
        )
    try:
        source = await _resolve_proof_source_adapter(request)
        source_title = source.source_title
        snapshot = runtime_snapshot or await _get_runtime_snapshot(request)
        if snapshot is None:
            if _is_non_appending_manual_source(request):
                raise RuntimeError(
                    "No manual proof runtime model configuration is available for this source. "
                    "Start the manual Aggregator or Single Paper Writer with configured proof roles, "
                    "or retry from a browser session with complete manual role settings."
                )
            raise RuntimeError("No proof runtime model configuration is available yet.")

        async with _manual_proof_run_lock:
            role_config = _configure_manual_roles(
                request.source_type,
                snapshot,
                proof_run_id=run_control.snapshot.proof_run_id,
            )
        stage = autonomous_coordinator._proof_verification_stage
        broadcast_fn = (
            _broadcast_manual_aggregator_proof_event
            if _is_manual_aggregator_request(request)
            else websocket.broadcast_event
        )
        scoped_proof_database = source.proof_database
        active_run_id = await scoped_proof_database.get_or_create_active_run_id()
        pruning_coordinator = ProofPruningCoordinator(
            proof_database=scoped_proof_database,
            runtime_snapshot=snapshot,
            proof_run_id=run_control.snapshot.proof_run_id,
            run_mode=run_control.snapshot.run_mode,
            run_id=active_run_id,
            lifecycle_generation=run_control.snapshot.lifecycle_generation,
            scope="manual",
            source_type=request.source_type,
            source_id=request.source_id,
            canonical_user_prompt=source.canonical_user_prompt,
            proof_store_id=source.proof_store_id,
            broadcast_fn=broadcast_fn,
            persist_fn=lambda state: proof_run_manager.save_pruning_state(run_control, state),
            load_fn=lambda: proof_run_manager.load_pruning_state(run_control),
            invalidate_fn=_invalidate_pruned_proof_context,
            should_stop=run_control.stop_event.is_set,
        )
        run_control.pruning_coordinator = pruning_coordinator
        try:
            await pruning_coordinator.restore()
        except Exception:
            logger.exception("Noncritical pruning restore failure for %s", run_control.snapshot.proof_run_id)

        expected_store_id = source.proof_store_id
        persisted_fingerprint = run_control.snapshot.source_content_fingerprint
        current_fingerprint = await source.fingerprint()
        if persisted_fingerprint and persisted_fingerprint != current_fingerprint:
            await proof_run_manager.clear_candidate_checkpoint(run_control)
            await proof_run_manager.update(
                run_control,
                source_revision=run_control.snapshot.source_revision + 1,
                candidate_checkpoint_reference="",
                last_round_summary="",
                last_round_reference="",
            )
        assistant_boundary_keys: set[tuple[int, str]] = set()
        retry_activity_payload: Optional[Dict[str, Any]] = None
        await proof_run_manager.update(
            run_control,
            source_title=source.source_title,
            source_content_fingerprint=current_fingerprint,
            proof_set_revision=await scoped_proof_database.get_proof_set_revision(),
        )

        async def execute_round(
            round_index: int,
            round_trigger: str,
            prior_round_results: str,
            reservation_token: str,
        ):
            nonlocal source_title, current_fingerprint, retry_activity_payload
            if legacy_sleep_owner is not None and round_index == 1:
                refreshed = source
            else:
                try:
                    refreshed = await _resolve_proof_source_adapter(request)
                except HTTPException as exc:
                    if exc.status_code == 404:
                        raise ProofRunSourceInvalidError(str(exc.detail)) from exc
                    raise
            if refreshed.proof_store_id != expected_store_id:
                raise ProofRunSourceInvalidError("The proof source storage boundary changed.")
            refreshed_fingerprint = await refreshed.fingerprint()
            source_changed = refreshed_fingerprint != current_fingerprint
            if source_changed:
                current_fingerprint = refreshed_fingerprint
                await proof_run_manager.clear_candidate_checkpoint(run_control)
                await proof_run_manager.update(
                    run_control,
                    source_content_fingerprint=current_fingerprint,
                    source_revision=run_control.snapshot.source_revision + 1,
                    candidate_checkpoint_reference="",
                    last_round_summary="",
                    last_round_reference="",
                )
                prior_round_results = ""
            source_title = refreshed.source_title
            await proof_run_manager.begin_round(
                run_control,
                round_number=round_index,
            )
            await proof_run_manager.update(run_control, source_title=source_title)
            assistant_boundary_key = (round_index, refreshed_fingerprint)
            if assistant_boundary_key not in assistant_boundary_keys:
                assistant_boundary_keys.add(assistant_boundary_key)
                await _refresh_manual_assistant_memory(
                    source_type=request.source_type,
                    source_id=request.source_id,
                    source_title=source_title,
                    source_content=refreshed.source_content,
                    user_prompt=refreshed.user_prompt,
                    scoped_proof_database=scoped_proof_database,
                )
            checkpoint_status = ""
            checkpoint_candidates: List[ProofCandidate] = []
            proof_candidate_indexes: Dict[str, int] = {}
            checkpoint_attempts: Dict[str, List[ProofAttemptFeedback]] = {}
            checkpoint_theorem_names: Dict[str, str] = {}
            checkpoint_truncation_streak: List[Dict[str, Any]] = []
            checkpoint_result: Optional[ProofStageResult] = None
            saved_checkpoint = await proof_run_manager.load_candidate_checkpoint(
                run_control
            )
            if saved_checkpoint:
                (
                    checkpoint_candidates,
                    proof_candidate_indexes,
                    checkpoint_attempts,
                    checkpoint_theorem_names,
                    checkpoint_truncation_streak,
                    checkpoint_result,
                ) = autonomous_coordinator._deserialize_proof_checkpoint(
                    saved_checkpoint
                )
                if checkpoint_candidates:
                    logger.info(
                        "Resuming manual proof run %s round %s from checkpoint with %s remaining candidate(s)",
                        run_control.snapshot.proof_run_id,
                        round_index,
                        len(checkpoint_candidates),
                    )

            async def save_manual_checkpoint(payload: dict) -> None:
                nonlocal checkpoint_status
                checkpoint_status = str(payload.get("status") or "")
                reference = (
                    f"{run_control.snapshot.proof_run_id}:"
                    f"{round_index}:{int(payload.get('checkpoint_revision') or 0)}"
                )
                await proof_run_manager.save_candidate_checkpoint(
                    run_control,
                    payload,
                )
                await proof_run_manager.update(
                    run_control,
                    candidate_checkpoint_reference=reference,
                )

            try:
                result = await stage.run_manual(
                    content=refreshed.source_content,
                    source_type=request.source_type,
                    source_id=request.source_id,
                    user_prompt=refreshed.user_prompt,
                    canonical_user_prompt=refreshed.canonical_user_prompt,
                    run_id=active_run_id,
                    submitter_model=role_config.model_id,
                    submitter_context=role_config.context_window,
                    submitter_max_tokens=role_config.max_output_tokens,
                    validator_model=snapshot.validator.model_id,
                    validator_context=snapshot.validator.context_window,
                    validator_max_tokens=snapshot.validator.max_output_tokens,
                    broadcast_fn=broadcast_fn,
                    novel_proofs_db=scoped_proof_database,
                    source_title=source_title,
                    theorem_candidates=checkpoint_candidates or None,
                    source_reserved=True,
                    source_reservation_token=reservation_token or run_control.reservation_token,
                    append_to_source=refreshed.append_to_source,
                    append_proof_callback=refreshed.append_proof_callback,
                    should_stop=run_control.stop_event.is_set,
                    release_source_on_exit=False,
                    proof_run_context={
                        "proof_run_id": run_control.snapshot.proof_run_id,
                        "run_mode": run_control.snapshot.run_mode,
                        "lifecycle_generation": run_control.snapshot.lifecycle_generation,
                        "proof_run_unbounded": run_control.snapshot.run_mode == "loop_with_pruning",
                        "round_index": round_index,
                        "round_trigger": round_trigger,
                        "prior_round_results": prior_round_results,
                    },
                    proof_pruning_registered_callback=pruning_coordinator.notify_proof_registered,
                    proof_pruning_pressure_callback=pruning_coordinator.notify_context_pressure,
                    proof_pruning_route_fingerprint=pruning_coordinator.route_config_fingerprint(snapshot),
                    proof_round_index=round_index,
                    proof_max_rounds=(
                        0
                        if run_control.snapshot.run_mode == "loop_with_pruning"
                        else 1
                    ),
                    prior_round_results=prior_round_results,
                    proof_candidate_indexes=proof_candidate_indexes,
                    checkpoint_attempts_by_candidate=checkpoint_attempts,
                    checkpoint_theorem_names_by_candidate=checkpoint_theorem_names,
                    checkpoint_truncation_streak=checkpoint_truncation_streak,
                    checkpoint_result=checkpoint_result,
                    checkpoint_callback=save_manual_checkpoint,
                )
                if retry_activity_payload is not None:
                    await broadcast_fn(
                        "proof_run_provider_resumed",
                        {
                            **retry_activity_payload,
                            "proof_run_id": run_control.snapshot.proof_run_id,
                            "run_mode": run_control.snapshot.run_mode,
                            "lifecycle_generation": run_control.snapshot.lifecycle_generation,
                            "scope": run_control.snapshot.scope,
                            "source_type": run_control.snapshot.source_type,
                            "source_id": run_control.snapshot.source_id,
                            "round_index": round_index,
                            "proof_round_index": round_index,
                            "message": (
                                f"{retry_activity_payload.get('provider_label') or 'Provider'} "
                                f"retry {retry_activity_payload.get('retry_attempt') or 1} "
                                "succeeded; the preserved proof round resumed."
                            ),
                        },
                    )
                    retry_activity_payload = None
                if result is None:
                    return "completed", None
            except Exception as exc:
                if is_provider_credit_pause_error(exc):
                    await proof_run_manager.provider_paused(run_control)
                    await proof_run_manager.wait_for_provider_or_control_wake(run_control)
                    if run_control.stop_event.is_set():
                        return "stopped", None
                    await proof_run_manager.resumed(
                        run_control,
                        from_provider_pause=True,
                    )
                    return "retry_same_round", None
                if isinstance(exc, RetryableProviderError):
                    async def broadcast_retry_activity(
                        state: str,
                        payload: Dict[str, Any],
                    ) -> None:
                        nonlocal retry_activity_payload
                        if state == "waiting":
                            retry_activity_payload = dict(payload)
                        event_payload = {
                            **payload,
                            "proof_run_id": run_control.snapshot.proof_run_id,
                            "run_mode": run_control.snapshot.run_mode,
                            "lifecycle_generation": run_control.snapshot.lifecycle_generation,
                            "scope": run_control.snapshot.scope,
                            "source_type": run_control.snapshot.source_type,
                            "source_id": run_control.snapshot.source_id,
                            "round_index": round_index,
                            "proof_round_index": round_index,
                            "proof_run_unbounded": (
                                run_control.snapshot.run_mode == "loop_with_pruning"
                            ),
                            "message": (
                                f"{payload.get('provider_label') or 'Provider'} could not complete "
                                f"the current proof call. Retry {payload.get('retry_attempt') or 1} "
                                f"will resume this same round after "
                                f"{payload.get('retry_after_seconds') or 0} seconds."
                                if state == "waiting"
                                else (
                                    f"Retry {payload.get('retry_attempt') or 1} is starting now "
                                    "from the preserved proof-round checkpoint."
                                )
                            ),
                        }
                        await broadcast_fn(
                            "proof_retry_scheduled"
                            if state == "waiting"
                            else "proof_retry_started",
                            event_payload,
                        )

                    await api_client_manager.wait_for_retryable_provider_error(
                        exc,
                        role_id=exc.role_id or role_config.model_id,
                        should_stop=run_control.stop_event.is_set,
                        activity_callback=broadcast_retry_activity,
                    )
                    if run_control.stop_event.is_set():
                        return "stopped", None
                    return "retry_same_round", None
                if is_transient_model_call_error(exc):
                    await asyncio.sleep(5)
                    if run_control.stop_event.is_set():
                        return "stopped", None
                    return "retry_same_round", None
                if isinstance(exc, ProviderRepairRequiredError) or is_non_retryable_model_error(exc):
                    await proof_run_manager.repair_required(
                        run_control,
                        reason=ProofVerificationStage._summarize_error(str(exc), limit=1000),
                    )
                    return "stopped", None
                raise

            summary = summarize_round_result(round_index, result)
            if result.fatal_stop_reason:
                await proof_run_manager.update(
                    run_control,
                    terminal_reason=result.fatal_stop_reason,
                )
            await proof_run_manager.complete_round(
                run_control,
                round_number=round_index,
                valid_candidate_count=(
                    result.total_candidates
                    if not result.had_error and not result.fatal_stop_reason
                    else None
                ),
                summary=summary,
                candidate_checkpoint_reference=(
                    ""
                    if checkpoint_status in {"complete", "no_candidates"}
                    else run_control.snapshot.candidate_checkpoint_reference
                ),
                proof_set_revision=await scoped_proof_database.get_proof_set_revision(),
            )
            if result.fatal_stop_reason:
                return "fatal_stop", result
            if (
                run_control.snapshot.run_mode == "loop_with_pruning"
                and result.total_candidates == 0
                and not result.had_error
                and not result.fatal_stop_reason
            ):
                if run_control.stop_event.is_set():
                    return "stopped", result
                await asyncio.sleep(0)
                if run_control.stop_event.is_set():
                    return "stopped", result
                return "continue_reset", None
            return ("completed_reset" if source_changed else "completed"), result

        driver = ProofRoundDriver(
            policy=(
                ContinuousPruningPolicy()
                if run_control.snapshot.run_mode == "loop_with_pruning"
                else OneRoundPolicy()
            ),
            source_type=request.source_type,
            source_id=request.source_id,
            base_trigger="manual",
            execute_round=execute_round,
            should_stop=run_control.stop_event.is_set,
            reserve_source=lambda _source_type, _source_id: _reuse_run_reservation(run_control),
            release_source=lambda _source_type, _source_id, _token: _keep_run_reservation(),
            initial_round_index=max(1, run_control.snapshot.last_completed_round + 1),
        )
        driver_status = await driver.run()
        if driver_status == "error_preserved":
            await proof_run_manager.error(
                run_control,
                terminal_reason="proof_stage_error",
                reason="Proof verification preserved an error checkpoint. Review proof activity for details, then repair settings or retry.",
            )
            return
        if driver_status == "fatal_stop":
            await proof_run_manager.error(
                run_control,
                terminal_reason="proof_output_truncation_recovery_exhausted",
                reason="Proof output truncation recovery was exhausted for this manual proof run.",
            )
            return
        if driver_status == "stopped":
            await proof_run_manager.update(
                run_control,
                status="stopping",
                stop_requested=True,
            )
            return
    except Exception as exc:
        logger.exception("Manual proof check failed for %s %s", request.source_type, request.source_id)
        broadcast_fn = (
            _broadcast_manual_aggregator_proof_event
            if _is_manual_aggregator_request(request)
            else websocket.broadcast_event
        )
        if api_client_manager.is_provider_failure(exc):
            logger.warning(
                "Manual proof check left pending after provider failure for %s %s; "
                "provider notification owns user repair guidance",
                request.source_type,
                request.source_id,
            )
        else:
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
                        f"{ProofVerificationStage._summarize_error(str(exc), limit=1800)}"
                    ),
                },
            )
        if manager_owned_run:
            raise
    finally:
        try:
            await assistant_proof_search_coordinator.stop_all(
                broadcast=True,
                reason="manual_proof_check_complete",
            )
        except Exception:
            logger.exception(
                "Unable to stop Assistant memory after manual proof check %s %s",
                request.source_type,
                request.source_id,
            )
        if legacy_sleep_owner is not None:
            try:
                await ProofVerificationStage.release_source(
                    request.source_type,
                    request.source_id,
                )
            finally:
                sleep_inhibitor.release(legacy_sleep_owner)


@router.get("", response_model=CurrentProofListResponse)
async def list_proofs(
    response: Response,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return all verified proofs plus aggregate counts."""
    response.headers["Cache-Control"] = "no-store"
    scoped_proof_database = _get_scoped_proof_database(scope)
    proofs = await scoped_proof_database.get_all_proofs()
    return {
        "proofs": [_normalize_proof_response_provenance(proof) for proof in proofs],
        "counts": scoped_proof_database.count_proofs(),
        "scope": (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower(),
        "proof_set_revision": await scoped_proof_database.get_proof_set_revision(),
    }


@router.get("/novel")
async def list_novel_proofs(scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS)):
    """Return only novel verified proofs."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proofs = await scoped_proof_database.get_all_proofs(novel_only=True)
    return {
        "proofs": [_normalize_proof_response_provenance(proof) for proof in proofs],
        "counts": scoped_proof_database.count_proofs(),
        "scope": (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower(),
    }


@router.get("/known")
async def list_known_proofs(scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS)):
    """Return only known (non-novel) verified proofs."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proofs = await scoped_proof_database.get_all_proofs(novel_only=False)
    return {
        "proofs": [_normalize_proof_response_provenance(proof) for proof in proofs],
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
    """Return cached Lean 4 availability and proof-database status."""
    version = ""
    workspace_ready = False
    mathlib_commit = ""
    lsp_active = False
    z3_version = ""
    smt_available = False
    workspace_state = "disabled" if not system_config.lean4_enabled else "not_started"
    workspace_error = ""
    manual_check_ready, manual_check_message = await _get_manual_check_status()
    if system_config.lean4_enabled:
        try:
            client = get_lean4_client()
            workspace_status = client.get_workspace_status()
            workspace_state = str(workspace_status.get("state") or "not_started")
            workspace_ready = bool(workspace_status.get("ready"))
            workspace_error = str(workspace_status.get("error") or "")
            version = client.get_cached_version()
            if workspace_ready:
                mathlib_commit = client.get_mathlib_commit()
            lsp_active = client.is_server_active()
        except Exception as exc:
            workspace_state = "failed"
            workspace_error = str(exc)
        if manual_check_ready:
            version_text = (version or "").strip().lower()
            version_unavailable = (
                bool(version_text)
                and ("not found" in version_text
                or "no such file" in version_text
                or "not recognized" in version_text)
            )
            if version_unavailable:
                manual_check_ready = False
                manual_check_message = "Lean 4 executable is not available."
            elif workspace_state == "failed":
                manual_check_ready = False
                manual_check_message = workspace_error or "Lean 4 workspace preparation failed."
            elif not workspace_ready:
                manual_check_message = "Lean 4 will finish preparing before verification."

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
        "workspace_state": workspace_state,
        "workspace_error": workspace_error,
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
            initialize_lean4_client()

    if smt_settings_changed:
        clear_smt_client()

    try:
        save_proof_runtime_settings()
    except RuntimeSettingsError as exc:
        logger.error("Failed to persist proof runtime settings: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist proof runtime settings")

    return await get_proofs_status()


@router.post("/check", response_model=ProofRunQueueResponse)
async def run_manual_proof_check(request: ProofCheckRequest):
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
        source = await _resolve_proof_source_adapter(request)
        run_id = await source.proof_database.get_or_create_active_run_id()
        try:
            source_fingerprint = await source.fingerprint()
            response = await proof_run_manager.queue(
                scope=source.scope,
                source_type=request.source_type,
                source_id=request.source_id,
                proof_store_id=source.proof_store_id,
                run_id=run_id,
                worker=lambda control: _run_manual_proof_check(request, control, snapshot),
                run_mode=request.run_mode,
                source_title=source.source_title,
                source_content_fingerprint=source_fingerprint,
                proof_set_revision=await source.proof_database.get_proof_set_revision(),
                event_callback=(
                    _broadcast_manual_aggregator_proof_event
                    if _is_manual_aggregator_request(request)
                    else websocket.broadcast_event
                ),
            )
            get_lean4_client().start_workspace_bootstrap()
            return response
        except RuntimeError:
            raise HTTPException(
                status_code=409,
                detail="A proof verification is already running for that source.",
            )


@router.get("/runs", response_model=ProofRunCollectionResponse)
async def list_proof_runs(
    response: Response,
    scope: Optional[Literal["autonomous", "manual"]] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Return a bounded metadata-only proof-run collection for reconnect recovery."""
    response.headers["Cache-Control"] = "no-store"
    return await proof_run_manager.list_runs(limit=limit, scope=scope)


@router.get("/runs/by-source", response_model=ProofRunSourceLookupResponse)
async def lookup_proof_runs_by_source(
    response: Response,
    scope: Literal["autonomous", "manual"] = Query(...),
    source_type: Literal["brainstorm", "paper"] = Query(...),
    source_id: str = Query(..., min_length=1, max_length=512),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Resolve recent runs for one validated source when a queue response was lost."""
    response.headers["Cache-Control"] = "no-store"
    try:
        safe_source_id = _validate_proof_run_source_id(source_type, source_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await proof_run_manager.find_by_source(
        scope=scope,
        source_type=source_type,
        source_id=safe_source_id,
        limit=limit,
    )


@router.get("/runs/{proof_run_id}", response_model=ProofRunSnapshot)
async def get_proof_run(proof_run_id: str, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        safe_proof_run_id = validate_single_path_component(
            proof_run_id,
            "proof run ID",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    snapshot = await proof_run_manager.get(safe_proof_run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Proof run not found")
    return snapshot


@router.post("/runs/{proof_run_id}/stop", response_model=ProofRunSnapshot)
async def stop_proof_run(proof_run_id: str, request: ProofRunStopRequest):
    try:
        safe_proof_run_id = validate_single_path_component(
            proof_run_id,
            "proof run ID",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        snapshot = await proof_run_manager.stop(
            safe_proof_run_id,
            request.expected_lifecycle_generation,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Proof run not found")
    return snapshot


@router.patch(
    "/{proof_id}/live-context",
    response_model=ProofLiveContextMutationResponse,
)
async def update_proof_live_context(
    proof_id: str,
    request: ProofLiveContextMutationRequest,
    scope: Literal["autonomous", "manual"] = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    scoped_database = _get_scoped_proof_database(scope)
    try:
        proof = await scoped_database.get_proof(proof_id)
    except ValueError:
        proof = None
    if proof is None:
        raise HTTPException(status_code=404, detail="Proof not found")

    warnings = []
    dependents = await scoped_database.get_proofs_depending_on(proof_id)
    if dependents:
        warnings.append(f"{len(dependents)} stored proof(s) depend on this occurrence.")
    if proof.source_type in {"leanoj_final"}:
        warnings.append("This occurrence is a verified final-solution proof.")
    try:
        updated, revision = await scoped_database.set_live_context_status(
            proof_id=proof_id,
            status=request.status,
            expected_run_id=request.expected_run_id,
            expected_proof_set_revision=request.expected_proof_set_revision,
            actor=request.actor,
            reason=request.reason,
            expected_theorem_hash=request.expected_theorem_hash,
            expected_lean_hash=request.expected_lean_hash,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Proof not found")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    assistant_proof_search_coordinator.invalidate_live_context_occurrence(proof_id)

    async def _refresh_proof_search_after_live_context_update() -> None:
        try:
            await proof_search_service.rebuild_index()
        except Exception as exc:
            logger.warning(
                "Proof-search refresh failed after live-context update for %s: %s",
                str(proof_id)[:120],
                str(exc)[:240],
            )

    asyncio.create_task(_refresh_proof_search_after_live_context_update())
    normalized_scope = (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower()
    await websocket.broadcast_event(
        "proof_live_context_updated",
        {
            "scope": normalized_scope,
            "proof_id": proof_id,
            "run_id": request.expected_run_id,
            "live_context_status": updated.live_context_status,
            "proof_set_revision": revision,
        },
    )
    return ProofLiveContextMutationResponse(
        scope=normalized_scope,
        proof_id=proof_id,
        run_id=request.expected_run_id,
        live_context_status=updated.live_context_status,
        live_context_pruned_at=updated.live_context_pruned_at,
        proof_search_refresh_scheduled=True,
        proof_set_revision=revision,
        warnings=warnings,
    )


@router.get("/library", response_model=ProofLibraryResponse)
async def get_proof_library(
    novel_only: Optional[bool] = None,
    category: Optional[ProofLibraryCategory] = Query(default=None),
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return archived proofs for the selected proof-library scope."""
    normalized_scope = (scope or PROOF_SCOPE_AUTONOMOUS).strip().lower()
    normalized_category = normalize_proof_library_category(category, novel_only)
    if normalized_scope == PROOF_SCOPE_MANUAL:
        all_proofs = await manual_proof_database.list_proof_library_from_history(
            _manual_proof_history_root(),
            novel_only=None,
            category="all",
        )
        proofs = await manual_proof_database.list_proof_library_from_history(
            _manual_proof_history_root(),
            novel_only=novel_only,
            category=normalized_category,
        )
    elif normalized_scope == PROOF_SCOPE_AUTONOMOUS:
        all_proofs = await proof_database.list_proof_library(
            novel_only=None,
            category="all",
        )
        proofs = await proof_database.list_proof_library(
            novel_only=novel_only,
            category=normalized_category,
        )
    else:
        raise HTTPException(status_code=400, detail="Proof scope must be 'autonomous' or 'manual'.")
    novel_count = sum(
        1 for p in all_proofs
        if p.get("novel") and is_prompt_injection_novel_tier(p.get("novelty_tier", ""))
    )
    duplicate_novel_count = sum(
        1 for p in all_proofs if is_duplicate_novel_tier(p.get("novelty_tier", ""))
    )
    not_novel_count = sum(
        1 for p in all_proofs if is_not_novel_tier(p.get("novelty_tier", "not_novel"))
    )
    normalized_all = [_normalize_proof_response_provenance(p) for p in all_proofs]
    normalized_proofs = [_normalize_proof_response_provenance(p) for p in proofs]
    return {
        "proofs": normalized_proofs,
        "counts": {
            "total": len(normalized_all),
            "listed": len(normalized_proofs),
            "novel": novel_count,
            "duplicate_novel": duplicate_novel_count,
            "not_novel": not_novel_count,
            "live_context_active": sum(
                1
                for proof in normalized_all
                if proof.get("live_context_status", "active") != "pruned"
            ),
            "live_context_pruned": sum(
                1
                for proof in normalized_all
                if proof.get("live_context_status", "active") == "pruned"
            ),
        },
        "scope": normalized_scope,
        "category": normalized_category,
    }


@router.get("/library/{session_id}/{proof_id}", response_model=ProofLibraryEntry)
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
    return _normalize_proof_response_provenance(proof)


@router.get("/library/{session_id}/{proof_id}/certificate", response_model=ProofCertificateResponse)
async def get_library_proof_certificate(
    session_id: str,
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Export an archived proof certificate keyed by its validated run and proof IDs."""
    proof, lean_code = await _get_archived_export(session_id, proof_id, scope)
    return await _certificate_response(proof, lean_code)


@router.get("/library/{session_id}/{proof_id}/certificate.lean")
async def get_library_proof_certificate_lean(
    session_id: str,
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Export archived Lean source keyed by its validated run and proof IDs."""
    proof, lean_code = await _get_archived_export(session_id, proof_id, scope)
    return PlainTextResponse(
        content=lean_code,
        headers={"Content-Disposition": f'attachment; filename="{proof["proof_id"]}.lean"'},
    )


async def _certificate_response(proof, lean_code: str) -> JSONResponse:
    """Build the common typed certificate payload for live and archived proofs."""
    lean_version = ""
    mathlib_commit = ""
    if system_config.lean4_enabled:
        try:
            client = get_lean4_client()
            lean_version = await asyncio.wait_for(client.get_version(), timeout=5.0)
            mathlib_commit = client.get_mathlib_commit()
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Lean 4 certificate metadata lookup timed out or failed: %s", exc)
    normalized = _normalize_proof_response_provenance(proof)
    payload = ProofCertificateResponse(
        proof_id=normalized["proof_id"],
        theorem_statement=normalized["theorem_statement"],
        theorem_name=normalized.get("theorem_name", ""),
        lean_code=lean_code,
        solver=normalized.get("solver") or "Lean 4",
        lean_version=lean_version,
        mathlib_commit=mathlib_commit,
        verified_at=(
            normalized["created_at"].isoformat()
            if normalized.get("created_at") and hasattr(normalized["created_at"], "isoformat")
            else str(normalized.get("created_at") or "") or None
        ),
        source_type=normalized.get("source_type", ""),
        source_id=normalized.get("source_id", ""),
        source_title=normalized.get("source_title", ""),
        run_id=normalized["run_id"],
        user_prompt=normalized["user_prompt"],
        novel=bool(normalized.get("novel")),
        novelty_tier=normalized["novelty_tier"],
        novelty_reasoning=normalized.get("novelty_reasoning", ""),
        independent_novelty_tier=normalized["independent_novelty_tier"],
        independent_novelty_reasoning=normalized["independent_novelty_reasoning"],
        exact_duplicate_proof_id=normalized.get("exact_duplicate_proof_id", ""),
        exact_duplicate_run_id=normalized.get("exact_duplicate_run_id", ""),
        artifact_purpose=normalized.get("artifact_purpose") or "verified_occurrence",
        canonical_identity_version=normalized.get("canonical_identity_version", ""),
        canonical_theorem_statement_hash=normalized.get(
            "canonical_theorem_statement_hash",
            "",
        ),
        canonical_lean_code_hash=normalized.get("canonical_lean_code_hash", ""),
        attempt_count=normalized.get("attempt_count") or 0,
        solver_hints=list(normalized.get("solver_hints") or []),
        dependencies=list(normalized.get("dependencies") or []),
        live_context_status=normalized.get("live_context_status", "active"),
        live_context_owner_run_id=normalized.get("live_context_owner_run_id", ""),
        live_context_pruned_at=normalized.get("live_context_pruned_at"),
        live_context_pruned_by=normalized.get("live_context_pruned_by"),
        live_context_prune_reason=normalized.get("live_context_prune_reason", ""),
        live_context_prune_validator_reasoning=normalized.get(
            "live_context_prune_validator_reasoning", ""
        ),
        live_context_prune_snapshot_revision=normalized.get(
            "live_context_prune_snapshot_revision"
        ),
        live_context_prune_trigger_reasons=list(
            normalized.get("live_context_prune_trigger_reasons") or []
        ),
    )
    return JSONResponse(content=payload.model_dump(mode="json"))


@router.get("/{proof_id}/certificate", response_model=ProofCertificateResponse)
async def get_proof_certificate(
    proof_id: str,
    scope: str = Query(default=PROOF_SCOPE_AUTONOMOUS),
):
    """Return a machine-readable proof certificate JSON payload."""
    scoped_proof_database = _get_scoped_proof_database(scope)
    proof = await _get_export_proof_or_404(proof_id, scoped_proof_database)

    lean_code = await _get_export_lean_code(proof_id, scoped_proof_database)
    response = await _certificate_response(proof, lean_code)
    response.headers["Content-Disposition"] = f'attachment; filename="{proof_id}_certificate.json"'
    return response


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
                        "live_context_status": dependent.live_context_status,
                        "live_context_owner_run_id": dependent.live_context_owner_run_id,
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
                "live_context_status": dependent.live_context_status,
                "live_context_owner_run_id": dependent.live_context_owner_run_id,
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
                "live_context_status": dependent.live_context_status,
                "live_context_owner_run_id": dependent.live_context_owner_run_id,
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
    return _normalize_proof_response_provenance(proof)
