"""
Aggregator API routes.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
import logging
from pathlib import Path
import aiofiles

from backend.shared.models import AggregatorStartRequest, SystemStatus, ModelInfo
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.config import system_config, rag_config
from backend.shared.token_tracker import token_tracker
from backend.shared.path_safety import resolve_path_within_root, validate_single_path_component
from backend.shared.log_redaction import redact_log_text
from backend.shared.manual_proof_context import get_manual_proof_context_lock
from backend.shared.workflow_start_guard import workflow_start_guard
from backend.aggregator.core.coordinator import coordinator
from backend.aggregator.core.context_allocator import context_allocator
from backend.aggregator.memory.event_log import event_log
from backend.aggregator.memory.shared_training import (
    clear_manual_aggregator_prompt,
    load_manual_aggregator_prompt,
    save_manual_aggregator_prompt,
    shared_training_memory,
)
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.autonomous.memory.proof_database import manual_proof_database
from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/aggregator", tags=["aggregator"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

MANUAL_PROOF_ACTIVE_KEYS = {
    "brainstorm:manual_aggregator",
    "paper:manual_compiler_current",
}


async def _manual_proof_clear_blocker() -> Optional[str]:
    """Return a blocker message if manual proof work could write stale proofs."""
    active_keys = await ProofVerificationStage.active_source_keys()
    for key in active_keys:
        if (
            key in MANUAL_PROOF_ACTIVE_KEYS
            or key.startswith("paper:compiler_manual_")
            or key.startswith("paper:manual_compiler_")
        ):
            return "Cannot clear the manual run while manual proof verification is running. Stop or wait for proof verification to finish first."
    return None


def _require_positive_setting(value: int, label: str) -> int:
    """Reject missing context/max-output settings before workflow state mutates."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be configured as a positive integer in Settings.",
        )
    return parsed


def _require_valid_role_limits(context_window: int, max_output_tokens: int, label: str) -> None:
    context = _require_positive_setting(context_window, f"{label} context window")
    max_tokens = _require_positive_setting(max_output_tokens, f"{label} max output tokens")
    if max_tokens >= context:
        raise HTTPException(
            status_code=400,
            detail=f"{label} max output tokens must be smaller than its context window.",
        )


def _get_start_conflict() -> Optional[str]:
    """Return a user-facing conflict message if another workflow is active."""
    if coordinator.is_running:
        return "Aggregator is already running"

    if compiler_coordinator.is_running:
        return "Cannot start Aggregator while Compiler is running. Stop Compiler first."

    autonomous_state = autonomous_coordinator.get_state()
    if autonomous_state.is_running or autonomous_coordinator.is_active:
        return "Cannot start Aggregator while Autonomous Research is running. Stop Autonomous Research first."

    if leanoj_coordinator.is_active:
        return "Cannot start Aggregator while Proof Solver is running. Stop Proof Solver first."

    return None


async def _ensure_manual_aggregator_memory_loaded_for_read() -> None:
    """Load the persisted manual Aggregator database without starting a workflow."""
    manual_path = Path(system_config.shared_training_file)

    if coordinator.is_running:
        if shared_training_memory.file_path != manual_path:
            try:
                paths_match = shared_training_memory.file_path.resolve() == manual_path.resolve()
            except Exception:
                paths_match = False
            if not paths_match:
                return
        await shared_training_memory.refresh_proof_appendix_from_file()
        return

    if autonomous_coordinator.is_active:
        return

    if shared_training_memory.file_path != manual_path:
        shared_training_memory.file_path = manual_path

    if manual_path.exists() and manual_path.stat().st_size > 0:
        await shared_training_memory.reload_insights_from_current_path()
        return

    async with shared_training_memory._lock:
        shared_training_memory.insights.clear()
        shared_training_memory.proof_appendix = ""
        shared_training_memory.submission_count = 0
        shared_training_memory.last_ragged_submission_count = 0

    if not manual_path.exists():
        manual_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(manual_path, "w", encoding="utf-8") as handle:
            await handle.write("")


async def _ensure_manual_event_log_loaded_for_read() -> None:
    """Load persisted manual Aggregator events without starting the Aggregator."""
    if coordinator.is_running:
        return
    await event_log.initialize()


@router.post("/start")
async def start_aggregator(request: AggregatorStartRequest):
    """Start the aggregator system."""
    try:
        async with workflow_start_guard.reserve():
            conflict = _get_start_conflict()
            if conflict:
                raise HTTPException(status_code=400, detail=conflict)

            # Validate submitter configs
            num_submitters = len(request.submitter_configs)
            if not (system_config.min_submitters <= num_submitters <= system_config.max_submitters):
                raise HTTPException(
                    status_code=400,
                    detail=f"Number of submitters must be {system_config.min_submitters}-{system_config.max_submitters}, got {num_submitters}"
                )
            _require_valid_role_limits(
                request.validator_context_size,
                request.validator_max_output_tokens,
                "Validator",
            )
            for config in request.submitter_configs:
                label = "Main submitter" if config.submitter_id == 1 else f"Submitter {config.submitter_id}"
                _require_valid_role_limits(config.context_window, config.max_output_tokens, label)

            # Update validator context window configuration
            rag_config.validator_context_window = request.validator_context_size
            rag_config.validator_max_output_tokens = request.validator_max_output_tokens

            # Use first submitter's context for context_allocator (for compatibility)
            if request.submitter_configs:
                first_submitter = request.submitter_configs[0]
                rag_config.submitter_context_window = first_submitter.context_window
                rag_config.submitter_max_output_tokens = first_submitter.max_output_tokens
                context_allocator.set_context_windows(
                    first_submitter.context_window,
                    request.validator_context_size,
                    first_submitter.max_output_tokens,
                    request.validator_max_output_tokens
                )

            # Log submitter configurations
            for config in request.submitter_configs:
                label = "(Main Submitter)" if config.submitter_id == 1 else ""
                logger.info(
                    "Submitter %s %s: model=%s, context=%s, max_tokens=%s",
                    config.submitter_id,
                    label,
                    redact_log_text(config.model_id, 160),
                    redact_log_text(config.context_window, 40),
                    redact_log_text(config.max_output_tokens, 40),
                )
            logger.info(
                "Validator: model=%s, context=%s, max_tokens=%s",
                redact_log_text(request.validator_model, 160),
                redact_log_text(request.validator_context_size, 40),
                redact_log_text(request.validator_max_output_tokens, 40),
            )

            # Initialize coordinator with per-submitter configs (includes OpenRouter provider fields)
            await coordinator.initialize(
                user_prompt=request.user_prompt,
                submitter_configs=request.submitter_configs,
                validator_model=request.validator_model,
                user_files=request.uploaded_files,
                validator_context_window=request.validator_context_size,
                validator_max_tokens=request.validator_max_output_tokens,
                # Pass OpenRouter provider config for validator
                validator_provider=request.validator_provider,
                validator_openrouter_provider=request.validator_openrouter_provider,
                validator_openrouter_reasoning_effort=request.validator_openrouter_reasoning_effort,
                validator_lm_studio_fallback=request.validator_lm_studio_fallback,
                validator_supercharge_enabled=request.validator_supercharge_enabled,
                creativity_emphasis_boost_enabled=request.creativity_emphasis_boost_enabled,
            )
            await save_manual_aggregator_prompt(request.user_prompt)

            # Start coordinator
            token_tracker.reset()
            token_tracker.start_timer()
            await coordinator.start()

            return {
                "status": "started",
                "message": f"Aggregator system started with {num_submitters} submitters",
                "num_submitters": num_submitters
            }

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Aggregator configuration error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Other errors
        logger.error(f"Failed to start aggregator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stop")
async def stop_aggregator():
    """Stop the aggregator system."""
    try:
        await coordinator.stop()
        token_tracker.stop_timer()
        return {"status": "stopped", "message": "Aggregator system stopped"}
    except Exception as e:
        logger.error(f"Failed to stop aggregator: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status", response_model=SystemStatus)
async def get_status():
    """Get current system status."""
    try:
        status = await coordinator.get_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/results")
async def get_results():
    """Get all accepted submissions with formatting for display."""
    try:
        await _ensure_manual_aggregator_memory_loaded_for_read()
        # Return formatted results with submission separators for GUI display
        results = await coordinator.get_results_formatted()
        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/save-results")
async def save_results():
    """Save results to a .txt file with formatting."""
    try:
        await _ensure_manual_aggregator_memory_loaded_for_read()
        # Get formatted results with metadata headers
        results = await coordinator.get_results_formatted()
        
        # Save to downloads directory
        output_path = Path(system_config.data_dir) / "aggregator_results.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(results)
        
        return {
            "status": "saved",
            "path": output_path.name,
            "message": f"Results saved to {output_path.name}"
        }
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/clear-all")
async def clear_all_submissions():
    """Clear all accepted submissions and reset the system."""
    try:
        async with get_manual_proof_context_lock():
            if compiler_coordinator.is_running:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot clear Aggregator data while Compiler is running. Stop Compiler first.",
                )
            blocker = await _manual_proof_clear_blocker()
            if blocker:
                raise HTTPException(status_code=409, detail=blocker)
            if coordinator.is_running:
                await coordinator.stop()
            archived_proofs = await manual_proof_database.archive_current_run(
                Path(system_config.data_dir) / "manual_proof_runs",
                user_prompt=await load_manual_aggregator_prompt(),
                reason="manual_aggregator_clear_all",
            )
            await coordinator.clear_all_submissions()
            await clear_manual_aggregator_prompt()
        
        return {
            "status": "cleared",
            "message": "All submissions cleared and system reset",
            "archived_manual_proofs": archived_proofs,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear submissions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a user file."""
    try:
        safe_filename = validate_single_path_component(file.filename, "filename")
        if not safe_filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt uploads are supported")

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds 5 MB limit")

        uploads_dir = Path(system_config.user_uploads_dir)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        file_path = resolve_path_within_root(uploads_dir, safe_filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        return {
            "status": "uploaded",
            "filename": safe_filename,
            "path": safe_filename
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Rejected unsafe upload filename: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get available models from LM Studio."""
    try:
        models = await lm_studio_client.list_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/settings")
async def get_aggregator_settings():
    """Get current aggregator model settings."""
    try:
        settings = await coordinator.get_model_settings()
        return settings
    except Exception as e:
        logger.error(f"Failed to get aggregator settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/events")
async def get_events():
    """Get persisted aggregator events (acceptances, rejections, cleanup removals)."""
    try:
        await _ensure_manual_event_log_loaded_for_read()
        events = await event_log.get_all_events()
        return {"events": events}
    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
