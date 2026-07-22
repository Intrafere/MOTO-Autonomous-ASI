"""
Aggregator API routes.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
import asyncio
import logging
from pathlib import Path
import aiofiles

from backend.shared.models import AggregatorStartRequest, SystemStatus, ModelInfo, ModelConfig
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.openrouter_client import OpenRouterClient
from backend.shared.config import system_config, rag_config
from backend.shared.embedding_readiness import require_embedding_provider_ready
from backend.shared.token_tracker import token_tracker
from backend.shared.path_safety import resolve_path_within_root, validate_single_path_component
from backend.shared.log_redaction import redact_log_text
from backend.shared.manual_proof_context import get_manual_proof_context_lock
from backend.shared.workflow_start_guard import WorkflowLease, workflow_start_guard
from backend.shared.api_client_manager import api_client_manager
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.aggregator.core.coordinator import coordinator
from backend.aggregator.core.context_allocator import context_allocator
from backend.aggregator.memory.event_log import event_log
from backend.aggregator.memory.shared_training import (
    clear_manual_aggregator_prompt,
    clear_manual_main_submitter_config,
    load_manual_aggregator_prompt,
    save_manual_aggregator_prompt,
    save_manual_main_submitter_config,
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
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".lean"}
AGGREGATOR_WORKFLOW_OWNER = "manual_aggregator"
_aggregator_workflow_lease: WorkflowLease | None = None


def _release_aggregator_workflow_lease() -> None:
    global _aggregator_workflow_lease
    workflow_start_guard.release(_aggregator_workflow_lease)
    _aggregator_workflow_lease = None


coordinator.top_level_terminal_callback = _release_aggregator_workflow_lease

MANUAL_PROOF_ACTIVE_KEYS = {
    "brainstorm:manual_aggregator",
    "paper:manual_compiler_current",
}


async def _delete_uploaded_file(file_ref: str) -> bool:
    """Delete a logical upload filename from the upload root."""
    safe_filename = validate_single_path_component(file_ref, "filename")
    if Path(safe_filename).suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError("Only .txt and .lean uploads are managed here")

    uploads_dir = Path(system_config.user_uploads_dir)
    file_path = resolve_path_within_root(uploads_dir, safe_filename)
    if not file_path.exists():
        return False
    if not file_path.is_file():
        raise ValueError("Upload path is not a file")

    await asyncio.to_thread(file_path.unlink)
    return True


async def _clear_uploaded_files() -> int:
    """Clear text/Lean user uploads so stale files cannot seed later workflows."""
    uploads_dir = Path(system_config.user_uploads_dir)
    if not uploads_dir.exists():
        return 0

    deleted = 0
    for file_path in uploads_dir.iterdir():
        if (
            file_path.is_file()
            and file_path.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS
        ):
            await asyncio.to_thread(file_path.unlink)
            deleted += 1
    return deleted


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


async def _require_openrouter_host_provider_available(
    *,
    label: str,
    provider: str,
    model_id: str,
    host_provider: Optional[str],
) -> None:
    """Reject stale pinned OpenRouter host providers before starting work."""
    if provider != "openrouter" or not model_id or not host_provider:
        return
    if not rag_config.openrouter_api_key:
        return

    client = OpenRouterClient(rag_config.openrouter_api_key)
    try:
        endpoints = await client.get_model_endpoints(model_id)
    finally:
        await client.close()

    available_hosts = {
        endpoint.get("provider_name")
        for endpoint in endpoints
        if isinstance(endpoint, dict) and endpoint.get("provider_name")
    }
    if host_provider not in available_hosts:
        hosts_text = ", ".join(sorted(available_hosts)) if available_hosts else "none"
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} OpenRouter host provider '{host_provider}' is not currently "
                f"available for model '{model_id}'. Set Host Provider to Auto or choose "
                f"one of the currently available hosts: {hosts_text}."
            ),
        )


def _get_start_conflict() -> Optional[str]:
    """Return a user-facing conflict message if another workflow is active."""
    if workflow_start_guard.active_owner:
        if workflow_start_guard.active_owner == AGGREGATOR_WORKFLOW_OWNER:
            return "Aggregator is already running"
        return "Cannot start Aggregator while another workflow is running. Stop it first."
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
    global _aggregator_workflow_lease
    manual_solution_path = None
    parent_start_committed = False
    coordinator_started = False
    try:
        async with workflow_start_guard.reserve():
            conflict = _get_start_conflict()
            if conflict:
                raise HTTPException(status_code=400, detail=conflict)

            if not request.user_prompt.strip():
                raise HTTPException(status_code=400, detail="Aggregator user prompt is required.")

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
            effective_assistant_context_size = (
                request.assistant_context_size
                if request.assistant_model
                else request.validator_context_size
            )
            effective_assistant_max_output_tokens = (
                request.assistant_max_output_tokens
                if request.assistant_model
                else request.validator_max_output_tokens
            )
            _require_valid_role_limits(
                effective_assistant_context_size,
                effective_assistant_max_output_tokens,
                "Assistant",
            )
            for config in request.submitter_configs:
                label = "Main submitter" if config.submitter_id == 1 else f"Submitter {config.submitter_id}"
                _require_valid_role_limits(config.context_window, config.max_output_tokens, label)
            await save_manual_aggregator_prompt(request.user_prompt)
            await require_embedding_provider_ready()

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
            assistant_model = request.assistant_model or request.validator_model
            assistant_provider = (
                request.assistant_provider
                if request.assistant_model
                else request.validator_provider
            )
            assistant_openrouter_provider = (
                request.assistant_openrouter_provider
                if request.assistant_model
                else request.validator_openrouter_provider
            )
            assistant_reasoning_effort = (
                request.assistant_openrouter_reasoning_effort
                if request.assistant_model
                else request.validator_openrouter_reasoning_effort
            )
            assistant_fallback = (
                request.assistant_lm_studio_fallback
                if request.assistant_model
                else request.validator_lm_studio_fallback
            )
            assistant_context_size = (
                effective_assistant_context_size
            )
            assistant_max_output_tokens = (
                effective_assistant_max_output_tokens
            )
            assistant_supercharge_enabled = (
                request.assistant_supercharge_enabled
                if request.assistant_model
                else request.validator_supercharge_enabled
            )
            for config in request.submitter_configs:
                label = "Main submitter" if config.submitter_id == 1 else f"Submitter {config.submitter_id}"
                await _require_openrouter_host_provider_available(
                    label=label,
                    provider=config.provider,
                    model_id=config.model_id,
                    host_provider=config.openrouter_provider,
                )
            await _require_openrouter_host_provider_available(
                label="Validator",
                provider=request.validator_provider,
                model_id=request.validator_model,
                host_provider=request.validator_openrouter_provider,
            )
            await _require_openrouter_host_provider_available(
                label="Assistant",
                provider=assistant_provider,
                model_id=assistant_model,
                host_provider=assistant_openrouter_provider,
            )
            api_client_manager.configure_role(
                "aggregator_assistant",
                ModelConfig(
                    provider=assistant_provider,
                    model_id=assistant_model,
                    openrouter_provider=assistant_openrouter_provider,
                    openrouter_reasoning_effort=assistant_reasoning_effort,
                    lm_studio_fallback_id=assistant_fallback,
                    context_window=assistant_context_size,
                    max_output_tokens=assistant_max_output_tokens,
                    supercharge_enabled=assistant_supercharge_enabled,
                ),
            )

            # One durable plan spans the active manual Aggregator -> Compiler run.
            from pathlib import Path
            from backend.shared.solution_path import (
                build_review_prompt,
                compact_review_prompt,
                review_with_json_retry,
                solution_path_registry,
            )
            primary_submitter = next(
                (
                    config
                    for config in request.submitter_configs
                    if config.submitter_id == 1
                ),
                None,
            )
            if primary_submitter is None:
                raise ValueError(
                    "Main Submitter 1 is required for solution-path review."
                )
            await save_manual_main_submitter_config(
                primary_submitter.model_dump(mode="json")
            )
            reviewer_role_id = "manual_solution_path_reviewer"
            api_client_manager.configure_role(
                reviewer_role_id,
                ModelConfig(
                    provider=primary_submitter.provider,
                    model_id=primary_submitter.model_id,
                    openrouter_model_id=(
                        primary_submitter.model_id
                        if primary_submitter.provider == "openrouter"
                        else None
                    ),
                    openrouter_provider=primary_submitter.openrouter_provider,
                    openrouter_reasoning_effort=primary_submitter.openrouter_reasoning_effort,
                    lm_studio_fallback_id=primary_submitter.lm_studio_fallback_id,
                    context_window=primary_submitter.context_window,
                    max_output_tokens=primary_submitter.max_output_tokens,
                    supercharge_enabled=primary_submitter.supercharge_enabled,
                ),
            )

            async def review_solution_path(proposal, current_plan):
                prompt = build_review_prompt(
                    user_prompt=request.user_prompt,
                    proposal=proposal,
                    current_plan=current_plan,
                )
                from backend.shared.response_extraction import extract_message_text

                async def call(messages):
                    return await api_client_manager.generate_completion(
                        task_id=f"agg_sub1_solution_path_{proposal.review_count:03d}",
                        role_id=reviewer_role_id,
                        model=primary_submitter.model_id,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=primary_submitter.max_output_tokens,
                    )

                return await review_with_json_retry(
                    prompt=prompt,
                    call_completion=call,
                    extract_text=lambda response: extract_message_text(
                        response["choices"][0]["message"]
                    ),
                    context_window=primary_submitter.context_window,
                    max_output_tokens=primary_submitter.max_output_tokens,
                    compact_prompt=compact_review_prompt(
                        user_prompt=request.user_prompt,
                        proposal=proposal,
                        current_plan=current_plan,
                    ),
                )

            manual_solution_path = await solution_path_registry.acquire(
                Path(system_config.data_dir) / "solution_paths",
                workflow_mode="manual",
                user_prompt=request.user_prompt,
                stable_run_id="manual",
                reviewer=review_solution_path,
            )
            # Keep Assistant as the final configured role for compatibility
            # with settings/defaulting observers; the dedicated reviewer keeps
            # its independent role ID and immutable copied configuration.
            api_client_manager.configure_role(
                "aggregator_assistant",
                ModelConfig(
                    provider=assistant_provider,
                    model_id=assistant_model,
                    openrouter_provider=assistant_openrouter_provider,
                    openrouter_reasoning_effort=assistant_reasoning_effort,
                    lm_studio_fallback_id=assistant_fallback,
                    context_window=assistant_context_size,
                    max_output_tokens=assistant_max_output_tokens,
                    supercharge_enabled=assistant_supercharge_enabled,
                ),
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
                solution_path_manager=manual_solution_path,
            )
            # Start coordinator
            token_tracker.reset()
            token_tracker.start_timer()
            await coordinator.start()
            coordinator_started = coordinator.is_running
            if not coordinator_started:
                raise RuntimeError("Aggregator did not enter running state")
            _aggregator_workflow_lease = workflow_start_guard.commit(
                AGGREGATOR_WORKFLOW_OWNER
            )
            parent_start_committed = True

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
    finally:
        if coordinator_started and not parent_start_committed:
            await coordinator.stop()
            token_tracker.stop_timer()
        if manual_solution_path is not None and not parent_start_committed:
            await manual_solution_path.stop()


@router.post("/stop")
async def stop_aggregator():
    """Stop the aggregator system."""
    async with workflow_start_guard.reserve():
        try:
            await coordinator.stop()
            if getattr(coordinator, "solution_path_manager", None) is not None:
                await coordinator.solution_path_manager.stop()
            await assistant_proof_search_coordinator.stop_all(
                broadcast=True,
                reason="aggregator_stopped",
            )
            if coordinator.is_running:
                raise RuntimeError("Aggregator remained active after stop")
            token_tracker.stop_timer()
            _release_aggregator_workflow_lease()
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


@router.get("/prompt")
async def get_prompt():
    """Get the durable manual Aggregator prompt."""
    try:
        return {"prompt": await load_manual_aggregator_prompt()}
    except Exception as e:
        logger.error(f"Failed to get manual Aggregator prompt: {e}")
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
        async with workflow_start_guard.reserve(), get_manual_proof_context_lock():
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
            _release_aggregator_workflow_lease()
            archived_proofs = await manual_proof_database.archive_current_run(
                Path(system_config.data_dir) / "manual_proof_runs",
                user_prompt=await load_manual_aggregator_prompt(),
                reason="manual_aggregator_clear_all",
            )
            await coordinator.clear_all_submissions()
            from backend.shared.solution_path import solution_path_registry
            await solution_path_registry.clear_run(
                Path(system_config.data_dir) / "solution_paths", "manual"
            )
            coordinator.solution_path_manager = None
            await assistant_proof_search_coordinator.stop_all(
                broadcast=True,
                reason="aggregator_cleared",
            )
            await assistant_proof_search_coordinator.clear_cooldown_state()
            await clear_manual_aggregator_prompt()
            await clear_manual_main_submitter_config()
            deleted_uploads = await _clear_uploaded_files()
        
        return {
            "status": "cleared",
            "message": "All submissions cleared and system reset",
            "archived_manual_proofs": archived_proofs,
            "deleted_uploads": deleted_uploads,
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
        if Path(safe_filename).suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only .txt and .lean uploads are supported")

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds 5 MB limit")
        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Uploads must be UTF-8 encoded text files")
        if not decoded_content.strip():
            raise HTTPException(status_code=400, detail="Upload is empty or contains only whitespace")

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


@router.delete("/upload-file/{filename}")
async def delete_uploaded_file(filename: str):
    """Remove a previously uploaded user file."""
    try:
        deleted = await _delete_uploaded_file(filename)
        return {
            "status": "deleted" if deleted else "not_found",
            "filename": validate_single_path_component(filename, "filename"),
            "deleted": deleted,
        }
    except ValueError as e:
        logger.warning("Rejected unsafe upload deletion request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete uploaded file: {e}")
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
