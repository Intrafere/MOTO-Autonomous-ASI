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
from backend.shared.workflow_start_guard import workflow_start_guard
from backend.aggregator.core.coordinator import coordinator
from backend.aggregator.core.context_allocator import context_allocator
from backend.aggregator.memory.event_log import event_log
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/aggregator", tags=["aggregator"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


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
                    f"Submitter {config.submitter_id} {label}: model={config.model_id}, "
                    f"context={config.context_window}, max_tokens={config.max_output_tokens}"
                )
            logger.info(
                f"Validator: model={request.validator_model}, "
                f"context={request.validator_context_size}, max_tokens={request.validator_max_output_tokens}"
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
                validator_supercharge_enabled=request.validator_supercharge_enabled
            )

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
        # Model compatibility errors
        logger.error(f"Model compatibility error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Model compatibility error")
    
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
        await coordinator.clear_all_submissions()
        
        return {
            "status": "cleared",
            "message": "All submissions cleared and system reset"
        }
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
        events = await event_log.get_all_events()
        return {"events": events}
    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
