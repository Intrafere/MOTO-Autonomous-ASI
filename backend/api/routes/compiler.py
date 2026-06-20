"""
Compiler API routes.
"""
import asyncio
import hashlib
from fastapi import APIRouter, HTTPException
import logging
from pathlib import Path
from datetime import datetime
import aiofiles

from backend.api.routes import websocket
from backend.shared.models import CompilerStartRequest, CompilerState, CritiqueRequest, ModelConfig
from backend.shared.config import system_config
from backend.shared.embedding_readiness import require_embedding_provider_ready
from backend.shared.token_tracker import token_tracker
from backend.shared.api_client_manager import api_client_manager
from backend.shared.log_redaction import redact_log_text
from backend.shared.manual_proof_context import get_manual_proof_context_lock
from backend.shared.workflow_start_guard import workflow_start_guard
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.compiler.core.compiler_coordinator import CRITIQUE_ATTEMPT_TARGET, compiler_coordinator
from backend.compiler.memory.manual_prompt import (
    clear_manual_compiler_prompt,
    load_manual_compiler_prompt,
    save_manual_compiler_prompt,
)
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import paper_memory
from backend.aggregator.core.coordinator import coordinator
from backend.aggregator.memory.shared_training import (
    append_proof_to_manual_shared_training,
    clear_manual_shared_training_proof_appendix,
)
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.proof_database import manual_proof_database
from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator
from backend.shared.response_extraction import extract_message_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compiler", tags=["compiler"])

_compiler_proof_only_task: asyncio.Task | None = None
_saved_compiler_proof_tasks: set[asyncio.Task] = set()
MANUAL_AGGREGATOR_SOURCE_ID = "manual_aggregator"
MANUAL_PROOF_ACTIVE_KEYS = {
    "brainstorm:manual_aggregator",
    "paper:manual_compiler_current",
}


async def _release_pre_reserved_source(source_type: str, source_id: str, reserved: bool) -> None:
    if reserved and source_id:
        await ProofVerificationStage.release_source(source_type, source_id)


def _positive_int_setting(value, setting_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        raise ValueError(f"{setting_name} must be explicitly configured as a positive integer.")
    return parsed


def _validate_positive_role_limits(role_limits: dict[str, tuple[object, object]]) -> None:
    """Validate context/max-output limits before mutating shared runtime state."""
    for role, (context_window, max_tokens) in role_limits.items():
        context = _positive_int_setting(context_window, f"{role} context window")
        output = _positive_int_setting(max_tokens, f"{role} max output tokens")
        if output >= context:
            raise ValueError(f"{role} max output tokens must be smaller than its context window.")


def _strip_manual_aggregator_proof_appendix(content: str) -> str:
    marker = "=== PROOFS GENERATED FROM THIS BRAINSTORM"
    if marker not in content:
        return content
    return content.split(marker, 1)[0].rstrip()


async def _read_manual_aggregator_context(*, strip_proofs: bool = True) -> str:
    try:
        shared_path = Path(system_config.shared_training_file)
        if not shared_path.exists():
            return ""
        content = await asyncio.to_thread(shared_path.read_text, encoding="utf-8")
        return _strip_manual_aggregator_proof_appendix(content) if strip_proofs else content
    except Exception as exc:
        logger.debug("Unable to read manual compiler aggregator context for proof check: %s", exc)
        return ""


async def _build_saved_compiler_proof_content(full_content: str) -> str:
    paper_content = paper_library.strip_verified_proofs_from_content(full_content or "")
    source_context = (await _read_manual_aggregator_context()).strip()
    parts = [f"SAVED MANUAL COMPILER PAPER:\n{paper_content.strip()}"]
    if source_context:
        parts.append(f"PART 1 AGGREGATOR DATABASE CONTEXT:\n{source_context}")
    return "\n\n---\n\n".join(part for part in parts if part.strip())


async def _append_proof_to_saved_compiler_paper(output_path: Path, proof_record) -> bool:
    """Append a novel verified proof to the saved manual compiler paper file."""
    try:
        existing_content = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
        updated_content = paper_library.attach_verified_proofs_to_content(
            existing_content,
            proof_record,
            source_context="manual saved-paper proof check",
        )
        if updated_content == existing_content:
            return True
        await asyncio.to_thread(output_path.write_text, updated_content, encoding="utf-8")
        return True
    except Exception as exc:
        logger.error("Failed to append proof to saved compiler paper %s: %s", output_path.name, exc)
        return False


async def _run_saved_compiler_paper_proof_check(
    full_content: str,
    source_title: str,
    proof_config: dict,
    output_path: Path,
    *,
    source_id: str = "",
    source_reserved: bool = False,
) -> None:
    """Run autonomous proof extraction/tiering for a saved manual compiler paper."""
    try:
        if not proof_config.get("lean4_enabled"):
            logger.info("Skipping saved compiler paper proof check: Lean 4 disabled")
            return
        if not full_content.strip():
            return
        source_content = paper_library.strip_verified_proofs_from_content(full_content)
        proof_content = await _build_saved_compiler_proof_content(full_content)
        submitter_model = str(proof_config.get("submitter_model") or "")
        validator_model = str(proof_config.get("validator_model") or "")
        if not submitter_model:
            logger.warning("Skipping saved compiler paper proof check: Rigor & Proofs model is unavailable")
            return
        if not validator_model:
            logger.warning("Skipping saved compiler paper proof check: validator model is unavailable")
            return

        source_hash = hashlib.sha256(source_content.encode("utf-8")).hexdigest()[:16]
        source_id = source_id or f"compiler_manual_{source_hash}"
        role_suffix = "compiler_manual_paper"

        submitter_context = proof_config.get("submitter_context")
        submitter_max_tokens = proof_config.get("submitter_max_tokens")
        validator_context = proof_config.get("validator_context")
        validator_max_tokens = proof_config.get("validator_max_tokens")

        submitter_config = ModelConfig(
            provider=str(proof_config.get("submitter_provider") or "lm_studio"),
            model_id=submitter_model,
            openrouter_provider=proof_config.get("submitter_openrouter_provider"),
            openrouter_reasoning_effort=proof_config.get("submitter_openrouter_reasoning_effort", "auto"),
            lm_studio_fallback_id=proof_config.get("submitter_lm_studio_fallback"),
            context_window=_positive_int_setting(submitter_context, "submitter proof context window"),
            max_output_tokens=_positive_int_setting(submitter_max_tokens, "submitter proof max output tokens"),
            supercharge_enabled=bool(proof_config.get("submitter_supercharge_enabled", False)),
        )
        validator_config = ModelConfig(
            provider=str(proof_config.get("validator_provider") or "lm_studio"),
            model_id=validator_model,
            openrouter_provider=proof_config.get("validator_openrouter_provider"),
            openrouter_reasoning_effort=proof_config.get("validator_openrouter_reasoning_effort", "auto"),
            lm_studio_fallback_id=proof_config.get("validator_lm_studio_fallback"),
            context_window=_positive_int_setting(validator_context, "validator proof context window"),
            max_output_tokens=_positive_int_setting(validator_max_tokens, "validator proof max output tokens"),
            supercharge_enabled=bool(proof_config.get("validator_supercharge_enabled", False)),
        )
        for role_id in (
            f"autonomous_proof_identification_{role_suffix}",
            f"autonomous_proof_lemma_search_{role_suffix}",
            f"autonomous_proof_formalization_{role_suffix}",
        ):
            api_client_manager.configure_role(role_id, submitter_config)
        api_client_manager.configure_role("autonomous_proof_novelty", validator_config)

        stage = ProofVerificationStage()
        await stage.run(
            content=proof_content,
            source_type="paper",
            source_id=source_id,
            user_prompt=manual_proof_database.inject_into_prompt(str(proof_config.get("user_prompt") or "")),
            submitter_model=submitter_model,
            submitter_context=submitter_config.context_window,
            submitter_max_tokens=submitter_config.max_output_tokens,
            validator_model=validator_model,
            validator_context=validator_config.context_window,
            validator_max_tokens=validator_config.max_output_tokens,
            broadcast_fn=websocket.broadcast_event,
            novel_proofs_db=manual_proof_database,
            source_title=source_title,
            role_suffix_override=role_suffix,
            trigger="manual_compiler_save",
            source_reserved=source_reserved,
            release_source_on_exit=False,
            append_to_source=False,
            append_proof_callback=lambda proof: _append_proof_to_saved_compiler_paper(output_path, proof),
        )
    finally:
        await _release_pre_reserved_source("paper", source_id, source_reserved)
        await assistant_proof_search_coordinator.stop_all(
            broadcast=True,
            reason="saved_compiler_paper_proof_check_complete",
        )


def _get_start_conflict() -> str | None:
    """Return a user-facing conflict message if another workflow is active."""
    if compiler_coordinator.is_running:
        return "Compiler is already running"

    if _compiler_proof_only_task and not _compiler_proof_only_task.done():
        return "Compiler proof verification is already running"

    if coordinator.is_running:
        return "Cannot start Compiler while Aggregator is running. Stop Aggregator first."

    autonomous_state = autonomous_coordinator.get_state()
    if autonomous_state.is_running or autonomous_coordinator.is_active:
        return "Cannot start Compiler while Autonomous Research is running. Stop Autonomous Research first."

    if leanoj_coordinator.is_active:
        return "Cannot start Compiler while Proof Solver is running. Stop Proof Solver first."

    return None


async def _run_compiler_aggregator_proof_check(
    request: CompilerStartRequest,
    *,
    source_reserved: bool = False,
) -> None:
    """Run proof verification over the manual Aggregator database without writing a paper."""
    try:
        token_tracker.reset()
        token_tracker.start_timer()
        content = await _read_manual_aggregator_context()
        if not content.strip():
            await websocket.broadcast_event(
                "compiler_proof_check_skipped",
                {"reason": "Aggregator database is empty."},
            )
            return

        source_id = MANUAL_AGGREGATOR_SOURCE_ID
        role_suffix = "compiler_aggregator"

        submitter_config = ModelConfig(
            provider=request.high_param_provider,
            model_id=request.high_param_model,
            openrouter_provider=request.high_param_openrouter_provider,
            openrouter_reasoning_effort=request.high_param_openrouter_reasoning_effort,
            lm_studio_fallback_id=request.high_param_lm_studio_fallback,
            context_window=request.high_param_context_size,
            max_output_tokens=request.high_param_max_output_tokens,
            supercharge_enabled=request.high_param_supercharge_enabled,
        )
        validator_config = ModelConfig(
            provider=request.validator_provider,
            model_id=request.validator_model,
            openrouter_provider=request.validator_openrouter_provider,
            openrouter_reasoning_effort=request.validator_openrouter_reasoning_effort,
            lm_studio_fallback_id=request.validator_lm_studio_fallback,
            context_window=request.validator_context_size,
            max_output_tokens=request.validator_max_output_tokens,
            supercharge_enabled=request.validator_supercharge_enabled,
        )
        for role_id in (
            f"autonomous_proof_identification_{role_suffix}",
            f"autonomous_proof_lemma_search_{role_suffix}",
            f"autonomous_proof_formalization_{role_suffix}",
        ):
            api_client_manager.configure_role(role_id, submitter_config)
        api_client_manager.configure_role("autonomous_proof_novelty", validator_config)
        api_client_manager.configure_role(
            "compiler_assistant",
            ModelConfig(
                provider=(
                    request.assistant_provider
                    if request.assistant_model
                    else request.validator_provider
                ),
                model_id=request.assistant_model or request.validator_model,
                openrouter_provider=(
                    request.assistant_openrouter_provider
                    if request.assistant_model
                    else request.validator_openrouter_provider
                ),
                openrouter_reasoning_effort=(
                    request.assistant_openrouter_reasoning_effort
                    if request.assistant_model
                    else request.validator_openrouter_reasoning_effort
                ),
                lm_studio_fallback_id=(
                    request.assistant_lm_studio_fallback
                    if request.assistant_model
                    else request.validator_lm_studio_fallback
                ),
                context_window=request.assistant_context_size if request.assistant_model else request.validator_context_size,
                max_output_tokens=request.assistant_max_output_tokens if request.assistant_model else request.validator_max_output_tokens,
                supercharge_enabled=(
                    request.assistant_supercharge_enabled
                    if request.assistant_model
                    else request.validator_supercharge_enabled
                ),
            ),
        )

        await websocket.broadcast_event(
            "compiler_proof_check_started",
            {"source_type": "brainstorm", "source_id": source_id},
        )
        stage = ProofVerificationStage()
        await stage.run(
            content=f"PART 1 AGGREGATOR DATABASE:\n{content}",
            source_type="brainstorm",
            source_id=source_id,
            user_prompt=manual_proof_database.inject_into_prompt(request.compiler_prompt),
            submitter_model=request.high_param_model,
            submitter_context=request.high_param_context_size,
            submitter_max_tokens=request.high_param_max_output_tokens,
            validator_model=request.validator_model,
            validator_context=request.validator_context_size,
            validator_max_tokens=request.validator_max_output_tokens,
            broadcast_fn=websocket.broadcast_event,
            novel_proofs_db=manual_proof_database,
            source_title=request.compiler_prompt or "Compiler Aggregator Database",
            role_suffix_override=role_suffix,
            trigger="manual_compiler_aggregator",
            source_reserved=source_reserved,
            append_to_source=False,
            append_proof_callback=append_proof_to_manual_shared_training,
        )
        await websocket.broadcast_event(
            "compiler_proof_check_complete",
            {"source_type": "brainstorm", "source_id": source_id},
        )
    finally:
        await _release_pre_reserved_source("brainstorm", MANUAL_AGGREGATOR_SOURCE_ID, source_reserved)
        await assistant_proof_search_coordinator.stop_all(
            broadcast=True,
            reason="compiler_aggregator_proof_check_complete",
        )
        token_tracker.stop_timer()


def _log_background_task_failure(task: asyncio.Task) -> None:
    _saved_compiler_proof_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info("Saved compiler paper proof check was cancelled")
    except Exception:
        logger.exception("Saved compiler paper proof check failed")


async def _manual_proof_clear_blocker() -> str | None:
    """Return a blocker message if manual proof work could write stale proofs."""
    active_keys = await ProofVerificationStage.active_source_keys()
    for key in active_keys:
        if (
            key in MANUAL_PROOF_ACTIVE_KEYS
            or key.startswith("paper:compiler_manual_")
            or key.startswith("paper:manual_compiler_")
        ):
            return "Cannot clear the manual run while manual proof verification is running. Stop or wait for proof verification to finish first."
    if _compiler_proof_only_task and not _compiler_proof_only_task.done():
        return "Cannot clear the manual run while compiler proof verification is running."
    if any(not task.done() for task in _saved_compiler_proof_tasks):
        return "Cannot clear the manual run while saved-paper proof verification is running."
    return None


@router.post("/start")
async def start_compiler(request: CompilerStartRequest):
    """Start the compiler system."""
    global _compiler_proof_only_task
    try:
        async with workflow_start_guard.reserve():
            conflict = _get_start_conflict()
            if conflict:
                raise HTTPException(status_code=400, detail=conflict)

            if not request.compiler_prompt.strip():
                raise HTTPException(status_code=400, detail="Compiler prompt is required.")

            if not request.allow_mathematical_proofs and not request.allow_research_papers:
                raise HTTPException(
                    status_code=400,
                    detail="At least one allowed output must be enabled.",
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
            _validate_positive_role_limits({
                "validator": (request.validator_context_size, request.validator_max_output_tokens),
                "Writing Submitter": (request.writer_context_size, request.writer_max_output_tokens),
                "Rigor & Proofs submitter": (request.high_param_context_size, request.high_param_max_output_tokens),
                "assistant": (effective_assistant_context_size, effective_assistant_max_output_tokens),
            })
            await save_manual_compiler_prompt(request.compiler_prompt)

            effective_allow_mathematical_proofs = bool(
                request.allow_mathematical_proofs and not system_config.generic_mode
            )
            if request.allow_mathematical_proofs and not system_config.lean4_enabled:
                if not (system_config.generic_mode and request.allow_research_papers):
                    raise HTTPException(
                        status_code=501,
                        detail={
                            "lean4_enabled": False,
                            "message": "Mathematical proof output requires Lean 4 proof verification to be enabled.",
                        },
                    )

            if not request.allow_research_papers:
                if not effective_allow_mathematical_proofs:
                    raise HTTPException(status_code=400, detail="At least one allowed output must be enabled.")
                if not system_config.lean4_enabled:
                    raise HTTPException(
                        status_code=501,
                        detail={
                            "lean4_enabled": False,
                            "message": "Mathematical proof output requires Lean 4 proof verification to be enabled.",
                        },
                    )
                async with get_manual_proof_context_lock():
                    try:
                        await ProofVerificationStage.reserve_source("brainstorm", MANUAL_AGGREGATOR_SOURCE_ID)
                    except RuntimeError:
                        raise HTTPException(status_code=409, detail="A proof verification is already running for the manual Aggregator database.")
                    _compiler_proof_only_task = asyncio.create_task(
                        _run_compiler_aggregator_proof_check(request, source_reserved=True)
                    )
                    _compiler_proof_only_task.add_done_callback(_log_background_task_failure)
                return {
                    "status": "proof_check_started",
                    "message": "Compiler proof verification started over the Aggregator database",
                }

            await require_embedding_provider_ready()
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
            api_client_manager.configure_role(
                "compiler_assistant",
                ModelConfig(
                    provider=assistant_provider,
                    model_id=assistant_model,
                    openrouter_provider=assistant_openrouter_provider,
                    openrouter_reasoning_effort=assistant_reasoning_effort,
                    lm_studio_fallback_id=assistant_fallback,
                    context_window=effective_assistant_context_size,
                    max_output_tokens=effective_assistant_max_output_tokens,
                    supercharge_enabled=(
                        request.assistant_supercharge_enabled
                        if request.assistant_model
                        else request.validator_supercharge_enabled
                    ),
                ),
            )

            # Update system config with user-provided context sizes
            system_config.compiler_validator_context_window = request.validator_context_size
            system_config.compiler_writer_context_window = request.writer_context_size
            system_config.compiler_high_param_context_window = request.high_param_context_size
            system_config.compiler_critique_submitter_context_window = request.high_param_context_size

            # Update max output token configurations
            system_config.compiler_validator_max_output_tokens = request.validator_max_output_tokens
            system_config.compiler_writer_max_output_tokens = request.writer_max_output_tokens
            system_config.compiler_high_param_max_output_tokens = request.high_param_max_output_tokens
            system_config.compiler_critique_submitter_max_tokens = request.high_param_max_output_tokens

            # Deprecated critique fields are compatibility aliases for Rigor & Proofs.
            system_config.compiler_critique_submitter_model = request.high_param_model

            logger.info(
                "Compiler max output tokens - Validator: %s, Writing Submitter: %s, Rigor & Proofs: %s",
                redact_log_text(request.validator_max_output_tokens, 40),
                redact_log_text(request.writer_max_output_tokens, 40),
                redact_log_text(request.high_param_max_output_tokens, 40),
            )

            # Initialize coordinator with OpenRouter provider configurations
            await compiler_coordinator.initialize(
                compiler_prompt=request.compiler_prompt,
                validator_model=request.validator_model,
                writer_model=request.writer_model,
                high_param_model=request.high_param_model,
                critique_submitter_model=request.high_param_model,
                # OpenRouter provider configs for each role
                validator_provider=request.validator_provider,
                validator_openrouter_provider=request.validator_openrouter_provider,
                validator_openrouter_reasoning_effort=request.validator_openrouter_reasoning_effort,
                validator_lm_studio_fallback=request.validator_lm_studio_fallback,
                writer_provider=request.writer_provider,
                writer_openrouter_provider=request.writer_openrouter_provider,
                writer_openrouter_reasoning_effort=request.writer_openrouter_reasoning_effort,
                writer_lm_studio_fallback=request.writer_lm_studio_fallback,
                high_param_provider=request.high_param_provider,
                high_param_openrouter_provider=request.high_param_openrouter_provider,
                high_param_openrouter_reasoning_effort=request.high_param_openrouter_reasoning_effort,
                high_param_lm_studio_fallback=request.high_param_lm_studio_fallback,
                critique_submitter_provider=request.high_param_provider,
                critique_submitter_openrouter_provider=request.high_param_openrouter_provider,
                critique_submitter_openrouter_reasoning_effort=request.high_param_openrouter_reasoning_effort,
                critique_submitter_lm_studio_fallback=request.high_param_lm_studio_fallback,
                validator_supercharge_enabled=request.validator_supercharge_enabled,
                writer_supercharge_enabled=request.writer_supercharge_enabled,
                high_param_supercharge_enabled=request.high_param_supercharge_enabled,
                critique_submitter_supercharge_enabled=request.high_param_supercharge_enabled,
                allow_mathematical_proofs=effective_allow_mathematical_proofs
            )

            # Start coordinator
            token_tracker.reset()
            token_tracker.start_timer()
            await compiler_coordinator.start()

            return {"status": "started", "message": "Compiler started successfully"}
    
    except HTTPException:
        raise
    except ValueError as e:
        # Configuration/model compatibility errors - provide structured error response
        error_msg = str(e)
        is_settings_error = any(
            marker in error_msg.lower()
            for marker in ("context", "max output", "max_output", "tokens", "positive integer", "configured")
        )
        logger.error(f"Compiler configuration error: {e}", exc_info=True)
        
        # Determine which model failed
        failed_model_type = "unknown"
        failed_model_name = ""
        
        if request.validator_model in error_msg:
            failed_model_type = "validator"
            failed_model_name = request.validator_model
        elif request.writer_model in error_msg:
            failed_model_type = "writer"
            failed_model_name = request.writer_model
        elif request.high_param_model in error_msg:
            failed_model_type = "high_param"
            failed_model_name = request.high_param_model
        
        # Extract reason from error message
        reason = error_msg
        if "Model incompatibility detected:" in error_msg:
            reason = error_msg.split("Model incompatibility detected:")[1].split(".")[0].strip()
        
        error_response = {
            "error": "configuration_error" if is_settings_error else "model_compatibility",
            "failed_model_type": failed_model_type,
            "failed_model_name": failed_model_name,
            "reason": reason,
            "suggestion": (
                "Configure positive context window and max output token values for every compiler role in Settings."
                if is_settings_error
                else "Try using a compatible model or click 'Use Aggregator Models' to auto-fill working models."
            ),
            "full_error": error_msg
        }
        
        raise HTTPException(status_code=400, detail=error_response)
    
    except Exception as e:
        # Other errors
        logger.error(f"Failed to start compiler: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stop")
async def stop_compiler():
    """Stop the compiler system."""
    global _compiler_proof_only_task
    try:
        if _compiler_proof_only_task and not _compiler_proof_only_task.done():
            _compiler_proof_only_task.cancel()
            await asyncio.gather(_compiler_proof_only_task, return_exceptions=True)
            _compiler_proof_only_task = None
        await compiler_coordinator.stop()
        await assistant_proof_search_coordinator.stop_all(
            broadcast=True,
            reason="compiler_stopped",
        )
        token_tracker.stop_timer()
        return {"status": "stopped", "message": "Compiler stopped"}
    except Exception as e:
        logger.error(f"Failed to stop compiler: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/test-models")
async def test_models(request: CompilerStartRequest):
    """Test model compatibility without starting the compiler."""
    if system_config.generic_mode:
        raise HTTPException(
            status_code=501,
            detail={
                "generic_mode": True,
                "message": "LM Studio model diagnostics are unavailable in generic hosted mode.",
            },
        )

    from backend.shared.lm_studio_client import lm_studio_client
    
    results = {
        "validator": {"model": request.validator_model, "passed": False, "error": "", "details": {}},
        "writer": {"model": request.writer_model, "passed": False, "error": "", "details": {}},
        "high_param": {"model": request.high_param_model, "passed": False, "error": "", "details": {}}
    }
    
    # Test validator model
    is_compat, error, details = await lm_studio_client.test_model_compatibility(
        request.validator_model,
        request.validator_max_output_tokens,
    )
    results["validator"]["passed"] = is_compat
    results["validator"]["error"] = error
    results["validator"]["details"] = details
    
    # Test writer model
    is_compat, error, details = await lm_studio_client.test_model_compatibility(
        request.writer_model,
        request.writer_max_output_tokens,
    )
    results["writer"]["passed"] = is_compat
    results["writer"]["error"] = error
    results["writer"]["details"] = details
    
    # Test Rigor & Proofs model
    is_compat, error, details = await lm_studio_client.test_model_compatibility(
        request.high_param_model,
        request.high_param_max_output_tokens,
    )
    results["high_param"]["passed"] = is_compat
    results["high_param"]["error"] = error
    results["high_param"]["details"] = details
    
    all_passed = all(r["passed"] for r in results.values())
    
    return {
        "all_passed": all_passed,
        "results": results,
        "suggestion": "Use 'openai/gpt-oss-20b' or 'openai/gpt-oss-20b:3' for best compatibility" if not all_passed else ""
    }


@router.get("/status", response_model=CompilerState)
async def get_status():
    """Get current compiler status."""
    try:
        if _compiler_proof_only_task and not _compiler_proof_only_task.done():
            return CompilerState(is_running=True, current_mode="proof_verification")
        status = await compiler_coordinator.get_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/prompt")
async def get_prompt():
    """Get the durable manual Compiler prompt."""
    try:
        return {"prompt": await load_manual_compiler_prompt()}
    except Exception as e:
        logger.error(f"Failed to get manual Compiler prompt: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/paper")
async def get_paper():
    """Get current paper content (includes outline prepended)."""
    try:
        outline = await outline_memory.get_outline()
        paper = await paper_memory.get_paper()
        word_count = await paper_memory.get_word_count()
        
        # Prepend outline to paper for display
        full_content = ""
        if outline:
            full_content = f"OUTLINE:\n{'='*80}\n\n{outline}\n\n{'='*80}\n\nPAPER:\n{'='*80}\n\n{paper}"
        else:
            full_content = paper
        
        return {
            "paper": full_content,
            "word_count": word_count,
            "version": paper_memory.get_version()
        }
    except Exception as e:
        logger.error(f"Failed to get paper: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/outline")
async def get_outline():
    """Get current outline."""
    try:
        outline = await outline_memory.get_outline()
        
        return {
            "outline": outline,
            "version": outline_memory.get_version()
        }
    except Exception as e:
        logger.error(f"Failed to get outline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/save-paper")
async def save_paper():
    """Save paper to a .txt file (includes author attribution, outline, and paper content)."""
    async with get_manual_proof_context_lock():
        return await _save_paper_unlocked()


async def _save_paper_unlocked():
    """Save paper while the manual proof context lock is held."""
    try:
        outline = await outline_memory.get_outline()
        paper = await paper_memory.get_paper()
        word_count = await paper_memory.get_word_count()
        persisted_prompt = compiler_coordinator.user_prompt or await load_manual_compiler_prompt()
        
        # Get model tracking data for author attribution
        model_data = compiler_coordinator.get_model_tracking_data()
        
        # Generate author attribution if model tracking data is available
        attribution_section = ""
        credits_section = ""
        if model_data and model_data.get("model_usage"):
            from backend.autonomous.memory.paper_model_tracker import (
                generate_attribution_for_existing_paper,
                generate_credits_for_existing_paper
            )
            
            # Parse generation date if available
            gen_date = None
            if model_data.get("generation_date"):
                try:
                    gen_date = datetime.fromisoformat(model_data["generation_date"])
                except (TypeError, ValueError) as exc:
                    logger.debug("Ignoring invalid saved compiler generation date: %s", exc)
            
            # Generate attribution header (no reference papers for manual mode)
            attribution_section = generate_attribution_for_existing_paper(
                user_prompt=persisted_prompt,
                paper_title=compiler_coordinator.paper_title or persisted_prompt,
                model_usage=model_data["model_usage"],
                generation_date=gen_date,
                reference_paper_models=None  # No reference papers in manual mode
            )
            
            # Generate credits footer (including Wolfram calls if available)
            wolfram_count = model_data.get("wolfram_calls", 0)
            credits_section = generate_credits_for_existing_paper(
                model_data["model_usage"],
                wolfram_calls=wolfram_count
            )
        
        # Build full content with attribution
        full_content_parts = []
        
        # Add attribution header if available
        if attribution_section:
            full_content_parts.append(attribution_section)
        
        # Add outline and paper content
        if outline:
            full_content_parts.append(f"OUTLINE:\n{'='*80}\n\n{outline}\n\n{'='*80}\n\nPAPER:\n{'='*80}\n\n{paper}")
        else:
            full_content_parts.append(paper)
        
        # Add credits footer if available
        if credits_section:
            full_content_parts.append(credits_section)
        
        full_content = "\n".join(full_content_parts)
        
        # Save to output directory
        output_path = Path(system_config.data_dir) / "compiler_paper_saved.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(full_content)

        rigor_submitter = compiler_coordinator.high_param_submitter
        proof_check_scheduled = bool(
            system_config.lean4_enabled
            and getattr(compiler_coordinator, "allow_mathematical_proofs", True)
            and full_content.strip()
            and rigor_submitter is not None
            and getattr(rigor_submitter, "model_name", "")
            and compiler_coordinator.validator_model
        )
        if proof_check_scheduled:
            source_title = compiler_coordinator.paper_title or persisted_prompt or "Compiler Paper"
            proof_source_content = paper_library.strip_verified_proofs_from_content(full_content)
            proof_source_hash = hashlib.sha256(proof_source_content.encode("utf-8")).hexdigest()[:16]
            proof_source_id = f"compiler_manual_{proof_source_hash}"
            proof_config = {
                "lean4_enabled": system_config.lean4_enabled,
                "user_prompt": persisted_prompt,
                "submitter_model": rigor_submitter.model_name,
                "submitter_provider": compiler_coordinator.high_param_provider,
                "submitter_openrouter_provider": compiler_coordinator.high_param_openrouter_provider,
                "submitter_openrouter_reasoning_effort": compiler_coordinator.high_param_openrouter_reasoning_effort,
                "submitter_lm_studio_fallback": compiler_coordinator.high_param_lm_studio_fallback,
                "submitter_context": system_config.compiler_high_param_context_window,
                "submitter_max_tokens": system_config.compiler_high_param_max_output_tokens,
                "submitter_supercharge_enabled": getattr(compiler_coordinator, "high_param_supercharge_enabled", False),
                "validator_model": compiler_coordinator.validator_model,
                "validator_provider": compiler_coordinator.validator_provider,
                "validator_openrouter_provider": compiler_coordinator.validator_openrouter_provider,
                "validator_openrouter_reasoning_effort": compiler_coordinator.validator_openrouter_reasoning_effort,
                "validator_lm_studio_fallback": compiler_coordinator.validator_lm_studio_fallback,
                "validator_context": compiler_coordinator.validator_context_window,
                "validator_max_tokens": compiler_coordinator.validator_max_tokens,
                "validator_supercharge_enabled": getattr(compiler_coordinator, "validator_supercharge_enabled", False),
            }
            try:
                await ProofVerificationStage.reserve_source("paper", proof_source_id)
            except RuntimeError:
                proof_check_scheduled = False
                logger.info("Saved compiler paper proof check already running for source %s", proof_source_id)
                return {
                    "status": "saved",
                    "path": output_path.name,
                    "word_count": word_count,
                    "message": f"Paper saved to {output_path.name} ({word_count} words)",
                    "has_attribution": bool(attribution_section),
                    "proof_check_scheduled": False,
                }
            task = asyncio.create_task(
                _run_saved_compiler_paper_proof_check(
                    full_content,
                    source_title,
                    proof_config,
                    output_path,
                    source_id=proof_source_id,
                    source_reserved=True,
                )
            )
            _saved_compiler_proof_tasks.add(task)
            task.add_done_callback(_log_background_task_failure)
        
        return {
            "status": "saved",
            "path": output_path.name,
            "word_count": word_count,
            "message": f"Paper saved to {output_path.name} ({word_count} words)",
            "has_attribution": bool(attribution_section),
            "proof_check_scheduled": proof_check_scheduled
        }
    except Exception as e:
        logger.error(f"Failed to save paper: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metrics")
async def get_metrics():
    """Get compiler metrics."""
    try:
        status = await compiler_coordinator.get_status()
        
        # Calculate acceptance rates
        construction_total = status.construction_acceptances + status.construction_rejections
        construction_rate = (
            status.construction_acceptances / construction_total 
            if construction_total > 0 else 0.0
        )
        
        rigor_total = status.rigor_acceptances + status.rigor_rejections
        rigor_rate = (
            status.rigor_acceptances / rigor_total 
            if rigor_total > 0 else 0.0
        )
        
        return {
            "total_submissions": status.total_submissions,
            "construction": {
                "acceptances": status.construction_acceptances,
                "rejections": status.construction_rejections,
                "declines": status.construction_declines,
                "acceptance_rate": construction_rate
            },
            "rigor": {
                "acceptances": status.rigor_acceptances,
                "rejections": status.rigor_rejections,
                "declines": status.rigor_declines,
                "acceptance_rate": rigor_rate
            },
            "outline": {
                "acceptances": status.outline_acceptances,
                "rejections": status.outline_rejections,
                "declines": status.outline_declines
            },
            "review": {
                "acceptances": status.review_acceptances,
                "rejections": status.review_rejections,
                "declines": status.review_declines
            },
            "minuscule_edit_count": status.minuscule_edit_count,
            "paper_word_count": status.paper_word_count
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/clear-paper")
async def clear_paper(confirm: bool = False):
    """Clear the current paper and outline, reset to fresh start.
    
    Args:
        confirm: Must be True to proceed with reset (prevents accidental resets)
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Pass confirm=true to clear paper."
        )
    
    try:
        async with get_manual_proof_context_lock():
            blocker = await _manual_proof_clear_blocker()
            if blocker:
                raise HTTPException(status_code=409, detail=blocker)
            if compiler_coordinator.is_running:
                await compiler_coordinator.stop()
            persisted_prompt = compiler_coordinator.user_prompt or await load_manual_compiler_prompt()
            archived_proofs = await manual_proof_database.archive_current_run(
                Path(system_config.data_dir) / "manual_proof_runs",
                user_prompt=persisted_prompt,
                reason="manual_compiler_clear_paper",
            )
            await clear_manual_shared_training_proof_appendix()
            await compiler_coordinator.clear_paper()
            await assistant_proof_search_coordinator.stop_all(
                broadcast=True,
                reason="compiler_cleared",
            )
            await clear_manual_compiler_prompt()
        
        # Also clear any paper critiques
        from backend.shared.critique_memory import clear_critiques
        try:
            await clear_critiques("compiler_paper")
            logger.info("Cleared compiler paper critiques")
        except Exception as e:
            logger.warning(f"Failed to clear compiler critiques: {e}")
        
        return {
            "status": "cleared",
            "message": "Paper and outline cleared - system reset to fresh start",
            "archived_manual_proofs": archived_proofs,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear paper: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/critique-status")
async def get_critique_status():
    """Get critique phase status."""
    try:
        return {
            "in_critique_phase": compiler_coordinator.in_critique_phase,
            "critique_acceptances": compiler_coordinator.critique_acceptances,
            "paper_version": compiler_coordinator.paper_version,
            "target_critiques": CRITIQUE_ATTEMPT_TARGET
        }
    except Exception as e:
        logger.error(f"Failed to get critique status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/previous-versions")
async def get_previous_versions():
    """Get all previous body versions."""
    try:
        versions = await paper_memory.get_previous_versions()
        return {"previous_versions": versions}
    except Exception as e:
        logger.error(f"Failed to get previous versions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# PAPER CRITIQUE ENDPOINTS (Validator Critique Feature)
# ============================================================================


@router.post("/critique-paper")
async def request_compiler_critique(critique_request: CritiqueRequest = None):
    """
    Request a critique of the current compiler paper from the validator model.
    
    The paper is direct-injected into the validator model for an honest critique.
    If the paper exceeds the validator's context window, an error is returned.
    
    Validator configuration can be provided in the request body, otherwise falls back
    to system config. This allows critique generation without the compiler running.
    
    Args:
        critique_request: Request body containing custom prompt and optional validator config
    
    Returns:
        The critique with ratings and feedback
    """
    from typing import Optional
    from backend.shared.critique_prompts import build_critique_prompt, DEFAULT_CRITIQUE_PROMPT
    from backend.shared.critique_memory import save_critique
    from backend.shared.models import PaperCritique
    from backend.shared.utils import count_tokens
    import uuid
    
    # Handle None critique_request (for backwards compatibility)
    if critique_request is None:
        critique_request = CritiqueRequest()
    
    try:
        # Get current paper content
        paper_content = await paper_memory.get_paper()
        if not paper_content or not paper_content.strip():
            raise HTTPException(
                status_code=400,
                detail="No paper content available. Please start the compiler and generate some content first."
            )
        
        # Extract custom prompt from request body
        custom_prompt = critique_request.custom_prompt
        
        # Initialize validator config with values from the request body, if provided
        validator_model = critique_request.validator_model
        validator_context_window = critique_request.validator_context_window
        validator_max_tokens = critique_request.validator_max_tokens
        validator_provider = critique_request.validator_provider
        validator_openrouter_provider = critique_request.validator_openrouter_provider
        validator_openrouter_reasoning_effort = critique_request.validator_openrouter_reasoning_effort
        validator_supercharge_enabled = bool(critique_request.validator_supercharge_enabled)
        
        # If validator config not provided in request, fall back to coordinator config
        if not validator_model:
            validator_model = getattr(compiler_coordinator, 'validator_model', None)
            validator_context_window = system_config.compiler_validator_context_window
            validator_max_tokens = system_config.compiler_validator_max_output_tokens
            validator_provider = getattr(compiler_coordinator, 'validator_provider', 'lm_studio')
            validator_openrouter_provider = getattr(compiler_coordinator, 'validator_openrouter_provider', None)
            validator_openrouter_reasoning_effort = getattr(compiler_coordinator, 'validator_openrouter_reasoning_effort', 'auto')
            validator_supercharge_enabled = bool(getattr(compiler_coordinator, 'validator_supercharge_enabled', False))
        
        if not validator_model:
            raise HTTPException(
                status_code=400,
                detail="No validator model configured. Please configure a validator model in Compiler Settings."
            )
        try:
            validator_context_window = _positive_int_setting(
                validator_context_window,
                "validator critique context window",
            )
            validator_max_tokens = _positive_int_setting(
                validator_max_tokens,
                "validator critique max output tokens",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        
        # Get paper title from coordinator or use prompt
        paper_title = None
        if compiler_coordinator.paper_title:
            paper_title = compiler_coordinator.paper_title
        elif compiler_coordinator.user_prompt:
            paper_title = compiler_coordinator.user_prompt[:100]  # Use first 100 chars of prompt as title
        
        # Build the critique prompt
        prompt_to_use = custom_prompt if custom_prompt else DEFAULT_CRITIQUE_PROMPT
        full_prompt = build_critique_prompt(paper_content, paper_title, prompt_to_use)
        
        # Count tokens in the prompt
        prompt_tokens = count_tokens(full_prompt)
        
        # Calculate available input tokens (context window - output reserve - safety margin)
        output_reserve = validator_max_tokens
        safety_margin = int(validator_context_window * 0.1)  # 10% safety margin
        available_input = validator_context_window - output_reserve - safety_margin
        
        # Check if paper fits in context window
        if prompt_tokens > available_input:
            excess_tokens = prompt_tokens - available_input
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Paper is too long for the validator's context window. "
                    f"The paper requires {prompt_tokens:,} tokens, but the validator can only accept {available_input:,} tokens "
                    f"(context window: {validator_context_window:,}, output reserve: {output_reserve:,}, safety margin: {safety_margin:,}). "
                    f"The paper exceeds the limit by {excess_tokens:,} tokens. "
                    f"A complete and honest review requires direct context injection - please select a validator with a larger context window."
                )
            )
        
        # Build messages for API call
        messages = [
            {"role": "user", "content": full_prompt}
        ]
        
        # Configure the paper_critic role with the validator settings BEFORE making the API call
        # This ensures routing goes to the correct provider (OpenRouter vs LM Studio)
        api_client_manager.configure_role(
            "paper_critic",
            ModelConfig(
                provider=validator_provider,
                model_id=validator_model,
                openrouter_model_id=validator_model if validator_provider == "openrouter" else None,
                openrouter_provider=validator_openrouter_provider,
                openrouter_reasoning_effort=validator_openrouter_reasoning_effort,
                lm_studio_fallback_id=None,  # No fallback for direct critique calls
                context_window=validator_context_window,
                max_output_tokens=validator_max_tokens,
                supercharge_enabled=validator_supercharge_enabled
            )
        )
        
        # Make the API call to the validator model
        logger.info(
            "Requesting critique for compiler paper from validator model %s",
            redact_log_text(validator_model, 160),
        )
        
        response = await api_client_manager.generate_completion(
            task_id=f"compiler_paper_critique_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            role_id="paper_critic",
            model=validator_model,
            messages=messages,
            max_tokens=validator_max_tokens,
            temperature=0.0
        )
        
        # Parse the response - extract from OpenAI-compatible response structure
        response_content = ""
        if response.get("choices"):
            message = response["choices"][0].get("message", {})
            response_content = extract_message_text(message)
        
        if not response_content:
            raise HTTPException(status_code=500, detail="Empty response from validator model")
        
        # Parse with lenient fallback for truncated critique responses
        from backend.shared.critique_prompts import parse_critique_response
        critique_data = parse_critique_response(response_content)
        
        # Create critique object
        critique = PaperCritique(
            critique_id=str(uuid.uuid4()),
            model_id=validator_model,
            provider=validator_provider,
            host_provider=validator_openrouter_provider,
            date=datetime.now(),
            prompt_used=prompt_to_use,
            critique_source="user_request",
            novelty_rating=critique_data.get("novelty_rating", 0),
            novelty_feedback=critique_data.get("novelty_feedback", ""),
            correctness_rating=critique_data.get("correctness_rating", 0),
            correctness_feedback=critique_data.get("correctness_feedback", ""),
            impact_rating=critique_data.get("impact_rating", 0),
            impact_feedback=critique_data.get("impact_feedback", ""),
            full_critique=critique_data.get("full_critique", "")
        )
        
        # Save the critique
        saved_critique = await save_critique("compiler_paper", critique)
        
        return {
            "success": True,
            "critique": saved_critique.model_dump(),
            "paper_title": paper_title or "Compiler Paper"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to request compiler paper critique: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/critiques")
async def get_compiler_critiques():
    """
    Get all critiques for the current compiler paper.
    
    Returns:
        List of critiques for the compiler paper
    """
    from backend.shared.critique_memory import get_critiques
    
    try:
        critiques = await get_critiques("compiler_paper")
        
        # Get paper title if available
        paper_title = None
        if compiler_coordinator.paper_title:
            paper_title = compiler_coordinator.paper_title
        elif compiler_coordinator.user_prompt:
            paper_title = compiler_coordinator.user_prompt[:100]
        
        return {
            "success": True,
            "paper_title": paper_title or "Compiler Paper",
            "critiques": [c.model_dump() for c in critiques],
            "count": len(critiques)
        }
        
    except Exception as e:
        logger.error(f"Failed to get compiler paper critiques: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/critiques")
async def delete_compiler_critiques(confirm: bool = False):
    """
    Delete all critiques for the current compiler paper.
    
    Args:
        confirm: Must be True to confirm deletion
    
    Returns:
        Success status
    """
    from backend.shared.critique_memory import clear_critiques
    
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion with confirm=true"
            )
        
        await clear_critiques("compiler_paper")
        
        return {
            "success": True,
            "message": "Compiler paper critiques cleared"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete compiler paper critiques: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/default-critique-prompt")
async def get_compiler_default_critique_prompt():
    """
    Get the default critique prompt text.
    
    Returns:
        The default critique prompt that can be customized by users.
    """
    from backend.shared.critique_prompts import DEFAULT_CRITIQUE_PROMPT
    
    return {
        "success": True,
        "prompt": DEFAULT_CRITIQUE_PROMPT
    }


# =============================================================================
# WOLFRAM ALPHA ENDPOINTS
# =============================================================================

@router.post("/wolfram/set-api-key")
async def set_wolfram_api_key(request: dict):
    """
    Set and validate Wolfram Alpha API key.
    
    Args:
        request: {"api_key": str}
    
    Returns:
        Success status and validation result
    """
    from backend.shared.secret_store import SecretStoreError, store_wolfram_api_key
    from backend.shared.wolfram_alpha_client import initialize_wolfram_client, get_wolfram_client
    
    try:
        api_key = request.get("api_key", "").strip()
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize client
        initialize_wolfram_client(api_key)
        
        # Test connection with simple query
        client = get_wolfram_client()
        test_result = await client.query("What is 2+2?")
        
        if test_result is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to connect to Wolfram Alpha - invalid API key or network error"
            )
        
        # Store in system config
        system_config.wolfram_alpha_api_key = api_key
        system_config.wolfram_alpha_enabled = True

        if system_config.generic_mode:
            logger.info("Generic mode active - keeping Wolfram Alpha API key in runtime memory only")
            success_message = "Wolfram Alpha API key validated and loaded into runtime memory"
        else:
            # Persist to secure backend storage so the key survives restarts.
            store_wolfram_api_key(api_key)
            success_message = "Wolfram Alpha API key validated successfully"
        
        logger.info("Wolfram Alpha API key set and validated")
        
        return {
            "success": True,
            "message": success_message,
            "test_result": test_result
        }
        
    except SecretStoreError as e:
        logger.error(f"Failed to persist Wolfram Alpha API key securely: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set Wolfram Alpha API key: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/wolfram/api-key")
async def clear_wolfram_api_key():
    """
    Clear Wolfram Alpha API key.
    
    Returns:
        Success status
    """
    from backend.shared.secret_store import SecretStoreError, clear_wolfram_api_key as clear_persisted_wolfram_api_key
    from backend.shared.wolfram_alpha_client import clear_wolfram_client
    
    try:
        # Clear client
        clear_wolfram_client()
        
        # Clear from config
        system_config.wolfram_alpha_api_key = None
        system_config.wolfram_alpha_enabled = False

        if system_config.generic_mode:
            logger.info("Generic mode active - cleared in-memory Wolfram Alpha API key")
            success_message = "Wolfram Alpha API key cleared from runtime memory"
        else:
            clear_persisted_wolfram_api_key()
            success_message = "Wolfram Alpha API key cleared"
        
        logger.info("Wolfram Alpha API key cleared")
        
        return {
            "success": True,
            "message": success_message
        }
        
    except SecretStoreError as e:
        logger.error(f"Failed to clear Wolfram Alpha API key from secure storage: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logger.error(f"Failed to clear Wolfram Alpha API key: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/wolfram/status")
async def get_wolfram_status():
    """
    Get Wolfram Alpha configuration status.
    
    Returns:
        enabled: bool, has_key: bool
    """
    return {
        "enabled": system_config.wolfram_alpha_enabled,
        "has_key": system_config.wolfram_alpha_api_key is not None
    }


@router.post("/wolfram/test-query")
async def test_wolfram_query(request: dict):
    """
    Test Wolfram Alpha query without saving API key.
    
    Args:
        request: {"query": str, "api_key": str}
    
    Returns:
        Query result or error
    """
    from backend.shared.wolfram_alpha_client import WolframAlphaClient
    
    try:
        query = request.get("query", "").strip()
        api_key = request.get("api_key", "").strip()
        
        if not query or not api_key:
            raise HTTPException(status_code=400, detail="Both query and api_key are required")
        
        # Create temporary client (don't initialize singleton)
        temp_client = WolframAlphaClient(api_key)
        
        try:
            result = await temp_client.query(query)
            
            if result is None:
                return {
                    "success": False,
                    "message": "Query failed - check API key and query format",
                    "result": None
                }
            
            return {
                "success": True,
                "message": "Query successful",
                "result": result
            }
            
        finally:
            await temp_client.close()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test Wolfram Alpha query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

