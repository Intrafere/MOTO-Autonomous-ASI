"""
Compiler API routes.
"""
from fastapi import APIRouter, HTTPException
import logging
from pathlib import Path
import aiofiles

from backend.shared.models import CompilerStartRequest, CompilerState, CritiqueRequest
from backend.shared.config import system_config
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import paper_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compiler", tags=["compiler"])


@router.post("/start")
async def start_compiler(request: CompilerStartRequest):
    """Start the compiler system."""
    try:
        # Update system config with user-provided context sizes
        system_config.compiler_validator_context_window = request.validator_context_size
        system_config.compiler_high_context_context_window = request.high_context_context_size
        system_config.compiler_high_param_context_window = request.high_param_context_size
        system_config.compiler_critique_submitter_context_window = request.critique_submitter_context_window
        
        # Update max output token configurations
        system_config.compiler_validator_max_output_tokens = request.validator_max_output_tokens
        system_config.compiler_high_context_max_output_tokens = request.high_context_max_output_tokens
        system_config.compiler_high_param_max_output_tokens = request.high_param_max_output_tokens
        system_config.compiler_critique_submitter_max_tokens = request.critique_submitter_max_tokens
        
        # Store critique submitter model
        system_config.compiler_critique_submitter_model = request.critique_submitter_model
        
        logger.info(
            f"Compiler max output tokens - "
            f"Validator: {request.validator_max_output_tokens}, "
            f"High-context: {request.high_context_max_output_tokens}, "
            f"High-param: {request.high_param_max_output_tokens}"
        )
        
        # Initialize coordinator with OpenRouter provider configurations
        await compiler_coordinator.initialize(
            compiler_prompt=request.compiler_prompt,
            validator_model=request.validator_model,
            high_context_model=request.high_context_model,
            high_param_model=request.high_param_model,
            critique_submitter_model=request.critique_submitter_model,
            # OpenRouter provider configs for each role
            validator_provider=request.validator_provider,
            validator_openrouter_provider=request.validator_openrouter_provider,
            validator_lm_studio_fallback=request.validator_lm_studio_fallback,
            high_context_provider=request.high_context_provider,
            high_context_openrouter_provider=request.high_context_openrouter_provider,
            high_context_lm_studio_fallback=request.high_context_lm_studio_fallback,
            high_param_provider=request.high_param_provider,
            high_param_openrouter_provider=request.high_param_openrouter_provider,
            high_param_lm_studio_fallback=request.high_param_lm_studio_fallback,
            critique_submitter_provider=request.critique_submitter_provider,
            critique_submitter_openrouter_provider=request.critique_submitter_openrouter_provider,
            critique_submitter_lm_studio_fallback=request.critique_submitter_lm_studio_fallback
        )
        
        # Start coordinator
        await compiler_coordinator.start()
        
        return {"status": "started", "message": "Compiler started successfully"}
    
    except ValueError as e:
        # Model compatibility errors - provide structured error response
        error_msg = str(e)
        logger.error(f"Model compatibility error: {e}", exc_info=True)
        
        # Determine which model failed
        failed_model_type = "unknown"
        failed_model_name = ""
        
        if request.validator_model in error_msg:
            failed_model_type = "validator"
            failed_model_name = request.validator_model
        elif request.high_context_model in error_msg:
            failed_model_type = "high_context"
            failed_model_name = request.high_context_model
        elif request.high_param_model in error_msg:
            failed_model_type = "high_param"
            failed_model_name = request.high_param_model
        
        # Extract reason from error message
        reason = error_msg
        if "Model incompatibility detected:" in error_msg:
            reason = error_msg.split("Model incompatibility detected:")[1].split(".")[0].strip()
        
        error_response = {
            "error": "model_compatibility",
            "failed_model_type": failed_model_type,
            "failed_model_name": failed_model_name,
            "reason": reason,
            "suggestion": "Try using 'openai/gpt-oss-20b' or 'openai/gpt-oss-20b:3' which are known to work. You can also click 'Use Aggregator Models' to auto-fill working models.",
            "full_error": error_msg
        }
        
        raise HTTPException(status_code=400, detail=error_response)
    
    except Exception as e:
        # Other errors
        logger.error(f"Failed to start compiler: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_compiler():
    """Stop the compiler system."""
    try:
        await compiler_coordinator.stop()
        return {"status": "stopped", "message": "Compiler stopped"}
    except Exception as e:
        logger.error(f"Failed to stop compiler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/skip-critique")
async def skip_critique():
    """Skip the critique phase (immediately or pre-emptively)."""
    try:
        if not compiler_coordinator.is_running:
            raise HTTPException(status_code=400, detail="Compiler is not running")
        
        was_in_critique = compiler_coordinator.in_critique_phase
        success = await compiler_coordinator.skip_critique_phase()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to skip critique phase")
        
        if was_in_critique:
            message = "Critique phase skipped, continuing to conclusion"
        else:
            message = "Critique skip queued - will skip when critique phase is reached"
        
        return {
            "success": True,
            "message": message,
            "was_immediate": was_in_critique
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to skip critique: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-models")
async def test_models(request: CompilerStartRequest):
    """Test model compatibility without starting the compiler."""
    from backend.shared.lm_studio_client import lm_studio_client
    
    results = {
        "validator": {"model": request.validator_model, "passed": False, "error": "", "details": {}},
        "high_context": {"model": request.high_context_model, "passed": False, "error": "", "details": {}},
        "high_param": {"model": request.high_param_model, "passed": False, "error": "", "details": {}}
    }
    
    # Test validator model
    is_compat, error, details = await lm_studio_client.test_model_compatibility(request.validator_model)
    results["validator"]["passed"] = is_compat
    results["validator"]["error"] = error
    results["validator"]["details"] = details
    
    # Test high-context model
    is_compat, error, details = await lm_studio_client.test_model_compatibility(request.high_context_model)
    results["high_context"]["passed"] = is_compat
    results["high_context"]["error"] = error
    results["high_context"]["details"] = details
    
    # Test high-param model
    is_compat, error, details = await lm_studio_client.test_model_compatibility(request.high_param_model)
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
        status = await compiler_coordinator.get_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-paper")
async def save_paper():
    """Save paper to a .txt file (includes author attribution, outline, and paper content)."""
    try:
        outline = await outline_memory.get_outline()
        paper = await paper_memory.get_paper()
        word_count = await paper_memory.get_word_count()
        
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
            from datetime import datetime
            
            # Parse generation date if available
            gen_date = None
            if model_data.get("generation_date"):
                try:
                    gen_date = datetime.fromisoformat(model_data["generation_date"])
                except:
                    pass
            
            # Generate attribution header (no reference papers for manual mode)
            attribution_section = generate_attribution_for_existing_paper(
                user_prompt=compiler_coordinator.user_prompt,
                paper_title=compiler_coordinator.paper_title or compiler_coordinator.user_prompt,
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
        output_path = Path("backend/data/compiler_paper_saved.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(full_content)
        
        return {
            "status": "saved",
            "path": str(output_path),
            "word_count": word_count,
            "message": f"Paper saved to {output_path} ({word_count} words)",
            "has_attribution": bool(attribution_section)
        }
    except Exception as e:
        logger.error(f"Failed to save paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            "miniscule_edit_count": status.miniscule_edit_count,
            "paper_word_count": status.paper_word_count
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        await compiler_coordinator.clear_paper()
        
        # Also clear any paper critiques
        from backend.shared.critique_memory import clear_critiques
        try:
            await clear_critiques("compiler_paper")
            logger.info("Cleared compiler paper critiques")
        except Exception as e:
            logger.warning(f"Failed to clear compiler critiques: {e}")
        
        return {
            "status": "cleared",
            "message": "Paper and outline cleared - system reset to fresh start"
        }
    except Exception as e:
        logger.error(f"Failed to clear paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/critique-status")
async def get_critique_status():
    """Get critique phase status."""
    try:
        return {
            "in_critique_phase": compiler_coordinator.in_critique_phase,
            "critique_acceptances": compiler_coordinator.critique_acceptances,
            "paper_version": compiler_coordinator.paper_version,
            "target_critiques": 10
        }
    except Exception as e:
        logger.error(f"Failed to get critique status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/previous-versions")
async def get_previous_versions():
    """Get all previous body versions."""
    try:
        versions = await paper_memory.get_previous_versions()
        return {"previous_versions": versions}
    except Exception as e:
        logger.error(f"Failed to get previous versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    from backend.shared.api_client_manager import api_client_manager
    from backend.shared.json_parser import parse_json
    from backend.shared.utils import count_tokens
    import uuid
    from datetime import datetime
    
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
        
        # If validator config not provided in request, fall back to system config
        if not validator_model:
            validator_model = system_config.compiler_validator_model
            validator_context_window = system_config.compiler_validator_context_window
            validator_max_tokens = system_config.compiler_validator_max_output_tokens
            validator_provider = getattr(system_config, 'compiler_validator_provider', 'lm_studio')
            validator_openrouter_provider = getattr(system_config, 'compiler_validator_openrouter_provider', None)
        
        if not validator_model:
            raise HTTPException(
                status_code=400,
                detail="No validator model configured. Please configure a validator model in Compiler Settings."
            )
        
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
        from backend.shared.models import ModelConfig
        
        api_client_manager.configure_role(
            "paper_critic",
            ModelConfig(
                provider=validator_provider,
                model_id=validator_model,
                openrouter_model_id=validator_model if validator_provider == "openrouter" else None,
                openrouter_provider=validator_openrouter_provider,
                lm_studio_fallback_id=None,  # No fallback for direct critique calls
                context_window=validator_context_window,
                max_output_tokens=validator_max_tokens
            )
        )
        
        # Make the API call to the validator model
        logger.info(f"Requesting critique for compiler paper from validator model {validator_model}")
        
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
            response_content = message.get("content", "") or message.get("reasoning", "")
        
        if not response_content:
            raise HTTPException(status_code=500, detail="Empty response from validator model")
        
        # Try to parse as JSON
        try:
            critique_data = parse_json(response_content)
        except Exception as e:
            # If JSON parsing fails, create a structured response from raw text
            logger.warning(f"Failed to parse critique JSON, using raw response: {e}")
            critique_data = {
                "novelty_rating": 0,
                "novelty_feedback": "Unable to parse structured response",
                "correctness_rating": 0,
                "correctness_feedback": "Unable to parse structured response",
                "impact_rating": 0,
                "impact_feedback": "Unable to parse structured response",
                "full_critique": response_content
            }
        
        # Create critique object
        critique = PaperCritique(
            critique_id=str(uuid.uuid4()),
            model_id=validator_model,
            provider=validator_provider,
            host_provider=validator_openrouter_provider,
            date=datetime.now(),
            prompt_used=prompt_to_use,
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        
        logger.info("Wolfram Alpha API key set and validated")
        
        return {
            "success": True,
            "message": "Wolfram Alpha API key validated successfully",
            "test_result": test_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set Wolfram Alpha API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/wolfram/api-key")
async def clear_wolfram_api_key():
    """
    Clear Wolfram Alpha API key.
    
    Returns:
        Success status
    """
    from backend.shared.wolfram_alpha_client import clear_wolfram_client
    
    try:
        # Clear client
        clear_wolfram_client()
        
        # Clear from config
        system_config.wolfram_alpha_api_key = None
        system_config.wolfram_alpha_enabled = False
        
        logger.info("Wolfram Alpha API key cleared")
        
        return {
            "success": True,
            "message": "Wolfram Alpha API key cleared"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear Wolfram Alpha API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))

