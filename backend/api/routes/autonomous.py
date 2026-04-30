"""
Autonomous Research API Routes - REST endpoints for autonomous research mode.
Includes Tier 1 (Brainstorm), Tier 2 (Paper Writing), and Tier 3 (Final Answer) endpoints.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.shared.models import AutonomousResearchStartRequest, CritiqueRequest
from backend.shared.path_safety import (
    resolve_path_within_root,
    validate_single_path_component,
)
from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
from backend.autonomous.memory.research_metadata import research_metadata, ResearchMetadata
from backend.autonomous.memory.brainstorm_memory import brainstorm_memory, BrainstormMemory
from backend.autonomous.memory.paper_library import paper_library, PaperLibrary
from backend.autonomous.memory.final_answer_memory import final_answer_memory, FinalAnswerMemory
from backend.autonomous.memory.session_manager import session_manager
from backend.autonomous.memory.autonomous_api_logger import autonomous_api_logger
from backend.aggregator.core.coordinator import coordinator
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.shared.boost_logger import boost_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auto-research", tags=["autonomous"])


def _get_active_autonomous_session_id() -> str:
    """Return the active autonomous session identifier, falling back to legacy mode."""
    return session_manager.session_id if session_manager.is_session_active else "legacy"


def _validate_history_session_id(session_id: str) -> None:
    """Reject malformed history session identifiers before building any filesystem paths."""
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")


def _parse_api_log_timestamp(timestamp: Optional[str]) -> datetime:
    """Parse log timestamps for sorting and deduplication."""
    if not timestamp:
        return datetime.min

    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        return datetime.min


def _normalize_autonomous_api_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize autonomous log entries into the combined API log shape."""
    return {
        **entry,
        "source": "api",
        "boosted": bool(entry.get("boosted", False)),
        "boost_mode": entry.get("boost_mode"),
        "provider": entry.get("provider") or "unknown",
        "phase": entry.get("phase") or "unknown",
        "prompt_preview": entry.get("prompt_preview") or "",
        "prompt_full": entry.get("prompt_full") or entry.get("prompt_preview") or "",
        "response_preview": entry.get("response_preview") or "",
        "response_full": entry.get("response_full") or entry.get("response_preview") or "",
    }


def _normalize_boost_api_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize boost log entries so they can be shown in the main API log view."""
    prompt_preview = entry.get("prompt_preview") or ""
    response_preview = entry.get("response_preview") or ""
    response_full = entry.get("response_full") or response_preview

    return {
        **entry,
        "source": "boost",
        "boosted": True,
        "provider": entry.get("provider") or "openrouter",
        "phase": entry.get("phase") or "boost",
        "prompt_preview": prompt_preview,
        "prompt_full": entry.get("prompt_full") or prompt_preview,
        "response_preview": response_preview,
        "response_full": response_full,
    }


def _build_api_log_match_key(entry: Dict[str, Any]) -> tuple:
    """Build a stable key used to deduplicate boost entries already logged elsewhere."""
    return (
        entry.get("task_id") or "",
        entry.get("role_id") or "",
        entry.get("model") or "",
        bool(entry.get("success", True)),
    )


def _merge_combined_api_logs(
    autonomous_logs: List[Dict[str, Any]],
    boost_logs: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Combine autonomous and boost logs into one list.

    Autonomous boosted calls are often already mirrored into the autonomous logger,
    so boost entries are merged into matching autonomous entries when possible.
    """
    combined_logs = [_normalize_autonomous_api_log(entry) for entry in autonomous_logs]
    candidate_map: Dict[tuple, List[int]] = {}

    for index, entry in enumerate(combined_logs):
        candidate_map.setdefault(_build_api_log_match_key(entry), []).append(index)

    for raw_boost_entry in boost_logs:
        boost_entry = _normalize_boost_api_log(raw_boost_entry)
        boost_timestamp = _parse_api_log_timestamp(boost_entry.get("timestamp"))
        matching_indices = candidate_map.get(_build_api_log_match_key(boost_entry), [])
        merged = False

        for index in matching_indices:
            existing_entry = combined_logs[index]
            if existing_entry.get("_boost_merged"):
                continue

            existing_timestamp = _parse_api_log_timestamp(existing_entry.get("timestamp"))
            if abs((existing_timestamp - boost_timestamp).total_seconds()) > 2:
                continue

            existing_entry["source"] = "api+boost"
            existing_entry["boosted"] = True
            existing_entry["boost_mode"] = boost_entry.get("boost_mode")
            existing_entry["_boost_merged"] = True

            if not existing_entry.get("prompt_full"):
                existing_entry["prompt_full"] = boost_entry.get("prompt_full") or ""
            if not existing_entry.get("prompt_preview"):
                existing_entry["prompt_preview"] = boost_entry.get("prompt_preview") or ""
            if not existing_entry.get("response_full"):
                existing_entry["response_full"] = boost_entry.get("response_full") or ""
            if not existing_entry.get("response_preview"):
                existing_entry["response_preview"] = boost_entry.get("response_preview") or ""

            merged = True
            break

        if not merged:
            combined_logs.append(boost_entry)

    combined_logs.sort(
        key=lambda entry: _parse_api_log_timestamp(entry.get("timestamp")),
        reverse=True,
    )

    for entry in combined_logs:
        entry.pop("_boost_merged", None)

    return combined_logs[:limit]


def _build_combined_api_stats(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build summary stats for the merged API log view."""
    if not logs:
        return {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "success_rate": 0.0,
            "boosted_calls": 0,
            "by_phase": {},
            "by_model": {},
            "by_provider": {},
            "by_source": {},
            "by_boost_mode": {},
        }

    successful_calls = sum(1 for log in logs if log.get("success", True))
    boosted_calls = sum(1 for log in logs if log.get("boosted"))
    by_phase: Dict[str, int] = {}
    by_model: Dict[str, int] = {}
    by_provider: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_boost_mode: Dict[str, int] = {}

    for log in logs:
        phase = log.get("phase") or "unknown"
        model = log.get("model") or "unknown"
        provider = log.get("provider") or "unknown"
        source = log.get("source") or "unknown"

        by_phase[phase] = by_phase.get(phase, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        by_provider[provider] = by_provider.get(provider, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1

        if log.get("boosted"):
            boost_mode = log.get("boost_mode") or "unknown"
            by_boost_mode[boost_mode] = by_boost_mode.get(boost_mode, 0) + 1

    total_calls = len(logs)

    return {
        "total_calls": total_calls,
        "successful_calls": successful_calls,
        "failed_calls": total_calls - successful_calls,
        "success_rate": successful_calls / total_calls if total_calls else 0.0,
        "boosted_calls": boosted_calls,
        "by_phase": by_phase,
        "by_model": by_model,
        "by_provider": by_provider,
        "by_source": by_source,
        "by_boost_mode": by_boost_mode,
    }


async def _get_combined_api_logs(limit: int = 100) -> Dict[str, Any]:
    """Fetch, deduplicate, and summarize the combined autonomous + boost API logs."""
    fetch_limit = max(limit * 3, 300)
    autonomous_logs = await autonomous_api_logger.get_logs(limit=fetch_limit)
    boost_logs = await boost_logger.get_logs(limit=fetch_limit)
    combined_logs = _merge_combined_api_logs(autonomous_logs, boost_logs, limit=limit)
    combined_stats = _build_combined_api_stats(
        _merge_combined_api_logs(
            autonomous_logs,
            boost_logs,
            limit=max(fetch_limit, len(autonomous_logs) + len(boost_logs)),
        )
    )
    return {"logs": combined_logs, "stats": combined_stats}

    if session_id == "legacy":
        return

    try:
        validate_single_path_component(session_id, "session ID")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid session ID: {session_id}")


def _get_start_conflict() -> Optional[str]:
    """Return a user-facing conflict message if another workflow is active."""
    autonomous_state = autonomous_coordinator.get_state()
    if autonomous_state.is_running:
        return "Autonomous research is already running"

    if coordinator.is_running:
        return "Cannot start Autonomous Research while Aggregator is running. Stop Aggregator first."

    if compiler_coordinator.is_running:
        return "Cannot start Autonomous Research while Compiler is running. Stop Compiler first."

    return None


def _resolve_history_session_paths(session_id: str) -> Dict[str, Path]:
    """Resolve all session-specific paths needed for Stage 2 paper history operations."""
    from backend.shared.config import system_config

    _validate_history_session_id(session_id)

    if session_id == "legacy":
        paths = {
            "papers_dir": Path(system_config.auto_papers_dir),
            "brainstorms_dir": Path(system_config.auto_brainstorms_dir),
            "metadata_path": Path(system_config.auto_research_metadata_file),
            "stats_path": Path(system_config.auto_research_stats_file),
            "workflow_state_path": Path(system_config.auto_workflow_state_file),
        }
    else:
        try:
            session_root = resolve_path_within_root(
                Path(system_config.auto_sessions_base_dir),
                validate_single_path_component(session_id, "session ID"),
            )
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid session ID: {session_id}")

        if not session_root.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        paths = {
            "papers_dir": session_root / "papers",
            "brainstorms_dir": session_root / "brainstorms",
            "metadata_path": session_root / "session_metadata.json",
            "stats_path": session_root / "session_stats.json",
            "workflow_state_path": session_root / "workflow_state.json",
        }

    if not paths["papers_dir"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"No Stage 2 papers directory found for session: {session_id}"
        )

    return paths


def _resolve_final_answer_dir(answer_id: str) -> Path:
    """Resolve a legacy or session-based final answer directory safely."""
    from backend.shared.config import system_config

    if answer_id == "legacy":
        base_dir = Path(system_config.data_dir) / "auto_final_answer"
    else:
        try:
            session_dir = resolve_path_within_root(
                Path(system_config.auto_sessions_base_dir),
                validate_single_path_component(answer_id, "final answer ID"),
            )
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid final answer ID: {answer_id}")

        base_dir = session_dir / "final_answer"

    if not base_dir.exists():
        raise HTTPException(status_code=404, detail=f"Final answer not found: {answer_id}")

    return base_dir


def _build_scoped_final_answer_memory(answer_id: str) -> FinalAnswerMemory:
    """Create a temporary FinalAnswerMemory rooted at one validated answer directory."""
    return FinalAnswerMemory.build_scoped_memory(_resolve_final_answer_dir(answer_id))


def _build_scoped_paper_library(paths: Dict[str, Path]) -> PaperLibrary:
    """Create a temporary PaperLibrary rooted at one legacy/session papers directory."""
    scoped_library = PaperLibrary()
    scoped_library._base_dir = paths["papers_dir"]
    scoped_library._archive_dir = paths["papers_dir"] / "archive"
    return scoped_library


def _build_scoped_brainstorm_memory(paths: Dict[str, Path]) -> BrainstormMemory:
    """Create a temporary BrainstormMemory rooted at one legacy/session brainstorms directory."""
    scoped_memory = BrainstormMemory()
    scoped_memory._base_dir = paths["brainstorms_dir"]
    return scoped_memory


async def _ensure_history_paper_is_visible(
    scoped_paper_library: PaperLibrary,
    *,
    session_id: str,
    paper_id: str,
) -> Any:
    """Ensure a history paper matches the completed/non-archived contract of the history UI."""
    metadata = await scoped_paper_library.get_metadata(paper_id)
    if not metadata or metadata.status != "complete":
        raise HTTPException(
            status_code=404,
            detail=f"Paper not found in history: session={session_id}, paper={paper_id}"
        )

    if not await scoped_paper_library.is_paper_complete(paper_id):
        raise HTTPException(
            status_code=404,
            detail=f"Paper is not available in history: session={session_id}, paper={paper_id}"
        )

    return metadata


async def _build_scoped_research_metadata(paths: Dict[str, Path]) -> ResearchMetadata:
    """Create a temporary ResearchMetadata instance rooted at one legacy/session metadata set."""
    scoped_metadata = ResearchMetadata()
    scoped_metadata._metadata_path = paths["metadata_path"]
    scoped_metadata._stats_path = paths["stats_path"]
    scoped_metadata._workflow_state_path = paths["workflow_state_path"]
    await scoped_metadata.initialize()
    return scoped_metadata


def _resolve_validator_config(request: Optional[CritiqueRequest]) -> Dict[str, Any]:
    """Resolve critique validator settings from the request or the active coordinator."""
    validator_model = None
    validator_context_window = None
    validator_max_tokens = None
    validator_provider = None
    validator_openrouter_provider = None
    custom_prompt = None

    if request:
        custom_prompt = request.custom_prompt
        if request.validator_model:
            validator_model = request.validator_model
            validator_context_window = request.validator_context_window or 131072
            validator_max_tokens = request.validator_max_tokens or 25000
            validator_provider = request.validator_provider or "lm_studio"
            validator_openrouter_provider = request.validator_openrouter_provider

    if not validator_model:
        coordinator_config = autonomous_coordinator.get_validator_config()
        if coordinator_config:
            validator_model = coordinator_config["validator_model"]
            validator_context_window = coordinator_config["validator_context_window"]
            validator_max_tokens = coordinator_config["validator_max_tokens"]
            validator_provider = coordinator_config["validator_provider"]
            validator_openrouter_provider = coordinator_config.get("validator_openrouter_provider")

    if not validator_model:
        raise HTTPException(
            status_code=400,
            detail="No validator model configured. Please configure a validator model in Autonomous Research Settings."
        )

    return {
        "custom_prompt": custom_prompt,
        "validator_model": validator_model,
        "validator_context_window": validator_context_window,
        "validator_max_tokens": validator_max_tokens,
        "validator_provider": validator_provider,
        "validator_openrouter_provider": validator_openrouter_provider,
    }


async def _generate_autonomous_paper_critique(
    *,
    paper_id: str,
    paper_title: str,
    content: str,
    base_dir: Path,
    request: Optional[CritiqueRequest] = None,
) -> Dict[str, Any]:
    """Generate and persist a critique for an autonomous Stage 2 paper."""
    from backend.shared.critique_memory import save_critique
    from backend.shared.critique_prompts import (
        DEFAULT_CRITIQUE_PROMPT,
        build_critique_prompt,
        parse_critique_response,
    )
    from backend.shared.api_client_manager import api_client_manager
    from backend.shared.models import ModelConfig, PaperCritique
    from backend.shared.utils import count_tokens
    from datetime import datetime
    import uuid

    config = _resolve_validator_config(request)
    prompt_to_use = config["custom_prompt"] or DEFAULT_CRITIQUE_PROMPT
    full_prompt = build_critique_prompt(content, paper_title, prompt_to_use)
    prompt_tokens = count_tokens(full_prompt)

    output_reserve = config["validator_max_tokens"]
    safety_margin = int(config["validator_context_window"] * 0.1)
    available_input = config["validator_context_window"] - output_reserve - safety_margin

    if prompt_tokens > available_input:
        excess_tokens = prompt_tokens - available_input
        raise HTTPException(
            status_code=400,
            detail=(
                f"Paper is too long for the validator's context window. "
                f"The paper requires {prompt_tokens:,} tokens, but the validator can only accept {available_input:,} tokens "
                f"(context window: {config['validator_context_window']:,}, output reserve: {output_reserve:,}, safety margin: {safety_margin:,}). "
                f"The paper exceeds the limit by {excess_tokens:,} tokens. "
                f"A complete and honest review requires direct context injection - please select a validator with a larger context window."
            )
        )

    api_client_manager.configure_role(
        "paper_critic",
        ModelConfig(
            provider=config["validator_provider"],
            model_id=config["validator_model"],
            openrouter_model_id=config["validator_model"] if config["validator_provider"] == "openrouter" else None,
            openrouter_provider=config["validator_openrouter_provider"],
            lm_studio_fallback_id=None,
            context_window=config["validator_context_window"],
            max_output_tokens=config["validator_max_tokens"],
        )
    )

    logger.info(f"Requesting critique for paper {paper_id} from validator model {config['validator_model']}")

    response = await api_client_manager.generate_completion(
        task_id=f"paper_critique_{paper_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        role_id="paper_critic",
        model=config["validator_model"],
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=config["validator_max_tokens"],
        temperature=0.0,
    )

    response_content = ""
    if response.get("choices"):
        message = response["choices"][0].get("message", {})
        response_content = message.get("content") or message.get("reasoning") or ""

    if not response_content:
        raise HTTPException(status_code=500, detail="Empty response from validator model")

    critique_data = parse_critique_response(response_content)
    critique = PaperCritique(
        critique_id=str(uuid.uuid4()),
        model_id=config["validator_model"],
        provider=config["validator_provider"],
        host_provider=config["validator_openrouter_provider"],
        date=datetime.now(),
        prompt_used=prompt_to_use,
        critique_source="user_request",
        novelty_rating=critique_data.get("novelty_rating", 0),
        novelty_feedback=critique_data.get("novelty_feedback", ""),
        correctness_rating=critique_data.get("correctness_rating", 0),
        correctness_feedback=critique_data.get("correctness_feedback", ""),
        impact_rating=critique_data.get("impact_rating", 0),
        impact_feedback=critique_data.get("impact_feedback", ""),
        full_critique=critique_data.get("full_critique", ""),
    )

    saved_critique = await save_critique("autonomous_paper", critique, paper_id, base_dir)
    return {
        "success": True,
        "critique": saved_critique.model_dump(),
        "paper_id": paper_id,
        "paper_title": paper_title,
    }


async def _get_autonomous_paper_critiques_response(
    *,
    paper_id: str,
    paper_title: str,
    base_dir: Path,
) -> Dict[str, Any]:
    """Load critique history for an autonomous Stage 2 paper."""
    from backend.shared.critique_memory import get_critiques

    critiques = await get_critiques("autonomous_paper", paper_id, base_dir)
    return {
        "success": True,
        "paper_id": paper_id,
        "paper_title": paper_title,
        "critiques": [critique.model_dump() for critique in critiques],
        "count": len(critiques),
    }


async def _delete_autonomous_paper_from_scope(
    *,
    session_id: str,
    scoped_paper_library: PaperLibrary,
    scoped_brainstorm_memory: BrainstormMemory,
    scoped_research_metadata: ResearchMetadata,
    paper_id: str,
) -> Dict[str, Any]:
    """Delete a Stage 2 paper and clean its related metadata/critique state."""
    from backend.shared.critique_memory import clear_critiques

    state = autonomous_coordinator.get_state()
    active_session_id = _get_active_autonomous_session_id()
    if (
        state.is_running
        and state.current_tier == "tier2_paper_writing"
        and autonomous_coordinator._current_paper_id == paper_id
        and active_session_id == session_id
    ):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active paper while it's being compiled. Stop autonomous research first."
        )

    metadata = await scoped_paper_library.get_metadata(paper_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

    paper_path = scoped_paper_library.get_paper_path(paper_id)
    base_dir = Path(paper_path).parent
    source_brainstorms = metadata.source_brainstorm_ids or []

    success = await scoped_paper_library.delete_paper(paper_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete paper files for {paper_id}"
        )

    await scoped_research_metadata.delete_paper(paper_id)

    for topic_id in source_brainstorms:
        try:
            await scoped_brainstorm_memory.remove_paper_reference(topic_id, paper_id)
        except Exception as e:
            logger.warning(
                f"Failed to remove paper {paper_id} from brainstorm metadata {topic_id}: {e}"
            )

    try:
        await clear_critiques("autonomous_paper", paper_id, base_dir)
        logger.info(f"Cleared critiques for deleted paper {paper_id}")
    except Exception as e:
        logger.warning(f"Failed to clear critiques for paper {paper_id}: {e}")

    logger.info(
        f"Deleted paper {paper_id} from session {session_id} "
        f"(from brainstorms: {', '.join(source_brainstorms)})"
    )

    return {
        "success": True,
        "message": f"Paper {paper_id} deleted successfully",
        "paper_id": paper_id,
        "session_id": session_id,
        "source_brainstorms": source_brainstorms,
    }


@router.post("/start")
async def start_autonomous_research(
    request: AutonomousResearchStartRequest,
    background_tasks: BackgroundTasks
):
    """Start autonomous research mode."""
    try:
        from backend.shared.config import system_config

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
        
        # Log submitter configurations
        for config in request.submitter_configs:
            label = "(Main Submitter)" if config.submitter_id == 1 else ""
            logger.info(
                f"Brainstorm Submitter {config.submitter_id} {label}: model={config.model_id}, "
                f"context={config.context_window}, max_tokens={config.max_output_tokens}"
            )
        logger.info(
            f"Validator: model={request.validator_model}, "
            f"context={request.validator_context_window}, max_tokens={request.validator_max_tokens}"
        )
        
        # Initialize coordinator
        await autonomous_coordinator.initialize(
            user_research_prompt=request.user_research_prompt,
            submitter_configs=request.submitter_configs,
            validator_model=request.validator_model,
            validator_context_window=request.validator_context_window,
            validator_max_tokens=request.validator_max_tokens,
            high_context_model=request.high_context_model,
            high_context_context_window=request.high_context_context_window,
            high_context_max_tokens=request.high_context_max_tokens,
            high_param_model=request.high_param_model,
            high_param_context_window=request.high_param_context_window,
            high_param_max_tokens=request.high_param_max_tokens,
            critique_submitter_model=request.critique_submitter_model,
            critique_submitter_context_window=request.critique_submitter_context_window,
            critique_submitter_max_tokens=request.critique_submitter_max_tokens,
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
            critique_submitter_lm_studio_fallback=request.critique_submitter_lm_studio_fallback,
            tier3_enabled=request.tier3_enabled
        )
        
        # Start in background
        background_tasks.add_task(autonomous_coordinator.start)
        
        return {
            "success": True,
            "message": f"Autonomous research started with {num_submitters} brainstorm submitters",
            "num_submitters": num_submitters
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(f"Failed to start autonomous research: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to start autonomous research: {e}")


@router.post("/stop")
async def stop_autonomous_research():
    """Stop autonomous research mode gracefully."""
    try:
        state = autonomous_coordinator.get_state()
        if not state.is_running:
            return {
                "success": True,
                "message": "Autonomous research was not running"
            }
        
        await autonomous_coordinator.stop()
        
        # Get final stats
        stats = await research_metadata.get_stats()
        
        return {
            "success": True,
            "message": "Autonomous research stopped",
            "final_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to stop autonomous research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_autonomous_research(confirm: bool = False):
    """Clear all autonomous research data.
    
    Returns success even with non-critical warnings.
    Only fails if critical operations (brainstorms, papers, RAG) fail.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm with confirm=true"
            )
        
        state = autonomous_coordinator.get_state()
        if state.is_running:
            raise HTTPException(
                status_code=400,
                detail="Cannot clear data while autonomous research is running. Please stop research first."
            )
        
        logger.info("Starting autonomous research data clear...")
        
        try:
            await autonomous_coordinator.clear_all_data()
            logger.info("Autonomous research data clear completed successfully")
            
            return {
                "success": True,
                "message": "All autonomous research data cleared successfully"
            }
        
        except RuntimeError as e:
            # RuntimeError from clear_all_data - check if message indicates partial success
            error_msg = str(e)
            if "Failed to clear critical data" in error_msg:
                # Critical operations failed - this is a real failure
                logger.error(f"Critical errors during clear: {error_msg}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to clear critical data (brainstorms/papers/RAG). Error: {error_msg}"
                )
            else:
                # Generic RuntimeError - treat as failure
                logger.error(f"Error during clear: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(f"Failed to clear autonomous research data: {error_details}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to clear autonomous research data: {e}"
        )


@router.get("/status")
async def get_autonomous_status():
    """Get current status and metrics."""
    try:
        state = autonomous_coordinator.get_state()
        stats = await research_metadata.get_stats()
        
        # Get current brainstorm info if available
        current_brainstorm = None
        if stats.get("current_brainstorm_id"):
            metadata = await brainstorm_memory.get_metadata(stats["current_brainstorm_id"])
            if metadata:
                # Get queue size and acceptance/rejection counts from coordinator
                queue_size = 0
                acceptance_count = 0
                rejection_count = 0
                
                # Try to get aggregator queue size
                if autonomous_coordinator._brainstorm_aggregator:
                    from backend.aggregator.core.queue_manager import queue_manager
                    try:
                        queue_size = await queue_manager.size()
                    except Exception:
                        pass
                
                # Get counts from autonomous coordinator internal state
                acceptance_count = autonomous_coordinator._acceptance_count
                rejection_count = autonomous_coordinator._rejection_count
                cleanup_removals = autonomous_coordinator._cleanup_removals
                
                current_brainstorm = {
                    "topic_id": metadata.topic_id,
                    "topic_prompt": metadata.topic_prompt,
                    "submission_count": metadata.submission_count,
                    "queue_size": queue_size,
                    "acceptance_count": acceptance_count,
                    "rejection_count": rejection_count,
                    "cleanup_removals": cleanup_removals  # Actual pruned/cleanup count
                }
        
        # Get current paper info if available
        current_paper = None
        if stats.get("current_paper_id"):
            paper_meta = await paper_library.get_metadata(stats["current_paper_id"])
            if paper_meta:
                current_paper = {
                    "paper_id": paper_meta.paper_id,
                    "title": paper_meta.title
                }
        
        # Get Tier 3 final answer info if available
        tier3_status = None
        tier3_state = final_answer_memory.get_state()
        if tier3_state.is_active or tier3_state.status == "complete":
            tier3_status = {
                "is_active": tier3_state.is_active,
                "status": tier3_state.status,
                "answer_format": tier3_state.answer_format,
                "certainty_level": tier3_state.certainty_assessment.certainty_level if tier3_state.certainty_assessment else None
            }
        
        return {
            "is_running": state.is_running,
            "current_tier": state.current_tier,
            "current_brainstorm": current_brainstorm,
            "current_paper": current_paper,
            "tier3_status": tier3_status,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get autonomous status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brainstorms")
async def get_all_brainstorms():
    """Get list of all brainstorm topics with metadata."""
    try:
        brainstorms = await brainstorm_memory.get_all_brainstorms()
        
        return {
            "brainstorms": [
                {
                    "topic_id": b.topic_id,
                    "topic_prompt": b.topic_prompt,
                    "status": b.status,
                    "submission_count": b.submission_count,
                    "papers_generated": b.papers_generated,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                    "last_activity": b.last_activity.isoformat() if b.last_activity else None
                }
                for b in brainstorms
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get brainstorms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers")
async def get_all_papers():
    """Get list of all completed papers with abstracts and critique ratings."""
    try:
        from backend.shared.critique_memory import get_latest_critique
        from pathlib import Path
        
        papers = await paper_library.get_all_papers()
        
        # Build response with critique ratings
        paper_responses = []
        for p in papers:
            # Get latest critique for this paper
            paper_path = paper_library.get_paper_path(p.paper_id)
            base_dir = None
            if paper_path:
                base_dir = Path(paper_path).parent
            
            latest_critique = await get_latest_critique(
                paper_type="autonomous_paper",
                paper_id=p.paper_id,
                base_dir=base_dir
            )
            
            # Calculate average rating if critique exists
            critique_avg = None
            if latest_critique:
                critique_avg = round(
                    (latest_critique.novelty_rating + 
                     latest_critique.correctness_rating + 
                     latest_critique.impact_rating) / 3.0,
                    1
                )
            
            paper_responses.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "word_count": p.word_count,
                "source_brainstorm_ids": p.source_brainstorm_ids,
                "referenced_papers": p.referenced_papers,
                "status": p.status,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "model_usage": p.model_usage,
                "critique_avg": critique_avg
            })
        
        return {
            "papers": paper_responses
        }
        
    except Exception as e:
        logger.error(f"Failed to get papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brainstorm/{topic_id}")
async def get_brainstorm(topic_id: str):
    """Get specific brainstorm database content."""
    try:
        metadata = await brainstorm_memory.get_metadata(topic_id)
        
        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Brainstorm not found: {topic_id}"
            )
        
        content = await brainstorm_memory.get_database_content(topic_id)
        submissions = await brainstorm_memory.get_submissions_list(topic_id)
        
        return {
            "topic_id": metadata.topic_id,
            "topic_prompt": metadata.topic_prompt,
            "status": metadata.status,
            "submission_count": metadata.submission_count,
            "papers_generated": metadata.papers_generated,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
            "content": content,
            "submissions": submissions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get brainstorm {topic_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper/{paper_id}")
async def get_paper(paper_id: str):
    """Get specific paper content."""
    try:
        metadata = await paper_library.get_metadata(paper_id)
        
        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Paper not found: {paper_id}"
            )
        
        content = await paper_library.get_paper_content(paper_id)
        outline = await paper_library.get_outline(paper_id)
        
        return {
            "paper_id": metadata.paper_id,
            "title": metadata.title,
            "abstract": metadata.abstract,
            "word_count": metadata.word_count,
            "source_brainstorm_ids": metadata.source_brainstorm_ids,
            "referenced_papers": metadata.referenced_papers,
            "status": metadata.status,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
            "model_usage": metadata.model_usage,
            "content": content,
            "outline": outline
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper-history")
async def get_paper_history():
    """Get all completed, non-archived Stage 2 papers from legacy and session history."""
    try:
        papers = await paper_library.list_history_papers()
        return {
            "success": True,
            "papers": papers,
            "total_count": len(papers)
        }
    except Exception as e:
        logger.error(f"Failed to get Stage 2 paper history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper-history/{session_id}/{paper_id}")
async def get_history_paper(session_id: str, paper_id: str):
    """Get one completed, non-archived Stage 2 paper from legacy/session history."""
    try:
        paper = await paper_library.get_history_paper(session_id, paper_id)
        if not paper:
            raise HTTPException(
                status_code=404,
                detail=f"Paper not found in history: session={session_id}, paper={paper_id}"
            )

        return {
            "success": True,
            **paper
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history paper {session_id}/{paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-paper-progress")
async def get_current_paper_progress():
    """Get current paper being compiled (if any).
    
    Works for both Tier 2 (regular paper writing) and Tier 3 (final answer chapters).
    Returns the in-progress compiler memory content regardless of which tier.
    """
    try:
        state = autonomous_coordinator.get_state()
        
        # Check if in paper writing tier (either Tier 2 or Tier 3)
        is_tier2 = state.current_tier == "tier2_paper_writing"
        is_tier3 = state.current_tier == "tier3_final_answer"
        
        if not is_tier2 and not is_tier3:
            return {
                "is_compiling": False,
                "paper_id": None,
                "title": None,
                "content": "",
                "word_count": 0,
                "tier": None
            }
        
        # Get paper content from compiler memory (used by both Tier 2 and Tier 3)
        from backend.compiler.memory.paper_memory import paper_memory as compiler_paper_memory
        from backend.compiler.memory.outline_memory import outline_memory
        
        content = await compiler_paper_memory.get_paper()
        word_count = await compiler_paper_memory.get_word_count()
        outline = await outline_memory.get_outline()
        
        # Build response based on tier
        if is_tier2:
            # Tier 2: Regular paper writing
            current_paper_id = autonomous_coordinator._current_paper_id
            title = autonomous_coordinator._current_paper_title or "Untitled Paper"
            
            return {
                "is_compiling": True,
                "paper_id": current_paper_id,
                "title": title,
                "content": content,
                "outline": outline,
                "word_count": word_count,
                "tier": "tier2"
            }
        else:
            # Tier 3: Final answer chapter writing
            tier3_state = final_answer_memory.get_state()
            
            # Determine what's being written
            chapter_info = None
            title = "Final Answer"
            
            if tier3_state.answer_format == "short_form":
                title = "Final Answer (Short Form)"
            elif tier3_state.answer_format == "long_form" and tier3_state.volume_organization:
                vol = tier3_state.volume_organization
                title = vol.volume_title or "Final Answer Volume"
                
                # Find current chapter being written
                if tier3_state.current_writing_chapter:
                    for ch in vol.chapters:
                        if ch.order == tier3_state.current_writing_chapter:
                            chapter_info = {
                                "order": ch.order,
                                "title": ch.title,
                                "type": ch.chapter_type,
                                "status": ch.status
                            }
                            break
            
            return {
                "is_compiling": True,
                "paper_id": f"tier3_chapter_{tier3_state.current_writing_chapter}" if tier3_state.current_writing_chapter else "tier3",
                "title": title,
                "content": content,
                "outline": outline,
                "word_count": word_count,
                "tier": "tier3",
                "tier3_format": tier3_state.answer_format,
                "tier3_status": tier3_state.status,
                "tier3_chapter": chapter_info
            }
        
    except Exception as e:
        logger.error(f"Failed to get current paper progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get autonomous research statistics."""
    try:
        stats = await research_metadata.get_stats()
        paper_counts = await paper_library.count_papers()
        
        return {
            **stats,
            "paper_counts": paper_counts
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions():
    """
    List all research sessions organized by user prompt.
    Each session contains brainstorms, papers, and final answers.
    """
    try:
        from backend.shared.config import system_config
        
        sessions = await session_manager.list_all_sessions(system_config.auto_sessions_base_dir)
        
        return {
            "sessions": sessions,
            "current_session_id": session_manager.session_id if session_manager.is_session_active else None,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-session")
async def get_current_session():
    """Get information about the current active session."""
    try:
        if not session_manager.is_session_active:
            return {
                "is_active": False,
                "session_id": None,
                "path": None
            }
        
        return {
            "is_active": True,
            "session_id": session_manager.session_id,
            "path": str(session_manager.session_path) if session_manager.session_path else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get current session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/force-paper-writing")
async def force_paper_writing():
    """
    Manual override to force current brainstorm to transition to paper writing.
    Bypasses automatic completion review - user acts as special submitter reviewer.
    """
    try:
        state = autonomous_coordinator.get_state()
        
        # Validate state
        if not state.is_running:
            raise HTTPException(
                status_code=400,
                detail="Autonomous research is not running"
            )
        
        if state.current_tier != "tier1_aggregation":
            raise HTTPException(
                status_code=400,
                detail=f"Can only force paper writing during brainstorm aggregation (tier1). Current tier: {state.current_tier}"
            )
        
        # Get current brainstorm info
        topic_id = autonomous_coordinator._current_topic_id
        if not topic_id:
            raise HTTPException(
                status_code=400,
                detail="No active brainstorm found"
            )
        
        metadata = await brainstorm_memory.get_metadata(topic_id)
        if not metadata:
            raise HTTPException(
                status_code=400,
                detail=f"Brainstorm metadata not found: {topic_id}"
            )
        
        # Log manual override
        logger.info(f"Manual override: Forcing paper writing for brainstorm {topic_id}")
        
        # Trigger manual transition
        success = await autonomous_coordinator.force_paper_writing()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to transition to paper writing"
            )
        
        return {
            "success": True,
            "message": f"Brainstorm {topic_id} will now transition to paper writing",
            "topic_id": topic_id,
            "topic_prompt": metadata.topic_prompt,
            "submission_count": metadata.submission_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to force paper writing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/skip-critique")
async def skip_critique():
    """Skip critique phase during autonomous paper writing (immediately or pre-emptively)."""
    try:
        state = autonomous_coordinator.get_state()
        
        if not state.is_running:
            raise HTTPException(status_code=400, detail="Autonomous research is not running")
        
        if state.current_tier != "tier2_paper_writing":
            raise HTTPException(
                status_code=400,
                detail=f"Not in paper writing tier (current: {state.current_tier})"
            )
        
        success = await autonomous_coordinator.skip_critique_phase()
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="No active compiler found for paper writing"
            )
        
        return {
            "success": True,
            "message": "Critique phase will be skipped (immediately or when reached)"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to skip critique: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-current-paper")
async def reset_current_paper(confirm: bool = False):
    """
    Reset the current paper being written and restart from appropriate phase.
    
    Requires confirmation parameter to prevent accidental resets.
    
    Behavior:
    - Tier 3 Short-Form: Reset to title selection
    - Tier 3 Long-Form: Reset current chapter only
    - Tier 2 (during autonomous): Reset to outline creation
    
    Args:
        confirm: Must be True to proceed with reset (prevents accidental resets)
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Pass confirm=true to reset paper."
        )
    
    if not autonomous_coordinator._running:
        raise HTTPException(
            status_code=400,
            detail="Autonomous research not running"
        )
    
    try:
        result = await autonomous_coordinator.reset_current_paper()
        return {
            "success": True,
            "message": "Paper reset successfully",
            **result
        }
    except Exception as e:
        logger.error(f"Failed to reset paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/force-tier3")
async def force_tier3(mode: str = "complete_current"):
    """
    Force transition to Tier 3 final answer generation.
    
    Modes:
    - complete_current: Finish current brainstorm->paper cycle first, then trigger Tier 3
    - skip_incomplete: Skip incomplete work, proceed to Tier 3 with completed papers only
    
    Returns current state info and mode selected.
    """
    try:
        state = autonomous_coordinator.get_state()
        
        # Validate state
        if not state.is_running:
            raise HTTPException(
                status_code=400,
                detail="Autonomous research is not running"
            )
        
        # Validate mode
        if mode not in ["complete_current", "skip_incomplete"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Must be 'complete_current' or 'skip_incomplete'"
            )
        
        # Check if already in Tier 3
        if state.current_tier == "tier3_final_answer":
            raise HTTPException(
                status_code=400,
                detail="Already in Tier 3 final answer generation"
            )
        
        # Get context info for response
        context_info = {
            "current_tier": state.current_tier,
            "current_topic_id": autonomous_coordinator._current_topic_id,
            "current_paper_id": None,
        }
        
        # Get additional context based on current tier
        if state.current_tier == "tier1_aggregation":
            topic_id = autonomous_coordinator._current_topic_id
            if topic_id:
                metadata = await brainstorm_memory.get_metadata(topic_id)
                if metadata:
                    context_info["brainstorm_submissions"] = metadata.submission_count
                    context_info["brainstorm_prompt"] = metadata.topic_prompt[:100] + "..." if len(metadata.topic_prompt) > 100 else metadata.topic_prompt
        
        elif state.current_tier == "tier2_paper_writing":
            # Get current paper info from compiler if available
            try:
                from backend.compiler.core.compiler_coordinator import compiler_coordinator
                compiler_state = await compiler_coordinator.get_status()
                context_info["compiler_mode"] = compiler_state.current_mode or "unknown"
            except Exception:
                pass
        
        # Get count of completed papers
        all_papers = await paper_library.get_all_papers()
        context_info["completed_papers_count"] = len(all_papers)
        
        # Require at least 1 completed paper for Tier 3
        if len(all_papers) == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot trigger Tier 3 without any completed papers. At least 1 paper is required."
            )
        
        # Log the force action
        logger.info(f"Force Tier 3 requested with mode: {mode}, current tier: {state.current_tier}")
        
        # Trigger the force Tier 3 method on coordinator
        result = await autonomous_coordinator.force_tier3_final_answer(mode)
        
        # Handle the result dict from coordinator
        if not result.get("success", False):
            # Actual failure to initiate/run Tier 3
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to trigger Tier 3 final answer generation")
            )
        
        # Success cases: "initiated", "no_answer_known", or "complete"
        tier3_result = result.get("result", "initiated")
        message = result.get("message", f"Tier 3 final answer generation initiated with mode: {mode}")
        
        return {
            "success": True,
            "result": tier3_result,  # "initiated" | "no_answer_known" | "complete"
            "message": message,
            "mode": mode,
            "context": context_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to force Tier 3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-current-compiler-paper")
async def save_current_compiler_paper():
    """
    Endpoint to save the current compiler paper to the autonomous library.
    Useful for recovering papers that got stuck before abstract was written.
    """
    try:
        import re
        from backend.compiler.memory.paper_memory import paper_memory as compiler_paper_memory
        from backend.compiler.memory.outline_memory import outline_memory as compiler_outline_memory
        from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
        from backend.autonomous.memory.research_metadata import research_metadata
        
        # Get current paper from compiler memory
        current_paper = await compiler_paper_memory.get_paper()
        current_outline = await compiler_outline_memory.get_outline()
        
        if not current_paper:
            raise HTTPException(status_code=404, detail="No paper in compiler memory")
        
        # Generate paper ID
        paper_id = await research_metadata.generate_paper_id()
        
        # Extract title from paper or use default
        title_match = re.search(r"^#\s+(.+)$", current_paper, re.MULTILINE)
        title = title_match.group(1) if title_match else "Recovered Paper - Langlands Correspondence"
        
        # Get brainstorm content (if available)
        topic_id = autonomous_coordinator._current_topic_id if autonomous_coordinator else None
        brainstorm_content = ""
        if topic_id:
            brainstorm_content = await brainstorm_memory.get_database_content(topic_id)
        
        # Save paper
        metadata = await paper_library.save_paper(
            paper_id=paper_id,
            title=title,
            content=current_paper,
            outline=current_outline or "[Outline not available]",
            abstract="[Abstract not completed - paper recovered from compiler memory]",
            source_brainstorm_ids=[topic_id] if topic_id else [],
            source_brainstorm_content=brainstorm_content,
            referenced_papers=[]
        )
        
        # Register in metadata
        await research_metadata.register_paper(metadata)
        
        # Update brainstorm to reference this paper
        if topic_id:
            await brainstorm_memory.add_paper_reference(topic_id, paper_id)
        
        logger.info(f"Paper save successful: {paper_id}")
        
        return {
            "success": True,
            "paper_id": paper_id,
            "title": title,
            "word_count": metadata.word_count,
            "message": "Paper successfully recovered and saved to library"
        }
        
    except Exception as e:
        logger.error(f"Failed to save current compiler paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/brainstorm/{topic_id}")
async def delete_brainstorm(topic_id: str, confirm: bool = False):
    """
    Delete a brainstorm and optionally all its associated papers.
    
    Query params:
        confirm: Must be True to execute deletion (safety check)
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion with confirm=true"
            )
        
        # Check if running
        state = autonomous_coordinator.get_state()
        if state.is_running and state.current_tier == "tier1_aggregation":
            # Check if this is the active brainstorm
            if autonomous_coordinator._current_topic_id == topic_id:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot delete active brainstorm while it's being aggregated. Stop autonomous research first."
                )
        
        # Get brainstorm metadata
        metadata = await brainstorm_memory.get_metadata(topic_id)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Brainstorm not found: {topic_id}"
            )
        
        # Get associated papers
        associated_papers = metadata.papers_generated or []
        
        # Delete brainstorm files
        success = await brainstorm_memory.delete_brainstorm(topic_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete brainstorm files for {topic_id}"
            )
        
        # Remove from central metadata
        await research_metadata.delete_brainstorm(topic_id)
        
        logger.info(f"Deleted brainstorm {topic_id} (had {len(associated_papers)} associated papers)")
        
        return {
            "success": True,
            "message": f"Brainstorm {topic_id} deleted successfully",
            "topic_id": topic_id,
            "associated_papers": associated_papers
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete brainstorm {topic_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/paper/{paper_id}")
async def delete_paper(paper_id: str, confirm: bool = False):
    """
    Delete a paper and optionally its source brainstorm.
    
    Query params:
        confirm: Must be True to execute deletion (safety check)
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion with confirm=true"
            )

        return await _delete_autonomous_paper_from_scope(
            session_id=_get_active_autonomous_session_id(),
            scoped_paper_library=paper_library,
            scoped_brainstorm_memory=brainstorm_memory,
            scoped_research_metadata=research_metadata,
            paper_id=paper_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/paper-history/{session_id}/{paper_id}")
async def delete_history_paper(session_id: str, paper_id: str, confirm: bool = False):
    """Delete a completed Stage 2 history paper from a specific legacy/session scope."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion with confirm=true"
            )

        paths = _resolve_history_session_paths(session_id)
        scoped_paper_library = _build_scoped_paper_library(paths)
        scoped_brainstorm_memory = _build_scoped_brainstorm_memory(paths)
        scoped_research_metadata = await _build_scoped_research_metadata(paths)
        await _ensure_history_paper_is_visible(
            scoped_paper_library,
            session_id=session_id,
            paper_id=paper_id,
        )

        return await _delete_autonomous_paper_from_scope(
            session_id=session_id,
            scoped_paper_library=scoped_paper_library,
            scoped_brainstorm_memory=scoped_brainstorm_memory,
            scoped_research_metadata=scoped_research_metadata,
            paper_id=paper_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete history paper {session_id}/{paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TIER 3: FINAL ANSWER ENDPOINTS
# ============================================================================


@router.get("/tier3/status")
async def get_tier3_status():
    """
    Get current Tier 3 final answer status.
    Returns state, progress, and volume/paper information.
    """
    try:
        # Ensure final answer memory is initialized (loads state from disk)
        await final_answer_memory.initialize()
        state = final_answer_memory.get_state()
        
        # Get volume info if long form
        volume_info = None
        if state.volume_organization:
            vol = state.volume_organization
            volume_info = {
                "title": vol.volume_title,
                "total_chapters": len(vol.chapters),
                "chapters_complete": len([ch for ch in vol.chapters if ch.status == "complete"]),
                "outline_complete": vol.outline_complete,
                "chapters": [
                    {
                        "order": ch.order,
                        "title": ch.title,
                        "type": ch.chapter_type,
                        "status": ch.status,
                        "paper_id": ch.paper_id
                    }
                    for ch in sorted(vol.chapters, key=lambda x: x.order)
                ]
            }
        
        # Get certainty info if available
        certainty_info = None
        if state.certainty_assessment:
            certainty_info = {
                "level": state.certainty_assessment.certainty_level,
                "summary": state.certainty_assessment.known_certainties_summary[:500] + "..." if len(state.certainty_assessment.known_certainties_summary) > 500 else state.certainty_assessment.known_certainties_summary
            }
        
        return {
            "is_active": state.is_active,
            "status": state.status,
            "answer_format": state.answer_format,
            "certainty_assessment": certainty_info,
            "short_form_paper_id": state.short_form_paper_id,
            "short_form_references": state.short_form_reference_papers,
            "volume": volume_info,
            "current_writing_chapter": state.current_writing_chapter,
            "rejections": {
                "assessment": state.tier3_assessment_rejections,
                "format": state.tier3_format_rejections,
                "volume": state.tier3_volume_rejections,
                "writing": state.tier3_writing_rejections
            },
            "timestamp": state.timestamp.isoformat() if state.timestamp else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get Tier 3 status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tier3/final-answer")
async def get_final_answer():
    """
    Get the final answer content.
    Returns either the short form paper or the assembled volume.
    """
    try:
        # Ensure final answer memory is initialized (loads state from disk)
        await final_answer_memory.initialize()
        state = final_answer_memory.get_state()
        
        if not state.is_active and state.status != "complete":
            return {
                "has_final_answer": False,
                "format": None,
                "content": "",
                "message": "No final answer available yet. Tier 3 has not completed."
            }
        
        if state.answer_format == "short_form":
            # Get short form paper
            if state.short_form_paper_id:
                content = await paper_library.get_paper_content(state.short_form_paper_id)
                metadata = await paper_library.get_metadata(state.short_form_paper_id)
                
                return {
                    "has_final_answer": True,
                    "format": "short_form",
                    "paper_id": state.short_form_paper_id,
                    "title": metadata.title if metadata else "Final Answer",
                    "content": content,
                    "word_count": len(content.split()) if content else 0,
                    "status": state.status
                }
            else:
                return {
                    "has_final_answer": False,
                    "format": "short_form",
                    "content": "",
                    "message": "Short form paper not yet written"
                }
        
        elif state.answer_format == "long_form":
            # Get assembled volume
            volume_content = await final_answer_memory.get_final_volume()
            volume_org = state.volume_organization
            
            if volume_content:
                return {
                    "has_final_answer": True,
                    "format": "long_form",
                    "title": volume_org.volume_title if volume_org else "Research Volume",
                    "content": volume_content,
                    "word_count": len(volume_content.split()),
                    "chapters": [
                        {
                            "order": ch.order,
                            "title": ch.title,
                            "type": ch.chapter_type
                        }
                        for ch in sorted(volume_org.chapters, key=lambda x: x.order)
                    ] if volume_org else [],
                    "status": state.status
                }
            else:
                return {
                    "has_final_answer": False,
                    "format": "long_form",
                    "content": "",
                    "message": "Volume not yet assembled",
                    "is_writing": state.current_writing_chapter is not None
                }
        
        return {
            "has_final_answer": False,
            "format": None,
            "content": "",
            "message": "Format not yet selected"
        }
        
    except Exception as e:
        logger.error(f"Failed to get final answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tier3/volume-progress")
async def get_volume_progress():
    """
    Get current volume writing progress (for long form answers).
    Returns detailed status of each chapter.
    """
    try:
        state = final_answer_memory.get_state()
        
        if state.answer_format != "long_form" or not state.volume_organization:
            return {
                "is_long_form": False,
                "volume": None
            }
        
        vol = state.volume_organization
        
        # Get chapter contents
        chapters_data = []
        for ch in sorted(vol.chapters, key=lambda x: x.order):
            chapter_data = {
                "order": ch.order,
                "title": ch.title,
                "type": ch.chapter_type,
                "status": ch.status,
                "paper_id": ch.paper_id,
                "description": ch.description,
                "content_preview": ""
            }
            
            # Get preview of content
            if ch.chapter_type == "existing_paper" and ch.paper_id:
                content = await paper_library.get_paper_content(ch.paper_id)
                chapter_data["content_preview"] = content[:500] + "..." if content and len(content) > 500 else content or ""
            elif ch.status == "complete":
                content = await final_answer_memory.get_chapter_paper(ch.order)
                chapter_data["content_preview"] = content[:500] + "..." if content and len(content) > 500 else content or ""
            
            chapters_data.append(chapter_data)
        
        return {
            "is_long_form": True,
            "volume_title": vol.volume_title,
            "outline_complete": vol.outline_complete,
            "current_writing_chapter": state.current_writing_chapter,
            "total_chapters": len(vol.chapters),
            "completed_chapters": len([ch for ch in vol.chapters if ch.status == "complete"]),
            "chapters": chapters_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get volume progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tier3/rejections")
async def get_tier3_rejections(phase: str = None):
    """
    Get Tier 3 rejection logs.
    
    Query params:
        phase: Optional filter - "assessment", "format", "volume", or "writing"
    """
    try:
        rejections = await final_answer_memory.get_rejections(phase)
        
        return {
            "rejections": rejections,
            "count": len(rejections),
            "phase_filter": phase
        }
        
    except Exception as e:
        logger.error(f"Failed to get Tier 3 rejections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tier3/clear")
async def clear_tier3_data(confirm: bool = False):
    """
    Clear Tier 3 final answer data.
    Does NOT affect Tier 1 brainstorms or Tier 2 papers.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm with confirm=true"
            )
        
        state = autonomous_coordinator.get_state()
        if state.is_running and autonomous_coordinator._tier3_active:
            raise HTTPException(
                status_code=400,
                detail="Cannot clear Tier 3 data while final answer generation is in progress"
            )
        
        await final_answer_memory.clear()
        
        # Also clear any final answer critiques
        from backend.shared.critique_memory import clear_critiques
        try:
            await clear_critiques("final_answer")
            logger.info("Cleared final answer critiques")
        except Exception as e:
            logger.warning(f"Failed to clear final answer critiques: {e}")
        
        return {
            "success": True,
            "message": "Tier 3 final answer data cleared"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear Tier 3 data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FINAL ANSWER LIBRARY - Browse all completed volumes/papers
# ============================================================================

@router.get("/final-answer-library")
async def get_final_answer_library():
    """
    Get a list of ALL completed final answers from all sessions (legacy + session-based).
    
    Returns a library of all finished volumes and papers with metadata:
    - answer_id: Unique identifier
    - format: "short_form" or "long_form"
    - title: Volume/paper title
    - user_prompt: Research question
    - certainty_level: Assessment result
    - word_count: Total words
    - chapter_count: Number of chapters (long form only)
    - completion_date: When it was completed
    - location: Path to the answer
    - session_id: Session identifier
    """
    try:
        await final_answer_memory.initialize()
        final_answers = await final_answer_memory.list_all_final_answers()
        
        return {
            "success": True,
            "final_answers": final_answers,
            "total_count": len(final_answers)
        }
    except Exception as e:
        logger.error(f"Failed to get final answer library: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/final-answer-library/{answer_id}")
async def get_final_answer_by_id(answer_id: str):
    """
    Get full content of a specific final answer by its ID.
    
    Args:
        answer_id: Either "legacy" or a session folder name
    
    Returns:
        - metadata: Title, format, word count, etc.
        - content: Full text of the volume/paper
        - chapters: List of chapter details (long form only)
    """
    try:
        await final_answer_memory.initialize()
        final_answer = await final_answer_memory.get_final_answer_by_id(answer_id)
        
        if not final_answer:
            raise HTTPException(
                status_code=404,
                detail=f"Final answer '{answer_id}' not found"
            )
        
        return {
            "success": True,
            **final_answer
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get final answer {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FINAL ANSWER ARCHIVE ENDPOINTS (Papers & Brainstorms)
# =============================================================================

@router.get("/final-answer/{answer_id}/archive/papers")
async def get_final_answer_archived_papers(answer_id: str):
    """
    Get list of archived papers for a final answer.
    
    Args:
        answer_id: Either "legacy" or session folder name
    
    Returns:
        List of paper metadata
    """
    try:
        memory = _build_scoped_final_answer_memory(answer_id)
        papers = await memory.get_archived_papers_list()
        return {"papers": papers}
    except Exception as e:
        logger.error(f"Failed to get archived papers for {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/final-answer/{answer_id}/archive/papers/{paper_id}")
async def get_final_answer_archived_paper(answer_id: str, paper_id: str):
    """
    Get full archived paper content.
    
    Args:
        answer_id: Either "legacy" or session folder name
        paper_id: Paper ID
    
    Returns:
        Paper content, abstract, outline, metadata
    """
    try:
        memory = _build_scoped_final_answer_memory(answer_id)
        paper = await memory.get_archived_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"Archived paper {paper_id} not found")
        
        return paper
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get archived paper {paper_id} for {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/final-answer/{answer_id}/archive/brainstorms")
async def get_final_answer_archived_brainstorms(answer_id: str):
    """
    Get list of archived brainstorms for a final answer.
    
    Args:
        answer_id: Either "legacy" or session folder name
    
    Returns:
        List of brainstorm metadata
    """
    try:
        memory = _build_scoped_final_answer_memory(answer_id)
        brainstorms = await memory.get_archived_brainstorms_list()
        return {"brainstorms": brainstorms}
    except Exception as e:
        logger.error(f"Failed to get archived brainstorms for {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/final-answer/{answer_id}/archive/brainstorms/{topic_id}")
async def get_final_answer_archived_brainstorm(answer_id: str, topic_id: str):
    """
    Get full archived brainstorm content.
    
    Args:
        answer_id: Either "legacy" or session folder name
        topic_id: Brainstorm topic ID
    
    Returns:
        Brainstorm content and metadata
    """
    try:
        memory = _build_scoped_final_answer_memory(answer_id)
        brainstorm = await memory.get_archived_brainstorm(topic_id)
        if brainstorm is None:
            raise HTTPException(status_code=404, detail=f"Archived brainstorm {topic_id} not found")
        
        return brainstorm
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get archived brainstorm {topic_id} for {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PAPER CRITIQUE ENDPOINTS (Validator Critique Feature)
# ============================================================================


@router.post("/paper/{paper_id}/critique")
async def request_paper_critique(paper_id: str, request: CritiqueRequest = None):
    """
    Request a critique of a paper from the user's validator model.
    
    The paper is direct-injected into the validator model for an honest critique.
    If the paper exceeds the validator's context window, an error is returned.
    
    Validator config can be provided in request body (allows critiques without starting research),
    or falls back to the autonomous coordinator's stored config if research is running.
    
    Args:
        paper_id: The paper ID to critique
        request: Optional request body with custom_prompt and/or validator config
    
    Returns:
        The critique with ratings and feedback
    """
    try:
        metadata = await paper_library.get_metadata(paper_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        content = await paper_library.get_paper_content(paper_id)
        if not content:
            raise HTTPException(status_code=404, detail=f"Paper content not found: {paper_id}")

        paper_path = paper_library.get_paper_path(paper_id)
        base_dir = Path(paper_path).parent

        return await _generate_autonomous_paper_critique(
            paper_id=paper_id,
            paper_title=metadata.title,
            content=content,
            base_dir=base_dir,
            request=request,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to request paper critique for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper/{paper_id}/critiques")
async def get_paper_critiques(paper_id: str):
    """
    Get all critiques for a paper.
    
    Args:
        paper_id: The paper ID
    
    Returns:
        List of critiques for the paper
    """
    try:
        metadata = await paper_library.get_metadata(paper_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        paper_path = paper_library.get_paper_path(paper_id)
        base_dir = Path(paper_path).parent

        return await _get_autonomous_paper_critiques_response(
            paper_id=paper_id,
            paper_title=metadata.title,
            base_dir=base_dir,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get critiques for paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/paper/{paper_id}/critiques")
async def delete_paper_critiques(paper_id: str, confirm: bool = False):
    """
    Delete all critiques for a paper.
    
    Args:
        paper_id: The paper ID
        confirm: Must be True to confirm deletion
    
    Returns:
        Success status
    """
    from backend.shared.critique_memory import clear_critiques
    import os
    
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion with confirm=true"
            )
        
        # Verify paper exists
        metadata = await paper_library.get_metadata(paper_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
        
        base_dir = Path(paper_library.get_paper_path(paper_id)).parent
        
        await clear_critiques("autonomous_paper", paper_id, base_dir)
        
        return {
            "success": True,
            "message": f"Critiques cleared for paper {paper_id}",
            "paper_id": paper_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete critiques for paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STAGE 2 PAPER HISTORY CRITIQUE ENDPOINTS
# ============================================================================


@router.post("/paper-history/{session_id}/{paper_id}/critique")
async def request_history_paper_critique(
    session_id: str,
    paper_id: str,
    request: CritiqueRequest = None,
):
    """Request a validator critique for a Stage 2 history paper from a specific session."""
    try:
        paths = _resolve_history_session_paths(session_id)
        scoped_paper_library = _build_scoped_paper_library(paths)
        metadata = await _ensure_history_paper_is_visible(
            scoped_paper_library,
            session_id=session_id,
            paper_id=paper_id,
        )

        content = await scoped_paper_library.get_paper_content(paper_id)
        if not content:
            raise HTTPException(
                status_code=404,
                detail=f"Paper content not found: session={session_id}, paper={paper_id}"
            )

        return await _generate_autonomous_paper_critique(
            paper_id=paper_id,
            paper_title=metadata.title,
            content=content,
            base_dir=paths["papers_dir"],
            request=request,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to request history critique for {session_id}/{paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper-history/{session_id}/{paper_id}/critiques")
async def get_history_paper_critiques(session_id: str, paper_id: str):
    """Get all validator critiques for a Stage 2 history paper from a specific session."""
    try:
        paths = _resolve_history_session_paths(session_id)
        scoped_paper_library = _build_scoped_paper_library(paths)
        metadata = await _ensure_history_paper_is_visible(
            scoped_paper_library,
            session_id=session_id,
            paper_id=paper_id,
        )

        return await _get_autonomous_paper_critiques_response(
            paper_id=paper_id,
            paper_title=metadata.title,
            base_dir=paths["papers_dir"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history critiques for {session_id}/{paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FINAL ANSWER CRITIQUE ENDPOINTS
# ============================================================================


@router.post("/final-answer-library/{answer_id}/critique")
async def request_final_answer_critique(answer_id: str, request: CritiqueRequest = None):
    """
    Request a critique of a final answer from the user's validator model.
    
    The final answer is direct-injected into the validator model for an honest critique.
    If the content exceeds the validator's context window, an error is returned.
    
    Validator config can be provided in request body (allows critiques without starting research),
    or falls back to the autonomous coordinator's stored config if research is running.
    
    Args:
        answer_id: The final answer ID to critique (either "legacy" or a session folder name)
        request: Optional request body with custom_prompt and/or validator config
    
    Returns:
        The critique with ratings and feedback
    """
    from backend.shared.critique_prompts import build_critique_prompt, DEFAULT_CRITIQUE_PROMPT
    from backend.shared.critique_memory import save_critique
    from backend.shared.models import PaperCritique, CritiqueRequest
    from backend.shared.api_client_manager import api_client_manager
    from backend.shared.utils import count_tokens
    import uuid
    from datetime import datetime
    
    try:
        # Get final answer content
        await final_answer_memory.initialize()
        final_answer = await final_answer_memory.get_final_answer_by_id(answer_id)
        
        if not final_answer:
            raise HTTPException(status_code=404, detail=f"Final answer not found: {answer_id}")
        
        content = final_answer.get("content", "")
        title = final_answer.get("title", "Final Answer")
        
        if not content:
            raise HTTPException(status_code=404, detail=f"Final answer content not found: {answer_id}")
        
        base_dir = _resolve_final_answer_dir(answer_id)
        
        # Try to get validator config from request body first (allows critiques without starting research)
        # Then fall back to autonomous coordinator's stored config
        validator_model = None
        validator_context_window = None
        validator_max_tokens = None
        validator_provider = None
        validator_openrouter_provider = None
        custom_prompt = None
        
        if request:
            custom_prompt = request.custom_prompt
            # Check if request provides validator config
            if request.validator_model:
                validator_model = request.validator_model
                validator_context_window = request.validator_context_window or 131072
                validator_max_tokens = request.validator_max_tokens or 25000
                validator_provider = request.validator_provider or "lm_studio"
                validator_openrouter_provider = request.validator_openrouter_provider
        
        # If no validator config from request, try coordinator
        if not validator_model:
            coordinator_config = autonomous_coordinator.get_validator_config()
            if coordinator_config:
                validator_model = coordinator_config["validator_model"]
                validator_context_window = coordinator_config["validator_context_window"]
                validator_max_tokens = coordinator_config["validator_max_tokens"]
                validator_provider = coordinator_config["validator_provider"]
                validator_openrouter_provider = coordinator_config.get("validator_openrouter_provider")
        
        # If still no config, error
        if not validator_model:
            raise HTTPException(
                status_code=400,
                detail="No validator model configured. Please configure a validator model in Autonomous Research Settings."
            )
        
        # Build the critique prompt
        prompt_to_use = custom_prompt if custom_prompt else DEFAULT_CRITIQUE_PROMPT
        full_prompt = build_critique_prompt(content, title, prompt_to_use)
        
        # Count tokens in the prompt
        prompt_tokens = count_tokens(full_prompt)
        
        # Calculate available input tokens
        output_reserve = validator_max_tokens
        safety_margin = int(validator_context_window * 0.1)
        available_input = validator_context_window - output_reserve - safety_margin
        
        # Check if content fits in context window
        if prompt_tokens > available_input:
            excess_tokens = prompt_tokens - available_input
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Final answer is too long for the validator's context window. "
                    f"The content requires {prompt_tokens:,} tokens, but the validator can only accept {available_input:,} tokens "
                    f"(context window: {validator_context_window:,}, output reserve: {output_reserve:,}, safety margin: {safety_margin:,}). "
                    f"The content exceeds the limit by {excess_tokens:,} tokens. "
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
        
        # Make the API call
        logger.info(f"Requesting critique for final answer {answer_id} from validator model {validator_model}")
        
        response = await api_client_manager.generate_completion(
            task_id=f"final_answer_critique_{answer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
            response_content = message.get("content") or message.get("reasoning") or ""
        
        if not response_content:
            raise HTTPException(status_code=500, detail="Empty response from validator model")
        
        # Parse with lenient fallback for truncated critique responses
        from backend.shared.critique_prompts import parse_critique_response
        critique_data = parse_critique_response(response_content)
        
        # Create critique object with correct field names
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
        
        # Save the critique with session-aware path
        saved_critique = await save_critique("final_answer", critique, answer_id, base_dir)
        
        return {
            "success": True,
            "critique": saved_critique.model_dump(),
            "answer_id": answer_id,
            "title": title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to request final answer critique for {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/final-answer-library/{answer_id}/critiques")
async def get_final_answer_critiques(answer_id: str):
    """
    Get all critiques for a final answer.
    
    Args:
        answer_id: The final answer ID (either "legacy" or a session folder name)
    
    Returns:
        List of critiques for the final answer
    """
    from backend.shared.critique_memory import get_critiques
    
    try:
        # Verify final answer exists
        await final_answer_memory.initialize()
        final_answer = await final_answer_memory.get_final_answer_by_id(answer_id)
        
        if not final_answer:
            raise HTTPException(status_code=404, detail=f"Final answer not found: {answer_id}")
        
        base_dir = _resolve_final_answer_dir(answer_id)
        
        title = final_answer.get("title", "Final Answer")
        critiques = await get_critiques("final_answer", answer_id, base_dir)
        
        return {
            "success": True,
            "answer_id": answer_id,
            "title": title,
            "critiques": [c.model_dump() for c in critiques],
            "count": len(critiques)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get critiques for final answer {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/final-answer-library/{answer_id}/critiques")
async def delete_final_answer_critiques(answer_id: str, confirm: bool = False):
    """
    Delete all critiques for a final answer.
    
    Args:
        answer_id: The final answer ID (either "legacy" or a session folder name)
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
        
        # Verify final answer exists
        await final_answer_memory.initialize()
        final_answer = await final_answer_memory.get_final_answer_by_id(answer_id)
        
        if not final_answer:
            raise HTTPException(status_code=404, detail=f"Final answer not found: {answer_id}")
        
        base_dir = _resolve_final_answer_dir(answer_id)
        
        await clear_critiques("final_answer", answer_id, base_dir)
        
        return {
            "success": True,
            "message": f"Critiques cleared for final answer {answer_id}",
            "answer_id": answer_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete critiques for final answer {answer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/default-critique-prompt")
async def get_default_critique_prompt():
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


# ============================================================================
# API CALL LOGGING ENDPOINTS
# ============================================================================

@router.get("/api-logs")
async def get_autonomous_api_logs(limit: int = 100):
    """
    Get autonomous research API call logs.
    
    Args:
        limit: Maximum number of log entries to return (default 100)
        
    Returns:
        Dict with logs and statistics
    """
    try:
        combined = await _get_combined_api_logs(limit=limit)
        
        return {
            "success": True,
            "logs": combined["logs"],
            "stats": combined["stats"],
        }
    except Exception as e:
        logger.error(f"Failed to get autonomous API logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api-logs/clear")
async def clear_autonomous_api_logs():
    """
    Clear all autonomous API logs.
    
    Returns:
        Success status
    """
    try:
        await autonomous_api_logger.clear_logs()
        await boost_logger.clear_logs()
        
        return {
            "success": True,
            "message": "Combined API logs cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear autonomous API logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-logs/stats")
async def get_autonomous_api_stats():
    """
    Get statistics about autonomous API calls.
    
    Returns:
        Statistics dict (total calls, by phase, by model, success rate, etc.)
    """
    try:
        combined = await _get_combined_api_logs(limit=1000)
        
        return {
            "success": True,
            "stats": combined["stats"]
        }
    except Exception as e:
        logger.error(f"Failed to get autonomous API stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))