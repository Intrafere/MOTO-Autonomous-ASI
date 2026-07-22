"""
Compiler Coordinator - orchestrates sequential Markov chain workflow.
Manages mode switching, submission/validation loop, and paper/outline updates.
"""
import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional, Dict, Callable, List, Tuple
from datetime import datetime

from backend.shared.config import system_config, rag_config
from backend.shared.models import CompilerState, CompilerSubmission, CompilerValidationResult, WorkflowTask, ValidationResult, ModelConfig
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError, OpenRouterInvalidResponseError
from backend.shared.brainstorm_proof_gate import BRAINSTORM_LEAN_PROOF_MARKER
from backend.shared.free_model_manager import free_model_manager
from backend.shared.json_parser import parse_json, sanitize_model_output_for_retry_context
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens
from backend.shared.context_overflow import (
    CONTEXT_OVERFLOW_RESOLUTION,
    CONTEXT_OVERFLOW_STOP_MESSAGE,
    CONTEXT_OVERFLOW_STOP_REASON,
    context_overflow_model_payload,
)
from backend.compiler.agents.writer_submitter import WritingSubmitter
from backend.compiler.agents.high_param_submitter import HighParamSubmitter
from backend.compiler.validation.compiler_validator import CompilerValidator, normalize_unicode_hyphens, find_with_normalized_hyphens
from backend.compiler.memory.outline_memory import outline_memory, OUTLINE_ANCHOR
from backend.compiler.memory.paper_memory import (
    paper_memory, 
    PAPER_ANCHOR,
    ABSTRACT_PLACEHOLDER,
    INTRO_PLACEHOLDER,
    CONCLUSION_PLACEHOLDER
)
from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log
from backend.compiler.memory.critique_memory import critique_memory
from backend.compiler.core.compiler_rag_manager import compiler_rag_manager
from backend.autonomous.memory.paper_model_tracker import PaperModelTracker
from backend.autonomous.memory.proof_database import manual_proof_database, proof_database

logger = logging.getLogger(__name__)

CRITIQUE_ATTEMPT_TARGET = 3
MAX_RIGOR_CYCLES_PER_LOOP = 5


def _is_rigor_model_call_failure(exc: Exception) -> bool:
    """Return true for provider/config failures that must not become proof declines."""
    from backend.shared.api_client_manager import RetryableProviderError

    message = str(exc or "").lower()
    return (
        isinstance(exc, (OpenRouterInvalidResponseError, RetryableProviderError))
        or is_non_retryable_model_error(exc)
        or is_transient_model_call_error(exc)
        or "model output incomplete" in message
        or "transient provider error" in message
        or "upstream provider timeout" in message
        or "response missing 'choices'" in message
    )


async def _cancel_and_drain_task(task: asyncio.Task) -> None:
    """Cancel a task, suppressing only cancellation while preserving real failures."""
    task.cancel()
    for result in await asyncio.gather(task, return_exceptions=True):
        if isinstance(result, asyncio.CancelledError):
            continue
        if isinstance(result, BaseException):
            raise result


LEAN_PROOF_EDIT_DENIAL_REASON = (
    "REJECTION REASON: Protected Lean 4 Proof\n\n"
    "ISSUE: The paper-writing retroactive brainstorm operation attempted to edit, delete, "
    "or add context to a Lean 4 verified proof in the brainstorm database.\n\n"
    "WHY THIS IS AN ISSUE: Lean 4 proof blocks are immutable from paper-writing modes. "
    "Paper writing may cite or discuss verified proofs in the paper, but it cannot mutate "
    "the proof text or attach context to the proof record. Only the normal brainstorm prune "
    "system may remove Lean 4 proof entries.\n\n"
    "FIX REQUIRED: Do not target Lean 4 proof submissions with brainstorm_operation. If a proof "
    "is unhelpful, let the scheduled brainstorm prune system handle removal. If the paper needs "
    "commentary, write that commentary in the paper prose instead of editing the proof."
)


def _classify_submitter_error(err: BaseException) -> tuple[str, str]:
    """
    Classify an exception raised by a WritingSubmitter.submit_* call.

    Distinguishes true context / prompt-size overflows (which are meaningful
    "decline to submit" signals) from upstream transport / API failures
    (non-JSON responses, connection errors, generic API errors) which are NOT
    context overflows and should not be reported to the user as such.

    Returns:
        (label, reason_prefix) where:
            - label is a short human-readable classification used in logs
              and UI messages (e.g. "Context overflow", "API transport error")
            - reason_prefix is the leading text used when building the
              full reason/reasoning string (e.g. "Context overflow: ...")
    """
    msg = str(err) if err is not None else ""
    msg_lower = msg.lower()

    if isinstance(err, OpenRouterInvalidResponseError):
        return ("API transport error", "API transport error")

    if (
        "prompt too large" in msg_lower
        or "tokens > " in msg_lower
        or "exceeds context limit" in msg_lower
        or "exceeds available input budget" in msg_lower
        or "exceeds the configured input budget" in msg_lower
    ):
        return ("Context overflow", "Context overflow")

    if msg_lower.startswith("openrouter api error") or msg_lower.startswith("openrouter connection failed") or msg_lower.startswith("openrouter rate limit"):
        return ("API transport error", "API transport error")

    return ("Submitter error", "Submitter error")


class CompilerCoordinator:
    """
    Coordinates the compiler system.
    - Sequential Markov chain workflow (submit → validate → update → submit)
    - Mode switching (construction → outline → review → rigor)
    - Paper and outline memory management
    - WebSocket broadcasting
    """
    
    def __init__(self):
        self.writer_submitter: Optional[WritingSubmitter] = None
        self.high_param_submitter: Optional[HighParamSubmitter] = None
        self.validator: Optional[CompilerValidator] = None
        self.solution_path_manager = None
        
        self.is_running = False
        self.current_mode = "idle"
        self.outline_accepted = False
        self.fatal_error_type: Optional[str] = None
        self.fatal_error_message: str = ""
        self.fatal_error_payload: Optional[Dict[str, Any]] = None
        self.top_level_terminal_callback: Optional[Callable[[], Any]] = None
        
        # Stats
        self.total_submissions = 0
        self.construction_acceptances = 0
        self.construction_rejections = 0
        self.construction_declines = 0
        self.rigor_acceptances = 0
        self.rigor_rejections = 0
        self.rigor_declines = 0
        self.outline_acceptances = 0
        self.outline_rejections = 0
        self.outline_declines = 0
        self.review_acceptances = 0
        self.review_rejections = 0
        self.review_declines = 0
        self.minuscule_edit_count = 0
        
        # Workflow state
        self.construction_cycle_count = 0
        self.rigor_cycle_active = False
        
        # Autonomous mode (for Part 3 integration)
        self.autonomous_mode = False
        self.autonomous_section_phase = None  # "body", "conclusion", "introduction", "abstract"
        self._current_topic_id = None  # Set by autonomous coordinator for retroactive brainstorm corrections
        self._current_paper_id: Optional[str] = None  # Set by autonomous coordinator for proof source identity
        self._current_rigor_proof_source_title: Optional[str] = None
        self._current_reference_paper_ids: List[str] = []  # Autonomous/Tier 3 references preserved for critique context
        self.allow_mathematical_proofs: bool = True
        
        # Critique phase state (post-body peer review)
        self.critique_submitter = None  # Rigor & Proofs submitter reused for critique generation
        self.critique_aggregator = None  # Coordinator instance for critique workflow
        self.in_critique_phase = False
        self.critique_acceptances = 0
        self.paper_version = 1  # Track version number
        self.paper_title: Optional[str] = None  # Track current paper title
        
        # Aggregator monitoring for incremental re-RAG
        self.aggregator_acceptances_last_rag = 0
        
        # WebSocket broadcaster
        self.websocket_broadcaster: Optional[Callable] = None
        
        # Main loop task
        self._main_task: Optional[asyncio.Task] = None
        self._aggregator_monitor_task: Optional[asyncio.Task] = None
        
        # Per-paper model tracking for manual mode (Part 2)
        self._paper_model_tracker: Optional[PaperModelTracker] = None
        self._current_paper_tracker: Optional[PaperModelTracker] = None
        
        # Workflow tracking
        self.workflow_tasks: List[WorkflowTask] = []
        self.completed_task_ids: set = set()
        self.current_task_sequence: int = 0
        self.current_task_id: Optional[str] = None  # Currently executing task

    def _notify_top_level_terminal(self) -> None:
        """Notify a route-owned top-level lifecycle without affecting child compilers."""
        if self.top_level_terminal_callback:
            try:
                self.top_level_terminal_callback()
            except Exception:
                logger.exception("Compiler top-level terminal callback failed")
    
    async def initialize(
        self,
        compiler_prompt: str,
        validator_model: str,
        writer_model: str,
        high_param_model: str,
        critique_submitter_model: str,
        skip_aggregator_db: bool = False,
        # OpenRouter provider config for validator
        validator_provider: str = "lm_studio",
        validator_openrouter_provider: Optional[str] = None,
        validator_openrouter_reasoning_effort: str = "auto",
        validator_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider config for writing submitter
        writer_provider: str = "lm_studio",
        writer_openrouter_provider: Optional[str] = None,
        writer_openrouter_reasoning_effort: str = "auto",
        writer_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider config for Rigor & Proofs submitter
        high_param_provider: str = "lm_studio",
        high_param_openrouter_provider: Optional[str] = None,
        high_param_openrouter_reasoning_effort: str = "auto",
        high_param_lm_studio_fallback: Optional[str] = None,
        # Deprecated critique compatibility fields, mirrored from Rigor & Proofs
        critique_submitter_provider: str = "lm_studio",
        critique_submitter_openrouter_provider: Optional[str] = None,
        critique_submitter_openrouter_reasoning_effort: str = "auto",
        critique_submitter_lm_studio_fallback: Optional[str] = None,
        validator_supercharge_enabled: bool = False,
        writer_supercharge_enabled: bool = False,
        high_param_supercharge_enabled: bool = False,
        critique_submitter_supercharge_enabled: bool = False,
        allow_mathematical_proofs: bool = True
    ) -> None:
        """
        Initialize the compiler coordinator.
        
        Args:
            compiler_prompt: User's compiler-directing prompt
            validator_model: Model for validator
            writer_model: Model for writing submitter
            high_param_model: Model for Rigor & Proofs submitter
            critique_submitter_model: Model for critique generation
            skip_aggregator_db: If True, don't load Part 1 aggregator database (for autonomous mode)
            validator_provider: Provider for validator ("lm_studio" or "openrouter")
            validator_openrouter_provider: OpenRouter host provider for validator
            validator_openrouter_reasoning_effort: OpenRouter reasoning effort for validator
            validator_lm_studio_fallback: LM Studio fallback model for validator
            writer_provider: Provider for writing submitter
            writer_openrouter_provider: OpenRouter host provider for writing submitter
            writer_openrouter_reasoning_effort: OpenRouter reasoning effort for writing submitter
            writer_lm_studio_fallback: LM Studio fallback model for writing submitter
            high_param_provider: Provider for Rigor & Proofs submitter
            high_param_openrouter_provider: OpenRouter host provider for Rigor & Proofs submitter
            high_param_openrouter_reasoning_effort: OpenRouter reasoning effort for Rigor & Proofs submitter
            high_param_lm_studio_fallback: LM Studio fallback model for Rigor & Proofs submitter
            critique_submitter_*: Deprecated compatibility aliases mirrored from Rigor & Proofs
        """
        logger.info("Initializing compiler coordinator...")
        
        # Store user prompt, paper title, and model configs
        self.user_prompt = compiler_prompt
        self.paper_title = compiler_prompt  # Initial title is the compiler prompt
        self._current_rigor_proof_source_title = compiler_prompt
        self.validator_model = validator_model
        self.validator_context_window = system_config.compiler_validator_context_window
        self.validator_max_tokens = system_config.compiler_validator_max_output_tokens
        # Deprecated critique role fields are compatibility aliases. Critique
        # generation now runs on the Rigor & Proofs submitter settings.
        self.critique_submitter_model = high_param_model
        
        # Store OpenRouter provider configs for all roles
        self.validator_provider = validator_provider
        self.validator_openrouter_provider = validator_openrouter_provider
        self.validator_openrouter_reasoning_effort = validator_openrouter_reasoning_effort
        self.validator_lm_studio_fallback = validator_lm_studio_fallback
        self.writer_provider = writer_provider
        self.writer_openrouter_provider = writer_openrouter_provider
        self.writer_openrouter_reasoning_effort = writer_openrouter_reasoning_effort
        self.writer_lm_studio_fallback = writer_lm_studio_fallback
        self.high_param_provider = high_param_provider
        self.high_param_openrouter_provider = high_param_openrouter_provider
        self.high_param_openrouter_reasoning_effort = high_param_openrouter_reasoning_effort
        self.high_param_lm_studio_fallback = high_param_lm_studio_fallback
        self.critique_submitter_provider = high_param_provider
        self.critique_submitter_openrouter_provider = high_param_openrouter_provider
        self.critique_submitter_openrouter_reasoning_effort = high_param_openrouter_reasoning_effort
        self.critique_submitter_lm_studio_fallback = high_param_lm_studio_fallback
        self.validator_supercharge_enabled = validator_supercharge_enabled
        self.writer_supercharge_enabled = writer_supercharge_enabled
        self.high_param_supercharge_enabled = high_param_supercharge_enabled
        self.critique_submitter_supercharge_enabled = high_param_supercharge_enabled
        self.allow_mathematical_proofs = bool(allow_mathematical_proofs)
        
        # Reset workflow state for fresh start
        self.outline_accepted = False
        self.is_running = False
        self.current_mode = "idle"
        
        # Reset stats
        self.total_submissions = 0
        self.construction_acceptances = 0
        self.construction_rejections = 0
        self.construction_declines = 0
        self.rigor_acceptances = 0
        self.rigor_rejections = 0
        self.rigor_declines = 0
        self.outline_acceptances = 0
        self.outline_rejections = 0
        self.outline_declines = 0
        self.review_acceptances = 0
        self.review_rejections = 0
        self.review_declines = 0
        self.minuscule_edit_count = 0
        self.construction_cycle_count = 0
        self.rigor_cycle_active = False
        self.aggregator_acceptances_last_rag = 0
        
        # Initialize memory
        await outline_memory.initialize()
        await paper_memory.initialize()
        await compiler_rejection_log.initialize()
        
        # Initialize Wolfram Alpha client if enabled
        if system_config.wolfram_alpha_enabled and system_config.wolfram_alpha_api_key:
            from backend.shared.wolfram_alpha_client import initialize_wolfram_client
            initialize_wolfram_client(system_config.wolfram_alpha_api_key)
            logger.info("Wolfram Alpha client initialized (available as a construction-mode tool)")
        
        # Note: Resume logic is handled in _main_workflow() to properly skip startup loops
        
        # Reset RAG manager state flags for fresh session
        # This ensures _aggregator_db_loaded and _initialized don't prevent proper setup
        compiler_rag_manager.reset()
        
        # Initialize RAG manager
        await compiler_rag_manager.initialize()
        
        # Load aggregator database (skip for autonomous mode - it loads brainstorm DB separately)
        if not skip_aggregator_db:
            # CRITICAL: Clear RAG for manual mode to prevent cross-contamination
            # from autonomous brainstorm content that may have been loaded in a prior session
            from backend.aggregator.core.rag_manager import rag_manager
            logger.info("Clearing RAG for fresh Part 2 compiler session...")
            await rag_manager.clear_all_documents_async()
            logger.info("RAG cleared successfully for Part 2 compiler")
            
            # Now load the Part 1 aggregator database into clean RAG
            await compiler_rag_manager.load_aggregator_database()
            
            # Load user-uploaded files into RAG (critical for manual mode context)
            user_uploads_dir = Path(system_config.user_uploads_dir)
            if user_uploads_dir.exists():
                logger.info(f"Loading user files from {user_uploads_dir}")
                user_files = list(user_uploads_dir.glob("*"))
                loaded_count = 0
                
                for file_path in user_files:
                    if file_path.is_file():
                        try:
                            await rag_manager.add_document(
                                str(file_path),
                                chunk_sizes=rag_config.submitter_chunk_intervals,  # All 4 configs (256/512/768/1024)
                                is_user_file=True  # High priority, permanent (never evicted)
                            )
                            logger.info(f"Loaded user file: {file_path.name}")
                            loaded_count += 1
                        except Exception as e:
                            logger.error(f"Failed to load user file {file_path.name}: {e}")
                
                logger.info(f"Manual compiler: Loaded {loaded_count} user files into RAG")
            else:
                logger.info(f"No user uploads directory found at {user_uploads_dir}")
        else:
            logger.info("Skipping Part 1 aggregator database load (autonomous mode)")
        
        # Create agents
        self.writer_submitter = WritingSubmitter(
            writer_model,
            compiler_prompt,
            websocket_broadcaster=self.websocket_broadcaster,
            proof_database_store=proof_database if self.autonomous_mode else None,
        )
        self.writer_submitter.solution_path_manager = self.solution_path_manager
        await self.writer_submitter.initialize()
        # Set up task tracking callback for workflow panel integration
        self.writer_submitter.set_task_tracking_callback(self._handle_task_event)
        # Configure API client manager for writing submitter (OpenRouter/LM Studio routing)
        api_client_manager.configure_role(
            role_id="compiler_writer",
            config=ModelConfig(
                provider=self.writer_provider,
                model_id=writer_model,
                openrouter_provider=self.writer_openrouter_provider,
                openrouter_reasoning_effort=self.writer_openrouter_reasoning_effort,
                lm_studio_fallback_id=self.writer_lm_studio_fallback,
                context_window=system_config.compiler_writer_context_window,
                max_output_tokens=system_config.compiler_writer_max_output_tokens,
                supercharge_enabled=self.writer_supercharge_enabled
            )
        )
        
        self.high_param_submitter = HighParamSubmitter(
            high_param_model, 
            compiler_prompt,
            websocket_broadcaster=self.websocket_broadcaster,
            validator_model=validator_model,
            validator_context_window=self.validator_context_window,
            validator_max_tokens=self.validator_max_tokens,
            proof_database_store=proof_database if self.autonomous_mode else manual_proof_database,
        )
        self.high_param_submitter.solution_path_manager = self.solution_path_manager
        self.high_param_submitter.set_rigor_proof_source(
            self._current_paper_id or "",
            self._current_rigor_proof_source_title or compiler_prompt,
        )
        self.high_param_submitter.set_rigor_proof_source(
            self._current_paper_id or "",
            self._current_rigor_proof_source_title or compiler_prompt,
        )
        await self.high_param_submitter.initialize()
        # Set up task tracking callback for workflow panel integration
        self.high_param_submitter.set_task_tracking_callback(self._handle_task_event)
        # Configure API client manager for Rigor & Proofs submitter (OpenRouter/LM Studio routing)
        api_client_manager.configure_role(
            role_id="compiler_high_param",
            config=ModelConfig(
                provider=self.high_param_provider,
                model_id=high_param_model,
                openrouter_provider=self.high_param_openrouter_provider,
                openrouter_reasoning_effort=self.high_param_openrouter_reasoning_effort,
                lm_studio_fallback_id=self.high_param_lm_studio_fallback,
                context_window=system_config.compiler_high_param_context_window,
                max_output_tokens=system_config.compiler_high_param_max_output_tokens,
                supercharge_enabled=self.high_param_supercharge_enabled
            )
        )
        high_param_role_config = ModelConfig(
            provider=self.high_param_provider,
            model_id=high_param_model,
            openrouter_provider=self.high_param_openrouter_provider,
            openrouter_reasoning_effort=self.high_param_openrouter_reasoning_effort,
            lm_studio_fallback_id=self.high_param_lm_studio_fallback,
            context_window=system_config.compiler_high_param_context_window,
            max_output_tokens=system_config.compiler_high_param_max_output_tokens,
            supercharge_enabled=self.high_param_supercharge_enabled
        )
        api_client_manager.configure_role(
            role_id="compiler_rigor_formalization",
            config=high_param_role_config,
        )
        validator_role_config = ModelConfig(
            provider=self.validator_provider,
            model_id=validator_model,
            openrouter_provider=self.validator_openrouter_provider,
            openrouter_reasoning_effort=self.validator_openrouter_reasoning_effort,
            lm_studio_fallback_id=self.validator_lm_studio_fallback,
            context_window=self.validator_context_window,
            max_output_tokens=self.validator_max_tokens,
            supercharge_enabled=self.validator_supercharge_enabled
        )
        api_client_manager.configure_role(
            role_id="compiler_rigor_novelty",
            config=validator_role_config,
        )
        
        self.validator = CompilerValidator(
            validator_model, 
            compiler_prompt,
            websocket_broadcaster=self.websocket_broadcaster,
            proof_database_store=proof_database if self.autonomous_mode else None,
            solution_path_manager=self.solution_path_manager,
        )
        await self.validator.initialize()
        # Set up task tracking callback for workflow panel integration
        self.validator.set_task_tracking_callback(self._handle_task_event)
        # Configure API client manager for validator (OpenRouter/LM Studio routing)
        api_client_manager.configure_role(
            role_id="compiler_validator",
            config=ModelConfig(
                provider=self.validator_provider,
                model_id=validator_model,
                openrouter_provider=self.validator_openrouter_provider,
                openrouter_reasoning_effort=self.validator_openrouter_reasoning_effort,
                lm_studio_fallback_id=self.validator_lm_studio_fallback,
                context_window=self.validator_context_window,
                max_output_tokens=self.validator_max_tokens,
                supercharge_enabled=self.validator_supercharge_enabled
            )
        )
        
        # Initialize per-paper model tracking (for manual Part 2 mode)
        # Only set up tracking if NOT in autonomous mode (autonomous coordinator handles its own tracking)
        if not self.autonomous_mode:
            self._paper_model_tracker = PaperModelTracker(
                user_prompt=compiler_prompt,
                paper_title=compiler_prompt  # Initial title is the compiler prompt
            )
            
            # Set up model tracking callback
            async def paper_model_tracking_callback(model_id: str) -> None:
                if self._paper_model_tracker:
                    self._paper_model_tracker.track_call(model_id)
            
            api_client_manager.set_model_tracking_callback(paper_model_tracking_callback)
            logger.info("Per-paper model tracking enabled for manual compiler mode")
        
        logger.info("Compiler coordinator initialized successfully")
        
        # Initialize workflow predictions
        await self.refresh_workflow_predictions()
    
    async def refresh_workflow_predictions(self) -> None:
        """Refresh workflow predictions based on actual agent state."""
        try:
            from backend.shared.boost_manager import boost_manager
            
            # Get actual sequence counters from agents
            writer_seq = self.writer_submitter.task_sequence if self.writer_submitter else 0
            hp_seq = self.high_param_submitter.task_sequence if self.high_param_submitter else 0
            val_seq = self.validator.task_sequence if self.validator else 0
            
            # Build workflow tasks based on current mode and actual sequences
            tasks = []
            
            if not self.outline_accepted:
                # Outline creation phase: writer -> validator -> writer -> validator ...
                for i in range(20):
                    if i % 2 == 0:
                        task_id = f"comp_writer_{writer_seq:03d}"
                        role = "Writing Submitter"
                        mode = "Outline Creation"
                        writer_seq += 1
                    else:
                        task_id = f"comp_val_{val_seq:03d}"
                        role = "Validator"
                        mode = "Outline Review"
                        val_seq += 1
                    
                    tasks.append(WorkflowTask(
                        task_id=task_id,
                        sequence_number=i + 1,
                        role=role,
                        mode=mode,
                        provider="lm_studio",
                        using_boost=boost_manager.should_use_boost(task_id)
                    ))
            else:
                # Construction cycle pattern
                cycle_pattern = [
                    ("writer", "Writing Submitter", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("writer", "Writing Submitter", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("writer", "Writing Submitter", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("writer", "Writing Submitter", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("writer", "Writing Submitter", "Outline Update"),
                    ("val", "Validator", "Outline Review"),
                    ("writer", "Writing Submitter", "Paper Review"),
                    ("val", "Validator", "Review Validation"),
                    ("writer", "Writing Submitter", "Paper Review"),
                    ("val", "Validator", "Review Validation"),
                    ("hp", "Rigor & Proofs", "Rigor Enhancement"),
                    ("val", "Validator", "Rigor Review"),
                ]
                
                for i in range(20):
                    pattern_idx = i % len(cycle_pattern)
                    agent_type, role, mode = cycle_pattern[pattern_idx]
                    
                    if agent_type == "writer":
                        task_id = f"comp_writer_{writer_seq:03d}"
                        writer_seq += 1
                    elif agent_type == "hp":
                        task_id = f"comp_hp_{hp_seq:03d}"
                        hp_seq += 1
                    else:
                        task_id = f"comp_val_{val_seq:03d}"
                        val_seq += 1
                    
                    tasks.append(WorkflowTask(
                        task_id=task_id,
                        sequence_number=i + 1,
                        role=role,
                        mode=mode,
                        provider="lm_studio",
                        using_boost=boost_manager.should_use_boost(task_id)
                    ))
            
            self.workflow_tasks = tasks
            
            # Broadcast workflow update
            if self.websocket_broadcaster:
                await self.websocket_broadcaster("workflow_updated", {
                    "tasks": [task.model_dump() for task in self.workflow_tasks],
                    "mode": "compiler"
                })
            
            logger.debug(f"Refreshed compiler workflow predictions: {len(self.workflow_tasks)} tasks")
        except Exception as e:
            logger.error(f"Failed to refresh compiler workflow predictions: {e}")
    
    async def get_next_task(self) -> Optional[WorkflowTask]:
        """Get the next task in the workflow queue."""
        for task in self.workflow_tasks:
            if not task.completed and task.task_id not in self.completed_task_ids:
                return task
        return None
    
    async def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        self.completed_task_ids.add(task_id)
        self.current_task_sequence += 1
        
        # Update task in workflow list
        for task in self.workflow_tasks:
            if task.task_id == task_id:
                task.completed = True
                task.active = False
                break
        
        # Broadcast task completion
        if self.websocket_broadcaster:
            await self.websocket_broadcaster("task_completed", {
                "task_id": task_id,
                "sequence": self.current_task_sequence
            })
        
        # Refresh predictions after mode changes or phase transitions
        await self.refresh_workflow_predictions()
    
    def _handle_task_event(self, event_type: str, task_id: str) -> None:
        """
        Handle task events from submitters and validator.
        Called synchronously by agents; schedules async work on event loop.
        
        Args:
            event_type: "started" or "completed"
            task_id: The task ID (e.g., "comp_writer_001", "comp_hp_002", "comp_val_003")
        """
        if event_type == "started":
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.mark_task_started(task_id))
                else:
                    loop.run_until_complete(self.mark_task_started(task_id))
            except Exception as e:
                logger.debug(f"Could not mark task started: {e}")
        elif event_type == "completed":
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.mark_task_completed(task_id))
                else:
                    loop.run_until_complete(self.mark_task_completed(task_id))
            except Exception as e:
                logger.debug(f"Could not mark task completed: {e}")
    
    async def mark_task_started(self, task_id: str) -> None:
        """Mark a task as actively running."""
        self.current_task_id = task_id
        
        # Update task in workflow list
        for task in self.workflow_tasks:
            if task.task_id == task_id:
                task.active = True
            else:
                task.active = False
        
        # Broadcast task start
        if self.websocket_broadcaster:
            await self.websocket_broadcaster("task_started", {
                "task_id": task_id
            })
    
    def enable_autonomous_mode(self):
        """Enable autonomous mode with fixed section order (Body → Conclusion → Intro → Abstract)."""
        self.autonomous_mode = True
        self.autonomous_section_phase = "body"
        logger.info("Autonomous mode enabled - section order: Body → Conclusion → Intro → Abstract")

    def set_rigor_proof_source(self, paper_id: Optional[str], paper_title: Optional[str] = None) -> None:
        """Bind compiler rigor proof records to the real paper being written."""
        self._current_paper_id = (paper_id or "").strip() or None
        self._current_rigor_proof_source_title = (
            (paper_title or "").strip()
            or self.paper_title
            or getattr(self, "user_prompt", "")
        )
        if self.high_param_submitter:
            self.high_param_submitter.set_rigor_proof_source(
                self._current_paper_id or "",
                self._current_rigor_proof_source_title or "",
            )
    
    def _is_body_complete(self, paper: str) -> bool:
        """
        Check if body construction is complete.
        
        Used to determine whether to skip rigor enhancement and outline updates,
        which should ONLY run during body construction phase.
        
        Args:
            paper: Current paper content
        
        Returns:
            True if body is complete (should skip rigor/outline updates), False otherwise
        """
        # Autonomous mode: use explicit phase tracking
        if self.autonomous_mode:
            return self.autonomous_section_phase != "body"
        
        # Manual mode: body is complete if Conclusion section exists in paper
        # This uses the same flexible pattern as other conclusion detection in the codebase
        return bool(re.search(
            r"(?:^|\n)\s*(?:#+\s*)?(?:[IVXLCDM]+\.?\s*)?(?:Conclusion|Summary|Discussion|Final\s*Remarks|Concluding\s*Remarks)",
            paper, re.IGNORECASE | re.MULTILINE
        ))
    
    async def start(self) -> None:
        """Start the compiler system."""
        if self.is_running:
            logger.warning("Compiler already running")
            return
        
        self.is_running = True
        self.fatal_error_type = None
        self.fatal_error_message = ""
        self.fatal_error_payload = None
        logger.info("Starting compiler...")
        
        try:
            free_model_manager.reset()
            await self.refresh_workflow_predictions()
            self._main_task = asyncio.create_task(self._main_workflow())

            if not self.autonomous_mode:
                self._aggregator_monitor_task = asyncio.create_task(self._monitor_aggregator_for_rerag())
            else:
                self._aggregator_monitor_task = None

            await self._broadcast("compiler_started", {"message": "Compiler started"})
            logger.info("Compiler started successfully")
        except BaseException:
            self.is_running = False
            for task in (self._main_task, self._aggregator_monitor_task):
                if task and not task.done():
                    await _cancel_and_drain_task(task)
            self._main_task = None
            self._aggregator_monitor_task = None
            raise

    async def _load_rigor_source_material_context(self) -> tuple[str, str]:
        """Load direct brainstorm/aggregator context for paper-writing proof mode."""
        if self.autonomous_mode and self._current_topic_id:
            try:
                from backend.autonomous.memory.brainstorm_memory import brainstorm_memory

                content = await brainstorm_memory.get_database_content(
                    self._current_topic_id,
                    strip_proofs=True,
                )
                return content or "", f"Source brainstorm {self._current_topic_id}"
            except Exception as exc:
                logger.debug("Unable to load autonomous brainstorm context for rigor: %s", exc)
                return "", ""

        try:
            shared_path = Path(system_config.shared_training_file)
            if not shared_path.exists():
                return "", ""
            content = await asyncio.to_thread(shared_path.read_text, encoding="utf-8")
            proof_marker = "=== PROOFS GENERATED FROM THIS BRAINSTORM"
            if proof_marker in content:
                content = content.split(proof_marker, 1)[0].rstrip()
            return content or "", "Part 1 aggregator database"
        except Exception as exc:
            logger.debug("Unable to load manual aggregator context for rigor: %s", exc)
            return "", ""
    
    async def stop(self) -> None:
        """Stop the compiler system."""
        tasks_alive = any(
            task is not None and not task.done()
            for task in (self._main_task, self._aggregator_monitor_task)
        )
        if not self.is_running and not tasks_alive:
            self._notify_top_level_terminal()
            return
        
        self.is_running = False
        logger.info("Stopping compiler...")
        
        # Clear model tracking callback if this is manual mode
        if not self.autonomous_mode and self._paper_model_tracker:
            api_client_manager.set_model_tracking_callback(None)
            logger.info(f"Compiler stopped: tracked {len(self._paper_model_tracker.get_models_dict())} models, "
                       f"{self._paper_model_tracker.total_calls} API calls")
        
        if self._main_task:
            await _cancel_and_drain_task(self._main_task)
        self._main_task = None
        
        if self._aggregator_monitor_task:
            await _cancel_and_drain_task(self._aggregator_monitor_task)
        self._aggregator_monitor_task = None
        
        await self._broadcast("compiler_stopped", {"message": "Compiler stopped"})
        logger.info("Compiler stopped")
        self._notify_top_level_terminal()

    async def _handle_context_overflow(self, error: BaseException, *, role_id: str, mode: Optional[str] = None) -> None:
        """Stop the compiler when mandatory direct context cannot fit."""
        self.fatal_error_type = CONTEXT_OVERFLOW_STOP_REASON
        self.fatal_error_message = str(error)
        self.is_running = False
        config_role_id = {
            "compiler_rigor": "compiler_high_param",
        }.get(role_id, role_id)
        feedback = getattr(error, "feedback", None)
        feedback_route_payload = {
            key: value
            for key, value in {
                "configured_model": getattr(feedback, "configured_model", None),
                "configured_provider": getattr(feedback, "configured_provider", None),
                "effective_model": getattr(feedback, "effective_model", None),
                "effective_provider": getattr(feedback, "effective_provider", None),
            }.items()
            if value
        }
        payload = {
            "workflow_mode": "autonomous" if self.autonomous_mode else "compiler",
            "role_id": role_id,
            **context_overflow_model_payload(
                api_client_manager.get_role_config(config_role_id)
            ),
            **feedback_route_payload,
            "mode": mode or self.current_mode,
            "reason": CONTEXT_OVERFLOW_STOP_REASON,
            "message": CONTEXT_OVERFLOW_STOP_MESSAGE,
            "error_detail": str(error),
            "resolution": CONTEXT_OVERFLOW_RESOLUTION,
        }
        self.fatal_error_payload = dict(payload)
        logger.error("Compiler context overflow in %s: %s", role_id, error)
        await self._broadcast("context_overflow_error", payload)
    
    async def _main_workflow(self) -> None:
        """Main compiler workflow loop."""
        logger.info("Compiler workflow started")
        
        try:
            # Check if we're resuming from an existing paper
            current_paper = await paper_memory.get_paper()
            current_outline = await outline_memory.get_outline()
            is_resuming_paper = bool(current_paper and current_paper.strip())
            is_resuming_outline = bool(current_outline and current_outline.strip())
            
            if is_resuming_paper:
                logger.info(f"Resuming from existing paper ({await paper_memory.get_word_count()} words) - skipping startup loops")
                # Ensure outline_accepted flag is set when resuming
                self.outline_accepted = True
                
                # CRITICAL: Ensure placeholders exist in the paper
                # Papers created before the placeholder system or from older versions
                # may not have placeholders, causing "old_string not found" failures
                # when the model tries to use placeholder text as old_string.
                placeholders_added = await paper_memory.ensure_placeholders_exist()
                if placeholders_added:
                    logger.info("Placeholders were missing and have been added to the paper")
            elif is_resuming_outline:
                logger.info("Resuming from existing outline (no paper yet) - skipping outline creation, starting paper construction")
                # Outline exists but no paper - skip outline creation, go to initial paper
                self.outline_accepted = True
                await self._initial_paper_loop()
            else:
                logger.info("Starting fresh - no existing paper or outline found")
                # STARTUP SEQUENCE: Create initial outline
                await self._outline_creation_loop()
                
                # STARTUP SEQUENCE: Create initial paper portion
                await self._initial_paper_loop()
            
            # MAIN LOOPS: Alternate between construction and rigor
            while self.is_running:
                # LOOP 1: Paper Construction (steps 5-18)
                await self._construction_loop()
                
                # LOOP 2: Rigor Enhancement (steps 19-21) - ONLY during body phase
                current_paper = await paper_memory.get_paper()
                if not self._is_body_complete(current_paper):
                    await self._rigor_loop()
                else:
                    logger.info("Skipping rigor loop - body construction complete")
                
        except asyncio.CancelledError:
            logger.info("Compiler workflow cancelled")
        except FreeModelExhaustedError as e:
            # All free models exhausted after retries - wait briefly and retry
            logger.warning(f"Compiler: all free models exhausted: {e}")
            await self._broadcast("free_models_exhausted", {
                "role_id": "compiler",
                "message": "All free models exhausted, waiting to retry",
            })
            await asyncio.sleep(120)  # Wait before retrying (all models exhausted)
            if self.is_running:
                self._main_task = asyncio.create_task(self._main_workflow())
        except RetryableProviderError as e:
            logger.warning("Compiler paused for retryable provider failure: %s", e)
            await api_client_manager.wait_for_retryable_provider_error(
                e,
                role_id=e.role_id or "compiler",
                should_stop=lambda: not self.is_running,
            )
            if self.is_running:
                self._main_task = asyncio.create_task(self._main_workflow())
        except ValueError as e:
            is_context_overflow = (
                "mandatory full source context" in str(e).lower()
                or _classify_submitter_error(e)[0] == "Context overflow"
            )
            if not is_context_overflow:
                logger.error(f"Compiler workflow error: {e}", exc_info=True)
                self.is_running = False
                await self._broadcast("compiler_error", {
                    "error": "Compiler workflow encountered an internal error",
                    "mode": self.current_mode,
                    "total_submissions": self.total_submissions
                })
                return
            await self._handle_context_overflow(e, role_id="compiler_rigor")
        except Exception as e:
            logger.error(f"Compiler workflow error: {e}", exc_info=True)
            self.is_running = False
            await self._broadcast("compiler_error", {
                "error": "Compiler workflow encountered an internal error",
                "mode": self.current_mode,
                "total_submissions": self.total_submissions
            })
        finally:
            if not self.is_running:
                self._notify_top_level_terminal()
    
    def _pre_validate_outline_structure(self, content: str) -> Optional[str]:
        """
        Pre-validate outline structure before sending to LLM validator.
        Returns detailed error message if critical issues found, None if OK.
        
        This provides IMMEDIATE, CONSISTENT feedback on structural issues
        without relying on LLM validator interpretation.
        """
        # Abstract is OPTIONAL. If included, it must be the unnumbered first
        # section. Introduction, at least one body section, and Conclusion are
        # required in that order.
        heading_pattern = re.compile(
            r"^\s*(?P<number>(?:[IVXLCDM]+|\d+)\.\s*)?"
            r"(?P<title>Abstract|Introduction|Conclusion)\s*$",
            re.MULTILINE | re.IGNORECASE,
        )
        headings = list(heading_pattern.finditer(content))
        abstract_headings = [
            match for match in headings
            if match.group("title").lower() == "abstract"
        ]
        if abstract_headings:
            abstract = abstract_headings[0]
            if abstract.group("number"):
                return (
                    "PRE-VALIDATION FAILED: INCORRECT_SECTION_NAME - Abstract\n\n"
                    "If included, Abstract must use the unnumbered heading 'Abstract'. "
                    "Numbered forms such as 'I. Abstract' and '0. Abstract' are invalid."
                )
            first_nonempty_line = next(
                (line.strip() for line in content.splitlines() if line.strip()),
                "",
            )
            if first_nonempty_line.lower() != "abstract":
                return (
                    "PRE-VALIDATION FAILED: INCORRECT_SECTION_ORDER - Abstract\n\n"
                    "If included, the unnumbered Abstract must be the first section."
                )
        
        # Check for "Introduction" header
        intro_pattern = r'^\s*(?:I\.\s*)?Introduction\s*$'
        if not re.search(intro_pattern, content, re.MULTILINE | re.IGNORECASE):
            return """PRE-VALIDATION FAILED: MISSING_REQUIRED_SECTION - Introduction

Your outline must include a line with 'Introduction' or 'I. Introduction' as a section header.

VALID EXAMPLES:
✓ Introduction
✓ I. Introduction
✓ INTRODUCTION

INVALID:
❌ Overview
❌ Background
❌ Intro"""
        
        # Check for "Conclusion" header  
        concl_pattern = r'^\s*(?:[IVX]+\.\s*)?Conclusion\s*$'
        if not re.search(concl_pattern, content, re.MULTILINE | re.IGNORECASE):
            return """PRE-VALIDATION FAILED: MISSING_REQUIRED_SECTION - Conclusion

Your outline must include a line with 'Conclusion' or 'N. Conclusion' (where N is a Roman numeral) as a section header.

VALID EXAMPLES:
✓ Conclusion
✓ V. Conclusion
✓ VII. CONCLUSION

INVALID:
❌ Summary
❌ Final Remarks
❌ Closing"""

        intro_match = re.search(
            intro_pattern, content, re.MULTILINE | re.IGNORECASE
        )
        conclusion_match = re.search(
            concl_pattern, content, re.MULTILINE | re.IGNORECASE
        )
        if intro_match and conclusion_match and intro_match.start() >= conclusion_match.start():
            return (
                "PRE-VALIDATION FAILED: INCORRECT_SECTION_ORDER\n\n"
                "Introduction must appear before the body sections and Conclusion."
            )
        body_between = (
            content[intro_match.end():conclusion_match.start()].strip()
            if intro_match and conclusion_match
            else ""
        )
        body_heading_pattern = r"^\s*(?:[IVXLCDM]+|\d+)\.\s+\S.*$"
        if not re.search(
            body_heading_pattern,
            body_between,
            re.MULTILINE | re.IGNORECASE,
        ):
            return (
                "PRE-VALIDATION FAILED: MISSING_REQUIRED_SECTION - Body\n\n"
                "Your outline must include at least one numbered body section "
                "between Introduction and Conclusion."
            )
        
        return None  # All critical checks passed
    
    async def _outline_creation_loop(self) -> None:
        """
        PHASE 1: Iterative outline creation with validator feedback.
        
        Submitter refines outline multiple times until satisfied (outline_complete=true)
        or iteration limit reached (15).
        
        This replaces the old single-shot outline creation with an iterative refinement process.
        """
        MAX_ITERATIONS = 15
        iteration = 0
        last_submission = None
        
        self.current_mode = "outline_create"
        
        logger.info("=" * 60)
        logger.info("PHASE 1: ITERATIVE OUTLINE CREATION")
        logger.info("=" * 60)
        
        while iteration < MAX_ITERATIONS and self.is_running:
            iteration += 1
            logger.info(f"\n--- Outline Creation Iteration {iteration}/{MAX_ITERATIONS} ---")
            
            # Generate outline (creation or refinement)
            try:
                submission = await self.writer_submitter.submit_outline_create()
            except FreeModelExhaustedError:
                raise
            except ValueError as e:
                label, _ = _classify_submitter_error(e)
                if label == "Context overflow":
                    await self._handle_context_overflow(e, role_id="compiler_writer", mode="outline_create")
                    return
                logger.error(f"Iteration {iteration}: Outline submission failed with error: {e} - retrying")
                await asyncio.sleep(5)
                iteration -= 1
                continue
            except Exception as e:
                logger.error(f"Iteration {iteration}: Outline submission failed with error: {e} - retrying")
                await asyncio.sleep(5)
                iteration -= 1
                continue
            
            if submission is None:
                logger.error(f"Iteration {iteration}: Failed to generate outline submission")
                # Don't count as iteration, retry
                await asyncio.sleep(1)
                continue
            
            last_submission = submission
            self.total_submissions += 1
            
            await self._broadcast("compiler_submission", {
                "mode": "outline_create",
                "iteration": iteration,
                "submission_id": submission.submission_id
            })
            
            # PRE-VALIDATION: Check critical structure before sending to LLM validator
            pre_validation_error = self._pre_validate_outline_structure(submission.content)
            if pre_validation_error:
                # Create rejection result without calling validator
                logger.warning(f"Iteration {iteration}: PRE-VALIDATION FAILED - {pre_validation_error[:100]}...")
                
                result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=pre_validation_error,
                    summary=pre_validation_error[:750],  # Truncate for summary
                    json_valid=True,
                    validation_stage="pre-validation"
                )
                
                # Store feedback in rolling window
                await outline_memory.add_creation_feedback(
                    reasoning=pre_validation_error,
                    is_accepted=False
                )
                
                # Log rejection
                await compiler_rejection_log.add_rejection(result, "outline_create", submission.content)
                
                logger.info(f"✗ Iteration {iteration}: Outline REJECTED (pre-validation)")
                logger.info(f"  Reason: {pre_validation_error[:150]}...")
                logger.info(f"  Continuing to iteration {iteration + 1} with feedback...")
                
                # Continue to next iteration
                continue
            
            # Validate outline with LLM validator
            result = await self.validator.validate_submission(
                submission,
                current_paper="",
                current_outline=submission.content
            )
            
            # Store feedback in rolling window (regardless of accept/reject)
            # CRITICAL: Include outline content when accepted so model can see its own work
            await outline_memory.add_creation_feedback(
                reasoning=result.reasoning,
                is_accepted=(result.decision == "accept"),
                outline_content=submission.content if result.decision == "accept" else ""
            )
            
            if result.decision == "accept":
                self.outline_acceptances += 1
                
                await compiler_rejection_log.add_acceptance(
                    submission.submission_id,
                    "outline_create",
                    submission.content[:500]
                )
                
                await self._broadcast("compiler_acceptance", {
                    "mode": "outline_create",
                    "iteration": iteration,
                    "submission_id": submission.submission_id
                })
                
                logger.info(f"✓ Iteration {iteration}: Outline ACCEPTED")
                logger.info(f"  Validator feedback: {result.reasoning[:200]}...")
                
                # Check if submitter marked outline complete
                if submission.outline_complete:
                    logger.info("=" * 60)
                    logger.info("OUTLINE MARKED COMPLETE BY SUBMITTER - LOCKING")
                    logger.info("=" * 60)
                    
                    await outline_memory.update_outline(submission.content)
                    await outline_memory.clear_creation_feedback()
                    self.outline_accepted = True
                    
                    await self._broadcast("outline_locked", {
                        "iteration": iteration,
                        "total_iterations": iteration,
                        "outline_length": len(submission.content),
                        "reason": "submitter_complete"
                    })
                    
                    await self._broadcast("outline_updated", {
                        "outline": submission.content
                    })
                    
                    logger.info(f"Outline locked after {iteration} iterations")
                    return
                else:
                    logger.info(f"  Submitter wants to refine further (outline_complete=false)")
                    logger.info(f"  Continuing to iteration {iteration + 1}...")
                    # Continue to next iteration - do NOT update outline yet
            else:
                # Rejected
                self.outline_rejections += 1
                
                await compiler_rejection_log.add_rejection(result, "outline_create", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "outline_create",
                    "iteration": iteration,
                    "submission_id": submission.submission_id,
                    "reasoning": result.reasoning
                })
                
                logger.info(f"✗ Iteration {iteration}: Outline REJECTED")
                logger.info(f"  Reason: {result.reasoning[:200]}...")
                logger.info(f"  Continuing to iteration {iteration + 1} with feedback...")
        
        # Hit iteration limit without completion
        if iteration >= MAX_ITERATIONS:
            logger.warning("=" * 60)
            logger.warning(f"ITERATION LIMIT REACHED ({MAX_ITERATIONS}) - FORCE COMPLETING")
            logger.warning("=" * 60)
            
            # Force lock with last submission if available
            if last_submission and last_submission.content:
                logger.warning(f"Force-locking outline from iteration {iteration}")
                await outline_memory.update_outline(last_submission.content)
                await outline_memory.clear_creation_feedback()
                self.outline_accepted = True
                
                await self._broadcast("outline_forced_complete", {
                    "iterations": MAX_ITERATIONS,
                    "outline_length": len(last_submission.content),
                    "reason": "iteration_limit"
                })
                
                await self._broadcast("outline_updated", {
                    "outline": last_submission.content
                })
                
                logger.info("Outline force-locked due to iteration limit")
                return
            else:
                logger.error("No valid outline submission to force-lock")
                # Set outline_accepted to False and let the system handle the error
                self.outline_accepted = False
                return
        
        # Stopped by user
        logger.warning("Outline creation interrupted by user stop")
    
    async def _initial_paper_loop(self) -> None:
        """STARTUP: Initial paper portion loop (steps 3-4)."""
        logger.info("Starting initial paper loop...")
        self.current_mode = "construction"
        
        initial_portion_accepted = False
        attempt = 0
        rejection_feedback = None  # Store rejection feedback for retry
        
        while self.is_running and not initial_portion_accepted:
            # Writing submitter writes the first portion.
            submission = None
            attempt += 1
            backoff_time = min(2 ** (attempt - 1), 16)  # 1s, 2s, 4s, 8s, 16s max
            
            try:
                section_phase = self.autonomous_section_phase if self.autonomous_mode else None
                
                # Load brainstorm content for first construction too
                first_brainstorm_content = None
                first_brainstorm_source = None
                if self.autonomous_mode and self._current_topic_id:
                    try:
                        from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
                        first_brainstorm_content = await brainstorm_memory.get_database_content(self._current_topic_id, strip_proofs=True)
                        first_brainstorm_source = f"brainstorm_{self._current_topic_id}.txt"
                    except Exception as exc:
                        logger.debug("Unable to load initial brainstorm context for construction: %s", exc)
                
                submission = await self.writer_submitter.submit_construction(
                    is_first_portion=True,
                    section_phase=section_phase,
                    rejection_feedback=rejection_feedback,
                    brainstorm_content=first_brainstorm_content,
                    brainstorm_source_name=first_brainstorm_source
                )
                
                if submission is None:
                    logger.warning(
                        f"Construction submission returned None (attempt {attempt}). "
                        f"Retrying in {backoff_time}s..."
                    )
                    await self._broadcast("compiler_retry", {
                        "mode": "construction",
                        "attempt": attempt,
                        "reason": "Empty submission returned"
                    })
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    if attempt > 1:
                        logger.info(f"Construction submission succeeded after {attempt} attempts")
                    
            except FreeModelExhaustedError:
                raise
            except (ValueError, OpenRouterInvalidResponseError) as e:
                label, reason_prefix = _classify_submitter_error(e)
                if label == "Context overflow":
                    await self._handle_context_overflow(e, role_id="compiler_writer", mode="construction")
                    return
                logger.error(f"Construction {label.lower()} in initial loop (attempt {attempt}): {e}")
                await self._broadcast("compiler_rejection", {
                    "mode": "construction",
                    "reasoning": f"{reason_prefix}: {e}"
                })
                await compiler_rejection_log.add_rejection(
                    CompilerValidationResult(
                        submission_id=str(uuid.uuid4()),
                        decision="reject",
                        reasoning=str(e),
                        summary=str(e)[:750],
                        validation_stage="internal_error"
                    ), "construction", ""
                )
                await asyncio.sleep(backoff_time)
                continue
            except Exception as e:
                logger.error(f"Construction submission failed with error (attempt {attempt}): {e}")
                await self._broadcast("compiler_retry", {
                    "mode": "construction",
                    "attempt": attempt,
                    "reason": str(e)
                })
                logger.info(f"Retrying in {backoff_time}s...")
                await asyncio.sleep(backoff_time)
                continue
            
            # submission is valid if we reach here
            
            self.total_submissions += 1
            
            await self._broadcast("compiler_submission", {
                "mode": "construction",
                "submission_id": submission.submission_id,
                "is_first": True
            })
            
            # Validator reviews
            current_outline = await outline_memory.get_outline()
            result = await self.validator.validate_submission(
                submission,
                current_paper="",
                current_outline=current_outline
            )
            
            if result.decision == "accept":
                # Accept initial portion - initialize with placeholders
                # This adds visible placeholder markers for Conclusion, Introduction, and Abstract
                # so the AI can clearly see these sections don't exist yet
                await paper_memory.initialize_with_placeholders(submission.content)
                initial_portion_accepted = True
                self.construction_acceptances += 1
                self._track_submission_wolfram_calls(submission)
                
                await compiler_rejection_log.add_acceptance(
                    submission.submission_id,
                    "construction",
                    submission.content[:500]
                )
                
                word_count = await paper_memory.get_word_count()
                
                await self._broadcast("compiler_acceptance", {
                    "mode": "construction",
                    "submission_id": submission.submission_id
                })
                
                await self._broadcast("paper_updated", {
                    "word_count": word_count,
                    "preview": submission.content[:500]
                })
                
                # Check for phase transitions in autonomous mode (explicit section_complete)
                if self.autonomous_mode and self.autonomous_section_phase:
                    paper_complete = await self._check_phase_transition(section_complete=submission.section_complete)
                    if paper_complete:
                        logger.info("Paper fully complete after initial portion!")
                        return
                
                logger.info(f"Initial paper portion accepted with placeholders! ({word_count} words)")
            else:
                # Reject
                self.construction_rejections += 1
                
                await compiler_rejection_log.add_rejection(result, "construction", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "construction",
                    "submission_id": submission.submission_id,
                    "reasoning": result.reasoning
                })
                
                # Store rejection feedback for next retry attempt
                # This helps the LLM learn from the specific rejection reason
                rejection_feedback = result.reasoning
                logger.info(f"Initial portion rejected (feedback will guide next retry): {result.reasoning[:200]}...")
        
        logger.info("Initial paper loop complete")
    
    async def _construction_loop(self) -> None:
        """LOOP 1: Paper construction (steps 5-18)."""
        logger.info("Starting construction loop...")
        
        # Track rejection feedback to pass to next iteration
        rejection_feedback = None
        
        # 4× construction submissions (steps 5-12)
        for i in range(4):
            if not self.is_running:
                break
            
            accepted, rejection_reason = await self._submit_and_validate_construction(
                rejection_feedback=rejection_feedback
            )
            # Pass rejection to next iteration if rejected, clear if accepted
            rejection_feedback = rejection_reason if not accepted else None
        
        # 1× outline review/update (steps 13-14) - ONLY during body phase
        if self.is_running:
            current_paper = await paper_memory.get_paper()
            if not self._is_body_complete(current_paper):
                await self._submit_and_validate_outline_update()
            else:
                logger.info("Skipping outline update - body construction complete")
        
        # 2× paper review/cleanup (steps 15-18)
        for i in range(2):
            if not self.is_running:
                break
            
            await self._submit_and_validate_review()
        
        logger.info("Construction loop complete")

    def _track_submission_wolfram_calls(self, submission: CompilerSubmission) -> None:
        """Record accepted construction-mode Wolfram tool calls in paper credits.

        WritingSubmitter stores the full Wolfram audit trail on
        `submission.metadata["wolfram_calls"]`. PaperModelTracker only tracks a
        count (and accepts the query for logging), so we bridge the two here
        after the paper operation has been accepted.
        """
        wolfram_calls = (submission.metadata or {}).get("wolfram_calls") or []
        if not wolfram_calls:
            return

        tracker = (
            self._current_paper_tracker
            if self.autonomous_mode
            else self._paper_model_tracker
        )
        if not tracker:
            logger.debug(
                "Accepted submission had %s Wolfram call(s), but no paper tracker is active.",
                len(wolfram_calls),
            )
            return

        for call in wolfram_calls:
            query = ""
            if isinstance(call, dict):
                query = str(call.get("query", "") or "").strip()
            else:
                query = str(call or "").strip()
            tracker.track_wolfram_call(query)
        logger.info("Tracked %s accepted Wolfram Alpha construction call(s)", len(wolfram_calls))
    
    async def _rigor_loop(self) -> None:
        """LOOP 2: Rigor enhancement.

        With the Lean-4-verified-theorem flow, every verified theorem lands
        somewhere (inline or appendix). The loop continues while
        `_submit_and_validate_rigor` returns True, but yields back to
        construction after at most MAX_RIGOR_CYCLES_PER_LOOP consecutive
        theorem cycles.
        """
        logger.info("Starting rigor loop...")
        self.rigor_cycle_active = True
        rigor_cycles_run = 0

        # Continue until first decline or the consecutive-cycle cap.
        while (
            self.is_running
            and self.rigor_cycle_active
            and rigor_cycles_run < MAX_RIGOR_CYCLES_PER_LOOP
        ):
            rigor_cycles_run += 1
            logger.info(
                "Running rigor cycle %s/%s",
                rigor_cycles_run,
                MAX_RIGOR_CYCLES_PER_LOOP,
            )
            continued = await self._submit_and_validate_rigor()
            
            if not continued:
                # Decline - end this rigor loop and return to construction.
                self.rigor_cycle_active = False
                logger.info("Rigor cycle ended (decline: no more theorems or Lean failed)")
        
        if (
            self.is_running
            and self.rigor_cycle_active
            and rigor_cycles_run >= MAX_RIGOR_CYCLES_PER_LOOP
        ):
            self.rigor_cycle_active = False
            logger.info(
                "Rigor cycle cap reached (%s); returning to construction",
                MAX_RIGOR_CYCLES_PER_LOOP,
            )

        logger.info("Rigor loop complete")
    
    # Maximum retries for premature decline/completion rejections
    MAX_PREMATURE_RETRIES = 5
    PRE_ABSTRACT_RED_TEAM_MAX_PASSES = 2
    
    async def _submit_and_validate_construction(self, rejection_feedback: Optional[str] = None, retry_count: int = 0) -> Tuple[bool, Optional[str]]:
        """
        Submit and validate construction.
        
        Args:
            rejection_feedback: Feedback from a previous rejection to guide the model
            retry_count: Current retry count (for premature rejection retry loop)
            
        Returns:
            Tuple[bool, Optional[str]]: (accepted, rejection_reason)
                - accepted: True if submission accepted, False otherwise
                - rejection_reason: Rejection reason if rejected, None otherwise
        """
        self.current_mode = "construction"
        
        # FIX: Ensure placeholders exist before non-body phase construction
        # This prevents "old_string not found" errors when model tries to use placeholder as old_string
        # Placeholders can go missing during normal operation, not just crash recovery
        if self.autonomous_mode and self.autonomous_section_phase and self.autonomous_section_phase != "body":
            try:
                placeholders_added = await paper_memory.ensure_placeholders_exist()
                if placeholders_added:
                    logger.info(f"[{self.autonomous_section_phase.upper()} PHASE] Placeholders were missing and have been added to the paper")
            except Exception as e:
                logger.warning(f"Failed to ensure placeholders exist: {e}")
        
        # Single attempt - None means no work needed, not error
        section_phase = self.autonomous_section_phase if self.autonomous_mode else None
        
        # Load brainstorm content for retroactive corrections (autonomous mode only)
        brainstorm_content_for_submitter = None
        brainstorm_source_for_submitter = None
        if self.autonomous_mode and self._current_topic_id:
            try:
                from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
                brainstorm_content_for_submitter = await brainstorm_memory.get_database_content(self._current_topic_id, strip_proofs=True)
                brainstorm_source_for_submitter = f"brainstorm_{self._current_topic_id}.txt"
                if brainstorm_content_for_submitter:
                    logger.info(f"Loaded brainstorm content for retroactive corrections: {len(brainstorm_content_for_submitter)} chars")
            except Exception as e:
                logger.warning(f"Failed to load brainstorm for retroactive corrections: {e}")
        
        submission = None
        try:
            submission = await self.writer_submitter.submit_construction(
                is_first_portion=False,
                section_phase=section_phase,
                rejection_feedback=rejection_feedback,
                brainstorm_content=brainstorm_content_for_submitter,
                brainstorm_source_name=brainstorm_source_for_submitter
            )
        except (ValueError, OpenRouterInvalidResponseError) as e:
            label, reason_prefix = _classify_submitter_error(e)
            if label == "Context overflow":
                await self._handle_context_overflow(e, role_id="compiler_writer", mode="construction")
                return False, CONTEXT_OVERFLOW_STOP_MESSAGE
            logger.error(f"Construction {label.lower()}: {e}")
            self.construction_rejections += 1
            overflow_reason = f"{reason_prefix}: {e}"
            await compiler_rejection_log.add_rejection(
                CompilerValidationResult(
                    submission_id=str(uuid.uuid4()),
                    decision="reject",
                    reasoning=overflow_reason,
                    summary=overflow_reason[:750],
                    validation_stage="internal_error"
                ), "construction", ""
            )
            await self._broadcast("compiler_rejection", {
                "mode": "construction",
                "reasoning": overflow_reason
            })
            return False, overflow_reason
        
        if submission is None:
            logger.info("Construction not needed - paper is complete")
            self.construction_declines += 1
            
            # CRITICAL: Check for premature decline in autonomous mode
            # FIX: Check if CURRENT phase content exists (not NEXT phase)
            if self.autonomous_mode and self.autonomous_section_phase:
                current_paper = await paper_memory.get_paper()
                
                # Check if required sections exist using flexible patterns
                # CRITICAL: Must distinguish between real content and fake placeholders inserted by model
                def has_real_section_content(section_pattern: str, paper_text: str) -> bool:
                    """Check if section exists with real content, not just fake placeholder text."""
                    match = re.search(section_pattern, paper_text, re.IGNORECASE | re.MULTILINE)
                    if not match:
                        return False
                    
                    # Get sample for keyword detection (300 chars)
                    after_header_sample = paper_text[match.end():match.end() + 300].strip()
                    
                    # Get FULL content length to check if substantial
                    full_content_after = paper_text[match.end():].strip()
                    
                    fake_placeholder_indicators = [
                        'will be replaced',
                        'to be written', 
                        'placeholder',
                        'this placeholder'
                    ]
                    has_placeholder_keywords = any(phrase in after_header_sample.lower() for phrase in fake_placeholder_indicators)
                    
                    # Decision logic:
                    # - If FULL content >300 chars: REAL (substantial, keywords don't matter)
                    # - If <300 chars WITH keywords: FAKE (short placeholder-style)
                    # - If <300 chars NO keywords: REAL if >50 chars
                    if len(full_content_after) > 300:
                        return True  # Substantial content is always real
                    elif has_placeholder_keywords:
                        return False  # Short with keywords = fake
                    else:
                        return len(after_header_sample) > 50  # No keywords, check substance
                
                has_conclusion = has_real_section_content(
                    r"(?:^|\n)\s*(?:#+\s*)?(?:[IVXLCDM]+\.?\s*)?(?:Conclusion|Summary|Discussion|Final\s*Remarks|Concluding\s*Remarks)",
                    current_paper
                )
                has_introduction = has_real_section_content(
                    r"(?:^|\n)\s*(?:#+\s*)?(?:I\.?\s*)?Introduction",
                    current_paper
                )
                has_abstract = has_real_section_content(
                    r"(?:^|\n)\s*(?:#+\s*)?\*{0,2}Abstract\*{0,2}",
                    current_paper
                )
                
                logger.debug(f"Phase: {self.autonomous_section_phase}, has_conclusion={has_conclusion}, has_introduction={has_introduction}, has_abstract={has_abstract}")
                
                # Reject premature decline - check if CURRENT phase content is missing
                # The model is declining to write, but the current phase content doesn't exist yet
                premature = False
                rejection_reason = ""
                
                # FIX: Check CURRENT phase, not next phase
                # In conclusion phase: model declining means it thinks conclusion is done, but is it written?
                if self.autonomous_section_phase == "conclusion" and not has_conclusion:
                    premature = True
                    rejection_reason = "Cannot decline: Conclusion phase is active but no Conclusion section found in paper. You must write the Conclusion. Check CURRENT DOCUMENT PROGRESS - there is no Conclusion section present."
                elif self.autonomous_section_phase == "introduction" and not has_introduction:
                    premature = True
                    rejection_reason = "Cannot decline: Introduction phase is active but no Introduction section found in paper. You must write the Introduction. Check CURRENT DOCUMENT PROGRESS - there is no Introduction section present."
                elif self.autonomous_section_phase == "abstract" and not has_abstract:
                    premature = True
                    rejection_reason = "Cannot decline: Abstract phase is active but no Abstract found in paper. You must write the Abstract. Check CURRENT DOCUMENT PROGRESS - there is no Abstract present."
                
                if premature:
                    logger.warning(f"Rejecting premature decline: {rejection_reason}")
                    self.construction_rejections += 1
                    
                    # Log as rejection
                    rejection_result = CompilerValidationResult(
                        submission_id=str(uuid.uuid4()),
                        decision="reject",
                        reasoning=rejection_reason,
                        summary=rejection_reason[:750],
                        validation_stage="internal_error"  # Phase enforcement check
                    )
                    await compiler_rejection_log.add_rejection(rejection_result, "construction", "")
                    
                    await self._broadcast("compiler_rejection", {
                        "mode": "construction",
                        "reasoning": rejection_reason
                    })
                    
                    # RETRY WITH FEEDBACK - instead of just returning False
                    if retry_count < self.MAX_PREMATURE_RETRIES:
                        logger.info(f"Retrying with rejection feedback (attempt {retry_count + 1}/{self.MAX_PREMATURE_RETRIES})")
                        return await self._submit_and_validate_construction(
                            rejection_feedback=rejection_reason,
                            retry_count=retry_count + 1
                        )
                    else:
                        logger.error(f"Max retries ({self.MAX_PREMATURE_RETRIES}) reached for premature decline in {section_phase} phase")
                        return False, rejection_reason
            
            # If not premature, log as normal decline
            await compiler_rejection_log.add_decline("construction", "Paper already complete")
            
            await self._broadcast("compiler_decline", {
                "mode": "construction",
                "reasoning": "Paper already complete"
            })
            
            return False, None
        
        # CRITICAL: Handle completion signals directly without validation
        # When section_complete=True with empty content, this is a phase completion signal
        # NOT a construction submission that needs validation
        if submission.section_complete and not submission.content.strip():
            logger.info(f"Received phase completion signal for phase: {self.autonomous_section_phase}")
            
            # Handle phase transition directly
            if self.autonomous_mode and self.autonomous_section_phase:
                # FIX: Validate that CURRENT phase content actually exists before transitioning
                current_paper = await paper_memory.get_paper()
                current_phase = self.autonomous_section_phase
                
                # Check if current phase's content exists using flexible patterns
                phase_content_exists = True
                missing_section = ""
                
                if current_phase == "conclusion":
                    has_conclusion = bool(re.search(
                        r"(?:^|\n)\s*(?:#+\s*)?(?:[IVXLCDM]+\.?\s*)?(?:Conclusion|Summary|Discussion|Final\s*Remarks|Concluding\s*Remarks)",
                        current_paper, re.IGNORECASE | re.MULTILINE
                    ))
                    if not has_conclusion:
                        phase_content_exists = False
                        missing_section = "Conclusion"
                elif current_phase == "introduction":
                    has_introduction = bool(re.search(
                        r"(?:^|\n)\s*(?:#+\s*)?(?:I\.?\s*)?Introduction",
                        current_paper, re.IGNORECASE | re.MULTILINE
                    ))
                    if not has_introduction:
                        phase_content_exists = False
                        missing_section = "Introduction"
                elif current_phase == "abstract":
                    has_abstract = bool(re.search(
                        r"(?:^|\n)\s*(?:#+\s*)?\*{0,2}Abstract\*{0,2}",
                        current_paper, re.IGNORECASE | re.MULTILINE
                    ))
                    if not has_abstract:
                        phase_content_exists = False
                        missing_section = "Abstract"
                
                if not phase_content_exists:
                    # Reject the completion signal - content doesn't exist
                    rejection_reason = f"Cannot complete {current_phase} phase: No {missing_section} section found in paper. You must write {missing_section} content before marking phase complete. Check CURRENT DOCUMENT PROGRESS - there is no {missing_section} section present."
                    logger.warning(f"Rejecting empty phase completion: {rejection_reason}")
                    self.construction_rejections += 1
                    
                    rejection_result = CompilerValidationResult(
                        submission_id=str(uuid.uuid4()),
                        decision="reject",
                        reasoning=rejection_reason,
                        summary=rejection_reason[:750],
                        validation_stage="internal_error"  # Phase enforcement check
                    )
                    await compiler_rejection_log.add_rejection(rejection_result, "construction", "")
                    
                    await self._broadcast("compiler_rejection", {
                        "mode": "construction",
                        "reasoning": rejection_reason
                    })
                    
                    # RETRY WITH FEEDBACK - instead of just returning False
                    if retry_count < self.MAX_PREMATURE_RETRIES:
                        logger.info(f"Retrying with rejection feedback for empty completion (attempt {retry_count + 1}/{self.MAX_PREMATURE_RETRIES})")
                        return await self._submit_and_validate_construction(
                            rejection_feedback=rejection_reason,
                            retry_count=retry_count + 1
                        )
                    else:
                        logger.error(f"Max retries ({self.MAX_PREMATURE_RETRIES}) reached for empty phase completion in {current_phase} phase")
                        return False, rejection_reason
                
                # Content exists, proceed with phase transition
                paper_complete = await self._check_phase_transition(section_complete=True)
                
                if paper_complete:
                    logger.info("Paper fully complete!")
                    self.is_running = False
                    return True, None
                
                # Phase transitioned successfully - this is a success, not a rejection
                logger.info(f"Phase transition successful. New phase: {self.autonomous_section_phase}")
                self._track_submission_wolfram_calls(submission)
                
                await self._broadcast("phase_completion_signal", {
                    "previous_phase": submission.metadata.get("phase", "unknown"),
                    "new_phase": self.autonomous_section_phase,
                    "reasoning": submission.reasoning
                })
                
                return True, None  # Signal success - phase transitioned
            
            # In non-autonomous mode with section_complete, treat as decline
            logger.info("Completion signal in non-autonomous mode - treating as decline")
            await compiler_rejection_log.add_decline("construction", "Section marked complete")
            return False, None
        
        self.total_submissions += 1
        
        await self._broadcast("compiler_submission", {
            "mode": "construction",
            "submission_id": submission.submission_id
        })
        
        current_paper = await paper_memory.get_paper()
        current_outline = await outline_memory.get_outline()
        
        result = await self.validator.validate_submission(
            submission,
            current_paper=current_paper,
            current_outline=current_outline
        )
        
        if result.decision == "accept":
            # Update paper. For phase-section creation, prefer placeholder replacement
            # only when the submission actually targets that placeholder. If the
            # placeholder is already gone (resume/retry) or the model submitted a
            # validated edit against existing section text, apply the edit normally.
            section_phase = self.autonomous_section_phase if self.autonomous_mode else None
            placeholder_replaced = False
            phase_placeholder = {
                "conclusion": CONCLUSION_PLACEHOLDER,
                "introduction": INTRO_PLACEHOLDER,
                "abstract": ABSTRACT_PLACEHOLDER,
            }.get(section_phase)

            if phase_placeholder:
                old_string = (submission.old_string or "").strip()
                uses_placeholder_target = (
                    phase_placeholder in current_paper
                    and (
                        submission.operation == "full_content"
                        or (
                            submission.operation == "replace"
                            and (not old_string or old_string == phase_placeholder)
                        )
                    )
                )

                if uses_placeholder_target:
                    success = await paper_memory.replace_placeholder(phase_placeholder, submission.content)
                    if not success:
                        logger.error("%s placeholder was present but replacement failed.", section_phase.capitalize())
                        updated_paper = None  # Trigger rejection
                    else:
                        placeholder_replaced = True
                        updated_paper = await paper_memory.get_paper()
                else:
                    if phase_placeholder not in current_paper:
                        logger.info(
                            "%s placeholder not present; applying validated edit operation instead",
                            section_phase.capitalize(),
                        )
                    else:
                        logger.info(
                            "%s phase submission targets existing content; applying validated edit operation",
                            section_phase.capitalize(),
                        )
                    updated_paper = self._apply_edit(current_paper, submission)
            else:
                # Body section or no phase - use standard _apply_edit
                updated_paper = self._apply_edit(current_paper, submission)
            
            # Check if exact string match failed (only for _apply_edit cases)
            if updated_paper is None:
                logger.error(
                    f"Placement execution failed despite validator acceptance. "
                    f"Treating as rejection. Submission: {submission.submission_id}"
                )
                self.construction_rejections += 1
                
                # Create rejection result for placement failure
                rejection_result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Exact string match failed: old_string='{submission.old_string[:100]}...' not found or not unique in document",
                    summary="Exact string match failed - old_string not found or not unique",
                    placement_check=False,
                    validation_stage="pre-validation"  # Exact string match check
                )
                
                await compiler_rejection_log.add_rejection(rejection_result, "construction", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "construction",
                    "submission_id": submission.submission_id,
                    "reasoning": "Exact string match failed"
                })
                
                return False, rejection_result.reasoning
            
            # Only skip update_paper when replace_placeholder already saved.
            if not placeholder_replaced:
                await paper_memory.update_paper(updated_paper)
            
            self.construction_acceptances += 1
            self._track_submission_wolfram_calls(submission)
            
            await compiler_rejection_log.add_acceptance(
                submission.submission_id,
                "construction",
                submission.content[:500]
            )
            
            word_count = await paper_memory.get_word_count()
            
            await self._broadcast("compiler_acceptance", {
                "mode": "construction",
                "submission_id": submission.submission_id
            })
            
            await self._broadcast("paper_updated", {
                "word_count": word_count,
                "preview": submission.content[:500]
            })
            
            # Check for phase transitions in autonomous mode (explicit section_complete)
            if self.autonomous_mode and self.autonomous_section_phase:
                paper_complete = await self._check_phase_transition(section_complete=submission.section_complete)
                if paper_complete:
                    logger.info("Paper fully complete!")
                    # Signal completion to stop further construction
                    self.is_running = False
                    return True, None
            
            logger.info(f"Construction accepted ({word_count} words)")
            paper_accepted = True
            paper_rejection_reason = None
        else:
            self.construction_rejections += 1
            
            await compiler_rejection_log.add_rejection(result, "construction", submission.content)
            
            await self._broadcast("compiler_rejection", {
                "mode": "construction",
                "submission_id": submission.submission_id,
                "reasoning": result.reasoning
            })
            
            logger.info("Construction rejected")
            paper_accepted = False
            paper_rejection_reason = result.reasoning
        
        # ================================================================
        # RETROACTIVE BRAINSTORM OPERATION (independent from paper result)
        # ================================================================
        if submission.brainstorm_operation and self.autonomous_mode and hasattr(self, '_current_topic_id') and self._current_topic_id:
            await self._handle_brainstorm_retroactive_operation(submission.brainstorm_operation)
        
        return paper_accepted, paper_rejection_reason
    
    async def _handle_brainstorm_retroactive_operation(self, brainstorm_op) -> None:
        """
        Handle a retroactive brainstorm operation independently from the paper operation.
        Validates the operation using the compiler validator with brainstorm-only context,
        then applies if accepted and refreshes RAG.
        """
        from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
        
        topic_id = self._current_topic_id
        logger.info(f"Processing retroactive brainstorm {brainstorm_op.action} for topic {topic_id}")
        
        try:
            brainstorm_content = await brainstorm_memory.get_database_content(topic_id, strip_proofs=True)
            if not brainstorm_content:
                logger.warning(f"Brainstorm {topic_id} is empty, skipping retroactive operation")
                return
            
            denial_reason = self._get_lean_proof_retroactive_denial_reason(
                brainstorm_op,
                brainstorm_content,
            )
            if denial_reason:
                await self._reject_brainstorm_retroactive_operation(
                    brainstorm_op,
                    topic_id,
                    denial_reason,
                )
                return

            result = await self.validator.validate_brainstorm_operation(
                brainstorm_op, brainstorm_content
            )
            
            if result.decision == "accept":
                success = False
                action = brainstorm_op.action
                
                if action == "edit":
                    success = await brainstorm_memory.edit_submission(
                        topic_id, brainstorm_op.submission_number, brainstorm_op.new_content
                    )
                elif action == "delete":
                    success = await brainstorm_memory.remove_submission(
                        topic_id, brainstorm_op.submission_number
                    )
                elif action == "add":
                    new_num = await brainstorm_memory.add_submission_retroactive(
                        topic_id, brainstorm_op.new_content
                    )
                    success = new_num is not None
                
                if success:
                    logger.info(f"Retroactive brainstorm {action} accepted and applied for topic {topic_id}")
                    
                    # Refresh RAG with updated brainstorm content
                    try:
                        db_path = brainstorm_memory.get_database_path(topic_id)
                        from backend.aggregator.core.rag_manager import rag_manager
                        await rag_manager.add_document(
                            db_path,
                            chunk_sizes=[512],
                            is_user_file=True
                        )
                        logger.info("RAG refreshed with updated brainstorm content")
                    except Exception as e:
                        logger.error(f"Failed to refresh RAG after brainstorm {action}: {e}")
                    
                    await self._broadcast("brainstorm_retroactive_accepted", {
                        "action": action,
                        "topic_id": topic_id,
                        "submission_number": brainstorm_op.submission_number,
                    })
                else:
                    logger.error(f"Retroactive brainstorm {action} was validated but failed to apply")
            else:
                logger.info(f"Retroactive brainstorm {brainstorm_op.action} rejected: {result.reasoning[:200]}")
                await self._broadcast("brainstorm_retroactive_rejected", {
                    "action": brainstorm_op.action,
                    "topic_id": topic_id,
                    "reasoning": result.reasoning[:500],
                })
        except Exception as e:
            logger.error(f"Error handling retroactive brainstorm operation: {e}")

    @staticmethod
    def _parse_brainstorm_submission_content(brainstorm_content: str, submission_number: int) -> str:
        separator = "=" * 80
        parts = (brainstorm_content or "").split(separator)
        for index, part in enumerate(parts):
            if "SUBMISSION #" not in part:
                continue
            match = re.search(r"SUBMISSION #(\d+)\s*\|", part)
            if not match or int(match.group(1)) != submission_number:
                continue
            if index + 1 < len(parts):
                return parts[index + 1].strip()
        return ""

    @staticmethod
    def _is_lean_verified_brainstorm_content(content: str) -> bool:
        text = content or ""
        if BRAINSTORM_LEAN_PROOF_MARKER in text:
            return True
        lowered = text.lower()
        return (
            "lean 4 code:" in lowered
            and "lean verification: accepted" in lowered
            and "theorem statement:" in lowered
        )

    @staticmethod
    def _looks_like_lean_proof_annotation_attempt(brainstorm_op) -> bool:
        text = f"{getattr(brainstorm_op, 'new_content', '')}\n{getattr(brainstorm_op, 'reasoning', '')}".lower()
        if BRAINSTORM_LEAN_PROOF_MARKER.lower() in text:
            return True
        proof_reference = (
            "lean 4 proof" in text
            or "lean-verified proof" in text
            or "lean verified proof" in text
            or "verified proof" in text
            or "proof id:" in text
        )
        annotation_intent = any(
            phrase in text
            for phrase in (
                "add context",
                "additional context",
                "annotate",
                "annotation",
                "clarify proof",
                "context for proof",
                "context to proof",
                "update proof",
                "edit proof",
            )
        )
        return proof_reference and annotation_intent

    def _get_lean_proof_retroactive_denial_reason(self, brainstorm_op, brainstorm_content: str) -> str:
        action = getattr(brainstorm_op, "action", "")
        if action in {"edit", "delete"}:
            submission_number = getattr(brainstorm_op, "submission_number", None)
            if submission_number is None:
                return ""
            target_content = self._parse_brainstorm_submission_content(
                brainstorm_content,
                int(submission_number),
            )
            if self._is_lean_verified_brainstorm_content(target_content):
                return LEAN_PROOF_EDIT_DENIAL_REASON
        if action == "add" and self._looks_like_lean_proof_annotation_attempt(brainstorm_op):
            return LEAN_PROOF_EDIT_DENIAL_REASON
        return ""

    async def _reject_brainstorm_retroactive_operation(
        self,
        brainstorm_op,
        topic_id: str,
        reasoning: str,
    ) -> None:
        logger.info(
            "Retroactive brainstorm %s automatically rejected for protected Lean 4 proof mutation: %s",
            getattr(brainstorm_op, "action", ""),
            getattr(brainstorm_op, "submission_number", None),
        )
        result = CompilerValidationResult(
            submission_id=str(uuid.uuid4()),
            decision="reject",
            reasoning=reasoning,
            summary=reasoning[:750],
            json_valid=True,
            validation_stage="pre-validation",
        )
        await compiler_rejection_log.add_rejection(
            result,
            "brainstorm_retroactive",
            getattr(brainstorm_op, "new_content", "") or getattr(brainstorm_op, "reasoning", ""),
        )
        await self._broadcast("brainstorm_retroactive_rejected", {
            "action": getattr(brainstorm_op, "action", ""),
            "topic_id": topic_id,
            "submission_number": getattr(brainstorm_op, "submission_number", None),
            "reasoning": reasoning[:500],
            "automatic": True,
            "protected_lean_proof": True,
        })
    
    async def _submit_and_validate_outline_update(self) -> bool:
        """Submit and validate outline update. Returns True if accepted."""
        self.current_mode = "outline_update"
        
        try:
            submission = await self.writer_submitter.submit_outline_update()
        except ValueError as e:
            label, _ = _classify_submitter_error(e)
            if label == "Context overflow":
                await self._handle_context_overflow(e, role_id="compiler_writer", mode="outline_update")
                return False
            logger.error(f"Outline update submission failed with error: {e} - skipping this cycle")
            return False
        except Exception as e:
            logger.error(f"Outline update submission failed with error: {e} - skipping this cycle")
            return False
        
        if submission is None:
            logger.info("No outline update needed")
            self.outline_declines += 1
            await compiler_rejection_log.add_decline("outline_update", "Outline already complete")
            
            await self._broadcast("compiler_decline", {
                "mode": "outline_update",
                "reasoning": "Outline already complete"
            })
            
            return False
        
        self.total_submissions += 1
        
        await self._broadcast("compiler_submission", {
            "mode": "outline_update",
            "submission_id": submission.submission_id
        })
        
        current_paper = await paper_memory.get_paper()
        current_outline = await outline_memory.get_outline()
        
        result = await self.validator.validate_submission(
            submission,
            current_paper=current_paper,
            current_outline=current_outline
        )
        
        if result.decision == "accept":
            # Update outline (use _apply_edit for line-based additions)
            updated_outline = self._apply_edit_to_outline(current_outline, submission)
            
            # Check if exact string match failed
            if updated_outline is None:
                logger.error(
                    f"Outline exact string match failed despite validator acceptance. "
                    f"Treating as rejection. Submission: {submission.submission_id}"
                )
                self.outline_rejections += 1
                
                # Create rejection result for outline placement failure
                rejection_result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Outline exact string match failed: old_string='{submission.old_string[:100]}...' not found or not unique in outline",
                    summary="Outline exact string match failed - old_string not found or not unique",
                    placement_check=False,
                    validation_stage="pre-validation"  # Exact string match check
                )
                
                await compiler_rejection_log.add_rejection(rejection_result, "outline_update", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "outline_update",
                    "submission_id": submission.submission_id,
                    "reasoning": "Outline exact string match failed"
                })
                
                return False
            
            await outline_memory.update_outline(updated_outline)
            
            self.outline_acceptances += 1
            
            await compiler_rejection_log.add_acceptance(
                submission.submission_id,
                "outline_update",
                submission.content[:500]
            )
            
            await self._broadcast("compiler_acceptance", {
                "mode": "outline_update",
                "submission_id": submission.submission_id
            })
            
            await self._broadcast("outline_updated", {
                "outline": submission.content
            })
            
            logger.info("Outline update accepted")
            return True
        else:
            self.outline_rejections += 1
            
            await compiler_rejection_log.add_rejection(result, "outline_update", submission.content)
            
            await self._broadcast("compiler_rejection", {
                "mode": "outline_update",
                "submission_id": submission.submission_id,
                "reasoning": result.reasoning
            })
            
            logger.info("Outline update rejected")
            return False
    
    async def _submit_and_validate_review(self, review_focus: str = "general") -> bool:
        """Submit and validate review. Returns True if accepted."""
        self.current_mode = "review"
        review_label = "empirical red-team review" if review_focus == "empirical_red_team" else "review"
        
        submission = None
        try:
            submission = await self.writer_submitter.submit_review(review_focus=review_focus)
        except (ValueError, OpenRouterInvalidResponseError) as e:
            label, reason_prefix = _classify_submitter_error(e)
            if label == "Context overflow":
                await self._handle_context_overflow(e, role_id="compiler_writer", mode="review")
                return False
            logger.error(f"{review_label.capitalize()} {label.lower()}: {e}")
            self.review_declines += 1
            await compiler_rejection_log.add_decline("review", f"{reason_prefix}: {e}")
            await self._broadcast("compiler_decline", {
                "mode": "review",
                "review_focus": review_focus,
                "reasoning": f"{reason_prefix}: {e}"
            })
            return False
        
        if submission is None:
            logger.info(f"No {review_label} edit needed")
            self.review_declines += 1
            decline_reason = (
                "No fabricated experiments or unsupported metrics found"
                if review_focus == "empirical_red_team"
                else "No errors or improvements needed"
            )
            await compiler_rejection_log.add_decline("review", decline_reason)
            
            await self._broadcast("compiler_decline", {
                "mode": "review",
                "review_focus": review_focus,
                "reasoning": decline_reason
            })
            
            return False
        
        self.total_submissions += 1
        
        # Check for minuscule edit
        if submission.metadata.get("is_minuscule", False):
            self.minuscule_edit_count += 1
        
        await self._broadcast("compiler_submission", {
            "mode": "review",
            "submission_id": submission.submission_id,
            "review_focus": review_focus
        })
        
        current_paper = await paper_memory.get_paper()
        current_outline = await outline_memory.get_outline()
        
        result = await self.validator.validate_submission(
            submission,
            current_paper=current_paper,
            current_outline=current_outline
        )
        
        if result.decision == "accept":
            # Apply edit
            updated_paper = self._apply_edit(current_paper, submission)
            
            # Check if exact string match failed
            if updated_paper is None:
                logger.error(
                    f"Placement execution failed despite validator acceptance. "
                    f"Treating as rejection. Submission: {submission.submission_id}"
                )
                self.review_rejections += 1
                
                # Create rejection result for placement failure
                rejection_result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Exact string match failed: old_string='{submission.old_string[:100]}...' not found or not unique in document",
                    summary="Exact string match failed - old_string not found or not unique",
                    placement_check=False,
                    validation_stage="pre-validation"  # Exact string match check
                )
                
                await compiler_rejection_log.add_rejection(rejection_result, "review", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "review",
                    "submission_id": submission.submission_id,
                    "review_focus": review_focus,
                    "reasoning": "Exact string match failed"
                })
                
                return False
            
            await paper_memory.update_paper(updated_paper)
            
            self.review_acceptances += 1
            
            await compiler_rejection_log.add_acceptance(
                submission.submission_id,
                "review",
                submission.content[:500]
            )
            
            word_count = await paper_memory.get_word_count()
            
            await self._broadcast("compiler_acceptance", {
                "mode": "review",
                "submission_id": submission.submission_id,
                "review_focus": review_focus
            })
            
            await self._broadcast("paper_updated", {
                "word_count": word_count,
                "preview": updated_paper[:500]
            })
            
            logger.info(f"{review_label.capitalize()} edit accepted ({word_count} words)")
            return True
        else:
            self.review_rejections += 1
            
            await compiler_rejection_log.add_rejection(result, "review", submission.content)
            
            await self._broadcast("compiler_rejection", {
                "mode": "review",
                "submission_id": submission.submission_id,
                "review_focus": review_focus,
                "reasoning": result.reasoning
            })
            
            logger.info(f"{review_label.capitalize()} edit rejected")
            return False

    async def _run_pre_abstract_red_team_review(self) -> None:
        """Run a dedicated empirical-provenance red-team pass before abstract writing."""
        logger.info("=" * 80)
        logger.info("STARTING PRE-ABSTRACT EMPIRICAL RED-TEAM REVIEW")
        logger.info("=" * 80)

        await self._broadcast("empirical_red_team_started", {
            "phase": "pre_abstract",
            "max_passes": self.PRE_ABSTRACT_RED_TEAM_MAX_PASSES
        })

        edits_applied = 0
        passes_run = 0

        for _ in range(self.PRE_ABSTRACT_RED_TEAM_MAX_PASSES):
            if not self.is_running:
                break

            passes_run += 1
            accepted = await self._submit_and_validate_review(review_focus="empirical_red_team")
            if not accepted:
                break
            edits_applied += 1

        await self._broadcast("empirical_red_team_complete", {
            "phase": "pre_abstract",
            "passes_run": passes_run,
            "edits_applied": edits_applied
        })

        logger.info(
            f"Pre-abstract empirical red-team review complete "
            f"(passes={passes_run}, edits_applied={edits_applied})"
        )
    
    async def _submit_and_validate_rigor(self) -> bool:
        """Run one rigor cycle.

        New Lean-4-verified-theorem flow:
          1. If Lean 4 is disabled in config, decline immediately (no work).
          2. Submitter does discovery + 5 Lean attempts + novelty + store.
             If it returns None, decline and end the rigor cycle.
          3. Coordinator owns the 2-attempt validator placement loop.
          4. If both placement attempts reject (or the submitter never
             produced a legal attempt-1), the theorem is routed to the
             Theorems Appendix (its Lean 4 verification is preserved).
             Counts as a rigor_acceptance per the build plan.

        Returns True to signal "continue the rigor loop" (a theorem landed
        somewhere). Returns False on decline (no theorem to propose / Lean
        5-attempt failure / Lean 4 disabled) so the outer loop ends this
        rigor cycle.
        """
        self.current_mode = "rigor"

        # Hard guard: Lean 4 disabled system-wide means rigor mode has no work.
        if not self.allow_mathematical_proofs or not system_config.lean4_enabled:
            reason = (
                "Mathematical proof outputs are disabled for this run"
                if not self.allow_mathematical_proofs
                else "Lean 4 is disabled in system configuration"
            )
            logger.info("Rigor loop: %s; declining cycle", reason)
            self.rigor_declines += 1
            await compiler_rejection_log.add_decline(
                "rigor", reason
            )
            await self._broadcast(
                "compiler_decline",
                {"mode": "rigor", "reasoning": reason},
            )
            return False

        try:
            if self.high_param_submitter:
                source_context, source_label = await self._load_rigor_source_material_context()
                self.high_param_submitter.set_source_material_context(source_context, source_label)
            lean_result = await self.high_param_submitter.submit_rigor_lean_theorem()
        except ValueError as exc:
            if (
                "mandatory full source context" in str(exc).lower()
                or _classify_submitter_error(exc)[0] == "Context overflow"
                or _is_rigor_model_call_failure(exc)
            ):
                raise
            logger.error(f"Rigor lean flow error: {exc}")
            self.rigor_declines += 1
            await compiler_rejection_log.add_decline("rigor", f"LLM error: {exc}")
            await self._broadcast(
                "compiler_decline", {"mode": "rigor", "reasoning": f"LLM error: {exc}"}
            )
            return False
        except Exception as exc:
            if _is_rigor_model_call_failure(exc):
                raise
            logger.error(f"Rigor lean flow raised: {exc}", exc_info=True)
            self.rigor_declines += 1
            await compiler_rejection_log.add_decline(
                "rigor", f"Internal error: {exc}"
            )
            await self._broadcast(
                "compiler_decline",
                {"mode": "rigor", "reasoning": f"Internal error: {exc}"},
            )
            return False

        if lean_result is None:
            logger.info("Rigor loop: no theorem attempted this cycle (decline)")
            self.rigor_declines += 1
            await compiler_rejection_log.add_decline(
                "rigor",
                "No theorem to formalize or 5 Lean 4 attempts failed",
            )
            await self._broadcast(
                "compiler_decline",
                {
                    "mode": "rigor",
                    "reasoning": "No theorem to formalize or 5 Lean 4 attempts failed",
                },
            )
            return False

        # At this point a Lean-4-verified proof exists in proof_database.
        # The submitter may or may not have produced an attempt-1 placement.
        return await self._place_or_appendix_fallback(lean_result)

    async def _place_or_appendix_fallback(self, lean_result) -> bool:
        """Drive the 2-attempt placement validator loop.

        On explicit appendix-only discovery, double rejection, or when the
        submitter never produced a legal attempt, the theorem is appended to
        the Theorems Appendix and the cycle is counted as a rigor_acceptance.
        """
        from backend.compiler.agents.high_param_submitter import (
            format_theorem_appendix_entry,
        )

        submission = lean_result.initial_placement_submission
        validator_feedback = ""
        requested_appendix_only = (
            getattr(lean_result, "placement_preference", "") == "appendix_only"
            or (getattr(lean_result, "metadata", {}) or {}).get("placement_preference")
            == "appendix_only"
        )
        appendix_outcome = (
            "appendix_requested" if requested_appendix_only else "appendix_fallback"
        )

        for placement_attempt in (1, 2):
            if submission is None:
                route_reason = (
                    "discovery requested appendix-only placement"
                    if requested_appendix_only
                    else "submitter returned no placement submission"
                )
                logger.info(
                    "Rigor placement attempt %s: %s; routing directly to appendix",
                    placement_attempt,
                    route_reason,
                )
                break

            self.total_submissions += 1
            await self._broadcast(
                "compiler_submission",
                {
                    "mode": "rigor",
                    "submission_id": submission.submission_id,
                    "lean_proof_id": lean_result.proof_id,
                    "placement_attempt": placement_attempt,
                },
            )

            current_paper = await paper_memory.get_paper()
            current_outline = await outline_memory.get_outline()

            result = await self.validator.validate_submission(
                submission,
                current_paper=current_paper,
                current_outline=current_outline,
            )

            if result.decision == "accept":
                updated_paper = self._apply_edit(current_paper, submission)
                if updated_paper is None:
                    logger.error(
                        "Rigor placement attempt %s: exact-string apply failed after "
                        "validator acceptance for submission %s",
                        placement_attempt,
                        submission.submission_id,
                    )
                    # Treat apply failure as a placement rejection for retry
                    validator_feedback = (
                        f"Exact-string match failed when applying your edit: "
                        f"old_string='{(submission.old_string or '')[:120]}...' was not "
                        "found or not unique in the current paper. Pick a more "
                        "specific anchor."
                    )
                    rejection_result = CompilerValidationResult(
                        submission_id=submission.submission_id,
                        decision="reject",
                        reasoning=validator_feedback,
                        summary=validator_feedback[:750],
                        placement_check=False,
                        validation_stage="pre-validation",
                    )
                    await compiler_rejection_log.add_rejection(
                        rejection_result, "rigor", submission.content
                    )
                    await self._broadcast(
                        "compiler_rejection",
                        {
                            "mode": "rigor",
                            "submission_id": submission.submission_id,
                            "reasoning": validator_feedback,
                            "placement_attempt": placement_attempt,
                        },
                    )
                    self.rigor_rejections += 1
                    if placement_attempt == 1:
                        submission = await self.high_param_submitter.submit_rigor_placement_retry(
                            lean_result, validator_feedback
                        )
                    continue

                # Success: inline placement accepted + applied.
                await paper_memory.update_paper(updated_paper)

                # Also drop a short cross-reference stub into the appendix so
                # the full Lean proof is preserved and easy to look up.
                appendix_stub = format_theorem_appendix_entry(
                    proof_id=lean_result.proof_id,
                    theorem_statement=lean_result.theorem_statement,
                    lean_code=lean_result.lean_code,
                    is_novel=lean_result.is_novel,
                    theorem_name=lean_result.theorem_name,
                    novelty_tier=lean_result.novelty_tier,
                    placement_outcome="inline",
                )
                try:
                    stub_appended = await paper_memory.append_to_theorems_appendix(
                        appendix_stub
                    )
                    if not stub_appended:
                        await paper_memory.ensure_markers_intact()
                        stub_appended = await paper_memory.append_to_theorems_appendix(
                            appendix_stub
                        )
                    if not stub_appended:
                        logger.error(
                            "Inline theorem persisted, but appendix stub could not "
                            "be written after marker repair"
                        )
                        await self._broadcast(
                            "compiler_rejection",
                            {
                                "mode": "rigor",
                                "submission_id": submission.submission_id,
                                "reasoning": (
                                    "The theorem was placed inline, but its Lean "
                                    "appendix entry could not be persisted."
                                ),
                                "placement_outcome": "inline_appendix_persistence_failed",
                                "lean_proof_id": lean_result.proof_id,
                            },
                        )
                except Exception as exc:
                    logger.warning(
                        "Inline-placed theorem appendix stub append failed (non-fatal): %s",
                        exc,
                    )

                self.rigor_acceptances += 1
                await compiler_rejection_log.add_acceptance(
                    submission.submission_id,
                    "rigor",
                    submission.content[:500],
                )

                word_count = await paper_memory.get_word_count()
                await self._broadcast(
                    "compiler_acceptance",
                    {
                        "mode": "rigor",
                        "submission_id": submission.submission_id,
                        "placement_outcome": "inline",
                        "lean_proof_id": lean_result.proof_id,
                        "is_novel": lean_result.is_novel,
                        "placement_attempt": placement_attempt,
                    },
                )
                await self._broadcast(
                    "paper_updated",
                    {"word_count": word_count, "preview": updated_paper[:500]},
                )
                logger.info(
                    "Rigor theorem %s placed inline on attempt %s (%s words)",
                    lean_result.proof_id,
                    placement_attempt,
                    word_count,
                )
                return True

            # Validator rejected this placement attempt
            self.rigor_rejections += 1
            validator_feedback = result.reasoning or "Placement rejected without reason"
            await compiler_rejection_log.add_rejection(
                result, "rigor", submission.content
            )
            await self._broadcast(
                "compiler_rejection",
                {
                    "mode": "rigor",
                    "submission_id": submission.submission_id,
                    "reasoning": result.reasoning,
                    "placement_attempt": placement_attempt,
                },
            )
            logger.info(
                "Rigor placement attempt %s rejected: %s",
                placement_attempt,
                (result.reasoning or "")[:160],
            )

            if placement_attempt == 1:
                submission = await self.high_param_submitter.submit_rigor_placement_retry(
                    lean_result, validator_feedback
                )

        # Appendix storage: either explicitly requested by discovery or used
        # as fallback when inline placement failed / was impossible. The math
        # is already Lean-verified, but paper-placement success is reported only
        # after the appendix entry is durably written.
        appendix_entry = format_theorem_appendix_entry(
            proof_id=lean_result.proof_id,
            theorem_statement=lean_result.theorem_statement,
            lean_code=lean_result.lean_code,
            is_novel=lean_result.is_novel,
            theorem_name=lean_result.theorem_name,
            novelty_tier=lean_result.novelty_tier,
            placement_outcome=appendix_outcome,
        )
        appended = await paper_memory.append_to_theorems_appendix(appendix_entry)
        if not appended:
            # Paper markers might be missing - try one repair pass then retry.
            logger.warning(
                "Appendix append returned False; attempting marker repair before retry"
            )
            await paper_memory.ensure_markers_intact()
            appended = await paper_memory.append_to_theorems_appendix(appendix_entry)
            if not appended:
                logger.error(
                    "Appendix append still failed after marker repair; preserving "
                    "the canonical proof record without claiming paper persistence"
                )
                await self._broadcast(
                    "compiler_rejection",
                    {
                        "mode": "rigor",
                        "submission_id": (
                            lean_result.initial_placement_submission.submission_id
                            if lean_result.initial_placement_submission
                            else f"rigor_{appendix_outcome}_{lean_result.proof_id}"
                        ),
                        "reasoning": (
                            "Lean verification succeeded, but the theorem could not "
                            "be persisted to the paper appendix after marker repair."
                        ),
                        "placement_outcome": "appendix_persistence_failed",
                        "lean_proof_id": lean_result.proof_id,
                    },
                )
                return False

        self.rigor_acceptances += 1
        acceptance_submission_id = (
            lean_result.initial_placement_submission.submission_id
            if lean_result.initial_placement_submission
            else f"rigor_{appendix_outcome}_{lean_result.proof_id}"
        )
        await compiler_rejection_log.add_acceptance(
            acceptance_submission_id,
            "rigor",
            appendix_entry[:500],
        )
        word_count = await paper_memory.get_word_count()
        await self._broadcast(
            "compiler_acceptance",
            {
                "mode": "rigor",
                "submission_id": acceptance_submission_id,
                "placement_outcome": appendix_outcome,
                "lean_proof_id": lean_result.proof_id,
                "is_novel": lean_result.is_novel,
            },
        )
        await self._broadcast("paper_updated", {"word_count": word_count})
        logger.info(
            "Rigor theorem %s stored in Theorems Appendix (%s)",
            lean_result.proof_id,
            appendix_outcome,
        )
        return True

    
    def _apply_edit_to_outline(self, current_outline: str, submission: CompilerSubmission) -> Optional[str]:
        """
        Apply edit to outline using exact string matching.
        
        Uses the submission's operation, old_string, and new_string fields:
        - operation="full_content": Replace entire outline with new_string (for outline_create)
        - operation="replace": Find exact old_string, replace with new_string
        - operation="insert_after": Find exact old_string (anchor), insert new_string after it
        - operation="delete": Find exact old_string, remove it
        
        Returns:
            Updated outline string, or None if operation fails
        """
        # Strip outline anchor from new_string (memory will re-add it)
        # Note: Use rstrip() to preserve leading indentation for subsections
        new_content = submission.new_string.replace(OUTLINE_ANCHOR, "").rstrip()
        old_content = submission.old_string.strip()
        operation = submission.operation
        
        logger.info(f"_apply_edit_to_outline: operation={operation}, mode={submission.mode}")
        
        # For outline_create, always use full_content behavior (replaces entire outline)
        if submission.mode == "outline_create" or operation == "full_content":
            if not new_content:
                logger.error("outline_create/full_content requires new_string content")
                return None
            return new_content
        
        # Handle empty outline
        if not current_outline or not current_outline.strip():
            return new_content
        
        try:
            # Check old_string exists and is unique (with Unicode hyphen normalization)
            if operation in ("replace", "insert_after", "delete"):
                if not old_content:
                    logger.error(f"Operation '{operation}' requires old_string to be non-empty")
                    return None
                
                # Use normalized comparison to handle Unicode hyphen variants
                pos, actual_old_content = find_with_normalized_hyphens(old_content, current_outline)
                if pos < 0:
                    logger.error(f"_apply_edit_to_outline FAILED: old_string not found in outline")
                    logger.error(f"   old_string length: {len(old_content)} chars")
                    logger.error(f"   old_string preview: {repr(old_content[:150])}")
                    logger.error(f"   outline length: {len(current_outline)} chars")
                    # Deep diagnostic already logged by find_with_normalized_hyphens
                    return None
                
                # Update old_content to actual outline text (may differ in Unicode hyphens)
                if actual_old_content != old_content:
                    logger.info(f"Unicode hyphen normalization applied in _apply_edit_to_outline")
                    old_content = actual_old_content
                
                # Check uniqueness using normalized comparison
                normalized_outline = normalize_unicode_hyphens(current_outline)
                normalized_old = normalize_unicode_hyphens(old_content)
                count = normalized_outline.count(normalized_old)
                if count > 1:
                    logger.error(f"old_string appears {count} times in outline. Provide more context to make it unique.")
                    logger.debug(f"old_string (first 200 chars): {old_content[:200]}")
                    return None
            
            # OPERATION: replace
            if operation == "replace":
                if not new_content:
                    # Replace with empty = delete
                    return current_outline.replace(old_content, "")
                
                result = current_outline.replace(old_content, new_content, 1)
                logger.info(f"Outline replace: replaced {len(old_content)} chars with {len(new_content)} chars")
                return result
            
            # OPERATION: insert_after
            elif operation == "insert_after":
                if not new_content:
                    logger.error("insert_after operation requires new_string content")
                    return None
                
                pos = current_outline.find(old_content)
                insert_pos = pos + len(old_content)
                
                # Insert with newline (outline uses single newlines between entries)
                result = (current_outline[:insert_pos].rstrip() + 
                         "\n" + new_content + "\n" + 
                         current_outline[insert_pos:].lstrip())
                logger.info(f"Outline insert after: inserted {len(new_content)} chars")
                return result
            
            # OPERATION: delete
            elif operation == "delete":
                result = current_outline.replace(old_content, "", 1)
                
                # Clean up multiple newlines
                while "\n\n\n" in result:
                    result = result.replace("\n\n\n", "\n\n")
                
                logger.info(f"Outline delete: removed {len(old_content)} chars")
                return result
            
            else:
                logger.error(f"Unknown outline operation: {operation}")
                return None
                
        except Exception as e:
            logger.error(f"Error in _apply_edit_to_outline: {e}")
            return None
    
    def _apply_edit(self, current_paper: str, submission: CompilerSubmission) -> Optional[str]:
        """
        Apply edit to paper using exact string matching.
        
        Uses the submission's operation, old_string, and new_string fields:
        - operation="replace": Find exact old_string, replace with new_string
        - operation="insert_after": Find exact old_string (anchor), insert new_string after it
        - operation="delete": Find exact old_string, remove it
        - operation="full_content": Replace entire paper with new_string (for first content)
        
        Returns:
            Updated paper string, or None if operation fails (triggers rejection)
        """
        # Strip paper anchor from new_string (memory will re-add it)
        new_content = submission.new_string.replace(PAPER_ANCHOR, "").strip()
        old_content = submission.old_string.strip()
        operation = submission.operation
        
        logger.info(f"_apply_edit: operation={operation}, old_string_len={len(old_content)}, new_string_len={len(new_content)}")
        
        # AUTO-CORRECTION: When paper is empty but LLM used wrong operation, 
        # automatically convert to full_content (prevents unnecessary rejections)
        paper_is_empty = not current_paper or not current_paper.strip()
        if paper_is_empty and operation in ("replace", "insert_after"):
            logger.warning(
                f"AUTO-CORRECTING: Paper is empty but operation='{operation}'. "
                f"Converting to 'full_content' operation automatically."
            )
            operation = "full_content"
            # Use new_string as the content, ignore old_string (which won't match anything anyway)
        
        # Handle empty paper or full_content operation (first insertion)
        if operation == "full_content" or paper_is_empty:
            if not new_content:
                logger.error("full_content operation requires new_string content")
                return None
            return new_content
        
        try:
            # Check old_string exists and is unique (with Unicode hyphen normalization)
            if operation in ("replace", "insert_after", "delete"):
                if not old_content:
                    logger.error(f"Operation '{operation}' requires old_string to be non-empty")
                    return None
                
                # Use normalized comparison to handle Unicode hyphen variants
                pos, actual_old_content = find_with_normalized_hyphens(old_content, current_paper)
                if pos < 0:
                    logger.error(f"_apply_edit FAILED: old_string not found in document")
                    logger.error(f"   old_string length: {len(old_content)} chars")
                    logger.error(f"   old_string preview: {repr(old_content[:150])}")
                    logger.error(f"   document length: {len(current_paper)} chars")
                    # Deep diagnostic already logged by find_with_normalized_hyphens
                    return None
                
                # Update old_content to actual document text (may differ in Unicode hyphens)
                if actual_old_content != old_content:
                    logger.info(f"Unicode hyphen normalization applied in _apply_edit")
                    old_content = actual_old_content
                
                # Check uniqueness using normalized comparison
                normalized_paper = normalize_unicode_hyphens(current_paper)
                normalized_old = normalize_unicode_hyphens(old_content)
                count = normalized_paper.count(normalized_old)
                if count > 1:
                    logger.error(f"old_string appears {count} times in document. Provide more context to make it unique.")
                    logger.debug(f"old_string (first 200 chars): {old_content[:200]}")
                    return None
            
            # OPERATION: replace
            if operation == "replace":
                if not new_content:
                    # Replace with empty = delete
                    logger.info(f"Replace with empty new_string - treating as delete")
                    return current_paper.replace(old_content, "")
                
                # CRITICAL: PLACEHOLDER BOUNDARY ENFORCEMENT FOR REPLACE
                # Ensure replace operation doesn't violate placeholder boundaries
                conclusion_pos = current_paper.find(CONCLUSION_PLACEHOLDER)
                old_content_pos = current_paper.find(old_content)
                
                if conclusion_pos != -1 and old_content_pos != -1:
                    # Check if we're trying to replace something that includes or is after the placeholder
                    if old_content_pos >= conclusion_pos and CONCLUSION_PLACEHOLDER not in old_content:
                        # Replacing content after the placeholder - only allowed for placeholder replacement itself
                        logger.warning(
                            f"Replace operation targets content after CONCLUSION_PLACEHOLDER. "
                            f"This may be intentional for placeholder replacement. Proceeding with caution."
                        )
                
                result = current_paper.replace(old_content, new_content, 1)
                logger.info(f"Replace: replaced {len(old_content)} chars with {len(new_content)} chars")
                return result
            
            # OPERATION: insert_after
            elif operation == "insert_after":
                if not new_content:
                    logger.error("insert_after operation requires new_string content")
                    return None
                
                pos = current_paper.find(old_content)
                insert_pos = pos + len(old_content)
                
                # CRITICAL: CONCLUSION_PLACEHOLDER BOUNDARY ENFORCEMENT
                # Body content must NEVER be inserted after the conclusion placeholder.
                # If the anchor is after the placeholder, or if the insertion would place
                # content after the placeholder, we must reject or relocate.
                conclusion_pos = current_paper.find(CONCLUSION_PLACEHOLDER)
                
                if conclusion_pos != -1:
                    # Conclusion placeholder exists - enforce boundary
                    if insert_pos > conclusion_pos:
                        # AUTO-CORRECTION: Anchor point is AFTER the conclusion placeholder.
                        # Instead of rejecting (which causes infinite loops), automatically 
                        # place the content just BEFORE the conclusion placeholder.
                        # This matches the auto-correction pattern for empty paper operations.
                        logger.warning(
                            f"AUTO-CORRECTING BOUNDARY VIOLATION: insert_after anchor is after CONCLUSION_PLACEHOLDER. "
                            f"Relocating insertion to just before the placeholder. "
                            f"Original anchor position: {insert_pos}, Conclusion placeholder position: {conclusion_pos}"
                        )
                        # Relocate insertion point to just before conclusion placeholder
                        insert_pos = conclusion_pos
                    
                    # Insert with proper spacing, ensuring content stays BEFORE placeholder
                    result = (current_paper[:insert_pos].rstrip() + 
                             "\n\n" + new_content + "\n\n" + 
                             current_paper[insert_pos:].lstrip())
                    
                    # VERIFY: Conclusion placeholder should still be AFTER all inserted content
                    new_conclusion_pos = result.find(CONCLUSION_PLACEHOLDER)
                    new_content_end = result.find(new_content) + len(new_content)
                    
                    if new_conclusion_pos != -1 and new_content_end > new_conclusion_pos:
                        logger.error(
                            f"PLACEHOLDER BOUNDARY VIOLATION: Insertion would place content after "
                            f"CONCLUSION_PLACEHOLDER. Content end: {new_content_end}, Placeholder: {new_conclusion_pos}"
                        )
                        return None
                    
                    logger.info(f"Insert after: inserted {len(new_content)} chars (conclusion boundary preserved)")
                    return result
                else:
                    # No conclusion placeholder - standard insert
                    result = (current_paper[:insert_pos].rstrip() + 
                             "\n\n" + new_content + "\n\n" + 
                             current_paper[insert_pos:].lstrip())
                    logger.info(f"Insert after: inserted {len(new_content)} chars after anchor")
                    return result
            
            # OPERATION: delete
            elif operation == "delete":
                result = current_paper.replace(old_content, "", 1)
                
                # Clean up double newlines that may result from deletion
                while "\n\n\n" in result:
                    result = result.replace("\n\n\n", "\n\n")
                
                logger.info(f"Delete: removed {len(old_content)} chars")
                return result
            
            else:
                logger.error(f"Unknown operation: {operation}")
                return None
                
        except Exception as e:
            logger.error(f"Error in _apply_edit: {e}")
            return None
    
    # =========================================================================
    # CRITIQUE PHASE WORKFLOW (POST-BODY PEER REVIEW)
    # =========================================================================
    
    async def _start_critique_phase(self) -> None:
        """
        Start critique aggregation sub-workflow.
        Runs after body is complete, before conclusion.
        Uses simple generate-validate loop (similar to aggregator workflow).
        """
        logger.info("=" * 80)
        logger.info("STARTING CRITIQUE PHASE")
        logger.info("=" * 80)
        
        self.in_critique_phase = True
        self.critique_acceptances = 0
        
        # Initialize critique memory
        paper_id = f"paper_v{self.paper_version}"
        critique_memory.initialize(paper_id)
        await critique_memory.clear()

        logger.info(f"Critique memory initialized for {paper_id}")
        
        if not self.high_param_submitter:
            logger.error("Cannot start critique phase: Rigor & Proofs submitter is not initialized")
            await self._end_critique_phase(self_review_appended=False)
            return

        self.critique_submitter = self.high_param_submitter
        
        # Clear rejection feedback from previous critique phases (fresh start)
        await self.critique_submitter.reset_critique_rejection_memory()
        logger.info("Cleared critique rejection feedback for fresh start")
        
        logger.info(
            "Critique generation using Rigor & Proofs submitter model: %s",
            self.critique_submitter.model_name,
        )
        
        # Configure API client manager for critique validator (uses same settings as compiler_validator)
        api_client_manager.configure_role(
            role_id="critique_validator",
            config=ModelConfig(
                provider=self.validator_provider,
                model_id=self.validator_model,
                openrouter_provider=self.validator_openrouter_provider,
                openrouter_reasoning_effort=self.validator_openrouter_reasoning_effort,
                lm_studio_fallback_id=self.validator_lm_studio_fallback,
                context_window=self.validator_context_window,
                max_output_tokens=self.validator_max_tokens,
                supercharge_enabled=self.validator_supercharge_enabled
            )
        )
        
        # Configure API client manager for critique cleanup (uses same settings as compiler_validator)
        api_client_manager.configure_role(
            role_id="critique_cleanup",
            config=ModelConfig(
                provider=self.validator_provider,
                model_id=self.validator_model,
                openrouter_provider=self.validator_openrouter_provider,
                openrouter_reasoning_effort=self.validator_openrouter_reasoning_effort,
                lm_studio_fallback_id=self.validator_lm_studio_fallback,
                context_window=self.validator_context_window,
                max_output_tokens=self.validator_max_tokens,
                supercharge_enabled=self.validator_supercharge_enabled
            )
        )
        
        # Broadcast critique phase started
        await self._broadcast("critique_phase_started", {
            "paper_version": self.paper_version,
            "target_critiques": CRITIQUE_ATTEMPT_TARGET
        })
        
        # Start critique aggregation loop
        await self._run_critique_aggregation()

    async def _get_reference_papers_context_for_critique(
        self,
        current_outline: str = "",
        current_body: str = "",
        aggregator_db: str = "",
        critique_feedback: str = ""
    ) -> Optional[str]:
        """
        Prepare reference-paper context for critique prompts in autonomous mode.

        This preserves the reference papers selected for the paper instead of
        silently dropping them once the critique phase begins.
        """
        if not self.autonomous_mode or not self._current_reference_paper_ids:
            return None

        try:
            from backend.autonomous.core.autonomous_rag_manager import autonomous_rag_manager
            from backend.autonomous.memory.brainstorm_memory import brainstorm_memory

            max_input_tokens = rag_config.get_available_input_tokens(
                system_config.compiler_high_param_context_window,
                system_config.compiler_high_param_max_output_tokens
            )

            direct_injected_context = "\n\n".join(
                part for part in [
                    self.user_prompt or "",
                    self.paper_title or "",
                    current_outline or "",
                    current_body or "",
                    aggregator_db or "",
                    critique_feedback or "",
                ]
                if part
            )
            direct_tokens = count_tokens(direct_injected_context)

            # Scale reference budget from the user-configured critique context
            # instead of applying a hidden fixed cap/headroom.
            available_after_direct = max_input_tokens - direct_tokens
            reference_budget = max(0, int(available_after_direct * 0.35))
            if reference_budget <= 0:
                logger.warning(
                    "Skipping critique reference context due to prompt budget "
                    f"(direct={direct_tokens}, max_input={max_input_tokens})"
                )
                return None

            exclude_sources = ["compiler_outline.txt", "compiler_paper.txt"]
            if self._current_topic_id:
                brainstorm_db_path = brainstorm_memory.get_database_path(self._current_topic_id)
                exclude_sources.append(Path(brainstorm_db_path).name)

            query = "\n\n".join(
                part for part in [
                    self.user_prompt or "",
                    self.paper_title or "",
                    current_outline or "",
                    current_body or "",
                    critique_feedback or "",
                ]
                if part
            )

            reference_context, _ = await autonomous_rag_manager.get_reference_papers_context(
                self._current_reference_paper_ids,
                max_total_tokens=reference_budget,
                query=query,
                exclude_sources=exclude_sources
            )

            return reference_context or None
        except Exception as e:
            logger.warning(f"Failed to prepare critique reference context: {e}")
            return None
    
    async def _run_critique_aggregation(self) -> None:
        """
        Run critique aggregation until the configured total attempt target.
        Uses simple generate-validate loop similar to aggregator workflow.
        """
        logger.info(
            "Starting critique aggregation loop "
            f"(target: {CRITIQUE_ATTEMPT_TARGET} total attempts, accepted OR rejected)"
        )
        
        rejection_count = 0
        consecutive_rejections = 0
        total_attempts = 0  # Track all attempts (accepted + rejected + declines)
        
        while self.is_running and self.in_critique_phase:
            try:
                # Get current critique count
                critique_count = await critique_memory.get_critique_count()
                self.critique_acceptances = critique_count
                
                # Broadcast progress
                await self._broadcast("critique_progress", {
                    "acceptances": critique_count,
                    "rejections": rejection_count,
                    "total_attempts": total_attempts,
                    "target": CRITIQUE_ATTEMPT_TARGET,
                    "version": self.paper_version
                })
                
                # Check if target reached
                if total_attempts >= CRITIQUE_ATTEMPT_TARGET:
                    logger.info(f"Critique phase complete: {total_attempts} total attempts ({critique_count} accepted, {rejection_count} rejected)")

                    if critique_count == 0:
                        logger.info("No critiques accepted - moving to next section")
                        await self._continue_without_self_review()
                    else:
                        await self._append_accepted_critiques_as_self_review()
                    break
                
                # Generate critique
                logger.info(
                    f"Generating critique (attempts: {total_attempts}/{CRITIQUE_ATTEMPT_TARGET}, "
                    f"accepted: {critique_count}, rejected: {rejection_count})"
                )
                
                current_body = await paper_memory.get_paper()
                current_outline = await outline_memory.get_outline()
                
                # Get aggregator database
                from backend.aggregator.memory.shared_training import shared_training_memory
                aggregator_db = await shared_training_memory.get_all_content()
                
                # Get existing critiques
                existing_critiques = await critique_memory.get_all_critiques()
                
                # Keep autonomous reference papers available during critique.
                reference_papers = await self._get_reference_papers_context_for_critique(
                    current_outline=current_outline,
                    current_body=current_body,
                    aggregator_db=aggregator_db,
                    critique_feedback=existing_critiques
                )
                
                # Generate critique submission
                submission = await self.critique_submitter.submit_critique(
                    user_prompt=self.user_prompt,
                    current_body=current_body,
                    current_outline=current_outline,
                    aggregator_db=aggregator_db,
                    reference_papers=reference_papers,
                    existing_critiques=existing_critiques
                )
                
                if submission is None:
                    logger.warning("Critique generation returned None - retrying")
                    await asyncio.sleep(5)
                    continue
                
                logger.info(f"Critique generated: {submission.submission_id}")
                
                # Validate critique using critique-specific validator prompts.
                validation_result = await self._validate_critique(submission)
                
                # Handle decline submissions differently
                if submission.is_decline:
                    total_attempts += 1  # Count decline as attempt
                    # This is a decline assessment (critique_needed=false)
                    if validation_result and validation_result.decision == "accept":
                        # Validator agrees - body is academically acceptable, no critique needed
                        logger.info(f"Decline ACCEPTED - validator agrees body is academically acceptable")
                        
                        await self._broadcast("critique_decline_accepted", {
                            "submission_id": submission.submission_id,
                            "reasoning": submission.reasoning,
                            "version": self.paper_version,
                            "total_attempts": total_attempts,
                            "target": CRITIQUE_ATTEMPT_TARGET
                        })
                    else:
                        # Validator disagrees - there ARE issues that need critique
                        consecutive_rejections += 1
                        rejection_count += 1
                        logger.info(f"Decline REJECTED - validator found issues: {validation_result.reasoning if validation_result else 'Unknown'}")
                        
                        await self._broadcast("critique_decline_rejected", {
                            "submission_id": submission.submission_id,
                            "reasoning": validation_result.reasoning if validation_result else "Unknown",
                            "consecutive": consecutive_rejections,
                            "total_attempts": total_attempts,
                            "target": CRITIQUE_ATTEMPT_TARGET
                        })
                else:
                    # Regular critique submission
                    total_attempts += 1  # Count critique as attempt
                    
                    if validation_result and validation_result.decision == "accept":
                        # Accept critique
                        consecutive_rejections = 0
                        await critique_memory.add_accepted_critique(submission.content)
                        
                        new_count = await critique_memory.get_critique_count()
                        logger.info(f"Critique ACCEPTED ({new_count}/{CRITIQUE_ATTEMPT_TARGET}): {submission.submission_id}")
                        
                        await self._broadcast("critique_accepted", {
                            "critique_id": submission.submission_id,
                            "count": new_count,
                            "target": CRITIQUE_ATTEMPT_TARGET,
                            "version": self.paper_version,
                            "total_attempts": total_attempts,
                            "rejections": rejection_count
                        })
                        
                        # Check for cleanup/pruning every 7 acceptances
                        if new_count % 7 == 0 and new_count > 0:
                            logger.info(f"Critique count reached {new_count} (multiple of 7) - checking for cleanup")
                            await self._perform_critique_cleanup()
                        
                    else:
                        # Reject critique
                        consecutive_rejections += 1
                        rejection_count += 1
                        logger.info(f"Critique REJECTED: {validation_result.reasoning if validation_result else 'Unknown reason'}")
                        
                        # Store rejection feedback for learning
                        if validation_result and validation_result.summary:
                            await self.critique_submitter.handle_critique_rejection(
                                summary=validation_result.summary,
                                content=submission.content
                            )
                        
                        await self._broadcast("critique_rejected", {
                            "critique_id": submission.submission_id,
                            "reasoning": validation_result.reasoning if validation_result else "Unknown",
                            "consecutive": consecutive_rejections,
                            "total_attempts": total_attempts,
                            "target": CRITIQUE_ATTEMPT_TARGET
                        })
                
                # Brief delay between critiques
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error in critique aggregation loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _validate_critique(self, submission) -> Optional[ValidationResult]:
        """
        Validate a critique submission using the validator.
        Reuses validator's validation logic with critique-specific prompts.
        
        Args:
            submission: The critique submission to validate
            
        Returns:
            ValidationResult or None
        """
        try:
            # Build validation prompt for critique
            # We pass the critique as "submission" and existing critiques as "context"
            current_body = await paper_memory.get_paper()
            current_outline = await outline_memory.get_outline()
            existing_critiques = await critique_memory.get_all_critiques()
            
            # Build prompt using critique validator prompts
            from backend.compiler.prompts.critique_prompts import (
                get_critique_validator_system_prompt,
                get_critique_validation_json_schema
            )
            
            # Assemble validation prompt
            parts = [
                get_critique_validator_system_prompt(),
                "\n---\n",
                get_critique_validation_json_schema(),
                "\n---\n",
                f"USER COMPILER-DIRECTING PROMPT:\n{self.user_prompt}",
                "\n---\n",
                f"PAPER TITLE:\n{self.paper_title}",
                "\n---\n",
                f"CURRENT OUTLINE:\n{current_outline}",
                "\n---\n",
                f"CURRENT BODY SECTION:\n{current_body}",
                "\n---\n",
                f"CRITIQUE TO VALIDATE:\n{submission.content}",
                "\n---\n",
                f"EXISTING ACCEPTED CRITIQUES:\n{existing_critiques if existing_critiques else 'None yet'}",
                "\n---\n",
                "Evaluate this critique and provide your decision as JSON:"
            ]
            
            prompt = ''.join(parts)
            from backend.shared.solution_path.integration import with_validator_hook
            prompt = with_validator_hook(prompt, self.solution_path_manager)
            
            # Generate task ID
            task_id = self.critique_submitter.next_critique_task_id("critique_val")
            
            # Call validator
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id="critique_validator",
                model=self.validator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.validator_max_tokens
            )
            
            # Extract text from response dict
            message = response.get("choices", [{}])[0].get("message", {})
            response_text = extract_message_text(message)
            
            # Parse response, with one bounded same-role JSON repair.
            try:
                data = parse_json(response_text)
            except Exception as parse_error:
                safe = sanitize_model_output_for_retry_context(
                    response_text, max_chars=2000
                )
                repair = (
                    "Your previous critique validation response was invalid JSON. "
                    f"Validation error: {parse_error}. Return the exact same primary "
                    "decision as one corrected JSON object. Retain the one valid "
                    "optional top-level `solution_path_update` if present; omission "
                    "remains valid."
                )
                retry = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_retry",
                    role_id="critique_validator",
                    model=self.validator_model,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": safe},
                        {"role": "user", "content": repair},
                    ],
                    temperature=0.0,
                    max_tokens=self.validator_max_tokens,
                )
                retry_message = retry.get("choices", [{}])[0].get("message", {})
                data = parse_json(extract_message_text(retry_message))
            
            if data is None:
                logger.error("Failed to parse critique validation response")
                return None
            
            # Handle array responses
            if isinstance(data, list):
                logger.warning("Validator returned array - using first element")
                if not data:
                    return None
                data = data[0]
            from backend.shared.solution_path.integration import enqueue_optional_update
            critique_decision = str(data.get("decision", "reject")).lower()
            reasoning = data.get("reasoning")
            if critique_decision not in {"accept", "reject"}:
                logger.error("Critique validator returned invalid decision")
                return None
            if not isinstance(reasoning, str) or not reasoning.strip():
                logger.error("Critique validator returned empty reasoning")
                return None
            await enqueue_optional_update(
                data,
                self.solution_path_manager,
                proposer_role="compiler_validator",
                source_task_id=task_id,
                source_phase="critique_validation",
                source_decision=(
                    critique_decision
                ),
            )
            
            # Create ValidationResult
            result = ValidationResult(
                submission_id=submission.submission_id,
                decision=data.get("decision", "reject"),
                reasoning=data.get("reasoning", ""),
                summary=data.get("summary", ""),
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            if _is_rigor_model_call_failure(e):
                raise
            logger.error(f"Error validating critique: {e}", exc_info=True)
            return None
    
    async def _perform_critique_cleanup(self) -> None:
        """
        Perform cleanup/pruning of critique database (every 7 acceptances).
        Similar to aggregator cleanup review.
        """
        try:
            logger.info("Starting critique cleanup review...")
            
            # Use validator to review critiques and identify one for removal
            from backend.aggregator.prompts.validator_prompts import (
                get_cleanup_review_system_prompt,
                get_cleanup_review_json_schema
            )
            
            all_critiques = await critique_memory.get_all_critiques()
            
            # Build cleanup prompt
            parts = [
                get_cleanup_review_system_prompt(),
                "\n---\n",
                get_cleanup_review_json_schema(),
                "\n---\n",
                f"USER PROMPT (the goal this critique database is improving):\n{self.user_prompt}",
                "\n---\n",
                f"CURRENT ACCEPTED CRITIQUES DATABASE:\n{all_critiques}",
                "\n---\n",
                "Review the critique database and provide your cleanup decision as JSON:"
            ]
            
            prompt = ''.join(parts)
            from backend.shared.solution_path.integration import with_validator_hook
            prompt = with_validator_hook(prompt, self.solution_path_manager)
            
            # Call validator
            task_id = self.critique_submitter.next_critique_task_id("critique_cleanup")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id="critique_cleanup",
                model=self.validator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.validator_max_tokens
            )
            
            # Extract text from response dict
            message = response.get("choices", [{}])[0].get("message", {})
            response_text = extract_message_text(message)
            
            # Parse response, with one bounded same-role JSON repair.
            try:
                data = parse_json(response_text)
            except Exception as parse_error:
                safe = sanitize_model_output_for_retry_context(
                    response_text, max_chars=2000
                )
                repair = (
                    "Your previous critique cleanup response was invalid JSON. "
                    f"Validation error: {parse_error}. Return the exact same primary "
                    "decision as one corrected JSON object. Retain the one valid "
                    "optional top-level `solution_path_update` if present; omission "
                    "remains valid."
                )
                retry = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_retry",
                    role_id="critique_cleanup",
                    model=self.validator_model,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": safe},
                        {"role": "user", "content": repair},
                    ],
                    temperature=0.0,
                    max_tokens=self.validator_max_tokens,
                )
                retry_message = retry.get("choices", [{}])[0].get("message", {})
                data = parse_json(extract_message_text(retry_message))
            if data is None:
                logger.info("Critique cleanup: No removal needed")
                return
            should_remove = data.get("should_remove", False)
            critique_number = data.get("submission_number")
            if should_remove and not critique_number:
                logger.warning(
                    "Critique cleanup requested removal without submission_number"
                )
                return
            from backend.shared.solution_path.integration import enqueue_optional_update
            await enqueue_optional_update(
                data,
                self.solution_path_manager,
                proposer_role="compiler_validator",
                source_task_id=task_id,
                source_phase="critique_cleanup_validation",
                source_decision="accept" if should_remove else "reject",
            )
            if not should_remove:
                logger.info("Critique cleanup: No removal needed")
                return
            
            # Remove the critique
            if critique_number:
                success = await critique_memory.remove_critique(critique_number)
                if success:
                    logger.info(f"Critique cleanup: Removed critique #{critique_number}")
                    await self._broadcast("critique_removed", {
                        "critique_number": critique_number,
                        "reasoning": data.get("reasoning", "")[:200]
                    })
            
        except Exception as e:
            logger.error(f"Error in critique cleanup: {e}", exc_info=True)
    
    async def _append_accepted_critiques_as_self_review(self) -> None:
        """Append validator-accepted critiques as an honest AI self-review section."""
        critique_feedback = await critique_memory.get_all_critiques()
        if not critique_feedback.strip():
            await self._continue_without_self_review()
            return

        appended = await paper_memory.append_self_review_section(critique_feedback)
        await self._broadcast("self_review_appended", {
            "version": self.paper_version,
            "critique_count": await critique_memory.get_critique_count(),
            "appended": appended,
        })
        await self._end_critique_phase(self_review_appended=appended)
    
    async def _end_critique_phase(self, self_review_appended: bool = False) -> None:
        """
        End critique phase and clean up.
        
        Args:
            self_review_appended: Whether accepted critiques were appended to the paper
        """
        logger.info(f"Ending critique phase (self_review_appended={self_review_appended})")
        
        self.in_critique_phase = False
        
        # Stop critique aggregator
        if self.critique_aggregator:
            try:
                await self.critique_aggregator.stop()
            except Exception as e:
                logger.error(f"Error stopping critique aggregator: {e}")
            finally:
                self.critique_aggregator = None
        
        # Broadcast end
        await self._broadcast("critique_phase_ended", {
            "self_review_appended": self_review_appended,
            "version": self.paper_version
        })

        self.autonomous_section_phase = "conclusion"
        logger.info("Critique phase complete - transitioning to CONCLUSION phase")
        await self._broadcast("phase_transition", {
            "from_phase": "critique",
            "to_phase": "conclusion",
            "trigger": "critiques_reviewed",
            "paper_word_count": await paper_memory.get_word_count()
        })
    
    async def _continue_without_self_review(self) -> None:
        """
        Continue when no critiques were accepted for the self-review section.
        """
        logger.info("=" * 80)
        logger.info("NO SELF-REVIEW APPENDED - No critiques accepted")
        logger.info("=" * 80)

        await self._end_critique_phase(self_review_appended=False)
        
        # The _end_critique_phase already transitions to conclusion.
        logger.info("Transitioning to CONCLUSION phase (body accepted as-is)")

    async def _monitor_aggregator_for_rerag(self) -> None:
        """Monitor aggregator acceptances and trigger incremental re-RAG every 10."""
        logger.info("Aggregator monitoring started - will check for new acceptances every 30 seconds")
        
        try:
            while self.is_running:
                try:
                    # Import here to avoid circular dependency
                    from backend.aggregator.memory.shared_training import shared_training_memory
                    
                    current_count = await shared_training_memory.get_insights_count()
                    
                    # Check if we have 10+ new acceptances
                    if current_count >= self.aggregator_acceptances_last_rag + 10:
                        logger.info(
                            f"Aggregator monitoring: Detected {current_count - self.aggregator_acceptances_last_rag} "
                            f"new acceptances (threshold: 10). Triggering incremental re-RAG..."
                        )
                        
                        # Trigger incremental re-RAG
                        await compiler_rag_manager.incremental_rerag_aggregator_database()
                        
                        # Update counter
                        self.aggregator_acceptances_last_rag = current_count
                        
                        await self._broadcast("aggregator_rerag_complete", {
                            "total_acceptances": current_count,
                            "last_rag_at": self.aggregator_acceptances_last_rag
                        })
                    
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Aggregator monitoring error: {e}", exc_info=True)
                    await asyncio.sleep(30)
        
        except asyncio.CancelledError:
            logger.info("Aggregator monitoring stopped")
            raise
    
    async def _check_phase_transition(self, section_complete: bool = False) -> bool:
        """
        Check if current phase is complete and transition to next phase.
        
        PHASE-BASED CONSTRUCTION:
        Phase transitions are triggered by explicit section_complete=True from the submitter.
        This replaces unreliable regex-based detection.
        
        Args:
            section_complete: Explicit signal from submitter that current section is complete
        
        Returns:
            True if paper is fully complete (abstract phase completed), False otherwise
        """
        current_phase = self.autonomous_section_phase
        
        if not section_complete:
            return False
        
        # Log current state for debugging
        word_count = await paper_memory.get_word_count()
        logger.info(f"Phase transition requested: current={current_phase}, paper_words={word_count}")
        
        # Phase transition logic based on explicit completion signal
        if current_phase == "body":
            # BODY COMPLETE - TRIGGER CRITIQUE PHASE BEFORE CONCLUSION
            logger.info("Body section complete - transitioning to CRITIQUE PHASE")
            await self._broadcast("phase_transition", {
                "from_phase": "body",
                "to_phase": "critique",
                "trigger": "section_complete",
                "paper_word_count": word_count
            })
            
            # Start critique aggregation sub-workflow
            await self._start_critique_phase()
            return False  # Don't advance to conclusion yet - critique phase will handle that
        
        elif current_phase == "conclusion":
            # VERIFY CONCLUSION ACTUALLY EXISTS BEFORE TRANSITIONING
            current_paper = await paper_memory.get_paper()
            has_conclusion = bool(re.search(
                r"(?:^|\n)\s*(?:(?:#+\s*)?(?:[IVXLCDM]+\.?\s*)?(?:Conclusion|Summary|Discussion|Final\s*Remarks|Concluding\s*Remarks)|\\(?:section|chapter)\*?\{(?:Conclusion|Summary|Discussion|Final\s*Remarks|Concluding\s*Remarks)\})",
                current_paper, re.IGNORECASE | re.MULTILINE
            ))
            
            if not has_conclusion:
                logger.error("Cannot transition from conclusion phase: No Conclusion section found in paper")
                return False  # Block transition
            
            self.autonomous_section_phase = "introduction"
            logger.info("Phase transition: conclusion → introduction (explicit section_complete)")
            await self._broadcast("phase_transition", {
                "from_phase": "conclusion",
                "to_phase": "introduction",
                "trigger": "section_complete",
                "paper_word_count": word_count
            })
            return False
        
        elif current_phase == "introduction":
            # VERIFY INTRODUCTION ACTUALLY EXISTS BEFORE TRANSITIONING
            current_paper = await paper_memory.get_paper()
            has_introduction = bool(re.search(
                r"(?:^|\n)\s*(?:(?:#+\s*)?(?:I\.?\s*)?Introduction|\\(?:section|chapter)\*?\{(?:I\.?\s*)?Introduction\})",
                current_paper, re.IGNORECASE | re.MULTILINE
            ))
            
            if not has_introduction:
                logger.error("Cannot transition from introduction phase: No Introduction section found in paper")
                return False  # Block transition

            logger.info("Introduction complete - running pre-abstract empirical red-team review")
            await self._run_pre_abstract_red_team_review()
            
            self.autonomous_section_phase = "abstract"
            logger.info("Phase transition: introduction → abstract (explicit section_complete)")
            await self._broadcast("phase_transition", {
                "from_phase": "introduction",
                "to_phase": "abstract",
                "trigger": "section_complete",
                "paper_word_count": word_count
            })
            return False
        
        elif current_phase == "abstract":
            # VERIFY ABSTRACT ACTUALLY EXISTS BEFORE MARKING PAPER COMPLETE
            current_paper = await paper_memory.get_paper()
            has_abstract = bool(re.search(
                r"(?:^|\n)\s*(?:(?:#+\s*)?\*{0,2}Abstract\*{0,2}|\\(?:section|chapter)\*?\{Abstract\}|\\begin\{abstract\})",
                current_paper, re.IGNORECASE | re.MULTILINE
            ))
            
            if not has_abstract:
                logger.error("Cannot complete paper: No Abstract section found in paper")
                return False  # Block completion
            
            logger.info(f"Paper COMPLETE: Abstract phase completed (explicit section_complete). Final word count: {word_count}")
            await self._broadcast("paper_complete", {
                "trigger": "section_complete",
                "final_word_count": word_count
            })
            return True  # Paper is fully complete
        
        return False
    
    async def get_status(self) -> CompilerState:
        """Get current compiler status."""
        word_count = await paper_memory.get_word_count()
        
        return CompilerState(
            is_running=self.is_running,
            current_mode=self.current_mode,
            outline_accepted=self.outline_accepted,
            paper_word_count=word_count,
            total_submissions=self.total_submissions,
            construction_acceptances=self.construction_acceptances,
            construction_rejections=self.construction_rejections,
            construction_declines=self.construction_declines,
            rigor_acceptances=self.rigor_acceptances,
            rigor_rejections=self.rigor_rejections,
            rigor_declines=self.rigor_declines,
            outline_acceptances=self.outline_acceptances,
            outline_rejections=self.outline_rejections,
            outline_declines=self.outline_declines,
            review_acceptances=self.review_acceptances,
            review_rejections=self.review_rejections,
            review_declines=self.review_declines,
            minuscule_edit_count=self.minuscule_edit_count,
            in_critique_phase=self.in_critique_phase,
            critique_acceptances=self.critique_acceptances,
            paper_version=self.paper_version
        )
    
    def get_model_tracking_data(self) -> Optional[Dict]:
        """
        Get per-paper model tracking data (for manual Part 2 mode).
        Includes Wolfram Alpha call counts.
        
        Returns:
            Dict with model_usage, total_calls, generation_date, or None if no tracking
        """
        if not self._paper_model_tracker:
            return None
        
        return {
            "model_usage": self._paper_model_tracker.get_models_dict(),
            "total_calls": self._paper_model_tracker.total_calls,
            "generation_date": self._paper_model_tracker.generation_date.isoformat(),
            "authors": self._paper_model_tracker.get_author_list(),
            "wolfram_calls": self._paper_model_tracker.get_wolfram_call_count()
        }
    
    async def clear_paper(self) -> None:
        """Clear the current paper and outline, reset to fresh start."""
        logger.info("Clearing paper and outline...")
        
        # Stop compiler if running
        was_running = self.is_running
        if was_running:
            await self.stop()
        
        # Clear paper and outline
        await paper_memory.update_paper("")
        await outline_memory.update_outline("")
        
        # Clear critique memory
        try:
            await critique_memory.clear()
            logger.info("Cleared critique memory")
        except Exception as e:
            logger.warning(f"Failed to clear critique memory: {e}")
        
        # Clear rejection logs
        async with compiler_rejection_log._lock:
            compiler_rejection_log.rejections.clear()
            compiler_rejection_log.acceptances.clear()
            await compiler_rejection_log._write_rejections()
            await compiler_rejection_log._write_acceptances()
        logger.info("Cleared rejection/acceptance logs")
        
        # Reset per-paper model tracker for fresh start
        if self._paper_model_tracker:
            self._paper_model_tracker.reset()
            logger.info("Reset per-paper model tracker")
        
        # Reset workflow state
        self.outline_accepted = False
        self.total_submissions = 0
        self.construction_acceptances = 0
        self.construction_rejections = 0
        self.construction_declines = 0
        self.rigor_acceptances = 0
        self.rigor_rejections = 0
        self.rigor_declines = 0
        self.outline_acceptances = 0
        self.outline_rejections = 0
        self.outline_declines = 0
        self.review_acceptances = 0
        self.review_rejections = 0
        self.review_declines = 0
        self.minuscule_edit_count = 0
        self.construction_cycle_count = 0
        self.rigor_cycle_active = False
        
        # Reset autonomous mode state (if in autonomous mode)
        if self.autonomous_mode:
            self.autonomous_section_phase = "body"  # Reset to body phase
            logger.info("Reset autonomous section phase to body")
        self._current_reference_paper_ids = []
        
        # Reset critique phase state
        self.in_critique_phase = False
        self.critique_acceptances = 0
        self.paper_version = 1
        self.paper_title = None
        logger.info("Reset critique phase state")
        
        logger.info("Paper and outline cleared - system reset to fresh start")
        await self._broadcast("paper_cleared", {"message": "Paper and outline cleared"})
    
    def set_websocket_broadcaster(self, broadcaster: Callable) -> None:
        """Set WebSocket broadcaster function."""
        self.websocket_broadcaster = broadcaster
    
    async def _broadcast(self, event_type: str, data: Dict) -> None:
        """Broadcast event via WebSocket."""
        if self.websocket_broadcaster:
            try:
                await self.websocket_broadcaster(event_type, data)
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")


# Global compiler coordinator instance
compiler_coordinator = CompilerCoordinator()

