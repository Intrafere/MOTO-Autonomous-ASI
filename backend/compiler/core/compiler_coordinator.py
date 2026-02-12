"""
Compiler Coordinator - orchestrates sequential Markov chain workflow.
Manages mode switching, submission/validation loop, and paper/outline updates.
"""
import asyncio
import logging
import re
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional, Dict, Callable, List, Tuple
from datetime import datetime

from backend.shared.config import system_config, rag_config
from backend.shared.models import CompilerState, CompilerSubmission, CompilerValidationResult, WorkflowTask, SubmitterConfig, ValidationResult, ModelConfig
from backend.shared.workflow_predictor import workflow_predictor
from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.compiler.agents.high_context_submitter import HighContextSubmitter
from backend.compiler.agents.high_param_submitter import HighParamSubmitter
from backend.compiler.agents.critique_submitter import CritiqueSubmitterAgent
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

logger = logging.getLogger(__name__)


class CompilerCoordinator:
    """
    Coordinates the compiler system.
    - Sequential Markov chain workflow (submit → validate → update → submit)
    - Mode switching (construction → outline → review → rigor)
    - Paper and outline memory management
    - WebSocket broadcasting
    """
    
    def __init__(self):
        self.high_context_submitter: Optional[HighContextSubmitter] = None
        self.high_param_submitter: Optional[HighParamSubmitter] = None
        self.validator: Optional[CompilerValidator] = None
        
        self.is_running = False
        self.current_mode = "idle"
        self.outline_accepted = False
        
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
        self.miniscule_edit_count = 0
        
        # Workflow state
        self.construction_cycle_count = 0
        self.rigor_cycle_active = False
        
        # Autonomous mode (for Part 3 integration)
        self.autonomous_mode = False
        self.autonomous_section_phase = None  # "body", "conclusion", "introduction", "abstract"
        
        # Critique phase state (post-body peer review)
        self.critique_submitter = None  # CritiqueSubmitterAgent instance
        self.critique_aggregator = None  # Coordinator instance for critique workflow
        self.in_critique_phase = False
        self.critique_acceptances = 0
        self.paper_version = 1  # Track version number
        self.rewrite_count = 0  # Track COMPLETED rewrites (max 1)
        self.rewrite_pending = False  # Track if rewrite initiated but not yet succeeded
        self.accumulated_critique_history: List[Dict] = []  # Store all critiques from all versions
        self.previous_body_versions: List[Dict] = []  # Store prior versions
        self.needs_critique_after_rewrite = False  # Flag to trigger another critique round
        self.paper_title: Optional[str] = None  # Track current paper title
        self._skip_critique_requested = False  # Pre-emptive skip flag (user can set before critique phase)
        self.pre_critique_paper: Optional[str] = None  # Snapshot of paper at critique phase start
        self.current_critique_feedback: Optional[str] = None  # Accepted critiques for current version (for rewrite context)
        
        # Aggregator monitoring for incremental re-RAG
        self.aggregator_acceptances_last_rag = 0
        
        # WebSocket broadcaster
        self.websocket_broadcaster: Optional[Callable] = None
        
        # Main loop task
        self._main_task: Optional[asyncio.Task] = None
        self._aggregator_monitor_task: Optional[asyncio.Task] = None
        
        # Per-paper model tracking for manual mode (Part 2)
        self._paper_model_tracker: Optional[PaperModelTracker] = None
        
        # Workflow tracking
        self.workflow_tasks: List[WorkflowTask] = []
        self.completed_task_ids: set = set()
        self.current_task_sequence: int = 0
        self.current_task_id: Optional[str] = None  # Currently executing task
    
    async def initialize(
        self,
        compiler_prompt: str,
        validator_model: str,
        high_context_model: str,
        high_param_model: str,
        critique_submitter_model: str,
        skip_aggregator_db: bool = False,
        # OpenRouter provider config for validator
        validator_provider: str = "lm_studio",
        validator_openrouter_provider: Optional[str] = None,
        validator_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider config for high-context submitter
        high_context_provider: str = "lm_studio",
        high_context_openrouter_provider: Optional[str] = None,
        high_context_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider config for high-param submitter
        high_param_provider: str = "lm_studio",
        high_param_openrouter_provider: Optional[str] = None,
        high_param_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider config for critique submitter
        critique_submitter_provider: str = "lm_studio",
        critique_submitter_openrouter_provider: Optional[str] = None,
        critique_submitter_lm_studio_fallback: Optional[str] = None
    ) -> None:
        """
        Initialize the compiler coordinator.
        
        Args:
            compiler_prompt: User's compiler-directing prompt
            validator_model: Model for validator
            high_context_model: Model for high-context submitter
            high_param_model: Model for high-param submitter
            critique_submitter_model: Model for critique generation and rewrite decisions
            skip_aggregator_db: If True, don't load Part 1 aggregator database (for autonomous mode)
            validator_provider: Provider for validator ("lm_studio" or "openrouter")
            validator_openrouter_provider: OpenRouter host provider for validator
            validator_lm_studio_fallback: LM Studio fallback model for validator
            high_context_provider: Provider for high-context submitter
            high_context_openrouter_provider: OpenRouter host provider for high-context submitter
            high_context_lm_studio_fallback: LM Studio fallback model for high-context submitter
            high_param_provider: Provider for high-param submitter
            high_param_openrouter_provider: OpenRouter host provider for high-param submitter
            high_param_lm_studio_fallback: LM Studio fallback model for high-param submitter
            critique_submitter_provider: Provider for critique submitter
            critique_submitter_openrouter_provider: OpenRouter host provider for critique submitter
            critique_submitter_lm_studio_fallback: LM Studio fallback model for critique submitter
        """
        logger.info("Initializing compiler coordinator...")
        
        # Store user prompt, paper title, and model configs
        self.user_prompt = compiler_prompt
        self.paper_title = compiler_prompt  # Initial title is the compiler prompt
        self.validator_model = validator_model
        self.validator_context_window = system_config.compiler_validator_context_window
        self.validator_max_tokens = system_config.compiler_validator_max_output_tokens
        self.critique_submitter_model = critique_submitter_model
        
        # Store OpenRouter provider configs for all roles
        self.validator_provider = validator_provider
        self.validator_openrouter_provider = validator_openrouter_provider
        self.validator_lm_studio_fallback = validator_lm_studio_fallback
        self.high_context_provider = high_context_provider
        self.high_context_openrouter_provider = high_context_openrouter_provider
        self.high_context_lm_studio_fallback = high_context_lm_studio_fallback
        self.high_param_provider = high_param_provider
        self.high_param_openrouter_provider = high_param_openrouter_provider
        self.high_param_lm_studio_fallback = high_param_lm_studio_fallback
        self.critique_submitter_provider = critique_submitter_provider
        self.critique_submitter_openrouter_provider = critique_submitter_openrouter_provider
        self.critique_submitter_lm_studio_fallback = critique_submitter_lm_studio_fallback
        
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
        self.miniscule_edit_count = 0
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
            logger.info("Wolfram Alpha client initialized for rigor mode")
        
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
            await asyncio.to_thread(rag_manager.clear_all_documents)
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
        self.high_context_submitter = HighContextSubmitter(
            high_context_model, 
            compiler_prompt,
            websocket_broadcaster=self.websocket_broadcaster
        )
        await self.high_context_submitter.initialize()
        # Set up task tracking callback for workflow panel integration
        self.high_context_submitter.set_task_tracking_callback(self._handle_task_event)
        # Configure API client manager for high-context submitter (OpenRouter/LM Studio routing)
        api_client_manager.configure_role(
            role_id="compiler_high_context",
            config=ModelConfig(
                provider=self.high_context_provider,
                model_id=high_context_model,
                openrouter_provider=self.high_context_openrouter_provider,
                lm_studio_fallback_id=self.high_context_lm_studio_fallback,
                context_window=system_config.compiler_high_context_context_window,
                max_output_tokens=system_config.compiler_high_context_max_output_tokens
            )
        )
        
        self.high_param_submitter = HighParamSubmitter(
            high_param_model, 
            compiler_prompt,
            websocket_broadcaster=self.websocket_broadcaster
        )
        await self.high_param_submitter.initialize()
        # Set up task tracking callback for workflow panel integration
        self.high_param_submitter.set_task_tracking_callback(self._handle_task_event)
        # Configure API client manager for high-param submitter (OpenRouter/LM Studio routing)
        api_client_manager.configure_role(
            role_id="compiler_high_param",
            config=ModelConfig(
                provider=self.high_param_provider,
                model_id=high_param_model,
                openrouter_provider=self.high_param_openrouter_provider,
                lm_studio_fallback_id=self.high_param_lm_studio_fallback,
                context_window=system_config.compiler_high_param_context_window,
                max_output_tokens=system_config.compiler_high_param_max_output_tokens
            )
        )
        
        self.validator = CompilerValidator(
            validator_model, 
            compiler_prompt,
            websocket_broadcaster=self.websocket_broadcaster
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
                lm_studio_fallback_id=self.validator_lm_studio_fallback,
                context_window=self.validator_context_window,
                max_output_tokens=self.validator_max_tokens
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
            hc_seq = self.high_context_submitter.task_sequence if self.high_context_submitter else 0
            hp_seq = self.high_param_submitter.task_sequence if self.high_param_submitter else 0
            val_seq = self.validator.task_sequence if self.validator else 0
            
            # Build workflow tasks based on current mode and actual sequences
            tasks = []
            
            if not self.outline_accepted:
                # Outline creation phase: HC -> V -> HC -> V ...
                for i in range(20):
                    if i % 2 == 0:
                        task_id = f"comp_hc_{hc_seq:03d}"
                        role = "High-Context"
                        mode = "Outline Creation"
                        hc_seq += 1
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
                    ("hc", "High-Context", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("hc", "High-Context", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("hc", "High-Context", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("hc", "High-Context", "Construction"),
                    ("val", "Validator", "Construction Review"),
                    ("hc", "High-Context", "Outline Update"),
                    ("val", "Validator", "Outline Review"),
                    ("hc", "High-Context", "Paper Review"),
                    ("val", "Validator", "Review Validation"),
                    ("hc", "High-Context", "Paper Review"),
                    ("val", "Validator", "Review Validation"),
                    ("hp", "High-Param", "Rigor Enhancement"),
                    ("val", "Validator", "Rigor Review"),
                ]
                
                for i in range(20):
                    pattern_idx = i % len(cycle_pattern)
                    agent_type, role, mode = cycle_pattern[pattern_idx]
                    
                    if agent_type == "hc":
                        task_id = f"comp_hc_{hc_seq:03d}"
                        hc_seq += 1
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
            task_id: The task ID (e.g., "comp_hc_001", "comp_hp_002", "comp_val_003")
        """
        import asyncio
        
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
        # If rewrite is pending (initiated but not yet succeeded), body is NOT complete
        if self.rewrite_pending:
            return False
        
        # Check if max rewrites completed - skip critique entirely
        if self.rewrite_count >= 1:
            logger.info("Max rewrites completed (1) - treating body as complete")
            return True
        
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
        logger.info("Starting compiler...")
        
        # Refresh workflow predictions at start
        await self.refresh_workflow_predictions()
        
        # Start main workflow loop
        self._main_task = asyncio.create_task(self._main_workflow())
        
        # Start aggregator monitoring for incremental re-RAG
        self._aggregator_monitor_task = asyncio.create_task(self._monitor_aggregator_for_rerag())
        
        await self._broadcast("compiler_started", {"message": "Compiler started"})
        logger.info("Compiler started successfully")
    
    async def stop(self) -> None:
        """Stop the compiler system."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping compiler...")
        
        # Clear model tracking callback if this is manual mode
        if not self.autonomous_mode and self._paper_model_tracker:
            api_client_manager.set_model_tracking_callback(None)
            logger.info(f"Compiler stopped: tracked {len(self._paper_model_tracker.get_models_dict())} models, "
                       f"{self._paper_model_tracker.total_calls} API calls")
        
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        if self._aggregator_monitor_task:
            self._aggregator_monitor_task.cancel()
            try:
                await self._aggregator_monitor_task
            except asyncio.CancelledError:
                pass
        
        await self._broadcast("compiler_stopped", {"message": "Compiler stopped"})
        logger.info("Compiler stopped")
    
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
        except Exception as e:
            logger.error(f"Compiler workflow error: {e}", exc_info=True)
            self.is_running = False
            # Broadcast error to frontend
            await self._broadcast("compiler_error", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "mode": self.current_mode,
                "total_submissions": self.total_submissions
            })
    
    def _pre_validate_outline_structure(self, content: str) -> Optional[str]:
        """
        Pre-validate outline structure before sending to LLM validator.
        Returns detailed error message if critical issues found, None if OK.
        
        This provides IMMEDIATE, CONSISTENT feedback on structural issues
        without relying on LLM validator interpretation.
        """
        import re
        
        # Abstract is OPTIONAL - if included, it must be properly formatted
        # Valid formats: "Abstract", "I. Abstract", "0. Abstract" (case-insensitive)
        # If Abstract is not present, that's also fine - outline can start with Introduction
        # We don't enforce Abstract presence here - let validator handle acceptance logic
        
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
            submission = await self.high_context_submitter.submit_outline_create()
            
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
        max_retries = 5
        retry_count = 0
        rejection_feedback = None  # Store rejection feedback for retry
        
        while self.is_running and not initial_portion_accepted:
            # High-context submitter writes first portion
            submission = None
            
            # Retry with exponential backoff if submission fails
            for attempt in range(max_retries):
                try:
                    section_phase = self.autonomous_section_phase if self.autonomous_mode else None
                    submission = await self.high_context_submitter.submit_construction(
                        is_first_portion=True,
                        section_phase=section_phase,
                        rejection_feedback=rejection_feedback  # Pass rejection feedback for retry
                    )
                    
                    if submission is None:
                        retry_count += 1
                        backoff_time = min(2 ** attempt, 16)  # 1s, 2s, 4s, 8s, 16s max
                        logger.warning(
                            f"Construction submission returned None (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {backoff_time}s..."
                        )
                        
                        await self._broadcast("compiler_retry", {
                            "mode": "construction",
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "reason": "Empty submission returned"
                        })
                        
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        # Success!
                        if attempt > 0:
                            logger.info(f"Construction submission succeeded after {attempt + 1} attempts")
                        break
                        
                except Exception as e:
                    retry_count += 1
                    backoff_time = min(2 ** attempt, 16)
                    logger.error(
                        f"Construction submission failed with error (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    
                    await self._broadcast("compiler_retry", {
                        "mode": "construction",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "reason": str(e)
                    })
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {backoff_time}s...")
                        await asyncio.sleep(backoff_time)
                    else:
                        logger.error(f"Construction submission failed permanently after {max_retries} attempts")
                        raise
            
            # Check if we got a valid submission
            if submission is None:
                logger.error(f"Failed to get construction submission after {max_retries} attempts")
                raise RuntimeError("Construction submission failed permanently - returned None after all retries")
            
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
    
    async def _rigor_loop(self) -> None:
        """LOOP 2: Rigor enhancement (steps 19-21)."""
        logger.info("Starting rigor loop...")
        self.rigor_cycle_active = True
        
        # Continue until first rejection
        while self.is_running and self.rigor_cycle_active:
            accepted = await self._submit_and_validate_rigor()
            
            if not accepted:
                # First rejection - return to Loop 1
                self.rigor_cycle_active = False
                logger.info("Rigor cycle ended (first rejection)")
        
        logger.info("Rigor loop complete")
    
    # Maximum retries for premature decline/completion rejections
    MAX_PREMATURE_RETRIES = 5
    
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
        
        # Pass critique context during body rewrite (when critique_feedback is set)
        critique_feedback_for_construction = None
        pre_critique_paper_for_construction = None
        if section_phase == "body" and self.current_critique_feedback:
            critique_feedback_for_construction = self.current_critique_feedback
            pre_critique_paper_for_construction = self.pre_critique_paper
            logger.info("Body construction with critique context (rewrite mode)")
        
        submission = await self.high_context_submitter.submit_construction(
            is_first_portion=False,
            section_phase=section_phase,
            rejection_feedback=rejection_feedback,
            critique_feedback=critique_feedback_for_construction,
            pre_critique_paper=pre_critique_paper_for_construction
        )
        
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
                    from backend.shared.models import CompilerValidationResult
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
                    
                    from backend.shared.models import CompilerValidationResult
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
            # Update paper - use placeholder replacement for conclusion/intro/abstract
            # This ensures content replaces the placeholder marker instead of generic insertion
            section_phase = self.autonomous_section_phase if self.autonomous_mode else None
            
            if section_phase == "conclusion":
                # Replace the conclusion placeholder with validated content
                success = await paper_memory.replace_placeholder(CONCLUSION_PLACEHOLDER, submission.content)
                if not success:
                    logger.error("Conclusion placeholder not found - paper is in invalid state. Rejecting.")
                    updated_paper = None  # Trigger rejection
                else:
                    updated_paper = await paper_memory.get_paper()
            elif section_phase == "introduction":
                # Replace the introduction placeholder with validated content
                success = await paper_memory.replace_placeholder(INTRO_PLACEHOLDER, submission.content)
                if not success:
                    logger.error("Introduction placeholder not found - paper is in invalid state. Rejecting.")
                    updated_paper = None  # Trigger rejection
                else:
                    updated_paper = await paper_memory.get_paper()
            elif section_phase == "abstract":
                # Replace the abstract placeholder with validated content
                success = await paper_memory.replace_placeholder(ABSTRACT_PLACEHOLDER, submission.content)
                if not success:
                    logger.error("Abstract placeholder not found - paper is in invalid state. Rejecting.")
                    updated_paper = None  # Trigger rejection
                else:
                    updated_paper = await paper_memory.get_paper()
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
                
                # Create emergency rejection
                from backend.shared.models import CompilerValidationResult
                emergency_result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Exact string match failed: old_string='{submission.old_string[:100]}...' not found or not unique in document",
                    summary="Exact string match failed - old_string not found or not unique",
                    placement_check=False,
                    validation_stage="pre-validation"  # Exact string match check
                )
                
                await compiler_rejection_log.add_rejection(emergency_result, "construction", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "construction",
                    "submission_id": submission.submission_id,
                    "reasoning": "Emergency rejection: exact string match failed"
                })
                
                return False, emergency_result.reasoning
            
            # Only call update_paper for _apply_edit cases (placeholder replacement already saved)
            if section_phase not in ("conclusion", "introduction", "abstract"):
                await paper_memory.update_paper(updated_paper)
            
            self.construction_acceptances += 1
            
            # If rewrite was pending, mark it as completed now (first successful acceptance)
            if self.rewrite_pending:
                self.rewrite_count += 1
                self.rewrite_pending = False
                logger.info(f"Rewrite #{self.rewrite_count} completed successfully (first acceptance after rewrite)")
            
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
            return True, None
        else:
            self.construction_rejections += 1
            
            await compiler_rejection_log.add_rejection(result, "construction", submission.content)
            
            await self._broadcast("compiler_rejection", {
                "mode": "construction",
                "submission_id": submission.submission_id,
                "reasoning": result.reasoning
            })
            
            logger.info("Construction rejected")
            return False, result.reasoning
    
    async def _submit_and_validate_outline_update(self) -> bool:
        """Submit and validate outline update. Returns True if accepted."""
        self.current_mode = "outline_update"
        
        submission = await self.high_context_submitter.submit_outline_update()
        
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
                
                # Create emergency rejection
                from backend.shared.models import CompilerValidationResult
                emergency_result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Outline exact string match failed: old_string='{submission.old_string[:100]}...' not found or not unique in outline",
                    summary="Outline exact string match failed - old_string not found or not unique",
                    placement_check=False,
                    validation_stage="pre-validation"  # Exact string match check
                )
                
                await compiler_rejection_log.add_rejection(emergency_result, "outline_update", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "outline_update",
                    "submission_id": submission.submission_id,
                    "reasoning": "Emergency rejection: outline exact string match failed"
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
    
    async def _submit_and_validate_review(self) -> bool:
        """Submit and validate review. Returns True if accepted."""
        self.current_mode = "review"
        
        submission = await self.high_context_submitter.submit_review()
        
        if submission is None:
            logger.info("No review edit needed")
            self.review_declines += 1
            await compiler_rejection_log.add_decline("review", "No errors or improvements needed")
            
            await self._broadcast("compiler_decline", {
                "mode": "review",
                "reasoning": "No errors or improvements needed"
            })
            
            return False
        
        self.total_submissions += 1
        
        # Check for miniscule edit
        if submission.metadata.get("is_miniscule", False):
            self.miniscule_edit_count += 1
        
        await self._broadcast("compiler_submission", {
            "mode": "review",
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
            # Apply edit
            updated_paper = self._apply_edit(current_paper, submission)
            
            # Check if exact string match failed
            if updated_paper is None:
                logger.error(
                    f"Placement execution failed despite validator acceptance. "
                    f"Treating as rejection. Submission: {submission.submission_id}"
                )
                self.review_rejections += 1
                
                # Create emergency rejection
                from backend.shared.models import CompilerValidationResult
                emergency_result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Exact string match failed: old_string='{submission.old_string[:100]}...' not found or not unique in document",
                    summary="Exact string match failed - old_string not found or not unique",
                    placement_check=False,
                    validation_stage="pre-validation"  # Exact string match check
                )
                
                await compiler_rejection_log.add_rejection(emergency_result, "review", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "review",
                    "submission_id": submission.submission_id,
                    "reasoning": "Emergency rejection: exact string match failed"
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
                "submission_id": submission.submission_id
            })
            
            await self._broadcast("paper_updated", {
                "word_count": word_count,
                "preview": updated_paper[:500]
            })
            
            logger.info(f"Review edit accepted ({word_count} words)")
            return True
        else:
            self.review_rejections += 1
            
            await compiler_rejection_log.add_rejection(result, "review", submission.content)
            
            await self._broadcast("compiler_rejection", {
                "mode": "review",
                "submission_id": submission.submission_id,
                "reasoning": result.reasoning
            })
            
            logger.info("Review edit rejected")
            return False
    
    async def _submit_and_validate_rigor(self) -> bool:
        """Submit and validate rigor enhancement. Returns True if accepted."""
        self.current_mode = "rigor"
        
        submission = await self.high_param_submitter.submit_rigor_enhancement()
        
        if submission is None:
            logger.info("No rigor enhancement needed")
            self.rigor_declines += 1
            await compiler_rejection_log.add_decline("rigor", "Rigor already adequate")
            
            await self._broadcast("compiler_decline", {
                "mode": "rigor",
                "reasoning": "Rigor already adequate"
            })
            
            # Treat as rejection for loop purposes
            return False
        
        self.total_submissions += 1
        
        await self._broadcast("compiler_submission", {
            "mode": "rigor",
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
            # Apply enhancement
            updated_paper = self._apply_edit(current_paper, submission)
            
            # Check if exact string match failed
            if updated_paper is None:
                logger.error(
                    f"Placement execution failed despite validator acceptance. "
                    f"Treating as rejection. Submission: {submission.submission_id}"
                )
                self.rigor_rejections += 1
                
                # Create emergency rejection
                from backend.shared.models import CompilerValidationResult
                emergency_result = CompilerValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Exact string match failed: old_string='{submission.old_string[:100]}...' not found or not unique in document",
                    summary="Exact string match failed - old_string not found or not unique",
                    placement_check=False,
                    validation_stage="pre-validation"  # Exact string match check
                )
                
                await compiler_rejection_log.add_rejection(emergency_result, "rigor", submission.content)
                
                await self._broadcast("compiler_rejection", {
                    "mode": "rigor",
                    "submission_id": submission.submission_id,
                    "reasoning": "Emergency rejection: exact string match failed"
                })
                
                return False
            
            await paper_memory.update_paper(updated_paper)
            
            self.rigor_acceptances += 1
            
            # Track Wolfram Alpha call if applicable (only for accepted submissions)
            if submission.metadata.get("wolfram_query"):
                if self.autonomous_mode and self._current_paper_tracker:
                    # Autonomous mode (Part 3)
                    self._current_paper_tracker.track_wolfram_call(
                        submission.metadata["wolfram_query"]
                    )
                    logger.info(f"Tracked Wolfram Alpha call (autonomous): {submission.metadata['wolfram_query']}")
                elif not self.autonomous_mode and self._paper_model_tracker:
                    # Manual mode (Part 2)
                    self._paper_model_tracker.track_wolfram_call(
                        submission.metadata["wolfram_query"]
                    )
                    logger.info(f"Tracked Wolfram Alpha call (manual): {submission.metadata['wolfram_query']}")
            
            await compiler_rejection_log.add_acceptance(
                submission.submission_id,
                "rigor",
                submission.content[:500]
            )
            
            word_count = await paper_memory.get_word_count()
            
            await self._broadcast("compiler_acceptance", {
                "mode": "rigor",
                "submission_id": submission.submission_id
            })
            
            await self._broadcast("paper_updated", {
                "word_count": word_count,
                "preview": updated_paper[:500]
            })
            
            logger.info(f"Rigor enhancement accepted ({word_count} words)")
            return True
        else:
            self.rigor_rejections += 1
            
            await compiler_rejection_log.add_rejection(result, "rigor", submission.content)
            
            await self._broadcast("compiler_rejection", {
                "mode": "rigor",
                "submission_id": submission.submission_id,
                "reasoning": result.reasoning
            })
            
            logger.info("Rigor enhancement rejected")
            return False
    
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
        # Check for pre-emptive skip request
        if self._skip_critique_requested:
            logger.info("=" * 80)
            logger.info("PRE-EMPTIVE SKIP: User requested critique skip before phase started")
            logger.info("Skipping critique phase, transitioning directly to conclusion")
            logger.info("=" * 80)
            
            self._skip_critique_requested = False  # Reset flag
            
            await self._broadcast("critique_phase_skipped", {
                "reason": "user_override_preemptive",
                "version": self.paper_version
            })
            
            # Transition directly to conclusion phase
            self.autonomous_section_phase = "conclusion"
            await self._broadcast("phase_transition", {
                "from_phase": "body",
                "to_phase": "conclusion",
                "skip_reason": "preemptive_user_override"
            })
            return
        
        logger.info("=" * 80)
        logger.info("STARTING CRITIQUE PHASE")
        logger.info("=" * 80)
        
        self.in_critique_phase = True
        self.critique_acceptances = 0
        
        # Snapshot paper at critique phase start (for rewrite context)
        self.pre_critique_paper = await paper_memory.get_paper()
        logger.info(f"Snapshot pre-critique paper: {len(self.pre_critique_paper)} chars")
        
        # Clear current critique feedback for this round
        self.current_critique_feedback = None
        
        # Initialize critique memory
        paper_id = f"paper_v{self.paper_version}"
        critique_memory.initialize(paper_id)
        
        # Before clearing, accumulate any existing critiques from previous phases
        existing = await critique_memory.get_all_critiques()
        if existing.strip():
            self.accumulated_critique_history.append({
                "version": self.paper_version,
                "critiques": existing
            })
            logger.info(f"Accumulated {len(self.accumulated_critique_history)} critique history version(s)")
        
        await critique_memory.clear()
        
        # Load from file for crash recovery (if file exists)
        await critique_memory.load_from_file()
        
        logger.info(f"Critique memory initialized for {paper_id}")
        
        # Create critique submitter agent
        self.critique_submitter = CritiqueSubmitterAgent(
            model=self.critique_submitter_model,
            context_window=system_config.compiler_critique_submitter_context_window,
            max_tokens=system_config.compiler_critique_submitter_max_tokens,
            submitter_id=1
        )
        
        # Initialize rejection memory
        await self.critique_submitter.initialize()
        
        # Clear rejection feedback from previous critique phases (fresh start)
        await self.critique_submitter.rejection_memory.reset()
        logger.info("Cleared critique rejection feedback for fresh start")
        
        logger.info(f"Critique submitter created with model: {self.critique_submitter.model}")
        
        # Set up task tracking callback for workflow panel integration
        self.critique_submitter.set_task_tracking_callback(self._handle_task_event)
        
        # Configure API client manager for critique submitter (OpenRouter/LM Studio routing)
        api_client_manager.configure_role(
            role_id="compiler_critique_submitter",
            config=ModelConfig(
                provider=self.critique_submitter_provider,
                model_id=self.critique_submitter_model,
                openrouter_provider=self.critique_submitter_openrouter_provider,
                lm_studio_fallback_id=self.critique_submitter_lm_studio_fallback,
                context_window=system_config.compiler_critique_submitter_context_window,
                max_output_tokens=system_config.compiler_critique_submitter_max_tokens
            )
        )
        
        # Configure API client manager for critique validator (uses same settings as compiler_validator)
        api_client_manager.configure_role(
            role_id="critique_validator",
            config=ModelConfig(
                provider=self.validator_provider,
                model_id=self.validator_model,
                openrouter_provider=self.validator_openrouter_provider,
                lm_studio_fallback_id=self.validator_lm_studio_fallback,
                context_window=self.validator_context_window,
                max_output_tokens=self.validator_max_tokens
            )
        )
        
        # Configure API client manager for critique cleanup (uses same settings as compiler_validator)
        api_client_manager.configure_role(
            role_id="critique_cleanup",
            config=ModelConfig(
                provider=self.validator_provider,
                model_id=self.validator_model,
                openrouter_provider=self.validator_openrouter_provider,
                lm_studio_fallback_id=self.validator_lm_studio_fallback,
                context_window=self.validator_context_window,
                max_output_tokens=self.validator_max_tokens
            )
        )
        
        # Broadcast critique phase started
        await self._broadcast("critique_phase_started", {
            "paper_version": self.paper_version,
            "target_critiques": 5
        })
        
        # Start critique aggregation loop
        await self._run_critique_aggregation()
    
    async def _run_critique_aggregation(self) -> None:
        """
        Run critique aggregation until 5 total attempts.
        Uses simple generate-validate loop similar to aggregator workflow.
        """
        logger.info("Starting critique aggregation loop (target: 5 total attempts, accepted OR rejected)")
        
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
                    "target": 5,  # Now means total attempts, not just acceptances
                    "version": self.paper_version
                })
                
                # Check if target reached
                if total_attempts >= 5:
                    logger.info(f"Critique phase complete: {total_attempts} total attempts ({critique_count} accepted, {rejection_count} rejected)")
                    
                    # If 0 acceptances, skip rewrite and continue
                    if critique_count == 0:
                        logger.info("No critiques accepted - skipping rewrite phase, moving to next section")
                        await self._skip_rewrite_and_continue()
                    else:
                        # Trigger rewrite decision with accepted critiques
                        await self._trigger_rewrite_decision()
                    break
                
                # Generate critique
                logger.info(f"Generating critique (attempts: {total_attempts}/5, accepted: {critique_count}, rejected: {rejection_count})")
                
                current_body = await paper_memory.get_paper()
                current_outline = await outline_memory.get_outline()
                
                # Get aggregator database
                from backend.aggregator.memory.shared_training import shared_training_memory
                aggregator_db = await shared_training_memory.get_all_content()
                
                # Get existing critiques
                existing_critiques = await critique_memory.get_all_critiques()
                
                # Get reference papers if available
                reference_papers = None  # TODO: Load if applicable
                
                # Format accumulated critique history from previous failed versions
                accumulated_history = self._format_accumulated_critique_history()
                
                # Generate critique submission
                submission = await self.critique_submitter.submit_critique(
                    user_prompt=self.user_prompt,
                    current_body=current_body,
                    current_outline=current_outline,
                    aggregator_db=aggregator_db,
                    reference_papers=reference_papers,
                    existing_critiques=existing_critiques,
                    accumulated_history=accumulated_history
                )
                
                if submission is None:
                    logger.warning("Critique generation returned None - retrying")
                    await asyncio.sleep(5)
                    continue
                
                logger.info(f"Critique generated: {submission.submission_id}")
                
                # Validate critique using aggregator validator prompts
                from backend.aggregator.agents.validator import ValidatorAgent
                from backend.aggregator.memory.shared_training import shared_training_memory
                from backend.aggregator.prompts.validator_prompts import build_validator_prompt
                
                # Build critique validation prompt (reuses aggregator validator structure)
                # We'll use the validator's validate method but with critique-specific context
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
                            "target": 5
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
                            "target": 5
                        })
                else:
                    # Regular critique submission
                    total_attempts += 1  # Count critique as attempt
                    
                    if validation_result and validation_result.decision == "accept":
                        # Accept critique
                        consecutive_rejections = 0
                        await critique_memory.add_accepted_critique(submission.content)
                        
                        new_count = await critique_memory.get_critique_count()
                        logger.info(f"Critique ACCEPTED ({new_count}/5): {submission.submission_id}")
                        
                        await self._broadcast("critique_accepted", {
                            "critique_id": submission.submission_id,
                            "count": new_count,
                            "target": 5,
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
                            await self.critique_submitter.handle_rejection(
                                summary=validation_result.summary,
                                content=submission.content
                            )
                        
                        await self._broadcast("critique_rejected", {
                            "critique_id": submission.submission_id,
                            "reasoning": validation_result.reasoning if validation_result else "Unknown",
                            "consecutive": consecutive_rejections,
                            "total_attempts": total_attempts,
                            "target": 5
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
            # Import prompt builders
            from backend.aggregator.prompts.validator_prompts import build_validator_prompt
            
            # Build validation prompt for critique
            # We pass the critique as "submission" and existing critiques as "context"
            current_body = await paper_memory.get_paper()
            current_outline = await outline_memory.get_outline()
            existing_critiques = await critique_memory.get_all_critiques()
            
            from backend.aggregator.memory.shared_training import shared_training_memory
            aggregator_db = await shared_training_memory.get_all_content()
            
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
            
            # Generate task ID
            task_id = f"critique_val_{self.critique_submitter.task_sequence:03d}"
            self.critique_submitter.task_sequence += 1
            
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
            response_text = message.get("content", "") or message.get("reasoning", "")
            
            # Parse response
            data = parse_json(response_text)
            
            if data is None:
                logger.error("Failed to parse critique validation response")
                return None
            
            # Handle array responses
            if isinstance(data, list):
                logger.warning("Validator returned array - using first element")
                if not data:
                    return None
                data = data[0]
            
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
            
            # Call validator
            task_id = f"critique_cleanup_{self.critique_submitter.task_sequence:03d}"
            self.critique_submitter.task_sequence += 1
            
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
            response_text = message.get("content", "") or message.get("reasoning", "")
            
            # Parse response
            data = parse_json(response_text)
            
            if data is None or not data.get("should_remove", False):
                logger.info("Critique cleanup: No removal needed")
                return
            
            # Remove the critique
            critique_number = data.get("submission_number")
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
    
    async def _trigger_rewrite_decision(self) -> None:
        """
        Trigger rewrite vs continue decision after 5 critiques.
        Includes retry logic if decision is rejected by validator.
        """
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.info("=" * 80)
                logger.info(f"Critique phase complete (5 total attempts) - triggering rewrite decision (attempt {retry_count + 1})")
                logger.info("=" * 80)
                
                # Get all critiques
                critique_feedback = await critique_memory.get_all_critiques()
                current_body = await paper_memory.get_paper()
                current_outline = await outline_memory.get_outline()
                current_title = self.paper_title if self.paper_title else self.user_prompt
                
                # Get context (aggregator DB, reference papers, etc.)
                from backend.aggregator.memory.shared_training import shared_training_memory
                aggregator_db = await shared_training_memory.get_all_content()
                reference_papers = None  # TODO: Load if applicable
                
                # Format accumulated critique history from previous failed versions
                accumulated_history = self._format_accumulated_critique_history()
                
                # Critique submitter makes decision
                logger.info("Critique submitter generating rewrite decision...")
                decision_result = await self.critique_submitter.submit_rewrite_decision(
                    user_prompt=self.user_prompt,
                    current_body=current_body,
                    current_outline=current_outline,
                    current_title=current_title,
                    aggregator_db=aggregator_db,
                    critique_feedback=critique_feedback,
                    pre_critique_paper=self.pre_critique_paper,  # Paper snapshot from start of critique phase
                    reference_papers=reference_papers,
                    accumulated_history=accumulated_history
                )
                
                if decision_result is None:
                    logger.error("Rewrite decision generation returned None")
                    retry_count += 1
                    await asyncio.sleep(5)
                    continue
                
                logger.info(f"Rewrite decision: {decision_result['decision']}")
                
                # Validator reviews decision
                logger.info("Validator reviewing rewrite decision...")
                validated = await self.validator.validate_rewrite_decision(
                    decision_result=decision_result,
                    user_prompt=self.user_prompt,
                    current_body=current_body,
                    current_outline=current_outline,
                    current_title=current_title,
                    critique_feedback=critique_feedback,
                    aggregator_db=aggregator_db
                )
                
                if not validated:
                    # Decision rejected - retry
                    logger.warning("Rewrite decision rejected by validator - retrying")
                    await self._broadcast("rewrite_decision_rejected", {
                        "attempt": retry_count + 1,
                        "max_retries": max_retries
                    })
                    retry_count += 1
                    await asyncio.sleep(5)
                    continue
                
                # Decision validated - execute it
                logger.info("Rewrite decision validated - executing")
                
                # Execute decision
                if decision_result["decision"] == "continue":
                    logger.info("Decision: CONTINUE to conclusion (critiques minor/incorrect)")
                    await self._end_critique_phase(rewrite=False)
                    
                elif decision_result["decision"] == "partial_revision":
                    logger.info("Decision: PARTIAL REVISION (iterative targeted edits)")
                    await self._execute_partial_revision(
                        new_title=decision_result.get("new_title"),
                        new_outline=decision_result.get("new_outline"),
                        critique_feedback=critique_feedback,
                        accumulated_history=accumulated_history
                    )
                    
                elif decision_result["decision"] == "total_rewrite":
                    logger.info("Decision: TOTAL REWRITE body section")
                    await self._execute_body_rewrite(
                        new_title=decision_result.get("new_title"),
                        new_outline=decision_result.get("new_outline"),
                        critique_feedback=critique_feedback
                    )
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                logger.error(f"Error in rewrite decision (attempt {retry_count + 1}): {e}", exc_info=True)
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(5)
                # Note: Don't call _end_critique_phase here - let it fall through to unified fallback below
        
        # Unified fallback if while loop exited due to retry exhaustion
        # Handles both: validation failures (returned False 5 times) OR exceptions (5 exceptions occurred)
        if retry_count >= max_retries:
            logger.error("Rewrite decision validation failed after max retries - defaulting to CONTINUE")
            await self._broadcast("rewrite_decision_max_retries_exceeded", {
                "action": "continue_to_conclusion"
            })
            await self._end_critique_phase(rewrite=False)
    
    async def _execute_body_rewrite(
        self,
        new_title: Optional[str],
        new_outline: Optional[str],
        critique_feedback: str
    ) -> None:
        """
        Execute full body section rewrite.
        
        Args:
            new_title: New paper title (or None to keep current)
            new_outline: Updated outline (or None to keep current)
            critique_feedback: All accepted critiques
        """
        logger.info("=" * 80)
        logger.info("EXECUTING BODY REWRITE")
        logger.info("=" * 80)
        
        # Mark rewrite as pending (will count as completed only after first successful body acceptance)
        self.rewrite_pending = True
        logger.info(f"Rewrite initiated (pending successful completion, max: 1)")
        
        # Store previous version
        current_body = await paper_memory.get_paper()
        old_title = self.paper_title if self.paper_title else self.user_prompt
        
        await paper_memory.store_previous_version(
            version=self.paper_version,
            title=old_title,
            body=current_body,
            critique_feedback=critique_feedback
        )
        
        logger.info(f"Stored Version {self.paper_version}: {old_title}")
        
        # Update title if changed
        title_changed = False
        if new_title and new_title != old_title:
            self.paper_title = new_title
            self.paper_version += 1
            title_changed = True
            logger.info(f"Paper title changed: {new_title} (Version {self.paper_version})")
        else:
            logger.info("Paper title unchanged")
        
        # Update outline if provided
        if new_outline:
            await outline_memory.update_outline(new_outline)
            logger.info("Outline updated with new structure")
        
        # Clear paper body (keep only placeholders)
        await paper_memory.clear_body_section()
        logger.info("Body section cleared - preserving placeholders")
        
        # Broadcast rewrite started
        await self._broadcast("body_rewrite_started", {
            "version": self.paper_version,
            "title": self.paper_title if self.paper_title else self.user_prompt,
            "title_changed": title_changed,
            "critique_feedback_preview": critique_feedback[:500]
        })
        
        # End critique phase
        await self._end_critique_phase(rewrite=True)
        
        # Reset to body phase with new context
        self.autonomous_section_phase = "body"
        
        # Store critique feedback for passing to construction prompts
        # This provides rewrite context so the model knows what to fix
        self.current_critique_feedback = critique_feedback
        logger.info(f"Stored critique feedback for construction: {len(critique_feedback)} chars")
        
        # Set flag for re-critique if title changed
        if title_changed:
            logger.info("Title changed - will run critique phase again after rewrite completes")
            self.needs_critique_after_rewrite = True
        else:
            logger.info("Title unchanged - will continue to conclusion after rewrite completes")
            self.needs_critique_after_rewrite = False
        
        logger.info("=" * 80)
        logger.info(f"BODY REWRITE PREPARED - Starting body construction for Version {self.paper_version}")
        logger.info("Body reconstruction will have: pre_critique_paper + accepted critique feedback")
        logger.info("=" * 80)
    
    async def _execute_partial_revision(
        self,
        new_title: Optional[str],
        new_outline: Optional[str],
        critique_feedback: str,
        accumulated_history: Optional[str] = None
    ) -> None:
        """
        Execute partial revision using ITERATIVE targeted edit operations.
        
        Proposes edits one at a time, validates each, applies, and shows updated paper
        before proposing the next edit.
        
        Args:
            new_title: New paper title (or None to keep current)
            new_outline: Updated outline (or None to keep current)
            critique_feedback: All accepted critiques
            accumulated_history: Optional accumulated critique history from previous versions
        """
        logger.info("=" * 80)
        logger.info("EXECUTING PARTIAL REVISION (ITERATIVE EDITS)")
        logger.info("=" * 80)
        
        # Mark rewrite as pending (will count as completed only after first successful edit acceptance)
        self.rewrite_pending = True
        logger.info(f"Partial revision initiated (pending successful completion, max: 1)")
        
        # Store current state (for history tracking)
        old_title = self.paper_title if self.paper_title else self.user_prompt
        
        # Update title if changed
        title_changed = False
        if new_title and new_title != old_title:
            self.paper_title = new_title
            self.paper_version += 1
            title_changed = True
            logger.info(f"Paper title changed: {new_title} (Version {self.paper_version})")
        else:
            logger.info("Paper title unchanged")
        
        # Update outline if provided
        if new_outline:
            await outline_memory.update_outline(new_outline)
            logger.info("Outline updated with new structure")
        
        # Get current outline
        current_outline = await outline_memory.get_outline()
        
        # ITERATIVE EDIT LOOP
        MAX_EDITS = 20  # Safety limit to prevent infinite loops
        edits_applied: List[Dict] = []
        successful_edits = 0
        failed_edits = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3
        
        logger.info("Starting iterative edit loop...")
        
        more_edits_needed = True
        while more_edits_needed and len(edits_applied) < MAX_EDITS:
            try:
                # Get current paper state
                current_paper = await paper_memory.get_paper()
                
                # Ask critique submitter for next edit
                logger.info(f"Requesting edit #{len(edits_applied) + 1}...")
                edit_proposal = await self.critique_submitter.submit_iterative_edit(
                    user_prompt=self.user_prompt,
                    pre_critique_paper=self.pre_critique_paper,
                    current_paper=current_paper,
                    current_outline=current_outline,
                    critique_feedback=critique_feedback,
                    edits_applied=edits_applied,
                    accumulated_history=accumulated_history
                )
                
                if edit_proposal is None:
                    logger.error("Failed to get edit proposal - stopping iterative loop")
                    break
                
                operation = edit_proposal.get("operation")
                old_string = edit_proposal.get("old_string", "")
                new_string = edit_proposal.get("new_string", "")
                reasoning = edit_proposal.get("reasoning", "")
                more_edits_needed = edit_proposal.get("more_edits_needed", False)
                
                logger.info(f"Edit proposal: {operation} - {reasoning[:100]}...")
                
                # Validate the edit via validator
                is_valid, validation_reason = await self._validate_partial_revision_edit(
                    edit_proposal=edit_proposal,
                    current_paper=current_paper,
                    current_outline=current_outline,
                    critique_feedback=critique_feedback
                )
                
                if not is_valid:
                    logger.warning(f"Edit #{len(edits_applied) + 1} rejected by validator: {validation_reason}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error(f"Max consecutive failures ({MAX_CONSECUTIVE_FAILURES}) reached - stopping iterative loop")
                        break
                    
                    # Don't add to edits_applied, loop will retry with same state
                    failed_edits += 1
                    continue
                
                # Apply the validated edit
                success = await self._apply_edit(operation, old_string, new_string)
                
                if success:
                    logger.info(f"Edit #{len(edits_applied) + 1} applied successfully")
                    edits_applied.append(edit_proposal)
                    successful_edits += 1
                    consecutive_failures = 0  # Reset on success
                    
                    # Broadcast progress
                    await self._broadcast("partial_revision_edit_applied", {
                        "edit_number": len(edits_applied),
                        "operation": operation,
                        "reasoning": reasoning[:200],
                        "more_edits_needed": more_edits_needed
                    })
                else:
                    logger.warning(f"Edit #{len(edits_applied) + 1} failed to apply (old_string not found)")
                    consecutive_failures += 1
                    failed_edits += 1
                    
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error(f"Max consecutive failures ({MAX_CONSECUTIVE_FAILURES}) reached - stopping iterative loop")
                        break
                
            except Exception as e:
                logger.error(f"Error in iterative edit loop: {e}", exc_info=True)
                consecutive_failures += 1
                failed_edits += 1
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(f"Max consecutive failures ({MAX_CONSECUTIVE_FAILURES}) reached - stopping iterative loop")
                    break
        
        logger.info(f"Iterative edit loop complete: {successful_edits} successful, {failed_edits} failed")
        
        if len(edits_applied) >= MAX_EDITS:
            logger.warning(f"Reached max edit limit ({MAX_EDITS}) - stopping iterative loop")
        
        # Mark rewrite completion on first successful edit
        if self.rewrite_pending:
            if successful_edits > 0:
                self.rewrite_count += 1
                logger.info(f"Rewrite #{self.rewrite_count} completed successfully (first accepted partial edit)")
            self.rewrite_pending = False
        
        # Broadcast partial revision complete
        await self._broadcast("partial_revision_complete", {
            "version": self.paper_version,
            "title": self.paper_title if self.paper_title else self.user_prompt,
            "title_changed": title_changed,
            "edits_applied": successful_edits,
            "edits_failed": failed_edits,
            "critique_feedback_preview": critique_feedback[:500]
        })
        
        # End critique phase
        await self._end_critique_phase(rewrite=False)
        
        # Set flag for re-critique if title changed
        if title_changed:
            logger.info("Title changed - would run critique phase again, but max rewrites reached")
            # Note: With max 1 rewrite, title changes won't trigger re-critique
            self.needs_critique_after_rewrite = False
        else:
            logger.info("Title unchanged - continuing to conclusion")
            self.needs_critique_after_rewrite = False
        
        # Continue to conclusion (partial revision doesn't loop back to body phase)
        # Clear critique context (no longer needed after body phase)
        self.current_critique_feedback = None
        self.autonomous_section_phase = "conclusion"
        
        logger.info("=" * 80)
        logger.info("PARTIAL REVISION COMPLETE - Continuing to CONCLUSION")
        logger.info("=" * 80)
    
    async def _validate_partial_revision_edit(
        self,
        edit_proposal: Dict,
        current_paper: str,
        current_outline: str,
        critique_feedback: str
    ) -> Tuple[bool, str]:
        """
        Validate a single partial revision edit using the compiler validator.
        
        Args:
            edit_proposal: The proposed edit with operation, old_string, new_string, reasoning
            current_paper: Current paper content
            current_outline: Paper outline
            critique_feedback: All accepted critiques
            
        Returns:
            Tuple of (is_valid: bool, rejection_reason: str)
        """
        try:
            # Delegate to the compiler validator which has comprehensive validation logic
            return await self.validator.validate_partial_revision_edit(
                edit_proposal=edit_proposal,
                current_paper=current_paper,
                current_outline=current_outline,
                critique_feedback=critique_feedback
            )
            
        except Exception as e:
            logger.error(f"Error validating partial revision edit: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"
    
    def _format_accumulated_critique_history(self) -> str:
        """
        Format all historical critiques from previous failed versions.
        Returns formatted string with clear version labeling.
        """
        if not self.accumulated_critique_history:
            return ""
        
        parts = ["=" * 80]
        parts.append("CRITIQUE HISTORY FROM PREVIOUS FAILED VERSIONS")
        parts.append("(These critiques are from earlier attempts that were rewritten)")
        parts.append("=" * 80 + "\n")
        
        for i, entry in enumerate(self.accumulated_critique_history, 1):
            parts.append(f"--- FAILED VERSION #{i} (REWRITTEN) ---")
            parts.append(entry['critiques'])
            parts.append("")
        
        return "\n".join(parts)
    
    async def _end_critique_phase(self, rewrite: bool) -> None:
        """
        End critique phase and clean up.
        
        Args:
            rewrite: Whether a rewrite was approved
        """
        logger.info(f"Ending critique phase (rewrite={rewrite})")
        
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
            "rewrite": rewrite,
            "version": self.paper_version
        })
        
        if not rewrite:
            # Continue to conclusion
            # Clear critique context (no longer needed after body phase)
            self.current_critique_feedback = None
            self.autonomous_section_phase = "conclusion"
            logger.info("Critique phase complete - transitioning to CONCLUSION phase")
            await self._broadcast("phase_transition", {
                "from_phase": "critique",
                "to_phase": "conclusion",
                "trigger": "critiques_reviewed",
                "paper_word_count": await paper_memory.get_word_count()
            })
        else:
            logger.info("Critique phase complete - body will be rewritten")
    
    async def _skip_rewrite_and_continue(self) -> None:
        """
        Skip rewrite phase when body is academically acceptable.
        Called when 5 total attempts complete with 0 accepted critiques.
        """
        logger.info("=" * 80)
        logger.info("SKIPPING REWRITE - No critiques accepted, body is acceptable")
        logger.info("=" * 80)
        
        await self._broadcast("critique_phase_skipped", {
            "reason": "no_critiques_accepted",
            "version": self.paper_version
        })
        
        # End critique phase without rewrite
        await self._end_critique_phase(rewrite=False)
        
        # The _end_critique_phase already transitions to conclusion when rewrite=False
        logger.info("Transitioning to CONCLUSION phase (body accepted as-is)")
    
    async def skip_critique_phase(self) -> bool:
        """
        Skip the critique phase and continue to conclusion.
        User override to bypass peer review and rewrite cycle.
        
        Can be called:
        - During critique phase: immediately skips
        - Before critique phase: sets flag to auto-skip when reached
        
        Returns:
            True if successfully skipped or queued for skip
        """
        if self.in_critique_phase:
            # Currently in critique phase - skip immediately
            logger.info("=" * 80)
            logger.info("USER OVERRIDE: Skipping critique phase NOW, continuing to conclusion")
            logger.info("=" * 80)
            
            await self._broadcast("critique_phase_skipped", {
                "reason": "user_override",
                "version": self.paper_version
            })
            
            await self._end_critique_phase(rewrite=False)
            return True
        else:
            # Not in critique phase yet - set flag to skip when reached
            logger.info("=" * 80)
            logger.info("USER OVERRIDE: Pre-emptive critique skip requested - will skip when phase is reached")
            logger.info("=" * 80)
            
            self._skip_critique_requested = True
            
            await self._broadcast("critique_skip_queued", {
                "message": "Critique phase will be skipped when reached",
                "version": self.paper_version
            })
            
            return True
    
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
            # Check if max rewrites reached - skip critique phase entirely
            if self.rewrite_count >= 1:
                logger.info(f"Max rewrites ({self.rewrite_count}) reached - skipping critique phase, proceeding to conclusion")
                # Clear critique context (no longer needed after body phase)
                self.current_critique_feedback = None
                self.autonomous_section_phase = "conclusion"
                await self._broadcast("phase_transition", {
                    "from_phase": "body",
                    "to_phase": "conclusion",
                    "trigger": "section_complete",
                    "reason": "max_rewrites_reached",
                    "rewrite_count": self.rewrite_count,
                    "paper_word_count": word_count
                })
                return False
            
            # Check if this is a rewrite completion that needs another critique round
            if self.needs_critique_after_rewrite:
                # Body rewrite complete, title changed - run critique phase again
                logger.info(f"Body rewrite complete (Version {self.paper_version}) - triggering ANOTHER critique phase (title changed)")
                self.needs_critique_after_rewrite = False  # Reset flag
                
                await self._broadcast("phase_transition", {
                    "from_phase": "body",
                    "to_phase": "critique",
                    "trigger": "rewrite_complete_title_changed",
                    "paper_word_count": word_count,
                    "version": self.paper_version
                })
                
                # Start critique aggregation sub-workflow again
                await self._start_critique_phase()
                return False
            
            # Check if this is a rewrite completion with unchanged title - skip to conclusion
            if self.rewrite_count > 0:
                # Rewrite completed but title unchanged - critique loop ends, proceed to conclusion
                logger.info(f"Rewrite #{self.rewrite_count} complete (title unchanged) - skipping additional critique, proceeding to conclusion")
                # Clear critique context (no longer needed after body phase)
                self.current_critique_feedback = None
                self.autonomous_section_phase = "conclusion"
                await self._broadcast("phase_transition", {
                    "from_phase": "body",
                    "to_phase": "conclusion",
                    "trigger": "rewrite_complete_title_unchanged",
                    "rewrite_count": self.rewrite_count,
                    "paper_word_count": word_count
                })
                return False
            
            # BODY COMPLETE - TRIGGER CRITIQUE PHASE BEFORE CONCLUSION (first time only)
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
                r"(?:^|\n)\s*(?:#+\s*)?(?:[IVXLCDM]+\.?\s*)?(?:Conclusion|Summary|Discussion|Final\s*Remarks|Concluding\s*Remarks)",
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
                r"(?:^|\n)\s*(?:#+\s*)?(?:I\.?\s*)?Introduction",
                current_paper, re.IGNORECASE | re.MULTILINE
            ))
            
            if not has_introduction:
                logger.error("Cannot transition from introduction phase: No Introduction section found in paper")
                return False  # Block transition
            
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
                r"(?:^|\n)\s*(?:#+\s*)?\*{0,2}Abstract\*{0,2}",
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
            miniscule_edit_count=self.miniscule_edit_count,
            in_critique_phase=self.in_critique_phase,
            critique_acceptances=self.critique_acceptances,
            paper_version=self.paper_version,
            skip_critique_requested=self._skip_critique_requested
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
        self.miniscule_edit_count = 0
        self.construction_cycle_count = 0
        self.rigor_cycle_active = False
        
        # Reset autonomous mode state (if in autonomous mode)
        if self.autonomous_mode:
            self.autonomous_section_phase = "body"  # Reset to body phase
            logger.info("Reset autonomous section phase to body")
        
        # Reset critique phase state
        self.in_critique_phase = False
        self.critique_acceptances = 0
        self.paper_version = 1
        self.rewrite_count = 0
        self.rewrite_pending = False
        self.accumulated_critique_history.clear()
        self.previous_body_versions.clear()
        self.needs_critique_after_rewrite = False
        self.paper_title = None
        self._skip_critique_requested = False
        self.pre_critique_paper = None
        self.current_critique_feedback = None
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

