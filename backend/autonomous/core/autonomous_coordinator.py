"""
Autonomous Coordinator - Main orchestrator for autonomous research mode.
Manages the two-tier workflow: brainstorm aggregation -> paper compilation.
"""
import asyncio
import logging
import os
import re
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path

import aiofiles

from backend.shared.config import system_config
from backend.shared.models import (
    AutonomousResearchState,
    BrainstormMetadata,
    TopicSelectionSubmission,
    SubmitterConfig,
    WorkflowTask,
    ModelConfig
)
from backend.shared.api_client_manager import api_client_manager
from backend.shared.workflow_predictor import workflow_predictor

# Memory managers
from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.research_metadata import research_metadata
from backend.autonomous.memory.autonomous_rejection_logs import autonomous_rejection_logs
from backend.autonomous.memory.session_manager import session_manager
from backend.autonomous.memory.autonomous_api_logger import autonomous_api_logger

# RAG manager
from backend.autonomous.core.autonomous_rag_manager import autonomous_rag_manager

# Agents
from backend.autonomous.agents.topic_selector import TopicSelectorAgent
from backend.autonomous.agents.topic_validator import TopicValidatorAgent
from backend.autonomous.agents.completion_reviewer import CompletionReviewerAgent
from backend.autonomous.agents.reference_selector import ReferenceSelectorAgent
from backend.autonomous.agents.paper_title_selector import PaperTitleSelectorAgent

# Validation
from backend.autonomous.validation.paper_redundancy_checker import PaperRedundancyChecker

# Tier 3: Final Answer Agents
from backend.autonomous.agents.final_answer.certainty_assessor import CertaintyAssessor
from backend.autonomous.agents.final_answer.answer_format_selector import AnswerFormatSelector
from backend.autonomous.agents.final_answer.volume_organizer import VolumeOrganizer
from backend.autonomous.memory.final_answer_memory import final_answer_memory
from backend.autonomous.memory.paper_model_tracker import PaperModelTracker

# Part 1 Aggregator Integration
from backend.aggregator.core.coordinator import Coordinator as AggregatorCoordinator
from backend.aggregator.memory.shared_training import shared_training_memory

# Part 2 Compiler Integration
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.compiler.memory.paper_memory import paper_memory as compiler_paper_memory
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.core.compiler_rag_manager import compiler_rag_manager

# RAG manager for document loading
from backend.aggregator.core.rag_manager import rag_manager

# API Client Manager for model tracking
from backend.shared.api_client_manager import api_client_manager

logger = logging.getLogger(__name__)


class AutonomousCoordinator:
    """
    Main orchestrator for autonomous research mode.
    Manages topic selection, brainstorm aggregation, and paper compilation.
    """
    
    def __init__(self):
        # State
        self._running = False
        self._state = AutonomousResearchState()
        self._stop_event = asyncio.Event()
        
        # Configuration (set during initialize)
        self._user_research_prompt: str = ""
        self._submitter_configs: List[SubmitterConfig] = []  # Per-submitter configs for brainstorm aggregation
        self._validator_model: str = ""
        self._validator_context: int = 131072
        self._validator_max_tokens: int = 15000
        self._validator_provider: str = "lm_studio"
        self._validator_openrouter_provider: Optional[str] = None
        self._validator_lm_studio_fallback: Optional[str] = None
        
        # Compiler models (separate from aggregator submitters)
        self._high_context_model: str = ""
        self._high_param_model: str = ""
        self._high_context_context: int = 131072
        self._high_param_context: int = 10000
        self._high_context_max_tokens: int = 25000
        self._high_param_max_tokens: int = 15000
        
        # Agents (initialized during setup)
        self._topic_selector: Optional[TopicSelectorAgent] = None
        self._topic_validator: Optional[TopicValidatorAgent] = None
        self._completion_reviewer: Optional[CompletionReviewerAgent] = None
        self._reference_selector: Optional[ReferenceSelectorAgent] = None
        self._title_selector: Optional[PaperTitleSelectorAgent] = None
        self._redundancy_checker: Optional[PaperRedundancyChecker] = None
        
        # Tier 3: Final Answer Agents
        self._certainty_assessor: Optional[CertaintyAssessor] = None
        self._format_selector: Optional[AnswerFormatSelector] = None
        self._volume_organizer: Optional[VolumeOrganizer] = None
        
        # Part 1 & 2 Integration
        self._brainstorm_aggregator: Optional[AggregatorCoordinator] = None
        self._paper_compiler: Optional[CompilerCoordinator] = None
        
        # Callbacks
        self._broadcast_callback: Optional[Callable] = None
        
        # Workflow tracking
        self._current_topic_id: Optional[str] = None
        self._current_paper_id: Optional[str] = None
        self._current_paper_title: Optional[str] = None
        self._current_reference_papers: List[str] = []  # Reference papers for current topic cycle
        self._acceptance_count: int = 0
        self._rejection_count: int = 0
        self._cleanup_removals: int = 0  # Track actual cleanup/pruning removals from aggregator
        self._consecutive_rejections: int = 0
        self._exhaustion_signals: int = 0
        self._papers_completed_count: int = 0
        self._last_redundancy_check_at: int = 0
        self._manual_paper_writing_triggered: bool = False
        self._resume_paper_phase: Optional[str] = None  # Saved phase for resume (body/conclusion/intro/abstract)
        
        # Tier 3 Final Answer tracking
        self._last_tier3_check_at: int = 0  # Paper count at last Tier 3 check
        self._tier3_active: bool = False  # Is Tier 3 final answer generation active
        self._force_tier3_after_paper: bool = False  # Force Tier 3 after current paper completes
        self._force_tier3_immediate: bool = False  # Force Tier 3 immediately (skip incomplete work)
        
        # Per-paper model tracking (tracks API calls for current paper being built)
        self._current_paper_tracker: Optional[PaperModelTracker] = None
        
        # Workflow task tracking (for WorkflowPanel)
        self.workflow_tasks: List['WorkflowTask'] = []
        self.completed_task_ids: set = set()
        self.current_task_sequence: int = 0
        self.current_task_id: Optional[str] = None
    
    def set_broadcast_callback(self, callback: Callable) -> None:
        """Set callback for broadcasting WebSocket events."""
        self._broadcast_callback = callback
    
    async def _broadcast(self, event: str, data: Dict[str, Any] = None) -> None:
        """Broadcast an event through WebSocket."""
        if self._broadcast_callback:
            # broadcast_event expects (event_type, data) as separate arguments
            await self._broadcast_callback(event, data or {})
    
    async def initialize(
        self,
        user_research_prompt: str,
        submitter_configs: List[SubmitterConfig],
        validator_model: str,
        validator_context_window: int = 131072,
        validator_max_tokens: int = 15000,
        high_context_model: str = "",
        high_context_context_window: int = 131072,
        high_context_max_tokens: int = 25000,
        high_param_model: str = "",
        high_param_context_window: int = 10000,
        high_param_max_tokens: int = 15000,
        critique_submitter_model: str = "",
        critique_submitter_context_window: int = 131072,
        critique_submitter_max_tokens: int = 25000,
        # OpenRouter provider configs for validator
        validator_provider: str = "lm_studio",
        validator_openrouter_provider: Optional[str] = None,
        validator_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider configs for high-context submitter
        high_context_provider: str = "lm_studio",
        high_context_openrouter_provider: Optional[str] = None,
        high_context_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider configs for high-param submitter
        high_param_provider: str = "lm_studio",
        high_param_openrouter_provider: Optional[str] = None,
        high_param_lm_studio_fallback: Optional[str] = None,
        # OpenRouter provider configs for critique submitter
        critique_submitter_provider: str = "lm_studio",
        critique_submitter_openrouter_provider: Optional[str] = None,
        critique_submitter_lm_studio_fallback: Optional[str] = None
    ) -> None:
        """Initialize the coordinator with configuration."""
        # Store configuration
        self._user_research_prompt = user_research_prompt
        self._submitter_configs = submitter_configs
        self._validator_model = validator_model
        self._validator_context = validator_context_window
        self._validator_max_tokens = validator_max_tokens
        
        # Use first submitter config for autonomous agents (topic selector, etc.)
        # These agents are single-instance, not parallel like brainstorm submitters
        first_submitter_model = submitter_configs[0].model_id if submitter_configs else ""
        first_submitter_context = submitter_configs[0].context_window if submitter_configs else 131072
        first_submitter_max_tokens = submitter_configs[0].max_output_tokens if submitter_configs else 25000
        
        # Compiler settings (separate from aggregator submitters)
        # Fallback to first submitter model if compiler models not specified
        self._high_context_model = high_context_model if high_context_model else first_submitter_model
        self._high_param_model = high_param_model if high_param_model else first_submitter_model
        self._high_context_context = high_context_context_window
        self._high_param_context = high_param_context_window
        self._high_context_max_tokens = high_context_max_tokens
        self._high_param_max_tokens = high_param_max_tokens
        # Critique submitter fallback: use high_context_model if not specified
        self._critique_submitter_model = critique_submitter_model if critique_submitter_model else self._high_context_model
        self._critique_submitter_context = critique_submitter_context_window
        self._critique_submitter_max_tokens = critique_submitter_max_tokens
        
        # Store OpenRouter provider configs for all roles
        self._validator_provider = validator_provider
        self._validator_openrouter_provider = validator_openrouter_provider
        self._validator_lm_studio_fallback = validator_lm_studio_fallback
        self._high_context_provider = high_context_provider
        self._high_context_openrouter_provider = high_context_openrouter_provider
        self._high_context_lm_studio_fallback = high_context_lm_studio_fallback
        self._high_param_provider = high_param_provider
        self._high_param_openrouter_provider = high_param_openrouter_provider
        self._high_param_lm_studio_fallback = high_param_lm_studio_fallback
        self._critique_submitter_provider = critique_submitter_provider
        self._critique_submitter_openrouter_provider = critique_submitter_openrouter_provider
        self._critique_submitter_lm_studio_fallback = critique_submitter_lm_studio_fallback
        
        logger.info(f"Autonomous coordinator initializing with {len(submitter_configs)} submitters")
        for config in submitter_configs:
            label = "(Main Submitter)" if config.submitter_id == 1 else ""
            logger.info(f"  Submitter {config.submitter_id} {label}: model={config.model_id}, context={config.context_window}")
        
        # PRIORITY 1: Check for interrupted session in auto_sessions/
        # This takes precedence over legacy paths and new session creation
        interrupted_session = await session_manager.find_interrupted_session(system_config.auto_sessions_base_dir)
        
        if interrupted_session:
            session_id = interrupted_session["session_id"]
            logger.info(f"Found interrupted session: {session_id}")
            logger.info(f"  User prompt: {interrupted_session['user_prompt'][:100]}...")
            logger.info(f"  Last updated: {interrupted_session['last_updated']}")
            logger.info(f"  Tier: {interrupted_session['workflow_state'].get('current_tier')}")
            logger.info(f"  Topic: {interrupted_session['workflow_state'].get('current_topic_id')}")
            logger.info(f"  Acceptances: {interrupted_session['workflow_state'].get('acceptance_count', 0)}")
            
            # Resume the interrupted session
            await session_manager.resume_session(session_id, system_config.auto_sessions_base_dir)
            logger.info(f"Session resumed: {session_manager.session_id}")
            
            # Configure memory systems to use session paths
            brainstorm_memory.set_session_manager(session_manager)
            paper_library.set_session_manager(session_manager)
            research_metadata.set_session_manager(session_manager)
            final_answer_memory.set_session_manager(session_manager)
            
            # Override the user_research_prompt with the one from the interrupted session
            # This ensures we continue with the same research goal
            self._user_research_prompt = interrupted_session["user_prompt"]
        else:
            # PRIORITY 2: Check for existing legacy data 
            # If legacy data exists, use it instead of creating empty new session
            legacy_papers_dir = Path(system_config.auto_papers_dir)
            legacy_brainstorms_dir = Path(system_config.auto_brainstorms_dir)
            
            # Count existing papers and brainstorms in legacy locations
            legacy_paper_count = len(list(legacy_papers_dir.glob("paper_*.txt"))) if legacy_papers_dir.exists() else 0
            legacy_brainstorm_count = len(list(legacy_brainstorms_dir.glob("brainstorm_*.txt"))) if legacy_brainstorms_dir.exists() else 0
            
            use_legacy_paths = legacy_paper_count > 0 or legacy_brainstorm_count > 0
            
            if use_legacy_paths:
                logger.info(f"Found existing legacy data: {legacy_paper_count} papers, {legacy_brainstorm_count} brainstorms")
                logger.info("Using legacy paths instead of creating new session (to preserve existing work)")
                # Don't set session manager - memory modules will use default legacy paths
                # Clear any previous session manager state
                await session_manager.clear()
            else:
                # PRIORITY 3: No interrupted session, no legacy data - create new session folder
                await session_manager.initialize(user_research_prompt, system_config.auto_sessions_base_dir)
                logger.info(f"New session initialized: {session_manager.session_id}")
                
                # Configure memory systems to use session paths
                brainstorm_memory.set_session_manager(session_manager)
                paper_library.set_session_manager(session_manager)
                research_metadata.set_session_manager(session_manager)
                final_answer_memory.set_session_manager(session_manager)
        
        # Initialize memory systems
        await brainstorm_memory.initialize()
        await paper_library.initialize()
        await research_metadata.initialize(user_research_prompt)
        await autonomous_rejection_logs.initialize()
        
        # CRITICAL: Reset and clear all RAG state for fresh autonomous session
        # This prevents cross-contamination from Part 1 manual mode
        # Autonomous mode should start with a clean RAG that only contains:
        # - Brainstorm database (for the current topic)
        # - Reference papers (selected by the reference selector)
        # Old user uploads or Part 1 aggregator content must be removed
        logger.info("Resetting RAG state for fresh autonomous research mode...")
        autonomous_rag_manager.reset()  # Reset tracking state (indexed sets)
        await asyncio.to_thread(rag_manager.clear_all_documents)  # Clear all RAG content (non-blocking)
        logger.info("RAG state reset and cleared for autonomous mode")
        
        # Now initialize with fresh state
        await autonomous_rag_manager.initialize()
        
        # Initialize agents (use first submitter for single-instance agents)
        self._topic_selector = TopicSelectorAgent(
            model_id=first_submitter_model,
            context_window=first_submitter_context,
            max_output_tokens=first_submitter_max_tokens
        )
        
        self._topic_validator = TopicValidatorAgent(
            model_id=validator_model,
            context_window=validator_context_window,
            max_output_tokens=validator_max_tokens
        )
        
        self._completion_reviewer = CompletionReviewerAgent(
            model_id=first_submitter_model,  # Same model for self-validation
            context_window=first_submitter_context,
            max_output_tokens=first_submitter_max_tokens
        )
        
        self._reference_selector = ReferenceSelectorAgent(
            model_id=first_submitter_model,
            context_window=first_submitter_context,
            max_output_tokens=first_submitter_max_tokens
        )
        
        self._title_selector = PaperTitleSelectorAgent(
            model_id=first_submitter_model,
            validator_model_id=validator_model,
            context_window=first_submitter_context,
            max_output_tokens=first_submitter_max_tokens
        )
        
        self._redundancy_checker = PaperRedundancyChecker(
            model_id=validator_model,
            context_window=validator_context_window,
            max_output_tokens=validator_max_tokens
        )
        
        # Initialize Tier 3 Final Answer Agents
        self._certainty_assessor = CertaintyAssessor(
            submitter_model=first_submitter_model,
            validator_model=validator_model,
            context_window=first_submitter_context,
            max_output_tokens=first_submitter_max_tokens
        )
        
        self._format_selector = AnswerFormatSelector(
            submitter_model=first_submitter_model,
            validator_model=validator_model,
            context_window=first_submitter_context,
            max_output_tokens=first_submitter_max_tokens
        )
        
        self._volume_organizer = VolumeOrganizer(
            submitter_model=first_submitter_model,
            validator_model=validator_model,
            context_window=first_submitter_context,
            max_output_tokens=first_submitter_max_tokens
        )
        
        # Initialize Tier 3 memory
        await final_answer_memory.initialize()
        
        # CRITICAL: Configure roles with api_client_manager so routing works correctly
        # Configure first submitter (used by topic selector, completion reviewer, reference selector, title selector)
        first_config = submitter_configs[0] if submitter_configs else SubmitterConfig(submitter_id=1, model_id=first_submitter_model)
        api_client_manager.configure_role(
            "autonomous_topic_selector",
            ModelConfig(
                provider=first_config.provider if hasattr(first_config, 'provider') else "lm_studio",
                model_id=first_submitter_model,
                openrouter_model_id=first_config.openrouter_model_id if hasattr(first_config, 'openrouter_model_id') else None,
                openrouter_provider=first_config.openrouter_provider if hasattr(first_config, 'openrouter_provider') else None,
                lm_studio_fallback_id=first_config.lm_studio_fallback_id if hasattr(first_config, 'lm_studio_fallback_id') else None,
                context_window=first_submitter_context,
                max_output_tokens=first_submitter_max_tokens
            )
        )
        
        api_client_manager.configure_role(
            "autonomous_completion_reviewer",
            ModelConfig(
                provider=first_config.provider if hasattr(first_config, 'provider') else "lm_studio",
                model_id=first_submitter_model,
                openrouter_model_id=first_config.openrouter_model_id if hasattr(first_config, 'openrouter_model_id') else None,
                openrouter_provider=first_config.openrouter_provider if hasattr(first_config, 'openrouter_provider') else None,
                lm_studio_fallback_id=first_config.lm_studio_fallback_id if hasattr(first_config, 'lm_studio_fallback_id') else None,
                context_window=first_submitter_context,
                max_output_tokens=first_submitter_max_tokens
            )
        )
        
        api_client_manager.configure_role(
            "autonomous_reference_selector",
            ModelConfig(
                provider=first_config.provider if hasattr(first_config, 'provider') else "lm_studio",
                model_id=first_submitter_model,
                openrouter_model_id=first_config.openrouter_model_id if hasattr(first_config, 'openrouter_model_id') else None,
                openrouter_provider=first_config.openrouter_provider if hasattr(first_config, 'openrouter_provider') else None,
                lm_studio_fallback_id=first_config.lm_studio_fallback_id if hasattr(first_config, 'lm_studio_fallback_id') else None,
                context_window=first_submitter_context,
                max_output_tokens=first_submitter_max_tokens
            )
        )
        
        api_client_manager.configure_role(
            "autonomous_paper_title_selector",
            ModelConfig(
                provider=first_config.provider if hasattr(first_config, 'provider') else "lm_studio",
                model_id=first_submitter_model,
                openrouter_model_id=first_config.openrouter_model_id if hasattr(first_config, 'openrouter_model_id') else None,
                openrouter_provider=first_config.openrouter_provider if hasattr(first_config, 'openrouter_provider') else None,
                lm_studio_fallback_id=first_config.lm_studio_fallback_id if hasattr(first_config, 'lm_studio_fallback_id') else None,
                context_window=first_submitter_context,
                max_output_tokens=first_submitter_max_tokens
            )
        )
        
        # Configure validator
        api_client_manager.configure_role(
            "autonomous_topic_validator",
            ModelConfig(
                provider=validator_provider,
                model_id=validator_model,
                openrouter_model_id=validator_model if validator_provider == "openrouter" else None,
                openrouter_provider=validator_openrouter_provider,
                lm_studio_fallback_id=validator_lm_studio_fallback,
                context_window=validator_context_window,
                max_output_tokens=validator_max_tokens
            )
        )
        
        api_client_manager.configure_role(
            "autonomous_paper_redundancy_checker",
            ModelConfig(
                provider=validator_provider,
                model_id=validator_model,
                openrouter_model_id=validator_model if validator_provider == "openrouter" else None,
                openrouter_provider=validator_openrouter_provider,
                lm_studio_fallback_id=validator_lm_studio_fallback,
                context_window=validator_context_window,
                max_output_tokens=validator_max_tokens
            )
        )
        
        # Configure Tier 3 Final Answer agents (certainty assessor, format selector, volume organizer)
        # These use the first submitter model configuration
        api_client_manager.configure_role(
            "autonomous_certainty_assessor",
            ModelConfig(
                provider=first_config.provider if hasattr(first_config, 'provider') else "lm_studio",
                model_id=first_submitter_model,
                openrouter_model_id=first_config.openrouter_model_id if hasattr(first_config, 'openrouter_model_id') else None,
                openrouter_provider=first_config.openrouter_provider if hasattr(first_config, 'openrouter_provider') else None,
                lm_studio_fallback_id=first_config.lm_studio_fallback_id if hasattr(first_config, 'lm_studio_fallback_id') else None,
                context_window=first_submitter_context,
                max_output_tokens=first_submitter_max_tokens
            )
        )
        
        api_client_manager.configure_role(
            "autonomous_format_selector",
            ModelConfig(
                provider=first_config.provider if hasattr(first_config, 'provider') else "lm_studio",
                model_id=first_submitter_model,
                openrouter_model_id=first_config.openrouter_model_id if hasattr(first_config, 'openrouter_model_id') else None,
                openrouter_provider=first_config.openrouter_provider if hasattr(first_config, 'openrouter_provider') else None,
                lm_studio_fallback_id=first_config.lm_studio_fallback_id if hasattr(first_config, 'lm_studio_fallback_id') else None,
                context_window=first_submitter_context,
                max_output_tokens=first_submitter_max_tokens
            )
        )
        
        api_client_manager.configure_role(
            "autonomous_volume_organizer",
            ModelConfig(
                provider=first_config.provider if hasattr(first_config, 'provider') else "lm_studio",
                model_id=first_submitter_model,
                openrouter_model_id=first_config.openrouter_model_id if hasattr(first_config, 'openrouter_model_id') else None,
                openrouter_provider=first_config.openrouter_provider if hasattr(first_config, 'openrouter_provider') else None,
                lm_studio_fallback_id=first_config.lm_studio_fallback_id if hasattr(first_config, 'lm_studio_fallback_id') else None,
                context_window=first_submitter_context,
                max_output_tokens=first_submitter_max_tokens
            )
        )
        
        logger.info("Configured Tier 3 Final Answer agents with api_client_manager")
        
        # Set up task tracking callbacks for workflow panel integration
        self._topic_selector.set_task_tracking_callback(self._handle_task_event)
        self._topic_validator.set_task_tracking_callback(self._handle_task_event)
        self._completion_reviewer.set_task_tracking_callback(self._handle_task_event)
        self._reference_selector.set_task_tracking_callback(self._handle_task_event)
        self._title_selector.set_task_tracking_callback(self._handle_task_event)
        self._redundancy_checker.set_task_tracking_callback(self._handle_task_event)
        self._certainty_assessor.set_task_tracking_callback(self._handle_task_event)
        self._format_selector.set_task_tracking_callback(self._handle_task_event)
        self._volume_organizer.set_task_tracking_callback(self._handle_task_event)
        
        # Load existing stats
        stats = await research_metadata.get_stats()
        self._papers_completed_count = stats.get("total_papers_completed", 0)
        self._last_redundancy_check_at = self._papers_completed_count
        
        # Check for interrupted workflow (crash recovery) BEFORE initializing tier3 tracking
        # This ensures we restore the saved value if it exists
        has_crash_recovery = research_metadata.has_interrupted_workflow()
        await self._check_resume_state()
        
        # Initialize Tier 3 check tracking from actual library count
        # (only if not restored from crash recovery, or if restored value was 0)
        if not has_crash_recovery or self._last_tier3_check_at == 0:
            paper_counts = await paper_library.count_papers()
            self._last_tier3_check_at = paper_counts["active"]
        
        logger.info("AutonomousCoordinator initialized")
        
        # Initialize workflow predictions
        await self.refresh_workflow_predictions()
    
    async def _check_resume_state(self) -> None:
        """Check if there's an interrupted workflow to resume."""
        if research_metadata.has_interrupted_workflow():
            workflow_state = await research_metadata.get_workflow_state()
            logger.info(f"Found interrupted workflow state: tier={workflow_state.get('current_tier')}")
            
            # Restore internal state from saved workflow state
            self._current_topic_id = workflow_state.get("current_topic_id")
            self._current_paper_id = workflow_state.get("current_paper_id")
            self._current_reference_papers = workflow_state.get("reference_paper_ids", [])
            self._acceptance_count = workflow_state.get("acceptance_count", 0)
            self._rejection_count = workflow_state.get("rejection_count", 0)
            self._consecutive_rejections = workflow_state.get("consecutive_rejections", 0)
            self._exhaustion_signals = workflow_state.get("exhaustion_signals", 0)
            self._papers_completed_count = workflow_state.get("papers_completed_count", 0)
            self._last_redundancy_check_at = workflow_state.get("last_redundancy_check_at", 0)
            self._last_tier3_check_at = workflow_state.get("last_tier3_check_at", 0)
            
            # Restore Tier 3 flags for proper resume
            self._tier3_active = workflow_state.get("tier3_active", False)
            
            # CRITICAL: Restore paper phase for proper resume
            # This ensures the compiler continues from the correct phase (body/conclusion/intro/abstract)
            self._resume_paper_phase = workflow_state.get("paper_phase")
            
            # Get Tier 3 specific info for logging
            tier3_phase = workflow_state.get("tier3_phase")
            tier3_format = workflow_state.get("tier3_format")
            
            logger.info(f"Workflow state restored: topic={self._current_topic_id}, "
                       f"paper={self._current_paper_id}, phase={self._resume_paper_phase}, "
                       f"acceptances={self._acceptance_count}, "
                       f"reference_papers={len(self._current_reference_papers)}, "
                       f"tier3_active={self._tier3_active}, tier3_phase={tier3_phase}, "
                       f"tier3_format={tier3_format}")
        else:
            self._resume_paper_phase = None
            logger.info("No interrupted workflow found - checking for incomplete papers")
            
            # Check for incomplete papers that were saved mid-construction
            # This handles the case where a paper was saved but not completed
            await self._check_for_incomplete_papers()
    
    async def _check_for_incomplete_papers(self) -> None:
        """
        Check for incomplete papers (papers with placeholders that need to be resumed).
        
        This handles the scenario where:
        1. A paper was being written
        2. The system was stopped/crashed
        3. The paper was saved (to prevent data loss)
        4. But the paper is incomplete (has placeholders)
        
        In this case, we set up the resume state to continue writing the incomplete paper.
        """
        incomplete_paper = await paper_library.get_most_recent_incomplete_paper()
        
        if incomplete_paper:
            logger.info(f"Found incomplete paper: {incomplete_paper.paper_id} "
                       f"(title: {incomplete_paper.title}, "
                       f"from brainstorm: {incomplete_paper.source_brainstorm_ids})")
            
            # Set up resume state for the incomplete paper
            self._current_paper_id = incomplete_paper.paper_id
            self._current_paper_title = incomplete_paper.title
            
            # Get source brainstorm (use first one if multiple)
            if incomplete_paper.source_brainstorm_ids:
                self._current_topic_id = incomplete_paper.source_brainstorm_ids[0]
            
            # Restore reference papers
            self._current_reference_papers = incomplete_paper.referenced_papers or []
            
            # Detect which phase the paper needs to resume from based on content
            paper_content = await self._get_paper_content_for_resume(incomplete_paper.paper_id)
            self._resume_paper_phase = self._detect_paper_phase(paper_content)
            
            logger.info(f"Will resume incomplete paper {incomplete_paper.paper_id} "
                       f"from phase: {self._resume_paper_phase}")
            
            # Save workflow state so the resume logic kicks in
            await self._save_workflow_state(tier="tier2_paper_writing", phase=self._resume_paper_phase)
    
    async def _get_paper_content_for_resume(self, paper_id: str) -> str:
        """Get paper content for detecting resume phase."""
        try:
            paper_path = paper_library._get_paper_path(paper_id)
            if paper_path.exists():
                async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                    return await f.read()
        except Exception as e:
            logger.error(f"Failed to read paper content for {paper_id}: {e}")
        return ""
    
    def _detect_paper_phase(self, paper_content: str) -> str:
        """
        Detect which phase a paper is in based on its content.
        
        Phase order: body -> conclusion -> introduction -> abstract
        
        Returns the phase that needs to be written next.
        
        Uses actual section existence as source of truth, with placeholder checks
        as fallback for robustness against file corruption.
        """
        # Check for actual sections (source of truth)
        has_abstract = self._has_section(paper_content, "Abstract")
        has_intro = self._has_section(paper_content, "Introduction")
        has_conclusion = self._has_section(paper_content, "Conclusion")
        
        # Also check placeholders as fallback indicator
        has_abstract_placeholder = "[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION" in paper_content
        has_intro_placeholder = "[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION" in paper_content
        has_conclusion_placeholder = "[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION" in paper_content
        
        # Check for body content (Roman numeral sections like II., III., IV., etc.)
        # This helps distinguish between "body incomplete" vs "body done, need conclusion"
        has_body_content = bool(re.search(r'^[IVX]+\.\s+\w', paper_content, re.MULTILINE))
        
        # Determine phase based on what sections actually exist
        # Order: body -> conclusion -> introduction -> abstract
        
        # Use actual content as primary indicator, placeholder as secondary
        conclusion_missing = not has_conclusion or has_conclusion_placeholder
        intro_missing = not has_intro or has_intro_placeholder
        abstract_missing = not has_abstract or has_abstract_placeholder
        
        if conclusion_missing:
            # Conclusion not written yet
            if has_body_content:
                # Body content exists, ready to write conclusion
                return "conclusion"
            else:
                # No body content yet, need to write body first
                return "body"
        elif intro_missing:
            # Conclusion written, ready to write introduction
            return "introduction"
        elif abstract_missing:
            # Introduction written, ready to write abstract
            return "abstract"
        else:
            # All sections present, paper is complete
            return "abstract"
    
    def _has_section(self, content: str, section_name: str) -> bool:
        """
        Check if a section exists in the paper content.
        
        Handles different section naming conventions:
        - Abstract: Never numbered (just "Abstract")
        - Introduction: Always numbered as "I. Introduction"
        - Conclusion: Can be numbered (e.g., "VI. Conclusion") or plain
        - Body sections: Numbered with Roman numerals (II, III, IV, etc.)
        """
        # Base patterns that work for all sections
        base_patterns = [
            rf"##\s*{section_name}",       # Markdown heading
            rf"#\s*{section_name}",        # Markdown heading
            rf"\*\*{section_name}\*\*",    # Bold text
            rf"^{section_name}\s*$",       # Plain section name
        ]
        
        # Add section-specific patterns
        if section_name == "Introduction":
            # Introduction is always numbered as "I."
            base_patterns.append(rf"^I\.\s*{section_name}")
        elif section_name == "Conclusion":
            # Conclusion can have Roman numeral (variable position in paper)
            base_patterns.append(rf"^[IVXLC]+\.\s*{section_name}")
        # Abstract never has a number, so no additional pattern needed
        
        for pattern in base_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    async def _load_saved_paper_to_compiler(self, paper_id: str) -> None:
        """
        Load a saved paper from the library back into compiler memory.
        
        This is used when resuming an incomplete paper after a system restart.
        The paper content needs to be loaded from the paper library files
        back into the compiler's paper_memory and outline_memory.
        """
        try:
            # Load paper content from library
            paper_path = paper_library._get_paper_path(paper_id)
            if paper_path.exists():
                async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                    paper_content = await f.read()
                
                # Strip the attribution header if present (starts with ===)
                # The attribution is added at save time, we don't want it in compiler memory
                if paper_content.startswith("=" * 80):
                    # Find the end of the attribution block
                    lines = paper_content.split("\n")
                    content_start = 0
                    in_header = True
                    for i, line in enumerate(lines):
                        if in_header and line.startswith("=" * 80) and i > 0:
                            # Found end of header block
                            content_start = i + 1
                            in_header = False
                            break
                    
                    if content_start > 0:
                        paper_content = "\n".join(lines[content_start:]).strip()
                        logger.info(f"Stripped attribution header from saved paper ({content_start} lines)")
                
                # Also strip model credits footer if present
                if "=" * 80 + "\nMODEL CREDITS" in paper_content:
                    idx = paper_content.find("=" * 80 + "\nMODEL CREDITS")
                    paper_content = paper_content[:idx].strip()
                    logger.info("Stripped model credits footer from saved paper")
                
                # Save to compiler paper memory
                await compiler_paper_memory.update_paper(paper_content)
                logger.info(f"Loaded saved paper to compiler memory ({len(paper_content)} chars)")
            else:
                logger.warning(f"Saved paper not found: {paper_path}")
            
            # Load outline from library
            outline_path = paper_library._get_outline_path(paper_id)
            if outline_path.exists():
                async with aiofiles.open(outline_path, 'r', encoding='utf-8') as f:
                    outline_content = await f.read()
                
                await outline_memory.update_outline(outline_content)
                logger.info(f"Loaded saved outline to compiler memory ({len(outline_content)} chars)")
            else:
                logger.warning(f"Saved outline not found: {outline_path}")
                
        except Exception as e:
            logger.error(f"Failed to load saved paper {paper_id} to compiler: {e}")
    
    async def _save_workflow_state(self, tier: str = None, phase: str = None) -> None:
        """Save current workflow state for crash recovery."""
        # Serialize submitter configs for storage
        submitter_configs_data = [
            {
                "submitter_id": config.submitter_id,
                "model_id": config.model_id,
                "context_window": config.context_window,
                "max_output_tokens": config.max_output_tokens
            }
            for config in self._submitter_configs
        ]
        
        # Get Tier 3 state for crash recovery
        tier3_state = final_answer_memory.get_state()
        tier3_format = final_answer_memory.get_answer_format()
        
        state = {
            "is_running": self._running,
            "current_tier": tier or self._state.current_tier,
            "current_topic_id": self._current_topic_id,
            "current_paper_id": self._current_paper_id,
            "paper_phase": phase,
            "reference_paper_ids": self._current_reference_papers,  # Persist reference papers across restarts
            "acceptance_count": self._acceptance_count,
            "rejection_count": self._rejection_count,
            "consecutive_rejections": self._consecutive_rejections,
            "exhaustion_signals": self._exhaustion_signals,
            "papers_completed_count": self._papers_completed_count,
            "last_redundancy_check_at": self._last_redundancy_check_at,
            "last_tier3_check_at": self._last_tier3_check_at,
            # Tier 3 Final Answer crash recovery fields
            "tier3_active": self._tier3_active,
            "tier3_format": tier3_format,
            "tier3_phase": tier3_state.status if tier3_state and tier3_state.is_active else None,
            "model_config": {
                "submitter_configs": submitter_configs_data,
                "validator_model": self._validator_model,
                "validator_context_window": self._validator_context,
                "validator_max_tokens": self._validator_max_tokens,
                "high_context_model": self._high_context_model,
                "high_param_model": self._high_param_model,
                "high_context_context_window": self._high_context_context,
                "high_param_context_window": self._high_param_context,
                "high_context_max_tokens": self._high_context_max_tokens,
                "high_param_max_tokens": self._high_param_max_tokens
            }
        }
        await research_metadata.save_workflow_state(state)
    
    async def start(self) -> None:
        """Start the autonomous research loop."""
        if self._running:
            logger.warning("AutonomousCoordinator already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._state.is_running = True
        
        # Set up autonomous API logging callback
        async def log_callback(task_id, role_id, model, provider, prompt, response, 
                              tokens_used, duration_ms, success, error, phase):
            """Callback for logging autonomous API calls."""
            try:
                await autonomous_api_logger.log_api_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=model,
                    provider=provider,
                    prompt=prompt,
                    response_content=response,
                    tokens_used=tokens_used,
                    duration_ms=duration_ms,
                    success=success,
                    error=error,
                    phase=phase
                )
            except Exception as e:
                logger.error(f"Failed to log API call in autonomous logger: {e}")
        
        api_client_manager.set_autonomous_logger_callback(log_callback)
        logger.info("Autonomous API logging enabled")
        
        # Refresh workflow predictions at start
        await self.refresh_workflow_predictions()
        
        await self._broadcast("auto_research_started")
        logger.info("AutonomousCoordinator started")
        
        # Check for interrupted workflow to resume
        resume_state = await self._get_resume_point()
        
        try:
            # Main research loop
            while self._running and not self._stop_event.is_set():
                # Check if resuming from interrupted state (CHECK THIS FIRST)
                if resume_state:
                    resume_tier = resume_state.get("current_tier")
                    resume_topic = resume_state.get("current_topic_id")
                    resume_paper = resume_state.get("current_paper_id")
                    
                    logger.info(f"Resuming from interrupted workflow: tier={resume_tier}, "
                               f"topic={resume_topic}, paper={resume_paper}")
                    
                    await self._broadcast("auto_research_resumed", {
                        "tier": resume_tier,
                        "topic_id": resume_topic,
                        "paper_id": resume_paper
                    })
                    
                    # DEFENSIVE: Check for Tier 3 state mismatch
                    # If tier says "tier2_paper_writing" but tier3_phase is set, this was actually Tier 3
                    if resume_tier == "tier2_paper_writing" and resume_state.get("tier3_phase"):
                        logger.warning(
                            f"⚠️ DETECTED TIER 3 STATE MISMATCH: "
                            f"tier={resume_tier} but tier3_phase={resume_state.get('tier3_phase')}. "
                            f"Correcting to tier3_final_answer for proper resume."
                        )
                        resume_tier = "tier3_final_answer"
                    
                    if resume_tier == "tier2_paper_writing" and resume_topic:
                        # Resume paper writing - skip to compilation
                        # CRITICAL: Restore paper_id so compilation workflow knows to resume
                        self._current_topic_id = resume_topic
                        self._current_paper_id = resume_paper  # FIX: Restore paper_id
                        await self._paper_compilation_workflow()
                        resume_state = None  # Clear resume state after handling
                        continue
                    elif resume_tier == "tier1_aggregation" and resume_topic:
                        # Resume brainstorm aggregation
                        self._current_topic_id = resume_topic
                        
                        # Verify topic still exists; if missing, clear resume state
                        metadata = await brainstorm_memory.get_metadata(resume_topic)
                        if metadata is None:
                            logger.warning(f"Resume state references missing brainstorm {resume_topic}; clearing resume state")
                            await research_metadata.clear_workflow_state()
                            resume_state = None
                            continue
                        
                        write_paper = await self._brainstorm_aggregation_loop()
                        resume_state = None  # Clear resume state after handling
                        
                        if self._stop_event.is_set():
                            break
                        
                        if write_paper:
                            await self._paper_compilation_workflow()
                            
                            if self._stop_event.is_set():
                                break
                            
                            await self._check_paper_redundancy()
                        
                        continue
                    elif resume_tier == "tier3_final_answer":
                        # Resume Tier 3 final answer generation
                        tier3_state = final_answer_memory.get_state()
                        
                        logger.info(f"Resuming Tier 3 final answer: format={tier3_state.answer_format}, "
                                   f"status={tier3_state.status}, is_active={tier3_state.is_active}")
                        
                        if tier3_state and tier3_state.is_active:
                            # Resume Tier 3 from saved state
                            completed = await self._resume_tier3_workflow(tier3_state)
                            resume_state = None  # Clear resume state after handling
                            
                            if completed:
                                # Final answer complete - system should stop
                                logger.info("Tier 3 resumed and completed - autonomous research complete")
                                await self._broadcast("final_answer_complete", {
                                    "resumed": True,
                                    "format": tier3_state.answer_format
                                })
                                break
                            else:
                                # no_answer_known - continue research
                                logger.info("Tier 3 resumed but needs more research - continuing")
                                continue
                        else:
                            # Tier 3 state not active or invalid - start fresh
                            logger.warning("Tier 3 state not active, starting fresh topic selection")
                            resume_state = None
                            continue
                    else:
                        # Unknown resume state, start fresh
                        resume_state = None
                
                # CRITICAL: Check if there's an unsaved paper from previous topic
                # Skip this check if we were resuming tier2 (paper is intentionally "unsaved" during resume)
                if self._current_paper_id and not await self._is_paper_saved(self._current_paper_id):
                    logger.warning(f"Unsaved paper detected: {self._current_paper_id}. Saving before starting new topic.")
                    
                    # Get current paper content
                    current_paper = await compiler_paper_memory.get_paper()
                    current_outline = await outline_memory.get_outline()
                    
                    if current_paper:
                        # Save paper without marking as complete (it's still in progress)
                        await self._handle_paper_completion(
                            paper_id=self._current_paper_id,
                            title=self._current_paper_title or "Untitled Paper",
                            content=current_paper,
                            outline=current_outline or "[Outline not available]",
                            reference_paper_ids=getattr(self, '_current_reference_papers', []),
                            mark_complete=False  # Keep paper state for resume
                        )
                        logger.info(f"Saved incomplete paper {self._current_paper_id} before topic switch")
                
                
                # Check for forced immediate Tier 3 (skip_incomplete mode)
                if self._force_tier3_immediate:
                    logger.info("Forced Tier 3 (skip_incomplete): Triggering immediately")
                    self._force_tier3_immediate = False  # Clear flag
                    
                    # Verify we have papers to work with
                    all_papers = await paper_library.get_all_papers()
                    if len(all_papers) > 0:
                        completed = await self._tier3_final_answer_workflow()
                        
                        if completed:
                            logger.info("FINAL ANSWER COMPLETE - Autonomous research finished")
                            await self._broadcast("final_answer_complete", {
                                "format": final_answer_memory.get_answer_format(),
                                "status": final_answer_memory.get_state().status
                            })
                            break
                        else:
                            logger.info("Tier 3 returned no_answer_known - continuing research")
                    else:
                        logger.warning("Cannot run forced Tier 3: no completed papers")
                
                # Phase 1: Topic selection
                topic_result = await self._topic_selection_loop()
                
                if self._stop_event.is_set():
                    break
                
                if not topic_result:
                    logger.error("Topic selection failed, retrying in 30 seconds")
                    await asyncio.sleep(30)
                    continue
                
                # Phase 1.5: Pre-brainstorm reference paper selection
                # This enables compounding knowledge across research cycles
                self._current_reference_papers = await self._pre_brainstorm_reference_selection()
                logger.info(f"Selected {len(self._current_reference_papers)} reference papers for brainstorm")
                
                if self._stop_event.is_set():
                    break
                
                # Save workflow state after topic and reference selection
                await self._save_workflow_state(tier="tier1_aggregation")
                
                # Phase 2: Brainstorm aggregation (with reference papers)
                write_paper = await self._brainstorm_aggregation_loop()
                
                if self._stop_event.is_set():
                    break
                
                if not write_paper:
                    # Continue with brainstorm, loop back
                    continue
                
                # Phase 3: Paper compilation
                await self._save_workflow_state(tier="tier2_paper_writing")
                paper_success = await self._paper_compilation_workflow()
                
                if self._stop_event.is_set():
                    break
                
                # Only check redundancy and log completion if paper was successful
                if paper_success:
                    # Check for paper redundancy (every 3 papers)
                    await self._check_paper_redundancy()
                    
                    # Check for Tier 3 final answer trigger (every 5 papers)
                    if await self._should_trigger_tier3():
                        logger.info("Tier 3 trigger: Attempting final answer generation")
                        completed = await self._tier3_final_answer_workflow()
                        
                        if completed:
                            # System stops after final answer is complete
                            logger.info("FINAL ANSWER COMPLETE - Autonomous research finished")
                            await self._broadcast("final_answer_complete", {
                                "format": final_answer_memory.get_answer_format(),
                                "status": final_answer_memory.get_state().status
                            })
                            break
                        else:
                            # Tier 3 decided we need more research - continue
                            logger.info("Tier 3: More research needed, returning to topic selection")
                    
                    logger.info("Paper complete, returning to topic selection")
                else:
                    logger.warning("Paper compilation failed, returning to topic selection")
                
        except Exception as e:
            logger.error(f"AutonomousCoordinator error: {e}")
            # Save workflow state before raising (for crash recovery)
            await self._save_workflow_state()
            raise
        finally:
            self._running = False
            self._state.is_running = False
            
            stats = await research_metadata.get_stats()
            await self._broadcast("auto_research_stopped", {
                "final_stats": stats
            })
            logger.info("AutonomousCoordinator stopped")
    
    async def _get_resume_point(self) -> Optional[Dict[str, Any]]:
        """Get resume point if there's an interrupted workflow."""
        if research_metadata.has_interrupted_workflow():
            return await research_metadata.get_workflow_state()
        return None
    
    async def stop(self) -> None:
        """Stop the autonomous research gracefully.
        
        IMPORTANT: This method preserves workflow state for resume capability.
        The user can resume their session by pressing Start again.
        Only the clear_all_data() method should delete workflow state.
        """
        logger.info("Stopping AutonomousCoordinator...")
        self._stop_event.set()
        self._running = False
        
        # Stop any running aggregator or compiler to prevent orphan tasks
        if self._brainstorm_aggregator:
            try:
                await self._brainstorm_aggregator.stop()
                logger.info("Stopped brainstorm aggregator")
            except Exception as e:
                logger.warning(f"Error stopping aggregator: {e}")
        
        if self._paper_compiler:
            try:
                await self._paper_compiler.stop()
                logger.info("Stopped paper compiler")
            except Exception as e:
                logger.warning(f"Error stopping compiler: {e}")
        
        # Clear autonomous API logging callback
        api_client_manager.set_autonomous_logger_callback(None)
        logger.info("Autonomous API logging disabled")
        
        # SAVE workflow state for resume (NOT clear it)
        # The user should be able to resume by pressing Start again
        # Only clear_all_data() should delete the workflow state
        try:
            # Determine current tier based on state
            current_tier = None
            if self._state.current_tier:
                current_tier = self._state.current_tier
            
            # Save the state with is_running=False but preserve all other data
            await self._save_workflow_state(tier=current_tier)
            logger.info(f"Workflow state saved for resume (tier={current_tier}, topic={self._current_topic_id})")
        except Exception as e:
            logger.warning(f"Could not save workflow state on stop: {e}")
        
        logger.info("Autonomous research stopped - press Start to resume from last state")
    
    async def reset_current_paper(self) -> Dict[str, Any]:
        """
        Reset current paper being written and restart from appropriate phase.
        
        Behavior varies by context:
        - Tier 3 Short-Form: Reset to title selection phase (clear paper, outline, title)
        - Tier 3 Long-Form: Reset only current chapter being written (gap/intro/conclusion)
        - Tier 2 Paper (during autonomous): Delegate to compiler coordinator
        
        Returns:
            Dict with reset details (phase, what was cleared, restart point)
        """
        logger.info("Resetting current paper...")
        
        # Check if Tier 3 is active
        if self._tier3_active:
            answer_format = self._state.tier3_answer_format
            
            if answer_format == "short_form":
                # Tier 3 Short-Form: Reset to title selection
                logger.info("Tier 3 short-form reset: clearing paper, outline, and title")
                
                # Stop compiler if running
                if self._paper_compiler and self._paper_compiler.is_running:
                    await self._paper_compiler.stop()
                    logger.info("Stopped compiler for reset")
                
                # Clear paper and outline via compiler coordinator
                if self._paper_compiler:
                    await self._paper_compiler.clear_paper()
                
                # Reset title selection in final_answer_memory
                await final_answer_memory.reset_title_selection()
                
                # Broadcast reset event
                await self._broadcast("tier3_paper_reset", {
                    "format": "short_form",
                    "message": "Short-form paper reset to title selection phase"
                })
                
                return {
                    "reset_type": "tier3_short_form",
                    "cleared": ["paper", "outline", "title"],
                    "restart_phase": "title_selection",
                    "message": "Paper reset to title selection. Will select new title and rebuild from scratch."
                }
            
            elif answer_format == "long_form":
                # Tier 3 Long-Form: Reset only current chapter
                logger.info("Tier 3 long-form reset: clearing current chapter only")
                
                # Stop compiler if running
                if self._paper_compiler and self._paper_compiler.is_running:
                    await self._paper_compiler.stop()
                    logger.info("Stopped compiler for chapter reset")
                
                # Get current chapter index
                volume_org = self._state.volume_organization
                if not volume_org:
                    raise ValueError("No volume organization found for long-form reset")
                
                current_chapter_idx = None
                for idx, chapter in enumerate(volume_org.chapters):
                    if chapter.status == "writing":
                        current_chapter_idx = idx
                        break
                
                if current_chapter_idx is None:
                    raise ValueError("No chapter currently being written")
                
                # Clear current chapter
                await final_answer_memory.reset_current_chapter(current_chapter_idx)
                
                # Clear compiler paper and outline
                if self._paper_compiler:
                    await self._paper_compiler.clear_paper()
                
                chapter_type = volume_org.chapters[current_chapter_idx].chapter_type
                chapter_title = volume_org.chapters[current_chapter_idx].title
                
                # Broadcast reset event
                await self._broadcast("tier3_chapter_reset", {
                    "format": "long_form",
                    "chapter_index": current_chapter_idx,
                    "chapter_type": chapter_type,
                    "chapter_title": chapter_title,
                    "message": f"Chapter {current_chapter_idx} ({chapter_title}) reset to pending"
                })
                
                return {
                    "reset_type": "tier3_long_form_chapter",
                    "cleared": [f"chapter_{current_chapter_idx}_paper", f"chapter_{current_chapter_idx}_outline"],
                    "chapter_index": current_chapter_idx,
                    "chapter_title": chapter_title,
                    "restart_phase": "chapter_writing",
                    "message": f"Chapter {current_chapter_idx} ({chapter_title}) reset. Will rebuild this chapter from scratch."
                }
            
            else:
                raise ValueError(f"Unknown answer format: {answer_format}")
        
        else:
            # Tier 2 paper writing during autonomous (not Tier 3)
            logger.info("Tier 2 reset during autonomous: delegating to compiler coordinator")
            
            if not self._paper_compiler:
                raise ValueError("No compiler active to reset")
            
            # Delegate to compiler clear_paper
            await self._paper_compiler.clear_paper()
            
            # Broadcast reset event
            await self._broadcast("tier2_paper_reset", {
                "message": "Tier 2 paper reset to outline creation"
            })
            
            return {
                "reset_type": "tier2_autonomous",
                "cleared": ["paper", "outline", "rejection_logs"],
                "restart_phase": "outline_creation",
                "message": "Paper reset to outline creation. Will rebuild outline and paper from scratch."
            }
    
    async def _resume_research_loop_after_tier3(self) -> None:
        """
        Resume the main research loop after Tier 3 returns no_answer_known.
        
        This method is called as a background task when Tier 3 determines that
        more research is needed. It re-enters the main research loop to continue
        generating brainstorms and papers until the next Tier 3 trigger.
        """
        logger.info("Resuming research loop after Tier 3 no_answer_known")
        
        # Broadcast resume event
        await self._broadcast("auto_research_resumed", {
            "tier": "tier1_aggregation",
            "topic_id": None,  # Will select new topic
            "paper_id": None,
            "reason": "tier3_no_answer_known"
        })
        
        try:
            # Main research loop - same as in start() method
            while self._running and not self._stop_event.is_set():
                # Check for forced immediate Tier 3 (skip_incomplete mode)
                if self._force_tier3_immediate:
                    logger.info("Forced Tier 3 (skip_incomplete): Triggering immediately")
                    self._force_tier3_immediate = False  # Clear flag
                    
                    # Verify we have papers to work with
                    all_papers = await paper_library.get_all_papers()
                    if len(all_papers) > 0:
                        completed = await self._tier3_final_answer_workflow()
                        
                        if completed:
                            logger.info("FINAL ANSWER COMPLETE - Autonomous research finished")
                            await self._broadcast("final_answer_complete", {
                                "format": final_answer_memory.get_answer_format(),
                                "status": final_answer_memory.get_state().status
                            })
                            break
                        else:
                            logger.info("Tier 3 returned no_answer_known - continuing research")
                    else:
                        logger.warning("Cannot run forced Tier 3: no completed papers")
                
                # Phase 1: Topic selection
                topic_result = await self._topic_selection_loop()
                
                if self._stop_event.is_set():
                    break
                
                if not topic_result:
                    logger.error("Topic selection failed, retrying in 30 seconds")
                    await asyncio.sleep(30)
                    continue
                
                # Phase 1.5: Pre-brainstorm reference paper selection
                self._current_reference_papers = await self._pre_brainstorm_reference_selection()
                logger.info(f"Selected {len(self._current_reference_papers)} reference papers for brainstorm")
                
                if self._stop_event.is_set():
                    break
                
                # Save workflow state after topic and reference selection
                await self._save_workflow_state(tier="tier1_aggregation")
                
                # Phase 2: Brainstorm aggregation (with reference papers)
                write_paper = await self._brainstorm_aggregation_loop()
                
                if self._stop_event.is_set():
                    break
                
                if not write_paper:
                    # Continue with brainstorm, loop back
                    continue
                
                # Phase 3: Paper compilation
                await self._save_workflow_state(tier="tier2_paper_writing")
                paper_success = await self._paper_compilation_workflow()
                
                if self._stop_event.is_set():
                    break
                
                # Only check redundancy and log completion if paper was successful
                if paper_success:
                    # Check for paper redundancy (every 3 papers)
                    await self._check_paper_redundancy()
                    
                    # Check for Tier 3 final answer trigger (every 5 papers)
                    if await self._should_trigger_tier3():
                        logger.info("Tier 3 trigger: Attempting final answer generation")
                        completed = await self._tier3_final_answer_workflow()
                        
                        if completed:
                            # System stops after final answer is complete
                            logger.info("FINAL ANSWER COMPLETE - Autonomous research finished")
                            await self._broadcast("final_answer_complete", {
                                "format": final_answer_memory.get_answer_format(),
                                "status": final_answer_memory.get_state().status
                            })
                            break
                        else:
                            # Tier 3 decided we need more research - continue
                            logger.info("Tier 3: More research needed, returning to topic selection")
                    
                    logger.info("Paper complete, returning to topic selection")
                else:
                    logger.warning("Paper compilation failed, returning to topic selection")
                
        except Exception as e:
            logger.error(f"Error in resumed research loop: {e}")
            # Save workflow state before cleanup (for crash recovery)
            await self._save_workflow_state()
        finally:
            self._running = False
            self._state.is_running = False
            
            # Clear shared training memory in-memory cache to prevent data pollution
            # The insights list will be reloaded from the file when needed
            shared_training_memory.insights.clear()
            shared_training_memory.submission_count = 0
            shared_training_memory.last_ragged_submission_count = 0
            logger.info("Cleared shared_training_memory in-memory data (will reload from file when needed)")
            
            stats = await research_metadata.get_stats()
            await self._broadcast("auto_research_stopped", {
                "final_stats": stats
            })
            logger.info("Resumed research loop completed")
    
    def get_state(self) -> AutonomousResearchState:
        """Get current state."""
        return self._state
    
    def get_validator_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the current validator configuration.
        Returns None if not initialized.
        
        Returns:
            Dict with validator_model, validator_context_window, validator_max_tokens,
            validator_provider, and validator_openrouter_provider, or None if not initialized.
        """
        if not self._validator_model:
            return None
        
        return {
            "validator_model": self._validator_model,
            "validator_context_window": self._validator_context,
            "validator_max_tokens": self._validator_max_tokens,
            "validator_provider": self._validator_provider,
            "validator_openrouter_provider": self._validator_openrouter_provider,
        }
    
    async def skip_critique_phase(self) -> bool:
        """
        Skip critique phase for the currently compiling paper.
        Proxies to the paper compiler's skip_critique_phase method.
        
        Returns:
            True if successfully skipped, False if not in paper writing or no compiler
        """
        if self._state.current_tier != "tier2_paper_writing":
            logger.warning("Cannot skip critique: not in paper writing tier")
            return False
        
        if not self._paper_compiler:
            logger.warning("Cannot skip critique: no active paper compiler")
            return False
        
        return await self._paper_compiler.skip_critique_phase()
    
    # ========================================================================
    # PHASE 1: TOPIC SELECTION
    # ========================================================================
    
    async def _topic_selection_loop(self) -> Optional[str]:
        """
        Topic selection with validation.
        
        Returns:
            topic_id if successful, None if failed
        """
        # Set tier state immediately when entering topic selection
        # This ensures force_tier3 can reliably detect we're in the Tier 1 phase
        # (Previously, state remained "idle" until aggregation loop started, causing race conditions)
        self._state.current_tier = "tier1_aggregation"
        
        # Set phase for API logging
        api_client_manager.set_autonomous_phase("topic_selection")
        
        max_attempts = system_config.autonomous_topic_selection_retry_limit
        
        for attempt in range(max_attempts):
            if self._stop_event.is_set():
                return None
            
            logger.info(f"Topic selection attempt {attempt + 1}/{max_attempts}")
            
            # Get context
            brainstorms_summary = await autonomous_rag_manager.get_all_brainstorms_summary()
            papers_summary = await autonomous_rag_manager.get_all_papers_summary()
            
            # Generate topic selection
            submission = await self._topic_selector.select_topic(
                user_research_prompt=self._user_research_prompt,
                brainstorms_summary=brainstorms_summary,
                papers_summary=papers_summary
            )
            
            if submission is None:
                logger.warning("Failed to generate topic selection")
                await asyncio.sleep(5)
                continue
            
            # Validate
            validation = await self._topic_validator.validate(
                submission=submission,
                user_research_prompt=self._user_research_prompt,
                brainstorms_summary=brainstorms_summary,
                papers_summary=papers_summary
            )
            
            if validation.decision == "accept":
                # Check if we should stop before creating new topic
                if self._stop_event.is_set():
                    logger.info("Topic selection cancelled - stop event set after validation")
                    return None
                
                # Execute topic selection
                topic_id = await self._execute_topic_selection(submission)
                
                if topic_id:
                    await self._broadcast("topic_selected", {
                        "action": submission.action,
                        "topic_id": topic_id,
                        "topic_prompt": submission.topic_prompt or submission.topic_id
                    })
                    return topic_id
            else:
                # Handle rejection
                await self._topic_selector.handle_rejection(submission, validation.reasoning)
                await research_metadata.increment_stat("topic_selection_rejections")
                
                await self._broadcast("topic_selection_rejected", {
                    "reasoning": validation.reasoning
                })
                
                logger.info(f"Topic selection rejected: {validation.reasoning[:100]}...")
        
        logger.error(f"Topic selection failed after {max_attempts} attempts")
        return None
    
    async def _execute_topic_selection(
        self, 
        submission: TopicSelectionSubmission
    ) -> Optional[str]:
        """Execute the topic selection action."""
        # Early return if stopped
        if self._stop_event.is_set():
            logger.info("Topic execution cancelled - stop event set")
            return None
        
        try:
            if submission.action == "new_topic":
                # Create new brainstorm
                topic_id = await research_metadata.generate_topic_id()
                metadata = await brainstorm_memory.create_brainstorm(
                    topic_id=topic_id,
                    topic_prompt=submission.topic_prompt
                )
                await research_metadata.register_brainstorm(metadata)
                
                self._current_topic_id = topic_id
                self._acceptance_count = 0
                self._consecutive_rejections = 0
                
                logger.info(f"Created new brainstorm: {topic_id}")
                return topic_id
                
            elif submission.action == "continue_existing":
                # Continue existing brainstorm
                topic_id = submission.topic_id
                metadata = await brainstorm_memory.get_metadata(topic_id)
                
                if metadata is None:
                    logger.error(f"Brainstorm not found: {topic_id}")
                    return None
                
                self._current_topic_id = topic_id
                self._acceptance_count = metadata.submission_count
                self._consecutive_rejections = 0
                
                logger.info(f"Continuing brainstorm: {topic_id}")
                return topic_id
                
            elif submission.action == "combine_topics":
                # Combine multiple brainstorms
                topic_id = await research_metadata.generate_topic_id()
                metadata = await brainstorm_memory.combine_topics(
                    new_topic_id=topic_id,
                    new_topic_prompt=submission.topic_prompt,
                    source_topic_ids=submission.topic_ids
                )
                
                if metadata is None:
                    logger.error("Failed to combine topics")
                    return None
                
                await research_metadata.register_brainstorm(metadata)
                
                self._current_topic_id = topic_id
                self._acceptance_count = metadata.submission_count
                self._consecutive_rejections = 0
                
                logger.info(f"Combined topics into: {topic_id}")
                return topic_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing topic selection: {e}")
            return None
    
    async def _pre_brainstorm_reference_selection(self) -> List[str]:
        """
        Select reference papers BEFORE brainstorming begins.
        
        This is the crucial mechanism that enables compounding knowledge across research cycles.
        By selecting reference papers before brainstorming, submitters can:
        - Build upon proven mathematical frameworks from prior papers
        - Avoid re-exploring territory already covered in depth
        - Identify novel connections between new topics and established results
        - Accelerate convergence on valuable insights by standing on prior work
        
        Returns:
            List of selected paper_ids (max 6)
        """
        # Get available papers
        papers_summary = await autonomous_rag_manager.get_all_papers_summary()
        
        if not papers_summary:
            logger.info("No papers available for pre-brainstorm reference selection")
            return []
        
        # Get topic metadata
        metadata = await brainstorm_memory.get_metadata(self._current_topic_id)
        topic_prompt = metadata.topic_prompt if metadata else ""
        
        # For pre-brainstorm selection, we don't have a brainstorm summary yet
        # Use the topic prompt as the main context
        brainstorm_summary = f"[Brainstorm not yet started]\nTopic: {topic_prompt}"
        
        await self._broadcast("reference_selection_started", {
            "topic_id": self._current_topic_id,
            "mode": "pre_brainstorm",
            "already_selected": 0,  # No papers selected yet in pre-brainstorm mode
            "available_papers": len(papers_summary)
        })
        
        # Run reference selection in "initial" mode (before brainstorm)
        selected_ids = await self._reference_selector.select_references(
            user_research_prompt=self._user_research_prompt,
            topic_prompt=topic_prompt,
            brainstorm_summary=brainstorm_summary,
            available_papers=papers_summary,
            mode="initial",  # Pre-brainstorm mode
            already_selected=[]  # No papers selected yet
        )
        
        await self._broadcast("reference_selection_complete", {
            "topic_id": self._current_topic_id,
            "mode": "pre_brainstorm",
            "selected_count": len(selected_ids),
            "newly_added": len(selected_ids),  # For pre_brainstorm, all are new
            "selected_papers": selected_ids
        })
        
        logger.info(f"Pre-brainstorm reference selection: selected {len(selected_ids)} papers")
        return selected_ids
    
    def _get_reference_paper_paths(self) -> List[str]:
        """
        Get file paths for currently selected reference papers.
        Uses session-based paths if session manager is active.
        
        Returns:
            List of file paths to reference paper files
        """
        paths = []
        for paper_id in self._current_reference_papers:
            # Use paper_library to get session-aware path
            # paper_library handles both legacy flat structure and session-based paths
            paper_path = paper_library._get_paper_path(paper_id)
            if os.path.exists(paper_path):
                paths.append(str(paper_path))
            else:
                logger.warning(f"Reference paper not found: {paper_path}")
        return paths
    
    # ========================================================================
    # PHASE 2: BRAINSTORM AGGREGATION
    # ========================================================================
    
    async def _brainstorm_aggregation_loop(self) -> bool:
        """
        Brainstorm aggregation loop.
        Uses actual Part 1 aggregator infrastructure.
        
        Returns:
            True if should write paper, False if should continue
        """
        self._state.current_tier = "tier1_aggregation"
        
        # Set phase for API logging
        api_client_manager.set_autonomous_phase("brainstorm")
        
        # Get brainstorm metadata
        metadata = await brainstorm_memory.get_metadata(self._current_topic_id)
        if metadata is None:
            logger.error(f"Cannot start aggregation: brainstorm {self._current_topic_id} not found")
            return False
        
        # Initialize per-paper model tracker for this brainstorm/paper cycle
        self._current_paper_tracker = PaperModelTracker(
            user_prompt=self._user_research_prompt,
            paper_title=""  # Will be set later when paper title is selected
        )
        
        # Set up model tracking callback - tracks to BOTH per-paper tracker AND global Tier 3 tracker
        async def paper_model_tracking_callback(model_id: str) -> None:
            # Track to current paper tracker (always)
            if self._current_paper_tracker:
                self._current_paper_tracker.track_call(model_id)
            # Also track to global Tier 3 tracker if active
            if self._tier3_active:
                await final_answer_memory.track_model_call(model_id)
        
        api_client_manager.set_model_tracking_callback(paper_model_tracking_callback)
        logger.info(f"Per-paper model tracking enabled for brainstorm {self._current_topic_id}")
        
        # Initialize aggregator for this brainstorm
        self._brainstorm_aggregator = AggregatorCoordinator()
        
        # Override shared training memory path to brainstorm-specific
        # Use brainstorm_memory to get correct path (respects session manager)
        brainstorm_db_path = brainstorm_memory._get_database_path(self._current_topic_id)
        brainstorm_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Temporarily override shared training path
        original_shared_path = system_config.shared_training_file
        system_config.shared_training_file = str(brainstorm_db_path)
        
        # CRITICAL: Also update the shared_training_memory file path
        # since it's a global singleton that was initialized with the original path
        original_memory_path = shared_training_memory.file_path
        shared_training_memory.file_path = brainstorm_db_path
        
        # CRITICAL FIX: Reload insights from the brainstorm-specific file
        # This prevents data loss from overwriting the brainstorm file with insights
        # from the previous file (rag_shared_training.txt)
        await shared_training_memory.reload_insights_from_current_path()
        logger.info(f"Reloaded {len(shared_training_memory.insights)} existing submissions from brainstorm database")
        
        try:
            # Get reference paper paths for brainstorm context
            # This enables compounding knowledge - brainstorm submitters can build on prior papers
            reference_paper_paths = self._get_reference_paper_paths()
            if reference_paper_paths:
                logger.info(f"Loading {len(reference_paper_paths)} reference papers for brainstorm aggregation")
            
            # Initialize aggregator with topic prompt
            # CRITICAL: skip_stats_load=True to prevent loading manual aggregator stats
            # CRITICAL: Pass per-submitter configs for multi-submitter support
            # CRITICAL: Pass reference papers as user_files to enable compounding knowledge
            await self._brainstorm_aggregator.initialize(
                user_prompt=metadata.topic_prompt,
                submitter_configs=self._submitter_configs,  # Per-submitter configs (1-10 submitters)
                validator_model=self._validator_model,
                user_files=reference_paper_paths,  # Reference papers for compounding knowledge
                skip_stats_load=True,  # Start fresh for each brainstorm (don't load manual mode stats)
                validator_context_window=self._validator_context,
                validator_max_tokens=self._validator_max_tokens,
                # Pass OpenRouter provider configs for validator
                validator_provider=self._validator_provider,
                validator_openrouter_provider=self._validator_openrouter_provider,
                validator_lm_studio_fallback=self._validator_lm_studio_fallback
            )
            
            # CRITICAL FIX: Re-ingest existing submissions into RAG after resume
            # When resuming, shared_training_memory has loaded submissions from file
            # but they're not in RAG because we cleared RAG during _reset_rag_state()
            # We need to re-ingest them so submitters can retrieve context
            if len(shared_training_memory.insights) > 0:
                logger.info(f"Re-ingesting {len(shared_training_memory.insights)} existing submissions into RAG...")
                
                # Write current insights to temp file for RAG ingestion
                temp_db_path = brainstorm_db_path.with_suffix('.tmp')
                try:
                    async with aiofiles.open(temp_db_path, 'w', encoding='utf-8') as f:
                        # Extract content from dict (insights are now dicts with metadata)
                        insight_contents = [insight['content'] if isinstance(insight, dict) else insight for insight in shared_training_memory.insights]
                        await f.write('\n\n'.join(insight_contents))
                    
                    # Ingest into RAG with all 4 chunk configs (for cyclic variation)
                    from backend.shared.config import rag_config
                    await rag_manager.add_document(
                        str(temp_db_path),
                        chunk_sizes=rag_config.submitter_chunk_intervals,  # [256, 512, 768, 1024]
                        is_user_file=False  # This is dynamic content (subject to eviction)
                    )
                    logger.info(f"Successfully re-ingested {len(shared_training_memory.insights)} submissions into RAG")
                finally:
                    # Clean up temp file
                    if temp_db_path.exists():
                        temp_db_path.unlink()
            
            # Set WebSocket broadcaster for aggregator events
            if self._broadcast_callback:
                self._brainstorm_aggregator.websocket_broadcaster = self._broadcast_callback
            
            # Start aggregator
            await self._brainstorm_aggregator.start()
            logger.info(f"Aggregator started for brainstorm {self._current_topic_id}")
            
            # Monitor aggregator progress
            # CRITICAL: Start from current aggregator stats (0 for new brainstorm), not metadata
            # The aggregator's total_acceptances starts at 0 since we passed skip_stats_load=True
            status = await self._brainstorm_aggregator.get_status()
            last_acceptances = status.total_acceptances  # Should be 0 for new brainstorm
            last_rejections = status.total_rejections  # Track rejections for stat increments
            
            # CRITICAL BUG FIX: Don't reset counters if resuming from workflow state
            # Check if counters were already restored (non-zero means we're resuming)
            is_resuming = self._acceptance_count > 0 or self._rejection_count > 0
            
            if is_resuming:
                # Resuming: Use the restored counters from workflow state, don't reset
                logger.info(f"Resuming brainstorm with {self._acceptance_count} acceptances, "
                           f"{self._rejection_count} rejections from workflow state")
                # Set last_* to current values so we can track NEW acceptances/rejections from here
                last_acceptances = self._acceptance_count
                last_rejections = self._rejection_count
            else:
                # Fresh brainstorm: Initialize counters from aggregator stats (should be 0)
                self._acceptance_count = last_acceptances
                self._rejection_count = 0  # Reset rejection count for this brainstorm
                self._cleanup_removals = 0  # Reset cleanup removals for this brainstorm
                self._consecutive_rejections = 0
                self._exhaustion_signals = 0
                logger.info(f"Starting fresh brainstorm with {last_acceptances} acceptances")
            
            while self._running and not self._stop_event.is_set():
                # Get current aggregator stats
                status = await self._brainstorm_aggregator.get_status()
                current_acceptances = status.total_acceptances
                current_rejections = status.total_rejections
                current_cleanup_removals = status.removals_executed  # Track actual cleanup/pruning removals
                
                # Track cleanup removals for status display
                if current_cleanup_removals != self._cleanup_removals:
                    self._cleanup_removals = current_cleanup_removals
                
                # Track new acceptances/rejections
                if current_acceptances > last_acceptances:
                    new_acceptances = current_acceptances - last_acceptances
                    self._acceptance_count = current_acceptances
                    self._consecutive_rejections = 0
                    last_acceptances = current_acceptances
                    
                    # Increment total submissions accepted stat for acceptance rate calculation
                    await research_metadata.increment_stat("total_submissions_accepted", new_acceptances)
                    
                    # Update brainstorm metadata
                    await brainstorm_memory.update_metadata(
                        self._current_topic_id,
                        submission_count=current_acceptances
                    )
                    
                    # NOTE: Don't broadcast here - the aggregator already broadcasts 
                    # individual 'submission_accepted' events with submitter_id per submission
                    
                    # Reset consecutive rejections counter on acceptance
                    self._consecutive_rejections = 0
                    
                    # Save workflow state periodically (every 5 acceptances)
                    if current_acceptances % 5 == 0:
                        await self._save_workflow_state(tier="tier1_aggregation")
                    
                    # Check for hard limit of 80 acceptances (FORCE paper writing, skip completion review)
                    if self._acceptance_count >= 80:
                        logger.info(f"Hard limit of 80 acceptances reached for {self._current_topic_id}. Forcing paper writing transition.")
                        
                        # Broadcast hard limit reached event
                        await self._broadcast("brainstorm_hard_limit_reached", {
                            "topic_id": self._current_topic_id,
                            "acceptance_count": self._acceptance_count,
                            "message": "Brainstorm hard limit of 80 acceptances reached. Forcing paper writing."
                        })
                        
                        # Mark brainstorm complete
                        await brainstorm_memory.mark_complete(self._current_topic_id)
                        await research_metadata.mark_brainstorm_complete(self._current_topic_id)
                        
                        # Stop aggregator
                        await self._brainstorm_aggregator.stop()
                        
                        # Force transition to paper writing (skip completion review)
                        return True
                    
                    # Check for early completion triggers
                    early_trigger = await self._check_early_completion_triggers()
                    
                    # Check for completion review trigger (regular interval OR early trigger)
                    if self._should_run_completion_review() or early_trigger:
                        if early_trigger:
                            logger.info("EARLY completion trigger detected - bypassing interval check")
                        
                        write_paper = await self._run_completion_review()
                        
                        if write_paper:
                            # Stop aggregator
                            await self._brainstorm_aggregator.stop()
                            return True
                
                # Check for manual override trigger (before checking stop event)
                if self._manual_paper_writing_triggered:
                    logger.info("Manual override detected - transitioning to paper writing")
                    self._manual_paper_writing_triggered = False
                    return True
                
                # Track consecutive rejections and increment total rejections stat
                if current_rejections > last_rejections:
                    new_rejections = current_rejections - last_rejections
                    self._rejection_count = current_rejections
                    last_rejections = current_rejections
                    self._consecutive_rejections += new_rejections
                    
                    # Increment total submissions rejected stat for acceptance rate calculation
                    await research_metadata.increment_stat("total_submissions_rejected", new_rejections)
                    
                    # NOTE: Don't broadcast here - the aggregator already broadcasts 
                    # individual 'submission_rejected' events with submitter_id per submission
                    
                    # Check for hard limit of 10 consecutive rejections (with minimum 5 acceptances)
                    # This FORCES paper writing, similar to the 80 acceptance hard limit
                    if self._consecutive_rejections >= 10 and self._acceptance_count >= 5:
                        logger.info(f"Hard limit: {self._consecutive_rejections} consecutive rejections with {self._acceptance_count} acceptances. Forcing paper writing.")
                        
                        # Broadcast rejection hard limit event
                        await self._broadcast("brainstorm_rejection_limit_reached", {
                            "topic_id": self._current_topic_id,
                            "consecutive_rejections": self._consecutive_rejections,
                            "acceptance_count": self._acceptance_count,
                            "message": "10 consecutive rejections reached. Forcing paper writing."
                        })
                        
                        # Mark brainstorm complete
                        await brainstorm_memory.mark_complete(self._current_topic_id)
                        await research_metadata.mark_brainstorm_complete(self._current_topic_id)
                        
                        # Stop aggregator
                        await self._brainstorm_aggregator.stop()
                        
                        # Force transition to paper writing (skip completion review)
                        return True
                
                # Brief pause between checks
                await asyncio.sleep(2)
            
            # Stop aggregator on exit
            await self._brainstorm_aggregator.stop()
            return False
            
        finally:
            # Restore original shared training path
            system_config.shared_training_file = original_shared_path
            shared_training_memory.file_path = original_memory_path
            
            # CRITICAL: Clear in-memory data that was loaded from brainstorm database
            # This prevents cross-contamination if Part 1 manual mode starts after autonomous mode
            # Part 1 will re-initialize and reload its own database from the original path
            async with shared_training_memory._lock:
                shared_training_memory.insights.clear()
                shared_training_memory.submission_count = 0
                shared_training_memory.last_ragged_submission_count = 0
            logger.info("Cleared shared_training_memory in-memory data (will reload from file when needed)")
    
    async def _check_early_completion_triggers(self) -> bool:
        """
        Check for early completion review triggers:
        - 10+ consecutive rejections
        - 2+ submitter exhaustion signals
        
        Returns:
            True if early trigger detected
        """
        # Check consecutive rejections
        if self._consecutive_rejections >= 10:
            logger.info(f"Early completion trigger: {self._consecutive_rejections} consecutive rejections")
            return True
        
        # Check for exhaustion signals in recent rejection logs
        # Parse last 5 rejections from each submitter for exhaustion keywords
        exhaustion_keywords = [
            "cannot identify new mathematical content",
            "brainstorm topic appears thoroughly explored",
            "all major mathematical avenues have been covered",
            "exhausted",
            "no new insights"
        ]
        
        exhaustion_count = 0
        for submitter_id in [1, 2, 3]:
            rejections = await autonomous_rejection_logs.get_brainstorm_submitter_rejections(
                self._current_topic_id,
                submitter_id
            )
            
            # Check most recent rejection for exhaustion signals
            if rejections:
                recent_rejection = rejections[0]  # Most recent
                reasoning = recent_rejection.get("reasoning", "").lower()
                
                if any(keyword in reasoning for keyword in exhaustion_keywords):
                    exhaustion_count += 1
                    logger.info(f"Submitter {submitter_id} signaled exhaustion")
        
        if exhaustion_count >= 2:
            logger.info(f"Early completion trigger: {exhaustion_count} submitters signaled exhaustion")
            return True
        
        return False
    
    async def force_paper_writing(self) -> bool:
        """
        Manual override to force transition to paper writing.
        User acts as special submitter reviewer and decides brainstorm is ready.
        
        Returns:
            True if transition successful, False otherwise
        """
        try:
            # Validate state
            if not self._running or self._state.current_tier != "tier1_aggregation":
                logger.error("Cannot force paper writing: invalid state")
                return False
            
            if not self._current_topic_id:
                logger.error("Cannot force paper writing: no active brainstorm")
                return False
            
            if not self._brainstorm_aggregator:
                logger.error("Cannot force paper writing: no aggregator running")
                return False
            
            logger.info(f"MANUAL OVERRIDE: Forcing paper writing for brainstorm {self._current_topic_id}")
            
            # Broadcast manual override event
            await self._broadcast("manual_paper_writing_triggered", {
                "topic_id": self._current_topic_id,
                "submission_count": self._acceptance_count
            })
            
            # Stop the aggregator
            await self._brainstorm_aggregator.stop()
            logger.info("Brainstorm aggregator stopped by manual override")
            
            # Mark brainstorm complete
            await brainstorm_memory.mark_complete(self._current_topic_id)
            await research_metadata.mark_brainstorm_complete(self._current_topic_id)
            
            # Set flag to trigger paper writing on next loop iteration
            self._manual_paper_writing_triggered = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error forcing paper writing: {e}")
            return False
    
    async def force_tier3_final_answer(self, mode: str = "complete_current") -> dict:
        """
        Force transition to Tier 3 final answer generation.
        
        Args:
            mode: Either "complete_current" or "skip_incomplete"
                - complete_current: Finish current brainstorm->paper cycle first
                - skip_incomplete: Skip incomplete work, use completed papers only
        
        Returns:
            Dict with:
                - success: True if Tier 3 was initiated/ran successfully
                - result: "initiated" | "no_answer_known" | "complete" | "error"
                - message: Human-readable description
        """
        try:
            # Validate state
            if not self._running:
                logger.error("Cannot force Tier 3: autonomous research not running")
                return {"success": False, "result": "error", "message": "Autonomous research not running"}
            
            if self._state.current_tier == "tier3_final_answer":
                logger.error("Cannot force Tier 3: already in Tier 3")
                return {"success": False, "result": "error", "message": "Already in Tier 3"}
            
            # Check we have at least one completed paper
            all_papers = await paper_library.get_all_papers()
            if len(all_papers) == 0:
                logger.error("Cannot force Tier 3: no completed papers")
                return {"success": False, "result": "error", "message": "No completed papers"}
            
            logger.info(f"MANUAL OVERRIDE: Forcing Tier 3 with mode={mode}, current_tier={self._state.current_tier}")
            
            # Broadcast force event
            await self._broadcast("tier3_forced", {
                "mode": mode,
                "current_tier": self._state.current_tier,
                "completed_papers": len(all_papers)
            })
            
            if mode == "complete_current":
                # Complete current work first, then trigger Tier 3
                if self._state.current_tier == "tier1_aggregation":
                    # Force paper writing, then Tier 3 will be triggered after paper completes
                    logger.info("Force Tier 3 (complete_current): Forcing paper writing first")
                    
                    # Set flag to trigger Tier 3 after paper is done
                    self._force_tier3_after_paper = True
                    
                    # Trigger paper writing
                    paper_result = await self.force_paper_writing()
                    if paper_result:
                        return {"success": True, "result": "initiated", "message": "Tier 3 will start after paper writing completes"}
                    else:
                        return {"success": False, "result": "error", "message": "Failed to initiate paper writing"}
                    
                elif self._state.current_tier == "tier2_paper_writing":
                    # Set flag to trigger Tier 3 after current paper completes
                    logger.info("Force Tier 3 (complete_current): Will trigger after current paper")
                    self._force_tier3_after_paper = True
                    return {"success": True, "result": "initiated", "message": "Tier 3 will start after current paper completes"}
                    
            elif mode == "skip_incomplete":
                # Skip incomplete work, go directly to Tier 3
                logger.info("Force Tier 3 (skip_incomplete): Stopping current work and triggering Tier 3")
                
                # CRITICAL: Stop the main loop FIRST - this prevents race conditions
                # where the main loop continues creating new brainstorms while Tier 3 runs
                self._running = False
                self._stop_event.set()
                logger.info("Force Tier 3: Main loop stopped")
                
                # Stop current aggregator if it exists (don't check tier - state is unreliable)
                if self._brainstorm_aggregator:
                    try:
                        await self._brainstorm_aggregator.stop()
                        logger.info("Aggregator stopped for forced Tier 3")
                    except Exception as e:
                        logger.warning(f"Error stopping aggregator: {e}")
                    
                    # Mark current brainstorm as complete if we have one
                    if self._current_topic_id:
                        try:
                            await brainstorm_memory.mark_complete(self._current_topic_id)
                            logger.info(f"Marked brainstorm {self._current_topic_id} as complete (forced Tier 3)")
                        except Exception as e:
                            logger.warning(f"Error marking brainstorm complete: {e}")
                
                # Stop compiler if it exists (don't check tier - state is unreliable)
                if self._paper_compiler:
                    try:
                        await self._paper_compiler.stop()
                        logger.info("Compiler stopped for forced Tier 3")
                    except Exception as e:
                        logger.warning(f"Error stopping compiler: {e}")
                
                # CRITICAL: Wait for main loop to actually exit before resetting flags
                # The main loop checks these flags, and if we reset them too quickly,
                # the loop will see _running=True and continue creating brainstorms!
                # This delay ensures the main loop's next iteration sees _running=False and exits.
                await asyncio.sleep(0.5)
                logger.info("Force Tier 3: Waited for main loop to exit")
                
                # CRITICAL: Reset flags for Tier 3 execution
                # Now that the main loop has exited, we can reset flags for Tier 3's internal loops
                # (chapter writing, paper compilation monitoring)
                self._running = True
                self._stop_event.clear()
                logger.info("Force Tier 3: Flags reset for Tier 3 execution")
                
                # Run Tier 3 synchronously - main loop is stopped, we own the execution now
                # This prevents the race condition where main loop and Tier 3 run in parallel
                logger.info("Force Tier 3: Starting Tier 3 workflow synchronously")
                result = await self._tier3_final_answer_workflow()
                
                # Handle Tier 3 result
                if result:
                    # Tier 3 completed successfully with a final answer
                    # Reset flags to stopped state - research is complete
                    self._running = False
                    self._stop_event.set()
                    logger.info("Force Tier 3: Flags reset to stopped state after successful completion")
                    
                    try:
                        await research_metadata.clear_workflow_state()
                        logger.info("Force Tier 3: Workflow state cleared after successful completion")
                    except Exception as e:
                        logger.warning(f"Failed to clear workflow state after forced Tier 3: {e}")
                    
                    # Broadcast completion event for forced Tier 3
                    await self._broadcast("tier3_complete", {
                        "format": final_answer_memory.get_state().answer_format or "unknown",
                        "forced": True
                    })
                    return {"success": True, "result": "complete", "message": "Tier 3 completed with final answer generated"}
                else:
                    # Tier 3 ran successfully but determined no_answer_known
                    # This is a VALID outcome - it means more research is needed
                    # Per spec: "returns to normal research" when no_answer_known
                    logger.info("Force Tier 3: Tier 3 completed but determined more research is needed (no_answer_known)")
                    logger.info("Force Tier 3: Restarting main research loop to generate more papers")
                    
                    # Flags are already in running state (set at lines 1737-1738)
                    # Create a background task to resume the main research loop
                    asyncio.create_task(self._resume_research_loop_after_tier3())
                    
                    return {
                        "success": True, 
                        "result": "no_answer_known", 
                        "message": "Tier 3 ran successfully but determined more research is needed. Resuming autonomous research to generate more papers."
                    }
            
            return {"success": False, "result": "error", "message": "Invalid mode or state"}
            
        except Exception as e:
            logger.error(f"Error forcing Tier 3: {e}")
            return {"success": False, "result": "error", "message": str(e)}
    
    def _should_run_completion_review(self) -> bool:
        """Check if completion review should run."""
        # Run every 10 acceptances
        interval = system_config.autonomous_completion_review_interval
        return (self._acceptance_count > 0 and 
                self._acceptance_count % interval == 0)
    
    async def _run_completion_review(self) -> bool:
        """
        Run completion review with self-validation.
        
        Returns:
            True if should write paper, False if should continue
        """
        logger.info(f"Running completion review at {self._acceptance_count} acceptances")
        
        await self._broadcast("completion_review_started", {
            "topic_id": self._current_topic_id,
            "submission_count": self._acceptance_count
        })
        
        await research_metadata.increment_stat("completion_reviews_run")
        
        # Get brainstorm content
        metadata = await brainstorm_memory.get_metadata(self._current_topic_id)
        if metadata is None:
            logger.error("Cannot run completion review: brainstorm not found")
            return False
        
        brainstorm_content = await brainstorm_memory.get_database_content(self._current_topic_id)
        
        # Run completion review with self-validation
        result, is_validated = await self._completion_reviewer.review_completion(
            user_research_prompt=self._user_research_prompt,
            topic_id=self._current_topic_id,
            topic_prompt=metadata.topic_prompt,
            brainstorm_database=brainstorm_content,
            submission_count=self._acceptance_count
        )
        
        if result is None:
            logger.error("Completion review failed")
            return False
        
        await self._broadcast("completion_review_result", {
            "topic_id": self._current_topic_id,
            "decision": result.decision,
            "reasoning": result.reasoning[:500]
        })
        
        if result.decision == "write_paper":
            logger.info("Completion review: WRITE PAPER")
            
            # Mark brainstorm complete
            await brainstorm_memory.mark_complete(self._current_topic_id)
            await research_metadata.mark_brainstorm_complete(self._current_topic_id)
            
            return True
        else:
            logger.info("Completion review: CONTINUE BRAINSTORM")
            return False
    
    # ========================================================================
    # PHASE 3: PAPER COMPILATION
    # ========================================================================
    
    async def _paper_compilation_workflow(self) -> bool:
        """
        Complete paper compilation workflow.
        Order: Reference selection -> Title -> Body -> Conclusion -> Intro -> Abstract
        
        Supports RESUME: If self._current_paper_id is already set, skips title/reference
        selection and continues paper compilation where it left off.
        
        Returns:
            True if paper was successfully compiled, False otherwise.
        """
        self._state.current_tier = "tier2_paper_writing"
        
        # Set phase for API logging
        api_client_manager.set_autonomous_phase("paper_compilation")
        
        logger.info(f"Starting paper compilation for brainstorm {self._current_topic_id}")
        
        # Check if we're resuming an in-progress paper
        # This flag tracks whether we're resuming (for passing to _compile_paper)
        is_resuming_paper = False
        
        if self._current_paper_id:
            # RESUME MODE: Skip title/reference selection, continue with existing paper
            paper_id = self._current_paper_id
            is_resuming_paper = True
            
            # Get paper title from metadata
            paper_metadata = await research_metadata.get_paper_entry(paper_id)
            if paper_metadata:
                paper_title = paper_metadata.get("title", f"Paper {paper_id}")
                self._current_paper_title = paper_title
            else:
                # Fallback - try to get title from compiler outline
                from backend.compiler.memory.outline_memory import outline_memory as compiler_outline_memory
                outline = await compiler_outline_memory.get_outline()
                if outline and outline.strip():
                    # Use first line as title or generate default
                    paper_title = f"Paper {paper_id}"
                    self._current_paper_title = paper_title
                else:
                    paper_title = f"Paper {paper_id}"
                    self._current_paper_title = paper_title
            
            # Use already-selected reference papers
            reference_paper_ids = self._current_reference_papers
            
            logger.info(f"RESUME: Continuing paper {paper_id} compilation (title: {paper_title[:50]}...)")
            
            await self._broadcast("paper_writing_resumed", {
                "paper_id": paper_id,
                "title": paper_title,
                "source_brainstorm_id": self._current_topic_id
            })
        else:
            # FRESH START: Run full title/reference selection workflow
            # Step 1: Reference selection (if papers exist)
            reference_paper_ids = await self._reference_selection_workflow()
            
            if self._stop_event.is_set():
                return False
            
            # Step 2: Title selection
            paper_title = await self._paper_title_selection()
            
            if paper_title is None:
                logger.error("Paper title selection failed")
                return False
            
            if self._stop_event.is_set():
                return False
            
            # Generate paper ID
            paper_id = await research_metadata.generate_paper_id()
            self._current_paper_id = paper_id
            self._current_paper_title = paper_title
            
            # Update paper tracker with title
            if self._current_paper_tracker:
                self._current_paper_tracker.paper_title = paper_title
            
            await self._broadcast("paper_writing_started", {
                "paper_id": paper_id,
                "title": paper_title,
                "source_brainstorm_id": self._current_topic_id
            })
        
        # Save workflow state with paper details
        await self._save_workflow_state(tier="tier2_paper_writing", phase="outline")
        
        # Step 3: Paper compilation (using Part 2 compiler infrastructure)
        # Pass is_resume flag and phase to preserve existing paper content when resuming
        paper_content = await self._compile_paper(
            paper_id=paper_id,
            paper_title=paper_title,
            reference_paper_ids=reference_paper_ids,
            is_resume=is_resuming_paper,
            resume_phase=self._resume_paper_phase if is_resuming_paper else None
        )
        
        # Clear resume state after compilation attempt (whether success or failure)
        # This prevents stale resume state from affecting future papers
        if is_resuming_paper:
            self._resume_paper_phase = None
        
        if paper_content is None:
            logger.error("Paper compilation failed")
            return False
        
        # Get final outline
        from backend.compiler.memory.outline_memory import outline_memory as compiler_outline_memory
        final_outline = await compiler_outline_memory.get_outline()
        
        # Step 4: Save completed paper
        await self._handle_paper_completion(
            paper_id=paper_id,
            title=paper_title,
            content=paper_content,
            outline=final_outline or "[Outline not available]",
            reference_paper_ids=reference_paper_ids
        )
        
        return True
    
    async def _reference_selection_workflow(self) -> List[str]:
        """
        Run additional reference paper selection workflow before paper writing.
        
        This allows the AI to select ADDITIONAL references discovered to be relevant
        during brainstorming, up to the 6 paper limit.
        
        The papers already selected during pre-brainstorm reference selection are
        preserved and shown as "ALREADY SELECTED" to the AI.
        
        Returns:
            Combined list of all selected paper_ids (max 6 total)
        """
        # Start with papers already selected during pre-brainstorm
        already_selected = self._current_reference_papers.copy()
        
        # Check how many more we can select
        remaining_slots = 6 - len(already_selected)
        if remaining_slots <= 0:
            logger.info(f"Already have {len(already_selected)} reference papers (max 6), skipping additional selection")
            return already_selected
        
        # Get available papers
        papers_summary = await autonomous_rag_manager.get_all_papers_summary()
        
        if not papers_summary:
            logger.info("No papers available for additional reference selection")
            return already_selected
        
        # Filter out already selected papers from available list
        available_for_selection = [
            p for p in papers_summary 
            if p.get("paper_id") not in already_selected
        ]
        
        if not available_for_selection:
            logger.info("All available papers already selected, skipping additional selection")
            return already_selected
        
        # Get brainstorm summary
        brainstorm_summary = await autonomous_rag_manager.get_brainstorm_summary(
            self._current_topic_id
        )
        
        metadata = await brainstorm_memory.get_metadata(self._current_topic_id)
        topic_prompt = metadata.topic_prompt if metadata else ""
        
        await self._broadcast("reference_selection_started", {
            "topic_id": self._current_topic_id,
            "mode": "additional",
            "already_selected": len(already_selected),
            "available_papers": len(available_for_selection)
        })
        
        # Run reference selection in "additional" mode
        additional_ids = await self._reference_selector.select_references(
            user_research_prompt=self._user_research_prompt,
            topic_prompt=topic_prompt,
            brainstorm_summary=brainstorm_summary,
            available_papers=available_for_selection,
            mode="additional",  # Additional selection mode
            already_selected=already_selected  # Papers already selected
        )
        
        # Combine with already selected (respecting max 6 limit)
        combined = already_selected + additional_ids
        if len(combined) > 6:
            logger.warning(f"Combined references ({len(combined)}) exceeds limit, truncating to 6")
            combined = combined[:6]
        
        # Update current reference papers
        self._current_reference_papers = combined
        
        await self._broadcast("reference_selection_complete", {
            "topic_id": self._current_topic_id,
            "mode": "additional",
            "selected_count": len(combined),  # Total count after adding new papers
            "newly_added": len(additional_ids),
            "selected_papers": combined
        })
        
        logger.info(f"Additional reference selection: {len(additional_ids)} new + {len(already_selected)} existing = {len(combined)} total")
        return combined
    
    async def _paper_title_selection(self) -> Optional[str]:
        """Select paper title."""
        metadata = await brainstorm_memory.get_metadata(self._current_topic_id)
        if metadata is None:
            return None
        
        # Get brainstorm summary
        brainstorm_summary = await autonomous_rag_manager.get_brainstorm_summary(
            self._current_topic_id
        )
        
        # Get existing papers from this brainstorm
        existing_papers = await research_metadata.get_papers_by_brainstorm(
            self._current_topic_id
        )
        
        # Select title
        title = await self._title_selector.select_title(
            user_research_prompt=self._user_research_prompt,
            topic_prompt=metadata.topic_prompt,
            brainstorm_summary=brainstorm_summary,
            existing_papers_from_brainstorm=existing_papers
        )
        
        return title
    
    async def _compile_paper(
        self,
        paper_id: str,
        paper_title: str,
        reference_paper_ids: List[str],
        is_resume: bool = False,
        resume_phase: Optional[str] = None
    ) -> Optional[str]:
        """
        Compile paper using Part 2 compiler infrastructure.
        Order: Body -> Conclusion -> Introduction -> Abstract
        
        Integrates with actual compiler coordinator.
        
        Args:
            paper_id: Unique paper identifier
            paper_title: Title of the paper
            reference_paper_ids: List of reference paper IDs to include in RAG
            is_resume: If True, continue from existing paper content instead of starting fresh
            resume_phase: If resuming, the phase to continue from (body/conclusion/introduction/abstract)
        """
        if is_resume:
            logger.info(f"RESUME: Continuing paper compilation: {paper_title} (phase: {resume_phase})")
        else:
            logger.info(f"Compiling paper: {paper_title}")
        
        # Initialize compiler for this paper
        self._paper_compiler = CompilerCoordinator()
        
        try:
            # CRITICAL: Clear RAG before autonomous paper compilation to prevent cross-contamination
            # This removes any old user uploads, previous session data, or Part 1 aggregator content
            # Even on resume, we need to reload RAG since it's not persisted across restarts
            logger.info("Clearing RAG for autonomous paper compilation...")
            await asyncio.to_thread(rag_manager.clear_all_documents)
            logger.info("RAG cleared successfully")
            
            # Initialize compiler with paper title as prompt
            # CRITICAL: skip_aggregator_db=True prevents loading Part 1 aggregator database
            # Autonomous mode should ONLY use the brainstorm database for this topic
            await self._paper_compiler.initialize(
                compiler_prompt=f"Write a mathematical research paper titled: {paper_title}",
                validator_model=self._validator_model,
                high_context_model=self._high_context_model,
                high_param_model=self._high_param_model,
                critique_submitter_model=self._critique_submitter_model,
                skip_aggregator_db=True,  # Don't load Part 1 aggregator - use brainstorm DB only
                # Pass OpenRouter provider configs for all compiler roles
                validator_provider=self._validator_provider,
                validator_openrouter_provider=self._validator_openrouter_provider,
                validator_lm_studio_fallback=self._validator_lm_studio_fallback,
                high_context_provider=self._high_context_provider,
                high_context_openrouter_provider=self._high_context_openrouter_provider,
                high_context_lm_studio_fallback=self._high_context_lm_studio_fallback,
                high_param_provider=self._high_param_provider,
                high_param_openrouter_provider=self._high_param_openrouter_provider,
                high_param_lm_studio_fallback=self._high_param_lm_studio_fallback,
                critique_submitter_provider=self._critique_submitter_provider,
                critique_submitter_openrouter_provider=self._critique_submitter_openrouter_provider,
                critique_submitter_lm_studio_fallback=self._critique_submitter_lm_studio_fallback
            )
            
            # Set WebSocket broadcaster for compiler events
            if self._broadcast_callback:
                self._paper_compiler.websocket_broadcaster = self._broadcast_callback
            
            # Enable autonomous section order constraint
            self._paper_compiler.enable_autonomous_mode()
            
            # CRITICAL: Restore the saved phase when resuming
            # enable_autonomous_mode() sets phase to "body" by default
            # But when resuming, we need to continue from where we left off
            if is_resume and resume_phase:
                self._paper_compiler.autonomous_section_phase = resume_phase
                logger.info(f"RESUME: Restored compiler phase to '{resume_phase}'")
            
            # CRITICAL FIX: Only clear paper for fresh starts, NOT for resume
            # When resuming, the paper content in compiler_paper.txt should be preserved
            if not is_resume:
                # Clear paper and outline from any previous paper before starting new one
                # This is critical because paper_memory and outline_memory are global singletons
                # that persist content across paper compilations
                await self._paper_compiler.clear_paper()
                logger.info(f"Cleared previous paper/outline for fresh paper {paper_id}")
            else:
                # On resume, check if compiler memory has the paper content
                # If not (e.g., after system restart), load it from the saved paper in library
                existing_paper = await compiler_paper_memory.get_paper()
                existing_outline = await outline_memory.get_outline()
                paper_len = len(existing_paper) if existing_paper else 0
                outline_len = len(existing_outline) if existing_outline else 0
                
                if paper_len == 0:
                    # Compiler memory is empty - need to load from saved paper in library
                    logger.info(f"RESUME: Compiler memory empty, loading saved paper {paper_id} from library")
                    await self._load_saved_paper_to_compiler(paper_id)
                    # Re-check lengths after loading
                    existing_paper = await compiler_paper_memory.get_paper()
                    existing_outline = await outline_memory.get_outline()
                    paper_len = len(existing_paper) if existing_paper else 0
                    outline_len = len(existing_outline) if existing_outline else 0
                
                logger.info(f"RESUME: Paper state - paper ({paper_len} chars), outline ({outline_len} chars)")
            
            # Load brainstorm database into compiler RAG
            # This is now the ONLY aggregator content loaded (no Part 1 pollution)
            # IMPORTANT: Use brainstorm_memory.get_database_path() for session-aware path resolution
            brainstorm_db_path = brainstorm_memory.get_database_path(self._current_topic_id)
            if os.path.exists(brainstorm_db_path):
                logger.info(f"Loading brainstorm database into compiler RAG: {brainstorm_db_path}")
                await rag_manager.add_document(
                    brainstorm_db_path,
                    chunk_sizes=[512],  # Use standard chunk size for brainstorm
                    is_user_file=True  # High priority, permanent
                )
                logger.info("Brainstorm database loaded into compiler RAG")
            else:
                logger.warning(f"Brainstorm database not found: {brainstorm_db_path}")
            
            # Load reference papers into compiler RAG (if any)
            if reference_paper_ids:
                logger.info(f"Loading {len(reference_paper_ids)} reference papers into compiler RAG")
                for ref_paper_id in reference_paper_ids:
                    # IMPORTANT: Use paper_library.get_paper_path() for session-aware path resolution
                    paper_path = paper_library.get_paper_path(ref_paper_id)
                    if os.path.exists(paper_path):
                        await rag_manager.add_document(
                            paper_path,
                            chunk_sizes=[512],
                            is_user_file=False  # Lower priority than brainstorm
                        )
                        logger.info(f"Reference paper loaded: {ref_paper_id}")
                    else:
                        logger.warning(f"Reference paper not found: {paper_path}")
                logger.info("All reference papers loaded into compiler RAG")
            
            # Start compiler
            await self._paper_compiler.start()
            logger.info(f"Compiler started for paper {paper_id}")
            
            # Monitor compiler progress
            # The compiler runs its full workflow:
            # 1. Create and validate outline
            # 2. Write body sections in order
            # 3. Write conclusion
            # 4. Write introduction
            # 5. Write abstract (signals completion)
            
            abstract_written = False
            last_tracked_phase = None
            while self._running and not self._stop_event.is_set():
                # Get current paper content
                current_paper = await compiler_paper_memory.get_paper()
                
                # Sync compiler phase to workflow state (fix for phase tracking bug)
                # This ensures the workflow state accurately reflects compiler progress
                compiler_phase = self._paper_compiler.autonomous_section_phase
                if compiler_phase and compiler_phase != last_tracked_phase:
                    logger.info(f"Phase updated: {last_tracked_phase} → {compiler_phase}")
                    last_tracked_phase = compiler_phase
                    await self._save_workflow_state(tier="tier2_paper_writing", phase=compiler_phase)
                
                # Check if abstract has been written
                # Abstract is the LAST section written, so its presence signals completion
                if current_paper and self._has_abstract(current_paper):
                    logger.info("Abstract detected - paper compilation complete")
                    abstract_written = True
                    break
                
                # Check if compiler has stopped (error or other reason)
                if not self._paper_compiler.is_running:
                    logger.warning("Compiler stopped unexpectedly")
                    break
                
                # Brief pause between checks
                await asyncio.sleep(3)
            
            # Stop compiler
            await self._paper_compiler.stop()
            
            if not abstract_written:
                logger.error("Paper compilation did not complete (no abstract)")
                return None
            
            # Get final paper content
            final_paper = await compiler_paper_memory.get_paper()
            
            # Extract abstract for storage
            self._current_abstract = self._extract_abstract(final_paper)
            
            return final_paper
            
        except Exception as e:
            logger.error(f"Error during paper compilation: {e}")
            if self._paper_compiler:
                await self._paper_compiler.stop()
            return None
    
    def _has_abstract(self, paper_content: str) -> bool:
        """Check if paper contains an abstract section."""
        # Look for common abstract section markers
        abstract_patterns = [
            r"##\s*Abstract",
            r"#\s*Abstract",
            r"\*\*Abstract\*\*",
            r"Abstract\s*\n",
        ]
        
        for pattern in abstract_patterns:
            if re.search(pattern, paper_content, re.IGNORECASE):
                return True
        
        return False
    
    def _extract_abstract(self, paper_content: str) -> str:
        """Extract abstract text from paper."""
        # Try to find abstract section
        abstract_patterns = [
            r"##\s*Abstract\s*\n(.*?)(?=\n##|\n#|\Z)",
            r"#\s*Abstract\s*\n(.*?)(?=\n##|\n#|\Z)",
            r"\*\*Abstract\*\*\s*\n(.*?)(?=\n##|\n#|\n\*\*|\Z)",
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, paper_content, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Limit to first 500 chars for metadata
                return abstract[:500] if len(abstract) > 500 else abstract
        
        # Fallback: first paragraph after title
        lines = paper_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                # Found first non-heading line
                return lines[i].strip()[:500]
        
        return "[Abstract not found]"
    
    async def _handle_paper_completion(
        self,
        paper_id: str,
        title: str,
        content: str,
        outline: str,
        reference_paper_ids: List[str],
        mark_complete: bool = True
    ) -> None:
        """
        Handle paper save - optionally mark as complete.
        
        Args:
            mark_complete: If True, clears paper state and marks as finished.
                          If False, keeps paper state (for mid-progress saves).
        """
        # Get brainstorm content for caching
        brainstorm_content = await brainstorm_memory.get_database_content(
            self._current_topic_id
        )
        
        # Extract abstract (in full implementation, would be properly extracted)
        abstract = getattr(self, '_current_abstract', '[Abstract not available]')
        
        # Get model usage from per-paper tracker
        model_usage = None
        generation_date = None
        wolfram_calls = None
        if self._current_paper_tracker:
            model_usage = self._current_paper_tracker.get_models_dict()
            generation_date = self._current_paper_tracker.generation_date
            wolfram_calls = self._current_paper_tracker.get_wolfram_call_count()
            logger.info(f"Paper {paper_id}: tracked {len(model_usage)} models, {self._current_paper_tracker.total_calls} API calls, {wolfram_calls} Wolfram calls")
        
        # Get reference paper model usage for "Possible Models Used for Additional Reference" section
        reference_paper_models = None
        if reference_paper_ids:
            reference_model_usages = []
            for ref_paper_id in reference_paper_ids:
                ref_metadata = await paper_library.get_metadata(ref_paper_id)
                if ref_metadata and ref_metadata.model_usage:
                    reference_model_usages.append(ref_metadata.model_usage)
            
            if reference_model_usages:
                reference_paper_models = PaperModelTracker.aggregate_reference_models(reference_model_usages)
                logger.info(f"Aggregated reference models from {len(reference_model_usages)} papers: {len(reference_paper_models)} unique models")
        
        # Generate author attribution header and model credits footer
        final_content = content
        if self._current_paper_tracker:
            # Generate attribution header
            attribution_header = self._current_paper_tracker.generate_author_attribution(
                user_prompt=self._user_research_prompt,
                paper_title=title,
                reference_paper_models=reference_paper_models
            )
            
            # Generate model credits footer
            model_credits = self._current_paper_tracker.generate_model_credits()
            
            # Combine: header + content + footer
            final_content = attribution_header + "\n" + content
            if model_credits:
                final_content = final_content + "\n" + model_credits
            
            logger.info("Added author attribution and model credits to paper")
        
        # Save paper with appropriate status
        paper_metadata = await paper_library.save_paper(
            paper_id=paper_id,
            title=title,
            content=final_content,
            outline=outline,
            abstract=abstract,
            source_brainstorm_ids=[self._current_topic_id],
            source_brainstorm_content=brainstorm_content,
            referenced_papers=reference_paper_ids,
            model_usage=model_usage,
            generation_date=generation_date,
            status="complete" if mark_complete else "in_progress",
            wolfram_calls=wolfram_calls
        )
        
        # Register in central metadata
        await research_metadata.register_paper(paper_metadata)
        
        # Add paper reference to brainstorm
        await brainstorm_memory.add_paper_reference(self._current_topic_id, paper_id)
        
        # Update counts
        self._papers_completed_count += 1
        
        await self._broadcast("paper_completed", {
            "paper_id": paper_id,
            "title": title,
            "word_count": paper_metadata.word_count
        })
        
        # Trigger auto-critique generation in background (only if marking as complete)
        if mark_complete:
            asyncio.create_task(self._auto_generate_paper_critique(
                paper_id=paper_id,
                paper_title=title
            ))
        
        # Only clear paper state if marking as complete
        if mark_complete:
            # Clear paper-specific workflow state (paper is complete)
            self._current_paper_id = None
            self._current_paper_title = None
            
            # Clear per-paper model tracking (will be re-initialized for next paper)
            self._current_paper_tracker = None
            
            await self._save_workflow_state(tier=None, phase=None)
            
            logger.info(f"Paper completed: {paper_id} ({paper_metadata.word_count} words)")
        else:
            # Paper saved but still in progress - keep state
            logger.info(f"Paper saved (in progress): {paper_id} ({paper_metadata.word_count} words)")
    
    async def _auto_generate_paper_critique(
        self,
        paper_id: str,
        paper_title: str
    ) -> None:
        """
        Automatically generate a critique for a completed paper.
        If the average rating is >= 6.25, emit a WebSocket event for popup notification.
        
        This runs in the background and failures are logged but don't affect paper completion.
        """
        from backend.shared.critique_prompts import build_critique_prompt
        from backend.shared.critique_memory import save_critique
        from backend.shared.api_client_manager import api_client_manager
        from backend.shared.json_parser import parse_json
        from backend.shared.utils import count_tokens
        from backend.shared.models import PaperCritique, ModelConfig
        import uuid
        from datetime import datetime
        
        try:
            logger.info(f"Auto-generating critique for paper {paper_id}: {paper_title}")
            
            # Check if validator config exists
            if not self._validator_model:
                logger.warning(f"Cannot auto-generate critique: No validator model configured")
                return
            
            # Get paper content
            paper_content = await paper_library.get_paper_content(paper_id)
            if not paper_content:
                logger.error(f"Cannot auto-generate critique: Paper {paper_id} content not found")
                return
            
            # Build critique prompt string (returns str, not messages list)
            from backend.shared.critique_prompts import DEFAULT_CRITIQUE_PROMPT
            
            prompt = build_critique_prompt(
                paper_content=paper_content,
                paper_title=paper_title,
                custom_prompt=None  # Use default prompt
            )
            
            # Wrap in messages list for API call
            messages = [{"role": "user", "content": prompt}]
            
            # Check context window (attribute is _validator_context, not _validator_context_window)
            prompt_tokens = count_tokens(prompt)
            available_tokens = self._validator_context - self._validator_max_tokens - 500
            
            if prompt_tokens > available_tokens:
                logger.error(
                    f"Cannot auto-generate critique for paper {paper_id}: "
                    f"Content too large ({prompt_tokens} tokens > {available_tokens} available)"
                )
                return
            
            # Configure the paper_critic role before making the API call
            # This ensures proper routing to OpenRouter or LM Studio
            api_client_manager.configure_role(
                "paper_critic",
                ModelConfig(
                    provider=self._validator_provider,
                    model_id=self._validator_model,
                    openrouter_model_id=self._validator_model if self._validator_provider == "openrouter" else None,
                    openrouter_provider=self._validator_openrouter_provider,
                    lm_studio_fallback_id=self._validator_lm_studio_fallback,
                    context_window=self._validator_context,
                    max_output_tokens=self._validator_max_tokens
                )
            )
            
            # Generate critique
            response = await api_client_manager.generate_completion(
                task_id=f"auto_paper_critique_{paper_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                role_id="paper_critic",
                model=self._validator_model,
                messages=messages,
                max_tokens=self._validator_max_tokens,
                temperature=0.0
            )
            
            # Parse response
            response_content = ""
            if response.get("choices"):
                message = response["choices"][0].get("message", {})
                response_content = message.get("content", "") or message.get("reasoning", "")
            
            if not response_content:
                logger.error(f"Empty response from validator model for paper {paper_id}")
                return
            
            # Parse JSON
            try:
                critique_data = parse_json(response_content)
            except Exception as e:
                logger.warning(f"Failed to parse critique JSON for paper {paper_id}: {e}")
                # Create fallback structure
                critique_data = {
                    "novelty_rating": 0,
                    "novelty_feedback": "Unable to parse structured response",
                    "correctness_rating": 0,
                    "correctness_feedback": "Unable to parse structured response",
                    "impact_rating": 0,
                    "impact_feedback": "Unable to parse structured response",
                    "full_critique": response_content
                }
            
            # Extract ratings
            novelty = critique_data.get("novelty_rating", 0)
            correctness = critique_data.get("correctness_rating", 0)
            impact = critique_data.get("impact_rating", 0)
            
            # Calculate average rating
            average_rating = (novelty + correctness + impact) / 3.0
            
            # Create critique object
            critique = PaperCritique(
                critique_id=str(uuid.uuid4()),
                model_id=self._validator_model,
                provider=self._validator_provider,
                host_provider=self._validator_openrouter_provider,
                date=datetime.now(),
                prompt_used=DEFAULT_CRITIQUE_PROMPT,  # Always uses default for auto-critiques
                novelty_rating=novelty,
                novelty_feedback=critique_data.get("novelty_feedback", ""),
                correctness_rating=correctness,
                correctness_feedback=critique_data.get("correctness_feedback", ""),
                impact_rating=impact,
                impact_feedback=critique_data.get("impact_feedback", ""),
                full_critique=critique_data.get("full_critique", "")
            )
            
            # Save critique
            from pathlib import Path
            
            paper_path = paper_library.get_paper_path(paper_id)  # Synchronous, returns str
            if paper_path:
                base_path = Path(paper_path).parent
                await save_critique(
                    paper_type="autonomous_paper",
                    critique=critique,
                    paper_id=paper_id,
                    base_path=str(base_path)
                )
                logger.info(
                    f"Auto-critique saved for paper {paper_id}: "
                    f"avg={average_rating:.1f} (N={novelty}, C={correctness}, I={impact})"
                )
                
                # Always emit critique completion event (for badge refresh)
                await self._broadcast("paper_critique_completed", {
                    "paper_id": paper_id,
                    "average_rating": round(average_rating, 1)
                })
                
                # If average rating >= 6.25, also emit high-score event for popup notification
                if average_rating >= 6.25:
                    await self._broadcast("high_score_critique", {
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "average_rating": round(average_rating, 1),
                        "novelty_rating": novelty,
                        "correctness_rating": correctness,
                        "impact_rating": impact,
                        "timestamp": datetime.now().isoformat()
                    })
                    logger.info(f"High-score critique notification sent for paper {paper_id} (avg={average_rating:.1f})")
            else:
                logger.error(f"Cannot save critique: Paper path not found for {paper_id}")
        
        except Exception as e:
            # Log but don't crash - auto-critique is non-critical
            logger.error(f"Auto-critique generation failed for paper {paper_id}: {e}", exc_info=True)
    
    # ========================================================================
    # PAPER REDUNDANCY CHECK
    # ========================================================================
    
    async def _check_paper_redundancy(self) -> None:
        """Check paper library for redundancy (every 3 papers)."""
        # CRITICAL: Skip redundancy check if Tier 3 is active
        # This prevents accidentally purging papers that are being used in the final volume
        if self._tier3_active:
            logger.debug("Skipping paper redundancy check: Tier 3 is active")
            return
        
        if not self._redundancy_checker.should_check(
            self._papers_completed_count,
            self._last_redundancy_check_at
        ):
            return
        
        logger.info("Running paper redundancy check")
        await research_metadata.increment_stat("paper_redundancy_reviews_run")
        
        # Get papers summary
        papers_summary = await autonomous_rag_manager.get_all_papers_summary()
        
        # Check for redundancy
        result = await self._redundancy_checker.check_redundancy(
            user_research_prompt=self._user_research_prompt,
            papers_summary=papers_summary
        )
        
        if result and result.should_remove and result.paper_id:
            # Execute removal
            success = await self._redundancy_checker.execute_removal(result.paper_id)
            
            await self._broadcast("paper_redundancy_review", {
                "should_remove": True,
                "paper_id": result.paper_id,
                "reasoning": result.reasoning,
                "removed": success
            })
        else:
            await self._broadcast("paper_redundancy_review", {
                "should_remove": False,
                "paper_id": None,
                "reasoning": result.reasoning if result else "Check failed"
            })
        
        # Update tracking
        self._last_redundancy_check_at = self._papers_completed_count
    
    # ========================================================================
    # TIER 3: FINAL ANSWER GENERATION
    # ========================================================================
    
    async def _should_trigger_tier3(self) -> bool:
        """
        Check if Tier 3 final answer generation should be triggered.
        Triggers every 5 papers in the library, or if manually forced.
        Uses actual paper library count, not internal counters.
        """
        # Check force flags first
        if self._force_tier3_immediate:
            logger.info("Tier 3 trigger: Force immediate flag set")
            self._force_tier3_immediate = False  # Clear flag
            return True
        
        if self._force_tier3_after_paper:
            logger.info("Tier 3 trigger: Force after paper flag set")
            self._force_tier3_after_paper = False  # Clear flag
            return True
        
        # Normal trigger: every 5 papers in library
        interval = 5  # Check every 5 papers
        # Get actual paper count from library (not internal counter)
        paper_counts = await paper_library.count_papers()
        actual_paper_count = paper_counts["active"]
        
        papers_since_last_check = actual_paper_count - self._last_tier3_check_at
        return papers_since_last_check >= interval
    
    # ========================================================================
    # TIER 3 CRASH RECOVERY METHODS
    # ========================================================================
    
    async def _resume_tier3_workflow(self, tier3_state) -> bool:
        """
        Resume Tier 3 from saved state after a crash or restart.
        
        Args:
            tier3_state: The FinalAnswerState loaded from persistence
            
        Returns:
            True if final answer was successfully generated and system should stop,
            False if more research is needed (no_answer_known) and should continue.
        """
        logger.info("=" * 60)
        logger.info("TIER 3: RESUMING FROM SAVED STATE")
        logger.info(f"Status: {tier3_state.status}, Format: {tier3_state.answer_format}")
        logger.info("=" * 60)
        
        # Re-initialize Tier 3 context
        self._state.current_tier = "tier3_final_answer"
        self._tier3_active = True
        
        # Set phase for API logging
        api_client_manager.set_autonomous_phase("tier3")
        
        # Re-initialize model usage tracking
        await final_answer_memory.initialize_model_tracking(self._user_research_prompt)
        
        # Set up model tracking callback
        async def tier3_model_tracking_callback(model_id: str) -> None:
            await final_answer_memory.track_model_call(model_id)
            if self._current_paper_tracker:
                self._current_paper_tracker.track_call(model_id)
        
        api_client_manager.set_model_tracking_callback(tier3_model_tracking_callback)
        logger.info("Tier 3: Model tracking re-enabled for resume")
        
        await self._broadcast("tier3_resumed", {
            "status": tier3_state.status,
            "format": tier3_state.answer_format
        })
        
        try:
            status = tier3_state.status
            
            if status in ["idle", "phase1_assessment", "assessing"]:
                # Assessment not completed - restart Tier 3 from beginning
                logger.info("Tier 3 resume: Starting from beginning (assessment incomplete)")
                return await self._tier3_final_answer_workflow()
            
            elif status in ["phase2_format", "format_selecting"]:
                # Assessment complete but format not selected - resume from format selection
                logger.info("Tier 3 resume: Starting from format selection")
                return await self._resume_tier3_from_format_selection(tier3_state)
            
            elif status in ["phase3a_short_form", "selecting_references", "writing"]:
                # Short form workflow was in progress
                if tier3_state.answer_format == "short_form":
                    logger.info("Tier 3 resume: Resuming short form workflow")
                    return await self._resume_tier3_short_form(tier3_state)
                else:
                    # Status mismatch - restart Tier 3
                    logger.warning("Tier 3 resume: Status/format mismatch, restarting")
                    return await self._tier3_final_answer_workflow()
            
            elif status in ["phase3b_long_form", "organizing_volume"]:
                # Long form workflow was in progress
                if tier3_state.answer_format == "long_form":
                    logger.info("Tier 3 resume: Resuming long form workflow")
                    return await self._resume_tier3_long_form(tier3_state)
                else:
                    # Status mismatch - restart Tier 3
                    logger.warning("Tier 3 resume: Status/format mismatch, restarting")
                    return await self._tier3_final_answer_workflow()
            
            elif status == "complete":
                # Already complete - just return True
                logger.info("Tier 3 resume: Already complete")
                return True
            
            else:
                # Unknown status - start fresh
                logger.warning(f"Tier 3 resume: Unknown status '{status}', starting fresh")
                return await self._tier3_final_answer_workflow()
                
        except Exception as e:
            logger.error(f"Tier 3 resume error: {e}")
            await final_answer_memory.set_active(False)
            return False
        
        finally:
            # Always clear model tracking callback when Tier 3 ends
            api_client_manager.set_model_tracking_callback(None)
            self._tier3_active = False
    
    async def _resume_tier3_from_format_selection(self, tier3_state) -> bool:
        """
        Resume Tier 3 from format selection phase.
        Certainty assessment is already saved. If format was already selected and saved,
        use it; otherwise re-select format and proceed.
        """
        try:
            # Get saved certainty assessment with validation
            assessment = tier3_state.certainty_assessment
            if assessment is None:
                logger.error("Tier 3 resume: No certainty assessment found, restarting")
                return await self._tier3_final_answer_workflow()
            
            # Validate assessment has required fields
            if not hasattr(assessment, 'certainty_level') or not assessment.certainty_level:
                logger.error("Tier 3 resume: Certainty assessment incomplete (missing certainty_level), restarting")
                return await self._tier3_final_answer_workflow()
            
            # Get all papers for answer construction
            all_papers = await autonomous_rag_manager.get_all_papers_summary()
            if not all_papers:
                logger.error("Tier 3 resume: No papers available")
                await final_answer_memory.set_active(False)
                return False
            
            # Check if format was already selected and saved
            saved_format = tier3_state.answer_format
            if saved_format:
                # Format already selected - use saved format directly
                logger.info(f"Tier 3 resume: Using saved format '{saved_format}' (skipping format selection)")
                
                await self._broadcast("tier3_format_selected", {
                    "format": saved_format,
                    "reasoning": "[Resumed from saved state]"
                })
                
                # Proceed directly to answer construction
                if saved_format == "short_form":
                    success = await self._tier3_short_form_workflow(assessment, all_papers)
                else:
                    success = await self._tier3_long_form_workflow(assessment, all_papers)
            else:
                # Format not yet selected - need to select it
                await final_answer_memory.set_status("phase2_format")
                
                await self._broadcast("tier3_phase_changed", {
                    "phase": "format_selection",
                    "description": "Resuming format selection"
                })
                
                format_selection = await self._format_selector.select_format(
                    user_research_prompt=self._user_research_prompt,
                    certainty_assessment=assessment,
                    all_papers=all_papers
                )
                
                if format_selection is None:
                    logger.error("Tier 3 resume: Format selection failed")
                    await final_answer_memory.set_active(False)
                    return False
                
                await self._broadcast("tier3_format_selected", {
                    "format": format_selection.answer_format,
                    "reasoning": format_selection.reasoning
                })
                
                # Save workflow state after format selection
                await self._save_workflow_state(tier="tier3_final_answer")
                
                # Proceed to answer construction
                if format_selection.answer_format == "short_form":
                    success = await self._tier3_short_form_workflow(assessment, all_papers)
                else:
                    success = await self._tier3_long_form_workflow(assessment, all_papers)
            
            if success:
                await final_answer_memory.set_status("complete")
                return True
            else:
                await final_answer_memory.set_active(False)
                return False
                
        except Exception as e:
            logger.error(f"Tier 3 resume from format selection error: {e}")
            await final_answer_memory.set_active(False)
            return False
    
    async def _resume_tier3_short_form(self, tier3_state) -> bool:
        """
        Resume Tier 3 short form workflow.
        
        Since resuming mid-compilation is complex and error-prone (the compiler
        auto-detects existing paper but Tier 3 paper compilation clears the paper),
        we restart the short form workflow from the beginning. The certainty
        assessment is preserved, so we only need to redo reference selection,
        title selection, and paper compilation.
        """
        try:
            # Get saved state with validation
            assessment = tier3_state.certainty_assessment
            
            if assessment is None:
                logger.error("Tier 3 resume: No certainty assessment, restarting Tier 3 entirely")
                return await self._tier3_final_answer_workflow()
            
            # Validate assessment has required fields
            if not hasattr(assessment, 'certainty_level') or not assessment.certainty_level:
                logger.error("Tier 3 resume: Certainty assessment incomplete (missing certainty_level), restarting")
                return await self._tier3_final_answer_workflow()
            
            # Validate format is short_form (sanity check)
            if tier3_state.answer_format and tier3_state.answer_format != "short_form":
                logger.error(f"Tier 3 resume: Format mismatch - expected short_form, got {tier3_state.answer_format}")
                return await self._tier3_final_answer_workflow()
            
            # Get all papers
            all_papers = await autonomous_rag_manager.get_all_papers_summary()
            if not all_papers:
                logger.error("Tier 3 resume: No papers available")
                await final_answer_memory.set_active(False)
                return False
            
            await self._broadcast("tier3_phase_changed", {
                "phase": "short_form_writing",
                "description": "Resuming short form workflow (restarting from reference selection)"
            })
            
            # Restart short form workflow - assessment is preserved so we just need to
            # redo reference selection, title, and paper compilation
            logger.info("Tier 3 resume: Restarting short form workflow with preserved assessment")
            return await self._tier3_short_form_workflow(assessment, all_papers)
            
        except Exception as e:
            logger.error(f"Tier 3 resume short form error: {e}")
            await final_answer_memory.set_active(False)
            return False
    
    async def _resume_tier3_long_form(self, tier3_state) -> bool:
        """
        Resume Tier 3 long form workflow.
        Resume from the last completed chapter.
        """
        try:
            # Get saved state with validation
            assessment = tier3_state.certainty_assessment
            volume_org = tier3_state.volume_organization
            completed_chapters = tier3_state.completed_chapters or []
            current_chapter = tier3_state.current_writing_chapter
            
            if assessment is None:
                logger.error("Tier 3 resume: No certainty assessment, restarting")
                return await self._tier3_final_answer_workflow()
            
            # Validate assessment has required fields
            if not hasattr(assessment, 'certainty_level') or not assessment.certainty_level:
                logger.error("Tier 3 resume: Certainty assessment incomplete (missing certainty_level), restarting")
                return await self._tier3_final_answer_workflow()
            
            # Validate format is long_form (sanity check)
            if tier3_state.answer_format and tier3_state.answer_format != "long_form":
                logger.error(f"Tier 3 resume: Format mismatch - expected long_form, got {tier3_state.answer_format}")
                return await self._tier3_final_answer_workflow()
            
            # Get all papers
            all_papers = await autonomous_rag_manager.get_all_papers_summary()
            if not all_papers:
                logger.error("Tier 3 resume: No papers available")
                await final_answer_memory.set_active(False)
                return False
            
            # Check if volume was organized
            if volume_org is None:
                # Volume not organized - restart long form from beginning
                logger.info("Tier 3 resume: Volume not organized, restarting long form workflow")
                return await self._tier3_long_form_workflow(assessment, all_papers)
            
            # Validate volume organization has required fields
            if not hasattr(volume_org, 'volume_title') or not volume_org.volume_title:
                logger.error("Tier 3 resume: Volume organization incomplete (missing title), restarting long form")
                return await self._tier3_long_form_workflow(assessment, all_papers)
            
            if not hasattr(volume_org, 'chapters') or not volume_org.chapters:
                logger.error("Tier 3 resume: Volume organization incomplete (no chapters), restarting long form")
                return await self._tier3_long_form_workflow(assessment, all_papers)
            
            await final_answer_memory.set_status("phase3b_long_form")
            
            await self._broadcast("tier3_phase_changed", {
                "phase": "long_form_organization",
                "description": f"Resuming long form volume (chapters completed: {len(completed_chapters)})"
            })
            
            logger.info(f"Tier 3 resume: Volume '{volume_org.volume_title}' with "
                       f"{len(completed_chapters)} chapters already complete")
            
            # Get chapters that still need to be written
            chapters_to_write = self._volume_organizer.get_writing_order(volume_org)
            remaining_chapters = [
                ch for ch in chapters_to_write 
                if ch.order not in completed_chapters
            ]
            
            if not remaining_chapters:
                # All chapters complete - just assemble the volume
                logger.info("Tier 3 resume: All chapters complete, assembling volume")
                final_volume = await final_answer_memory.assemble_final_volume()
                
                await self._broadcast("tier3_long_form_complete", {
                    "title": volume_org.volume_title,
                    "total_chapters": len(volume_org.chapters),
                    "resumed": True
                })
                
                await self._broadcast("tier3_complete", {
                    "format": "long_form",
                    "title": volume_org.volume_title
                })
                
                logger.info(f"Tier 3 LONG FORM RESUMED AND COMPLETE: {volume_org.volume_title}")
                return True
            
            logger.info(f"Tier 3 resume: {len(remaining_chapters)} chapters remaining")
            
            # Continue writing remaining chapters
            for chapter in remaining_chapters:
                if self._stop_event.is_set():
                    break
                
                await final_answer_memory.set_current_writing_chapter(chapter.order)
                
                await self._broadcast("tier3_chapter_started", {
                    "chapter_order": chapter.order,
                    "chapter_type": chapter.chapter_type,
                    "title": chapter.title,
                    "resumed": True
                })
                
                success = await self._write_volume_chapter(chapter, volume_org, assessment)
                
                if not success:
                    logger.error(f"Tier 3 resume: Failed to write chapter {chapter.order}: {chapter.title}")
                    return False
                
                await final_answer_memory.update_chapter_status(chapter.order, "complete")
                
                # Save workflow state after each chapter
                await self._save_workflow_state(tier="tier3_final_answer")
                
                await self._broadcast("tier3_chapter_complete", {
                    "chapter_order": chapter.order,
                    "title": chapter.title
                })
            
            # Assemble final volume
            final_volume = await final_answer_memory.assemble_final_volume()
            
            await self._broadcast("tier3_long_form_complete", {
                "title": volume_org.volume_title,
                "total_chapters": len(volume_org.chapters)
            })
            
            await self._broadcast("tier3_complete", {
                "format": "long_form",
                "title": volume_org.volume_title
            })
            
            logger.info(f"Tier 3 LONG FORM RESUMED AND COMPLETE: {volume_org.volume_title}")
            return True
            
        except Exception as e:
            logger.error(f"Tier 3 resume long form error: {e}")
            await final_answer_memory.set_active(False)
            return False
    
    async def _tier3_final_answer_workflow(self) -> bool:
        """
        Complete Tier 3 final answer generation workflow.
        
        Returns:
            True if final answer was successfully generated and system should stop,
            False if more research is needed (no_answer_known) and should continue.
        """
        logger.info("=" * 60)
        logger.info("TIER 3: FINAL ANSWER GENERATION STARTED")
        logger.info("=" * 60)
        
        # Set current tier to Tier 3 for UI/state
        self._state.current_tier = "tier3_final_answer"
        
        # Set phase for API logging
        api_client_manager.set_autonomous_phase("tier3")
        
        # CRITICAL: Mark Tier 3 as active to disable paper redundancy checks
        # This prevents purging papers that are being used in the final volume
        self._tier3_active = True
        
        # Update tracking - use actual library count
        paper_counts = await paper_library.count_papers()
        self._last_tier3_check_at = paper_counts["active"]
        
        # Initialize Tier 3 memory for this session
        await final_answer_memory.set_active(True)
        await final_answer_memory.set_status("phase1_assessment")
        
        # Initialize model usage tracking for Tier 3
        # This tracks all models used and their API call counts for author attribution
        await final_answer_memory.initialize_model_tracking(self._user_research_prompt)
        
        # Set up model tracking callback - will be called after each API call
        # Track to BOTH global Tier 3 tracker AND per-paper tracker (if active)
        async def tier3_model_tracking_callback(model_id: str) -> None:
            # Always track to global Tier 3 tracker
            await final_answer_memory.track_model_call(model_id)
            # Also track to per-paper tracker if one is active (for gap/intro/conclusion papers)
            if self._current_paper_tracker:
                self._current_paper_tracker.track_call(model_id)
        
        api_client_manager.set_model_tracking_callback(tier3_model_tracking_callback)
        logger.info("Tier 3: Model tracking enabled (global + per-paper)")
        
        await self._broadcast("tier3_started", {
            "papers_count": self._papers_completed_count
        })
        
        try:
            # Get all papers for assessment
            all_papers = await autonomous_rag_manager.get_all_papers_summary()
            
            if not all_papers:
                logger.warning("Tier 3: No papers available for assessment")
                await final_answer_memory.set_active(False)
                return False
            
            # ============================================================
            # PHASE 1: Certainty Assessment
            # ============================================================
            logger.info("Tier 3 Phase 1: Assessing certainty")
            await final_answer_memory.set_status("phase1_assessment")
            
            await self._broadcast("tier3_phase_changed", {
                "phase": "assessment",
                "description": "Assessing what can be answered with certainty"
            })
            
            assessment = await self._certainty_assessor.assess_certainty(
                user_research_prompt=self._user_research_prompt,
                all_papers=all_papers
            )
            
            if assessment is None:
                logger.error("Tier 3: Certainty assessment failed")
                await final_answer_memory.set_active(False)
                return False
            
            # Check if we should continue research
            if assessment.certainty_level == "no_answer_known":
                logger.info("Tier 3: No answer known yet - continuing research")
                await self._broadcast("tier3_result", {
                    "result": "continue_research",
                    "certainty_level": assessment.certainty_level,
                    "reasoning": assessment.reasoning
                })
                # Return to normal workflow
                self._state.current_tier = "tier1_aggregation"
                await final_answer_memory.set_active(False)
                return False
            
            logger.info(f"Tier 3 certainty level: {assessment.certainty_level}")
            
            # Save workflow state after Phase 1 completion for crash recovery
            await self._save_workflow_state(tier="tier3_final_answer")
            
            # ============================================================
            # PHASE 2: Format Selection
            # ============================================================
            logger.info("Tier 3 Phase 2: Selecting answer format")
            await final_answer_memory.set_status("phase2_format")
            
            await self._broadcast("tier3_phase_changed", {
                "phase": "format_selection",
                "description": "Selecting short form (paper) or long form (volume)"
            })
            
            format_selection = await self._format_selector.select_format(
                user_research_prompt=self._user_research_prompt,
                certainty_assessment=assessment,
                all_papers=all_papers
            )
            
            if format_selection is None:
                logger.error("Tier 3: Format selection failed")
                await final_answer_memory.set_active(False)
                return False
            
            logger.info(f"Tier 3 format selected: {format_selection.answer_format}")
            
            await self._broadcast("tier3_format_selected", {
                "format": format_selection.answer_format,
                "reasoning": format_selection.reasoning
            })
            
            # Save workflow state after Phase 2 completion for crash recovery
            await self._save_workflow_state(tier="tier3_final_answer")
            
            # ============================================================
            # PHASE 3: Answer Construction
            # ============================================================
            if format_selection.answer_format == "short_form":
                # Phase 3A: Short Form - Single Paper
                success = await self._tier3_short_form_workflow(assessment, all_papers)
            else:
                # Phase 3B: Long Form - Volume
                success = await self._tier3_long_form_workflow(assessment, all_papers)
            
            if success:
                await final_answer_memory.set_status("complete")
                # DO NOT set is_active=False - keep state as "complete" so it persists for API access
                # The frontend needs to see the complete state to display the final answer
                return True
            else:
                await final_answer_memory.set_active(False)
                return False
            
        except Exception as e:
            logger.error(f"Tier 3 workflow error: {e}")
            await final_answer_memory.set_active(False)
            return False
        
        finally:
            # Always clear model tracking callback when Tier 3 ends
            api_client_manager.set_model_tracking_callback(None)
            logger.info("Tier 3: Model tracking disabled")
            
            # Always reset Tier 3 active flag to re-enable redundancy checks
            # (though if Tier 3 succeeded, the system stops anyway)
            self._tier3_active = False
    
    async def _tier3_short_form_workflow(
        self,
        assessment,
        all_papers: List[Dict[str, Any]]
    ) -> bool:
        """
        Tier 3 Short Form workflow - write a single paper answering the user's question.
        """
        logger.info("Tier 3 Phase 3A: Short Form - Writing final answer paper")
        await final_answer_memory.set_status("phase3a_short_form")
        
        await self._broadcast("tier3_phase_changed", {
            "phase": "short_form_writing",
            "description": "Writing final answer as single paper"
        })
        
        try:
            # Step 1: Select reference papers for the final answer
            # Uses the same reference selection as Tier 2, but with ALL papers available
            reference_papers = await self._tier3_reference_selection(all_papers)
            await final_answer_memory.set_short_form_references(reference_papers)
            
            logger.info(f"Tier 3: Selected {len(reference_papers)} reference papers for final answer")
            
            # Save workflow state after reference selection for crash recovery
            await self._save_workflow_state(tier="tier3_final_answer")
            
            # Step 2: Select paper title that directly answers the question
            paper_title = await self._tier3_title_selection(assessment, reference_papers)
            
            if not paper_title:
                logger.error("Tier 3: Failed to select final paper title")
                return False
            
            # Generate paper ID (use special prefix for final answer)
            paper_id = f"FINAL_ANSWER_{await research_metadata.generate_paper_id()}"
            await final_answer_memory.set_short_form_paper_id(paper_id)
            
            logger.info(f"Tier 3: Writing final answer paper: {paper_title}")
            
            # Save workflow state after title selection for crash recovery
            await self._save_workflow_state(tier="tier3_final_answer")
            
            await self._broadcast("tier3_paper_started", {
                "paper_id": paper_id,
                "title": paper_title,
                "is_final_answer": True
            })
            
            # Step 3: Compile the paper using Part 2 infrastructure
            # CRITICAL: We pass reference papers, NOT brainstorm database
            # Tier 3 operates ONLY on Tier 2 papers
            paper_content = await self._compile_tier3_paper(
                paper_id=paper_id,
                paper_title=paper_title,
                reference_paper_ids=reference_papers,
                assessment=assessment
            )
            
            if paper_content is None:
                logger.error("Tier 3: Failed to compile final answer paper")
                return False
            
            # Step 4: Assemble the final paper with author attribution and model credits
            assembled_paper = await final_answer_memory.assemble_short_form_paper(
                paper_content=paper_content,
                paper_title=paper_title
            )
            
            # Step 5: Save the final answer paper
            final_outline = await outline_memory.get_outline()
            
            await self._handle_paper_completion(
                paper_id=paper_id,
                title=paper_title,
                content=assembled_paper,
                outline=final_outline or "[Outline not available]",
                reference_paper_ids=reference_papers
            )
            
            await self._broadcast("tier3_short_form_complete", {
                "paper_id": paper_id,
                "title": paper_title
            })
            
            # Notify frontend that Tier 3 is complete
            await self._broadcast("tier3_complete", {
                "format": "short_form",
                "title": paper_title
            })
            
            logger.info(f"Tier 3 SHORT FORM COMPLETE: {paper_title}")
            return True
            
        except Exception as e:
            logger.error(f"Tier 3 short form error: {e}")
            return False
    
    async def _tier3_long_form_workflow(
        self,
        assessment,
        all_papers: List[Dict[str, Any]]
    ) -> bool:
        """
        Tier 3 Long Form workflow - create a volume collection of papers.
        """
        logger.info("Tier 3 Phase 3B: Long Form - Creating volume collection")
        await final_answer_memory.set_status("phase3b_long_form")
        
        await self._broadcast("tier3_phase_changed", {
            "phase": "long_form_organization",
            "description": "Organizing volume structure"
        })
        
        try:
            # Step 1: Organize volume structure
            volume = await self._volume_organizer.organize_volume(
                user_research_prompt=self._user_research_prompt,
                certainty_assessment=assessment,
                all_papers=all_papers
            )
            
            if volume is None:
                logger.error("Tier 3: Failed to organize volume")
                return False
            
            logger.info(f"Tier 3: Volume organized: {volume.volume_title} ({len(volume.chapters)} chapters)")
            
            await self._broadcast("tier3_volume_organized", {
                "title": volume.volume_title,
                "chapters": [ch.model_dump() for ch in volume.chapters]
            })
            
            # Save workflow state after volume organization for crash recovery
            await self._save_workflow_state(tier="tier3_final_answer")
            
            # Step 2: Write chapters in order
            # Order: Gap papers -> Conclusion -> Introduction
            chapters_to_write = self._volume_organizer.get_writing_order(volume)
            
            for chapter in chapters_to_write:
                if self._stop_event.is_set():
                    break
                
                await final_answer_memory.set_current_writing_chapter(chapter.order)
                
                await self._broadcast("tier3_chapter_started", {
                    "chapter_order": chapter.order,
                    "chapter_type": chapter.chapter_type,
                    "title": chapter.title
                })
                
                success = await self._write_volume_chapter(chapter, volume, assessment)
                
                if not success:
                    logger.error(f"Tier 3: Failed to write chapter {chapter.order}: {chapter.title}")
                    return False
                
                await final_answer_memory.update_chapter_status(chapter.order, "complete")
                
                # Save workflow state after each chapter completion for crash recovery
                await self._save_workflow_state(tier="tier3_final_answer")
                
                await self._broadcast("tier3_chapter_complete", {
                    "chapter_order": chapter.order,
                    "title": chapter.title
                })
            
            # Step 3: Assemble final volume
            final_volume = await final_answer_memory.assemble_final_volume()
            
            await self._broadcast("tier3_long_form_complete", {
                "title": volume.volume_title,
                "total_chapters": len(volume.chapters)
            })
            
            # Notify frontend that Tier 3 is complete
            await self._broadcast("tier3_complete", {
                "format": "long_form",
                "title": volume.volume_title
            })
            
            logger.info(f"Tier 3 LONG FORM COMPLETE: {volume.volume_title}")
            return True
            
        except Exception as e:
            logger.error(f"Tier 3 long form error: {e}")
            return False
    
    async def _tier3_reference_selection(
        self,
        all_papers: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Select reference papers for Tier 3 final answer.
        Directly selects papers without brainstorm context.
        """
        # For Tier 3, we browse ALL papers and select those most useful for answering
        selected_ids = await self._reference_selector.select_references(
            user_research_prompt=self._user_research_prompt,
            topic_prompt="[Tier 3 Final Answer - selecting papers to answer the research question]",
            brainstorm_summary="[No brainstorm - Tier 3 operates on completed papers only]",
            available_papers=all_papers,
            mode="initial",  # Fresh selection for Tier 3
            already_selected=[]
        )
        
        return selected_ids
    
    async def _tier3_title_selection(
        self,
        assessment,
        reference_papers: List[str]
    ) -> Optional[str]:
        """
        Select a title for the Tier 3 final answer paper.
        The title should directly and transparently answer the user's question.
        """
        from backend.autonomous.prompts.final_answer_prompts import build_final_paper_title_prompt
        
        # Get reference paper details
        reference_details = []
        for paper_id in reference_papers:
            content = await paper_library.get_paper_content(paper_id)
            metadata = await paper_library.get_metadata(paper_id)
            if metadata:
                reference_details.append({
                    "paper_id": paper_id,
                    "title": metadata.title,
                    "abstract": metadata.abstract
                })
        
        # Use the existing title selector with special context
        title = await self._title_selector.select_title(
            user_research_prompt=self._user_research_prompt,
            topic_prompt=f"[TIER 3 FINAL ANSWER] Certainty: {assessment.certainty_level}",
            brainstorm_summary=f"Known Certainties:\n{assessment.known_certainties_summary}",
            existing_papers_from_brainstorm=[]  # No previous papers for this "brainstorm"
        )
        
        return title
    
    async def _compile_tier3_paper(
        self,
        paper_id: str,
        paper_title: str,
        reference_paper_ids: List[str],
        assessment
    ) -> Optional[str]:
        """
        Compile Tier 3 final answer paper.
        CRITICAL: Uses ONLY reference papers, NOT brainstorm databases.
        """
        logger.info(f"Compiling Tier 3 paper: {paper_title}")
        
        # Initialize compiler for this paper
        self._paper_compiler = CompilerCoordinator()
        
        try:
            # Clear RAG for fresh Tier 3 compilation
            logger.info("Clearing RAG for Tier 3 paper compilation...")
            await asyncio.to_thread(rag_manager.clear_all_documents)
            
            # Initialize compiler
            await self._paper_compiler.initialize(
                compiler_prompt=f"Write a mathematical research paper titled: {paper_title}\n\n"
                               f"IMPORTANT: This paper directly answers the research question.\n"
                               f"Known Certainties: {assessment.known_certainties_summary}",
                validator_model=self._validator_model,
                high_context_model=self._high_context_model,
                high_param_model=self._high_param_model,
                critique_submitter_model=self._critique_submitter_model,
                skip_aggregator_db=True,  # CRITICAL: Don't load any aggregator database
                # Pass OpenRouter provider configs for all compiler roles
                validator_provider=self._validator_provider,
                validator_openrouter_provider=self._validator_openrouter_provider,
                validator_lm_studio_fallback=self._validator_lm_studio_fallback,
                high_context_provider=self._high_context_provider,
                high_context_openrouter_provider=self._high_context_openrouter_provider,
                high_context_lm_studio_fallback=self._high_context_lm_studio_fallback,
                high_param_provider=self._high_param_provider,
                high_param_openrouter_provider=self._high_param_openrouter_provider,
                high_param_lm_studio_fallback=self._high_param_lm_studio_fallback,
                critique_submitter_provider=self._critique_submitter_provider,
                critique_submitter_openrouter_provider=self._critique_submitter_openrouter_provider,
                critique_submitter_lm_studio_fallback=self._critique_submitter_lm_studio_fallback
            )
            
            # Set WebSocket broadcaster
            if self._broadcast_callback:
                self._paper_compiler.websocket_broadcaster = self._broadcast_callback
            
            # Enable autonomous mode
            self._paper_compiler.enable_autonomous_mode()
            
            # Clear any previous paper/outline
            await self._paper_compiler.clear_paper()
            
            # Load reference papers ONLY (no brainstorm database for Tier 3)
            if reference_paper_ids:
                logger.info(f"Loading {len(reference_paper_ids)} reference papers for Tier 3 compilation")
                for ref_paper_id in reference_paper_ids:
                    # IMPORTANT: Use paper_library.get_paper_path() for session-aware path resolution
                    paper_path = paper_library.get_paper_path(ref_paper_id)
                    if os.path.exists(paper_path):
                        await rag_manager.add_document(
                            paper_path,
                            chunk_sizes=[512],
                            is_user_file=True  # High priority
                        )
                        logger.info(f"Tier 3 reference loaded: {ref_paper_id}")
            
            # Start compiler
            await self._paper_compiler.start()
            logger.info(f"Tier 3 compiler started for {paper_id}")
            
            # Monitor for completion (same as Tier 2)
            abstract_written = False
            while self._running and not self._stop_event.is_set():
                current_paper = await compiler_paper_memory.get_paper()
                
                if current_paper and self._has_abstract(current_paper):
                    abstract_written = True
                    break
                
                if not self._paper_compiler.is_running:
                    break
                
                await asyncio.sleep(3)
            
            await self._paper_compiler.stop()
            
            if not abstract_written:
                logger.error("Tier 3 paper compilation did not complete")
                return None
            
            return await compiler_paper_memory.get_paper()
            
        except Exception as e:
            logger.error(f"Tier 3 paper compilation error: {e}")
            if self._paper_compiler:
                await self._paper_compiler.stop()
            return None
    
    async def _write_volume_chapter(
        self,
        chapter,
        volume,
        assessment
    ) -> bool:
        """
        Write a single chapter for the long form volume.
        
        Gap papers, introduction, and conclusion are written using the compiler.
        Existing papers are already complete (just linked).
        """
        if chapter.chapter_type == "existing_paper":
            # Existing paper - already written, just mark complete
            logger.info(f"Chapter {chapter.order} uses existing paper {chapter.paper_id}")
            return True
        
        logger.info(f"Writing chapter {chapter.order}: {chapter.title} ({chapter.chapter_type})")
        
        # Determine context based on chapter type
        if chapter.chapter_type == "introduction":
            # Introduction is written LAST - has access to all chapters
            context = "Write the INTRODUCTION for this volume. You have access to ALL chapters."
        elif chapter.chapter_type == "conclusion":
            # Conclusion is written second-to-last
            context = "Write the CONCLUSION for this volume. Synthesize findings from all body chapters."
        else:
            # Gap paper - fills content gap
            context = f"Write a paper to fill this content gap: {chapter.description}"
        
        # Get reference papers (existing papers in the volume)
        reference_ids = [
            ch.paper_id for ch in volume.chapters 
            if ch.chapter_type == "existing_paper" and ch.paper_id
        ]
        
        # Compile the chapter paper
        chapter_paper_id = f"volume_ch{chapter.order:02d}_{chapter.chapter_type}"
        
        paper_content = await self._compile_tier3_paper(
            paper_id=chapter_paper_id,
            paper_title=chapter.title,
            reference_paper_ids=reference_ids,
            assessment=assessment
        )
        
        if paper_content:
            # Save chapter content
            chapter_outline = await outline_memory.get_outline()
            await final_answer_memory.save_chapter_paper(
                chapter_order=chapter.order,
                content=paper_content,
                outline=chapter_outline or ""
            )
            return True
        
        return False
    
    # ========================================================================
    # CLEAR DATA
    # ========================================================================
    
    async def clear_all_data(self) -> None:
        """Clear all autonomous research data.
        
        Clears brainstorms, papers, metadata, RAG state, and session data.
        Uses graceful degradation: distinguishes critical vs non-critical failures.
        """
        # Check both internal flag and state object
        if self._running or self._state.is_running:
            raise RuntimeError("Cannot clear data while running")
        
        import shutil
        import time
        from pathlib import Path
        
        # Wait briefly for any pending async file operations to complete
        await asyncio.sleep(0.3)
        
        errors = []
        critical_errors = []  # Track critical errors separately
        successes = []  # Track successful operations
        
        def safe_rmtree(path: Path, max_retries: int = 5) -> bool:
            """Safely remove directory tree with retries for Windows file locking."""
            for attempt in range(max_retries):
                try:
                    if path.exists():
                        shutil.rmtree(path)
                    return True
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 0.5s, 1s, 2s, 4s
                        delay = 0.5 * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {path}: {e}")
                        time.sleep(delay)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error removing {path}: {type(e).__name__}: {e}")
                    raise
            return False
        
        # Step 0: Clear all session workflow states (prevents resume from old sessions)
        try:
            sessions_dir = Path(system_config.auto_sessions_base_dir)
            if sessions_dir.exists():
                for session_dir in sessions_dir.iterdir():
                    if session_dir.is_dir():
                        workflow_state_file = session_dir / "workflow_state.json"
                        if workflow_state_file.exists():
                            try:
                                workflow_state_file.unlink()
                                logger.info(f"Cleared workflow state from session: {session_dir.name}")
                            except Exception as e:
                                # Non-critical: workflow state files are small
                                logger.warning(f"Could not clear workflow state for {session_dir.name}: {e}")
            logger.info("Cleared all session workflow states")
        except Exception as e:
            errors.append(f"Failed to clear session workflow states: {e}")
            logger.error(errors[-1])
        
        # Step 1: Clear brainstorms directory
        try:
            brainstorms_dir = Path(system_config.auto_brainstorms_dir)
            safe_rmtree(brainstorms_dir)
            brainstorms_dir.mkdir(parents=True, exist_ok=True)
            successes.append("Cleared brainstorms directory")
            logger.info(f"Cleared brainstorms directory: {brainstorms_dir}")
        except Exception as e:
            critical_errors.append(f"Failed to clear brainstorms directory: {e}")
            logger.error(critical_errors[-1])
        
        # Step 2: Clear papers directory
        try:
            papers_dir = Path(system_config.auto_papers_dir)
            safe_rmtree(papers_dir)
            papers_dir.mkdir(parents=True, exist_ok=True)
            (papers_dir / "archive").mkdir(exist_ok=True)
            successes.append("Cleared papers directory")
            logger.info(f"Cleared papers directory: {papers_dir}")
        except Exception as e:
            critical_errors.append(f"Failed to clear papers directory: {e}")
            logger.error(critical_errors[-1])
        
        # Step 3: Clear metadata and stats files
        try:
            await research_metadata.clear_all()
            successes.append("Cleared research metadata and stats")
            logger.info("Cleared research metadata and stats")
        except Exception as e:
            # Non-critical: metadata can be regenerated
            errors.append(f"Failed to clear research metadata: {e}")
            logger.warning(errors[-1])
        
        # Step 4: Clear topic selection rejections file
        try:
            topic_rejections_path = Path(system_config.auto_research_topic_rejections_file)
            if topic_rejections_path.exists():
                topic_rejections_path.unlink()
            successes.append("Cleared topic rejections")
            logger.info(f"Cleared topic rejections file: {topic_rejections_path}")
        except Exception as e:
            # Non-critical: rejection logs are not essential data
            errors.append(f"Failed to clear topic rejections file: {e}")
            logger.warning(errors[-1])
        
        # Step 5: Clear autonomous rejection logs state (all logs)
        try:
            await autonomous_rejection_logs.clear_all()
            successes.append("Cleared autonomous rejection logs")
            logger.info("Cleared autonomous rejection logs")
        except Exception as e:
            # Non-critical: rejection logs can be regenerated
            errors.append(f"Failed to clear autonomous rejection logs: {e}")
            logger.warning(errors[-1])
        
        # Step 6: Clear RAG state (removes indexed brainstorm/paper content)
        try:
            # Wait a moment for any pending RAG operations to complete
            await asyncio.sleep(0.5)
            
            autonomous_rag_manager.reset()
            await asyncio.to_thread(rag_manager.clear_all_documents)
            successes.append("Cleared RAG state")
            logger.info("Cleared RAG state (ChromaDB collections)")
        except Exception as e:
            # Critical: RAG state affects future operations
            critical_errors.append(f"Failed to clear RAG state: {e}")
            logger.error(critical_errors[-1])
        
        # Step 7: Reset internal state
        self._current_topic_id = None
        self._current_paper_id = None
        self._current_paper_title = None
        self._current_reference_papers = []
        self._acceptance_count = 0
        self._rejection_count = 0
        self._cleanup_removals = 0
        self._consecutive_rejections = 0
        self._exhaustion_signals = 0
        self._papers_completed_count = 0
        self._last_redundancy_check_at = 0
        self._manual_paper_writing_triggered = False
        self._force_tier3_after_paper = False
        self._force_tier3_immediate = False
        self._tier3_active = False
        self._last_tier3_check_at = 0
        
        # Step 8: Reset state object
        self._state = AutonomousResearchState()
        
        # Step 9: Clear session manager state
        try:
            await session_manager.clear()
            successes.append("Cleared session manager state")
            logger.info("Cleared session manager state")
        except Exception as e:
            # Non-critical: session manager will reset on next start
            errors.append(f"Failed to clear session manager: {e}")
            logger.warning(errors[-1])
        
        # Report results with graceful degradation
        success_count = len(successes)
        error_count = len(errors)
        critical_count = len(critical_errors)
        
        if critical_count > 0:
            # Critical errors prevent full clear
            error_msg = "; ".join(critical_errors)
            logger.error(f"CRITICAL ERRORS during clear ({critical_count} critical, {error_count} non-critical): {error_msg}")
            raise RuntimeError(f"Failed to clear critical data: {error_msg}")
        
        if error_count > 0:
            # Non-critical errors: partial success
            error_msg = "; ".join(errors)
            logger.warning(f"Clear completed with {success_count} successes, {error_count} non-critical warnings: {error_msg}")
            # Don't raise - this is still a success
        
        logger.info(f"Autonomous research data cleared successfully ({success_count} operations completed, {error_count} non-critical warnings)")
    
    # ==================== WORKFLOW TRACKING METHODS ====================
    
    async def refresh_workflow_predictions(self) -> None:
        """Refresh workflow task predictions based on actual agent state."""
        try:
            from backend.shared.boost_manager import boost_manager
            
            tier = self._state.current_tier or "idle"
            tasks = []
            
            if tier == "tier1_aggregation" and self._brainstorm_aggregator:
                # Get tasks from the managed aggregator coordinator
                await self._brainstorm_aggregator.refresh_workflow_predictions()
                tasks = list(self._brainstorm_aggregator.workflow_tasks)
                
            elif tier == "tier2_paper_writing" and self._paper_compiler:
                # Get tasks from the managed compiler coordinator
                await self._paper_compiler.refresh_workflow_predictions()
                tasks = list(self._paper_compiler.workflow_tasks)
                
            else:
                # Topic selection/idle phase - use autonomous agents' sequences
                ts_seq = self._topic_selector.task_sequence if self._topic_selector else 0
                tv_seq = self._topic_validator.task_sequence if self._topic_validator else 0
                
                for i in range(20):
                    if i % 2 == 0:
                        task_id = f"auto_ts_{ts_seq:03d}"
                        role = "Topic Selector"
                        mode = "Topic Selection"
                        ts_seq += 1
                    else:
                        task_id = f"auto_tv_{tv_seq:03d}"
                        role = "Topic Validator"
                        mode = "Topic Validation"
                        tv_seq += 1
                    
                    tasks.append(WorkflowTask(
                        task_id=task_id,
                        sequence_number=i + 1,
                        role=role,
                        mode=mode,
                        provider="lm_studio",
                        using_boost=boost_manager.should_use_boost(task_id)
                    ))
            
            self.workflow_tasks = tasks
            
            # Broadcast update
            await self._broadcast("workflow_updated", {
                "tasks": [task.dict() for task in self.workflow_tasks],
                "mode": "autonomous"
            })
            
            logger.debug(f"Refreshed autonomous workflow predictions: {len(self.workflow_tasks)} tasks, tier={tier}")
            
        except Exception as e:
            logger.error(f"Failed to refresh workflow predictions: {e}")
            self.workflow_tasks = []
    
    def _is_single_model_mode(self) -> bool:
        """Check if all submitters and validator use the same model."""
        if not self._submitter_configs or not self._validator_model:
            return False
        
        # Check if all submitter models match the validator model
        all_models = [config.model_id for config in self._submitter_configs]
        all_models.append(self._validator_model)
        
        return len(set(all_models)) == 1
    
    async def get_next_task(self) -> Optional['WorkflowTask']:
        """Get the next uncompleted task from workflow queue."""
        for task in self.workflow_tasks:
            if not task.completed:
                return task
        return None
    
    async def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        self.completed_task_ids.add(task_id)
        
        # Update task in workflow_tasks
        for task in self.workflow_tasks:
            if task.task_id == task_id:
                task.completed = True
                break
        
        self.current_task_sequence += 1
        
        # Broadcast completion
        await self._broadcast("task_completed", {
            "task_id": task_id,
            "sequence": self.current_task_sequence
        })
        
        # Refresh predictions after each completion to keep workflow panel updated
        await self.refresh_workflow_predictions()
    
    async def mark_task_started(self, task_id: str) -> None:
        """Mark a task as actively running."""
        self.current_task_id = task_id
        
        # Update active status in workflow_tasks
        for task in self.workflow_tasks:
            task.active = (task.task_id == task_id)
        
        # Broadcast start
        await self._broadcast("task_started", {
            "task_id": task_id
        })
    
    def _handle_task_event(self, event_type: str, task_id: str) -> None:
        """
        Handle task events from agents (callback pattern).
        Called synchronously by agents; schedules async work on event loop.
        """
        import asyncio
        
        if event_type == "started":
            # Schedule async task start on event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.mark_task_started(task_id))
                else:
                    loop.run_until_complete(self.mark_task_started(task_id))
            except Exception as e:
                logger.debug(f"Could not mark task started: {e}")
        elif event_type == "completed":
            # Schedule async task completion on event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.mark_task_completed(task_id))
                else:
                    loop.run_until_complete(self.mark_task_completed(task_id))
            except Exception as e:
                logger.debug(f"Could not mark task completed: {e}")
    
    # ==================== END WORKFLOW TRACKING ====================
    
    async def _is_paper_saved(self, paper_id: str) -> bool:
        """Check if paper is already saved in library."""
        try:
            metadata = await paper_library.get_metadata(paper_id)
            return metadata is not None
        except:
            return False


# Global instance
autonomous_coordinator = AutonomousCoordinator()

