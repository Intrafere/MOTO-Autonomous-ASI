"""
Coordinator - orchestrates submitters, validator, queue, and RAG system.
Manages the overall aggregator workflow.
"""
import asyncio
import time
import json
from typing import List, Optional, Dict, Callable, Any
import logging
from pathlib import Path
from datetime import datetime
import aiofiles

from backend.shared.config import system_config, rag_config
from backend.shared.models import SystemStatus, Submission, ValidationResult, SubmitterConfig, WorkflowTask, ModelConfig
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.rag_lock import rag_operation_lock
from backend.shared.workflow_predictor import workflow_predictor
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.free_model_manager import free_model_manager
from backend.aggregator.agents.submitter import SubmitterAgent
from backend.aggregator.agents.validator import ValidatorAgent
from backend.aggregator.core.queue_manager import queue_manager
from backend.aggregator.core.rag_manager import rag_manager
from backend.aggregator.memory.shared_training import shared_training_memory
from backend.aggregator.memory.event_log import event_log

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinates the entire aggregator system.
    - Manages 1-10 submitter agents (parallel) - each can use different models
    - Manages 1 validator agent (sequential) - single Markov chain evolution
    - Handles queue and validation loop
    - Broadcasts events via WebSocket
    """
    
    def __init__(self):
        self.submitters: List[SubmitterAgent] = []
        self.validator: Optional[ValidatorAgent] = None
        self.is_running = False
        
        # Stats file path
        self.stats_file_path = Path(system_config.data_dir) / "aggregator_stats.json"
        
        # Stats
        self.total_submissions = 0
        self.total_acceptances = 0
        self.total_rejections = 0
        
        # Cleanup review stats
        self.cleanup_reviews_performed = 0
        self.removals_proposed = 0
        self.removals_executed = 0
        
        # WebSocket broadcaster (set by FastAPI app)
        self.websocket_broadcaster: Optional[Callable] = None
        
        # Tasks
        self._validator_task: Optional[asyncio.Task] = None
        self._main_task: Optional[asyncio.Task] = None  # For single-model mode
        self._rechunk_task: Optional[asyncio.Task] = None
        self._rechunk_callback_set = False
        
        # Chunk size cycling for incremental re-chunking
        self.current_rechunk_index = 0
        
        # Single-model mode detection
        self.single_model_mode = False
        self.submitter_configs: List[SubmitterConfig] = []
        self.validator_model = ""
        
        # Workflow tracking
        self.workflow_tasks: List[WorkflowTask] = []
        self.completed_task_ids: set = set()
        self.current_task_sequence: int = 0
        self.current_task_id: Optional[str] = None  # Currently executing task
        
        # Submitter pause control (queue overflow prevention)
        self.should_pause_submitters = False  # Flag to pause submitters when queue >= 10
        
        # Cleanup review toggle (disabled for short-lived mini-brainstorm phases)
        self.enable_cleanup_review = True
    
    async def _load_stats(self) -> None:
        """Load persisted stats from file."""
        if self.stats_file_path.exists():
            try:
                async with aiofiles.open(self.stats_file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content.strip():
                        stats = json.loads(content)
                        self.total_submissions = stats.get("total_submissions", 0)
                        self.total_acceptances = stats.get("total_acceptances", 0)
                        self.total_rejections = stats.get("total_rejections", 0)
                        self.cleanup_reviews_performed = stats.get("cleanup_reviews_performed", 0)
                        self.removals_proposed = stats.get("removals_proposed", 0)
                        self.removals_executed = stats.get("removals_executed", 0)
                        logger.info(f"Loaded persisted stats: {self.total_acceptances} acceptances, {self.total_rejections} rejections")
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")
    
    async def _save_stats(self) -> None:
        """Save current stats to file."""
        stats = {
            "total_submissions": self.total_submissions,
            "total_acceptances": self.total_acceptances,
            "total_rejections": self.total_rejections,
            "cleanup_reviews_performed": self.cleanup_reviews_performed,
            "removals_proposed": self.removals_proposed,
            "removals_executed": self.removals_executed
        }
        try:
            self.stats_file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.stats_file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(stats, indent=2))
            logger.debug("Saved stats to file")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    async def initialize(
        self,
        user_prompt: str,
        submitter_configs: List[SubmitterConfig],
        validator_model: str,
        user_files: List[str],
        skip_stats_load: bool = False,
        validator_context_window: Optional[int] = None,
        validator_max_tokens: Optional[int] = None,
        validator_provider: str = "lm_studio",
        validator_openrouter_provider: Optional[str] = None,
        validator_lm_studio_fallback: Optional[str] = None,
        enable_cleanup_review: bool = True
    ) -> None:
        """
        Initialize the coordinator with configuration.
        
        Args:
            user_prompt: User's prompt
            submitter_configs: Per-submitter configurations (1-10 submitters, each with own model/context)
            validator_model: Model for validator
            user_files: Paths to user-uploaded files
            skip_stats_load: If True, don't load stats from file (for autonomous mode)
            validator_context_window: Optional context window override for validator
            validator_max_tokens: Optional max output tokens override for validator
            validator_provider: Provider for validator ("lm_studio" or "openrouter")
            validator_openrouter_provider: OpenRouter host provider for validator (e.g., "Anthropic")
            validator_lm_studio_fallback: LM Studio fallback model for validator when using OpenRouter
        """
        logger.info("Initializing coordinator...")
        
        # Store cleanup review toggle
        self.enable_cleanup_review = enable_cleanup_review
        
        # Validate submitter count
        num_submitters = len(submitter_configs)
        if not (system_config.min_submitters <= num_submitters <= system_config.max_submitters):
            raise ValueError(
                f"Submitter count must be {system_config.min_submitters}-{system_config.max_submitters}, "
                f"got {num_submitters}"
            )
        
        # Store configurations
        self.submitter_configs = submitter_configs
        self.validator_model = validator_model
        self.validator_provider = validator_provider
        
        # Override validator context window if provided
        if validator_context_window is not None:
            rag_config.validator_context_window = validator_context_window
        if validator_max_tokens is not None:
            rag_config.validator_max_output_tokens = validator_max_tokens
        
        # Use the first submitter's context window for the shared RAG config (for compatibility)
        # Each submitter will use its own context window when generating
        if submitter_configs:
            rag_config.submitter_context_window = submitter_configs[0].context_window
            rag_config.submitter_max_output_tokens = submitter_configs[0].max_output_tokens
        
        # CRITICAL: Also update context_allocator's instance variables
        # (it caches rag_config values at initialization and doesn't auto-update)
        from backend.aggregator.core.context_allocator import context_allocator
        final_submitter_context = submitter_configs[0].context_window if submitter_configs else rag_config.submitter_context_window
        final_validator_context = validator_context_window if validator_context_window is not None else rag_config.validator_context_window
        final_submitter_max_output = submitter_configs[0].max_output_tokens if submitter_configs else rag_config.submitter_max_output_tokens
        final_validator_max_output = validator_max_tokens if validator_max_tokens is not None else rag_config.validator_max_output_tokens
        context_allocator.set_context_windows(final_submitter_context, final_validator_context, final_submitter_max_output, final_validator_max_output)
        
        # CRITICAL: Detect single-model mode ONLY based on configured model IDs
        # Boost routing is INDEPENDENT of this decision and does NOT affect concurrency
        # Single-model mode prevents queue overflow when all agents share the same LM Studio server
        # Boost can route calls to OpenRouter even in single-model mode (if enabled)
        all_models = [sc.model_id for sc in submitter_configs] + [validator_model]
        unique_models = set(all_models)
        self.single_model_mode = len(unique_models) == 1
        
        if self.single_model_mode:
            logger.info(
                f"Single-model mode ENABLED: All {num_submitters} submitters and validator use '{validator_model}'. "
                f"Submitters will run sequentially then validator processes all."
            )
        else:
            logger.info(
                f"Multi-model mode: {num_submitters} submitters with models "
                f"{[sc.model_id for sc in submitter_configs]} run in parallel, "
                f"validator ({validator_model}) runs independently."
            )
        
        # Log boost status if enabled (for transparency)
        from backend.shared.boost_manager import boost_manager
        if boost_manager.boost_config and boost_manager.boost_config.enabled:
            logger.info(
                f"Boost mode ACTIVE: Will route selected tasks to {boost_manager.boost_config.boost_model_id}. "
                f"This does NOT affect parallel execution mode."
            )
        
        # Log currently loaded models for diagnostics
        loaded_models = await lm_studio_client.get_loaded_models()
        logger.info(f"Currently loaded models: {loaded_models}")
        
        # CRITICAL: Warn user about potential context mismatches
        # LM Studio may not load models with requested context - this causes silent failures
        context_info = "\n".join([
            f"  - Submitter {sc.submitter_id}: {sc.context_window} tokens (model: {sc.model_id})"
            for sc in submitter_configs
        ])
        logger.info(
            f"Context window configuration:\n"
            f"{context_info}\n"
            f"  - Validator: {final_validator_context} tokens (model: {validator_model})"
        )
        
        # Initialize shared training memory
        await shared_training_memory.initialize()
        
        # Load persisted stats and event log (unless skipped for autonomous mode)
        if not skip_stats_load:
            await self._load_stats()
            
            # CRITICAL: Clear RAG for manual mode to prevent cross-contamination
            # from autonomous brainstorm content that may have been loaded in a prior session
            logger.info("Clearing RAG for fresh Part 1 aggregator session...")
            await asyncio.to_thread(rag_manager.clear_all_documents)
            logger.info("RAG cleared successfully for Part 1 aggregator")
        else:
            logger.info("Skipping stats load (autonomous mode - starting fresh)")
            # Reset stats to 0 for autonomous brainstorm
            self.total_submissions = 0
            self.total_acceptances = 0
            self.total_rejections = 0
            self.cleanup_reviews_performed = 0
            self.removals_proposed = 0
            self.removals_executed = 0
            # NOTE: For autonomous mode, RAG cleanup is handled by AutonomousCoordinator
            # to ensure proper isolation of brainstorm databases
        await event_log.initialize()
        
        # Load user files into RAG system
        user_files_content = {}
        for file_path in user_files:
            path = Path(file_path)
            if path.exists():
                # Add to RAG system with all 4 chunk configs
                await rag_manager.add_document(
                    file_path,
                    chunk_sizes=rag_config.submitter_chunk_intervals,
                    is_user_file=True
                )
                # Also load content for potential direct injection (async to avoid blocking)
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    user_files_content[path.name] = await f.read()
                logger.info(f"Loaded user file: {path.name}")
        
        # Create submitter agents from configs (1-10 submitters with individual settings)
        self.submitters = []
        for config in submitter_configs:
            submitter = SubmitterAgent(
                submitter_id=config.submitter_id,
                model_name=config.model_id,
                user_prompt=user_prompt,
                user_files_content=user_files_content,
                websocket_broadcaster=self.websocket_broadcaster,
                context_window=config.context_window,
                max_output_tokens=config.max_output_tokens,
                coordinator=self
            )
            await submitter.initialize()
            # Set callback to add submissions to queue
            submitter.set_submission_callback(self.add_submission_to_queue)
            # Set task tracking callback for workflow panel integration
            submitter.set_task_tracking_callback(self._handle_task_event)
            self.submitters.append(submitter)
            
            # Configure API client manager for this submitter (OpenRouter/LM Studio routing)
            api_client_manager.configure_role(
                role_id=f"aggregator_submitter_{config.submitter_id}",
                config=ModelConfig(
                    provider=config.provider,
                    model_id=config.model_id,
                    openrouter_provider=config.openrouter_provider,
                    lm_studio_fallback_id=config.lm_studio_fallback_id,
                    context_window=config.context_window,
                    max_output_tokens=config.max_output_tokens
                )
            )
            logger.info(f"Created Submitter {config.submitter_id}: model={config.model_id}, provider={config.provider}, context={config.context_window}")
        
        # Create validator agent
        self.validator = ValidatorAgent(
            model_name=validator_model,
            user_prompt=user_prompt,
            user_files_content=user_files_content,
            websocket_broadcaster=self.websocket_broadcaster
        )
        await self.validator.initialize()
        # Set task tracking callback for workflow panel integration
        self.validator.set_task_tracking_callback(self._handle_task_event)
        
        # Configure API client manager for validator (OpenRouter/LM Studio routing)
        api_client_manager.configure_role(
            role_id="aggregator_validator",
            config=ModelConfig(
                provider=validator_provider,
                model_id=validator_model,
                openrouter_provider=validator_openrouter_provider,
                lm_studio_fallback_id=validator_lm_studio_fallback,
                context_window=final_validator_context,
                max_output_tokens=final_validator_max_output
            )
        )
        logger.info(f"Created Validator: model={validator_model}, provider={validator_provider}")
        
        # Set up re-chunking callback
        if not self._rechunk_callback_set:
            shared_training_memory.set_rechunk_callback(self._on_training_update)
            self._rechunk_callback_set = True
        
        logger.info(f"Coordinator initialized successfully with {num_submitters} submitters")
        
        # Initialize workflow predictions
        await self.refresh_workflow_predictions()
    
    async def refresh_workflow_predictions(self) -> None:
        """Refresh workflow predictions based on actual agent state."""
        try:
            from backend.shared.boost_manager import boost_manager
            
            # Get actual sequence counters from agents
            submitter_sequences = {}
            for i, agent in enumerate(self.submitters):
                submitter_sequences[i + 1] = agent.task_sequence
            
            validator_sequence = self.validator.task_sequence if self.validator else 0
            
            # Build workflow tasks based on actual agent sequences
            tasks = []
            num_submitters = len(self.submitter_configs)
            cycle_length = num_submitters + 1
            
            # Copy sequences for prediction
            sub_seqs = dict(submitter_sequences)
            val_seq = validator_sequence
            
            for i in range(20):
                position_in_cycle = i % cycle_length
                
                if position_in_cycle < num_submitters:
                    submitter_id = position_in_cycle + 1
                    seq = sub_seqs.get(submitter_id, 0)
                    task_id = f"agg_sub{submitter_id}_{seq:03d}"
                    role = f"Submitter {submitter_id}" + (" (Main Submitter)" if submitter_id == 1 else "")
                    sub_seqs[submitter_id] = seq + 1
                else:
                    task_id = f"agg_val_{val_seq:03d}"
                    role = "Validator"
                    val_seq += 1
                
                tasks.append(WorkflowTask(
                    task_id=task_id,
                    sequence_number=i + 1,
                    role=role,
                    mode=None,
                    provider="lm_studio",
                    using_boost=boost_manager.should_use_boost(task_id)
                ))
            
            self.workflow_tasks = tasks
            
            # Broadcast workflow update
            if self.websocket_broadcaster:
                await self.websocket_broadcaster("workflow_updated", {
                    "tasks": [task.model_dump() for task in self.workflow_tasks],
                    "mode": "aggregator"
                })
            
            logger.debug(f"Refreshed workflow predictions: {len(self.workflow_tasks)} tasks")
        except Exception as e:
            logger.error(f"Failed to refresh workflow predictions: {e}")
    
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
        
        # Refresh predictions after each completion to keep workflow panel updated
        await self.refresh_workflow_predictions()
    
    def _handle_task_event(self, event_type: str, task_id: str) -> None:
        """
        Handle task events from submitters and validator.
        Called synchronously by agents; schedules async work on event loop.
        
        Args:
            event_type: "started" or "completed"
            task_id: The task ID (e.g., "agg_sub1_001", "agg_val_002")
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
    
    async def _check_and_update_pause_state(self) -> None:
        """Check queue size and update submitter pause state."""
        queue_size = await queue_manager.size()
        
        # Pause if queue >= 10, resume if < 10
        should_pause = queue_size >= system_config.queue_overflow_threshold
        
        if should_pause != self.should_pause_submitters:
            self.should_pause_submitters = should_pause
            if should_pause:
                logger.info(f"Queue size ({queue_size}) >= threshold ({system_config.queue_overflow_threshold}). Pausing submitters.")
                await self._broadcast("submitters_paused", {
                    "queue_size": queue_size,
                    "threshold": system_config.queue_overflow_threshold
                })
            else:
                logger.info(f"Queue size ({queue_size}) < threshold ({system_config.queue_overflow_threshold}). Resuming submitters.")
                await self._broadcast("submitters_resumed", {
                    "queue_size": queue_size
                })
    
    async def should_pause_submitter(self, submitter_id: int) -> bool:
        """
        Per-submitter fairness gate.
        
        Returns True if:
          - the global queue-overflow pause is active (queue >= queue_overflow_threshold), OR
          - this specific submitter already has more than per_submitter_queue_threshold
            of its own submissions waiting in the queue.
        
        The per-submitter cap is skipped when only one submitter is configured
        (no one else to be fair to - the global cap alone governs throughput).
        """
        if self.should_pause_submitters:
            return True
        if len(self.submitters) <= 1:
            return False
        own_count = await queue_manager.count_for_submitter(submitter_id)
        return own_count > system_config.per_submitter_queue_threshold
    
    async def start(self) -> None:
        """Start the aggregator system."""
        if self.is_running:
            logger.warning("Coordinator already running")
            return
        
        self.is_running = True
        logger.info("Starting coordinator...")
        
        # Reset free model manager state for fresh start
        free_model_manager.reset()
        
        # Refresh workflow predictions at start
        await self.refresh_workflow_predictions()
        
        if self.single_model_mode:
            # Single-model mode: Round-based sequential workflow
            logger.info("Starting single-model workflow (sequential submitters + validator)")
            self._main_task = asyncio.create_task(self._single_model_workflow())
        else:
            # Multi-model mode: Parallel submitters + independent validator
            logger.info("Starting multi-model workflow (parallel submitters)")
            for submitter in self.submitters:
                await submitter.start()
            self._validator_task = asyncio.create_task(self._validator_loop())
        
        await self._broadcast("system_started", {"message": "Aggregator system started"})
        logger.info("Coordinator started successfully")
    
    async def stop(self) -> None:
        """Stop the aggregator system."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping coordinator...")
        
        if self.single_model_mode:
            # Single-model mode: Cancel main task
            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
        else:
            # Multi-model mode: Stop submitters and validator task
            for submitter in self.submitters:
                await submitter.stop()
            
            if self._validator_task:
                self._validator_task.cancel()
                try:
                    await self._validator_task
                except asyncio.CancelledError:
                    pass
        
        # Cancel re-chunking task if running
        if self._rechunk_task and not self._rechunk_task.done():
            logger.info("Cancelling background re-chunking task...")
            self._rechunk_task.cancel()
            try:
                await self._rechunk_task
            except asyncio.CancelledError:
                pass
        
        await self._broadcast("system_stopped", {"message": "Aggregator system stopped"})
        logger.info("Coordinator stopped")
    
    async def add_submission_to_queue(self, submission: Submission) -> None:
        """Add a submission to the queue (called by submitters)."""
        await queue_manager.enqueue(submission)
        self.total_submissions += 1
        await self._broadcast("new_submission", {
            "submission_id": submission.submission_id,
            "submitter_id": submission.submitter_id,
            "queue_size": await queue_manager.size()
        })
    
    async def _validator_loop(self) -> None:
        """Main validator loop - continuously process queue with batch validation."""
        logger.info("Validator loop started - will run continuously until stopped (batch mode: up to 3)")
        iteration = 0
        
        while self.is_running:
            try:
                iteration += 1
                
                # Check queue size and update pause state BEFORE dequeuing (proactive pause)
                await self._check_and_update_pause_state()
                
                # Get batch of up to 3 submissions from queue
                submissions = await queue_manager.dequeue_batch(max_count=3)
                
                if not submissions:
                    logger.debug(f"Validator iteration {iteration} - queue empty, waiting...")
                    await asyncio.sleep(1)
                    continue
                
                logger.info(
                    f"Validator iteration {iteration} - batch validating {len(submissions)} submissions: "
                    f"{[s.submission_id for s in submissions]}"
                )
                
                # Batch validate
                results = await self.validator.validate_batch(submissions)
                
                # Process results
                for submission, result in zip(submissions, results):
                    if result.decision == "accept":
                        await self._handle_acceptance(submission, result)
                    else:
                        await self._handle_rejection(submission, result)
                
                # Check and update pause state after validation batch
                await self._check_and_update_pause_state()
                
            except asyncio.CancelledError:
                logger.info(f"Validator loop cancelled at iteration {iteration}")
                break
            except FreeModelExhaustedError as e:
                # All free models exhausted after retries - wait briefly and retry
                logger.warning(f"Validator: all free models exhausted: {e}")
                if self.broadcast_callback:
                    await self.broadcast_callback("free_models_exhausted", {
                        "role_id": "aggregator_validator",
                        "message": str(e),
                    })
                await asyncio.sleep(120)  # Wait before retrying (all models exhausted)
            except Exception as e:
                logger.error(f"Validator loop error on iteration {iteration}: {e}", exc_info=True)
                await asyncio.sleep(2)
        
        logger.warning(f"Validator loop EXITED after {iteration} iterations - is_running={self.is_running}")
    
    async def _single_model_workflow(self) -> None:
        """
        Round-based workflow for single-model mode.
        
        When submitter and validator use the same model, runs sequentially:
        1. All submitters generate submissions one by one (S1 → S2 → S3)
        2. Validator processes all queued submissions
        3. Repeat
        
        This prevents queue overflow that would occur from parallel tasks
        competing for the same model.
        """
        logger.info("Single-model workflow started - round-based sequential execution")
        round_number = 0
        
        while self.is_running:
            try:
                round_number += 1
                logger.info(f"Single-model workflow: Starting round {round_number}")
                
                # Phase 1: Run all submitters sequentially
                submissions_generated = 0
                for i, submitter in enumerate(self.submitters, 1):
                    if not self.is_running:
                        break
                    
                    logger.debug(f"Round {round_number}: Submitter {i} generating...")
                    submission = await submitter._generate_submission()
                    
                    if submission:
                        await self.add_submission_to_queue(submission)
                        submissions_generated += 1
                        logger.info(
                            f"Round {round_number}: Submitter {i} generated submission "
                            f"{submission.submission_id}"
                        )
                    else:
                        logger.debug(f"Round {round_number}: Submitter {i} returned None")
                
                if not self.is_running:
                    break
                
                logger.info(
                    f"Round {round_number}: All submitters complete, "
                    f"{submissions_generated} submissions queued. Starting validation..."
                )
                
                # Phase 2: Validate all queued submissions using batch validation (up to 3 at once)
                validations_done = 0
                while self.is_running:
                    # Check queue size and update pause state BEFORE dequeuing (proactive pause)
                    await self._check_and_update_pause_state()
                    
                    submissions = await queue_manager.dequeue_batch(max_count=3)
                    
                    if not submissions:
                        # Queue empty - move to next round
                        break
                    
                    logger.info(
                        f"Round {round_number}: Batch validating {len(submissions)} submissions: "
                        f"{[s.submission_id for s in submissions]}"
                    )
                    
                    results = await self.validator.validate_batch(submissions)
                    validations_done += len(submissions)
                    
                    for submission, result in zip(submissions, results):
                        if result.decision == "accept":
                            await self._handle_acceptance(submission, result)
                        else:
                            await self._handle_rejection(submission, result)
                
                # Check and update pause state after validation batch
                await self._check_and_update_pause_state()
                
                logger.info(
                    f"Round {round_number} complete: {submissions_generated} generated, "
                    f"{validations_done} validated"
                )
                
                # Brief delay between rounds
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info(f"Single-model workflow cancelled at round {round_number}")
                break
            except FreeModelExhaustedError as e:
                # All free models exhausted after retries - wait briefly and retry
                logger.warning(f"Single-model workflow: all free models exhausted: {e}")
                if self.broadcast_callback:
                    await self.broadcast_callback("free_models_exhausted", {
                        "role_id": "aggregator_single_model",
                        "message": str(e),
                    })
                await asyncio.sleep(120)  # Wait before retrying (all models exhausted)
            except Exception as e:
                logger.error(f"Single-model workflow error at round {round_number}: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.warning(f"Single-model workflow EXITED after {round_number} rounds")
    
    async def _handle_acceptance(self, submission: Submission, result: ValidationResult) -> None:
        """Handle accepted submission."""
        self.total_acceptances += 1
        
        # Add to shared training
        await shared_training_memory.add_accepted_submission(submission.content)
        
        # Notify submitter
        submitter = next((s for s in self.submitters if s.submitter_id == submission.submitter_id), None)
        if submitter:
            await submitter.handle_acceptance()
        
        # Get submitter config for model info
        submitter_config = self.submitter_configs[submission.submitter_id - 1] if submission.submitter_id <= len(self.submitter_configs) else None
        submitter_call = submission.metadata.get("llm_call", {}) if submission.metadata else {}
        validator_call = result.metadata.get("llm_call", {}) if result.metadata else {}
        configured_submitter_model = submitter_config.model_id if submitter_config else (submitter.model_name if submitter else "unknown")
        configured_submitter_provider = submitter_config.provider if submitter_config else ("openrouter" if submission.submitter_id == 11 else "lm_studio")
        actual_submitter_model = submitter_call.get("effective_model") or configured_submitter_model
        actual_submitter_provider = submitter_call.get("provider") or configured_submitter_provider
        actual_validator_model = validator_call.get("effective_model") or self.validator_model
        actual_validator_provider = validator_call.get("provider") or self.validator_provider
        
        # Broadcast
        await self._broadcast("submission_accepted", {
            "submission_id": submission.submission_id,
            "submitter_id": submission.submitter_id,
            "submitter_model": actual_submitter_model,
            "submitter_provider": actual_submitter_provider,
            "submitter_configured_model": configured_submitter_model,
            "submitter_configured_provider": configured_submitter_provider,
            "submitter_boosted": bool(submitter_call.get("boosted", False)),
            "submitter_boost_mode": submitter_call.get("boost_mode"),
            "content": submission.content,
            "reasoning": result.reasoning,
            "total_acceptances": self.total_acceptances,
            "validator_model": actual_validator_model,
            "validator_provider": actual_validator_provider,
            "validator_configured_model": self.validator_model,
            "validator_configured_provider": self.validator_provider,
            "validator_boosted": bool(validator_call.get("boosted", False)),
            "validator_boost_mode": validator_call.get("boost_mode"),
        })
        
        logger.info(f"Accepted submission from submitter {submission.submitter_id} (total: {self.total_acceptances})")
        
        # Log key event to persistent log
        await event_log.add_event(
            "submission_accepted",
            f"Submission from Submitter {submission.submitter_id} ACCEPTED (#{self.total_acceptances})",
            {"submitter_id": submission.submitter_id, "total_acceptances": self.total_acceptances}
        )
        
        # Save stats
        await self._save_stats()
        
        # Trigger cleanup review every 7 acceptances
        if self.enable_cleanup_review and self.total_acceptances % 7 == 0 and self.total_acceptances > 0:
            await self._perform_cleanup_review()
    
    async def _handle_rejection(self, submission: Submission, result: ValidationResult) -> None:
        """Handle rejected submission."""
        self.total_rejections += 1
        
        # Notify submitter (stores last 5 rejections in local memory)
        submitter = next((s for s in self.submitters if s.submitter_id == submission.submitter_id), None)
        if submitter:
            await submitter.handle_rejection(result.summary, submission.content)
        
        # Get submitter config for model info
        submitter_config = self.submitter_configs[submission.submitter_id - 1] if submission.submitter_id <= len(self.submitter_configs) else None
        submitter_call = submission.metadata.get("llm_call", {}) if submission.metadata else {}
        validator_call = result.metadata.get("llm_call", {}) if result.metadata else {}
        configured_submitter_model = submitter_config.model_id if submitter_config else (submitter.model_name if submitter else "unknown")
        configured_submitter_provider = submitter_config.provider if submitter_config else ("openrouter" if submission.submitter_id == 11 else "lm_studio")
        actual_submitter_model = submitter_call.get("effective_model") or configured_submitter_model
        actual_submitter_provider = submitter_call.get("provider") or configured_submitter_provider
        actual_validator_model = validator_call.get("effective_model") or self.validator_model
        actual_validator_provider = validator_call.get("provider") or self.validator_provider
        
        # Broadcast
        await self._broadcast("submission_rejected", {
            "submission_id": submission.submission_id,
            "submitter_id": submission.submitter_id,
            "submitter_model": actual_submitter_model,
            "submitter_provider": actual_submitter_provider,
            "submitter_configured_model": configured_submitter_model,
            "submitter_configured_provider": configured_submitter_provider,
            "submitter_boosted": bool(submitter_call.get("boosted", False)),
            "submitter_boost_mode": submitter_call.get("boost_mode"),
            "reasoning": result.reasoning,
            "total_rejections": self.total_rejections,
            "validator_model": actual_validator_model,
            "validator_provider": actual_validator_provider,
            "validator_configured_model": self.validator_model,
            "validator_configured_provider": self.validator_provider,
            "validator_boosted": bool(validator_call.get("boosted", False)),
            "validator_boost_mode": validator_call.get("boost_mode"),
        })
        
        logger.info(f"Rejected submission from submitter {submission.submitter_id} (total: {self.total_rejections})")
        
        # Log key event to persistent log
        rejection_reason = result.summary[:200] if result.summary else result.reasoning[:200]
        await event_log.add_event(
            "submission_rejected",
            f"Submission from Submitter {submission.submitter_id} REJECTED: {rejection_reason}",
            {"submitter_id": submission.submitter_id, "total_rejections": self.total_rejections}
        )
        
        # Save stats
        await self._save_stats()
    
    async def _perform_cleanup_review(self) -> None:
        """
        Perform a cleanup review of the accepted submissions database.
        
        Called every 7 acceptances to check if any previously accepted submission
        should now be removed due to redundancy, contradictions, etc.
        """
        try:
            self.cleanup_reviews_performed += 1
            
            logger.info("=" * 80)
            logger.info("CLEANUP DEBUG: ================== COORDINATOR CLEANUP REVIEW START ==================")
            logger.info("=" * 80)
            logger.info(f"CLEANUP DEBUG: Review #{self.cleanup_reviews_performed}")
            logger.info(f"CLEANUP DEBUG: Triggered at total_acceptances={self.total_acceptances} (every 7 acceptances)")
            logger.info(f"CLEANUP DEBUG: Trigger condition check: {self.total_acceptances} % 7 == {self.total_acceptances % 7}")
            logger.info(f"CLEANUP DEBUG: Stats - removals_proposed={self.removals_proposed}, removals_executed={self.removals_executed}")
            
            logger.info(f"Starting cleanup review #{self.cleanup_reviews_performed} (triggered at {self.total_acceptances} acceptances)")
            
            await self._broadcast("cleanup_review_started", {
                "review_number": self.cleanup_reviews_performed,
                "total_acceptances": self.total_acceptances
            })
            
            # Phase 1: Ask validator to review for potential removal
            logger.info("CLEANUP DEBUG: >>> PHASE 1: Calling validator.perform_cleanup_review()...")
            removal_proposal = await self.validator.perform_cleanup_review()
            logger.info(f"CLEANUP DEBUG: <<< PHASE 1 Complete: removal_proposal={removal_proposal}")
            
            if removal_proposal is None:
                # No removal needed
                logger.info("CLEANUP DEBUG: No removal proposal returned (None) - cleanup review complete")
                logger.info(f"Cleanup review #{self.cleanup_reviews_performed}: No removal needed")
                await self._broadcast("cleanup_review_complete", {
                    "review_number": self.cleanup_reviews_performed,
                    "removal_proposed": False,
                    "removal_executed": False
                })
                logger.info("CLEANUP DEBUG: ================== COORDINATOR CLEANUP REVIEW END (No Removal) ==================")
                return
            
            # Removal proposed
            self.removals_proposed += 1
            submission_number = removal_proposal["submission_number"]
            removal_reasoning = removal_proposal["reasoning"]
            
            logger.info(f"CLEANUP DEBUG: REMOVAL PROPOSED for submission #{submission_number}")
            logger.info(f"CLEANUP DEBUG: Removal reasoning: {removal_reasoning[:300]}...")
            logger.info(
                f"Cleanup review #{self.cleanup_reviews_performed}: Removal proposed for submission #{submission_number}"
            )
            
            await self._broadcast("cleanup_removal_proposed", {
                "review_number": self.cleanup_reviews_performed,
                "submission_number": submission_number,
                "reasoning": removal_reasoning[:500]  # Truncate for broadcast
            })
            
            # Phase 2: Get the submission content for validation
            logger.info(f"CLEANUP DEBUG: >>> PHASE 2: Getting content for submission #{submission_number}...")
            submission_content = await shared_training_memory.get_submission_content(submission_number)
            
            if submission_content is None:
                logger.warning(f"CLEANUP DEBUG: SUBMISSION NOT FOUND - submission #{submission_number} does not exist in database")
                logger.warning(
                    f"Cleanup review #{self.cleanup_reviews_performed}: "
                    f"Submission #{submission_number} not found for validation"
                )
                await self._broadcast("cleanup_review_complete", {
                    "review_number": self.cleanup_reviews_performed,
                    "removal_proposed": True,
                    "removal_executed": False,
                    "reason": "Submission not found"
                })
                logger.info("CLEANUP DEBUG: ================== COORDINATOR CLEANUP REVIEW END (Not Found) ==================")
                return
            
            logger.info(f"CLEANUP DEBUG: <<< PHASE 2 Complete: Got submission content, length={len(submission_content)} chars")
            logger.debug(f"CLEANUP DEBUG: Submission content preview:\n{submission_content[:500]}...")
            
            # Phase 3: Validate the removal decision
            logger.info("CLEANUP DEBUG: >>> PHASE 3: Calling validator.validate_removal()...")
            removal_validated = await self.validator.validate_removal(
                submission_number=submission_number,
                submission_content=submission_content,
                removal_reasoning=removal_reasoning
            )
            logger.info(f"CLEANUP DEBUG: <<< PHASE 3 Complete: removal_validated={removal_validated}")
            
            if not removal_validated:
                logger.info(f"CLEANUP DEBUG: REMOVAL NOT VALIDATED - keeping submission #{submission_number}")
                logger.info(
                    f"Cleanup review #{self.cleanup_reviews_performed}: "
                    f"Removal of submission #{submission_number} was NOT validated"
                )
                await self._broadcast("cleanup_review_complete", {
                    "review_number": self.cleanup_reviews_performed,
                    "removal_proposed": True,
                    "removal_executed": False,
                    "reason": "Removal not validated"
                })
                logger.info("CLEANUP DEBUG: ================== COORDINATOR CLEANUP REVIEW END (Not Validated) ==================")
                return
            
            # Phase 4: Execute the removal
            logger.info(f"CLEANUP DEBUG: >>> PHASE 4: Executing removal of submission #{submission_number}...")
            removal_success = await shared_training_memory.remove_submission(submission_number, trigger_rechunk=False)
            logger.info(f"CLEANUP DEBUG: <<< PHASE 4 Complete: removal_success={removal_success}")
            
            if removal_success:
                self.removals_executed += 1
                logger.info(f"CLEANUP DEBUG: REMOVAL EXECUTED SUCCESSFULLY for submission #{submission_number}")
                logger.info(f"CLEANUP DEBUG: Total removals executed: {self.removals_executed}")
                logger.info(
                    f"Cleanup review #{self.cleanup_reviews_performed}: "
                    f"Successfully removed submission #{submission_number}"
                )
                await self._broadcast("cleanup_submission_removed", {
                    "review_number": self.cleanup_reviews_performed,
                    "submission_number": submission_number,
                    "reasoning": removal_reasoning[:500],
                    "total_removals": self.removals_executed
                })
                
                # Full RAG rebuild so deleted content is no longer retrievable
                await self._rebuild_shared_training_rag_after_cleanup()
                
                # Log key event to persistent log
                await event_log.add_event(
                    "cleanup_submission_removed",
                    f"Cleanup removed submission #{submission_number}: {removal_reasoning[:200]}",
                    {"submission_number": submission_number, "total_removals": self.removals_executed}
                )
                
                # Save stats
                await self._save_stats()
            else:
                logger.warning(f"CLEANUP DEBUG: REMOVAL FAILED for submission #{submission_number}")
                logger.warning(
                    f"Cleanup review #{self.cleanup_reviews_performed}: "
                    f"Failed to remove submission #{submission_number}"
                )
            
            await self._broadcast("cleanup_review_complete", {
                "review_number": self.cleanup_reviews_performed,
                "removal_proposed": True,
                "removal_executed": removal_success,
                "submission_number": submission_number
            })
            
            logger.info(f"CLEANUP DEBUG: ================== COORDINATOR CLEANUP REVIEW END (Success={removal_success}) ==================")
            
        except Exception as e:
            logger.error(f"CLEANUP DEBUG: EXCEPTION in _perform_cleanup_review: {e}", exc_info=True)
            logger.error(f"Cleanup review failed: {e}", exc_info=True)
            await self._broadcast("cleanup_review_error", {
                "review_number": self.cleanup_reviews_performed,
                "error": str(e)
            })
    
    async def _on_training_update(self) -> None:
        """Callback when shared training is updated - trigger NON-BLOCKING re-chunking."""
        # Cancel previous re-chunking task if still running
        # CRITICAL: Don't await the cancellation - that would block the validator loop!
        if self._rechunk_task and not self._rechunk_task.done():
            logger.warning("Previous re-chunking still in progress, cancelling it...")
            self._rechunk_task.cancel()
            # Task will catch CancelledError and clean up in background
        
        # Launch re-chunking in background task
        self._rechunk_task = asyncio.create_task(self._rechunk_training_data())
        logger.info("Launched background re-chunking task (validator continues processing)")
    
    async def _rechunk_training_data(self) -> None:
        """Background task for incremental re-chunking training data with global lock."""
        try:
            # ACQUIRE GLOBAL RAG LOCK
            await rag_operation_lock.acquire("Aggregator immediate re-chunk")
            
            logger.info("Background incremental re-chunking started...")
            
            # Get only new submissions since last RAG
            new_submissions = await shared_training_memory.get_new_submissions_since_last_rag()
            
            if not new_submissions:
                logger.info("Incremental re-chunking: No new submissions to process")
                await self._broadcast("rechunk_complete", {
                    "chunk_size": None,
                    "new_submissions": 0,
                    "mode": "incremental"
                })
                return
            
            logger.info(f"Incremental re-chunking: Processing {len(new_submissions)} new submissions")
            
            # DESIGN NOTE: Incremental re-chunking intentionally accumulates chunks over time.
            # Each batch is added with source name "rag_shared_training_update_{chunk_size}".
            # This accumulation is INTENTIONAL because:
            # - RAG retrieves most relevant chunks regardless of which batch they're from
            # - MAX_CHUNKS_PER_SIZE (10,000 per size) caps growth to prevent unbounded memory
            # - Keeps all historical data available for semantic search
            # - Simpler than periodic full re-chunk with no risk of data loss during cleanup
            
            # Determine chunk size using coordinator-level cycling
            chunk_size = rag_config.submitter_chunk_intervals[self.current_rechunk_index]
            self.current_rechunk_index = (self.current_rechunk_index + 1) % len(rag_config.submitter_chunk_intervals)
            logger.info(f"Incremental re-chunking: Using chunk_size={chunk_size}")
            
            # Add each new submission as text chunks
            # Combine all new submissions into single text to add
            combined_new_content = "\n\n".join([
                f"{'=' * 80}\nSUBMISSION #{sub.get('number') or idx+self.last_ragged_submission_count} | Accepted: {sub.get('timestamp', 'Unknown')}\n{'=' * 80}\n\n{sub['content']}\n"
                for idx, sub in enumerate(new_submissions)
            ])
            
            # Add the combined new submissions with current chunk size
            await rag_manager.add_text(
                combined_new_content,
                f"rag_shared_training_update_{chunk_size}",  # Unique name per chunk size
                chunk_sizes=[chunk_size],
                is_permanent=False
            )
            
            # Mark submissions as RAG'd
            current_count = await shared_training_memory.get_insights_count()
            await shared_training_memory.mark_submissions_ragged(current_count)
            
            logger.info(f"Incremental re-chunking COMPLETE - {len(new_submissions)} submissions added, chunk_size={chunk_size}")
            
            # Broadcast success
            await self._broadcast("rechunk_complete", {
                "chunk_size": chunk_size,
                "new_submissions": len(new_submissions),
                "total_submissions": current_count,
                "mode": "incremental"
            })
            
        except asyncio.CancelledError:
            logger.info("Incremental re-chunking cancelled (newer update triggered)")
            raise  # Re-raise to properly clean up task
        except Exception as e:
            logger.error(f"Incremental re-chunking FAILED: {e}", exc_info=True)
            await self._broadcast("rechunk_error", {
                "error": str(e),
                "message": "Incremental re-chunking failed but system continues"
            })
        finally:
            # ALWAYS RELEASE LOCK
            rag_operation_lock.release()
    
    async def _rebuild_shared_training_rag_after_cleanup(self) -> None:
        """Full RAG rebuild of shared-training content after a cleanup removal.
        
        The normal incremental rechunk path is append-only and cannot remove
        deleted content from RAG. After a prune we must drop all shared-training
        RAG sources and re-add the current (post-removal) file so retrieval
        results stay consistent with the live database.
        """
        current_path = Path(shared_training_memory.file_path)
        current_count = await shared_training_memory.get_insights_count()
        
        await rag_operation_lock.acquire("Aggregator cleanup full re-rag")
        try:
            # Collect every source name that could contain shared-training chunks
            candidate_sources = [current_path.name, current_path.with_suffix(".tmp").name]
            for size in rag_config.submitter_chunk_intervals:
                candidate_sources.append(f"rag_shared_training_update_{size}")
            
            for source in dict.fromkeys(candidate_sources):
                if source in rag_manager.document_access_order:
                    await rag_manager.remove_document(source)
            
            if current_count > 0 and current_path.exists():
                await rag_manager.add_document(
                    str(current_path),
                    chunk_sizes=rag_config.submitter_chunk_intervals,
                    is_user_file=False,
                )
            
            await shared_training_memory.mark_submissions_ragged(current_count)
            logger.info(f"Cleanup full re-RAG complete: {current_count} live submissions re-indexed")
        except Exception as e:
            logger.error(f"Cleanup full re-RAG failed: {e}", exc_info=True)
            raise
        finally:
            rag_operation_lock.release()
    
    async def get_status(self) -> SystemStatus:
        """Get current system status."""
        queue_size = await queue_manager.size()
        shared_training_size = await shared_training_memory.get_insights_count()
        
        acceptance_rate = 0.0
        if self.total_submissions > 0:
            acceptance_rate = self.total_acceptances / self.total_submissions
        
        return SystemStatus(
            is_running=self.is_running,
            queue_size=queue_size,
            total_submissions=self.total_submissions,
            total_acceptances=self.total_acceptances,
            total_rejections=self.total_rejections,
            acceptance_rate=acceptance_rate,
            submitter_states=[s.get_state() for s in self.submitters],
            shared_training_size=shared_training_size,
            cleanup_reviews_performed=self.cleanup_reviews_performed,
            removals_proposed=self.removals_proposed,
            removals_executed=self.removals_executed
        )
    
    async def get_results(self) -> str:
        """Get all accepted submissions (plain content for API/display)."""
        return await shared_training_memory.get_all_content()
    
    async def get_results_formatted(self) -> str:
        """Get all accepted submissions with formatting for file export."""
        return await shared_training_memory.get_all_content_formatted()
    
    async def get_model_settings(self) -> Dict[str, Any]:
        """Get current aggregator model settings."""
        submitter_info = []
        for submitter in self.submitters:
            submitter_info.append({
                "submitter_id": submitter.submitter_id,
                "model_id": submitter.model_name,
                "context_window": getattr(submitter, 'context_window', rag_config.submitter_context_window),
                "max_output_tokens": getattr(submitter, 'max_output_tokens', rag_config.submitter_max_output_tokens)
            })
        
        settings = {
            "submitter_configs": submitter_info,
            "validator_model": self.validator.model_name if self.validator else "",
            "num_submitters": len(self.submitters)
        }
        return settings
    
    async def clear_all_submissions(self) -> None:
        """Clear all accepted submissions and reset the system."""
        logger.info("Clearing all accepted submissions and resetting system...")
        
        # Stop system if running
        was_running = self.is_running
        if was_running:
            await self.stop()
        
        # Clear shared training memory
        async with shared_training_memory._lock:
            shared_training_memory.insights.clear()
            shared_training_memory.submission_count = 0
            shared_training_memory.last_ragged_submission_count = 0
            await shared_training_memory._save()
        
        # Clear local rejection logs for all submitters
        for submitter in self.submitters:
            await submitter.local_memory.clear()
        
        # Clear RAG database
        try:
            await asyncio.to_thread(rag_manager.clear_all_documents)
            logger.info("Cleared RAG database")
        except Exception as e:
            logger.error(f"Failed to clear RAG database: {e}")
        
        # Reset coordinator stats
        self.total_submissions = 0
        self.total_acceptances = 0
        self.total_rejections = 0
        self.cleanup_reviews_performed = 0
        self.removals_proposed = 0
        self.removals_executed = 0
        
        # Clear persisted stats
        await self._save_stats()
        
        # Clear event log
        await event_log.clear()
        
        # Clear queue
        await queue_manager.clear()
        
        logger.info("All submissions cleared and system reset")
        await self._broadcast("system_reset", {"message": "All submissions cleared"})
    
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


# Global coordinator instance
coordinator = Coordinator()

