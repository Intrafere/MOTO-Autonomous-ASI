"""
Submitter agent - generates submissions in parallel with other submitters.
Cycles through chunk sizes (256 → 512 → 768 → 1024) independently.
"""
import asyncio
from typing import Optional, Dict, Callable
import logging
import httpx
from datetime import datetime
import uuid

from backend.shared.config import rag_config, system_config
from backend.shared.models import Submission, SubmitterState
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.aggregator.core.context_allocator import context_allocator
from backend.aggregator.core.queue_manager import queue_manager
from backend.aggregator.memory.shared_training import shared_training_memory
from backend.aggregator.memory.local_training import LocalTrainingMemory
from backend.aggregator.prompts.submitter_prompts import build_submitter_prompt
from backend.aggregator.validation.json_validator import json_validator

logger = logging.getLogger(__name__)


class SubmitterAgent:
    """
    Submitter agent that generates submissions.
    Runs in parallel with other submitters.
    Each submitter can use its own model and context window configuration.
    """
    
    def __init__(
        self,
        submitter_id: int,
        model_name: str,
        user_prompt: str,
        user_files_content: Dict[str, str],
        websocket_broadcaster: Optional[Callable] = None,
        context_window: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        coordinator: Optional['Coordinator'] = None
    ):
        self.submitter_id = submitter_id
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.user_files_content = user_files_content
        self.websocket_broadcaster = websocket_broadcaster
        self.coordinator = coordinator
        
        # Per-submitter context settings (fall back to global config if not provided)
        self.context_window = context_window if context_window is not None else rag_config.submitter_context_window
        self.max_output_tokens = max_output_tokens if max_output_tokens is not None else rag_config.submitter_max_output_tokens
        
        # State
        self.state = SubmitterState(submitter_id=submitter_id)
        self.chunk_sizes = rag_config.submitter_chunk_intervals
        self.current_chunk_index = 0
        
        # Memory
        self.local_memory = LocalTrainingMemory(submitter_id)
        
        # Control
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.submission_callback: Optional[callable] = None
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = f"aggregator_submitter_{submitter_id}"
        self.task_tracking_callback: Optional[Callable] = None
    
    async def initialize(self) -> None:
        """Initialize submitter agent."""
        await self.local_memory.initialize()
        logger.info(f"Submitter {self.submitter_id} initialized with model {self.model_name}")
    
    def set_submission_callback(self, callback: callable) -> None:
        """Set callback for when submissions are generated."""
        self.submission_callback = callback
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"agg_sub{self.submitter_id}_{self.task_sequence:03d}"
    
    async def start(self) -> None:
        """Start the submitter agent."""
        self.is_running = True
        self.state.is_active = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Submitter {self.submitter_id} started")
    
    async def stop(self) -> None:
        """Stop the submitter agent."""
        self.is_running = False
        self.state.is_active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Submitter {self.submitter_id} stopped")
    
    async def _run_loop(self) -> None:
        """Main run loop - continuously generate submissions."""
        iteration = 0
        logger.info(f"Submitter {self.submitter_id} run loop started - will run continuously until stopped")
        
        while self.is_running:
            try:
                # Check if we should pause due to queue overflow
                if self.coordinator and self.coordinator.should_pause_submitters:
                    logger.debug(f"Submitter {self.submitter_id} paused (queue overflow)")
                    await asyncio.sleep(2)  # Wait before checking again
                    continue
                
                iteration += 1
                logger.debug(f"Submitter {self.submitter_id} iteration {iteration} - generating submission...")
                
                submission = await self._generate_submission()
                if submission:
                    # Hold submission until queue has capacity (prevents overflow when
                    # the LLM call was already in-flight when the queue filled up)
                    while self.is_running:
                        queue_size = await queue_manager.size()
                        if queue_size < system_config.queue_overflow_threshold:
                            break
                        logger.debug(f"Submitter {self.submitter_id}: Queue full ({queue_size}), holding submission")
                        await asyncio.sleep(2)

                    if self.submission_callback and self.is_running:
                        await self.submission_callback(submission)
                        logger.info(f"Submitter {self.submitter_id} generated submission {submission.submission_id} (iteration {iteration})")
                else:
                    logger.warning(f"Submitter {self.submitter_id} iteration {iteration} - submission generation returned None (will retry)")
                
                # Brief delay between submissions
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Submitter {self.submitter_id} error on iteration {iteration}: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.warning(f"Submitter {self.submitter_id} run loop EXITED after {iteration} iterations - is_running={self.is_running}")
    
    async def _generate_submission(self) -> Optional[Submission]:
        """Generate a single submission."""
        try:
            # Get current chunk size
            chunk_size = self.chunk_sizes[self.current_chunk_index]
            
            # Cycle to next chunk size for next iteration
            self.current_chunk_index = (self.current_chunk_index + 1) % len(self.chunk_sizes)
            
            # Get context
            shared_training_content = await shared_training_memory.get_all_content()
            local_training_content = ""  # Local training would be added if implemented
            rejection_log_content = await self.local_memory.get_all_content()
            
            # Allocate context
            allocation = await context_allocator.allocate_submitter_context(
                user_prompt=self.user_prompt,
                json_schema=self._get_json_schema(),
                system_prompt=self._get_system_prompt(),
                shared_training_content=shared_training_content,
                local_training_content=local_training_content,
                rejection_log_content=rejection_log_content,
                user_files_content=self.user_files_content,
                chunk_size=chunk_size,
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens
            )
            
            # Build prompt
            rag_evidence = ""
            if allocation["rag_context"]:
                rag_evidence = allocation["rag_context"].text
            
            prompt = build_submitter_prompt(
                self.user_prompt,
                allocation["direct"],
                rag_evidence
            )
            
            # CRITICAL: Verify actual prompt size fits in context window
            from backend.shared.utils import count_tokens
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
            
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"Submitter {self.submitter_id}: Assembled prompt ({actual_prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed_tokens} tokens after safety margin). This indicates a context allocation bug."
                )
                return None  # Skip this submission
            
            logger.debug(f"Submitter {self.submitter_id} prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
            
            # Log RAG usage for transparency
            if allocation.get("rag_context") and allocation["rag_context"].text:
                logger.info(
                    f"Submitter {self.submitter_id} using RAG context: {len(allocation['rag_context'].evidence)} evidence chunks, "
                    f"{allocation['rag_context'].metadata.get('token_count', 0)} tokens, "
                    f"coverage={allocation['rag_context'].coverage:.2f}"
                )
            else:
                logger.debug(f"Submitter {self.submitter_id}: All content direct-injected, no RAG context used")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Generate completion with retry for 400 errors
            response = None
            max_retries = 3  # 400 errors won't fix themselves - fail fast
            
            for attempt in range(max_retries):
                try:
                    # Use api_client_manager for boost support
                    response = await api_client_manager.generate_completion(
                        task_id=task_id,
                        role_id=self.role_id,
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,  # Deterministic generation - evolving context provides diversity
                        max_tokens=self.max_output_tokens  # Per-submitter max output tokens
                    )
                    break  # Success
                    
                except (httpx.HTTPStatusError, ValueError) as e:
                    error_msg = str(e)
                    is_400_or_context = "400" in error_msg or "context" in error_msg.lower()
                    
                    if is_400_or_context and attempt < max_retries - 1:
                        backoff_time = min(2 ** attempt, 16)
                        logger.warning(
                            f"Submitter {self.submitter_id}: HTTP error (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                            f"Retrying in {backoff_time}s..."
                        )
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        # Final retry or non-recoverable error
                        logger.error(
                            f"Submitter {self.submitter_id}: Failed to generate completion after {attempt + 1} attempts: {e}"
                        )
                        # Notify task completed (failed but still completed)
                        if self.task_tracking_callback:
                            self.task_tracking_callback("completed", task_id)
                        return None  # Return None instead of crashing
                        
                except Exception as e:
                    logger.error(f"Submitter {self.submitter_id}: Unexpected error during completion: {e}")
                    # Notify task completed (failed but still completed)
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
            
            # Extract content
            if not response or not response.get("choices"):
                logger.error(f"Submitter {self.submitter_id}: No choices in response")
                return None
            
            # Extract content from either 'content' or 'reasoning' field
            # Some reasoning models (e.g., DeepSeek R1, certain GPT variants) output JSON in 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content", "") or message.get("reasoning", "")
            
            # Cache model config on first successful API call (only relevant for LM Studio)
            try:
                await lm_studio_client.cache_model_load_config(self.model_name, {
                    "context_length": self.context_window,
                    "model_path": self.model_name
                })
            except Exception:
                # Silently ignore - only applies to LM Studio models
                pass
            
            # Parse JSON
            try:
                parsed = parse_json(llm_output)
                valid = True
                error = None
            except Exception as parse_error:
                # Not corrupted, just invalid JSON - continue with conversational retry
                valid = False
                parsed = None
                error = str(parse_error)
            if not valid:
                # Two-stage conversational retry before final rejection
                logger.info(f"Submitter {self.submitter_id}: Initial JSON parse failed, attempting conversational retry")
                logger.debug(f"Parse error: {error}")
                
                # Stage 1: Guide proper JSON escaping for LaTeX
                retry_prompt_1 = (
                    "Your previous response could not be parsed as valid JSON.\n\n"
                    f"PARSE ERROR: {error}\n\n"
                    "JSON ESCAPING RULES FOR LaTeX:\n"
                    "LaTeX notation IS ALLOWED - but you must escape it properly in JSON:\n"
                    "1. Every backslash in your content needs ONE escape in JSON\n"
                    "   - To write \\mathbb{Z} in content, write: \"\\\\mathbb{Z}\" in JSON\n"
                    "   - To write \\( and \\), write: \"\\\\(\" and \"\\\\)\" in JSON\n"
                    "2. Do NOT double-escape: \\\\\\\\mathbb is WRONG, \\\\mathbb is CORRECT\n"
                    "3. Escape quotes inside strings: use \\\" for literal quotes\n"
                    "4. Avoid malformed unicode escapes (must be exactly \\uXXXX with 4 hex digits)\n\n"
                    "Please provide your submission again in valid JSON format:\n"
                    "{\n"
                    '  "submission": "your mathematical submission (LaTeX allowed, escape backslashes)",\n'
                    '  "reasoning": "your reasoning (LaTeX allowed, escape backslashes)"\n'
                    "}\n\n"
                    "Respond with ONLY the JSON object, no markdown, no explanation."
                )
                
                try:
                    # CRITICAL FIX: Don't include full failed output - it can be 90K+ tokens!
                    # Truncate to prevent context overflow during retry
                    max_failed_output_chars = 2000  # ~500 tokens - enough to show error context
                    if len(llm_output) > max_failed_output_chars:
                        failed_output_preview = llm_output[:max_failed_output_chars] + "\n[...output truncated for retry...]"
                    else:
                        failed_output_preview = llm_output
                    
                    # Calculate if conversation fits in context window
                    prompt_tokens = count_tokens(prompt)
                    preview_tokens = count_tokens(failed_output_preview)
                    retry_prompt_tokens = count_tokens(retry_prompt_1)
                    conversation_tokens = prompt_tokens + preview_tokens + retry_prompt_tokens
                    max_input = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
                    
                    if conversation_tokens > max_input:
                        # If even truncated conversation doesn't fit, just retry with original prompt
                        logger.warning(
                            f"Submitter {self.submitter_id}: Retry conversation too large ({conversation_tokens} > {max_input}), "
                            f"using simple retry without conversation context"
                        )
                        retry_response_1 = await api_client_manager.generate_completion(
                            task_id=f"{task_id}_retry1",
                            role_id=self.role_id,
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],  # Just retry original
                            temperature=0.0,
                            max_tokens=self.max_output_tokens
                        )
                    else:
                        # Build conversation with truncated failed output
                        retry_response_1 = await api_client_manager.generate_completion(
                            task_id=f"{task_id}_retry1",  # Track retry attempt
                            role_id=self.role_id,
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": failed_output_preview},
                                {"role": "user", "content": retry_prompt_1}
                            ],
                            temperature=0.0,  # Deterministic JSON formatting
                            max_tokens=self.max_output_tokens  # Respect max_tokens on retry
                        )
                    
                    if retry_response_1.get("choices"):
                        retry_output_1 = retry_response_1["choices"][0]["message"]["content"]
                        
                        try:
                            parsed = parse_json(retry_output_1)
                            valid = True
                            error = None
                            logger.info(f"Submitter {self.submitter_id}: First retry succeeded!")
                            llm_output = retry_output_1  # Use retry output for submission
                        except Exception as parse_error:
                            valid = False
                            parsed = None
                            error = str(parse_error)
                            logger.warning(f"Submitter {self.submitter_id}: First retry failed - {error}, attempting second retry")
                            
                            # Stage 2: More focused - emphasize correct escaping
                            retry_prompt_2 = (
                                "JSON parsing still failed. Focus on proper escaping.\n\n"
                                f"NEW PARSE ERROR: {error}\n\n"
                                "CRITICAL: For LaTeX in JSON, every backslash needs ONE escape:\n"
                                "- \\mathbb becomes \\\\mathbb in JSON\n"
                                "- \\( becomes \\\\( in JSON\n"
                                "- Do NOT double-escape: \\\\\\\\mathbb is WRONG\n\n"
                                "Example format:\n"
                                "{\n"
                                '  "submission": "For \\\\mathbb{Z}, we have \\\\phi: G \\\\to H",\n'
                                '  "reasoning": "This establishes the \\\\pi_1 connection"\n'
                                "}\n\n"
                                "Respond with ONLY the JSON, nothing else."
                            )
                            
                            try:
                                # Truncate retry output for second stage as well
                                if len(retry_output_1) > max_failed_output_chars:
                                    retry_output_1_preview = retry_output_1[:max_failed_output_chars] + "\n[...truncated...]"
                                else:
                                    retry_output_1_preview = retry_output_1
                                
                                # Check if second retry conversation fits
                                retry2_tokens = (prompt_tokens + preview_tokens + retry_prompt_tokens + 
                                               count_tokens(retry_output_1_preview) + count_tokens(retry_prompt_2))
                                
                                if retry2_tokens > max_input:
                                    # Too large - just retry with original prompt
                                    logger.warning(
                                        f"Submitter {self.submitter_id}: Second retry conversation too large, using simple retry"
                                    )
                                    retry_response_2 = await api_client_manager.generate_completion(
                                        task_id=f"{task_id}_retry2",
                                        role_id=self.role_id,
                                        model=self.model_name,
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=0.0,
                                        max_tokens=self.max_output_tokens
                                    )
                                else:
                                    retry_response_2 = await api_client_manager.generate_completion(
                                        task_id=f"{task_id}_retry2",  # Track second retry attempt
                                        role_id=self.role_id,
                                        model=self.model_name,
                                        messages=[
                                            {"role": "user", "content": prompt},
                                            {"role": "assistant", "content": failed_output_preview},
                                            {"role": "user", "content": retry_prompt_1},
                                            {"role": "assistant", "content": retry_output_1_preview},
                                            {"role": "user", "content": retry_prompt_2}
                                        ],
                                        temperature=0.0,  # Deterministic JSON formatting
                                        max_tokens=self.max_output_tokens  # Respect max_tokens on retry
                                    )
                                
                                if retry_response_2.get("choices"):
                                    retry_output_2 = retry_response_2["choices"][0]["message"]["content"]
                                    
                                    try:
                                        parsed = parse_json(retry_output_2)
                                        valid = True
                                        error = None
                                        logger.info(f"Submitter {self.submitter_id}: Second retry succeeded!")
                                        llm_output = retry_output_2
                                    except Exception as parse_error:
                                        valid = False
                                        parsed = None
                                        error = str(parse_error)
                                        logger.warning(f"Submitter {self.submitter_id}: Second retry also failed - {error}")
                            except Exception as e:
                                logger.error(f"Submitter {self.submitter_id}: Second retry request failed - {e}")
                    else:
                        logger.warning(f"Submitter {self.submitter_id}: First retry returned no choices")
                except Exception as e:
                    logger.error(f"Submitter {self.submitter_id}: First retry request failed - {e}")
                
                # If STILL invalid after both retries, proceed to rejection
                if not valid:
                    logger.warning(f"Submitter {self.submitter_id}: Invalid JSON after all retries - {error}")
                    
                    # Determine specific error pattern for feedback
                    error_type = "unknown"
                    if "LaTeX" in error or "\\(" in llm_output or "\\mathcal" in llm_output:
                        error_type = "LaTeX notation"
                    elif "unicode" in error.lower() or "\\u" in error:
                        error_type = "malformed unicode"
                    elif "escape" in error.lower():
                        error_type = "invalid escape sequence"
                    
                    # Create detailed feedback for rejection log
                    error_feedback = (
                        f"JSON validation failed ({error_type}): {error}\n\n"
                        f"CRITICAL JSON ESCAPING RULES:\n"
                        f"1. LaTeX IS ALLOWED but must be escaped: every \\ becomes \\\\ in JSON\n"
                        f"2. Example: to write \\mathbb{{Z}}, output \"\\\\mathbb{{Z}}\" in JSON\n"
                        f"3. Do NOT double-escape: \\\\\\\\mathbb is WRONG, \\\\mathbb is CORRECT\n"
                        f"4. Unicode must be exactly \\uXXXX (4 hex digits). Examples: \\u0394, \\u03B1\n"
                        f"5. For quotes in text, use \\\\\" (escaped quote)\n"
                        f"6. Valid JSON escapes: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX\n\n"
                        f"Your error was: {error_type}\n"
                        f"Make sure every LaTeX backslash has ONE escape in your JSON output."
                    )
                    # Record as rejection in local memory
                    await self.local_memory.add_rejection(
                        error_feedback,
                        llm_output[:750]
                    )
                    self._increment_rejection()
                    # Notify task completed (failed but still completed)
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
            
            # Create submission
            submission = Submission(
                submission_id=str(uuid.uuid4()),
                submitter_id=self.submitter_id,
                content=parsed["submission"],
                reasoning=parsed["reasoning"],
                chunk_size_used=chunk_size,
                metadata={
                    "chunk_size": chunk_size,
                    "rag_used": bool(allocation["rag_context"])
                }
            )
            
            # CRITICAL: Validate submission size before sending to validator
            # If submission is larger than output_reserve_tokens, it indicates an error or overflow
            from backend.shared.utils import count_tokens
            submission_tokens = count_tokens(parsed["submission"])
            max_submission_tokens = rag_config.output_reserve_tokens  # Should match max_tokens limit
            if submission_tokens > max_submission_tokens:
                logger.error(
                    f"Submitter {self.submitter_id}: Generated submission is too large "
                    f"({submission_tokens} tokens > {max_submission_tokens} max). "
                    f"This indicates the model generated excessive output despite max_tokens limit. "
                    f"Rejecting submission to prevent validator overflow."
                )
                # Record as a local error (not a validator rejection)
                await self.local_memory.add_rejection(
                    f"Submission exceeded size limit: {submission_tokens} tokens (max: {max_submission_tokens}). "
                    "This may indicate a mid-generation overflow or model error.",
                    parsed["submission"][:750]
                )
                self._increment_rejection()
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            self.state.total_submissions += 1
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            return submission
            
        except Exception as e:
            logger.error(f"Submitter {self.submitter_id} failed to generate submission: {e}")
            return None
    
    def _get_system_prompt(self) -> str:
        """Get system prompt."""
        from backend.aggregator.prompts.submitter_prompts import get_submitter_system_prompt
        return get_submitter_system_prompt()
    
    def _get_json_schema(self) -> str:
        """Get JSON schema."""
        from backend.aggregator.prompts.submitter_prompts import get_submitter_json_schema
        return get_submitter_json_schema()
    
    async def handle_acceptance(self) -> None:
        """Handle submission acceptance."""
        self.state.total_acceptances += 1
        self.state.consecutive_rejections = 0
        logger.info(f"Submitter {self.submitter_id}: Submission accepted (total: {self.state.total_acceptances})")
    
    async def handle_rejection(self, validator_summary: str, submission_content: str) -> None:
        """Handle submission rejection."""
        await self.local_memory.add_rejection(validator_summary, submission_content)
        self._increment_rejection()
        logger.info(f"Submitter {self.submitter_id}: Submission rejected (consecutive: {self.state.consecutive_rejections})")
    
    def _increment_rejection(self) -> None:
        """Increment rejection counter and check for reset."""
        self.state.consecutive_rejections += 1
        
        # Check if need to reset local database
        if self.state.consecutive_rejections >= system_config.consecutive_rejection_reset_threshold:
            asyncio.create_task(self._reset_local_memory())
    
    async def _reset_local_memory(self) -> None:
        """Reset local memory after too many consecutive rejections."""
        await self.local_memory.reset()
        self.state.consecutive_rejections = 0
        logger.warning(f"Submitter {self.submitter_id}: Reset local memory after {system_config.consecutive_rejection_reset_threshold} consecutive rejections")
    
    def get_state(self) -> SubmitterState:
        """Get current state."""
        return self.state

