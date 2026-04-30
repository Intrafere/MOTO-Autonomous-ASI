"""
Validator agent - validates submissions sequentially.
Always uses 512-char chunks for consistency.
"""
import asyncio
from typing import Optional, Dict, Callable, List, Any
import logging
import httpx

from backend.shared.config import rag_config
from backend.shared.models import Submission, ValidationResult
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.autonomous.memory.proof_database import proof_database
from backend.aggregator.core.context_allocator import context_allocator
from backend.aggregator.memory.shared_training import shared_training_memory
from backend.aggregator.prompts.validator_prompts import (
    build_validator_prompt,
    build_validator_dual_prompt,
    build_validator_triple_prompt,
    build_cleanup_review_prompt,
    build_removal_validation_prompt,
    get_validator_system_prompt,
    get_validator_json_schema,
    get_validator_dual_system_prompt,
    get_validator_dual_json_schema,
    get_validator_triple_system_prompt,
    get_validator_triple_json_schema,
    get_cleanup_review_system_prompt,
    get_cleanup_review_json_schema,
    get_removal_validation_system_prompt,
    get_removal_validation_json_schema
)
from backend.aggregator.validation.json_validator import json_validator
from backend.aggregator.validation.contradiction_checker import contradiction_checker

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """
    Validator agent that validates submissions sequentially.
    Always uses constant 512-char chunks.
    """
    
    def __init__(
        self,
        model_name: str,
        user_prompt: str,
        user_files_content: Dict[str, str],
        websocket_broadcaster: Optional[Callable] = None
    ):
        self.model_name = model_name
        self.user_prompt = proof_database.inject_into_prompt(user_prompt)
        self.user_files_content = user_files_content
        self.chunk_size = rag_config.validator_chunk_size  # Always 512
        self.websocket_broadcaster = websocket_broadcaster
        
        # Control
        self.is_running = False
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "aggregator_validator"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"agg_val_{self.task_sequence:03d}"
    
    async def initialize(self) -> None:
        """Initialize validator agent."""
        logger.info(f"Validator initialized with model {self.model_name}")
    
    async def validate_submission(self, submission: Submission) -> ValidationResult:
        """
        Validate a submission.
        
        Returns:
            ValidationResult with decision and reasoning
        """
        try:
            # Step 1: JSON validation (already done by submitter, but double-check)
            # Step 2: Contradiction check
            contradiction_passed, contradiction_reason = contradiction_checker.check_contradictions(
                submission.content
            )
            
            if not contradiction_passed:
                return ValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Contradiction check failed: {contradiction_reason}",
                    summary=f"Contains contradictions. {contradiction_reason}",
                    contradiction_check_passed=False,
                    json_valid=True
                )
            
            # Step 4: Quality assessment via LLM
            quality_result = await self._assess_quality(submission)
            
            return quality_result
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"Validation error: {e}",
                summary=f"Internal validation error",
                json_valid=False
            )
    
    async def _assess_quality(self, submission: Submission) -> ValidationResult:
        """Assess submission quality using LLM."""
        try:
            # Get context
            shared_training_content = await shared_training_memory.get_all_content()
            
            # Allocate context
            allocation = await context_allocator.allocate_validator_context(
                user_prompt=self.user_prompt,
                json_schema=self._get_json_schema(),
                system_prompt=self._get_system_prompt(),
                shared_training_content=shared_training_content,
                user_files_content=self.user_files_content,
                submission_content=submission.content,
                chunk_size=self.chunk_size
            )
            
            # Build prompt
            rag_evidence = ""
            if allocation["rag_context"]:
                rag_evidence = allocation["rag_context"].text
            
            prompt = build_validator_prompt(
                self.user_prompt,
                submission.content,
                allocation["direct"],
                rag_evidence
            )
            
            # CRITICAL: Verify actual prompt size fits in context window
            from backend.shared.utils import count_tokens
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(context_allocator.validator_context_window, rag_config.validator_max_output_tokens)
            configured_context = context_allocator.validator_context_window
            
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"Validator: Assembled prompt ({actual_prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed_tokens} tokens after safety margin). This indicates a context allocation bug."
                )
                return ValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning=f"Internal error: Prompt too large ({actual_prompt_tokens} tokens > {max_allowed_tokens} max)",
                    summary="Internal context overflow error",
                    json_valid=False
                )
            
            logger.debug(
                f"Validator prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens}, "
                f"configured: {configured_context})"
            )
            
            # Log RAG usage for transparency
            if allocation.get("rag_context") and allocation["rag_context"].text:
                logger.info(
                    f"Validator using RAG context: {len(allocation['rag_context'].evidence)} evidence chunks, "
                    f"{allocation['rag_context'].metadata.get('token_count', 0)} tokens, "
                    f"coverage={allocation['rag_context'].coverage:.2f}"
                )
            else:
                logger.debug("Validator: All content direct-injected, no RAG context used")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Generate validation with retry for 400 errors
            response = None
            call_metadata = {}
            max_retries = 3  # 400 errors won't fix themselves - fail fast
            
            for attempt in range(max_retries):
                try:
                    # Use api_client_manager for boost support
                    response = await api_client_manager.generate_completion(
                        task_id=task_id,
                        role_id=self.role_id,
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,  # Deterministic validation - evolving context provides diversity
                        max_tokens=rag_config.validator_max_output_tokens  # User-configurable max output tokens
                    )
                    call_metadata = api_client_manager.extract_call_metadata(response)
                    break  # Success
                    
                except (httpx.HTTPStatusError, ValueError) as e:
                    error_msg = str(e)
                    is_400_or_context = "400" in error_msg or "context" in error_msg.lower()
                    
                    # Detect LM Studio context overflow
                    if "context" in error_msg.lower() and "overflow" in error_msg.lower():
                        logger.critical(
                            f"⚠️  LM STUDIO CONTEXT MISMATCH DETECTED ⚠️\n"
                            f"LM Studio reported context overflow: {error_msg}\n"
                            f"Configured validator context: {configured_context} tokens\n"
                            f"Assembled prompt size: {actual_prompt_tokens} tokens\n\n"
                            f"DIAGNOSIS: LM Studio loaded the model with a SMALLER context than configured.\n"
                            f"SOLUTION: \n"
                            f"  1. Stop the aggregator\n"
                            f"  2. In LM Studio: Unload the validator model\n"
                            f"  3. In LM Studio: Reload with context={configured_context}\n"
                            f"  4. Restart the aggregator\n"
                        )
                    
                    if is_400_or_context and attempt < max_retries - 1:
                        backoff_time = min(2 ** attempt, 16)
                        logger.warning(
                            f"Validator: HTTP error (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                            f"Retrying in {backoff_time}s..."
                        )
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        # Final retry or non-recoverable error
                        logger.error(f"Validator: Failed to generate validation after {attempt + 1} attempts: {e}")
                        
                        # Provide context-specific error message
                        if "context" in error_msg.lower():
                            summary = "LM Studio context window mismatch - check logs"
                        else:
                            summary = "Internal error"
                        
                        # Notify task completed (failed but still completed)
                        if self.task_tracking_callback:
                            self.task_tracking_callback("completed", task_id)
                        
                        return ValidationResult(
                            submission_id=submission.submission_id,
                            decision="reject",
                            reasoning=f"Quality assessment error: {e}",
                            summary=summary,
                            json_valid=False
                        )
                        
                except Exception as e:
                    logger.error(f"Validator: Unexpected error during validation: {e}")
                    # Notify task completed (failed but still completed)
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return ValidationResult(
                        submission_id=submission.submission_id,
                        decision="reject",
                        reasoning=f"Quality assessment error: {e}",
                        summary="Internal error",
                        json_valid=False
                    )
            
            # Extract content
            if not response or not response.get("choices"):
                logger.error("Validator: No choices in response")
                return ValidationResult(
                    submission_id=submission.submission_id,
                    decision="reject",
                    reasoning="Validation LLM error: no response",
                    summary="Internal error",
                    json_valid=False
                )
            
            # Extract content from either 'content' or 'reasoning' field
            # Some reasoning models (e.g., DeepSeek R1, certain GPT variants) output JSON in 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            
            # Cache model config on first successful API call (only relevant for LM Studio)
            try:
                await lm_studio_client.cache_model_load_config(self.model_name, {
                    "context_length": context_allocator.validator_context_window,
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
                # Attempt conversational retry before final rejection
                logger.info("Validator: Initial JSON parse failed, attempting conversational retry")
                logger.debug(f"Parse error: {error}")
                
                retry_prompt = (
                    "Your previous response could not be parsed as valid JSON.\n\n"
                    f"PARSE ERROR: {error}\n\n"
                    "Please provide the same validation decision in valid JSON format:\n"
                    "{\n"
                    '  "decision": "accept" or "reject",\n'
                    '  "reasoning": "your reasoning here",\n'
                    '  "summary": "brief summary here"\n'
                    "}\n\n"
                    "CRITICAL: Properly escape all backslashes (use \\\\) and quotes (use \\\").\n"
                    "Respond with ONLY the JSON object, no markdown, no explanation."
                )
                
                try:
                    # CRITICAL FIX: Truncate failed output to prevent context overflow during retry
                    max_failed_output_chars = 2000  # ~500 tokens - enough for error context
                    if len(llm_output) > max_failed_output_chars:
                        failed_output_preview = llm_output[:max_failed_output_chars] + "\n[...output truncated for retry...]"
                    else:
                        failed_output_preview = llm_output
                    
                    # Calculate if conversation fits in context window
                    prompt_tokens = count_tokens(prompt)
                    preview_tokens = count_tokens(failed_output_preview)
                    retry_prompt_tokens = count_tokens(retry_prompt)
                    conversation_tokens = prompt_tokens + preview_tokens + retry_prompt_tokens
                    max_input = rag_config.get_available_input_tokens(context_allocator.validator_context_window, rag_config.validator_max_output_tokens)
                    
                    if conversation_tokens > max_input:
                        # Too large - just retry with original prompt
                        logger.warning(
                            f"Validator: Retry conversation too large ({conversation_tokens} > {max_input}), "
                            f"using simple retry without conversation context"
                        )
                        retry_response = await api_client_manager.generate_completion(
                            task_id=f"{task_id}_retry",
                            role_id=self.role_id,
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=rag_config.validator_max_output_tokens
                        )
                    else:
                        # Build conversation with truncated failed output
                        retry_response = await api_client_manager.generate_completion(
                            task_id=f"{task_id}_retry",  # Track retry attempt
                            role_id=self.role_id,
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": failed_output_preview},
                                {"role": "user", "content": retry_prompt}
                            ],
                            temperature=0.0,  # Deterministic JSON formatting
                            max_tokens=rag_config.validator_max_output_tokens  # Respect max_tokens on retry
                        )
                    
                    if retry_response.get("choices"):
                        call_metadata = api_client_manager.extract_call_metadata(retry_response)
                        retry_output = retry_response["choices"][0]["message"]["content"]
                        
                        try:
                            parsed = parse_json(retry_output)
                            valid = True
                            error = None
                            logger.info("Validator: Conversational retry succeeded!")
                            llm_output = retry_output  # Use retry output
                        except Exception as parse_error:
                            valid = False
                            parsed = None
                            error = str(parse_error)
                            logger.warning(f"Validator: Retry failed - {error}")
                    else:
                        logger.warning("Validator: Retry request returned no choices")
                except Exception as e:
                    logger.error(f"Validator: Retry request failed - {e}")
                
                # If STILL invalid after retry, return error ValidationResult
                if not valid:
                    logger.warning(f"Validator: Invalid JSON - {error}")
                    return ValidationResult(
                        submission_id=submission.submission_id,
                        decision="reject",
                        reasoning=f"Validator JSON error: {error}",
                        summary="Validator output error",
                        json_valid=False
                    )
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            # Extract summary with fallback for rejections
            # Schema requires summary for rejections, but some models may not provide it
            decision = parsed["decision"]
            summary = parsed.get("summary", "").strip()
            
            # For rejections, if summary is missing/empty, use reasoning as fallback
            if decision == "reject" and not summary:
                summary = parsed["reasoning"][:750]
                logger.debug(f"Validator: No summary provided for rejection, using reasoning as fallback")
            
            # Create validation result
            result = ValidationResult(
                submission_id=submission.submission_id,
                decision=decision,
                reasoning=parsed["reasoning"],
                summary=summary,
                contradiction_check_passed=True,
                json_valid=True,
                metadata={"llm_call": call_metadata}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return ValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"Quality assessment error: {e}",
                summary="Internal error",
                json_valid=False
            )
    
    # =========================================================================
    # BATCH VALIDATION METHODS
    # =========================================================================
    
    async def validate_batch(self, submissions: List[Submission]) -> List[ValidationResult]:
        """
        Validate 1-3 submissions at once using batch-specific prompts.
        
        - 1 submission: uses existing single prompt
        - 2 submissions: uses dual prompt (independent assessment + redundancy check)
        - 3 submissions: uses triple prompt (independent assessment + redundancy check)
        
        Each submission is first checked individually for contradiction issues.
        Then all are evaluated together for quality and intra-batch redundancy.
        
        Args:
            submissions: List of 1-3 submissions to validate
            
        Returns:
            List of ValidationResult in same order as input submissions
        """
        if not submissions:
            return []
        
        # For single submission, use existing method
        if len(submissions) == 1:
            result = await self.validate_submission(submissions[0])
            return [result]
        
        logger.info(f"Batch validation: Processing {len(submissions)} submissions")
        
        # Pre-checks for each submission (contradiction check)
        pre_check_results = []
        valid_submissions = []
        valid_indices = []
        
        for i, submission in enumerate(submissions):
            pre_check_result = await self._run_pre_checks(submission)
            if pre_check_result is not None:
                # Pre-check failed - add to results directly
                pre_check_results.append((i, pre_check_result))
                logger.info(f"Batch validation: Submission {i+1} failed pre-check")
            else:
                # Pre-check passed - will be evaluated by LLM
                valid_submissions.append(submission)
                valid_indices.append(i)
        
        # If all submissions failed pre-checks, return those results
        if not valid_submissions:
            results = [None] * len(submissions)
            for i, result in pre_check_results:
                results[i] = result
            return results
        
        # Quality assessment via LLM for submissions that passed pre-checks
        llm_results = await self._assess_batch_quality(valid_submissions)
        
        # Combine pre-check failures and LLM results in original order
        results = [None] * len(submissions)
        for i, result in pre_check_results:
            results[i] = result
        for idx, llm_idx in enumerate(valid_indices):
            results[llm_idx] = llm_results[idx]
        
        return results
    
    async def _run_pre_checks(self, submission: Submission) -> Optional[ValidationResult]:
        """
        Run contradiction pre-check on a submission.
        
        Returns:
            ValidationResult if pre-check failed, None if passed
        """
        # Contradiction check
        contradiction_passed, contradiction_reason = contradiction_checker.check_contradictions(
            submission.content
        )
        if not contradiction_passed:
            return ValidationResult(
                submission_id=submission.submission_id,
                decision="reject",
                reasoning=f"Contradiction check failed: {contradiction_reason}",
                summary=f"Contains contradictions. {contradiction_reason}",
                contradiction_check_passed=False,
                json_valid=True
            )
        
        return None  # Pre-checks passed
    
    async def _assess_batch_quality(self, submissions: List[Submission]) -> List[ValidationResult]:
        """
        Assess quality of multiple submissions using batch prompts.
        
        Args:
            submissions: List of 2-3 submissions that passed pre-checks
            
        Returns:
            List of ValidationResult in same order
        """
        try:
            batch_size = len(submissions)
            logger.info(f"Batch quality assessment: {batch_size} submissions")
            
            # If only 1 submission remains after pre-checks, use single submission path
            if batch_size == 1:
                result = await self._assess_quality(submissions[0])
                return [result]
            
            # Get appropriate prompts based on batch size
            if batch_size == 2:
                system_prompt = get_validator_dual_system_prompt()
                json_schema = get_validator_dual_json_schema()
                build_prompt = build_validator_dual_prompt
            else:  # batch_size == 3
                system_prompt = get_validator_triple_system_prompt()
                json_schema = get_validator_triple_json_schema()
                build_prompt = build_validator_triple_prompt
            
            # Get context
            shared_training_content = await shared_training_memory.get_all_content()
            
            # CRITICAL FIX: Properly account for batch-specific overhead
            # The context allocator expects single submission format, but batch prompts have:
            # - "SUBMISSION 1 TO VALIDATE:\n", "SUBMISSION 2 TO VALIDATE:\n", etc.
            # - Extra separators between submissions
            # Build batch-formatted content that matches actual prompt structure so token counting is accurate
            
            # Actual prompt format: "SUBMISSION 1 TO VALIDATE:\ncontent1\n---\nSUBMISSION 2 TO VALIDATE:\ncontent2"
            batch_formatted_parts = []
            for i, s in enumerate(submissions, 1):
                batch_formatted_parts.append(f"SUBMISSION {i} TO VALIDATE:\n{s.content}")
            batch_formatted_content = "\n---\n".join(batch_formatted_parts)
            
            allocation = await context_allocator.allocate_validator_context(
                user_prompt=self.user_prompt,
                json_schema=json_schema,
                system_prompt=system_prompt,
                shared_training_content=shared_training_content,
                user_files_content=self.user_files_content,
                submission_content=batch_formatted_content,
                chunk_size=self.chunk_size
            )
            
            # Build prompt
            rag_evidence = ""
            if allocation["rag_context"]:
                rag_evidence = allocation["rag_context"].text
            
            submission_contents = [s.content for s in submissions]
            prompt = build_prompt(
                self.user_prompt,
                submission_contents,
                allocation["direct"],
                rag_evidence
            )
            
            # Verify prompt size
            from backend.shared.utils import count_tokens
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(context_allocator.validator_context_window, rag_config.validator_max_output_tokens)
            
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"Batch validator: Prompt ({actual_prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed_tokens} tokens). Rejecting entire batch."
                )
                return [
                    ValidationResult(
                        submission_id=s.submission_id,
                        decision="reject",
                        reasoning="Internal error: Batch prompt too large",
                        summary="Internal context overflow error",
                        json_valid=False
                    )
                    for s in submissions
                ]
            
            logger.debug(f"Batch validator prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Generate validation using api_client_manager for boost support
            call_metadata = {}
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=rag_config.validator_max_output_tokens
            )
            call_metadata = api_client_manager.extract_call_metadata(response)
            
            if not response or not response.get("choices"):
                logger.error("Batch validator: No choices in response")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return [
                    ValidationResult(
                        submission_id=s.submission_id,
                        decision="reject",
                        reasoning="Validation LLM error: no response",
                        summary="Internal error",
                        json_valid=False
                    )
                    for s in submissions
                ]
            
            # Extract content
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            
            # Parse JSON
            try:
                parsed = parse_json(llm_output)
            except Exception as e:
                logger.warning(f"Batch validator: JSON parse failed: {e}")
                # Attempt conversational retry
                parsed, retry_call_metadata = await self._retry_batch_json_parse(prompt, llm_output, batch_size, task_id)
                if retry_call_metadata:
                    call_metadata = retry_call_metadata
                if parsed is None:
                    return [
                        ValidationResult(
                            submission_id=s.submission_id,
                            decision="reject",
                            reasoning=f"Validator JSON error: {e}",
                            summary="Validator output error",
                            json_valid=False,
                            metadata={"llm_call": call_metadata}
                        )
                        for s in submissions
                    ]
            
            # Extract decisions from parsed response
            decisions_list = parsed.get("decisions", [])
            
            if len(decisions_list) != batch_size:
                logger.error(
                    f"Batch validator: Expected {batch_size} decisions, got {len(decisions_list)}"
                )
                return [
                    ValidationResult(
                        submission_id=s.submission_id,
                        decision="reject",
                        reasoning=f"Invalid batch response: wrong number of decisions",
                        summary="Validator response format error",
                        json_valid=False
                    )
                    for s in submissions
                ]
            
            # Verify submission_number fields match expected order
            for i in range(batch_size):
                expected_number = i + 1  # 1-indexed
                actual_number = decisions_list[i].get("submission_number")
                if actual_number != expected_number:
                    logger.error(
                        f"Batch validator: Decision order mismatch at position {i}: "
                        f"submission_number={actual_number}, expected {expected_number}. "
                        f"LLM returned decisions out of order."
                    )
                    # Reject entire batch for safety
                    return [
                        ValidationResult(
                            submission_id=s.submission_id,
                            decision="reject",
                            reasoning=f"Validator response format error: decisions returned out of order",
                            summary="Internal error - decision ordering mismatch",
                            json_valid=False
                        )
                        for s in submissions
                    ]
            
            # Create validation results using verified order
            results = []
            for i, submission in enumerate(submissions):
                decision_data = decisions_list[i]
                
                # Extract summary with fallback for rejections
                decision = decision_data.get("decision", "reject")
                reasoning = decision_data.get("reasoning", "")
                summary = decision_data.get("summary", "").strip()
                
                # For rejections, if summary is missing/empty, use reasoning as fallback
                if decision == "reject" and not summary:
                    summary = reasoning[:750]
                    logger.debug(f"Batch validator: No summary provided for rejection {i+1}, using reasoning as fallback")
                
                results.append(ValidationResult(
                    submission_id=submission.submission_id,
                    decision=decision,
                    reasoning=reasoning,
                    summary=summary,
                    contradiction_check_passed=True,
                    json_valid=True,
                    metadata={"llm_call": call_metadata}
                ))
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            # Log batch results
            accept_count = sum(1 for r in results if r.decision == "accept")
            logger.info(f"Batch validation complete: {accept_count}/{batch_size} accepted")
            
            return results
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"Batch quality assessment failed: {e}", exc_info=True)
            return [
                ValidationResult(
                    submission_id=s.submission_id,
                    decision="reject",
                    reasoning=f"Quality assessment error: {e}",
                    summary="Internal error",
                    json_valid=False
                )
                for s in submissions
            ]
    
    async def _retry_batch_json_parse(
        self, 
        original_prompt: str, 
        failed_output: str, 
        batch_size: int,
        task_id: str
    ) -> tuple[Optional[Dict], Dict[str, Any]]:
        """
        Attempt conversational retry for batch JSON parsing.
        
        Args:
            original_prompt: The original validation prompt
            failed_output: The failed LLM output
            batch_size: Number of submissions in batch
            task_id: Task ID for tracking retry attempt
        
        Returns:
            Tuple of (parsed JSON dict if successful, call metadata dict)
        """
        logger.info("Batch validator: Attempting conversational retry for JSON parse")
        
        # Build example format based on batch size
        if batch_size == 2:
            example_format = '''
{
  "decisions": [
    {"submission_number": 1, "decision": "accept", "reasoning": "...", "summary": ""},
    {"submission_number": 2, "decision": "reject", "reasoning": "...", "summary": "..."}
  ]
}'''
        else:
            example_format = '''
{
  "decisions": [
    {"submission_number": 1, "decision": "accept", "reasoning": "...", "summary": ""},
    {"submission_number": 2, "decision": "reject", "reasoning": "...", "summary": "..."},
    {"submission_number": 3, "decision": "accept", "reasoning": "...", "summary": ""}
  ]
}'''
        
        retry_prompt = (
            "Your previous response could not be parsed as valid JSON.\n\n"
            f"Please provide the same validation decisions in valid JSON format:{example_format}\n\n"
            "CRITICAL: Properly escape all backslashes (use \\\\) and quotes (use \\\").\n"
            "Respond with ONLY the JSON object, no markdown, no explanation."
        )
        
        try:
            call_metadata = {}
            # CRITICAL FIX: Truncate failed output to prevent context overflow during retry
            max_failed_output_chars = 2000  # ~500 tokens - enough for error context
            if len(failed_output) > max_failed_output_chars:
                failed_output_preview = failed_output[:max_failed_output_chars] + "\n[...output truncated for retry...]"
            else:
                failed_output_preview = failed_output
            
            # Calculate if conversation fits in context window
            from backend.shared.utils import count_tokens
            prompt_tokens = count_tokens(original_prompt)
            preview_tokens = count_tokens(failed_output_preview)
            retry_prompt_tokens = count_tokens(retry_prompt)
            conversation_tokens = prompt_tokens + preview_tokens + retry_prompt_tokens
            max_input = rag_config.get_available_input_tokens(context_allocator.validator_context_window, rag_config.validator_max_output_tokens)
            
            if conversation_tokens > max_input:
                # Too large - just retry with original prompt
                logger.warning(
                    f"Batch validator: Retry conversation too large ({conversation_tokens} > {max_input}), "
                    f"using simple retry without conversation context"
                )
                retry_response = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_batch_retry",
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[{"role": "user", "content": original_prompt}],
                    temperature=0.0,
                    max_tokens=rag_config.validator_max_output_tokens
                )
            else:
                # Build conversation with truncated failed output
                retry_response = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_batch_retry",  # Track batch retry attempt
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": original_prompt},
                        {"role": "assistant", "content": failed_output_preview},
                        {"role": "user", "content": retry_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=rag_config.validator_max_output_tokens  # Respect max_tokens on retry
                )
            call_metadata = api_client_manager.extract_call_metadata(retry_response)
            
            if retry_response.get("choices"):
                retry_output = retry_response["choices"][0]["message"]["content"]
                parsed = parse_json(retry_output)
                logger.info("Batch validator: Conversational retry succeeded!")
                return parsed, call_metadata
        except Exception as e:
            logger.warning(f"Batch validator: Retry failed - {e}")
        
        return None, {}
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for single submission."""
        return get_validator_system_prompt()
    
    def _get_json_schema(self) -> str:
        """Get JSON schema for single submission."""
        return get_validator_json_schema()
    
    # =========================================================================
    # CLEANUP REVIEW METHODS
    # =========================================================================
    
    async def perform_cleanup_review(self) -> Optional[Dict]:
        """
        Perform a cleanup review of the accepted submissions database.
        
        Reviews all accepted submissions and identifies if any ONE submission
        should be removed due to redundancy, contradictions, or other issues.
        
        CRITICAL: This method NEVER skips due to prompt size. If submissions are
        too large for direct injection, it uses RAG to retrieve relevant context.
        
        Returns:
            Dict with {"submission_number": int, "reasoning": str} if removal proposed,
            None if no removal needed
        """
        try:
            logger.info("=" * 60)
            logger.info("CLEANUP DEBUG: ========== PHASE 1: CLEANUP REVIEW START ==========")
            logger.info("=" * 60)
            
            # Get all submissions formatted
            all_submissions = await shared_training_memory.get_all_content_formatted()
            submission_count = await shared_training_memory.get_insights_count()
            
            logger.info(f"CLEANUP DEBUG: Database has {submission_count} submissions")
            logger.info(f"CLEANUP DEBUG: Formatted submissions length: {len(all_submissions)} chars")
            
            if not all_submissions.strip():
                logger.info("CLEANUP DEBUG: No submissions in database, skipping cleanup review")
                logger.info("Cleanup review: No submissions in database, skipping")
                return None
            
            # Log a preview of the formatted submissions (first 1000 chars)
            logger.debug(f"CLEANUP DEBUG: Formatted submissions preview:\n{all_submissions[:1000]}...")
            
            # Use context allocator to handle large databases via RAG
            # CRITICAL: This NEVER skips - it uses RAG when content is too large
            context_result = await context_allocator.allocate_cleanup_review_context(
                user_prompt=self.user_prompt,
                json_schema=get_cleanup_review_json_schema(),
                system_prompt=get_cleanup_review_system_prompt(),
                all_submissions_formatted=all_submissions,
                user_files_content=self.user_files_content or {}
            )
            
            direct_context = context_result["direct"]
            rag_context = context_result["rag_context"]
            submissions_ragged = context_result["submissions_ragged"]
            user_files_ragged = context_result.get("user_files_ragged", False)
            
            if submissions_ragged:
                logger.info(
                    f"CLEANUP DEBUG: Submissions too large for direct injection - using RAG retrieval. "
                    f"RAG evidence available: {rag_context is not None and rag_context.text}"
                )
            else:
                logger.info("CLEANUP DEBUG: All submissions fit in direct injection")
            
            # Build user files context if not using RAG
            user_files_context = ""
            if not user_files_ragged and self.user_files_content:
                user_files_context = "\n\n".join([
                    f"--- {name} ---\n{content}"
                    for name, content in self.user_files_content.items()
                ])
                logger.info(f"CLEANUP DEBUG: User files context length: {len(user_files_context)} chars")
            
            # Build the cleanup review prompt
            # If submissions were RAGed, we pass empty string for all_submissions and use rag_evidence
            if submissions_ragged:
                # Get RAG evidence text
                rag_evidence = rag_context.text if rag_context and rag_context.text else ""
                
                # Build prompt with RAG evidence instead of all submissions
                prompt = build_cleanup_review_prompt(
                    user_prompt=self.user_prompt,
                    all_submissions_formatted="[Database too large for direct injection - see retrieved evidence below]",
                    context=user_files_context,
                    rag_evidence=rag_evidence
                )
                logger.info(f"CLEANUP DEBUG: Built cleanup review prompt WITH RAG evidence, length: {len(prompt)} chars")
            else:
                # Direct injection - all submissions fit
                prompt = build_cleanup_review_prompt(
                    user_prompt=self.user_prompt,
                    all_submissions_formatted=all_submissions,
                    context=user_files_context
                )
                logger.info(f"CLEANUP DEBUG: Built cleanup review prompt with direct injection, length: {len(prompt)} chars")
            
            # Verify prompt size (should now always fit due to RAG handling)
            from backend.shared.utils import count_tokens
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(context_allocator.validator_context_window, rag_config.validator_max_output_tokens)
            
            logger.info(f"CLEANUP DEBUG: Final prompt tokens: {actual_prompt_tokens}, Max allowed: {max_allowed_tokens}")
            logger.info(f"CLEANUP DEBUG: Context window: {context_allocator.validator_context_window}")
            
            # This should no longer happen due to RAG handling, but log if it does
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"CLEANUP DEBUG: UNEXPECTED - Prompt still too large after RAG ({actual_prompt_tokens} > {max_allowed_tokens}). "
                    f"This indicates a bug in context allocation. Proceeding anyway."
                )
            
            logger.info(f"CLEANUP DEBUG: Prompt size OK, sending to LLM model: {self.model_name}")
            logger.info(f"Cleanup review: Analyzing {submission_count} submissions")
            
            # Generate task ID for tracking and boost support
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Generate cleanup review using api_client_manager for boost support
            logger.info(f"CLEANUP DEBUG: Calling LLM with max_tokens={rag_config.validator_max_output_tokens}, temperature=0.0, task_id={task_id}")
            try:
                response = await api_client_manager.generate_completion(
                    task_id=task_id,
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Deterministic cleanup decisions
                    max_tokens=rag_config.validator_max_output_tokens
                )
            except Exception as api_error:
                logger.error(f"CLEANUP DEBUG: API call failed: {api_error}")
                # Notify task completed even on failure
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                raise
            
            # Notify task completed (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"CLEANUP DEBUG: LLM response received: {response is not None}")
            
            if not response or not response.get("choices"):
                logger.error("CLEANUP DEBUG: NO RESPONSE FROM LLM - response is None or has no choices")
                logger.error(f"CLEANUP DEBUG: Raw response object: {response}")
                logger.error("Cleanup review: No response from LLM")
                return None
            
            logger.info(f"CLEANUP DEBUG: Response has {len(response.get('choices', []))} choices")
            
            # Extract content from either 'content' or 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            
            logger.info(f"CLEANUP DEBUG: LLM output length: {len(llm_output)} chars")
            logger.info(f"CLEANUP DEBUG: Raw LLM output (first 1000 chars):\n{llm_output[:1000]}")
            if len(llm_output) > 1000:
                logger.debug(f"CLEANUP DEBUG: Full LLM output:\n{llm_output}")
            
            # Parse JSON
            try:
                logger.info("CLEANUP DEBUG: Attempting to parse JSON from LLM output...")
                parsed = parse_json(llm_output)
                logger.info(f"CLEANUP DEBUG: JSON parsed successfully: {parsed}")
            except Exception as e:
                logger.warning(f"CLEANUP DEBUG: JSON PARSE FAILED: {e}")
                logger.warning(f"CLEANUP DEBUG: Raw output that failed to parse:\n{llm_output}")
                logger.warning(f"Cleanup review: JSON parse failed: {e}")
                return None
            
            # Check if removal is recommended
            should_remove = parsed.get("should_remove", False)
            submission_number = parsed.get("submission_number")
            reasoning = parsed.get("reasoning", "")
            
            logger.info(f"CLEANUP DEBUG: Parsed fields - should_remove={should_remove}, submission_number={submission_number}")
            logger.info(f"CLEANUP DEBUG: Reasoning (first 300 chars): {reasoning[:300]}")
            
            if not should_remove:
                logger.info(f"CLEANUP DEBUG: NO REMOVAL NEEDED - should_remove is False")
                logger.info(f"Cleanup review: No removal needed - {reasoning[:200]}")
                return None
            
            if submission_number is None:
                logger.warning("CLEANUP DEBUG: INVALID RESPONSE - should_remove=true but no submission_number provided")
                logger.warning("Cleanup review: should_remove=true but no submission_number provided")
                return None
            
            logger.info(f"CLEANUP DEBUG: REMOVAL PROPOSED - submission #{submission_number}")
            logger.info(
                f"Cleanup review: Proposing removal of submission #{submission_number} - "
                f"{reasoning[:200]}..."
            )
            
            return {
                "submission_number": submission_number,
                "reasoning": reasoning
            }
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"CLEANUP DEBUG: EXCEPTION in perform_cleanup_review: {e}", exc_info=True)
            logger.error(f"Cleanup review failed: {e}", exc_info=True)
            return None
    
    async def validate_removal(
        self,
        submission_number: int,
        submission_content: str,
        removal_reasoning: str
    ) -> bool:
        """
        Validate a proposed submission removal.
        
        CRITICAL: This method NEVER skips or fails due to prompt size. If submissions are
        too large for direct injection, it uses RAG to retrieve relevant context.
        
        Args:
            submission_number: Number of submission proposed for removal
            submission_content: Content of the submission
            removal_reasoning: Reasoning for proposed removal
            
        Returns:
            True if removal should proceed, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("CLEANUP DEBUG: ========== PHASE 2: REMOVAL VALIDATION START ==========")
            logger.info("=" * 60)
            
            logger.info(f"CLEANUP DEBUG: Validating removal of submission #{submission_number}")
            logger.info(f"CLEANUP DEBUG: Submission content length: {len(submission_content)} chars")
            logger.info(f"CLEANUP DEBUG: Removal reasoning length: {len(removal_reasoning)} chars")
            logger.debug(f"CLEANUP DEBUG: Submission content preview:\n{submission_content[:500]}...")
            logger.debug(f"CLEANUP DEBUG: Removal reasoning:\n{removal_reasoning}")
            
            # Get all submissions for context
            all_submissions = await shared_training_memory.get_all_content_formatted()
            logger.info(f"CLEANUP DEBUG: Full database context length: {len(all_submissions)} chars")
            
            # Use context allocator to handle large databases via RAG
            # CRITICAL: This NEVER skips - it uses RAG when content is too large
            context_result = await context_allocator.allocate_cleanup_review_context(
                user_prompt=self.user_prompt,
                json_schema=get_removal_validation_json_schema(),
                system_prompt=get_removal_validation_system_prompt(),
                all_submissions_formatted=all_submissions,
                user_files_content=self.user_files_content or {},
                submission_proposed_for_removal=submission_content
            )
            
            direct_context = context_result["direct"]
            rag_context = context_result["rag_context"]
            submissions_ragged = context_result["submissions_ragged"]
            user_files_ragged = context_result.get("user_files_ragged", False)
            
            if submissions_ragged:
                logger.info(
                    f"CLEANUP DEBUG: Submissions too large for direct injection - using RAG retrieval. "
                    f"RAG evidence available: {rag_context is not None and rag_context.text}"
                )
            else:
                logger.info("CLEANUP DEBUG: All submissions fit in direct injection")
            
            # Build the removal validation prompt
            # If submissions were RAGed, we pass empty string for all_submissions and use rag_evidence
            if submissions_ragged:
                # Get RAG evidence text
                rag_evidence = rag_context.text if rag_context and rag_context.text else ""
                
                prompt = build_removal_validation_prompt(
                    user_prompt=self.user_prompt,
                    submission_number=submission_number,
                    submission_content=submission_content,
                    removal_reasoning=removal_reasoning,
                    all_submissions_formatted="[Database too large for direct injection - see retrieved evidence below]",
                    rag_evidence=rag_evidence
                )
                logger.info(f"CLEANUP DEBUG: Built removal validation prompt WITH RAG evidence, length: {len(prompt)} chars")
            else:
                prompt = build_removal_validation_prompt(
                    user_prompt=self.user_prompt,
                    submission_number=submission_number,
                    submission_content=submission_content,
                    removal_reasoning=removal_reasoning,
                    all_submissions_formatted=all_submissions
                )
                logger.info(f"CLEANUP DEBUG: Built removal validation prompt with direct injection, length: {len(prompt)} chars")
            
            # Verify prompt size (should now always fit due to RAG handling)
            from backend.shared.utils import count_tokens
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(context_allocator.validator_context_window, rag_config.validator_max_output_tokens)
            
            logger.info(f"CLEANUP DEBUG: Final prompt tokens: {actual_prompt_tokens}, Max allowed: {max_allowed_tokens}")
            
            # This should no longer happen due to RAG handling, but log if it does
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"CLEANUP DEBUG: UNEXPECTED - Prompt still too large after RAG ({actual_prompt_tokens} > {max_allowed_tokens}). "
                    f"This indicates a bug in context allocation. Proceeding anyway."
                )
            
            logger.info(f"CLEANUP DEBUG: Prompt size OK, sending to LLM model: {self.model_name}")
            logger.info(f"Removal validation: Validating proposed removal of submission #{submission_number}")
            
            # Generate task ID for tracking and boost support
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Generate validation using api_client_manager for boost support
            logger.info(f"CLEANUP DEBUG: Calling LLM with max_tokens={rag_config.validator_max_output_tokens}, temperature=0.0, task_id={task_id}")
            try:
                response = await api_client_manager.generate_completion(
                    task_id=task_id,
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Deterministic removal validation
                    max_tokens=rag_config.validator_max_output_tokens
                )
            except Exception as api_error:
                logger.error(f"CLEANUP DEBUG: API call failed: {api_error}")
                # Notify task completed even on failure
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                raise
            
            # Notify task completed (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"CLEANUP DEBUG: LLM response received: {response is not None}")
            
            if not response or not response.get("choices"):
                logger.error("CLEANUP DEBUG: NO RESPONSE FROM LLM - response is None or has no choices")
                logger.error(f"CLEANUP DEBUG: Raw response object: {response}")
                logger.error("Removal validation: No response from LLM")
                return False
            
            logger.info(f"CLEANUP DEBUG: Response has {len(response.get('choices', []))} choices")
            
            # Extract content from either 'content' or 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            
            logger.info(f"CLEANUP DEBUG: LLM output length: {len(llm_output)} chars")
            logger.info(f"CLEANUP DEBUG: Raw LLM output (first 1000 chars):\n{llm_output[:1000]}")
            if len(llm_output) > 1000:
                logger.debug(f"CLEANUP DEBUG: Full LLM output:\n{llm_output}")
            
            # Parse JSON
            try:
                logger.info("CLEANUP DEBUG: Attempting to parse JSON from LLM output...")
                parsed = parse_json(llm_output)
                logger.info(f"CLEANUP DEBUG: JSON parsed successfully: {parsed}")
            except Exception as e:
                logger.warning(f"CLEANUP DEBUG: JSON PARSE FAILED: {e}")
                logger.warning(f"CLEANUP DEBUG: Raw output that failed to parse:\n{llm_output}")
                logger.warning(f"Removal validation: JSON parse failed: {e}")
                return False
            
            decision = parsed.get("decision", "reject")
            reasoning = parsed.get("reasoning", "")
            
            logger.info(f"CLEANUP DEBUG: Parsed fields - decision={decision}")
            logger.info(f"CLEANUP DEBUG: Reasoning (first 300 chars): {reasoning[:300]}")
            
            if decision == "accept":
                logger.info(f"CLEANUP DEBUG: REMOVAL APPROVED for submission #{submission_number}")
                logger.info(
                    f"Removal validation: APPROVED removal of submission #{submission_number} - "
                    f"{reasoning[:200]}..."
                )
                return True
            else:
                logger.info(f"CLEANUP DEBUG: REMOVAL REJECTED for submission #{submission_number}")
                logger.info(
                    f"Removal validation: REJECTED removal of submission #{submission_number} - "
                    f"{reasoning[:200]}..."
                )
                return False
                
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"CLEANUP DEBUG: EXCEPTION in validate_removal: {e}", exc_info=True)
            logger.error(f"Removal validation failed: {e}", exc_info=True)
            return False
    
    async def get_submission_content(self, submission_number: int) -> Optional[str]:
        """
        Get the content of a specific submission by number.
        
        Args:
            submission_number: The submission number to retrieve
            
        Returns:
            The submission content if found, None otherwise
        """
        return await shared_training_memory.get_submission_content(submission_number)

