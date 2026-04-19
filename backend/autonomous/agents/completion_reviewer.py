"""
Completion Reviewer Agent - Assesses brainstorm completeness.
Implements SPECIAL SELF-VALIDATION MODE where the same model validates its own assessment.

CONTEXT HANDLING:
- Uses DIRECT INJECTION FIRST, RAG SECOND principle
- Tries to inject full brainstorm database for accurate exhaustion assessment
- Falls back to RAG if database doesn't fit in context
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, Tuple, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.shared.models import CompletionReviewResult, CompletionSelfValidationResult
from backend.autonomous.prompts.completion_prompts import (
    build_completion_review_prompt,
    build_completion_self_validation_prompt
)
from backend.autonomous.memory.autonomous_rejection_logs import autonomous_rejection_logs
from backend.autonomous.core.autonomous_rag_manager import autonomous_rag_manager

logger = logging.getLogger(__name__)


class CompletionReviewerAgent:
    """
    Agent that assesses brainstorm completeness.
    Uses SPECIAL SELF-VALIDATION MODE - same model validates its own assessment.
    
    Context handling:
    - Direct injects full brainstorm database when it fits
    - Uses RAG retrieval when database exceeds context limits
    - NEVER truncates (would lose critical information for exhaustion assessment)
    """
    
    def __init__(
        self,
        model_id: str,
        context_window: int = 131072,
        max_output_tokens: int = 25000
    ):
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "autonomous_completion_reviewer"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"agg_sub1_{self.task_sequence:03d}"
    
    def _calculate_available_context(self) -> int:
        """Calculate available tokens for brainstorm database content."""
        # Reserve for: output, system prompts, JSON schema, user prompt, topic prompt, etc.
        reserved_tokens = self.max_output_tokens + 10000  # Generous reserve for prompts
        available = self.context_window - reserved_tokens
        return max(available, 20000)  # Minimum 20k for brainstorm content
    
    async def review_completion(
        self,
        user_research_prompt: str,
        topic_id: str,
        topic_prompt: str,
        brainstorm_database: str,
        submission_count: int
    ) -> Tuple[Optional[CompletionReviewResult], bool]:
        """
        Perform completion review with self-validation.
        
        Args:
            user_research_prompt: The user's high-level research goal
            topic_id: Current brainstorm topic ID
            topic_prompt: The brainstorm topic prompt
            brainstorm_database: Full content of brainstorm database
            submission_count: Number of accepted submissions
        
        Returns:
            Tuple of (CompletionReviewResult or None, is_validated: bool)
        """
        # Prepare brainstorm context with proper direct injection / RAG handling
        brainstorm_context, used_rag = await self._prepare_brainstorm_context(
            topic_id,
            brainstorm_database,
            user_research_prompt
        )
        
        # Step 1: Generate completion assessment
        assessment = await self._generate_assessment(
            user_research_prompt,
            topic_id,
            topic_prompt,
            brainstorm_context,
            submission_count,
            used_rag
        )
        
        if assessment is None:
            logger.error("CompletionReviewer: Failed to generate assessment")
            return None, False
        
        # Step 2: Self-validate the assessment (SPECIAL SELF-VALIDATION MODE)
        is_validated = await self._self_validate(
            user_research_prompt,
            topic_prompt,
            brainstorm_context,
            assessment,
            used_rag
        )
        
        if not is_validated:
            # Log feedback that assessment was not validated
            await autonomous_rejection_logs.add_completion_feedback(
                topic_id=topic_id,
                decision=assessment.decision,
                reasoning=f"SELF-VALIDATION FAILED: {assessment.reasoning}",
                suggested_additions=assessment.suggested_additions
            )
            logger.info("CompletionReviewer: Self-validation FAILED - defaulting to continue")
            # Return a continue_brainstorm result when self-validation fails
            return CompletionReviewResult(
                decision="continue_brainstorm",
                reasoning="Self-validation failed - continuing brainstorm to be safe",
                suggested_additions=assessment.suggested_additions
            ), False
        
        # Log successful completion feedback
        if assessment.decision == "continue_brainstorm":
            await autonomous_rejection_logs.add_completion_feedback(
                topic_id=topic_id,
                decision=assessment.decision,
                reasoning=assessment.reasoning,
                suggested_additions=assessment.suggested_additions
            )
        
        return assessment, True
    
    async def _prepare_brainstorm_context(
        self,
        topic_id: str,
        brainstorm_database: str,
        query: str
    ) -> Tuple[str, bool]:
        """
        Prepare brainstorm context using DIRECT INJECTION FIRST, RAG SECOND.
        
        Returns:
            Tuple of (context_string, used_rag_boolean)
        """
        available_tokens = self._calculate_available_context()
        db_tokens = count_tokens(brainstorm_database)
        
        if db_tokens <= available_tokens:
            # Database fits - use direct injection (preferred for exhaustion assessment)
            logger.info(f"CompletionReviewer: Direct injection of brainstorm ({db_tokens} tokens <= {available_tokens} available)")
            return brainstorm_database, False
        else:
            # Database doesn't fit - use RAG
            logger.info(f"CompletionReviewer: Using RAG for brainstorm ({db_tokens} tokens > {available_tokens} available)")
            
            # Get brainstorm via RAG manager (handles indexing and retrieval)
            context, used_rag = await autonomous_rag_manager.get_brainstorm_context(
                topic_id,
                max_tokens=available_tokens,
                query=query
            )
            
            # Add note about RAG usage for model awareness
            if used_rag:
                context = f"[NOTE: Brainstorm database is large ({db_tokens} tokens). Showing RAG-retrieved relevant sections. " \
                         f"Total submissions: {brainstorm_database.count('Submission')} or more]\n\n{context}"
            
            return context, used_rag
    
    async def _generate_assessment(
        self,
        user_research_prompt: str,
        topic_id: str,
        topic_prompt: str,
        brainstorm_context: str,
        submission_count: int,
        used_rag: bool
    ) -> Optional[CompletionReviewResult]:
        """Generate the initial completion assessment."""
        try:
            # Get previous feedback
            completion_feedback = await autonomous_rejection_logs.format_completion_feedback_for_context(topic_id)
            
            # Build prompt with prepared context
            prompt = build_completion_review_prompt(
                user_research_prompt=user_research_prompt,
                topic_prompt=topic_prompt,
                brainstorm_database=brainstorm_context,
                submission_count=submission_count,
                completion_feedback=completion_feedback
            )
            
            # Validate prompt size before sending
            prompt_tokens = count_tokens(prompt)
            max_input_tokens = self.context_window - self.max_output_tokens
            
            if prompt_tokens > max_input_tokens:
                logger.error(f"CompletionReviewer: Prompt ({prompt_tokens} tokens) exceeds input limit ({max_input_tokens})")
                return None
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"CompletionReviewer: Generating assessment with model {self.model_id} "
                       f"(prompt={prompt_tokens}t, RAG={used_rag}, task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0  # Deterministic generation - evolving context provides diversity
            )
            
            if not response:
                logger.error("CompletionReviewer: Empty response from LLM")
                return None
            
            # Extract content from response (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                logger.error("CompletionReviewer: No content in response")
                return None
            
            # Parse JSON using central utility (handles sanitization + parsing + array handling)
            try:
                data = parse_json(content)
                
                # Validate required fields
                decision = data.get("decision", "").lower()
                if decision not in ["continue_brainstorm", "write_paper"]:
                    logger.error(f"CompletionReviewer: Invalid decision: {decision}")
                    return None
                
                # Notify task completed successfully
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                
                return CompletionReviewResult(
                    decision=decision,
                    reasoning=data.get("reasoning", "No reasoning provided"),
                    suggested_additions=data.get("suggested_additions", "")
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"CompletionReviewer: JSON parse error: {e}")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
                
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"CompletionReviewer: Error generating assessment: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
    
    async def _self_validate(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_context: str,
        assessment: CompletionReviewResult,
        used_rag: bool
    ) -> bool:
        """
        SPECIAL SELF-VALIDATION MODE.
        The same model validates its own completion assessment.
        
        Uses the same brainstorm context (direct or RAG) as the assessment phase.
        """
        try:
            # Convert assessment to dict
            original_assessment = {
                "decision": assessment.decision,
                "reasoning": assessment.reasoning,
                "suggested_additions": assessment.suggested_additions
            }
            
            # Build self-validation prompt with same context
            prompt = build_completion_self_validation_prompt(
                user_research_prompt=user_research_prompt,
                topic_prompt=topic_prompt,
                brainstorm_database=brainstorm_context,  # Use prepared context (may be RAG)
                original_assessment=original_assessment
            )
            
            # Validate prompt size before sending
            prompt_tokens = count_tokens(prompt)
            max_input_tokens = self.context_window - self.max_output_tokens
            
            if prompt_tokens > max_input_tokens:
                logger.error(f"CompletionReviewer: Self-validation prompt ({prompt_tokens} tokens) exceeds input limit")
                return False
            
            # Generate task ID for self-validation tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call SAME LLM for self-validation via api_client_manager
            logger.info(f"CompletionReviewer: Self-validating with SAME model {self.model_id} "
                       f"(prompt={prompt_tokens}t, RAG={used_rag}, task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,  # Use same role_id for self-validation
                model=self.model_id,  # SAME MODEL - critical for self-validation
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0  # Deterministic validation - evolving context provides diversity
            )
            
            if not response:
                logger.error("CompletionReviewer: Empty self-validation response")
                return False
            
            # Extract content from response (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                logger.error("CompletionReviewer: No content in self-validation response")
                return False
            
            # Log raw response for debugging
            logger.info(f"CompletionReviewer: Self-validation raw content (first 500 chars): {content[:500]}")
            
            # Parse JSON using central utility (handles sanitization + parsing + array handling + enhanced logging)
            try:
                data = parse_json(content)
                
                # Get validation result - check for string "true"/"false" as well
                validated_raw = data.get("validated", False)
                if isinstance(validated_raw, str):
                    validated = validated_raw.lower() == "true"
                    logger.warning(f"CompletionReviewer: 'validated' was a string '{validated_raw}', converted to bool: {validated}")
                else:
                    validated = bool(validated_raw)
                
                reasoning = data.get("reasoning", "No reasoning")
                
                # Notify task completed successfully
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                
                logger.info(f"CompletionReviewer: Self-validation result: validated={validated}")
                logger.info(f"CompletionReviewer: Self-validation reasoning: {reasoning[:300]}...")
                
                return validated
                
            except json.JSONDecodeError as e:
                logger.error(f"CompletionReviewer: JSON parse error in self-validation: {e}")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return False
                
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"CompletionReviewer: Error in self-validation: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return False
    
    async def check_early_completion_trigger(
        self,
        consecutive_rejections: int,
        submitter_exhaustion_signals: int
    ) -> bool:
        """
        Check if early completion review should be triggered.
        
        Args:
            consecutive_rejections: Number of consecutive rejections across all submitters
            submitter_exhaustion_signals: Number of submitters signaling exhaustion
        
        Returns:
            True if early completion review should be triggered
        """
        # Trigger on 10+ consecutive rejections
        if consecutive_rejections >= 10:
            logger.info(f"CompletionReviewer: Early trigger - {consecutive_rejections} consecutive rejections")
            return True
        
        # Trigger on 2+ submitter exhaustion signals
        if submitter_exhaustion_signals >= 2:
            logger.info(f"CompletionReviewer: Early trigger - {submitter_exhaustion_signals} submitters signaling exhaustion")
            return True
        
        return False

