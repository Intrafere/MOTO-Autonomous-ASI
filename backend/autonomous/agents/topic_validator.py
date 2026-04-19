"""
Topic Validator Agent - Validates topic selection decisions.

CONTEXT HANDLING:
- Uses same context as topic selector (metadata summaries)
- Validates prompt size before sending
- Truncates paper abstracts if context is too large

NO RAG BY DESIGN: Same rationale as topic selector — validates a strategic decision
using only metadata summaries (topic prompts, statuses, paper titles/abstracts).
Full content not needed for validating topic selection quality.
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens
from backend.shared.models import TopicSelectionSubmission, TopicValidationResult
from backend.autonomous.prompts.topic_prompts import build_topic_validation_prompt

logger = logging.getLogger(__name__)


class TopicValidatorAgent:
    """
    Agent that validates topic selection decisions.
    Uses same context as topic selector + the proposed action.
    
    Context handling:
    - Direct injects all metadata summaries (brainstorms, papers, proposed action)
    - Validates prompt fits in context before sending
    - Truncates paper abstracts if needed for context fit
    """
    
    def __init__(
        self,
        model_id: str,
        context_window: int = 131072,
        max_output_tokens: int = 15000
    ):
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "autonomous_topic_validator"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"agg_val_{self.task_sequence:03d}"
    
    def _calculate_max_input_tokens(self) -> int:
        """Calculate available tokens for input prompt."""
        return self.context_window - self.max_output_tokens
    
    async def validate(
        self,
        submission: TopicSelectionSubmission,
        user_research_prompt: str,
        brainstorms_summary: List[Dict[str, Any]],
        papers_summary: List[Dict[str, Any]],
        override_prompt: Optional[str] = None
    ) -> TopicValidationResult:
        """
        Validate a topic selection submission.
        
        Args:
            submission: The topic selection to validate
            user_research_prompt: The user's high-level research goal
            brainstorms_summary: List of all brainstorms with metadata
            papers_summary: List of all papers with title, abstract, word count
            override_prompt: If provided, use this prompt instead of building one
        
        Returns:
            TopicValidationResult with accept/reject decision
        """
        try:
            if override_prompt:
                prompt = override_prompt
            else:
                # Convert submission to dict for prompt
                proposed_action = {
                    "action": submission.action,
                    "topic_id": submission.topic_id,
                    "topic_ids": submission.topic_ids,
                    "topic_prompt": submission.topic_prompt,
                    "reasoning": submission.reasoning
                }
                
                # Build prompt
                prompt = build_topic_validation_prompt(
                    user_research_prompt=user_research_prompt,
                    brainstorms_summary=brainstorms_summary,
                    papers_summary=papers_summary,
                    proposed_action=proposed_action
                )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input_tokens = self._calculate_max_input_tokens()
            
            if prompt_tokens > max_input_tokens:
                if override_prompt:
                    logger.error(f"TopicValidator: Override prompt ({prompt_tokens} tokens) exceeds limit ({max_input_tokens}). Cannot truncate.")
                    return self._create_rejection("Override prompt too large for validation")
                
                logger.warning(f"TopicValidator: Prompt ({prompt_tokens} tokens) exceeds limit ({max_input_tokens}). "
                             f"Truncating paper abstracts.")
                
                truncated_papers = []
                for p in papers_summary:
                    truncated = p.copy()
                    if "abstract" in truncated and len(truncated.get("abstract", "")) > 500:
                        truncated["abstract"] = truncated["abstract"][:500] + "..."
                    truncated_papers.append(truncated)
                
                prompt = build_topic_validation_prompt(
                    user_research_prompt=user_research_prompt,
                    brainstorms_summary=brainstorms_summary,
                    papers_summary=truncated_papers,
                    proposed_action=proposed_action
                )
                
                prompt_tokens = count_tokens(prompt)
                if prompt_tokens > max_input_tokens:
                    logger.error(f"TopicValidator: Even after truncation, prompt ({prompt_tokens}) exceeds limit")
                    return self._create_rejection("Context too large for validation")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"TopicValidator: Validating topic selection with model {self.model_id} "
                       f"(prompt={prompt_tokens}t, task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0  # Deterministic validation - evolving context provides diversity
            )
            
            if not response:
                logger.error("TopicValidator: Empty response from LLM")
                return self._create_rejection("Empty response from validator")
            
            # Extract content from response (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                logger.error("TopicValidator: No content in response")
                return self._create_rejection("No content in validator response")
            
            # Parse JSON using central utility (handles sanitization + parsing + array handling)
            try:
                data = parse_json(content)
                
                # parse_json already handles array response, but keep check for safety
                if isinstance(data, list) and len(data) > 0:
                    logger.warning("TopicValidator: Model returned array, using first element")
                    data = data[0]
                
                # Validate required fields
                decision = data.get("decision", "").lower()
                reasoning = data.get("reasoning", "No reasoning provided")
                
                if decision not in ["accept", "reject"]:
                    logger.error(f"TopicValidator: Invalid decision: {decision}")
                    return self._create_rejection(f"Invalid decision format: {decision}")
                
                result = TopicValidationResult(
                    decision=decision,
                    reasoning=reasoning
                )
                
                # Notify task completed successfully
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                
                logger.info(f"TopicValidator: Decision={decision}")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"TopicValidator: JSON parse error: {e}")
                logger.debug(f"TopicValidator: Raw response: {content[:500]}...")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return self._create_rejection(f"JSON parse error: {str(e)}")
                
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"TopicValidator: Error during validation: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return self._create_rejection(f"Validation error: {str(e)}")
    
    def _create_rejection(self, reason: str) -> TopicValidationResult:
        """Create a rejection result with the given reason."""
        return TopicValidationResult(
            decision="reject",
            reasoning=f"Validation failed: {reason}"
        )

