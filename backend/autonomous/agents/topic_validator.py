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
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.model_output_contract import ModelOutputContractError
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.shared.models import TopicSelectionSubmission, TopicValidationResult
from backend.autonomous.prompts.topic_prompts import build_topic_validation_prompt
from backend.autonomous.agents.output_contract import (
    json_object,
    parse_with_contract_retry,
    require_choice,
    require_text,
    response_text,
)

logger = logging.getLogger(__name__)

_VALIDATION_SHAPE = """{
  "decision": "accept | reject",
  "reasoning": "string"
}"""


def _parse_validation(data: Any, *, role: str) -> Tuple[str, str, Dict[str, Any]]:
    data = json_object(data, role=role)
    decision = require_choice(
        data, "decision", ("accept", "reject"), role=role, case_insensitive=True
    )
    return decision, require_text(data, "reasoning", role=role), data


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
        context_window: int = 0,
        max_output_tokens: int = 0
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
        return rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
    
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

        Raises:
            ModelOutputContractError: the validator's reply did not match its JSON
                contract after one bounded retry. That is not a semantic decision,
                so callers must not record it as a rejection.
        """
        try:
            if override_prompt:
                prompt = override_prompt
            else:
                # Convert submission to dict for prompt
                proposed_action = {
                    "action": submission.action,
                    "topic_id": submission.topic_id,
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
                from backend.shared.solution_path.integration import with_validator_hook
                prompt = with_validator_hook(
                    prompt, getattr(self, "solution_path_manager", None)
                )
            if override_prompt:
                from backend.shared.solution_path.integration import with_validator_hook
                prompt = with_validator_hook(
                    prompt, getattr(self, "solution_path_manager", None)
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
                from backend.shared.solution_path.integration import with_validator_hook
                prompt = with_validator_hook(
                    prompt, getattr(self, "solution_path_manager", None)
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
            
            try:
                decision, reasoning, data = await parse_with_contract_retry(
                    response_text(response),
                    lambda parsed: _parse_validation(parsed, role=self.role_id),
                    prompt=prompt,
                    task_id=task_id,
                    role_id=self.role_id,
                    model=self.model_id,
                    context_window=self.context_window,
                    max_output_tokens=self.max_output_tokens,
                    response_noun="validation decision",
                    response_shape=_VALIDATION_SHAPE,
                )
            finally:
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)

            from backend.shared.solution_path.integration import enqueue_optional_update
            await enqueue_optional_update(
                data,
                getattr(self, "solution_path_manager", None),
                proposer_role="topic_validator",
                source_task_id=task_id,
                source_phase="topic_validation",
                source_decision=decision,
            )
            logger.info(f"TopicValidator: Decision={decision}")
            return TopicValidationResult(decision=decision, reasoning=reasoning)

        except FreeModelExhaustedError:
            raise
        except RetryableProviderError:
            raise
        except ModelOutputContractError:
            raise
        except Exception as e:
            if api_client_manager.is_provider_failure(e):
                raise
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

