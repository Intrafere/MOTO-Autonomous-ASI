"""
Topic Selector Agent - Selects the next research topic for brainstorming.

CONTEXT HANDLING:
- Uses DIRECT INJECTION for all context (metadata summaries are typically small)
- Validates prompt size before sending to prevent context overflow
- Truncates paper abstracts if context is too large (safe since abstracts are summaries)

NO RAG BY DESIGN: This agent makes a strategic decision about WHAT to work on next.
It only needs metadata summaries (topic prompts, statuses, paper titles/abstracts),
not full brainstorm databases or full paper content. Metadata is small enough to
direct-inject; abstract truncation is the overflow fallback.
"""
import logging
from typing import Optional, Dict, Any, List, Callable

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.model_output_contract import ModelOutputContractError
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.shared.models import TopicSelectionSubmission
from backend.shared.prompt_feedback_budget import fit_prompt_with_feedback
from backend.shared.provider_errors import ProviderContextLengthError
from backend.autonomous.prompts.topic_prompts import (
    build_topic_selection_prompt
)
from backend.autonomous.memory.autonomous_rejection_logs import autonomous_rejection_logs
from backend.autonomous.agents.output_contract import (
    json_object,
    optional_text,
    parse_with_contract_retry,
    require_choice,
    require_text,
    response_text,
)

_SELECTION_SHAPE = """{
  "action": "new_topic | continue_existing",
  "topic_id": "string (required for continue_existing)",
  "topic_prompt": "string (required for new_topic)",
  "reasoning": "string"
}"""


def _parse_selection(data: Any, *, role: str) -> TopicSelectionSubmission:
    data = json_object(data, role=role)
    action = require_choice(data, "action", ("new_topic", "continue_existing"), role=role)
    reasoning = require_text(data, "reasoning", role=role)
    if action == "continue_existing":
        topic_id = require_text(data, "topic_id", role=role)
        topic_prompt = optional_text(data, "topic_prompt", role=role)
    else:
        topic_id = optional_text(data, "topic_id", role=role) or None
        topic_prompt = require_text(data, "topic_prompt", role=role)
    return TopicSelectionSubmission(
        action=action,
        topic_id=topic_id,
        topic_prompt=topic_prompt,
        reasoning=reasoning,
    )

logger = logging.getLogger(__name__)


class TopicSelectorAgent:
    """
    Agent that selects the next brainstorm topic.
    Can choose to start a new topic or continue an existing incomplete topic.
    
    Context handling:
    - Direct injects all metadata summaries (brainstorms, papers, rejections)
    - Validates prompt fits in context before sending
    - Truncates paper abstracts (not full content) if needed for context fit
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
        self.role_id = "autonomous_topic_selector"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"agg_sub1_{self.task_sequence:03d}"
    
    def _calculate_max_input_tokens(self) -> int:
        """Calculate available tokens for input prompt."""
        return rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
    
    async def select_topic(
        self,
        user_research_prompt: str,
        brainstorms_summary: List[Dict[str, Any]],
        papers_summary: List[Dict[str, Any]],
        candidate_questions: str = ""
    ) -> Optional[TopicSelectionSubmission]:
        """
        Generate a topic selection submission.
        
        Args:
            user_research_prompt: The user's high-level research goal
            brainstorms_summary: List of all brainstorms with metadata
            papers_summary: List of all papers with title, abstract, word count
            candidate_questions: Formatted candidate questions from topic exploration phase
        
        Returns:
            TopicSelectionSubmission or None if generation failed

        Raises:
            ModelOutputContractError: the reply did not match its JSON contract
                after one bounded retry.
        """
        try:
            max_input_tokens = self._calculate_max_input_tokens()
            rejection_entries = await autonomous_rejection_logs.get_topic_selection_rejections()
            active_papers = papers_summary

            def build_prompt(entries):
                return build_topic_selection_prompt(
                    user_research_prompt=user_research_prompt,
                    brainstorms_summary=brainstorms_summary,
                    papers_summary=active_papers,
                    rejection_context=autonomous_rejection_logs.render_topic_rejections(entries),
                    candidate_questions=candidate_questions,
                )

            fit = fit_prompt_with_feedback(
                rejection_entries,
                build_prompt=build_prompt,
                available_tokens=max_input_tokens,
            )
            prompt = fit.prompt
            retained_rejections = fit.retained_entries
            prompt_tokens = fit.prompt_tokens
            
            if prompt_tokens > max_input_tokens:
                # Context too large - truncate paper abstracts to fit
                logger.warning(f"TopicSelector: Prompt ({prompt_tokens} tokens) exceeds limit ({max_input_tokens}). "
                             f"Truncating paper abstracts.")
                
                # Truncate abstracts in papers_summary
                truncated_papers = []
                for p in papers_summary:
                    truncated = p.copy()
                    if "abstract" in truncated and len(truncated.get("abstract", "")) > 500:
                        truncated["abstract"] = truncated["abstract"][:500] + "..."
                    truncated_papers.append(truncated)
                
                active_papers = truncated_papers
                fit = fit_prompt_with_feedback(
                    retained_rejections,
                    build_prompt=build_prompt,
                    available_tokens=max_input_tokens,
                )
                prompt = fit.prompt
                retained_rejections = fit.retained_entries
                prompt_tokens = fit.prompt_tokens
                if prompt_tokens > max_input_tokens:
                    logger.error(f"TopicSelector: Even after truncation, prompt ({prompt_tokens}) exceeds limit ({max_input_tokens})")
                    return None
            from backend.shared.solution_path.integration import with_budgeted_solver_plan
            prompt = with_budgeted_solver_plan(
                prompt,
                getattr(self, "solution_path_manager", None),
                max_input_tokens,
            )
            prompt_tokens = count_tokens(prompt)
            
            task_id = self.get_current_task_id()
            await api_client_manager.prewarm_assistant_memory_context(
                task_id=task_id,
                role_id=self.role_id,
                prompt=prompt,
            )

            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"TopicSelector: Generating topic selection with model {self.model_id} "
                       f"(prompt={prompt_tokens}t, task_id={task_id})")
            
            while True:
                try:
                    response = await api_client_manager.generate_completion(
                        task_id=task_id,
                        role_id=self.role_id,
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_output_tokens,
                        temperature=0.0
                    )
                    break
                except ProviderContextLengthError:
                    if len(retained_rejections) <= 1:
                        raise
                    retained_rejections = retained_rejections[1:]
                    prompt = build_prompt(retained_rejections)
            
            try:
                submission = await parse_with_contract_retry(
                    response_text(response),
                    lambda parsed: _parse_selection(parsed, role=self.role_id),
                    prompt=prompt,
                    task_id=task_id,
                    role_id=self.role_id,
                    model=self.model_id,
                    context_window=self.context_window,
                    max_output_tokens=self.max_output_tokens,
                    response_noun="topic selection",
                    response_shape=_SELECTION_SHAPE,
                )
            finally:
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)

            logger.info(f"TopicSelector: Generated submission - action={submission.action}")
            return submission

        except FreeModelExhaustedError:
            raise
        except RetryableProviderError:
            raise
        except ModelOutputContractError:
            raise
        except Exception as e:
            if api_client_manager.is_provider_failure(e):
                raise
            logger.error(f"TopicSelector: Error generating submission: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
    
    async def handle_rejection(
        self,
        submission: TopicSelectionSubmission,
        rejection_reasoning: str
    ) -> None:
        """
        Handle a rejected topic selection.
        
        Args:
            submission: The rejected submission
            rejection_reasoning: Why it was rejected
        """
        await autonomous_rejection_logs.add_topic_selection_rejection(
            action=submission.action,
            proposed_topic=submission.topic_prompt or submission.topic_id or "",
            rejection_reasoning=rejection_reasoning
        )
        logger.info(f"TopicSelector: Logged rejection for action={submission.action}")

