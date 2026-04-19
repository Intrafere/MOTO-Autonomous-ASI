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
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens
from backend.shared.models import TopicSelectionSubmission
from backend.autonomous.prompts.topic_prompts import (
    build_topic_selection_prompt
)
from backend.autonomous.memory.research_metadata import research_metadata
from backend.autonomous.memory.autonomous_rejection_logs import autonomous_rejection_logs

logger = logging.getLogger(__name__)


class TopicSelectorAgent:
    """
    Agent that selects the next brainstorm topic.
    Can choose to start a new topic, continue existing, or combine topics.
    
    Context handling:
    - Direct injects all metadata summaries (brainstorms, papers, rejections)
    - Validates prompt fits in context before sending
    - Truncates paper abstracts (not full content) if needed for context fit
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
        return self.context_window - self.max_output_tokens
    
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
        """
        try:
            # Get rejection context
            rejection_context = await autonomous_rejection_logs.format_topic_rejections_for_context()
            
            # Build prompt
            prompt = build_topic_selection_prompt(
                user_research_prompt=user_research_prompt,
                brainstorms_summary=brainstorms_summary,
                papers_summary=papers_summary,
                rejection_context=rejection_context,
                candidate_questions=candidate_questions
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input_tokens = self._calculate_max_input_tokens()
            
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
                
                # Rebuild prompt with truncated papers
                prompt = build_topic_selection_prompt(
                    user_research_prompt=user_research_prompt,
                    brainstorms_summary=brainstorms_summary,
                    papers_summary=truncated_papers,
                    rejection_context=rejection_context,
                    candidate_questions=candidate_questions
                )
                
                prompt_tokens = count_tokens(prompt)
                if prompt_tokens > max_input_tokens:
                    logger.error(f"TopicSelector: Even after truncation, prompt ({prompt_tokens}) exceeds limit ({max_input_tokens})")
                    return None
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"TopicSelector: Generating topic selection with model {self.model_id} "
                       f"(prompt={prompt_tokens}t, task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0  # Deterministic generation - evolving context provides diversity
            )
            
            if not response:
                logger.error("TopicSelector: Empty response from LLM")
                return None
            
            # Extract content from response (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                logger.error("TopicSelector: No content in response")
                return None
            
            # Parse JSON using central utility (handles sanitization + parsing + array handling)
            try:
                data = parse_json(content)
                
                # parse_json already handles array response, but keep check for safety
                if isinstance(data, list) and len(data) > 0:
                    logger.warning("TopicSelector: Model returned array, using first element")
                    data = data[0]
                
                # Validate required fields
                action = data.get("action", "")
                if action not in ["new_topic", "continue_existing", "combine_topics"]:
                    logger.error(f"TopicSelector: Invalid action: {action}")
                    return None
                
                # Create submission
                submission = TopicSelectionSubmission(
                    action=action,
                    topic_id=data.get("topic_id"),
                    topic_ids=data.get("topic_ids", []),
                    topic_prompt=data.get("topic_prompt", ""),
                    reasoning=data.get("reasoning", "No reasoning provided")
                )
                
                # Validate based on action
                if action == "continue_existing" and not submission.topic_id:
                    logger.error("TopicSelector: continue_existing requires topic_id")
                    return None
                
                if action == "combine_topics" and len(submission.topic_ids) < 2:
                    logger.error("TopicSelector: combine_topics requires at least 2 topic_ids")
                    return None
                
                if action in ["new_topic", "combine_topics"] and not submission.topic_prompt:
                    logger.error(f"TopicSelector: {action} requires topic_prompt")
                    return None
                
                # Notify task completed successfully
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                
                logger.info(f"TopicSelector: Generated submission - action={action}")
                return submission
                
            except json.JSONDecodeError as e:
                logger.error(f"TopicSelector: JSON parse error: {e}")
                logger.debug(f"TopicSelector: Raw response: {content[:500]}...")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
                
        except FreeModelExhaustedError:
            raise
        except Exception as e:
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
            proposed_topic=submission.topic_prompt or submission.topic_id or str(submission.topic_ids),
            rejection_reasoning=rejection_reasoning
        )
        logger.info(f"TopicSelector: Logged rejection for action={submission.action}")

