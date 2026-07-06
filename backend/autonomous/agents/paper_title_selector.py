"""
Paper Title Selector Agent - Selects titles for papers.

NO RAG BY DESIGN: This agent selects a title based on brainstorm SUMMARY (not full DB),
existing paper titles/abstracts from this brainstorm, and reference paper metadata.
All inputs are compact summaries that fit in direct injection. The full brainstorm
content is not needed — a summary is sufficient to choose an appropriate title.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import extract_message_text
from backend.shared.models import PaperTitleSelection
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.autonomous.prompts.paper_title_prompts import (
    build_paper_title_prompt,
    build_paper_title_validation_prompt
)

logger = logging.getLogger(__name__)


def _is_title_model_call_failure(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return (
        isinstance(exc, RetryableProviderError)
        or is_non_retryable_model_error(exc)
        or is_transient_model_call_error(exc)
        or "upstream provider timeout" in message
        or "response missing 'choices'" in message
        or "no api key" in message
        or "exceeds context" in message
        or "exceeds the configured" in message
    )


class PaperTitleSelectorAgent:
    """
    Agent that selects titles for papers.
    Title is validated before acceptance.
    """
    
    def __init__(
        self,
        model_id: str,
        validator_model_id: str,
        context_window: int = 0,
        max_output_tokens: int = 0,
        validator_context_window: Optional[int] = None,
        validator_max_output_tokens: Optional[int] = None,
    ):
        self.model_id = model_id
        self.validator_model_id = validator_model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.validator_context_window = validator_context_window or context_window
        self.validator_max_output_tokens = validator_max_output_tokens or max_output_tokens
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "autonomous_paper_title_selector"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"agg_sub1_{self.task_sequence:03d}"

    def get_current_validation_task_id(self) -> str:
        """Get a validator-routed task ID for title validation."""
        return f"agg_val_{self.task_sequence:03d}"
    
    async def select_title(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        existing_papers_from_brainstorm: List[Dict[str, Any]],
        reference_papers: List[Dict[str, Any]] = None,
        candidate_titles: str = "",
        stop_event: Optional[asyncio.Event] = None
    ) -> Optional[str]:
        """
        Select and validate a paper title.
        Retries indefinitely until a valid title is accepted or stop_event is set.
        Rejection feedback from each attempt is threaded into the next generation call
        so the model can correct its mistakes.

        Args:
            candidate_titles: Pre-validated candidate titles from exploration phase.
            stop_event: If provided, the loop exits when the event is set (user stop).

        Returns:
            Validated paper title, or None only if stop_event was set before success.
        """
        rejection_history: List[str] = []
        attempt = 0

        while True:
            # Honour user stop before each attempt
            if stop_event is not None and stop_event.is_set():
                logger.info("PaperTitleSelector: Stop event set - exiting title selection")
                return None

            attempt += 1
            logger.info(f"PaperTitleSelector: Attempt {attempt}")

            # Build accumulated rejection feedback string (keep last 5 for context budget)
            rejection_feedback = ""
            if rejection_history:
                recent = rejection_history[-5:]
                lines = []
                for i, r in enumerate(recent):
                    lines.append(f"Attempt {attempt - len(recent) + i}: {r}")
                rejection_feedback = "\n".join(lines)

            # Generate title selection (pass feedback so model learns from failures)
            selection = await self._generate_title(
                user_research_prompt,
                topic_prompt,
                brainstorm_summary,
                existing_papers_from_brainstorm,
                reference_papers,
                rejection_feedback=rejection_feedback,
                candidate_titles=candidate_titles
            )

            if selection is None:
                logger.error("PaperTitleSelector: Failed to generate title - will retry")
                await asyncio.sleep(5)
                continue

            # Validate title
            is_valid, rejection_reason = await self._validate_title(
                user_research_prompt,
                topic_prompt,
                brainstorm_summary,
                existing_papers_from_brainstorm,
                reference_papers,
                selection.paper_title,
                selection.reasoning
            )

            if is_valid:
                logger.info(f"PaperTitleSelector: Title accepted: '{selection.paper_title}'")
                return selection.paper_title
            else:
                logger.info(
                    f"PaperTitleSelector: Title rejected (attempt {attempt}): {rejection_reason}"
                )
                rejection_history.append(
                    f"Title '{selection.paper_title}' was rejected because: {rejection_reason}"
                )
                await asyncio.sleep(2)
    
    async def _generate_title(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        existing_papers_from_brainstorm: List[Dict[str, Any]],
        reference_papers: List[Dict[str, Any]] = None,
        rejection_feedback: str = "",
        candidate_titles: str = ""
    ) -> Optional[PaperTitleSelection]:
        """Generate a paper title selection."""
        try:
            max_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)

            # Build prompt with full rejection feedback first
            prompt = build_paper_title_prompt(
                user_research_prompt=user_research_prompt,
                topic_prompt=topic_prompt,
                brainstorm_summary=brainstorm_summary,
                existing_papers_from_brainstorm=existing_papers_from_brainstorm,
                reference_papers=reference_papers,
                rejection_feedback=rejection_feedback,
                candidate_titles=candidate_titles
            )

            # If prompt is too large, shed oldest rejection entries one at a time until it fits
            if rejection_feedback and count_tokens(prompt) > max_input_tokens:
                feedback_lines = [l for l in rejection_feedback.split("\n") if l.strip()]
                while feedback_lines and count_tokens(prompt) > max_input_tokens:
                    feedback_lines.pop(0)  # drop oldest entry
                    trimmed_feedback = "\n".join(feedback_lines)
                    prompt = build_paper_title_prompt(
                        user_research_prompt=user_research_prompt,
                        topic_prompt=topic_prompt,
                        brainstorm_summary=brainstorm_summary,
                        existing_papers_from_brainstorm=existing_papers_from_brainstorm,
                        reference_papers=reference_papers,
                        rejection_feedback=trimmed_feedback,
                        candidate_titles=candidate_titles
                    )
                if count_tokens(prompt) > max_input_tokens:
                    logger.warning(
                        "PaperTitleSelector: Prompt still exceeds context even with no rejection "
                        "feedback - sending without feedback"
                    )
                    prompt = build_paper_title_prompt(
                        user_research_prompt=user_research_prompt,
                        topic_prompt=topic_prompt,
                        brainstorm_summary=brainstorm_summary,
                        existing_papers_from_brainstorm=existing_papers_from_brainstorm,
                        reference_papers=reference_papers,
                        rejection_feedback="",
                        candidate_titles=candidate_titles
                    )

            # Progressive truncation if still too large after shedding rejection feedback
            if count_tokens(prompt) > max_input_tokens:
                logger.warning("PaperTitleSelector: Truncating existing paper outlines/abstracts to fit")
                truncated_existing = []
                for p in existing_papers_from_brainstorm:
                    tp = p.copy()
                    if tp.get("outline"):
                        tp["outline"] = ""
                    if tp.get("abstract") and len(tp["abstract"]) > 200:
                        tp["abstract"] = tp["abstract"][:200] + "..."
                    truncated_existing.append(tp)
                prompt = build_paper_title_prompt(
                    user_research_prompt=user_research_prompt,
                    topic_prompt=topic_prompt,
                    brainstorm_summary=brainstorm_summary,
                    existing_papers_from_brainstorm=truncated_existing,
                    reference_papers=reference_papers,
                    rejection_feedback="",
                    candidate_titles=candidate_titles
                )
            
            if count_tokens(prompt) > max_input_tokens:
                logger.warning("PaperTitleSelector: Truncating brainstorm summary to fit")
                prompt = build_paper_title_prompt(
                    user_research_prompt=user_research_prompt,
                    topic_prompt=topic_prompt,
                    brainstorm_summary=brainstorm_summary[:2000] + "\n... [truncated for context fit]",
                    existing_papers_from_brainstorm=truncated_existing,
                    reference_papers=reference_papers,
                    rejection_feedback="",
                    candidate_titles=candidate_titles
                )
            
            task_id = self.get_current_task_id()
            await api_client_manager.prewarm_assistant_memory_context(
                task_id=task_id,
                role_id=self.role_id,
                prompt=prompt,
            )

            if count_tokens(prompt) > max_input_tokens:
                logger.error("PaperTitleSelector: Cannot fit prompt even after all truncation")
                raise ValueError(
                    "Title generation prompt exceeds context limit even after shedding optional title context."
                )

            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"PaperTitleSelector: Generating title with model {self.model_id} (task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0  # Deterministic generation - evolving context provides diversity
            )
            
            if not response:
                return None
            
            # Extract content (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = extract_message_text(message)
            if not content:
                return None
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            title = data.get("paper_title", "")
            if not title:
                logger.error("PaperTitleSelector: No title in response")
                return None
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            return PaperTitleSelection(
                paper_title=title,
                reasoning=data.get("reasoning", "")
            )
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            if _is_title_model_call_failure(e):
                raise
            logger.error(f"PaperTitleSelector: Error generating title: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
    
    async def _validate_title(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        existing_papers_from_brainstorm: List[Dict[str, Any]],
        reference_papers: List[Dict[str, Any]],
        proposed_title: str,
        title_reasoning: str
    ) -> Tuple[bool, str]:
        """
        Validate a paper title.
        
        Returns:
            Tuple of (is_valid: bool, rejection_reason: str)
        """
        try:
            # Build validation prompt
            prompt = build_paper_title_validation_prompt(
                user_research_prompt=user_research_prompt,
                topic_prompt=topic_prompt,
                brainstorm_summary=brainstorm_summary,
                existing_papers_from_brainstorm=existing_papers_from_brainstorm,
                reference_papers=reference_papers,
                proposed_title=proposed_title,
                title_reasoning=title_reasoning
            )

            max_input_tokens = rag_config.get_available_input_tokens(
                self.validator_context_window,
                self.validator_max_output_tokens,
            )
            prompt_tokens = count_tokens(prompt)
            if prompt_tokens > max_input_tokens:
                raise ValueError(
                    "Title validation prompt exceeds the configured validator context window "
                    f"({prompt_tokens} > {max_input_tokens})."
                )
            
            # Generate task ID for validation tracking
            task_id = self.get_current_validation_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call validator LLM via api_client_manager (handles boost and fallback)
            logger.info(f"PaperTitleSelector: Validating with model {self.validator_model_id} (task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id="autonomous_paper_title_validator",
                model=self.validator_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.validator_max_output_tokens,
                temperature=0.0  # Deterministic validation - evolving context provides diversity
            )
            
            if not response:
                return False, "Empty validation response"
            
            # Extract content (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = extract_message_text(message)
            if not content:
                return False, "No content in validation response"
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            decision = data.get("decision", "").lower()
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            if decision == "accept":
                return True, ""
            else:
                return False, reasoning
                
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            if _is_title_model_call_failure(e):
                raise
            logger.error(f"PaperTitleSelector: Error validating title: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return False, f"Validation error: {str(e)}"

