"""
Paper Title Selector Agent - Selects titles for papers.
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.models import PaperTitleSelection
from backend.shared.utils import count_tokens
from backend.autonomous.prompts.paper_title_prompts import (
    build_paper_title_prompt,
    build_paper_title_validation_prompt
)

logger = logging.getLogger(__name__)


class PaperTitleSelectorAgent:
    """
    Agent that selects titles for papers.
    Title is validated before acceptance.
    """
    
    def __init__(
        self,
        model_id: str,
        validator_model_id: str,
        context_window: int = 131072,
        max_output_tokens: int = 25000
    ):
        self.model_id = model_id
        self.validator_model_id = validator_model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "autonomous_paper_title_selector"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"auto_pt_{self.task_sequence:03d}"
    
    async def select_title(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        existing_papers_from_brainstorm: List[Dict[str, Any]],
        reference_papers: List[Dict[str, Any]] = None,
        stop_event: Optional[asyncio.Event] = None
    ) -> Optional[str]:
        """
        Select and validate a paper title.
        Retries indefinitely until a valid title is accepted or stop_event is set.
        Rejection feedback from each attempt is threaded into the next generation call
        so the model can correct its mistakes.

        Args:
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
                rejection_feedback=rejection_feedback
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
        rejection_feedback: str = ""
    ) -> Optional[PaperTitleSelection]:
        """Generate a paper title selection."""
        try:
            max_input_tokens = self.context_window - self.max_output_tokens

            # Build prompt with full rejection feedback first
            prompt = build_paper_title_prompt(
                user_research_prompt=user_research_prompt,
                topic_prompt=topic_prompt,
                brainstorm_summary=brainstorm_summary,
                existing_papers_from_brainstorm=existing_papers_from_brainstorm,
                reference_papers=reference_papers,
                rejection_feedback=rejection_feedback
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
                        rejection_feedback=trimmed_feedback
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
                        rejection_feedback=""
                    )

            # Generate task ID for tracking
            task_id = self.get_current_task_id()
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
            content = message.get("content", "") or message.get("reasoning", "")
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
            
        except Exception as e:
            logger.error(f"PaperTitleSelector: Error generating title: {e}")
            # Notify task completed (failed but still completed)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
    
    async def _validate_title(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        existing_papers_from_brainstorm: List[Dict[str, Any]],
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
                proposed_title=proposed_title,
                title_reasoning=title_reasoning
            )
            
            # Generate task ID for validation tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call validator LLM via api_client_manager (handles boost and fallback)
            logger.info(f"PaperTitleSelector: Validating with model {self.validator_model_id} (task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,  # Use same role_id for validation
                model=self.validator_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15000,
                temperature=0.0  # Deterministic validation - evolving context provides diversity
            )
            
            if not response:
                return False, "Empty validation response"
            
            # Extract content (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "") or message.get("reasoning", "")
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
                
        except Exception as e:
            logger.error(f"PaperTitleSelector: Error validating title: {e}")
            # Notify task completed (failed but still completed)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return False, f"Validation error: {str(e)}"

