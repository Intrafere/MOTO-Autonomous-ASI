"""
Answer Format Selector Agent - Phase 2 of Tier 3 Final Answer Generation.

Selects whether the final answer should be:
- SHORT FORM: A single comprehensive paper directly answering the user's question
- LONG FORM: A curated volume/collection of papers with introduction and conclusion

CRITICAL: Operates ONLY on Tier 2 papers, NOT on Tier 1 brainstorm databases.

NO RAG BY DESIGN: This agent makes a strategic format decision using only the certainty
assessment result and paper metadata summaries (titles/abstracts). Full paper content
is not needed to decide short-form vs long-form — that's a structural question about
the research landscape, not a content-deep analysis.
"""
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, Callable

from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens
from backend.shared.models import AnswerFormatSelection, CertaintyAssessment
from backend.autonomous.prompts.final_answer_prompts import (
    build_format_selection_prompt,
    build_format_validation_prompt
)
from backend.autonomous.memory.final_answer_memory import final_answer_memory

logger = logging.getLogger(__name__)


class AnswerFormatSelector:
    """
    Agent that selects the format for the final answer (short vs long form).
    
    Phase 2 of Tier 3 workflow:
    1. Review certainty assessment from Phase 1
    2. Consider number and diversity of papers
    3. Select short_form or long_form
    4. Get selection validated
    5. Retry on rejection with feedback
    """
    
    MAX_RETRIES = 10  # Maximum retries with rejection feedback
    
    def __init__(
        self,
        submitter_model: str,
        validator_model: str,
        context_window: int = 131072,
        max_output_tokens: int = 25000
    ):
        self.submitter_model = submitter_model
        self.validator_model = validator_model
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "autonomous_format_selector"
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
    
    async def select_format(
        self,
        user_research_prompt: str,
        certainty_assessment: CertaintyAssessment,
        all_papers: List[Dict[str, Any]]
    ) -> Optional[AnswerFormatSelection]:
        """
        Complete format selection workflow with validation loop.
        
        Args:
            user_research_prompt: The user's original research question
            certainty_assessment: Result from Phase 1
            all_papers: List of all Tier 2 papers with metadata
        
        Returns:
            Validated AnswerFormatSelection or None if failed
        """
        if not all_papers:
            # No papers - default to short form
            logger.warning("AnswerFormatSelector: No papers available, defaulting to short_form")
            selection = AnswerFormatSelection(
                answer_format="short_form",
                reasoning="No research papers available. A short form answer will address the question based on available knowledge."
            )
            await final_answer_memory.set_answer_format(selection)
            return selection
        
        logger.info(f"AnswerFormatSelector: Starting format selection "
                   f"(certainty: {certainty_assessment.certainty_level}, papers: {len(all_papers)})")
        
        attempt = 0
        rejection_context = ""
        
        while attempt < self.MAX_RETRIES:
            attempt += 1
            logger.info(f"AnswerFormatSelector: Selection attempt {attempt}/{self.MAX_RETRIES}")
            
            # Generate selection
            selection = await self._generate_selection(
                user_research_prompt,
                certainty_assessment,
                all_papers,
                rejection_context
            )
            
            if selection is None:
                logger.error(f"AnswerFormatSelector: Failed to generate selection (attempt {attempt})")
                continue
            
            # Validate selection
            is_valid, feedback = await self._validate_selection(
                user_research_prompt,
                certainty_assessment,
                all_papers,
                selection
            )
            
            if is_valid:
                logger.info(f"AnswerFormatSelector: Format validated: {selection.answer_format}")
                await final_answer_memory.set_answer_format(selection)
                return selection
            else:
                # Log rejection and prepare for retry
                logger.info(f"AnswerFormatSelector: Selection rejected: {feedback[:100]}...")
                await final_answer_memory.add_rejection(
                    phase="format",
                    rejection_summary=feedback,
                    submission_preview=f"Format: {selection.answer_format}, Reasoning: {selection.reasoning[:400]}"
                )
                rejection_context = await final_answer_memory.get_rejection_context_async("format")
        
        logger.error(f"AnswerFormatSelector: Failed after {self.MAX_RETRIES} attempts")
        return None
    
    async def _generate_selection(
        self,
        user_research_prompt: str,
        certainty_assessment: CertaintyAssessment,
        all_papers: List[Dict[str, Any]],
        rejection_context: str = ""
    ) -> Optional[AnswerFormatSelection]:
        """Generate format selection."""
        try:
            # Build prompt
            prompt = build_format_selection_prompt(
                user_research_prompt=user_research_prompt,
                papers_summary=all_papers,
                certainty_assessment=certainty_assessment.model_dump(),
                rejection_context=rejection_context
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_max_input_tokens()
            
            if prompt_tokens > max_input:
                logger.error(f"AnswerFormatSelector: Prompt too large ({prompt_tokens} > {max_input})")
                return None
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            logger.info(f"AnswerFormatSelector: Generating selection (prompt={prompt_tokens}t, task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.submitter_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0
            )
            
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            if not response:
                return None
            
            # Extract content
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return None
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            # Validate answer_format value
            answer_format = data.get("answer_format", "short_form")
            if answer_format not in ["short_form", "long_form"]:
                logger.warning(f"AnswerFormatSelector: Invalid format '{answer_format}', defaulting to short_form")
                answer_format = "short_form"
            
            return AnswerFormatSelection(
                answer_format=answer_format,
                reasoning=data.get("reasoning", "")
            )
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"AnswerFormatSelector: Error generating selection: {e}")
            return None
    
    async def _validate_selection(
        self,
        user_research_prompt: str,
        certainty_assessment: CertaintyAssessment,
        all_papers: List[Dict[str, Any]],
        selection: AnswerFormatSelection
    ) -> tuple[bool, str]:
        """
        Validate the format selection.
        
        Returns:
            Tuple of (is_valid, feedback_if_rejected)
        """
        try:
            # Build validation prompt
            prompt = build_format_validation_prompt(
                user_research_prompt=user_research_prompt,
                papers_summary=all_papers,
                certainty_assessment=certainty_assessment.model_dump(),
                format_selection=selection.model_dump()
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_max_input_tokens()
            
            if prompt_tokens > max_input:
                logger.error(f"AnswerFormatSelector: Validation prompt too large ({prompt_tokens} > {max_input})")
                return False, "Validation prompt exceeds context limit"
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            logger.info(f"AnswerFormatSelector: Validating selection (task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=f"{self.role_id}_validator",
                model=self.validator_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0
            )
            
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            if not response:
                return False, "Empty response from validator"
            
            # Extract content
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return False, "No content in validator response"
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            decision = data.get("decision", "reject")
            reasoning = data.get("reasoning", "No reasoning provided")
            
            return decision == "accept", reasoning
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"AnswerFormatSelector: Error validating selection: {e}")
            return False, str(e)

