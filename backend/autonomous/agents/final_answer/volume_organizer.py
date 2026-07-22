"""
Volume Organizer Agent - Phase 3B of Tier 3 Final Answer Generation (Long Form).

Organizes a volume/collection structure when long form answer is selected:
- Selects existing papers as body chapters
- Identifies gap papers that need to be written
- Plans introduction and conclusion papers
- Iteratively refines until validator agrees

CRITICAL: Operates ONLY on Tier 2 papers, NOT on Tier 1 brainstorm databases.

NO RAG BY DESIGN: This agent organizes chapter order and identifies structural gaps
using only paper metadata summaries (titles/abstracts/outlines) and the certainty
assessment. Full paper content is not needed to plan volume structure — that's a
high-level organizational decision based on what each paper covers.
"""
import logging
from typing import Optional, List, Dict, Any, Callable

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.shared.models import (
    CertaintyAssessment,
    VolumeOrganization,
    VolumeChapter
)
from backend.autonomous.prompts.final_answer_prompts import (
    build_volume_organization_prompt,
    build_volume_validation_prompt
)
from backend.autonomous.memory.final_answer_memory import final_answer_memory

logger = logging.getLogger(__name__)


def _is_tier3_model_call_failure(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return (
        isinstance(exc, RetryableProviderError)
        or is_non_retryable_model_error(exc)
        or is_transient_model_call_error(exc)
        or "upstream provider timeout" in message
        or "response missing 'choices'" in message
        or "no api key" in message
    )


class VolumeOrganizer:
    """
    Agent that organizes volume structure for long form answers.
    
    Phase 3B of Tier 3 workflow (long form only):
    1. Review all Tier 2 papers
    2. Select papers as body chapters
    3. Identify gap papers needed
    4. Plan introduction and conclusion
    5. Validate with iterative refinement
    6. Repeat until outline_complete=true and validator accepts
    """
    
    MAX_ITERATIONS = 15  # Maximum iterations (like outline creation)
    
    def __init__(
        self,
        submitter_model: str,
        validator_model: str,
        context_window: int = 0,
        max_output_tokens: int = 0,
        validator_context_window: Optional[int] = None,
        validator_max_output_tokens: Optional[int] = None,
    ):
        self.submitter_model = submitter_model
        self.validator_model = validator_model
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.validator_context_window = validator_context_window or context_window
        self.validator_max_output_tokens = validator_max_output_tokens or max_output_tokens
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "autonomous_volume_organizer"
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

    def _calculate_validator_max_input_tokens(self) -> int:
        """Calculate available tokens for validator prompts."""
        return rag_config.get_available_input_tokens(
            self.validator_context_window,
            self.validator_max_output_tokens,
        )
    
    async def organize_volume(
        self,
        user_research_prompt: str,
        certainty_assessment: CertaintyAssessment,
        all_papers: List[Dict[str, Any]]
    ) -> Optional[VolumeOrganization]:
        """
        Complete volume organization workflow with validation loop.
        
        Args:
            user_research_prompt: The user's original research question
            certainty_assessment: Result from Phase 1
            all_papers: List of all Tier 2 papers with metadata
        
        Returns:
            Validated VolumeOrganization or None if failed
        """
        if not all_papers:
            logger.error("VolumeOrganizer: No papers available for volume organization")
            return None
        
        logger.info(f"VolumeOrganizer: Starting organization with {len(all_papers)} papers")
        
        iteration = 0
        current_volume: Dict[str, Any] = None
        rejection_context = ""
        validator_feedback = ""
        
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            logger.info(f"VolumeOrganizer: Iteration {iteration}/{self.MAX_ITERATIONS}")
            
            # Generate or refine organization
            organization = await self._generate_organization(
                user_research_prompt,
                certainty_assessment,
                all_papers,
                current_volume,
                rejection_context,
                validator_feedback
            )
            
            if organization is None:
                logger.error(f"VolumeOrganizer: Failed to generate organization (iteration {iteration})")
                continue
            
            # Validate organization
            is_valid, feedback = await self._validate_organization(
                user_research_prompt,
                all_papers,
                organization
            )
            
            if is_valid:
                # Check if submitter marked outline as complete
                if organization.outline_complete:
                    logger.info(f"VolumeOrganizer: Volume organization complete: {organization.volume_title}")
                    await final_answer_memory.save_volume_organization(organization)
                    return organization
                else:
                    # Accepted but not marked complete - continue refining
                    logger.info("VolumeOrganizer: Organization accepted, but not marked complete. Continuing refinement.")
                    current_volume = organization.model_dump()
                    validator_feedback = feedback  # Use positive feedback for improvement
            else:
                # Log rejection and prepare for retry
                logger.info(f"VolumeOrganizer: Organization rejected: {feedback[:100]}...")
                await final_answer_memory.add_rejection(
                    phase="volume",
                    rejection_summary=feedback,
                    submission_preview=f"Title: {organization.volume_title}, Chapters: {len(organization.chapters)}"
                )
                rejection_context = await final_answer_memory.get_rejection_context_async("volume")
                current_volume = organization.model_dump()
                validator_feedback = feedback
        
        logger.error(
            "VolumeOrganizer: No validator-approved completed organization after "
            f"{self.MAX_ITERATIONS} iterations"
        )
        return None
    
    async def _generate_organization(
        self,
        user_research_prompt: str,
        certainty_assessment: CertaintyAssessment,
        all_papers: List[Dict[str, Any]],
        current_volume: Dict[str, Any] = None,
        rejection_context: str = "",
        validator_feedback: str = ""
    ) -> Optional[VolumeOrganization]:
        """Generate or refine volume organization."""
        try:
            # Build prompt
            prompt = build_volume_organization_prompt(
                user_research_prompt=user_research_prompt,
                papers_summary=all_papers,
                certainty_assessment=certainty_assessment.model_dump(),
                current_volume=current_volume,
                rejection_context=rejection_context,
                validator_feedback=validator_feedback
            )
            
            task_id = self.get_current_task_id()
            await api_client_manager.prewarm_assistant_memory_context(
                task_id=task_id,
                role_id=self.role_id,
                prompt=prompt,
            )

            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_max_input_tokens()
            from backend.shared.solution_path.integration import with_budgeted_solver_plan
            prompt = with_budgeted_solver_plan(
                prompt, getattr(self, "solution_path_manager", None), max_input
            )
            prompt_tokens = count_tokens(prompt)
            
            if prompt_tokens > max_input:
                logger.error(f"VolumeOrganizer: Prompt too large ({prompt_tokens} > {max_input})")
                return None
            
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            logger.info(f"VolumeOrganizer: Generating organization (prompt={prompt_tokens}t, task_id={task_id})")
            
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
            content = extract_message_text(message)
            if not content:
                return None
            
            # Parse JSON using central utility
            data = parse_json(content)
            volume_title = data.get("volume_title")
            outline_complete = data.get("outline_complete")
            reasoning = data.get("reasoning")
            chapter_items = data.get("chapters")
            if not isinstance(volume_title, str) or not volume_title.strip():
                raise ValueError("Volume organization requires a non-empty volume_title")
            if type(outline_complete) is not bool:
                raise ValueError("Volume organization requires outline_complete to be a boolean")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("Volume organization requires non-empty reasoning")
            if not isinstance(chapter_items, list) or not chapter_items:
                raise ValueError("Volume organization requires a non-empty chapters list")

            chapters = []
            for index, ch in enumerate(chapter_items, start=1):
                if not isinstance(ch, dict):
                    raise ValueError(f"Volume chapter {index} must be an object")
                chapter_type = ch.get("chapter_type")
                title = ch.get("title")
                order = ch.get("order")
                status = ch.get("status", "pending")
                description = ch.get("description", "")
                paper_id = ch.get("paper_id")
                if chapter_type not in {
                    "existing_paper", "introduction", "conclusion", "gap_paper"
                }:
                    raise ValueError(f"Volume chapter {index} has invalid chapter_type")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError(f"Volume chapter {index} requires a non-empty title")
                if type(order) is not int or order < 1:
                    raise ValueError(f"Volume chapter {index} requires a positive integer order")
                if status not in {"pending", "writing", "complete"}:
                    raise ValueError(f"Volume chapter {index} has invalid status")
                if not isinstance(description, str):
                    raise ValueError(f"Volume chapter {index} description must be a string")
                if chapter_type == "existing_paper" and (
                    not isinstance(paper_id, str) or not paper_id.strip()
                ):
                    raise ValueError(
                        f"Existing-paper chapter {index} requires a non-empty paper_id"
                    )
                chapter = VolumeChapter(
                    chapter_type=chapter_type,
                    paper_id=paper_id.strip() if isinstance(paper_id, str) else None,
                    title=title.strip(),
                    order=order,
                    status=status,
                    description=description.strip(),
                )
                chapters.append(chapter)
            
            intro_count = sum(ch.chapter_type == "introduction" for ch in chapters)
            conclusion_count = sum(ch.chapter_type == "conclusion" for ch in chapters)
            if intro_count != 1 or conclusion_count != 1:
                raise ValueError(
                    "Volume organization requires exactly one introduction and one conclusion"
                )
            orders = [ch.order for ch in chapters]
            if sorted(orders) != list(range(1, len(chapters) + 1)):
                raise ValueError("Volume chapter orders must be unique and contiguous from 1")
            ordered = sorted(chapters, key=lambda chapter: chapter.order)
            if (
                ordered[0].chapter_type != "introduction"
                or ordered[-1].chapter_type != "conclusion"
            ):
                raise ValueError(
                    "Volume organization must place introduction first and conclusion last"
                )
            
            return VolumeOrganization(
                volume_title=volume_title.strip(),
                chapters=ordered,
                outline_complete=outline_complete,
                revision_reasoning=reasoning.strip(),
            )
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            if _is_tier3_model_call_failure(e):
                raise
            logger.error(f"VolumeOrganizer: Error generating organization: {e}")
            return None
    
    def _normalize_chapter_order(self, chapters: List[VolumeChapter]) -> List[VolumeChapter]:
        """
        Normalize chapter ordering:
        - Introduction is always first (order=1)
        - Conclusion is always last
        - Body chapters in between, preserving relative order
        """
        intro = [ch for ch in chapters if ch.chapter_type == "introduction"]
        conclusion = [ch for ch in chapters if ch.chapter_type == "conclusion"]
        body = [ch for ch in chapters if ch.chapter_type not in ["introduction", "conclusion"]]
        
        # Sort body chapters by current order
        body.sort(key=lambda x: x.order)
        
        # Reassign orders
        result = []
        order = 1
        
        for ch in intro:
            ch.order = order
            result.append(ch)
            order += 1
        
        for ch in body:
            ch.order = order
            result.append(ch)
            order += 1
        
        for ch in conclusion:
            ch.order = order
            result.append(ch)
            order += 1
        
        return result
    
    async def _validate_organization(
        self,
        user_research_prompt: str,
        all_papers: List[Dict[str, Any]],
        organization: VolumeOrganization
    ) -> tuple[bool, str]:
        """
        Validate the volume organization.
        
        Returns:
            Tuple of (is_valid, feedback)
        """
        try:
            # Build validation prompt
            prompt = build_volume_validation_prompt(
                user_research_prompt=user_research_prompt,
                papers_summary=all_papers,
                volume_organization=organization.model_dump()
            )
            from backend.shared.solution_path.integration import with_validator_hook
            prompt = with_validator_hook(
                prompt, getattr(self, "solution_path_manager", None)
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_validator_max_input_tokens()
            
            if prompt_tokens > max_input:
                logger.error(f"VolumeOrganizer: Validation prompt too large ({prompt_tokens} > {max_input})")
                return False, "Validation prompt exceeds context limit"
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            logger.info(f"VolumeOrganizer: Validating organization (task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=f"{self.role_id}_validator",
                model=self.validator_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.validator_max_output_tokens,
                temperature=0.0
            )
            
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            if not response:
                return False, "Empty response from validator"
            
            # Extract content
            message = response.get("choices", [{}])[0].get("message", {})
            content = extract_message_text(message)
            if not content:
                return False, "No content in validator response"
            
            # Parse JSON using central utility
            data = parse_json(content)
            from backend.shared.solution_path.integration import enqueue_optional_update
            decision = data.get("decision", "reject")
            await enqueue_optional_update(
                data,
                getattr(self, "solution_path_manager", None),
                proposer_role=f"{self.role_id}_validator",
                source_task_id=task_id,
                source_phase="volume_organization_validation",
                source_decision=decision if decision in {"accept", "reject"} else None,
            )
            reasoning = data.get("reasoning", "No reasoning provided")
            
            return decision == "accept", reasoning
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            if _is_tier3_model_call_failure(e):
                raise
            logger.error(f"VolumeOrganizer: Error validating organization: {e}")
            return False, str(e)
    
    def get_writing_order(self, volume: VolumeOrganization) -> List[VolumeChapter]:
        """
        Get the order in which chapters should be written.
        
        Writing order:
        1. Gap papers (body chapters) in order
        2. Conclusion paper
        3. Introduction paper
        
        Existing papers are skipped (already written).
        """
        if not volume or not volume.chapters:
            return []
        
        chapters_to_write = []
        
        # First, gap papers in order
        gap_papers = sorted(
            [ch for ch in volume.chapters if ch.chapter_type == "gap_paper"],
            key=lambda x: x.order
        )
        chapters_to_write.extend(gap_papers)
        
        # Then conclusion
        conclusion = [ch for ch in volume.chapters if ch.chapter_type == "conclusion"]
        chapters_to_write.extend(conclusion)
        
        # Finally introduction
        intro = [ch for ch in volume.chapters if ch.chapter_type == "introduction"]
        chapters_to_write.extend(intro)
        
        return chapters_to_write

