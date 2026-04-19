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
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, Callable

from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens
from backend.shared.models import (
    CertaintyAssessment,
    VolumeOrganization,
    VolumeChapter,
    VolumeOrganizationSubmission
)
from backend.autonomous.prompts.final_answer_prompts import (
    build_volume_organization_prompt,
    build_volume_validation_prompt
)
from backend.autonomous.memory.final_answer_memory import final_answer_memory

logger = logging.getLogger(__name__)


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
        context_window: int = 131072,
        max_output_tokens: int = 25000
    ):
        self.submitter_model = submitter_model
        self.validator_model = validator_model
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        
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
        return self.context_window - self.max_output_tokens
    
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
        
        # Force completion on max iterations
        logger.warning(f"VolumeOrganizer: Forcing completion at iteration {self.MAX_ITERATIONS}")
        
        if current_volume:
            organization = VolumeOrganization(
                volume_title=current_volume.get("volume_title", "Research Volume"),
                chapters=[VolumeChapter(**ch) for ch in current_volume.get("chapters", [])],
                outline_complete=True,
                revision_reasoning="Forced completion after maximum iterations"
            )
            await final_answer_memory.save_volume_organization(organization)
            return organization
        
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
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_max_input_tokens()
            
            if prompt_tokens > max_input:
                logger.error(f"VolumeOrganizer: Prompt too large ({prompt_tokens} > {max_input})")
                return None
            
            # Generate task ID
            task_id = self.get_current_task_id()
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
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return None
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            # Parse chapters
            chapters = []
            for ch in data.get("chapters", []):
                chapter = VolumeChapter(
                    chapter_type=ch.get("chapter_type", "existing_paper"),
                    paper_id=ch.get("paper_id"),
                    title=ch.get("title", "Untitled Chapter"),
                    order=ch.get("order", len(chapters) + 1),
                    status=ch.get("status", "pending"),
                    description=ch.get("description", "")
                )
                chapters.append(chapter)
            
            # Ensure we have introduction and conclusion
            has_intro = any(ch.chapter_type == "introduction" for ch in chapters)
            has_conclusion = any(ch.chapter_type == "conclusion" for ch in chapters)
            
            if not has_intro:
                chapters.insert(0, VolumeChapter(
                    chapter_type="introduction",
                    title="Introduction",
                    order=1,
                    description="Introduction to the volume"
                ))
            
            if not has_conclusion:
                chapters.append(VolumeChapter(
                    chapter_type="conclusion",
                    title="Conclusion",
                    order=len(chapters) + 1,
                    description="Conclusion of the volume"
                ))
            
            # Fix ordering if needed
            chapters = self._normalize_chapter_order(chapters)
            
            return VolumeOrganization(
                volume_title=data.get("volume_title", "Research Volume"),
                chapters=chapters,
                outline_complete=data.get("outline_complete", False),
                revision_reasoning=data.get("reasoning", "")
            )
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
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
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_max_input_tokens()
            
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

