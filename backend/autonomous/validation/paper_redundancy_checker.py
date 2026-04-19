"""
Paper Redundancy Checker - Reviews paper library for redundancy.
Runs every 3 completed papers to maintain library quality.

NO RAG BY DESIGN: Redundancy is assessed at the abstract/title level — comparing
high-level paper topics to find overlap. Full paper content is not needed to detect
whether two papers cover the same ground. All inputs are compact metadata summaries.
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.models import PaperRedundancyReviewResult
from backend.autonomous.prompts.paper_redundancy_prompts import build_paper_redundancy_prompt
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.research_metadata import research_metadata

logger = logging.getLogger(__name__)


class PaperRedundancyChecker:
    """
    Reviews paper library for redundancy every 3 completed papers.
    Conservative approach - only removes if clearly redundant.
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
        self.role_id = "autonomous_paper_redundancy_checker"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"agg_val_{self.task_sequence:03d}"
    
    async def check_redundancy(
        self,
        user_research_prompt: str,
        papers_summary: List[Dict[str, Any]]
    ) -> Optional[PaperRedundancyReviewResult]:
        """
        Check paper library for redundancy.
        
        Args:
            user_research_prompt: The user's high-level research goal
            papers_summary: List of all papers with title, abstract, word count
        
        Returns:
            PaperRedundancyReviewResult with removal decision
        """
        if len(papers_summary) < 3:
            logger.info("PaperRedundancyChecker: Not enough papers to check (need at least 3)")
            return PaperRedundancyReviewResult(
                should_remove=False,
                paper_id=None,
                reasoning="Library has fewer than 3 papers - redundancy check not applicable"
            )
        
        try:
            # Build prompt
            prompt = build_paper_redundancy_prompt(
                user_research_prompt=user_research_prompt,
                papers_summary=papers_summary
            )
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"PaperRedundancyChecker: Checking redundancy with model {self.model_id} (task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0  # Deterministic validation - evolving context provides diversity
            )
            
            if not response:
                logger.error("PaperRedundancyChecker: Empty response from LLM")
                return self._create_no_removal("Empty response from LLM")
            
            # Extract content (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return self._create_no_removal("No content in response")
            
            # Parse JSON using central utility
            try:
                data = parse_json(content)
                
                should_remove = data.get("should_remove", False)
                paper_id = data.get("paper_id")
                reasoning = data.get("reasoning", "No reasoning provided")
                
                # Validate paper_id if removal recommended
                if should_remove:
                    if not paper_id:
                        logger.warning("PaperRedundancyChecker: Removal recommended but no paper_id")
                        return self._create_no_removal("No paper_id provided for removal")
                    
                    # Verify paper exists
                    valid_ids = [p.get("paper_id") for p in papers_summary]
                    if paper_id not in valid_ids:
                        logger.warning(f"PaperRedundancyChecker: Invalid paper_id: {paper_id}")
                        return self._create_no_removal(f"Invalid paper_id: {paper_id}")
                
                result = PaperRedundancyReviewResult(
                    should_remove=should_remove,
                    paper_id=paper_id if should_remove else None,
                    reasoning=reasoning
                )
                
                # Notify task completed successfully
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                
                if should_remove:
                    logger.info(f"PaperRedundancyChecker: Recommending removal of {paper_id}")
                else:
                    logger.info("PaperRedundancyChecker: No removal recommended")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"PaperRedundancyChecker: JSON parse error: {e}")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return self._create_no_removal(f"JSON parse error: {str(e)}")
                
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"PaperRedundancyChecker: Error during check: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return self._create_no_removal(f"Error: {str(e)}")
    
    async def execute_removal(self, paper_id: str) -> bool:
        """
        Execute paper removal by archiving it.
        
        Args:
            paper_id: ID of paper to remove
        
        Returns:
            True if removal successful
        """
        try:
            # Archive the paper
            success = await paper_library.archive_paper(paper_id)
            
            if success:
                # Update central metadata
                await research_metadata.archive_paper(paper_id)
                logger.info(f"PaperRedundancyChecker: Successfully archived paper {paper_id}")
            else:
                logger.error(f"PaperRedundancyChecker: Failed to archive paper {paper_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"PaperRedundancyChecker: Error executing removal: {e}")
            return False
    
    def _create_no_removal(self, reason: str) -> PaperRedundancyReviewResult:
        """Create a no-removal result with the given reason."""
        return PaperRedundancyReviewResult(
            should_remove=False,
            paper_id=None,
            reasoning=f"No removal (conservative default): {reason}"
        )
    
    def should_check(self, total_papers_completed: int, last_check_at: int) -> bool:
        """
        Determine if redundancy check should run.
        
        Args:
            total_papers_completed: Total completed papers
            last_check_at: Paper count when last check was performed
        
        Returns:
            True if check should run (every 3 papers)
        """
        # Check every 3 completed papers
        if total_papers_completed < 3:
            return False
        
        # Run if we've completed 3 more papers since last check
        papers_since_last = total_papers_completed - last_check_at
        return papers_since_last >= 3

