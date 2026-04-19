"""
Reference Selector Agent - Selects reference papers for brainstorming and paper compilation.
Implements two-step workflow: show abstracts -> expand -> final selection.

Supports two modes:
- "initial": Pre-brainstorm selection (select papers to inform brainstorm exploration)
- "additional": Pre-paper selection (select additional papers, keeping already-selected ones)

This is the crucial mechanism that enables COMPOUNDING KNOWLEDGE across research cycles.
By selecting reference papers before brainstorming, submitters can:
- Build upon proven mathematical frameworks from prior papers
- Avoid re-exploring territory already covered in depth
- Identify novel connections between new topics and established results
- Accelerate convergence on valuable insights by standing on prior work

CONTEXT HANDLING:
- Uses DIRECT INJECTION FIRST, RAG SECOND principle
- Abstracts phase: Direct inject all abstracts (typically small)
- Expansion phase: Direct inject full papers if they fit, else use RAG
- Validates prompt size before sending to prevent overflow
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config, system_config
from backend.shared.models import ReferenceExpansionRequest, ReferenceSelectionResult
from backend.autonomous.prompts.paper_reference_prompts import (
    build_reference_expansion_prompt,
    build_reference_selection_prompt,
    build_pre_brainstorm_expansion_prompt,
    build_additional_reference_expansion_prompt
)
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.core.autonomous_rag_manager import autonomous_rag_manager

logger = logging.getLogger(__name__)


class ReferenceSelectorAgent:
    """
    Agent that selects reference papers for paper compilation.
    Two-step workflow: abstracts review -> expansion request -> final selection.
    
    Context handling:
    - Direct injects paper abstracts (small metadata)
    - Direct injects full papers when they fit
    - Uses RAG for papers that don't fit in context
    - Validates prompt size before sending
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
        self.role_id = "autonomous_reference_selector"
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
    
    async def select_references(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        available_papers: List[Dict[str, Any]],
        mode: str = "initial",
        already_selected: List[str] = None,
        already_selected_papers: List[Dict[str, Any]] = None,
        max_total_papers: Optional[int] = None,
    ) -> List[str]:
        """
        Complete reference selection workflow.
        
        This is the crucial mechanism that enables COMPOUNDING KNOWLEDGE across research cycles.
        
        Args:
            user_research_prompt: The user's high-level research goal
            topic_prompt: The brainstorm topic prompt
            brainstorm_summary: Summary of brainstorm content
            available_papers: List of papers with title, abstract, word count
            mode: Selection mode - "initial" (pre-brainstorm) or "additional" (pre-paper)
            already_selected: List of paper_ids already selected (for "additional" mode)
            max_total_papers: Total paper cap for this workflow. In "additional" mode,
                already selected papers count toward this total.
        
        Returns:
            List of selected paper_ids, capped by the caller's policy
        """
        if already_selected is None:
            already_selected = []
        if already_selected_papers is None:
            already_selected_papers = []
        if max_total_papers is None:
            max_total_papers = system_config.autonomous_tier3_short_form_max_reference_papers
        
        if not available_papers:
            logger.info(f"ReferenceSelector [{mode}]: No papers available, skipping reference selection")
            return []
        
        # Calculate max papers based on mode
        max_papers = max_total_papers if mode == "initial" else (max_total_papers - len(already_selected))
        if max_papers <= 0:
            logger.info(
                f"ReferenceSelector [{mode}]: Already at max capacity "
                f"({len(already_selected)} of {max_total_papers} papers)"
            )
            return []
        
        logger.info(
            f"ReferenceSelector [{mode}]: Starting selection "
            f"(limit={max_papers}, total_cap={max_total_papers}, "
            f"{len(available_papers)} available, {len(already_selected)} already selected)"
        )
        
        # Step 1: Show abstracts and ask which to expand
        expansion_request = await self._request_expansion(
            user_research_prompt,
            topic_prompt,
            brainstorm_summary,
            available_papers,
            mode=mode,
            already_selected=already_selected,
            already_selected_papers=already_selected_papers,
            max_total_papers=max_total_papers,
        )
        
        if expansion_request is None:
            logger.error(f"ReferenceSelector [{mode}]: Failed to get expansion request")
            return []
        
        # Check if proceeding without references
        if expansion_request.proceed_without_references:
            logger.info(f"ReferenceSelector [{mode}]: Proceeding without references")
            return []
        
        if not expansion_request.expand_papers:
            logger.info(f"ReferenceSelector [{mode}]: No papers to expand, proceeding without references")
            return []
        
        # Step 2: Get full content of requested papers
        expanded_papers = await self._get_expanded_papers(
            expansion_request.expand_papers,
            available_papers
        )
        
        if not expanded_papers:
            logger.warning(f"ReferenceSelector [{mode}]: Could not expand any papers")
            return []
        
        # Step 3: Final selection (respecting max_papers limit)
        selected_papers = await self._make_final_selection(
            user_research_prompt,
            topic_prompt,
            brainstorm_summary,
            expanded_papers,
            mode=mode,
            max_papers=max_papers
        )
        
        return selected_papers
    
    async def _request_expansion(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        papers_with_abstracts: List[Dict[str, Any]],
        mode: str = "initial",
        already_selected: List[str] = None,
        already_selected_papers: List[Dict[str, Any]] = None,
        max_total_papers: int = 6,
    ) -> Optional[ReferenceExpansionRequest]:
        """
        Request which papers to expand (Step 1: abstracts only).
        Direct injects all abstracts since they're small metadata.
        
        Args:
            mode: "initial" for pre-brainstorm, "additional" for pre-paper
            already_selected: Papers already selected (shown in additional mode)
        """
        if already_selected is None:
            already_selected = []
        if already_selected_papers is None:
            already_selected_papers = []
        
        try:
            # Build prompt based on mode
            if mode == "initial":
                # Pre-brainstorm: select papers to inform exploration
                prompt = build_pre_brainstorm_expansion_prompt(
                    user_research_prompt=user_research_prompt,
                    topic_prompt=topic_prompt,
                    brainstorm_summary=brainstorm_summary,
                    papers_with_abstracts=papers_with_abstracts,
                    max_papers=max_total_papers,
                )
            else:
                # Additional: select more papers before paper writing
                prompt = build_additional_reference_expansion_prompt(
                    user_research_prompt=user_research_prompt,
                    topic_prompt=topic_prompt,
                    brainstorm_summary=brainstorm_summary,
                    papers_with_abstracts=papers_with_abstracts,
                    already_selected=already_selected,
                    already_selected_papers=already_selected_papers,
                    max_total_papers=max_total_papers,
                )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_max_input_tokens()
            
            if prompt_tokens > max_input:
                logger.error(f"ReferenceSelector: Expansion prompt ({prompt_tokens} tokens) exceeds limit ({max_input})")
                return None
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"ReferenceSelector: Requesting expansion with model {self.model_id} "
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
                logger.error("ReferenceSelector: Empty response from LLM")
                return None
            
            # Extract content (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return None
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            return ReferenceExpansionRequest(
                expand_papers=data.get("expand_papers", []),
                proceed_without_references=data.get("proceed_without_references", False),
                reasoning=data.get("reasoning", "")
            )
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"ReferenceSelector: Error requesting expansion: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
    
    async def _get_expanded_papers(
        self,
        paper_ids: List[str],
        available_papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get full content and outlines of requested papers."""
        expanded = []
        
        for paper_id in paper_ids:
            # Get full paper content
            content = await paper_library.get_paper_content(paper_id)
            
            # NEW: Also get outline
            outline = await paper_library.get_outline(paper_id)
            
            if content:
                # Find metadata from available_papers
                paper_meta = next(
                    (p for p in available_papers if p.get("paper_id") == paper_id),
                    {}
                )
                
                expanded.append({
                    "paper_id": paper_id,
                    "title": paper_meta.get("title", "Unknown"),
                    "reference_title_display": paper_meta.get(
                        "reference_title_display",
                        paper_meta.get("title", "Unknown"),
                    ),
                    "word_count": paper_meta.get("word_count", len(content.split())),
                    "content": content,
                    "outline": outline  # NEW: Include outline
                })
            else:
                logger.warning(f"ReferenceSelector: Could not get content for paper {paper_id}")
        
        return expanded
    
    async def _make_final_selection(
        self,
        user_research_prompt: str,
        topic_prompt: str,
        brainstorm_summary: str,
        expanded_papers: List[Dict[str, Any]],
        mode: str = "initial",
        max_papers: int = 6
    ) -> List[str]:
        """
        Make final selection of reference papers.
        
        Uses DIRECT INJECTION FIRST, RAG SECOND:
        - If all papers fit: direct inject full content
        - If papers don't fit: use RAG to get relevant sections
        
        Args:
            mode: "initial" for pre-brainstorm, "additional" for pre-paper
            max_papers: Maximum papers to select for this call
        """
        try:
            max_input = self._calculate_max_input_tokens()
            
            # Calculate total tokens for all expanded papers
            total_paper_tokens = sum(
                count_tokens(p.get("content", "")) for p in expanded_papers
            )
            
            # Reserve ~40% of context for papers, rest for prompts/brainstorm
            paper_budget = int(max_input * 0.4)
            
            if total_paper_tokens <= paper_budget:
                # All papers fit - use direct injection
                logger.info(f"ReferenceSelector [{mode}]: Direct injection for {len(expanded_papers)} papers "
                           f"({total_paper_tokens} tokens <= {paper_budget} budget)")
                papers_for_prompt = expanded_papers
            else:
                # Papers don't fit - use RAG retrieval
                logger.info(f"ReferenceSelector [{mode}]: Papers ({total_paper_tokens} tokens) exceed budget ({paper_budget}). "
                           f"Using RAG for relevant sections.")
                
                # Get paper content via RAG
                paper_ids = [p.get("paper_id") for p in expanded_papers]
                rag_content, _ = await autonomous_rag_manager.get_reference_papers_context(
                    paper_ids,
                    max_total_tokens=paper_budget,
                    query=f"{user_research_prompt} {topic_prompt}"
                )
                
                # Create modified papers list with RAG content
                papers_for_prompt = [{
                    "paper_id": "combined_rag",
                    "title": f"RAG-retrieved content from {len(expanded_papers)} papers",
                    "content": rag_content
                }]
            
            # Build prompt with prepared papers
            prompt = build_reference_selection_prompt(
                user_research_prompt=user_research_prompt,
                topic_prompt=topic_prompt,
                brainstorm_summary=brainstorm_summary,
                expanded_papers=papers_for_prompt,
                mode=mode,
                max_papers=max_papers
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            if prompt_tokens > max_input:
                logger.error(f"ReferenceSelector [{mode}]: Prompt ({prompt_tokens} tokens) still exceeds limit ({max_input})")
                # Fall back to selecting from abstracts only
                return [p.get("paper_id") for p in expanded_papers[:max_papers]]
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM via api_client_manager (handles boost and fallback)
            logger.info(f"ReferenceSelector [{mode}]: Making final selection with model {self.model_id} "
                       f"(prompt={prompt_tokens}t, max={max_papers} papers, task_id={task_id})")
            
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0  # Deterministic validation - evolving context provides diversity
            )
            
            if not response:
                logger.error(f"ReferenceSelector [{mode}]: Empty response for final selection")
                return []
            
            # Extract content (check both content and reasoning fields)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return []
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            selected = data.get("selected_papers", [])
            
            # Enforce max papers limit
            if len(selected) > max_papers:
                logger.warning(f"ReferenceSelector [{mode}]: Limiting selection from {len(selected)} to {max_papers} papers")
                selected = selected[:max_papers]
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"ReferenceSelector [{mode}]: Selected {len(selected)} reference papers")
            return selected
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"ReferenceSelector [{mode}]: Error making final selection: {e}")
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return []

