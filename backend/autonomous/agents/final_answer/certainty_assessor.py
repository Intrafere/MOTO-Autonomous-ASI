"""
Certainty Assessor Agent - Phase 1 of Tier 3 Final Answer Generation.

Assesses what can be answered WITH CERTAINTY from existing Tier 2 research papers.
Uses a two-step workflow similar to reference selection:
1. Browse paper abstracts and request expansion
2. Review full content and assess certainties

CRITICAL: Operates ONLY on Tier 2 papers, NOT on Tier 1 brainstorm databases.

NO RAG FOR ABSTRACTS (by design): Step 1 browses abstracts/outlines which are small metadata.
EXPANDED PAPERS OVERFLOW: Step 2 uses RAG fallback for expanded papers when full direct injection does not fit.
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
from backend.shared.models import CertaintyAssessment
from backend.autonomous.prompts.final_answer_prompts import (
    build_certainty_assessment_prompt,
    build_certainty_validation_prompt
)
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.final_answer_memory import final_answer_memory
from backend.autonomous.core.autonomous_rag_manager import autonomous_rag_manager

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


class CertaintyAssessor:
    """
    Agent that assesses what can be answered with certainty from existing papers.
    
    Phase 1 of Tier 3 workflow:
    1. Browse all Tier 2 paper abstracts/outlines
    2. Request expansion of papers that seem relevant
    3. Assess what can be answered with certainty
    4. Get assessment validated
    5. Retry on rejection with feedback
    """
    
    MAX_RETRIES = 10  # Maximum retries with rejection feedback
    
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
        self.role_id = "autonomous_certainty_assessor"
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
    
    async def assess_certainty(
        self,
        user_research_prompt: str,
        all_papers: List[Dict[str, Any]]
    ) -> Optional[CertaintyAssessment]:
        """
        Complete certainty assessment workflow with validation loop.
        
        Args:
            user_research_prompt: The user's original research question
            all_papers: List of all Tier 2 papers with metadata (title, abstract, outline, word_count)
        
        Returns:
            Validated CertaintyAssessment or None if failed after max retries
        """
        if not all_papers:
            logger.warning("CertaintyAssessor: No papers available for assessment")
            return CertaintyAssessment(
                certainty_level="no_answer_known",
                known_certainties_summary="No research papers available to assess.",
                reasoning="The system has not generated any research papers yet."
            )
        
        logger.info(f"CertaintyAssessor: Starting assessment with {len(all_papers)} papers")
        
        # Step 1: Browse abstracts and request which papers to expand
        papers_to_review = await self._request_paper_expansion(
            user_research_prompt,
            all_papers
        )
        
        # Step 2: Get expanded paper content
        expanded_papers = None
        if papers_to_review:
            expanded_papers = await self._get_expanded_papers(papers_to_review, all_papers)
        
        # Step 3: Assess certainties with validation loop
        attempt = 0
        rejection_context = ""
        
        while attempt < self.MAX_RETRIES:
            attempt += 1
            logger.info(f"CertaintyAssessor: Assessment attempt {attempt}/{self.MAX_RETRIES}")
            
            # Generate assessment
            assessment = await self._generate_assessment(
                user_research_prompt,
                all_papers,
                expanded_papers,
                rejection_context
            )
            
            if assessment is None:
                logger.error(f"CertaintyAssessor: Failed to generate assessment (attempt {attempt})")
                continue
            
            # Validate assessment
            is_valid, feedback = await self._validate_assessment(
                user_research_prompt,
                all_papers,
                assessment,
                expanded_papers,
            )
            
            if is_valid:
                logger.info(f"CertaintyAssessor: Assessment validated: {assessment.certainty_level}")
                await final_answer_memory.save_certainty_assessment(assessment)
                return assessment
            else:
                # Log rejection and prepare for retry
                logger.info(f"CertaintyAssessor: Assessment rejected: {feedback[:100]}...")
                await final_answer_memory.add_rejection(
                    phase="assessment",
                    rejection_summary=feedback,
                    submission_preview=assessment.known_certainties_summary[:500]
                )
                rejection_context = await final_answer_memory.get_rejection_context_async("assessment")
        
        logger.error(f"CertaintyAssessor: Failed after {self.MAX_RETRIES} attempts")
        return None
    
    async def _request_paper_expansion(
        self,
        user_research_prompt: str,
        all_papers: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Request which papers to expand for full review.
        Shows all paper abstracts and asks which need full content.
        """
        try:
            # Build prompt showing all abstracts
            prompt = self._build_expansion_request_prompt(
                user_research_prompt,
                all_papers
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
            
            if prompt_tokens > max_input:
                logger.warning(f"CertaintyAssessor: Expansion prompt too large ({prompt_tokens} > {max_input}). "
                              "Proceeding with abstract-only assessment.")
                return []
            
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            logger.info(f"CertaintyAssessor: Requesting paper expansion (task_id={task_id})")
            
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
                return []
            
            # Extract content
            message = response.get("choices", [{}])[0].get("message", {})
            content = extract_message_text(message)
            if not content:
                return []
            
            # Parse JSON using central utility
            data = parse_json(content)
            
            if data.get("proceed_without_expansion", False):
                logger.info("CertaintyAssessor: Proceeding with abstract-only assessment")
                return []
            
            expand_papers = data.get("expand_papers", [])
            logger.info(f"CertaintyAssessor: Requested expansion of {len(expand_papers)} papers")
            return expand_papers
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            if _is_tier3_model_call_failure(e):
                raise
            logger.error(f"CertaintyAssessor: Error requesting expansion: {e}")
            return []
    
    def _build_expansion_request_prompt(
        self,
        user_research_prompt: str,
        papers: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for paper expansion request."""
        parts = [
            """You are reviewing research papers to assess what can be answered WITH CERTAINTY about the user's question.

YOUR TASK:
Review the paper abstracts and outlines below. Decide which papers you need to see in FULL CONTENT 
to accurately assess the strongest defensible answer and the evidence status of its components.
Expand papers when full details are needed to evaluate a mechanism, evidence or derivation,
implementation, risk or limitation, validation method, theorem, or proof.
Do not treat a proposed invention or experiment as demonstrated without the required evidence.

You may expand as many papers as needed for a thorough assessment.
If the abstracts/outlines provide sufficient information, you may proceed without expansion.

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash
2. Quotes: Escape double quotes as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your response ONLY as JSON:
{
  "expand_papers": ["paper_id_1", "paper_id_2", ...],
  "proceed_without_expansion": false,
  "reasoning": "Why these papers need full content OR why abstracts suffice"
}

---
USER'S RESEARCH QUESTION:
""",
            user_research_prompt,
            "\n---\nAVAILABLE RESEARCH PAPERS:\n"
        ]
        
        for p in papers:
            parts.append(f"\n--- Paper ID: {p.get('paper_id', 'Unknown')} ---")
            parts.append(f"\nTitle: {p.get('title', 'N/A')}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
            if p.get('outline'):
                parts.append(f"\nOutline:\n{p.get('outline')}")
            parts.append(f"\nWord Count: {p.get('word_count', 0)}")
            parts.append("\n")
        
        parts.append("\n---\nWhich papers need full content for certainty assessment? (respond as JSON):")
        
        return "".join(parts)
    
    async def _get_expanded_papers(
        self,
        paper_ids: List[str],
        all_papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get full content of requested papers."""
        expanded = []
        
        for paper_id in paper_ids:
            content = await paper_library.get_paper_content(
                paper_id,
                strip_proofs=True,
            )
            outline = await paper_library.get_outline(paper_id)
            
            if content:
                # Find metadata
                paper_meta = next(
                    (p for p in all_papers if p.get("paper_id") == paper_id),
                    {}
                )
                
                expanded.append({
                    "paper_id": paper_id,
                    "title": paper_meta.get("title", "Unknown"),
                    "abstract": paper_meta.get("abstract", ""),
                    "word_count": paper_meta.get("word_count", len(content.split())),
                    "content": content,
                    "outline": outline or ""
                })
            else:
                logger.warning(f"CertaintyAssessor: Could not get content for paper {paper_id}")
        
        return expanded
    
    async def _generate_assessment(
        self,
        user_research_prompt: str,
        all_papers: List[Dict[str, Any]],
        expanded_papers: List[Dict[str, Any]] = None,
        rejection_context: str = ""
    ) -> Optional[CertaintyAssessment]:
        """Generate certainty assessment from papers."""
        try:
            # Build prompt
            prompt = build_certainty_assessment_prompt(
                user_research_prompt=user_research_prompt,
                papers_summary=all_papers,
                expanded_papers=expanded_papers,
                rejection_context=rejection_context
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_max_input_tokens()
            
            if prompt_tokens > max_input:
                if expanded_papers:
                    # RAG the expanded papers instead of dropping them entirely
                    base_prompt = build_certainty_assessment_prompt(
                        user_research_prompt=user_research_prompt,
                        papers_summary=all_papers,
                        expanded_papers=None,
                        rejection_context=rejection_context
                    )
                    mandatory_tokens = count_tokens(base_prompt)
                    paper_budget = max_input - mandatory_tokens - 500
                    
                    if paper_budget > 2000:
                        logger.info(f"CertaintyAssessor: RAG fallback for expanded papers (budget={paper_budget}t)")
                        paper_ids = [p["paper_id"] for p in expanded_papers]
                        rag_content, _ = await autonomous_rag_manager.get_reference_papers_context(
                            paper_ids,
                            max_total_tokens=paper_budget,
                            query=user_research_prompt
                        )
                        
                        if rag_content:
                            rag_papers = [{
                                "paper_id": "rag_retrieved",
                                "title": f"RAG-retrieved content from {len(expanded_papers)} papers",
                                "content": rag_content
                            }]
                            prompt = build_certainty_assessment_prompt(
                                user_research_prompt=user_research_prompt,
                                papers_summary=all_papers,
                                expanded_papers=rag_papers,
                                rejection_context=rejection_context
                            )
                            prompt_tokens = count_tokens(prompt)
                        else:
                            logger.warning("CertaintyAssessor: RAG returned empty, falling back to abstracts-only")
                            prompt = base_prompt
                            prompt_tokens = mandatory_tokens
                    else:
                        logger.warning("CertaintyAssessor: Insufficient budget for RAG, using abstracts-only")
                        prompt = base_prompt
                        prompt_tokens = mandatory_tokens
                else:
                    prompt = build_certainty_assessment_prompt(
                        user_research_prompt=user_research_prompt,
                        papers_summary=all_papers,
                        expanded_papers=None,
                        rejection_context=rejection_context
                    )
                    prompt_tokens = count_tokens(prompt)
                
                if prompt_tokens > max_input:
                    logger.error("CertaintyAssessor: Cannot fit even summary-only prompt")
                    return None
            from backend.shared.solution_path.integration import with_budgeted_solver_plan
            prompt = with_budgeted_solver_plan(
                prompt, getattr(self, "solution_path_manager", None), max_input
            )
            prompt_tokens = count_tokens(prompt)
            
            task_id = self.get_current_task_id()
            await api_client_manager.prewarm_assistant_memory_context(
                task_id=task_id,
                role_id=self.role_id,
                prompt=prompt,
            )

            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            logger.info(f"CertaintyAssessor: Generating assessment (prompt={prompt_tokens}t, task_id={task_id})")
            
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
            if not isinstance(data, dict):
                raise ValueError("Certainty assessment must be a JSON object")
            certainty_level = data.get("certainty_level")
            summary = data.get("known_certainties_summary")
            reasoning = data.get("reasoning")
            if certainty_level not in {
                "total_answer",
                "partial_answer",
                "no_answer_known",
                "appears_impossible",
                "other",
            }:
                raise ValueError("Certainty assessment has an invalid or missing certainty_level")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Certainty assessment requires a non-empty known_certainties_summary")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("Certainty assessment requires non-empty reasoning")

            return CertaintyAssessment(
                certainty_level=certainty_level,
                known_certainties_summary=summary.strip(),
                reasoning=reasoning.strip(),
            )
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            if _is_tier3_model_call_failure(e):
                raise
            logger.error(f"CertaintyAssessor: Error generating assessment: {e}")
            return None
    
    async def _validate_assessment(
        self,
        user_research_prompt: str,
        all_papers: List[Dict[str, Any]],
        assessment: CertaintyAssessment,
        expanded_papers: List[Dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """
        Validate the certainty assessment.
        
        Returns:
            Tuple of (is_valid, feedback_if_rejected)
        """
        try:
            # Build validation prompt
            prompt = build_certainty_validation_prompt(
                user_research_prompt=user_research_prompt,
                papers_summary=all_papers,
                assessment=assessment.model_dump(),
                expanded_papers=expanded_papers,
            )
            from backend.shared.solution_path.integration import with_validator_hook
            prompt = with_validator_hook(
                prompt, getattr(self, "solution_path_manager", None)
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_input = self._calculate_validator_max_input_tokens()
            
            if prompt_tokens > max_input:
                logger.error(f"CertaintyAssessor: Validation prompt too large ({prompt_tokens} > {max_input})")
                return False, "Validation prompt exceeds context limit"
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            logger.info(f"CertaintyAssessor: Validating assessment (task_id={task_id})")
            
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
                source_phase="certainty_validation",
                source_decision=decision if decision in {"accept", "reject"} else None,
            )
            reasoning = data.get("reasoning", "No reasoning provided")
            
            return decision == "accept", reasoning
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            if _is_tier3_model_call_failure(e):
                raise
            logger.error(f"CertaintyAssessor: Error validating assessment: {e}")
            return False, str(e)

