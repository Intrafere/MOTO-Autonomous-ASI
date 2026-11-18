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
from typing import Optional, List, Dict, Any, Callable, Tuple

from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.model_output_contract import (
    ModelOutputContractError,
    is_model_output_contract_error,
)
from backend.shared.utils import count_tokens
from backend.shared.config import rag_config
from backend.shared.models import CertaintyAssessment
from backend.shared.prompt_feedback_budget import fit_prompt_with_feedback
from backend.shared.provider_errors import ProviderContextLengthError
from backend.autonomous.prompts.final_answer_prompts import (
    build_certainty_assessment_prompt,
    build_certainty_validation_prompt
)
from backend.autonomous.memory.paper_library import paper_library
from backend.autonomous.memory.final_answer_memory import final_answer_memory
from backend.autonomous.core.autonomous_rag_manager import autonomous_rag_manager
from backend.autonomous.agents.output_contract import (
    ContractFailurePacer,
    json_object,
    parse_with_contract_retry,
    never_stop,
    report_model_output_contract_failure,
    require_bool,
    require_choice,
    require_string_list,
    require_text,
    response_text,
)

logger = logging.getLogger(__name__)

_CERTAINTY_LEVELS = (
    "total_answer",
    "partial_answer",
    "no_answer_known",
    "appears_impossible",
    "other",
)

_ASSESSMENT_SHAPE = """{
  "certainty_level": "total_answer | partial_answer | no_answer_known | appears_impossible | other",
  "known_certainties_summary": "string",
  "reasoning": "string"
}"""

_EXPANSION_SHAPE = """{
  "expand_papers": ["paper_id", ...],
  "proceed_without_expansion": false,
  "reasoning": "string"
}"""

_VALIDATION_SHAPE = """{
  "decision": "accept | reject",
  "reasoning": "string"
}"""


def _parse_validation(data: Any, *, role: str) -> Tuple[str, str, Dict[str, Any]]:
    data = json_object(data, role=role)
    decision = require_choice(data, "decision", ("accept", "reject"), role=role)
    return decision, require_text(data, "reasoning", role=role), data


def _parse_assessment(data: Any, *, role: str) -> CertaintyAssessment:
    data = json_object(data, role=role)
    return CertaintyAssessment(
        certainty_level=require_choice(data, "certainty_level", _CERTAINTY_LEVELS, role=role),
        known_certainties_summary=require_text(data, "known_certainties_summary", role=role),
        reasoning=require_text(data, "reasoning", role=role),
    )


def _parse_expansion(data: Any, *, role: str) -> List[str]:
    data = json_object(data, role=role)
    proceed_without_expansion = require_bool(data, "proceed_without_expansion", role=role)
    expand_papers = require_string_list(data, "expand_papers", role=role)
    return [] if proceed_without_expansion else expand_papers


def _is_tier3_model_call_failure(exc: Exception) -> bool:
    if is_model_output_contract_error(exc):
        return False
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
        # Bound to the owning workflow's stop state so contract backoff ends on Stop.
        self.should_stop: Callable[[], bool] = never_stop
    
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
        rejection_entries = tuple((await final_answer_memory.get_rejections("assessment"))[-5:])
        # An assessment whose validator reply was unusable is validated again rather
        # than regenerated or recorded as a rejection.
        pending_assessment: Optional[CertaintyAssessment] = None
        contract_pacer = ContractFailurePacer(self.should_stop)

        while attempt < self.MAX_RETRIES:
            attempt += 1
            logger.info(f"CertaintyAssessor: Assessment attempt {attempt}/{self.MAX_RETRIES}")

            # Generate assessment
            assessment = pending_assessment
            pending_assessment = None
            if assessment is None:
                try:
                    assessment = await self._generate_assessment(
                        user_research_prompt,
                        all_papers,
                        expanded_papers,
                        rejection_entries
                    )
                except ModelOutputContractError as exc:
                    await report_model_output_contract_failure(
                        self.role_id,
                        exc,
                        model=self.submitter_model,
                        outcome="No certainty assessment was proposed; the assessment runs again.",
                    )
                    await contract_pacer.after_failure(self.role_id)
                    if self.should_stop():
                        return None
                    # A contract failure is operational: it never uses up one of the
                    # fixed tries, so re-asks continue until a usable reply or Stop.
                    attempt -= 1
                    continue
                contract_pacer.reset()

            if assessment is None:
                logger.error(f"CertaintyAssessor: Failed to generate assessment (attempt {attempt})")
                continue

            # Validate assessment
            try:
                is_valid, feedback = await self._validate_assessment(
                    user_research_prompt,
                    all_papers,
                    assessment,
                    expanded_papers,
                )
            except ModelOutputContractError as exc:
                await report_model_output_contract_failure(
                    f"{self.role_id}_validator",
                    exc,
                    model=self.validator_model,
                    outcome=(
                        "The certainty assessment was not judged and will be validated again; "
                        "no rejection was recorded."
                    ),
                )
                pending_assessment = assessment
                await contract_pacer.after_failure(f"{self.role_id}_validator")
                if self.should_stop():
                    return None
                # A contract failure is operational: it never uses up one of the
                # fixed tries, so re-asks continue until a usable reply or Stop.
                attempt -= 1
                continue
            contract_pacer.reset()

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
                rejection_entries = tuple((await final_answer_memory.get_rejections("assessment"))[-5:])
        
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

            try:
                expand_papers = await parse_with_contract_retry(
                    response_text(response),
                    lambda parsed: _parse_expansion(parsed, role=self.role_id),
                    prompt=prompt,
                    task_id=task_id,
                    role_id=self.role_id,
                    model=self.submitter_model,
                    context_window=self.context_window,
                    max_output_tokens=self.max_output_tokens,
                    response_noun="paper expansion request",
                    response_shape=_EXPANSION_SHAPE,
                )
            except ModelOutputContractError as exc:
                await report_model_output_contract_failure(
                    self.role_id,
                    exc,
                    task_id=task_id,
                    model=self.submitter_model,
                    outcome=(
                        "No papers were chosen for full review; the certainty assessment "
                        "continues from paper abstracts and outlines."
                    ),
                )
                return []
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
        rejection_entries=(),
    ) -> Optional[CertaintyAssessment]:
        """Generate certainty assessment from papers."""
        try:
            max_input = self._calculate_max_input_tokens()
            from backend.shared.solution_path.integration import with_budgeted_solver_plan

            active_expanded_papers = expanded_papers

            def build_prompt(entries):
                prompt = build_certainty_assessment_prompt(
                    user_research_prompt=user_research_prompt,
                    papers_summary=all_papers,
                    expanded_papers=active_expanded_papers,
                    rejection_context=final_answer_memory.render_rejection_context(
                        "assessment", entries
                    ),
                )
                return with_budgeted_solver_plan(
                    prompt, getattr(self, "solution_path_manager", None), max_input
                )

            fit = fit_prompt_with_feedback(
                rejection_entries,
                build_prompt=build_prompt,
                available_tokens=max_input,
            )
            prompt = fit.prompt
            retained_rejections = fit.retained_entries
            prompt_tokens = fit.prompt_tokens
            
            if prompt_tokens > max_input:
                if expanded_papers:
                    # RAG the expanded papers instead of dropping them entirely
                    active_expanded_papers = None
                    base_fit = fit_prompt_with_feedback(
                        retained_rejections,
                        build_prompt=build_prompt,
                        available_tokens=max_input,
                    )
                    base_prompt = base_fit.prompt
                    retained_rejections = base_fit.retained_entries
                    mandatory_tokens = base_fit.prompt_tokens
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
                            active_expanded_papers = rag_papers
                            rag_fit = fit_prompt_with_feedback(
                                retained_rejections,
                                build_prompt=build_prompt,
                                available_tokens=max_input,
                            )
                            prompt = rag_fit.prompt
                            retained_rejections = rag_fit.retained_entries
                            prompt_tokens = rag_fit.prompt_tokens
                        else:
                            logger.warning("CertaintyAssessor: RAG returned empty, falling back to abstracts-only")
                            prompt = base_prompt
                            prompt_tokens = mandatory_tokens
                    else:
                        logger.warning("CertaintyAssessor: Insufficient budget for RAG, using abstracts-only")
                        prompt = base_prompt
                        prompt_tokens = mandatory_tokens
                else:
                    active_expanded_papers = None
                    summary_fit = fit_prompt_with_feedback(
                        retained_rejections,
                        build_prompt=build_prompt,
                        available_tokens=max_input,
                    )
                    prompt = summary_fit.prompt
                    retained_rejections = summary_fit.retained_entries
                    prompt_tokens = summary_fit.prompt_tokens
                
                if prompt_tokens > max_input:
                    logger.error("CertaintyAssessor: Cannot fit even summary-only prompt")
                    return None
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
            
            while True:
                try:
                    response = await api_client_manager.generate_completion(
                        task_id=task_id,
                        role_id=self.role_id,
                        model=self.submitter_model,
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
            
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)

            return await parse_with_contract_retry(
                response_text(response),
                lambda parsed: _parse_assessment(parsed, role=self.role_id),
                prompt=prompt,
                task_id=task_id,
                role_id=self.role_id,
                model=self.submitter_model,
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens,
                response_noun="certainty assessment",
                response_shape=_ASSESSMENT_SHAPE,
            )

        except FreeModelExhaustedError:
            raise
        except ModelOutputContractError:
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

        Raises:
            ModelOutputContractError: the validator reply did not match its JSON
                contract after one bounded retry (not a semantic rejection).
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

            validator_role = f"{self.role_id}_validator"
            decision, reasoning, data = await parse_with_contract_retry(
                response_text(response),
                lambda parsed: _parse_validation(parsed, role=validator_role),
                prompt=prompt,
                task_id=task_id,
                role_id=validator_role,
                model=self.validator_model,
                context_window=self.validator_context_window,
                max_output_tokens=self.validator_max_output_tokens,
                response_noun="validation decision",
                response_shape=_VALIDATION_SHAPE,
            )
            from backend.shared.solution_path.integration import enqueue_optional_update
            await enqueue_optional_update(
                data,
                getattr(self, "solution_path_manager", None),
                proposer_role=validator_role,
                source_task_id=task_id,
                source_phase="certainty_validation",
                source_decision=decision,
            )

            return decision == "accept", reasoning

        except FreeModelExhaustedError:
            raise
        except ModelOutputContractError:
            raise
        except Exception as e:
            if _is_tier3_model_call_failure(e):
                raise
            logger.error(f"CertaintyAssessor: Error validating assessment: {e}")
            return False, str(e)

