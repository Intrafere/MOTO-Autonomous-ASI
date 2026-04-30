"""
High-context submitter agent for compiler.
Handles 3 modes: construction, outline update, and review.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.models import CompilerSubmission
from backend.shared.config import system_config, rag_config
from backend.shared.utils import count_tokens
from backend.shared.json_parser import parse_json
from backend.autonomous.memory.proof_database import proof_database
from backend.aggregator.validation.json_validator import json_validator
from backend.compiler.prompts.outline_prompts import (
    build_outline_create_prompt,
    build_outline_update_prompt
)
from backend.compiler.prompts.construction_prompts import (
    build_construction_prompt,
    build_body_construction_prompt,
    build_conclusion_construction_prompt,
    build_introduction_construction_prompt,
    build_abstract_construction_prompt
)
from backend.compiler.prompts.review_prompts import build_review_prompt
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import (
    paper_memory,
    ABSTRACT_PLACEHOLDER,
    INTRO_PLACEHOLDER,
    CONCLUSION_PLACEHOLDER,
)
from backend.compiler.core.compiler_rag_manager import compiler_rag_manager

logger = logging.getLogger(__name__)


# =============================================================================
# WOLFRAM ALPHA TOOL (Phase 3)
# =============================================================================
# The main writer may invoke Wolfram Alpha as a real OpenAI-style tool during
# construction mode. Each submission gets a budget of 20 calls; the loop
# forces finalization once the budget is exhausted. Callers attach the full
# audit trail to `CompilerSubmission.metadata["wolfram_calls"]`.

WOLFRAM_MAX_CALLS_PER_SUBMISSION = 20

WOLFRAM_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "wolfram_alpha_query",
        "description": (
            "Query Wolfram Alpha to verify a mathematical or computational claim "
            "before writing it into the paper. Use for: numerical verifications, "
            "symbolic computations, well-known mathematical facts, unit "
            "conversions, named-constant values. Do NOT use for open research "
            "questions or narrative prose. You may call this tool up to "
            f"{WOLFRAM_MAX_CALLS_PER_SUBMISSION} times per submission."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language Wolfram Alpha query, e.g. 'Is pi "
                        "algebraic?', 'integral of x^2 from 0 to 1', "
                        "'prime factorization of 360'."
                    ),
                },
                "purpose": {
                    "type": "string",
                    "description": (
                        "Brief note on how the result will be used in the paper "
                        "(for audit trail)."
                    ),
                },
            },
            "required": ["query", "purpose"],
        },
    },
}


def _wolfram_tool_available() -> bool:
    """Return True iff Wolfram Alpha is configured AND its client is live.

    Registration of the tool with the LLM is gated on this so models never
    see a callable tool when the backend cannot actually service it.
    """
    if not system_config.wolfram_alpha_enabled:
        return False
    try:
        from backend.shared.wolfram_alpha_client import get_wolfram_client
    except ImportError:
        return False
    try:
        return get_wolfram_client() is not None
    except Exception:
        return False


def _normalize_string_field(value) -> str:
    """
    Normalize string field from LLM response.
    Some LLMs incorrectly return strings as lists.
    
    Args:
        value: Raw value from JSON (could be str, list, or other)
    
    Returns:
        Normalized string value
    """
    if isinstance(value, list):
        # LLM returned list - join into single string
        logger.warning(f"LLM returned field as list (length {len(value)}), converting to string")
        return " ".join(str(item) for item in value if item)
    elif isinstance(value, str):
        return value
    elif value is None:
        return ""
    else:
        # Fallback: convert to string
        logger.warning(f"LLM returned field as {type(value)}, converting to string")
        return str(value)


def _strip_paper_markers_for_llm(paper_content: str) -> str:
    """
    Prepare paper text before sending it to the LLM.
    
    The section placeholders are KEPT so the LLM can see and use them
    as exact old_string values for replacement operations.
    
    IMPORTANT: We do NOT replace placeholders with different text anymore.
    The prompts tell the LLM to use the exact placeholder text as old_string.
    If we replaced them with different labels, the LLM would generate
    old_string values that don't match the actual paper file.
    
    Args:
        paper_content: Full paper content with markers
    
    The writer must see the same editable paper text that exact-match
    validation checks. Keep placeholders and theorem appendix bracket markers
    visible so old_string anchors can be copied verbatim from the real paper.

    Returns:
        Paper content with all system markers intact
    """
    # Keep markers intact so LLM can use them as exact old_string values.
    return paper_content.strip()


class HighContextSubmitter:
    """
    High-context, low-parameter submitter for compiler.
    
    Modes:
    - outline_create: Generate initial outline
    - outline_update: Review and potentially update outline
    - construction: Write next portion of paper
    - review: Review paper for errors/improvements (no aggregator DB)
    """
    
    def __init__(self, model_name: str, user_prompt: str, websocket_broadcaster: Optional[Callable] = None):
        self.model_name = model_name
        self.user_prompt = proof_database.inject_into_prompt(user_prompt)
        self.websocket_broadcaster = websocket_broadcaster
        self._initialized = False
        
        # Calculate context budget (user-configurable, default 131072)
        self.context_window = system_config.compiler_high_context_context_window
        self.max_output_tokens = system_config.compiler_high_context_max_output_tokens
        self.available_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "compiler_high_context"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"comp_hc_{self.task_sequence:03d}"
    
    async def initialize(self) -> None:
        """Initialize submitter."""
        if self._initialized:
            return
        
        # Re-read context window from config (in case it was updated)
        self.context_window = system_config.compiler_high_context_context_window
        self.max_output_tokens = system_config.compiler_high_context_max_output_tokens
        self.available_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        
        self._initialized = True
        logger.info(f"High-context submitter initialized with model: {self.model_name}")
        logger.info(f"Context budget: {self.available_input_tokens} tokens (window: {self.context_window})")
    
    async def submit_outline_create(self) -> CompilerSubmission:
        """
        Create initial outline submission.
        
        Returns:
            CompilerSubmission for outline creation
        """
        logger.info("Starting outline creation submission generation...")
        
        try:
            # Retrieve aggregator database evidence
            logger.info("Retrieving aggregator database evidence via RAG...")
            context_pack = await compiler_rag_manager.retrieve_for_mode(
                query=self.user_prompt,
                mode="outline_create"
            )
            logger.info(f"RAG retrieval complete: {len(context_pack.text)} chars retrieved")
            
            # Build prompt
            logger.info("Building outline creation prompt...")
            prompt = await build_outline_create_prompt(
                user_prompt=self.user_prompt,
                rag_evidence=context_pack.text
            )
            logger.info(f"Prompt built: {len(prompt)} chars")
            
            # Validate prompt size
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(system_config.compiler_high_context_context_window, system_config.compiler_high_context_max_output_tokens)
            
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"outline_create: Assembled prompt ({actual_prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed_tokens} tokens after safety margin). This indicates a context allocation bug."
                )
                raise ValueError(f"Prompt too large: {actual_prompt_tokens} tokens > {max_allowed_tokens} max")
            
            logger.debug(f"outline_create prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Get completion via api_client_manager (handles boost and fallback)
            logger.info(f"Generating LLM completion via api_client_manager (task_id={task_id})...")
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic generation - evolving context provides diversity
                max_tokens=system_config.compiler_high_context_max_output_tokens  # User-configurable (outline creation, update, construction, review)
            )
            
            # Check for empty response
            if not response.get("choices") or not response["choices"][0].get("message"):
                logger.error("outline_create: LLM returned empty response structure")
                raise ValueError("LLM returned empty response")
            
            # Extract content from either 'content' or 'reasoning' field
            # Some reasoning models (e.g., DeepSeek R1, certain GPT variants) output JSON in 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            logger.info(f"LLM completion received: {len(llm_output)} chars")
            
            # Check for empty content
            if not llm_output or len(llm_output.strip()) == 0:
                logger.error("outline_create: LLM returned empty content in both 'content' and 'reasoning' fields")
                raise ValueError("LLM returned empty content")
            
            # Parse response with retry
            logger.info("Parsing JSON response...")
            data = await self._parse_json_response_with_retry(llm_output, "outline_create", prompt)
            
            if not data:
                raise ValueError("Failed to parse JSON response from outline creation")
            
            logger.info("JSON parsed successfully")
            
            # Handle case where model returns array instead of single object
            if isinstance(data, list):
                if len(data) == 0:
                    raise ValueError("Outline creation returned empty array")
                logger.warning(f"Outline creation returned array of {len(data)} objects, using first object only")
                data = data[0]
            
            # Validate required fields for outline_create (iterative refinement)
            if "outline_complete" not in data:
                logger.error("outline_create: Missing required 'outline_complete' field in JSON")
                raise ValueError("Missing 'outline_complete' field - this field is required for outline creation")
            
            outline_complete = data.get("outline_complete")
            if not isinstance(outline_complete, bool):
                logger.error(f"outline_create: 'outline_complete' must be boolean, got {type(outline_complete)}")
                raise ValueError(f"Invalid 'outline_complete' field type: {type(outline_complete)}")
            
            # Create submission
            # Content is already properly decoded by json.loads() - no additional processing needed
            content = data.get("content", "")
            
            submission = CompilerSubmission(
                submission_id=str(uuid.uuid4()),
                mode="outline_create",
                content=content,
                operation="full_content",  # Outline create uses full replacement
                old_string="",
                new_string=content,  # The complete outline
                reasoning=data.get("reasoning", ""),
                outline_complete=outline_complete,  # NEW: For iterative refinement
                metadata={"coverage": context_pack.coverage, "answerability": context_pack.answerability}
            )
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Outline creation submission generated: {submission.submission_id}, outline_complete={outline_complete}")
            return submission
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate outline creation submission: {e}", exc_info=True)
            # Notify task completed (failed but still completed)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            raise
    
    async def submit_outline_update(self) -> Optional[CompilerSubmission]:
        """
        Submit outline update (or no-op if update not needed).
        
        Returns:
            CompilerSubmission if update needed, None otherwise
        """
        logger.info("Starting outline update review...")
        
        try:
            # Get current outline and paper
            logger.info("Loading outline and paper state...")
            current_outline = await outline_memory.get_outline()
            current_paper = await paper_memory.get_paper()
            logger.info(f"State loaded: outline={len(current_outline)} chars, paper={len(current_paper)} chars")
            
            # Show the same marker-bearing paper that validation/apply will match.
            paper_for_llm = _strip_paper_markers_for_llm(current_paper)
            logger.info(f"Paper prepared for LLM: {len(current_paper)} chars → {len(paper_for_llm)} chars (markers preserved)")
            
            # Retrieve aggregator database evidence
            # Exclude outline and paper (both direct-injected in outline_update mode)
            logger.info("Retrieving aggregator database evidence via RAG...")
            context_pack = await compiler_rag_manager.retrieve_for_mode(
                query=self.user_prompt,
                mode="outline_update",
                exclude_sources=["compiler_outline.txt", "compiler_paper.txt"]
            )
            logger.info(f"RAG retrieval complete: {len(context_pack.text)} chars retrieved")
            
            # Build prompt
            logger.info("Building outline update prompt...")
            prompt = await build_outline_update_prompt(
                user_prompt=self.user_prompt,
                current_outline=current_outline,
                current_paper=paper_for_llm,
                rag_evidence=context_pack.text
            )
            logger.info(f"Prompt built: {len(prompt)} chars")
            
            # Validate prompt size
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(system_config.compiler_high_context_context_window, system_config.compiler_high_context_max_output_tokens)
            
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"outline_update: Assembled prompt ({actual_prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed_tokens} tokens after safety margin). This indicates a context allocation bug."
                )
                raise ValueError(f"Prompt too large: {actual_prompt_tokens} tokens > {max_allowed_tokens} max")
            
            logger.debug(f"outline_update prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Get completion via api_client_manager (handles boost and fallback)
            logger.info(f"Generating LLM completion via api_client_manager (task_id={task_id})...")
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic generation - evolving context provides diversity
                max_tokens=system_config.compiler_high_context_max_output_tokens  # User-configurable (outline creation, update, construction, review)
            )
            
            # Check for empty response
            if not response.get("choices") or not response["choices"][0].get("message"):
                logger.error("outline_update: LLM returned empty response structure")
                raise ValueError("LLM returned empty response")
            
            # Extract content from either 'content' or 'reasoning' field
            # Some reasoning models (e.g., DeepSeek R1, certain GPT variants) output JSON in 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            logger.info(f"LLM completion received: {len(llm_output)} chars")
            
            # Check for empty content
            if not llm_output or len(llm_output.strip()) == 0:
                logger.error("outline_update: LLM returned empty content in both 'content' and 'reasoning' fields")
                raise ValueError("LLM returned empty content")
            
            # Parse response with retry
            logger.info("Parsing JSON response...")
            data = await self._parse_json_response_with_retry(llm_output, "outline_update", prompt)
            
            if not data:
                raise ValueError("Failed to parse JSON response from outline update")
            
            logger.info("JSON parsed successfully")
            
            # Handle case where model returns array instead of single object
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Outline update returned empty array, treating as no update needed")
                    return None
                logger.warning(f"Outline update returned array of {len(data)} objects, using first object only")
                data = data[0]
            
            # Check if update needed
            needs_update = data.get("needs_update", False)
            
            if not needs_update:
                # Notify task completed even when no update needed
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                logger.info("Outline update not needed")
                return None
            
            # Create submission
            # Content is already properly decoded by json.loads() - no additional processing needed
            content = data.get("content", "")
            
            submission = CompilerSubmission(
                submission_id=str(uuid.uuid4()),
                mode="outline_update",
                content=content,
                operation=data.get("operation", "replace"),
                old_string=_normalize_string_field(data.get("old_string", "")),
                new_string=_normalize_string_field(data.get("new_string", "")),
                reasoning=data.get("reasoning", ""),
                metadata={}
            )
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Outline update submission generated: {submission.submission_id}")
            return submission
            
        except FreeModelExhaustedError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate outline update submission: {e}", exc_info=True)
            # Notify task completed (failed but still completed)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            raise
    
    async def submit_construction(
        self, 
        is_first_portion: bool = False, 
        section_phase: Optional[str] = None,
        rejection_feedback: Optional[str] = None,
        critique_feedback: Optional[str] = None,
        pre_critique_paper: Optional[str] = None,
        brainstorm_content: Optional[str] = None,
        brainstorm_source_name: Optional[str] = None
    ) -> Optional[CompilerSubmission]:
        """
        Submit next paper construction portion.
        
        Args:
            is_first_portion: Whether this is the first portion of the paper
            section_phase: Phase constraint for construction ("body", "conclusion", "introduction", "abstract")
                          When provided, uses phase-specific prompts with explicit section_complete feedback.
            rejection_feedback: Feedback from a previous rejection to guide the model (e.g., "Introduction not found in document")
            critique_feedback: Accepted critique feedback from peer review (for body rewrites only)
            pre_critique_paper: Paper state before critique phase (for body rewrites - shows what failed)
            brainstorm_content: Full brainstorm database with submission numbers (for retroactive corrections)
            brainstorm_source_name: RAG source name for brainstorm (e.g., "brainstorm_abc123.txt") to exclude from retrieval
        
        Returns:
            CompilerSubmission for construction
        """
        phase_info = f", phase={section_phase}" if section_phase else ""
        feedback_info = f", retry with feedback" if rejection_feedback else ""
        critique_info = f", rewrite with critique" if critique_feedback else ""
        logger.info(f"Starting construction submission generation (first={is_first_portion}{phase_info}{feedback_info}{critique_info})")
        
        try:
            # Get current outline and paper
            logger.info("Loading outline and paper state...")
            current_outline = await outline_memory.get_outline()
            current_paper = await paper_memory.get_paper()
            logger.info(f"State loaded: outline={len(current_outline)} chars, paper={len(current_paper)} chars")
            
            # Show the same marker-bearing paper that validation/apply will match.
            paper_for_llm = _strip_paper_markers_for_llm(current_paper)
            logger.info(f"Paper prepared for LLM: {len(current_paper)} chars → {len(paper_for_llm)} chars (markers preserved)")
            
            # Calculate RAG budget accounting for brainstorm content (prevents context overflow)
            max_allowed_tokens = rag_config.get_available_input_tokens(
                system_config.compiler_high_context_context_window,
                system_config.compiler_high_context_max_output_tokens
            )
            outline_tokens = count_tokens(current_outline)
            paper_tokens = count_tokens(paper_for_llm) if paper_for_llm else 0
            brainstorm_tokens = count_tokens(brainstorm_content) if brainstorm_content else 0
            system_overhead = 5000  # system prompt, JSON schema, headers, separators, rejection history
            
            reserved_tokens = outline_tokens + paper_tokens + brainstorm_tokens + system_overhead
            rag_budget = max(5000, max_allowed_tokens - reserved_tokens)
            
            if brainstorm_content and brainstorm_tokens > 0:
                logger.info(
                    f"Context budget: max={max_allowed_tokens}, outline={outline_tokens}, "
                    f"paper={paper_tokens}, brainstorm={brainstorm_tokens}, overhead={system_overhead}, "
                    f"rag_budget={rag_budget}"
                )
            
            # Retrieve aggregator database evidence
            # Exclude sources already direct-injected to prevent token waste
            exclude_sources = ["compiler_outline.txt", "compiler_paper.txt"]
            if brainstorm_source_name:
                exclude_sources.append(brainstorm_source_name)
            
            logger.info("Retrieving aggregator database evidence via RAG...")
            query = self.user_prompt
            if not is_first_portion and paper_for_llm:
                # Use last part of paper to guide next section
                query += " " + paper_for_llm[-500:]
            
            context_pack = await compiler_rag_manager.retrieve_for_mode(
                query=query,
                mode="construction",
                max_tokens=rag_budget,
                exclude_sources=exclude_sources
            )
            logger.info(f"RAG retrieval complete: {len(context_pack.text)} chars retrieved")
            
            # Build prompt based on section phase (uses phase-specific prompts for explicit completion tracking)
            logger.info(f"Building construction prompt for phase: {section_phase or 'generic'}...")
            
            if section_phase == "body":
                prompt = await build_body_construction_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=paper_for_llm,
                    rag_evidence=context_pack.text,
                    is_first_portion=is_first_portion,
                    rejection_feedback=rejection_feedback,
                    critique_feedback=critique_feedback,
                    pre_critique_paper=pre_critique_paper,
                    brainstorm_content=brainstorm_content
                )
            elif section_phase == "conclusion":
                prompt = await build_conclusion_construction_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=paper_for_llm,
                    rag_evidence=context_pack.text,
                    rejection_feedback=rejection_feedback,
                    brainstorm_content=brainstorm_content
                )
            elif section_phase == "introduction":
                prompt = await build_introduction_construction_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=paper_for_llm,
                    rag_evidence=context_pack.text,
                    rejection_feedback=rejection_feedback,
                    brainstorm_content=brainstorm_content
                )
            elif section_phase == "abstract":
                prompt = await build_abstract_construction_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=paper_for_llm,
                    rag_evidence=context_pack.text,
                    rejection_feedback=rejection_feedback,
                    brainstorm_content=brainstorm_content
                )
            else:
                # Fallback to generic prompt for backward compatibility
                prompt = await build_construction_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=paper_for_llm,
                    rag_evidence=context_pack.text,
                    is_first_portion=is_first_portion,
                    section_phase=section_phase,
                    rejection_feedback=rejection_feedback,
                    critique_feedback=critique_feedback,
                    pre_critique_paper=pre_critique_paper
                )
            logger.info(f"Prompt built: {len(prompt)} chars")
            
            # Validate prompt size (max_allowed_tokens already calculated above for RAG budget)
            actual_prompt_tokens = count_tokens(prompt)
            
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"construction: Assembled prompt ({actual_prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed_tokens} tokens after safety margin). This indicates a context allocation bug."
                )
                raise ValueError(
                    f"construction: Prompt too large ({actual_prompt_tokens} tokens > {max_allowed_tokens} max). "
                    f"Brainstorm={brainstorm_tokens} tokens, outline={outline_tokens}, paper={paper_tokens}, overhead={system_overhead}."
                )
            
            logger.debug(f"construction prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Get completion via api_client_manager with Wolfram tool-loop.
            # Phase 3: the main writer may invoke Wolfram Alpha up to
            # WOLFRAM_MAX_CALLS_PER_SUBMISSION times per submission. When
            # Wolfram is disabled this helper degrades to a single-shot call.
            logger.info(f"Generating LLM completion via api_client_manager (task_id={task_id})...")
            try:
                llm_output, wolfram_calls, _message = await self._generate_completion_with_wolfram_tool(
                    task_id=task_id,
                    initial_prompt=prompt,
                )
            except Exception as exc:
                # Any tool-loop failure falls back to the plain single-shot
                # path so construction still makes forward progress.
                logger.warning(
                    "Wolfram tool-loop failed (%s); falling back to single-shot construction call",
                    exc,
                )
                fallback = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_fallback",
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=system_config.compiler_high_context_max_output_tokens,
                )
                if not fallback.get("choices") or not fallback["choices"][0].get("message"):
                    logger.error("construction: LLM returned empty response structure")
                    raise ValueError("LLM returned empty response")
                fallback_msg = fallback["choices"][0]["message"]
                llm_output = fallback_msg.get("content") or fallback_msg.get("reasoning") or ""
                wolfram_calls = []
            logger.info(
                f"LLM completion received: {len(llm_output)} chars "
                f"({len(wolfram_calls)} Wolfram tool call(s))"
            )
            
            # Check for empty content
            # Parse response with retry
            logger.info("Parsing JSON response...")
            data = await self._parse_json_response_with_retry(llm_output, "construction", prompt)
            
            if not data:
                logger.error("construction: Failed to parse JSON response, returning None")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            logger.info("JSON parsed successfully")
            
            # Handle case where model returns array instead of single object
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Construction returned empty array, returning None")
                    # Notify task completed (failed but still completed)
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
                logger.warning(f"Construction returned array of {len(data)} objects, using first object only")
                data = data[0]
            
            # Check if construction needed
            needs_construction = data.get("needs_construction", True)  # Default True for backward compat
            
            # Extract section_complete flag (new phase-based system)
            section_complete = data.get("section_complete", False)
            
            if not needs_construction:
                logger.info(f"Construction not needed - section_complete={section_complete}")
                # Still return a submission to signal completion if section_complete is True
                if section_complete:
                    submission = CompilerSubmission(
                        submission_id=str(uuid.uuid4()),
                        mode="construction",
                        content="",  # No content, just completion signal
                        operation="full_content",  # No-op for completion signal
                        old_string="",
                        new_string="",
                        reasoning=data.get("reasoning", "Section marked as complete"),
                        section_complete=True,
                        metadata={
                            "coverage": context_pack.coverage,
                            "is_first": is_first_portion,
                            "phase": section_phase,
                            "wolfram_calls": wolfram_calls,
                        },
                    )
                    # Notify task completed successfully
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    logger.info(f"Section completion signal generated: {submission.submission_id} (phase={section_phase})")
                    return submission
                # Notify task completed even when no construction needed
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            # Validate content not empty when needs_construction=True
            # The actual content is in "new_string" field, NOT "content"
            new_string_content = _normalize_string_field(data.get("new_string", ""))
            if not new_string_content or not new_string_content.strip():
                logger.warning(f"Construction marked as needed but new_string is empty. Data keys: {list(data.keys())}")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            # Create submission with section_complete flag
            submission = CompilerSubmission(
                submission_id=str(uuid.uuid4()),
                mode="construction",
                content=new_string_content,  # Use new_string as the content
                operation=data.get("operation", "full_content"),
                old_string=_normalize_string_field(data.get("old_string", "")),
                new_string=new_string_content,  # Already normalized above
                reasoning=data.get("reasoning", ""),
                section_complete=section_complete,
                metadata={
                    "coverage": context_pack.coverage,
                    "is_first": is_first_portion,
                    "phase": section_phase,
                    "wolfram_calls": wolfram_calls,
                },
            )
            
            # Parse optional brainstorm retroactive operation
            brainstorm_op_data = data.get("brainstorm_operation")
            if brainstorm_op_data and isinstance(brainstorm_op_data, dict):
                try:
                    from backend.shared.models import BrainstormRetroactiveOperation
                    submission.brainstorm_operation = BrainstormRetroactiveOperation(
                        action=brainstorm_op_data.get("action", ""),
                        submission_number=brainstorm_op_data.get("submission_number"),
                        new_content=brainstorm_op_data.get("new_content", ""),
                        reasoning=brainstorm_op_data.get("reasoning", "")
                    )
                    logger.info(f"Brainstorm retroactive operation parsed: {submission.brainstorm_operation.action}")
                except Exception as e:
                    logger.warning(f"Failed to parse brainstorm_operation, ignoring: {e}")
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Construction submission generated: {submission.submission_id} (section_complete={section_complete})")
            return submission
            
        except FreeModelExhaustedError:
            raise
        except ValueError:
            raise
        except RuntimeError as e:
            if "credits exhausted" in str(e).lower():
                raise
            logger.error(f"Failed to generate construction submission: {e}", exc_info=True)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
        except Exception as e:
            logger.error(f"Failed to generate construction submission: {e}", exc_info=True)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
    
    async def submit_review(self, review_focus: str = "general") -> Optional[CompilerSubmission]:
        """
        Submit paper review (or no-op if no edit needed).
        
        NO RAG BY DESIGN: Review mode evaluates the paper on its own merits —
        checking for errors, coherence issues, and improvements against the outline.
        No aggregator DB, brainstorm, or reference papers in context. The reviewer
        must judge the paper as a standalone document without external source bias.
        
        Returns:
            CompilerSubmission if edit needed, None otherwise
        """
        logger.info(f"Starting paper review for errors/improvements (focus={review_focus})...")
        
        try:
            # Get current outline and paper (NO aggregator DB context for this mode)
            logger.info("Loading outline and paper state...")
            current_outline = await outline_memory.get_outline()
            current_paper = await paper_memory.get_paper()
            logger.info(f"State loaded: outline={len(current_outline)} chars, paper={len(current_paper)} chars")
            
            # Show the same marker-bearing paper that validation/apply will match.
            paper_for_llm = _strip_paper_markers_for_llm(current_paper)
            logger.info(f"Paper prepared for LLM: {len(current_paper)} chars → {len(paper_for_llm)} chars (markers preserved)")
            
            # Build prompt (no RAG, just direct outline + paper content)
            # CRITICAL: Outline is ALWAYS fully injected per architectural rules
            logger.info("Building review prompt (full outline + paper, no aggregator DB)...")
            prompt = await build_review_prompt(
                user_prompt=self.user_prompt,
                current_outline=current_outline,  # ALWAYS fully injected
                current_paper=paper_for_llm,
                review_focus=review_focus
            )
            logger.info(f"Prompt built: {len(prompt)} chars")
            
            # Validate prompt size
            actual_prompt_tokens = count_tokens(prompt)
            max_allowed_tokens = rag_config.get_available_input_tokens(system_config.compiler_high_context_context_window, system_config.compiler_high_context_max_output_tokens)
            
            if actual_prompt_tokens > max_allowed_tokens:
                logger.error(
                    f"review: Assembled prompt ({actual_prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed_tokens} tokens after safety margin). This indicates a context allocation bug."
                )
                raise ValueError(f"review: Prompt too large ({actual_prompt_tokens} tokens > {max_allowed_tokens} max)")
            
            logger.debug(f"review prompt: {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
            
            # Generate task ID for tracking
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            # Notify task started (for workflow panel)
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Get completion via api_client_manager (handles boost and fallback)
            logger.info(f"Generating LLM completion via api_client_manager (task_id={task_id})...")
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic generation - evolving context provides diversity
                max_tokens=system_config.compiler_high_context_max_output_tokens  # User-configurable (outline creation, update, construction, review)
            )
            
            # Check for empty response
            if not response.get("choices") or not response["choices"][0].get("message"):
                logger.error("review: LLM returned empty response structure")
                raise ValueError("LLM returned empty response")
            
            # Extract content from either 'content' or 'reasoning' field
            # Some reasoning models (e.g., DeepSeek R1, certain GPT variants) output JSON in 'reasoning' field
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            logger.info(f"LLM completion received: {len(llm_output)} chars")
            
            # Parse response with retry
            logger.info("Parsing JSON response...")
            data = await self._parse_json_response_with_retry(llm_output, "review", prompt)
            
            if not data:
                logger.warning("Review: JSON parse failed, treating as no edit needed")
                # Notify task completed (failed but still completed)
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            logger.info("JSON parsed successfully")
            
            # Handle case where model returns array instead of single object
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Review returned empty array, treating as no edit needed")
                    # Notify task completed (failed but still completed)
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
                logger.warning(f"Review returned array of {len(data)} objects, using first object only")
                data = data[0]
            
            # Check if edit needed
            needs_edit = data.get("needs_edit", False)
            
            if not needs_edit:
                # Notify task completed even when no edit needed
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                logger.info("Paper review: no edit needed")
                return None
            
            # Check if this is a minuscule edit
            is_minuscule = "minuscule" in data.get("reasoning", "").lower() or "minor" in data.get("reasoning", "").lower()
            
            # Create submission
            # Use new_string as content for logging
            new_string_content = _normalize_string_field(data.get("new_string", ""))
            
            submission = CompilerSubmission(
                submission_id=str(uuid.uuid4()),
                mode="review",
                content=new_string_content,  # Use new_string as the content
                operation=data.get("operation", "replace"),
                old_string=_normalize_string_field(data.get("old_string", "")),
                new_string=new_string_content,  # Already normalized above
                reasoning=data.get("reasoning", ""),
                metadata={
                    "is_minuscule": is_minuscule,
                    "review_focus": review_focus
                }
            )
            
            # Notify task completed successfully
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Review submission generated: {submission.submission_id} (minuscule={is_minuscule})")
            return submission
            
        except FreeModelExhaustedError:
            raise
        except ValueError:
            raise
        except RuntimeError as e:
            if "credits exhausted" in str(e).lower():
                raise
            logger.error(f"Failed to generate review submission: {e}", exc_info=True)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None
        except Exception as e:
            logger.error(f"Failed to generate review submission: {e}", exc_info=True)
            if self.task_tracking_callback and 'task_id' in dir():
                self.task_tracking_callback("completed", task_id)
            return None  # Don't crash workflow on review failure
    
    async def _generate_completion_with_wolfram_tool(
        self,
        *,
        task_id: str,
        initial_prompt: str,
    ) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Run the construction LLM call with the Wolfram tool attached.

        Returns (final_llm_text, wolfram_calls, raw_message_dict).

        Behavior:
        - If Wolfram is disabled / unavailable, behaves like a single-shot
          `generate_completion` (preserves pre-Phase-3 behavior).
        - Otherwise, registers WOLFRAM_TOOL_SCHEMA on the call and loops: on
          any `tool_calls` in the assistant response, executes each via
          `wolfram_client.query(...)`, appends a tool-role turn with the
          result, and re-prompts the LLM. Up to 20 tool calls per submission.
        - On budget exhaustion, injects a user-role reminder and re-calls
          the LLM with tools disabled so it finalizes with whatever data
          it has gathered.
        - If the model never emits tool_calls (or the backend returns a
          plain completion in one shot), this function behaves identically
          to the single-shot path.

        Websocket events:
        - `compiler_wolfram_call` broadcast per call with query + preview.
        """
        wolfram_enabled = _wolfram_tool_available()

        messages: List[Dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
        wolfram_calls: List[Dict[str, Any]] = []

        # Get the Wolfram client once per submission so we don't repeatedly
        # re-resolve the singleton. Only resolved when tool is enabled.
        wolfram_client = None
        if wolfram_enabled:
            try:
                from backend.shared.wolfram_alpha_client import get_wolfram_client
                wolfram_client = get_wolfram_client()
            except Exception as exc:
                logger.warning(f"Wolfram client init failed; disabling tool for this call: {exc}")
                wolfram_enabled = False

        # Hard cap on total LLM turns in the loop. Each tool round is 1
        # assistant turn + 1 user/tool turn; plus one finalization turn on
        # budget exhaustion. This bound prevents runaway if the model just
        # keeps calling tools.
        max_loop_iterations = WOLFRAM_MAX_CALLS_PER_SUBMISSION + 3

        for iteration in range(max_loop_iterations):
            # Attach tools when the budget is not yet exhausted
            tools_param = (
                [WOLFRAM_TOOL_SCHEMA]
                if wolfram_enabled and len(wolfram_calls) < WOLFRAM_MAX_CALLS_PER_SUBMISSION
                else None
            )

            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=system_config.compiler_high_context_max_output_tokens,
                tools=tools_param,
            )

            if not response.get("choices") or not response["choices"][0].get("message"):
                raise ValueError("LLM returned empty response")
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # Final turn - extract content and return
                content = message.get("content") or message.get("reasoning") or ""
                return content, wolfram_calls, message

            # Append assistant turn verbatim so tool-role replies have the
            # right pairing ids.
            assistant_turn: Dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
            messages.append(assistant_turn)

            # Execute each tool call and append tool-role replies.
            for tool_call in tool_calls:
                fn = tool_call.get("function") or {}
                name = fn.get("name", "")
                arguments_raw = fn.get("arguments") or "{}"
                if name != "wolfram_alpha_query":
                    # Unknown tool - return a structured error so the model
                    # learns not to call it again, but don't hard-fail.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": f"Tool '{name}' is not available; ignore.",
                    })
                    continue

                if not wolfram_enabled or wolfram_client is None:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": "Wolfram Alpha is not enabled; continue without external verification.",
                    })
                    continue

                if len(wolfram_calls) >= WOLFRAM_MAX_CALLS_PER_SUBMISSION:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": (
                            f"Wolfram Alpha call budget exhausted "
                            f"({WOLFRAM_MAX_CALLS_PER_SUBMISSION} calls used). "
                            "Do not call this tool again; finalize your JSON response."
                        ),
                    })
                    continue

                try:
                    args = json.loads(arguments_raw) if isinstance(arguments_raw, str) else dict(arguments_raw)
                except Exception as exc:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": f"Tool call arguments were not valid JSON ({exc}); re-issue the call with valid JSON.",
                    })
                    continue

                query = str(args.get("query", "") or "").strip()
                purpose = str(args.get("purpose", "") or "").strip()
                if not query:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": "Tool call missing 'query'; re-issue with a concrete query.",
                    })
                    continue

                try:
                    result_text = await wolfram_client.query(query)
                except Exception as exc:
                    logger.warning(f"Wolfram query raised: {exc}")
                    result_text = None
                result_text = result_text or "Wolfram Alpha returned no result."
                wolfram_calls.append({
                    "query": query,
                    "purpose": purpose,
                    "result": result_text,
                })
                logger.info(
                    "Wolfram Alpha call %d/%d: %s",
                    len(wolfram_calls),
                    WOLFRAM_MAX_CALLS_PER_SUBMISSION,
                    query[:120],
                )
                try:
                    await self._broadcast_wolfram_event(
                        task_id=task_id,
                        query=query,
                        purpose=purpose,
                        result=result_text,
                        calls_used=len(wolfram_calls),
                    )
                except Exception as exc:
                    logger.debug(f"Wolfram websocket broadcast failed (non-fatal): {exc}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": result_text,
                })

            # After exhausting the budget, inject a one-time reminder and
            # let the next loop iteration run without tools so the model
            # must finalize its JSON response.
            if wolfram_enabled and len(wolfram_calls) >= WOLFRAM_MAX_CALLS_PER_SUBMISSION:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have used all {WOLFRAM_MAX_CALLS_PER_SUBMISSION} "
                        "Wolfram Alpha calls for this submission. Finalize "
                        "your JSON response now using the information you "
                        "have gathered. Do not attempt further tool calls."
                    ),
                })

        # Loop cap reached without a clean finalization - surface whatever
        # text the last assistant turn produced, or empty string.
        for turn in reversed(messages):
            if turn.get("role") == "assistant" and turn.get("content"):
                return str(turn["content"]), wolfram_calls, turn
        return "", wolfram_calls, {}

    async def _broadcast_wolfram_event(
        self,
        *,
        task_id: str,
        query: str,
        purpose: str,
        result: str,
        calls_used: int,
    ) -> None:
        """Broadcast one compiler_wolfram_call websocket event."""
        if not self.websocket_broadcaster:
            return
        try:
            await self.websocket_broadcaster(
                "compiler_wolfram_call",
                {
                    "task_id": task_id,
                    "query": query,
                    "purpose": purpose,
                    "result_preview": (result or "")[:200],
                    "calls_used": calls_used,
                    "calls_remaining": max(0, WOLFRAM_MAX_CALLS_PER_SUBMISSION - calls_used),
                    "max_calls": WOLFRAM_MAX_CALLS_PER_SUBMISSION,
                },
            )
        except Exception as exc:
            logger.debug(f"Wolfram broadcast failed: {exc}")

    async def _parse_json_response_with_retry(
        self, 
        response: str, 
        mode: str,
        original_prompt: str
    ) -> Optional[dict]:
        """
        Parse JSON response with a single conversational retry on failure.
        
        Uses api_client_manager for retry calls to ensure boost/fallback work correctly.
        Only ONE retry is attempted to prevent cascading failures when the coordinator
        also has retry logic at the phase level.
        
        Args:
            response: LLM response
            mode: One of 'outline_create', 'outline_update', 'construction', 'review'
            original_prompt: Original prompt sent to LLM (for retry context)
        
        Returns:
            Parsed JSON dict or None if validation fails after retry
        """
        # First attempt: try to parse JSON directly
        try:
            parsed = parse_json(response)
            return parsed
        except Exception as parse_error:
            error = str(parse_error)
            logger.info(f"Compiler high-context submitter ({mode}): Initial JSON parse failed, attempting single retry")
            logger.debug(f"Parse error: {error}")
        
        # Build mode-specific retry prompt
        retry_prompt = self._build_retry_prompt(mode, error)
        
        # Single conversational retry using api_client_manager (supports boost/fallback)
        try:
            # Generate a retry task ID (append _retry to distinguish from original)
            retry_task_id = f"{self.get_current_task_id()}_retry"
            
            retry_response = await api_client_manager.generate_completion(
                task_id=retry_task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": retry_prompt}
                ],
                temperature=0.0,  # Deterministic JSON formatting
                max_tokens=self.max_output_tokens
            )
            
            if retry_response.get("choices"):
                message = retry_response["choices"][0]["message"]
                retry_output = message.get("content") or message.get("reasoning") or ""
                
                try:
                    parsed = parse_json(retry_output)
                    logger.info(f"Compiler high-context submitter ({mode}): Retry succeeded!")
                    return parsed
                except Exception as retry_parse_error:
                    logger.warning(f"Compiler high-context submitter ({mode}): Retry parse failed - {retry_parse_error}")
            else:
                logger.warning(f"Compiler high-context submitter ({mode}): Retry returned empty response")
                
        except Exception as e:
            logger.error(f"Compiler high-context submitter ({mode}): Retry request failed - {e}")
        
        # Retry failed - return None and let coordinator handle it
        logger.error(f"Compiler high-context submitter ({mode}): JSON validation failed after retry: {error}")
        return None
    
    def _build_retry_prompt(self, mode: str, error: str) -> str:
        """Build mode-specific retry prompt for JSON errors."""
        base_instructions = (
            f"Your previous response could not be parsed as valid JSON.\n\n"
            f"PARSE ERROR: {error}\n\n"
            "JSON ESCAPING RULES FOR LaTeX:\n"
            "LaTeX notation IS ALLOWED - but you must escape it properly in JSON:\n"
            "1. Every backslash in your content needs ONE escape in JSON\n"
            "   - To write \\mathbb{Z} in content, write: \"\\\\mathbb{Z}\" in JSON\n"
            "   - To write \\( and \\), write: \"\\\\(\" and \"\\\\)\" in JSON\n"
            "2. Do NOT double-escape: \\\\\\\\mathbb is WRONG, \\\\mathbb is CORRECT\n"
            "3. For old_string: copy text EXACTLY from the document, just escape backslashes\n"
            "4. Escape quotes inside strings: use \\\" for literal quotes\n"
            "5. Avoid malformed unicode escapes (must be exactly \\uXXXX with 4 hex digits)\n\n"
        )
        
        # Mode-specific schema examples
        schema_examples = {
            "outline_create": (
                '{\n'
                '  "content": "your outline (LaTeX allowed, escape backslashes)",\n'
                '  "outline_complete": true or false,\n'
                '  "reasoning": "explanation"\n'
                '}\n'
            ),
            "outline_update": (
                '{\n'
                '  "needs_update": true,\n'
                '  "operation": "insert_after | replace",\n'
                '  "old_string": "exact text from outline (escape backslashes)",\n'
                '  "new_string": "new sections (LaTeX allowed, escape backslashes)",\n'
                '  "reasoning": "explanation"\n'
                '}\n'
            ),
            "construction": (
                '{\n'
                '  "needs_construction": true or false,\n'
                '  "section_complete": true or false,\n'
                '  "operation": "full_content | replace | insert_after | delete",\n'
                '  "old_string": "exact text from paper (escape backslashes)",\n'
                '  "new_string": "replacement text (LaTeX allowed, escape backslashes)",\n'
                '  "reasoning": "explanation"\n'
                '}\n'
            ),
            "review": (
                '{\n'
                '  "needs_edit": true or false,\n'
                '  "operation": "replace | insert_after | delete",\n'
                '  "old_string": "exact text from paper (escape backslashes)",\n'
                '  "new_string": "replacement text (escape backslashes)",\n'
                '  "reasoning": "explanation"\n'
                '}\n'
            ),
        }
        
        schema = schema_examples.get(mode, "{}")
        
        return (
            f"{base_instructions}"
            f"Please provide your response again in valid JSON format:\n"
            f"{schema}\n"
            "Respond with ONLY the JSON object, no markdown, no explanation."
        )

