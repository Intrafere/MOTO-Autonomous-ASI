"""
High-parameter submitter agent for compiler.
Handles rigor enhancement mode (2-step process).
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.api_client_manager import api_client_manager
from backend.shared.models import CompilerSubmission
from backend.shared.config import system_config, rag_config
from backend.shared.json_parser import parse_json
from backend.aggregator.validation.json_validator import json_validator
from backend.compiler.prompts.rigor_prompts import (
    build_rigor_planning_prompt,
    build_rigor_execution_prompt,
    build_rigor_wolfram_execution_prompt
)
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import paper_memory
from backend.compiler.core.compiler_rag_manager import compiler_rag_manager

logger = logging.getLogger(__name__)


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


class HighParamSubmitter:
    """
    High-parameter, low-context submitter for compiler.
    
    Mode:
    - rigor: Enhance scientific rigor (2-step process)
      Step 1: Planning (unvalidated)
      Step 2: Execution (with self-refusal option)
    """
    
    def __init__(self, model_name: str, user_prompt: str, websocket_broadcaster: Optional[Callable] = None):
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.websocket_broadcaster = websocket_broadcaster
        self._initialized = False
        
        # Task tracking for workflow panel and boost integration
        self.task_sequence: int = 0
        self.role_id = "compiler_high_param"
        self.task_tracking_callback: Optional[Callable] = None
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"comp_hp_{self.task_sequence:03d}"
    
    async def initialize(self) -> None:
        """Initialize submitter."""
        if self._initialized:
            return
        
        # Set context window from system config
        self.context_window = system_config.compiler_high_param_context_window
        self.max_output_tokens = system_config.compiler_high_param_max_output_tokens
        self.available_input_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        
        self._initialized = True
        logger.info(f"High-param submitter initialized with model: {self.model_name}")
        logger.info(f"Context budget: {self.available_input_tokens} tokens (window: {self.context_window})")
    
    
    async def submit_rigor_enhancement(self) -> Optional[CompilerSubmission]:
        """
        Submit rigor enhancement using 2-step process.
        
        Step 1: Planning (unvalidated) - decide if work needed and choose mode
        Step 2: Execution (with self-refusal) - carry out the work
        
        Returns:
            CompilerSubmission if enhancement made, None otherwise
        """
        logger.info("Starting rigor enhancement (Step 1: Planning)...")
        
        try:
            # STEP 1: PLANNING
            planning_result = await self._step1_planning()
            
            if planning_result is None:
                logger.error("Step 1 planning failed (JSON parse error)")
                return None
            
            if not planning_result.get("needs_rigor_work", False):
                logger.info("Step 1: No rigor work needed (declined)")
                return None
            
            mode = planning_result.get("mode")
            target_section = planning_result.get("target_section", "")
            wolfram_query = planning_result.get("wolfram_query", "")
            
            logger.info(f"Step 1 complete: mode={mode}, target_section_len={len(target_section)}")
            
            # STEP 2: EXECUTION (mode-specific)
            if mode == "wolfram_verification":
                return await self._step2_wolfram_execution(
                    target_section,
                    wolfram_query
                )
            else:  # standard_enhancement or rewrite_focus
                return await self._step2_standard_execution(
                    mode,
                    target_section
                )
                
        except Exception as e:
            logger.error(f"Rigor enhancement failed: {e}", exc_info=True)
            raise
    
    async def _step1_planning(self) -> Optional[dict]:
        """
        Execute Step 1: Planning (unvalidated).
        
        LLM decides:
        - Does document need rigor work?
        - Which mode to use?
        - What section to work on?
        
        Returns:
            Planning JSON dict or None if parse fails
        """
        logger.info("Step 1: Loading document state for planning...")
        
        # Get current outline and paper
        current_outline = await outline_memory.get_outline()
        current_paper = await paper_memory.get_paper()
        
        logger.info(f"Step 1: State loaded - outline={len(current_outline)} chars, paper={len(current_paper)} chars")
        
        # Retrieve relevant paper sections via RAG (same as current rigor mode)
        from backend.shared.utils import count_tokens
        max_allowed_tokens = rag_config.get_available_input_tokens(
            system_config.compiler_high_param_context_window,
            system_config.compiler_high_param_max_output_tokens
        )
        
        # Try initial RAG retrieval - may overflow if outline + system prompts are large
        try:
            logger.info("Step 1: Retrieving relevant paper sections via RAG...")
            context_pack = await compiler_rag_manager.retrieve_for_mode(
                query=self.user_prompt + " " + current_paper[-1000:],
                mode="rigor"
            )
            logger.info(f"Step 1: RAG retrieval complete - {len(context_pack.text)} chars")
            
            # Build planning prompt
            logger.info("Step 1: Building planning prompt...")
            prompt = await build_rigor_planning_prompt(
                user_prompt=self.user_prompt,
                current_outline=current_outline,
                current_paper=context_pack.text
            )
            
            # Verify prompt size
            actual_prompt_tokens = count_tokens(prompt)
            
            if actual_prompt_tokens > max_allowed_tokens:
                raise ValueError(f"Prompt too large: {actual_prompt_tokens} tokens > {max_allowed_tokens} max")
            
            logger.debug(f"Step 1: Planning prompt {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
            
        except ValueError as e:
            if "Prompt too large" not in str(e):
                raise
            
            # Context overflow - reduce RAG budget
            logger.warning("Step 1: Initial prompt too large, calculating reduced RAG budget...")
            
            mandatory_tokens = count_tokens(
                await build_rigor_planning_prompt(self.user_prompt, current_outline, "")
            )
            
            remaining_budget = max_allowed_tokens - mandatory_tokens - 200
            
            if remaining_budget < 500:
                raise ValueError(
                    f"Context window too small for rigor mode: outline + system prompts require "
                    f"{mandatory_tokens} tokens, only {max_allowed_tokens} available. "
                    f"Increase compiler_high_param_context_window or reduce outline size."
                )
            
            logger.warning(f"Step 1: Retrying with reduced RAG budget: {remaining_budget} tokens")
            context_pack = await compiler_rag_manager.retrieve_for_mode(
                query=self.user_prompt + " " + current_paper[-1000:],
                mode="rigor",
                max_tokens=remaining_budget
            )
            
            prompt = await build_rigor_planning_prompt(
                user_prompt=self.user_prompt,
                current_outline=current_outline,
                current_paper=context_pack.text
            )
            
            actual_prompt_tokens = count_tokens(prompt)
            logger.info(f"Step 1: Adjusted prompt to {actual_prompt_tokens} tokens")
        
        # Generate task ID
        task_id = self.get_current_task_id()
        self.task_sequence += 1
        
        if self.task_tracking_callback:
            self.task_tracking_callback("started", task_id)
        
        # Call LLM
        logger.info(f"Step 1: Generating LLM completion (task_id={task_id})...")
        response = await api_client_manager.generate_completion(
            task_id=task_id,
            role_id=self.role_id,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=system_config.compiler_high_param_max_output_tokens
        )
        
        # Extract content
        message = response["choices"][0]["message"]
        llm_output = message.get("content", "") or message.get("reasoning", "")
        logger.info(f"Step 1: LLM completion received - {len(llm_output)} chars")
        
        # Parse JSON
        try:
            data = parse_json(llm_output)
            logger.info("Step 1: JSON parsed successfully")
            
            # Handle array responses
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Step 1: Empty array returned, treating as no work needed")
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
                logger.warning(f"Step 1: Array of {len(data)} objects returned, using first")
                data = data[0]
            
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            return data
            
        except Exception as e:
            logger.error(f"Step 1: JSON parse failed - {e}")
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            return None
    
    async def _step2_standard_execution(
        self,
        mode: str,
        target_section: str
    ) -> Optional[CompilerSubmission]:
        """
        Execute Step 2: Standard or rewrite enhancement.
        
        Args:
            mode: "standard_enhancement" or "rewrite_focus"
            target_section: Target section from Step 1 (guidance label)
        
        Returns:
            CompilerSubmission if enhancement made, None otherwise
        """
        logger.info(f"Starting Step 2: {mode} execution...")
        
        try:
            # Get current state (same RAG retrieval as Step 1)
            current_outline = await outline_memory.get_outline()
            current_paper = await paper_memory.get_paper()
            
            # Use same RAG retrieval approach as Step 1
            from backend.shared.utils import count_tokens
            max_allowed_tokens = rag_config.get_available_input_tokens(
                system_config.compiler_high_param_context_window,
                system_config.compiler_high_param_max_output_tokens
            )
            
            # Try RAG retrieval
            try:
                logger.info("Step 2: Retrieving paper sections via RAG...")
                context_pack = await compiler_rag_manager.retrieve_for_mode(
                    query=self.user_prompt + " " + current_paper[-1000:],
                    mode="rigor"
                )
                
                # Build execution prompt
                logger.info("Step 2: Building execution prompt...")
                prompt = await build_rigor_execution_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=context_pack.text,  # FULL paper via RAG
                    target_section=target_section,  # Guidance label
                    mode=mode
                )
                
                # Verify prompt size
                actual_prompt_tokens = count_tokens(prompt)
                
                if actual_prompt_tokens > max_allowed_tokens:
                    raise ValueError(f"Prompt too large: {actual_prompt_tokens} tokens > {max_allowed_tokens} max")
                
                logger.debug(f"Step 2: Execution prompt {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
                
            except ValueError as e:
                if "Prompt too large" not in str(e):
                    raise
                
                # Reduce RAG budget
                logger.warning("Step 2: Prompt too large, reducing RAG budget...")
                
                mandatory_tokens = count_tokens(
                    await build_rigor_execution_prompt(
                        self.user_prompt, current_outline, "", target_section, mode
                    )
                )
                
                remaining_budget = max_allowed_tokens - mandatory_tokens - 200
                
                if remaining_budget < 500:
                    raise ValueError(
                        f"Context window too small for Step 2: {mandatory_tokens} tokens required"
                    )
                
                logger.warning(f"Step 2: Retrying with reduced budget: {remaining_budget} tokens")
                context_pack = await compiler_rag_manager.retrieve_for_mode(
                    query=self.user_prompt + " " + current_paper[-1000:],
                    mode="rigor",
                    max_tokens=remaining_budget
                )
                
                prompt = await build_rigor_execution_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=context_pack.text,
                    target_section=target_section,
                    mode=mode
                )
                
                actual_prompt_tokens = count_tokens(prompt)
                logger.info(f"Step 2: Adjusted prompt to {actual_prompt_tokens} tokens")
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM
            logger.info(f"Step 2: Generating LLM completion (task_id={task_id})...")
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=system_config.compiler_high_param_max_output_tokens
            )
            
            # Extract content
            message = response["choices"][0]["message"]
            llm_output = message.get("content", "") or message.get("reasoning", "")
            logger.info(f"Step 2: LLM completion received - {len(llm_output)} chars")
            
            # Parse JSON
            data = await self._parse_json_response_with_retry(llm_output, prompt, task_id)
            
            if not data:
                logger.error("Step 2: JSON parse failed")
                return None
            
            # Handle array responses
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Step 2: Empty array returned, treating as refusal")
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
                logger.warning(f"Step 2: Array of {len(data)} objects returned, using first")
                data = data[0]
            
            # Check if LLM refused (self-refusal option)
            if not data.get("proceed", True):
                logger.info("Step 2: LLM refused (Step 1 made mistake)")
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            # Check if enhancement needed
            if not data.get("needs_enhancement", False):
                logger.info("Step 2: No enhancement needed")
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            # Create submission
            new_string_content = _normalize_string_field(data.get("new_string", ""))
            
            submission = CompilerSubmission(
                submission_id=str(uuid.uuid4()),
                mode="rigor",
                content=new_string_content,
                operation=data.get("operation", "replace"),
                old_string=_normalize_string_field(data.get("old_string", "")),
                new_string=new_string_content,
                reasoning=data.get("reasoning", ""),
                metadata={"rigor_mode": mode}  # No Wolfram data for standard mode
            )
            
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Step 2: Rigor enhancement submission generated - {submission.submission_id}")
            return submission
            
        except Exception as e:
            logger.error(f"Step 2 execution failed: {e}", exc_info=True)
            raise
    
    async def _step2_wolfram_execution(
        self,
        target_section: str,
        wolfram_query: str
    ) -> Optional[CompilerSubmission]:
        """
        Execute Step 2: Wolfram Alpha verification.
        
        Args:
            target_section: Target section from Step 1 (guidance label)
            wolfram_query: Natural language query for Wolfram Alpha
        
        Returns:
            CompilerSubmission if verification added, None otherwise
        """
        logger.info("Starting Step 2: Wolfram Alpha verification...")
        
        # Check if Wolfram Alpha enabled
        if not system_config.wolfram_alpha_enabled:
            logger.warning("Step 2: Wolfram Alpha requested but not enabled in config")
            return None
        
        # Get Wolfram Alpha client
        from backend.shared.wolfram_alpha_client import get_wolfram_client
        wolfram_client = get_wolfram_client()
        
        if not wolfram_client:
            logger.error("Step 2: Wolfram Alpha client not initialized")
            return None
        
        # Make Wolfram Alpha API query
        logger.info(f"Step 2: Querying Wolfram Alpha - '{wolfram_query}'")
        wolfram_result = await wolfram_client.query(wolfram_query)
        
        if not wolfram_result:
            logger.warning("Step 2: Wolfram Alpha query failed - treating as decline")
            return None
        
        logger.info(f"Step 2: Wolfram Alpha result - {wolfram_result[:200]}")
        
        # Get current state (same RAG retrieval as Step 1)
        try:
            current_outline = await outline_memory.get_outline()
            current_paper = await paper_memory.get_paper()
            
            # Use same RAG retrieval approach
            from backend.shared.utils import count_tokens
            max_allowed_tokens = rag_config.get_available_input_tokens(
                system_config.compiler_high_param_context_window,
                system_config.compiler_high_param_max_output_tokens
            )
            
            # Try RAG retrieval
            try:
                logger.info("Step 2 (Wolfram): Retrieving paper sections via RAG...")
                context_pack = await compiler_rag_manager.retrieve_for_mode(
                    query=self.user_prompt + " " + current_paper[-1000:],
                    mode="rigor"
                )
                
                # Build Wolfram execution prompt
                logger.info("Step 2 (Wolfram): Building execution prompt...")
                prompt = await build_rigor_wolfram_execution_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=context_pack.text,  # FULL paper via RAG
                    target_section=target_section,  # Guidance label
                    wolfram_query=wolfram_query,
                    wolfram_result=wolfram_result
                )
                
                # Verify prompt size
                actual_prompt_tokens = count_tokens(prompt)
                
                if actual_prompt_tokens > max_allowed_tokens:
                    raise ValueError(f"Prompt too large: {actual_prompt_tokens} tokens > {max_allowed_tokens} max")
                
                logger.debug(f"Step 2 (Wolfram): Prompt {actual_prompt_tokens} tokens (max: {max_allowed_tokens})")
                
            except ValueError as e:
                if "Prompt too large" not in str(e):
                    raise
                
                # Reduce RAG budget
                logger.warning("Step 2 (Wolfram): Prompt too large, reducing RAG budget...")
                
                mandatory_tokens = count_tokens(
                    await build_rigor_wolfram_execution_prompt(
                        self.user_prompt, current_outline, "", target_section, 
                        wolfram_query, wolfram_result
                    )
                )
                
                remaining_budget = max_allowed_tokens - mandatory_tokens - 200
                
                if remaining_budget < 500:
                    raise ValueError(
                        f"Context window too small for Step 2 (Wolfram): {mandatory_tokens} tokens required"
                    )
                
                logger.warning(f"Step 2 (Wolfram): Retrying with reduced budget: {remaining_budget} tokens")
                context_pack = await compiler_rag_manager.retrieve_for_mode(
                    query=self.user_prompt + " " + current_paper[-1000:],
                    mode="rigor",
                    max_tokens=remaining_budget
                )
                
                prompt = await build_rigor_wolfram_execution_prompt(
                    user_prompt=self.user_prompt,
                    current_outline=current_outline,
                    current_paper=context_pack.text,
                    target_section=target_section,
                    wolfram_query=wolfram_query,
                    wolfram_result=wolfram_result
                )
                
                actual_prompt_tokens = count_tokens(prompt)
                logger.info(f"Step 2 (Wolfram): Adjusted prompt to {actual_prompt_tokens} tokens")
            
            # Generate task ID
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM
            logger.info(f"Step 2 (Wolfram): Generating LLM completion (task_id={task_id})...")
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=system_config.compiler_high_param_max_output_tokens
            )
            
            # Extract content
            message = response["choices"][0]["message"]
            llm_output = message.get("content", "") or message.get("reasoning", "")
            logger.info(f"Step 2 (Wolfram): LLM completion received - {len(llm_output)} chars")
            
            # Parse JSON
            data = await self._parse_json_response_with_retry(llm_output, prompt, task_id)
            
            if not data:
                logger.error("Step 2 (Wolfram): JSON parse failed")
                return None
            
            # Handle array responses
            if isinstance(data, list):
                if len(data) == 0:
                    logger.warning("Step 2 (Wolfram): Empty array returned, treating as refusal")
                    if self.task_tracking_callback:
                        self.task_tracking_callback("completed", task_id)
                    return None
                logger.warning(f"Step 2 (Wolfram): Array of {len(data)} objects returned, using first")
                data = data[0]
            
            # Check if LLM refused
            if not data.get("proceed", True):
                logger.info("Step 2 (Wolfram): LLM refused (query inappropriate or Step 1 wrong)")
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            # Check if enhancement needed
            if not data.get("needs_enhancement", False):
                logger.info("Step 2 (Wolfram): No enhancement needed")
                if self.task_tracking_callback:
                    self.task_tracking_callback("completed", task_id)
                return None
            
            # Create submission
            new_string_content = _normalize_string_field(data.get("new_string", ""))
            
            submission = CompilerSubmission(
                submission_id=str(uuid.uuid4()),
                mode="rigor",
                content=new_string_content,
                operation=data.get("operation", "insert_after"),
                old_string=_normalize_string_field(data.get("old_string", "")),
                new_string=new_string_content,
                reasoning=data.get("reasoning", ""),
                metadata={
                    "rigor_mode": "wolfram_verification",
                    "wolfram_query": wolfram_query,
                    "wolfram_result": wolfram_result
                }
            )
            
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            logger.info(f"Step 2 (Wolfram): Verification submission generated - {submission.submission_id}")
            return submission
            
        except Exception as e:
            logger.error(f"Step 2 (Wolfram) execution failed: {e}", exc_info=True)
            raise
    
    async def _parse_json_response_with_retry(
        self,
        response: str,
        original_prompt: str,
        task_id: str
    ) -> Optional[dict]:
        """
        Parse JSON response with conversational retry on failure.
        
        Args:
            response: LLM response
            original_prompt: Original prompt sent to LLM (for retry context)
            task_id: Task ID for tracking retry attempt
        
        Returns:
            Parsed JSON dict or None if validation fails after retries
        """
        # Cache model config on first use (only relevant for LM Studio)
        try:
            await lm_studio_client.cache_model_load_config(self.model_name, {
                "context_length": self.context_window,
                "model_path": self.model_name
            })
        except Exception:
            # Silently ignore - only applies to LM Studio models
            pass
        
        # Parse JSON
        try:
            parsed = parse_json(response)
            return parsed
            
        except Exception as parse_error:
            # Not corrupted, just invalid JSON - continue with conversational retry
            valid = False
            parsed = None
            error = str(parse_error)
            
            # Initial parse failed - attempt conversational retry
            logger.info("Compiler high-param submitter (rigor): Initial JSON parse failed, attempting retry")
            logger.debug(f"Parse error: {error}")
        
        # Build retry prompt
        retry_prompt = (
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
            "Please provide your response again in valid JSON format.\n\n"
            "Respond with ONLY the JSON object, no markdown, no explanation."
        )
        
        try:
            # CRITICAL FIX: Truncate failed output to prevent context overflow during retry
            from backend.shared.utils import count_tokens
            
            max_failed_output_chars = 2000  # ~500 tokens - enough for error context
            if len(response) > max_failed_output_chars:
                failed_output_preview = response[:max_failed_output_chars] + "\n[...output truncated for retry...]"
            else:
                failed_output_preview = response
            
            # Calculate if conversation fits in context window
            prompt_tokens = count_tokens(original_prompt)
            preview_tokens = count_tokens(failed_output_preview)
            retry_prompt_tokens = count_tokens(retry_prompt)
            conversation_tokens = prompt_tokens + preview_tokens + retry_prompt_tokens
            
            if conversation_tokens > self.available_input_tokens:
                # Too large - just retry with original prompt
                logger.warning(
                    f"Compiler high-param submitter (rigor): Retry conversation too large "
                    f"({conversation_tokens} > {self.available_input_tokens}), using simple retry"
                )
                retry_response = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_retry",
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[{"role": "user", "content": original_prompt}],
                    temperature=0.0,
                    max_tokens=self.max_output_tokens
                )
            else:
                # Build conversation with truncated failed output
                retry_response = await api_client_manager.generate_completion(
                    task_id=f"{task_id}_retry",
                    role_id=self.role_id,
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": original_prompt},
                        {"role": "assistant", "content": failed_output_preview},
                        {"role": "user", "content": retry_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=self.max_output_tokens
                )
            
            if retry_response.get("choices"):
                retry_output = retry_response["choices"][0]["message"]["content"]
                
                try:
                    parsed = parse_json(retry_output)
                    logger.info("Compiler high-param submitter (rigor): Retry succeeded!")
                    return parsed
                except Exception as parse_error:
                    error = str(parse_error)
                    logger.warning(f"Compiler high-param submitter (rigor): Retry failed - {error}")
        except Exception as e:
            logger.error(f"Compiler high-param submitter (rigor): Retry request failed - {e}")
        
        # All retries failed
        logger.error(f"Compiler high-param submitter (rigor): JSON validation failed after retry: {error}")
        return None
