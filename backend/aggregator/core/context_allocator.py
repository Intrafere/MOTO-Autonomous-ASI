"""
Context allocation and routing between direct injection and RAG.
Implements priority-based offloading to RAG when content exceeds context window.
"""
from typing import Dict, List, Optional
import logging
from pathlib import Path

from backend.shared.config import rag_config
from backend.shared.models import ContextPack
from backend.shared.utils import count_tokens
from backend.aggregator.core.rag_manager import rag_manager

logger = logging.getLogger(__name__)


class ContextAllocationError(Exception):
    """Raised when context allocation fails."""
    pass


class ContextAllocator:
    """
    Manages context allocation between direct injection and RAG.
    
    Priority order for RAG offloading:
    - Submitter: shared training → local training → rejection logs → user files
    - Validator: shared training → user files → submission under review
    """
    
    def __init__(self):
        # Default context windows (can be overridden via set_context_windows)
        self.submitter_context_window = rag_config.submitter_context_window
        self.validator_context_window = rag_config.validator_context_window
        # Default max output tokens
        self.submitter_max_output_tokens = rag_config.submitter_max_output_tokens
        self.validator_max_output_tokens = rag_config.validator_max_output_tokens
    
    def set_context_windows(self, submitter_context: int, validator_context: int, 
                            submitter_max_output: int = None, validator_max_output: int = None):
        """Set separate context windows and max output tokens for submitter and validator."""
        self.submitter_context_window = submitter_context
        self.validator_context_window = validator_context
        if submitter_max_output is not None:
            self.submitter_max_output_tokens = submitter_max_output
        if validator_max_output is not None:
            self.validator_max_output_tokens = validator_max_output
        logger.info(f"Context windows updated - Submitter: {submitter_context}, Validator: {validator_context}")

    def _get_shared_training_rag_sources(self) -> List[str]:
        """
        Get RAG source names that map to shared-training content.

        Used to exclude shared-training chunks from RAG when the same
        shared-training content is already direct-injected into the prompt.
        """
        sources: List[str] = []

        # Current shared-training file source (manual mode: rag_shared_training.txt,
        # autonomous mode: brainstorm_<topic_id>.txt)
        try:
            from backend.aggregator.memory.shared_training import shared_training_memory
            current_source = Path(shared_training_memory.file_path).name
            if current_source:
                sources.append(current_source)
        except Exception as e:
            logger.debug(f"Could not resolve shared-training source name for exclusion: {e}")

        # Incremental re-RAG sources used by aggregator background updates
        for chunk_size in rag_config.submitter_chunk_intervals:
            sources.append(f"rag_shared_training_update_{chunk_size}")

        # De-dup while preserving insertion order
        return list(dict.fromkeys(sources))
    
    async def allocate_submitter_context(
        self,
        user_prompt: str,
        json_schema: str,
        system_prompt: str,
        shared_training_content: str,
        local_training_content: str,
        rejection_log_content: str,
        user_files_content: Dict[str, str],
        chunk_size: int,
        context_window: int = None,
        max_output_tokens: int = None
    ) -> Dict[str, any]:
        """
        Allocate context for a submitter.
        
        Priority:
        1. User prompt (ALWAYS direct inject - added by prompt builder)
        2. JSON schema (ALWAYS direct inject - added by prompt builder)
        3. System prompt (ALWAYS direct inject - added by prompt builder)
        4. Shared training → RAG if needed
        5. Local training → RAG if needed
        6. Rejection logs → RAG if needed
        7. User files → RAG only if absolutely necessary
        
        Args:
            context_window: Per-submitter context window override (uses global if None)
            max_output_tokens: Per-submitter max output override (uses global if None)
        
        Returns:
            Dict with 'direct' (str) and 'rag_context' (ContextPack) keys
        """
        # Use per-submitter settings if provided, otherwise fall back to global
        ctx_window = context_window if context_window is not None else self.submitter_context_window
        max_output = max_output_tokens if max_output_tokens is not None else self.submitter_max_output_tokens
        
        # Calculate context limits for this specific submitter
        available_tokens = rag_config.get_available_input_tokens(ctx_window, max_output)
        minimum_rag_allocation = rag_config.get_minimum_rag_allocation(ctx_window, max_output)
        
        # Calculate tokens for prompt template parts (added by prompt builder)
        # These are NOT included in the 'direct' context we return, but we need to account for them
        user_prompt_tokens = count_tokens(user_prompt)
        json_schema_tokens = count_tokens(json_schema)
        system_prompt_tokens = count_tokens(system_prompt)
        
        # Account for prompt assembly overhead (separators, headers, final instruction, etc.)
        # Format: system + "\n---\n" + schema + "\n---\n" + "USER PROMPT:\n" + user_prompt + "\n---\n" + context + "\n---\n" + optional_rag + "\n---\n" + final_instruction
        assembly_overhead = count_tokens("\n---\n" * 5 + "USER PROMPT:\n" + "RETRIEVED EVIDENCE:\n" + "Now generate your submission as JSON:")
        
        mandatory_tokens = user_prompt_tokens + json_schema_tokens + system_prompt_tokens + assembly_overhead
        
        # Check if user prompt alone exceeds limits
        if user_prompt_tokens > (available_tokens - minimum_rag_allocation):
            raise ContextAllocationError(
                f"User prompt ({user_prompt_tokens} tokens) exceeds maximum allowed "
                f"({available_tokens - minimum_rag_allocation} tokens). "
                f"Please shorten your prompt."
            )
        
        remaining_tokens = available_tokens - mandatory_tokens
        
        logger.debug(
            f"Submitter context allocation: window={ctx_window}, "
            f"available={available_tokens}, mandatory={mandatory_tokens} "
            f"(user={user_prompt_tokens}, json={json_schema_tokens}, system={system_prompt_tokens}, "
            f"overhead={assembly_overhead}), remaining={remaining_tokens}"
        )
        
        # Build direct injection parts (excluding system_prompt, user_prompt, json_schema - those are added by prompt builder)
        direct_parts = []
        
        # Track what needs RAG retrieval based on offload priority
        needs_shared_training_rag = False
        needs_local_training_rag = False
        needs_rejection_log_rag = False
        needs_user_files_rag = False
        
        # Priority 1: Shared training - try direct injection first
        # BUT: Reserve minimum space for RAG (at least 5000 tokens) if content needs to be offloaded
        minimum_rag_reserve = 5000  # Ensure meaningful RAG retrieval space
        if shared_training_content:
            formatted = f"[SHARED TRAINING]\n{shared_training_content}"
            tokens = count_tokens(formatted)
            # Direct inject only if it fits AND leaves enough space for other content + RAG
            if tokens <= remaining_tokens and (tokens < remaining_tokens - minimum_rag_reserve):
                direct_parts.append(formatted)
                remaining_tokens -= tokens
                logger.debug(f"Submitter: Shared training direct injected ({tokens} tokens)")
            else:
                needs_shared_training_rag = True
                if tokens > remaining_tokens:
                    logger.info(f"Submitter: Shared training offloaded to RAG ({tokens} tokens > {remaining_tokens} available)")
                else:
                    logger.info(f"Submitter: Shared training offloaded to RAG ({tokens} tokens would leave insufficient RAG space)")
        
        # Priority 2: Local training - try direct injection first
        if local_training_content:
            formatted = f"[LOCAL TRAINING]\n{local_training_content}"
            tokens = count_tokens(formatted)
            if tokens <= remaining_tokens:
                direct_parts.append(formatted)
                remaining_tokens -= tokens
                logger.debug(f"Submitter: Local training direct injected ({tokens} tokens)")
            else:
                needs_local_training_rag = True
                logger.info(f"Submitter: Local training offloaded to RAG ({tokens} tokens > {remaining_tokens} available)")
        
        # Priority 3: Rejection logs - try direct injection first
        if rejection_log_content:
            formatted = f"[REJECTION LOG]\n{rejection_log_content}"
            tokens = count_tokens(formatted)
            if tokens <= remaining_tokens:
                direct_parts.append(formatted)
                remaining_tokens -= tokens
                logger.debug(f"Submitter: Rejection log direct injected ({tokens} tokens)")
            else:
                needs_rejection_log_rag = True
                logger.info(f"Submitter: Rejection log offloaded to RAG ({tokens} tokens > {remaining_tokens} available)")
        
        # Priority 4: User files - try direct injection first (LAST priority to offload)
        user_files_str = ""
        for filename, file_content in user_files_content.items():
            user_files_str += f"[FILE: {filename}]\n{file_content}\n"
        
        if user_files_str:
            tokens = count_tokens(user_files_str)
            if tokens <= remaining_tokens:
                direct_parts.append(user_files_str)
                remaining_tokens -= tokens
                logger.debug(f"Submitter: User files direct injected ({tokens} tokens)")
            else:
                needs_user_files_rag = True
                logger.info(f"Submitter: User files offloaded to RAG ({tokens} tokens > {remaining_tokens} available)")
        
        # Perform RAG retrieval ONLY if content was offloaded
        rag_context = None
        if any([needs_shared_training_rag, needs_local_training_rag, needs_rejection_log_rag, needs_user_files_rag]):
            # Build exclusion list: sources that were direct-injected should not appear in RAG
            exclude_sources = []
            if not needs_shared_training_rag and shared_training_content:
                exclude_sources.extend(self._get_shared_training_rag_sources())
            if not needs_user_files_rag and user_files_content:
                exclude_sources.extend(user_files_content.keys())
            if exclude_sources:
                exclude_sources = list(dict.fromkeys(exclude_sources))
            
            # FIXED: Calculate RAG budget from REMAINING space after direct injection
            # This ensures we maximize context usage without exceeding limits
            direct_content_temp = "\n\n".join(direct_parts)
            direct_content_tokens = count_tokens(direct_content_temp)
            
            # Total tokens already allocated
            already_allocated = mandatory_tokens + direct_content_tokens
            
            # Available space for RAG (with buffer for RAG formatting overhead)
            # RAG content will be wrapped with "\n---\nRETRIEVED EVIDENCE:\n{rag_text}" which adds tokens
            rag_formatting_overhead = count_tokens("\n---\nRETRIEVED EVIDENCE:\n")
            safety_buffer = 500  # Increased buffer for final prompt assembly + RAG wrapping
            max_rag_space = available_tokens - already_allocated - safety_buffer - rag_formatting_overhead
            
            # Use as much as possible for RAG while respecting the limit
            rag_max_tokens = max(0, max_rag_space)  # Ensure non-negative
            
            logger.info(
                f"Submitter: Performing RAG retrieval (max {rag_max_tokens} tokens) for offloaded content. "
                f"Breakdown: available={available_tokens}, mandatory={mandatory_tokens}, "
                f"direct_content={direct_content_tokens}, rag_overhead={rag_formatting_overhead}, safety_buffer={safety_buffer}"
            )
            
            if rag_max_tokens < 1000:
                logger.warning(
                    f"Submitter: Very limited RAG space ({rag_max_tokens} tokens). "
                    f"Consider reducing direct-injected content or increasing context window."
                )
            
            rag_context = await rag_manager.retrieve(
                query=user_prompt,
                chunk_size=chunk_size,  # Cycles: 256→512→768→1024
                max_tokens=rag_max_tokens,
                exclude_sources=exclude_sources if exclude_sources else None
            )
            
            if rag_context and rag_context.text:
                logger.info(
                    f"Submitter: RAG retrieved {len(rag_context.evidence)} evidence chunks, "
                    f"coverage={rag_context.coverage:.2f}, answerability={rag_context.answerability:.2f}"
                )
            else:
                logger.warning("Submitter: RAG retrieval returned empty despite offloaded content")
        else:
            logger.debug("Submitter: All content fits in direct injection - no RAG needed")
        
        direct_content = "\n\n".join(direct_parts)
        
        return {
            "direct": direct_content,
            "rag_context": rag_context
        }
    
    async def allocate_validator_context(
        self,
        user_prompt: str,
        json_schema: str,
        system_prompt: str,
        shared_training_content: str,
        user_files_content: Dict[str, str],
        submission_content: str,
        chunk_size: int = 512
    ) -> Dict[str, any]:
        """
        Allocate context for the validator.
        
        Priority:
        1. User prompt (ALWAYS direct inject - added by prompt builder)
        2. JSON schema (ALWAYS direct inject - added by prompt builder)
        3. System prompt (ALWAYS direct inject - added by prompt builder)
        4. Submission under review (direct inject if fits)
        5. Shared training → RAG if needed
        6. User files → RAG if needed
        
        Returns:
            Dict with 'direct' (str) and 'rag_context' (ContextPack) keys
        """
        # Calculate context limits for validator
        available_tokens = rag_config.get_available_input_tokens(self.validator_context_window, self.validator_max_output_tokens)
        minimum_rag_allocation = rag_config.get_minimum_rag_allocation(self.validator_context_window, self.validator_max_output_tokens)
        
        # Calculate tokens for prompt template parts (added by prompt builder)
        # These are NOT included in the 'direct' context we return, but we need to account for them
        user_prompt_tokens = count_tokens(user_prompt)
        json_schema_tokens = count_tokens(json_schema)
        system_prompt_tokens = count_tokens(system_prompt)
        submission_tokens = count_tokens(submission_content)
        
        # Account for prompt assembly overhead (separators, headers, final instruction, etc.)
        # Format: system + "\n---\n" + schema + "\n---\n" + "USER PROMPT:\n" + user_prompt + "\n---\n" + "SUBMISSION TO VALIDATE:\n" + submission + "\n---\n" + context + optional_rag + "\n---\n" + final_instruction
        assembly_overhead = count_tokens("\n---\n" * 6 + "USER PROMPT:\n" + "SUBMISSION TO VALIDATE:\n" + "EXISTING KNOWLEDGE BASE (Retrieved):\n" + "Evaluate this submission and provide your decision as JSON:")
        
        mandatory_tokens = user_prompt_tokens + json_schema_tokens + system_prompt_tokens + submission_tokens + assembly_overhead
        
        # Check if user prompt alone exceeds limits
        if user_prompt_tokens > (available_tokens - minimum_rag_allocation):
            raise ContextAllocationError(
                f"User prompt ({user_prompt_tokens} tokens) exceeds maximum allowed. "
                f"Please shorten your prompt."
            )
        
        remaining_tokens = available_tokens - mandatory_tokens
        
        logger.debug(
            f"Validator context allocation: window={self.validator_context_window}, "
            f"available={available_tokens}, mandatory={mandatory_tokens} "
            f"(user={user_prompt_tokens}, json={json_schema_tokens}, system={system_prompt_tokens}, "
            f"submission={submission_tokens}, overhead={assembly_overhead}), remaining={remaining_tokens}"
        )
        
        # Build direct injection parts (excluding system_prompt, user_prompt, json_schema, submission - those are added by prompt builder)
        direct_parts = []
        
        # NOTE: submission_content is added by build_validator_prompt(), not here
        # We already counted it in mandatory_tokens above
        
        # Track what needs RAG retrieval based on offload priority
        needs_shared_training_rag = False
        needs_user_files_rag = False
        
        # Priority 1: Shared training - try direct injection first
        # BUT: Reserve minimum space for RAG (at least 5000 tokens) if content needs to be offloaded
        minimum_rag_reserve = 5000  # Ensure meaningful RAG retrieval space
        if shared_training_content:
            formatted_training = f"[SHARED TRAINING]\n{shared_training_content}"
            training_tokens = count_tokens(formatted_training)
            # Direct inject only if it fits AND leaves enough space for other content + RAG
            if training_tokens <= remaining_tokens and (training_tokens < remaining_tokens - minimum_rag_reserve):
                # Fits - use direct injection
                direct_parts.append(formatted_training)
                remaining_tokens -= training_tokens
                logger.debug(f"Validator: Shared training direct injected ({training_tokens} tokens)")
            else:
                # Doesn't fit - offload to RAG
                needs_shared_training_rag = True
                if training_tokens > remaining_tokens:
                    logger.info(f"Validator: Shared training offloaded to RAG ({training_tokens} tokens > {remaining_tokens} available)")
                else:
                    logger.info(f"Validator: Shared training offloaded to RAG ({training_tokens} tokens would leave insufficient RAG space)")
        
        # Priority 2: User files - try direct injection first
        user_files_str = ""
        for filename, file_content in user_files_content.items():
            user_files_str += f"[FILE: {filename}]\n{file_content}\n"
        
        if user_files_str:
            user_files_tokens = count_tokens(user_files_str)
            if user_files_tokens <= remaining_tokens:
                # Fits - use direct injection
                direct_parts.append(user_files_str)
                remaining_tokens -= user_files_tokens
                logger.debug(f"Validator: User files direct injected ({user_files_tokens} tokens)")
            else:
                # Doesn't fit - offload to RAG
                needs_user_files_rag = True
                logger.info(f"Validator: User files offloaded to RAG ({user_files_tokens} tokens > {remaining_tokens} available)")
        
        # Perform RAG retrieval ONLY if content was offloaded
        rag_context = None
        if needs_shared_training_rag or needs_user_files_rag:
            # Build exclusion list: sources that were direct-injected should not appear in RAG
            exclude_sources = []
            if not needs_shared_training_rag and shared_training_content:
                exclude_sources.extend(self._get_shared_training_rag_sources())
            if not needs_user_files_rag and user_files_content:
                exclude_sources.extend(user_files_content.keys())
            if exclude_sources:
                exclude_sources = list(dict.fromkeys(exclude_sources))
            
            # FIXED: Calculate RAG budget from REMAINING space after direct injection
            # This ensures we maximize context usage without exceeding limits
            direct_content_temp = "\n\n".join(direct_parts)
            direct_content_tokens = count_tokens(direct_content_temp)
            
            # Total tokens already allocated
            already_allocated = mandatory_tokens + direct_content_tokens
            
            # Available space for RAG (with buffer for RAG formatting overhead)
            # RAG content will be wrapped with "\n---\nEXISTING KNOWLEDGE BASE (Retrieved):\n{rag_text}" which adds tokens
            rag_formatting_overhead = count_tokens("\n---\nEXISTING KNOWLEDGE BASE (Retrieved):\n")
            safety_buffer = 500  # Increased buffer for final prompt assembly + RAG wrapping
            max_rag_space = available_tokens - already_allocated - safety_buffer - rag_formatting_overhead
            
            # Use as much as possible for RAG while respecting the limit
            rag_max_tokens = max(0, max_rag_space)  # Ensure non-negative
            
            logger.info(
                f"Validator: Performing RAG retrieval (max {rag_max_tokens} tokens) for offloaded content. "
                f"Breakdown: available={available_tokens}, mandatory={mandatory_tokens}, "
                f"direct_content={direct_content_tokens}, rag_overhead={rag_formatting_overhead}, safety_buffer={safety_buffer}"
            )
            
            if rag_max_tokens < 1000:
                logger.warning(
                    f"Validator: Very limited RAG space ({rag_max_tokens} tokens). "
                    f"Consider reducing direct-injected content or increasing context window."
                )
            
            rag_context = await rag_manager.retrieve(
                query=user_prompt,
                chunk_size=chunk_size,  # Always 512 for validator
                max_tokens=rag_max_tokens,
                exclude_sources=exclude_sources if exclude_sources else None
            )
            
            if rag_context and rag_context.text:
                logger.info(
                    f"Validator: RAG retrieved {len(rag_context.evidence)} evidence chunks, "
                    f"coverage={rag_context.coverage:.2f}, answerability={rag_context.answerability:.2f}"
                )
            else:
                logger.warning("Validator: RAG retrieval returned empty despite offloaded content")
        else:
            logger.debug("Validator: All content fits in direct injection - no RAG needed")
        
        direct_content = "\n\n".join(direct_parts)
        
        return {
            "direct": direct_content,
            "rag_context": rag_context
        }
    
    async def allocate_cleanup_review_context(
        self,
        user_prompt: str,
        json_schema: str,
        system_prompt: str,
        all_submissions_formatted: str,
        user_files_content: Dict[str, str],
        submission_proposed_for_removal: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Allocate context for cleanup review or removal validation.
        
        Unlike regular validation, cleanup review MUST work even with very large databases.
        If all submissions don't fit, use RAG to retrieve relevant content.
        
        CRITICAL: This method NEVER skips or fails due to size. It uses RAG when needed.
        
        Args:
            user_prompt: User's original prompt (for context on what the database is solving)
            json_schema: JSON schema for output format
            system_prompt: Cleanup review system prompt
            all_submissions_formatted: All accepted submissions with numbers
            user_files_content: User-provided files
            submission_proposed_for_removal: For removal validation, the specific submission
            
        Returns:
            Dict with 'direct' (str), 'rag_context' (ContextPack), and 'submissions_ragged' (bool) keys
        """
        # Calculate context limits for validator (cleanup uses validator context window)
        available_tokens = rag_config.get_available_input_tokens(self.validator_context_window, self.validator_max_output_tokens)
        
        # Calculate tokens for prompt template parts (added by prompt builder)
        user_prompt_tokens = count_tokens(user_prompt)
        json_schema_tokens = count_tokens(json_schema)
        system_prompt_tokens = count_tokens(system_prompt)
        
        # Account for prompt assembly overhead
        assembly_overhead = count_tokens(
            "\n---\n" * 6 + 
            "USER PROMPT (the goal this database is solving):\n" + 
            "CURRENT ACCEPTED SUBMISSIONS DATABASE:\n" +
            "USER PROVIDED FILES:\n" +
            "ADDITIONAL CONTEXT (Retrieved):\n" +
            "Review the database and provide your cleanup decision as JSON:"
        )
        
        # If there's a submission proposed for removal (removal validation phase), include it
        submission_removal_tokens = 0
        if submission_proposed_for_removal:
            submission_removal_tokens = count_tokens(submission_proposed_for_removal)
            assembly_overhead += count_tokens("SUBMISSION PROPOSED FOR REMOVAL:\n")
        
        mandatory_tokens = user_prompt_tokens + json_schema_tokens + system_prompt_tokens + assembly_overhead + submission_removal_tokens
        
        remaining_tokens = available_tokens - mandatory_tokens
        
        logger.debug(
            f"Cleanup context allocation: window={self.validator_context_window}, "
            f"available={available_tokens}, mandatory={mandatory_tokens}, remaining={remaining_tokens}"
        )
        
        # Build direct injection parts
        direct_parts = []
        submissions_ragged = False  # Track if we had to use RAG for submissions
        
        # Track what needs RAG retrieval
        needs_submissions_rag = False
        needs_user_files_rag = False
        
        # Reserve space for RAG if needed (at least 5000 tokens)
        minimum_rag_reserve = 5000
        
        # Priority 1: All submissions - try direct injection first
        if all_submissions_formatted:
            submissions_tokens = count_tokens(all_submissions_formatted)
            
            # Direct inject if it fits AND leaves space for other content
            if submissions_tokens <= remaining_tokens and (submissions_tokens < remaining_tokens - minimum_rag_reserve):
                direct_parts.append(f"[ALL SUBMISSIONS]\n{all_submissions_formatted}")
                remaining_tokens -= submissions_tokens
                logger.info(f"Cleanup: All submissions direct injected ({submissions_tokens} tokens)")
            else:
                # CRITICAL: Use RAG instead of skipping!
                needs_submissions_rag = True
                submissions_ragged = True
                logger.info(
                    f"Cleanup: All submissions ({submissions_tokens} tokens) exceed available space ({remaining_tokens} tokens). "
                    f"Using RAG retrieval instead of skipping."
                )
        
        # Priority 2: User files - try direct injection
        user_files_str = ""
        for filename, file_content in user_files_content.items():
            user_files_str += f"[FILE: {filename}]\n{file_content}\n"
        
        if user_files_str:
            user_files_tokens = count_tokens(user_files_str)
            if user_files_tokens <= remaining_tokens:
                direct_parts.append(user_files_str)
                remaining_tokens -= user_files_tokens
                logger.debug(f"Cleanup: User files direct injected ({user_files_tokens} tokens)")
            else:
                needs_user_files_rag = True
                logger.info(f"Cleanup: User files offloaded to RAG ({user_files_tokens} tokens > {remaining_tokens} available)")
        
        # Perform RAG retrieval if content was offloaded
        rag_context = None
        if needs_submissions_rag or needs_user_files_rag:
            # Build exclusion list: sources that were direct-injected should not appear in RAG
            exclude_sources = []
            if not needs_submissions_rag and all_submissions_formatted:
                exclude_sources.extend(self._get_shared_training_rag_sources())
            if not needs_user_files_rag and user_files_content:
                exclude_sources.extend(user_files_content.keys())
            if exclude_sources:
                exclude_sources = list(dict.fromkeys(exclude_sources))
            
            # Calculate RAG budget from remaining space
            direct_content_temp = "\n\n".join(direct_parts)
            direct_content_tokens = count_tokens(direct_content_temp)
            already_allocated = mandatory_tokens + direct_content_tokens
            
            # Available space for RAG
            rag_formatting_overhead = count_tokens("\n---\nADDITIONAL CONTEXT (Retrieved):\n")
            safety_buffer = 500
            max_rag_space = available_tokens - already_allocated - safety_buffer - rag_formatting_overhead
            rag_max_tokens = max(0, max_rag_space)
            
            logger.info(
                f"Cleanup: Performing RAG retrieval (max {rag_max_tokens} tokens) for offloaded content. "
                f"Breakdown: available={available_tokens}, mandatory={mandatory_tokens}, "
                f"direct_content={direct_content_tokens}"
            )
            
            rag_context = await rag_manager.retrieve(
                query=user_prompt,
                chunk_size=512,  # Use validator's standard chunk size
                max_tokens=rag_max_tokens,
                exclude_sources=exclude_sources if exclude_sources else None
            )
            
            if rag_context and rag_context.text:
                logger.info(
                    f"Cleanup: RAG retrieved {len(rag_context.evidence)} evidence chunks, "
                    f"coverage={rag_context.coverage:.2f}"
                )
            else:
                logger.warning("Cleanup: RAG retrieval returned empty - cleanup review may have limited context")
        else:
            logger.info("Cleanup: All content fits in direct injection - no RAG needed")
        
        direct_content = "\n\n".join(direct_parts)
        
        return {
            "direct": direct_content,
            "rag_context": rag_context,
            "submissions_ragged": submissions_ragged,
            "user_files_ragged": needs_user_files_rag
        }


# Global context allocator instance
context_allocator = ContextAllocator()

