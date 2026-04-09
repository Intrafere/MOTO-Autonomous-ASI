"""
Critique Submitter - generates peer review feedback on body section.
Also makes rewrite vs continue decision after 5 critiques received.
"""
import asyncio
from typing import Optional, Dict, Callable, List
import logging
import uuid
from datetime import datetime

from backend.shared.config import rag_config, system_config
from backend.shared.models import Submission
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.utils import count_tokens
from backend.compiler.prompts.critique_prompts import (
    build_critique_prompt,
    build_rewrite_decision_prompt,
    build_iterative_edit_prompt
)
from backend.compiler.memory.critique_rejection_memory import CritiqueRejectionMemory

logger = logging.getLogger(__name__)


class CritiqueSubmitterAgent:
    """
    Critique submitter agent for peer review aggregation phase.
    Generates critiques of body section and makes rewrite vs continue decisions.
    """
    
    def __init__(
        self,
        model: str,
        context_window: int,
        max_tokens: int,
        submitter_id: int = 1  # Default to 1 for single-submitter critique mode
    ):
        """
        Initialize critique submitter agent.
        
        Args:
            model: LM Studio model name
            context_window: Context window size in tokens
            max_tokens: Max output tokens
            submitter_id: Submitter ID (default 1 for critique mode)
        """
        self.model = model
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.submitter_id = submitter_id
        
        # State
        self.submission_count = 0
        self.task_sequence = 0  # For task tracking
        self.task_tracking_callback: Optional[Callable] = None
        
        # Role ID for API tracking (matches configuration in compiler_coordinator)
        self.role_id = "compiler_critique_submitter"
        
        # Rejection feedback memory
        self.rejection_memory = CritiqueRejectionMemory()
        
        logger.info(f"Critique submitter initialized with model {model}")
    
    async def initialize(self) -> None:
        """Initialize critique submitter and rejection memory."""
        await self.rejection_memory.initialize()
        logger.info("Critique submitter rejection memory initialized")
    
    def set_task_tracking_callback(self, callback: Callable) -> None:
        """Set callback for task tracking (workflow panel integration)."""
        self.task_tracking_callback = callback
    
    def get_current_task_id(self) -> str:
        """Get the task ID for the current/next API call."""
        return f"critique_sub{self.submitter_id}_{self.task_sequence:03d}"
    
    async def submit_critique(
        self,
        user_prompt: str,
        current_body: str,
        current_outline: str,
        aggregator_db: str,
        reference_papers: Optional[str] = None,
        existing_critiques: Optional[str] = None,
        accumulated_history: Optional[str] = None
    ) -> Optional[Submission]:
        """
        Generate critique of body section.
        
        Args:
            user_prompt: User's compiler-directing prompt
            current_body: Body section to critique
            current_outline: Paper outline
            aggregator_db: Aggregator database content
            reference_papers: Optional reference paper content
            existing_critiques: Optional existing critique feedback
            accumulated_history: Optional accumulated critique history from previous failed versions
            
        Returns:
            Submission object or None if generation failed
        """
        try:
            # Get rejection feedback
            rejection_feedback = await self.rejection_memory.get_all_content()
            
            # Build prompt
            prompt = build_critique_prompt(
                user_prompt=user_prompt,
                current_body=current_body,
                current_outline=current_outline,
                aggregator_db=aggregator_db,
                reference_papers=reference_papers,
                critique_feedback=existing_critiques,
                rejection_feedback=rejection_feedback,
                accumulated_history=accumulated_history
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_allowed = rag_config.get_available_input_tokens(
                self.context_window,
                self.max_tokens
            )
            
            if prompt_tokens > max_allowed:
                logger.error(
                    f"Critique prompt ({prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed} tokens available)"
                )
                return None
            
            logger.debug(f"Critique prompt: {prompt_tokens} tokens (max: {max_allowed})")
            
            # Generate task ID and notify start
            task_id = self.get_current_task_id()
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_tokens
            )
            
            # Notify completion
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            # Extract content from API response
            # Some reasoning models output JSON in 'reasoning' field instead of 'content'
            if not response.get("choices") or not response["choices"][0].get("message"):
                logger.error("Critique: LLM returned empty response structure")
                return None
            
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            
            # Parse JSON response
            data = parse_json(llm_output)
            
            if data is None:
                logger.error("Failed to parse critique JSON response")
                return None
            
            # Handle array responses (extract first element)
            if isinstance(data, list):
                logger.warning("Critique submitter returned array instead of object - using first element")
                if not data:
                    logger.error("Empty array response from critique submitter")
                    return None
                data = data[0]
            
            # Validate required fields
            if "critique_needed" not in data:
                logger.error("Critique response missing 'critique_needed' field")
                return None
            
            if "reasoning" not in data:
                logger.error("Critique response missing 'reasoning' field")
                return None
            
            critique_needed = data.get("critique_needed", True)
            is_decline = not critique_needed
            
            # For critiques, submission field is required
            if critique_needed and "submission" not in data:
                logger.error("Critique response missing 'submission' field when critique_needed=true")
                return None
            
            # Create submission object
            submission = Submission(
                submission_id=str(uuid.uuid4()),
                submitter_id=self.submitter_id,
                content=data.get("submission", ""),  # Empty for declines
                reasoning=data.get("reasoning", ""),
                chunk_size_used=512,  # Fixed for critique mode
                timestamp=datetime.now(),
                is_decline=is_decline
            )
            
            self.submission_count += 1
            if is_decline:
                logger.info(f"Critique submitter declined to critique (assessment #{self.submission_count})")
            else:
                logger.info(f"Critique submitter generated critique #{self.submission_count}")
            
            return submission
            
        except FreeModelExhaustedError:
            raise
        except RuntimeError as e:
            if "credits exhausted" in str(e).lower():
                raise
            logger.error(f"Error generating critique: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error generating critique: {e}", exc_info=True)
            return None
    
    async def submit_rewrite_decision(
        self,
        user_prompt: str,
        current_body: str,
        current_outline: str,
        current_title: str,
        aggregator_db: str,
        critique_feedback: str,
        pre_critique_paper: str,
        reference_papers: Optional[str] = None,
        accumulated_history: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Decide whether to rewrite body or continue to conclusion.
        
        Args:
            user_prompt: User's compiler-directing prompt
            current_body: Body section being evaluated
            current_outline: Paper outline
            current_title: Current paper title
            aggregator_db: Aggregator database content
            critique_feedback: All accepted critiques (typically 1-3 out of 5 total attempts)
            pre_critique_paper: Paper snapshot from START of critique phase (for context)
            reference_papers: Optional reference paper content
            accumulated_history: Optional accumulated critique history from previous failed versions
            
        Returns:
            Dict with decision details or None if generation failed
            Format: {
                "decision": "total_rewrite" | "partial_revision" | "continue",
                "new_title": str or None,
                "new_outline": str or None,
                "reasoning": str
            }
            Note: For partial_revision, edit operations are proposed iteratively (not upfront)
        """
        try:
            # Build prompt
            prompt = build_rewrite_decision_prompt(
                user_prompt=user_prompt,
                current_body=current_body,
                current_outline=current_outline,
                current_title=current_title,
                aggregator_db=aggregator_db,
                critique_feedback=critique_feedback,
                pre_critique_paper=pre_critique_paper,
                reference_papers=reference_papers,
                accumulated_history=accumulated_history
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_allowed = rag_config.get_available_input_tokens(
                self.context_window,
                self.max_tokens
            )
            
            if prompt_tokens > max_allowed:
                logger.error(
                    f"Rewrite decision prompt ({prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed} tokens available)"
                )
                return None
            
            logger.debug(f"Rewrite decision prompt: {prompt_tokens} tokens (max: {max_allowed})")
            
            # Generate task ID and notify start
            task_id = f"critique_decision_{self.task_sequence:03d}"
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM (uses same role as critique generation)
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,  # Use same role config as critique generation
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_tokens
            )
            
            # Notify completion
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            # Extract content from API response
            # Some reasoning models output JSON in 'reasoning' field instead of 'content'
            if not response.get("choices") or not response["choices"][0].get("message"):
                logger.error("Rewrite decision: LLM returned empty response structure")
                return None
            
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            
            # Parse JSON response
            data = parse_json(llm_output)
            
            if data is None:
                logger.error("Failed to parse rewrite decision JSON response")
                return None
            
            # Handle array responses (extract first element)
            if isinstance(data, list):
                logger.warning("Rewrite decision returned array instead of object - using first element")
                if not data:
                    logger.error("Empty array response from rewrite decision")
                    return None
                data = data[0]
            
            # Validate required fields
            required_fields = ["decision", "reasoning"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Rewrite decision response missing '{field}' field")
                    return None
            
            # Validate decision value
            if data["decision"] not in ["total_rewrite", "partial_revision", "continue"]:
                logger.error(f"Invalid decision value: {data['decision']} (must be 'total_rewrite', 'partial_revision', or 'continue')")
                return None
            
            # Note: For partial_revision, edit_operations are now proposed iteratively (not upfront)
            # So we no longer validate edit_operations field here
            
            logger.info(f"Rewrite decision generated: {data['decision']}")
            
            return data
            
        except FreeModelExhaustedError:
            raise
        except RuntimeError as e:
            if "credits exhausted" in str(e).lower():
                raise
            logger.error(f"Error generating rewrite decision: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error generating rewrite decision: {e}", exc_info=True)
            return None
    
    async def submit_iterative_edit(
        self,
        user_prompt: str,
        pre_critique_paper: str,
        current_paper: str,
        current_outline: str,
        critique_feedback: str,
        edits_applied: List[Dict],
        reference_papers: Optional[str] = None,
        accumulated_history: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Propose ONE edit for iterative partial revision.
        
        Called repeatedly until more_edits_needed=false or max iterations reached.
        Each call sees the updated paper after previous edits were applied.
        
        Args:
            user_prompt: User's compiler-directing prompt
            pre_critique_paper: Paper snapshot from START of critique phase
            current_paper: Current paper body (after any edits applied so far)
            current_outline: Paper outline
            critique_feedback: All accepted critiques from this revision cycle
            edits_applied: List of edits already applied in this iteration
            reference_papers: Optional reference paper content
            accumulated_history: Optional accumulated critique history from previous failed versions
            
        Returns:
            Dict with edit details or None if generation failed
            Format: {
                "operation": "replace" | "insert_after" | "delete",
                "old_string": str,
                "new_string": str,
                "reasoning": str,
                "more_edits_needed": bool
            }
        """
        try:
            # Build prompt
            prompt = build_iterative_edit_prompt(
                user_prompt=user_prompt,
                pre_critique_paper=pre_critique_paper,
                current_paper=current_paper,
                current_outline=current_outline,
                critique_feedback=critique_feedback,
                edits_applied=edits_applied,
                reference_papers=reference_papers,
                accumulated_critique_history=accumulated_history or ""
            )
            
            # Validate prompt size
            prompt_tokens = count_tokens(prompt)
            max_allowed = rag_config.get_available_input_tokens(
                self.context_window,
                self.max_tokens
            )
            
            if prompt_tokens > max_allowed:
                logger.error(
                    f"Iterative edit prompt ({prompt_tokens} tokens) exceeds context window "
                    f"({max_allowed} tokens available)"
                )
                return None
            
            logger.debug(f"Iterative edit prompt: {prompt_tokens} tokens (max: {max_allowed})")
            
            # Generate task ID and notify start
            task_id = f"partial_edit_{self.task_sequence:03d}"
            self.task_sequence += 1
            
            if self.task_tracking_callback:
                self.task_tracking_callback("started", task_id)
            
            # Call LLM
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_tokens
            )
            
            # Notify completion
            if self.task_tracking_callback:
                self.task_tracking_callback("completed", task_id)
            
            # Extract content from API response
            if not response.get("choices") or not response["choices"][0].get("message"):
                logger.error("Iterative edit: LLM returned empty response structure")
                return None
            
            message = response["choices"][0]["message"]
            llm_output = message.get("content") or message.get("reasoning") or ""
            
            # Parse JSON response
            data = parse_json(llm_output)
            
            if data is None:
                logger.error("Failed to parse iterative edit JSON response")
                return None
            
            # Handle array responses
            if isinstance(data, list):
                logger.warning("Iterative edit returned array instead of object - using first element")
                if not data:
                    logger.error("Empty array response from iterative edit")
                    return None
                data = data[0]
            
            # Validate required fields
            required_fields = ["operation", "old_string", "new_string", "reasoning", "more_edits_needed"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Iterative edit response missing '{field}' field")
                    return None
            
            # Validate operation type
            if data["operation"] not in ["replace", "insert_after", "delete"]:
                logger.error(f"Invalid operation: {data['operation']} (must be 'replace', 'insert_after', or 'delete')")
                return None
            
            # Validate more_edits_needed is boolean
            if not isinstance(data["more_edits_needed"], bool):
                logger.warning(f"more_edits_needed is not boolean: {data['more_edits_needed']}, converting to bool")
                data["more_edits_needed"] = bool(data["more_edits_needed"])
            
            edit_num = len(edits_applied) + 1
            logger.info(f"Iterative edit #{edit_num} proposed: {data['operation']} (more_edits_needed={data['more_edits_needed']})")
            
            return data
            
        except FreeModelExhaustedError:
            raise
        except RuntimeError as e:
            if "credits exhausted" in str(e).lower():
                raise
            logger.error(f"Error generating iterative edit: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error generating iterative edit: {e}", exc_info=True)
            return None
    
    async def handle_acceptance(self) -> None:
        """Handle critique acceptance (for compatibility with aggregator interface)."""
        # No special action needed for critique acceptances
        pass
    
    async def handle_rejection(self, summary: str, content: str) -> None:
        """Handle critique rejection - store feedback for learning."""
        await self.rejection_memory.add_rejection(summary, content)
        logger.info(f"Critique rejected - feedback stored: {summary[:100]}...")

