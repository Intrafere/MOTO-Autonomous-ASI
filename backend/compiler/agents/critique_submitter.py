"""
Critique Submitter - generates peer review feedback on body section.
"""
from typing import Optional, Callable
import logging
import uuid
from datetime import datetime

from backend.shared.config import rag_config
from backend.shared.models import Submission
from backend.shared.api_client_manager import api_client_manager
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import extract_message_text
from backend.shared.utils import count_tokens
from backend.compiler.prompts.critique_prompts import (
    build_critique_prompt,
)
from backend.compiler.memory.critique_rejection_memory import CritiqueRejectionMemory

logger = logging.getLogger(__name__)


class CritiqueSubmitterAgent:
    """
    Critique submitter agent for peer review aggregation phase.
    Generates critiques of the body section for the final self-review.
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
            llm_output = extract_message_text(message)
            
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
    
    async def handle_acceptance(self) -> None:
        """Handle critique acceptance (for compatibility with aggregator interface)."""
        # No special action needed for critique acceptances
        pass
    
    async def handle_rejection(self, summary: str, content: str) -> None:
        """Handle critique rejection - store feedback for learning."""
        await self.rejection_memory.add_rejection(summary, content)
        logger.info(f"Critique rejected - feedback stored: {summary[:100]}...")

