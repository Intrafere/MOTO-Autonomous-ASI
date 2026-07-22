"""
Outline memory manager for compiler.
Handles outline file I/O, re-chunking triggers, and version tracking.
"""
import aiofiles
import asyncio
from typing import Optional, Callable
from pathlib import Path
import logging
from datetime import datetime

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

# Hardcoded outline anchor - serves as non-chronological stop token
# Two-line format: (1) shows paper has end marker, (2) shows outline has its own end marker
OUTLINE_ANCHOR = "[HARD CODED BRACKETED DESIGNATION THAT SHOWS END-OF-PAPER DESIGNATION MARK]\n[HARD CODED END-OF-OUTLINE MARK -- ALL OUTLINE CONTENT SHOULD BE ABOVE THIS LINE]"

# Outline creation feedback file (stores last 5 validator feedbacks for iterative refinement)
OUTLINE_CREATION_FEEDBACK_FILE = "compiler_outline_creation_feedback.txt"


class OutlineMemory:
    """
    Manages the paper outline state.
    - File I/O to compiler_outline.txt
    - Triggers re-chunking when updated
    - Thread-safe operations
    - Version tracking
    """
    
    def __init__(self):
        self.version = 0
        self.rechunk_callback: Optional[Callable] = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._root_generation = system_config.runtime_root_generation
        self._file_path_override: Optional[Path] = None

    @property
    def file_path(self) -> Path:
        if self._file_path_override is not None:
            return self._file_path_override
        return Path(system_config.compiler_outline_file)

    @file_path.setter
    def file_path(self, value: str | Path) -> None:
        path = Path(value)
        default_path = Path(system_config.compiler_outline_file)
        self._file_path_override = (
            None
            if path.resolve(strict=False) == default_path.resolve(strict=False)
            else path
        )

    def _refresh_root(self) -> None:
        if self._root_generation != system_config.runtime_root_generation:
            self.version = 0
            self._initialized = False
            self._file_path_override = None
            self._root_generation = system_config.runtime_root_generation
    
    async def initialize(self) -> None:
        """Initialize outline memory."""
        async with self._lock:
            self._refresh_root()
            if self._initialized:
                return
            
            # Create file if doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                    await f.write("")
            
            self._initialized = True
            logger.info("Outline memory initialized")
    
    async def get_outline(self) -> str:
        """Get current outline content."""
        async with self._lock:
            if not self.file_path.exists():
                return ""
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return content
    
    def _strip_duplicate_anchors(self, content: str) -> str:
        """
        Remove all anchor occurrences from content.
        
        Args:
            content: Content to clean
        
        Returns:
            Content with anchors removed
        """
        return content.replace(OUTLINE_ANCHOR, "").strip()
    
    def _ensure_anchor(self, content: str) -> str:
        """
        Ensure outline ends with anchor marker.
        
        Args:
            content: Outline content
        
        Returns:
            Content with anchor appended at end
        """
        if not content.strip():
            return ""
        
        # Add anchor if not already at end
        content_stripped = content.rstrip()
        if not content_stripped.endswith(OUTLINE_ANCHOR):
            return content_stripped + "\n\n" + OUTLINE_ANCHOR
        return content_stripped
    
    async def update_outline(self, new_outline: str) -> None:
        """
        Update outline and trigger re-chunking.
        Automatically strips duplicate anchors and ensures single anchor at end.
        
        Args:
            new_outline: New outline content
        """
        async with self._lock:
            # Strip any duplicate anchors from input
            cleaned = self._strip_duplicate_anchors(new_outline)
            
            # Ensure single anchor at end
            final_content = self._ensure_anchor(cleaned)
            
            # Write to file
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(final_content)
            
            # Increment version
            self.version += 1
            
            logger.info(f"Outline updated (version {self.version})")
        
        # Trigger re-chunking callback OUTSIDE the lock to avoid deadlock
        if self.rechunk_callback:
            try:
                await self.rechunk_callback(final_content)
            except Exception as e:
                logger.error(f"Re-chunking callback failed: {e}")
    
    async def ensure_anchor_intact(self) -> bool:
        """
        Lightweight check to ensure OUTLINE_ANCHOR exists.
        
        Called BEFORE every old_string match operation on the outline
        to prevent failures caused by missing anchor during normal operation.
        
        Returns:
            True if anchor was repaired, False if it was already intact
        """
        async with self._lock:
            if not self.file_path.exists():
                return False
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                outline = await f.read()
            
            if not outline.strip():
                return False
            
            # Check if anchor exists
            if OUTLINE_ANCHOR in outline:
                return False
            
            # Anchor is missing - repair it
            final_content = self._ensure_anchor(outline)
            
            # Write repaired outline
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(final_content)
            
            # Increment version
            self.version += 1
            
            logger.info(f"Repaired missing outline anchor (version {self.version})")
            
            return True
    
    async def add_creation_feedback(self, reasoning: str, is_accepted: bool, outline_content: str = "") -> None:
        """
        Add validator feedback to creation feedback log (last 5).
        Used during iterative outline creation to provide submitter with feedback history.
        
        Args:
            reasoning: Validator's reasoning for accept/reject decision
            is_accepted: True if outline was accepted, False if rejected
            outline_content: The outline content (stored when accepted so model can see its own work)
        """
        async with self._lock:
            feedback_path = Path(system_config.data_dir) / OUTLINE_CREATION_FEEDBACK_FILE
            
            # Read existing feedbacks
            feedbacks = []
            if feedback_path.exists():
                async with aiofiles.open(feedback_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content.strip():
                        feedbacks = content.strip().split("\n\n---FEEDBACK SEPARATOR---\n\n")
            
            # Add new feedback with timestamp and status
            status = "ACCEPTED" if is_accepted else "REJECTED"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Include outline content when accepted so model can see its own work
            if is_accepted and outline_content:
                new_feedback = f"[{timestamp}] {status}\n{reasoning}\n\n---YOUR OUTLINE---\n{outline_content}"
            else:
                new_feedback = f"[{timestamp}] {status}\n{reasoning}"
            feedbacks.append(new_feedback)
            
            # Keep only last 5
            feedbacks = feedbacks[-5:]
            
            # Write back
            async with aiofiles.open(feedback_path, 'w', encoding='utf-8') as f:
                await f.write("\n\n---FEEDBACK SEPARATOR---\n\n".join(feedbacks))
            
            logger.debug(f"Added outline creation feedback: {status}")
    
    async def get_creation_feedback(self) -> str:
        """
        Get last 5 creation feedbacks formatted for prompt injection.
        
        CRITICAL: Extracts and prominently displays the last ACCEPTED outline
        so the model can see its own work and decide whether to refine or lock.
        
        Returns:
            Formatted feedback string for inclusion in outline_create prompt,
            or empty string if no feedback exists.
        """
        async with self._lock:
            feedback_path = Path(system_config.data_dir) / OUTLINE_CREATION_FEEDBACK_FILE
            
            if not feedback_path.exists():
                return ""
            
            async with aiofiles.open(feedback_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                return ""
            
            # Parse feedbacks and find last accepted outline
            feedbacks = content.strip().split("\n\n---FEEDBACK SEPARATOR---\n\n")
            formatted = []
            last_accepted_outline = None
            
            for i, feedback in enumerate(feedbacks, 1):
                # Extract outline content if present
                if "---YOUR OUTLINE---" in feedback:
                    # Use maxsplit=1 to handle edge case where outline contains the separator
                    parts = feedback.split("---YOUR OUTLINE---", 1)
                    reasoning_part = parts[0].strip()
                    outline_part = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Store the last accepted outline
                    if "ACCEPTED" in reasoning_part:
                        last_accepted_outline = outline_part
                    
                    # Show feedback without inline outline (outline shown separately at top)
                    formatted.append(f"FEEDBACK {i}:\n{reasoning_part}")
                else:
                    formatted.append(f"FEEDBACK {i}:\n{feedback}")
            
            # Build result with last accepted outline prominently displayed first
            result_parts = []
            
            if last_accepted_outline:
                result_parts.append(f"""YOUR LAST ACCEPTED OUTLINE (from previous iteration):
This outline was ACCEPTED by the validator. You can:
- Set outline_complete=true to LOCK this outline and begin paper construction
- Set outline_complete=false to continue refining (generate improved version)

---BEGIN OUTLINE---
{last_accepted_outline}
---END OUTLINE---""")
            
            result_parts.append("\n".join(formatted))
            
            return "\n\n".join(result_parts)
    
    async def clear_creation_feedback(self) -> None:
        """
        Clear creation feedback log (called when outline is locked).
        This prevents feedback from one paper leaking into the next paper's outline creation.
        """
        async with self._lock:
            feedback_path = Path(system_config.data_dir) / OUTLINE_CREATION_FEEDBACK_FILE
            if feedback_path.exists():
                feedback_path.unlink()
                logger.info("Outline creation feedback cleared")
    
    async def is_empty(self) -> bool:
        """Check if outline is empty."""
        content = await self.get_outline()
        return len(content.strip()) == 0
    
    def set_rechunk_callback(self, callback: Callable) -> None:
        """Set callback to trigger re-chunking when outline updated."""
        self.rechunk_callback = callback
    
    def get_version(self) -> int:
        """Get current outline version."""
        return self.version


# Global outline memory instance
outline_memory = OutlineMemory()

