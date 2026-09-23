"""
Critique rejection feedback memory.
Tracks last 5 rejections to help critique submitter learn from mistakes.
"""
import aiofiles
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Mapping
import asyncio
import logging

from backend.shared.config import system_config, rag_config
from backend.shared.utils import truncate_with_ellipsis

logger = logging.getLogger(__name__)


class CritiqueRejectionMemory:
    """
    Critique rejection log.
    Maintains rolling window of last 5 rejections.
    """
    
    def __init__(self):
        self.file_path = Path(
            f"{system_config.data_dir}/critique_rejection_feedback.txt"
        )
        self.rejections: List[dict] = []
        self.max_rejections = rag_config.max_local_rejections
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize critique rejection memory."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_path.exists():
            # Load existing rejections
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    # Parse rejection entries
                    entries = content.split('\n---\n')
                    for entry in entries:
                        if entry.strip():
                            parts = entry.split('\n[SUBMISSION PREVIEW]\n')
                            if len(parts) == 2:
                                self.rejections.append({
                                    'validator_summary': parts[0].replace('[VALIDATOR SUMMARY]\n', '').strip(),
                                    'submission_preview': parts[1].strip()
                                })
            logger.info(f"Loaded {len(self.rejections)} critique rejections")
        else:
            # Create empty file
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write("")
            logger.info("Created new critique rejection log")
    
    async def add_rejection(
        self,
        validator_summary: str,
        submission_content: str
    ) -> None:
        """
        Add a rejection to the log.
        
        Args:
            validator_summary: Validator's reasoning (max 750 chars)
            submission_content: Original submission (first 750 chars)
        """
        async with self._lock:
            # Truncate to limits
            summary = truncate_with_ellipsis(validator_summary, 750)
            preview = truncate_with_ellipsis(submission_content, 750)
            
            # Add rejection
            self.rejections.append({
                'validator_summary': summary,
                'submission_preview': preview
            })
            
            # Keep only last N rejections
            if len(self.rejections) > self.max_rejections:
                self.rejections.pop(0)
            
            # Save to file
            await self._save()
    
    async def reset(self) -> None:
        """Reset (clear) all rejections."""
        async with self._lock:
            self.rejections = []
            await self._save()
            logger.info("Reset critique rejection log")
    
    async def clear(self) -> None:
        """Alias for reset() - clear all rejections."""
        await self.reset()
    
    async def get_all_content(self) -> str:
        """
        Get all rejection feedback as formatted string.
        
        Returns:
            Formatted rejection log for prompt context
        """
        return self.render_rejections(await self.get_rejection_entries())

    async def get_rejection_entries(self) -> tuple[dict, ...]:
        """Return an immutable snapshot of chronological critique rejections."""
        async with self._lock:
            return tuple(deepcopy(self.rejections))

    @staticmethod
    def render_rejections(rejections: Iterable[Mapping[str, str]]) -> str:
        """Render a caller-selected whole-entry critique feedback projection."""
        sections = []
        for i, rejection in enumerate(rejections, 1):
            section = f"[REJECTION {i}]\n"
            section += f"Validator Feedback: {rejection['validator_summary']}\n"
            section += f"Your Submission Preview: {rejection['submission_preview']}\n"
            sections.append(section)
        return '\n---\n'.join(sections)
    
    async def _save(self) -> None:
        """Save rejections to file."""
        try:
            content_parts = []
            for rejection in self.rejections:
                part = f"[VALIDATOR SUMMARY]\n{rejection['validator_summary']}\n"
                part += f"[SUBMISSION PREVIEW]\n{rejection['submission_preview']}"
                content_parts.append(part)
            
            content = '\n---\n'.join(content_parts)
            
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
        except Exception as e:
            logger.error(f"Failed to save critique rejection log: {e}")

