"""
Critique Memory - manages critique feedback database for peer review aggregation.
Stores accepted critiques separately from main shared training database.
"""
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
import logging
import re
from datetime import datetime

from backend.shared.config import system_config

logger = logging.getLogger(__name__)


class CritiqueMemory:
    """
    Manages critique feedback database for peer review aggregation phase.
    Similar to SharedTrainingMemory but for critique-specific feedback.
    """
    
    def __init__(self):
        self.critique_file_path: Optional[Path] = None
        self.critiques: List[Dict[str, str]] = []
        self._lock = asyncio.Lock()
        self.critique_count = 0
    
    def initialize(self, paper_id: str):
        """
        Initialize critique database for specific paper.
        
        Args:
            paper_id: Unique identifier for paper (e.g., "paper_v1")
        """
        self.critique_file_path = Path(system_config.data_dir) / f"critique_feedback_{paper_id}.txt"
        logger.info(f"Initialized critique memory for {paper_id} at {self.critique_file_path}")
    
    async def add_accepted_critique(self, critique_content: str) -> None:
        """
        Add an accepted critique to the database.
        
        Args:
            critique_content: The full critique text
        """
        if not self.critique_file_path:
            logger.error("Critique memory not initialized - call initialize() first")
            return
            
        async with self._lock:
            # Increment critique count
            self.critique_count += 1
            
            # Add critique with metadata
            self.critiques.append({
                'content': critique_content,
                'timestamp': datetime.now().isoformat(),
                'number': self.critique_count
            })
            
            # Save to file
            await self._save()
            
            logger.info(f"Added critique #{self.critique_count} to critique memory")
    
    async def get_all_critiques(self) -> str:
        """
        Get all accepted critiques as formatted string.
        
        Returns:
            All critiques with formatting for display
        """
        async with self._lock:
            if not self.critiques:
                return ""
            
            formatted_sections = []
            for critique in self.critiques:
                section = f"{'=' * 80}\n"
                section += f"CRITIQUE #{critique['number']} | Accepted: {critique['timestamp']}\n"
                section += f"{'=' * 80}\n"
                section += f"{critique['content']}\n"
                formatted_sections.append(section)
            
            return '\n'.join(formatted_sections)
    
    async def get_all_submissions(self) -> str:
        """
        Get all critiques as plain content (for compatibility with aggregator interface).
        This method exists to match the SharedTrainingMemory interface.
        
        Returns:
            All critique content without formatting
        """
        async with self._lock:
            return '\n\n'.join([critique['content'] for critique in self.critiques])
    
    async def get_critique_count(self) -> int:
        """
        Count number of accepted critiques (non-pruned).
        
        Returns:
            Number of critiques currently in database
        """
        async with self._lock:
            return len(self.critiques)
    
    async def remove_critique(self, critique_number: int) -> bool:
        """
        Remove critique by number (for pruning during cleanup review).
        
        Args:
            critique_number: The critique number to remove
            
        Returns:
            True if critique was found and removed, False otherwise
        """
        async with self._lock:
            # Find critique with matching number
            for i, critique in enumerate(self.critiques):
                if critique.get('number') == critique_number:
                    self.critiques.pop(i)
                    logger.info(f"Removed critique #{critique_number} from critique memory")
                    
                    # Save updated database
                    await self._save()
                    return True
            
            logger.warning(f"Critique #{critique_number} not found for removal")
            return False
    
    async def get_submission_content(self, submission_number: int) -> Optional[str]:
        """
        Get critique content by number (for compatibility with cleanup review).
        
        Args:
            submission_number: The critique number to retrieve
            
        Returns:
            Critique content if found, None otherwise
        """
        async with self._lock:
            for critique in self.critiques:
                if critique.get('number') == submission_number:
                    return critique['content']
            return None
    
    async def clear(self) -> None:
        """Clear all critiques and reset (start fresh for new critique phase)."""
        if not self.critique_file_path:
            logger.error("Critique memory not initialized - call initialize() first")
            return
            
        async with self._lock:
            self.critiques = []
            self.critique_count = 0
            
            # Clear file
            if self.critique_file_path.exists():
                async with aiofiles.open(self.critique_file_path, 'w', encoding='utf-8') as f:
                    await f.write("")
            
            logger.info("Cleared all critiques from critique memory")
    
    async def _save(self) -> None:
        """Save critiques to file with formatting."""
        if not self.critique_file_path:
            logger.error("Cannot save - critique memory not initialized")
            return
            
        # Ensure parent directory exists
        self.critique_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format critiques
        formatted_sections = []
        for critique in self.critiques:
            section = f"{'=' * 80}\n"
            section += f"CRITIQUE #{critique['number']} | Accepted: {critique['timestamp']}\n"
            section += f"{'=' * 80}\n"
            section += f"{critique['content']}\n"
            formatted_sections.append(section)
        
        # Write to file
        async with aiofiles.open(self.critique_file_path, 'w', encoding='utf-8') as f:
            await f.write('\n'.join(formatted_sections))
    
    async def load_from_file(self) -> None:
        """Load critiques from file (for crash recovery)."""
        if not self.critique_file_path:
            logger.error("Cannot load - critique memory not initialized")
            return
            
        if not self.critique_file_path.exists():
            logger.info("No existing critique file to load")
            return
            
        async with self._lock:
            async with aiofiles.open(self.critique_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            if not content.strip():
                logger.info("Critique file is empty")
                return
            
            # Parse formatted critiques
            self.critiques = self._parse_formatted_file(content)
            
            # Set critique_count to highest number found
            if self.critiques:
                max_number = max(
                    (critique.get('number', 0) for critique in self.critiques),
                    default=0
                )
                self.critique_count = max_number
            else:
                self.critique_count = 0
            
            logger.info(f"Loaded {len(self.critiques)} critiques from file (max number: {self.critique_count})")
    
    def _parse_formatted_file(self, content: str) -> List[Dict[str, str]]:
        """Parse the formatted file to extract critiques and metadata."""
        critiques = []
        
        # Pattern matches: separator + header + separator + content
        pattern = r'={80}\s*CRITIQUE #(\d+)\s*\|\s*Accepted:\s*([^\n]+)\s*={80}\s*\n(.*?)(?=\n={80}\s*CRITIQUE|$)'
        
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            number = int(match.group(1))
            timestamp = match.group(2).strip()
            content_text = match.group(3).strip()
            
            if content_text:
                critiques.append({
                    'content': content_text,
                    'timestamp': timestamp,
                    'number': number
                })
        
        return critiques


# Global singleton instance
critique_memory = CritiqueMemory()

