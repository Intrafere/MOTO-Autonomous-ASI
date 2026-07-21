"""
Persistent event log for aggregator.
Stores user-visible manual Aggregator activity to file.
"""
import aiofiles
import json
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import logging
from datetime import datetime

from backend.shared.config import system_config

logger = logging.getLogger(__name__)


class EventLog:
    """
    Persistent event log for aggregator.
    Stores user-visible run events to file for persistence across restarts
    and tab switches.
    """
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._root_generation = system_config.runtime_root_generation
        self._file_path_override: Path | None = None

    @property
    def file_path(self) -> Path:
        return self._file_path_override or Path(system_config.data_dir) / "aggregator_event_log.txt"

    @file_path.setter
    def file_path(self, value: str | Path) -> None:
        """Allow focused tests/custom stores to override without stale default-root capture."""
        self._file_path_override = Path(value)
        self._root_generation = system_config.runtime_root_generation

    def _refresh_root(self) -> None:
        if self._root_generation != system_config.runtime_root_generation:
            self.events = []
            self._file_path_override = None
            self._root_generation = system_config.runtime_root_generation
    
    async def initialize(self) -> None:
        """Initialize event log, loading existing events from file."""
        self._refresh_root()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_path.exists():
            try:
                async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content.strip():
                        # Each line is a JSON object
                        self.events = []
                        for line in content.strip().split('\n'):
                            if line.strip():
                                try:
                                    event = json.loads(line)
                                    self.events.append(event)
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse event log line: {e}")
                logger.debug(f"Loaded {len(self.events)} events from event log")
            except Exception as e:
                logger.error(f"Failed to load event log: {e}")
                self.events = []
        else:
            # Create empty file
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write("")
            logger.info("Created new event log file")
    
    async def add_event(self, event_type: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a key event to the log.
        
        Args:
            event_type: Type of event (system_started, new_submission, submission_accepted, etc.)
            message: Human-readable message
            metadata: Optional additional data (submitter_id, submission_number, etc.)
        """
        async with self._lock:
            self._refresh_root()
            event = {
                'id': len(self.events) + 1,
                'type': event_type,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.events.append(event)
            
            # Append to file (one JSON object per line)
            try:
                async with aiofiles.open(self.file_path, 'a', encoding='utf-8') as f:
                    await f.write(json.dumps(event) + '\n')
                logger.debug(f"Logged event: {event_type}")
            except Exception as e:
                logger.error(f"Failed to write event to log: {e}")
    
    async def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events from the log."""
        async with self._lock:
            self._refresh_root()
            return list(self.events)
    
    async def clear(self) -> None:
        """Clear all events from the log."""
        async with self._lock:
            self._refresh_root()
            self.events = []
            try:
                async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                    await f.write("")
                logger.info("Cleared event log")
            except Exception as e:
                logger.error(f"Failed to clear event log: {e}")


# Global event log instance
event_log = EventLog()

