"""
Session Manager - Manages prompt-based session folder organization.
Each research session (user prompt) gets its own folder for brainstorms, papers, and final answers.
"""
import asyncio
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import aiofiles

from backend.shared.path_safety import (
    resolve_path_within_root,
    validate_single_path_component,
)
from backend.shared.config import system_config

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages prompt-based session folder organization.
    
    Creates a new session folder for each autonomous research start,
    based on sanitized user prompt + timestamp.
    
    Structure:
        backend/data/auto_sessions/
        └── {sanitized_prompt}_{timestamp}/
            ├── brainstorms/
            ├── papers/
            ├── proofs/
            ├── final_answer/
            └── session_metadata.json
    """
    
    _instance: Optional['SessionManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._base_dir: Optional[Path] = None
        self._session_path: Optional[Path] = None
        self._user_prompt: Optional[str] = None
        self._session_id: Optional[str] = None
        self._initialized = True
    
    @property
    def is_session_active(self) -> bool:
        """Check if a session is currently active."""
        return self._session_path is not None and self._session_path.exists()
    
    @property
    def session_path(self) -> Optional[Path]:
        """Get current session path."""
        return self._session_path
    
    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._session_id
    
    def sanitize_prompt_for_folder(self, prompt: str, max_length: int = 50) -> str:
        """
        Convert user prompt to a safe folder name.
        
        - Takes first max_length characters
        - Normalizes unicode
        - Replaces spaces and special chars with underscores
        - Removes consecutive underscores
        - Converts to lowercase
        
        Args:
            prompt: The user research prompt
            max_length: Maximum length for the folder name (default 50)
            
        Returns:
            Safe folder name string
        """
        # Normalize unicode to ASCII equivalents where possible
        normalized = unicodedata.normalize('NFKD', prompt)
        normalized = normalized.encode('ascii', 'ignore').decode('ascii')
        
        # Take first max_length characters
        truncated = normalized[:max_length]
        
        # Replace non-alphanumeric with underscores
        safe = re.sub(r'[^a-zA-Z0-9]+', '_', truncated)
        
        # Remove leading/trailing underscores
        safe = safe.strip('_')
        
        # Convert to lowercase
        safe = safe.lower()
        
        # Handle empty result
        if not safe:
            safe = "research_session"
        
        return safe
    
    def _generate_session_id(self, prompt: str) -> str:
        """
        Generate a unique session ID from prompt + timestamp.
        
        Format: {sanitized_prompt}_{YYYY-MM-DD_HH-MM}
        """
        sanitized = self.sanitize_prompt_for_folder(prompt)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return f"{sanitized}_{timestamp}"
    
    async def initialize(self, user_prompt: str, base_dir: Optional[str] = None) -> Path:
        """
        Initialize a new session for the given user prompt.
        
        Creates a new session folder with brainstorms, papers, and final_answer subdirectories.
        
        Args:
            user_prompt: The user's research prompt
            base_dir: Base directory for all sessions
            
        Returns:
            Path to the session folder
        """
        async with self._lock:
            self._base_dir = Path(base_dir or system_config.auto_sessions_base_dir)
            self._user_prompt = user_prompt
            self._session_id = self._generate_session_id(user_prompt)
            self._session_path = self._base_dir / self._session_id
            
            # Create directory structure
            self._session_path.mkdir(parents=True, exist_ok=True)
            (self._session_path / "brainstorms").mkdir(exist_ok=True)
            (self._session_path / "papers").mkdir(exist_ok=True)
            (self._session_path / "proofs").mkdir(exist_ok=True)
            (self._session_path / "final_answer").mkdir(exist_ok=True)
            
            # Save session metadata
            metadata = {
                "session_id": self._session_id,
                "user_prompt": user_prompt,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            metadata_path = self._session_path / "session_metadata.json"
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            logger.info(f"Session initialized: {self._session_id}")
            logger.info(f"Session path: {self._session_path}")
            
            return self._session_path
    
    async def resume_session(self, session_id: str, base_dir: Optional[str] = None) -> Optional[Path]:
        """
        Resume an existing session by ID.
        
        Args:
            session_id: The session ID to resume
            base_dir: Base directory for all sessions
            
        Returns:
            Path to the session folder, or None if not found
        """
        async with self._lock:
            self._base_dir = Path(base_dir or system_config.auto_sessions_base_dir)
            try:
                safe_session_id = validate_single_path_component(session_id, "session ID")
                self._session_path = resolve_path_within_root(self._base_dir, safe_session_id)
            except ValueError as e:
                logger.error(f"Invalid session ID: {session_id} ({e})")
                return None
            
            if not self._session_path.exists():
                logger.error(f"Session not found: {session_id}")
                return None
            
            # Load metadata
            metadata_path = self._session_path / "session_metadata.json"
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
                    self._user_prompt = metadata.get("user_prompt", "")
                    self._session_id = metadata.get("session_id", session_id)
            else:
                self._session_id = session_id
                self._user_prompt = ""
            
            # Update status
            await self._update_metadata({"status": "active", "resumed_at": datetime.now().isoformat()})
            
            logger.info(f"Session resumed: {self._session_id}")
            return self._session_path
    
    async def _update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update session metadata."""
        if not self._session_path:
            return
            
        metadata_path = self._session_path / "session_metadata.json"
        
        # Load existing metadata
        metadata = {}
        if metadata_path.exists():
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
        
        # Apply updates
        metadata.update(updates)
        metadata["last_updated"] = datetime.now().isoformat()
        
        # Save
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=2))

    async def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Public wrapper for updating session metadata fields."""
        await self._update_metadata(updates)
    
    def get_brainstorms_dir(self) -> Path:
        """Get brainstorms subdirectory for current session."""
        if not self._session_path:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._session_path / "brainstorms"
    
    def get_papers_dir(self) -> Path:
        """Get papers subdirectory for current session."""
        if not self._session_path:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._session_path / "papers"

    def get_proofs_dir(self) -> Path:
        """Get proofs subdirectory for current session."""
        if not self._session_path:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._session_path / "proofs"
    
    def get_final_answer_dir(self) -> Path:
        """Get final_answer subdirectory for current session."""
        if not self._session_path:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._session_path / "final_answer"
    
    def get_metadata_path(self) -> Path:
        """Get path to session metadata file."""
        if not self._session_path:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._session_path / "session_metadata.json"
    
    async def mark_complete(self) -> None:
        """Mark the current session as complete."""
        await self._update_metadata({
            "status": "complete",
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"Session marked complete: {self._session_id}")
    
    async def clear(self) -> None:
        """Clear the current session (reset singleton state)."""
        async with self._lock:
            self._session_path = None
            self._user_prompt = None
            self._session_id = None
            logger.info("Session manager cleared")
    
    async def find_interrupted_session(self, base_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find the most recent RESUMABLE session in its workflow_state.
        
        A session is resumable if:
        1. is_running=True (crashed/interrupted), OR
        2. is_running=False but has current_tier AND (current_topic_id OR papers_completed > 0)
           (user pressed stop but work can be resumed)
        
        Scans all session directories for a workflow_state.json that is resumable,
        returns the most recent one by last_updated timestamp.
        
        Args:
            base_dir: Base directory for all sessions
            
        Returns:
            Session info dict with session_id, path, user_prompt, workflow_state
            Or None if no resumable session found
        """
        base_path = Path(base_dir or system_config.auto_sessions_base_dir)
        
        if not base_path.exists():
            return None
        
        resumable_sessions = []
        
        for session_dir in base_path.iterdir():
            if not session_dir.is_dir():
                continue
                
            workflow_state_path = session_dir / "workflow_state.json"
            if not workflow_state_path.exists():
                continue
                
            try:
                async with aiofiles.open(workflow_state_path, 'r', encoding='utf-8') as f:
                    raw = await f.read()
                if not raw.strip().strip('\x00'):
                    continue  # Empty or null-padded file — skip silently
                workflow_state = json.loads(raw)
                
                # Check if this session is resumable
                # Resumable means: has a tier AND (has a topic OR has completed papers)
                has_tier = workflow_state.get("current_tier") is not None
                has_topic = workflow_state.get("current_topic_id") is not None
                has_papers = workflow_state.get("papers_completed_count", 0) > 0
                
                if has_tier and (has_topic or has_papers):
                    # Load session metadata for user prompt
                    session_metadata_path = session_dir / "session_metadata.json"
                    user_prompt = ""
                    if session_metadata_path.exists():
                        async with aiofiles.open(session_metadata_path, 'r', encoding='utf-8') as f:
                            session_metadata = json.loads(await f.read())
                            user_prompt = session_metadata.get("user_prompt", "")
                    
                    resumable_sessions.append({
                        "session_id": session_dir.name,
                        "path": str(session_dir),
                        "user_prompt": user_prompt,
                        "workflow_state": workflow_state,
                        "last_updated": workflow_state.get("last_updated", ""),
                        "was_running": workflow_state.get("is_running", False)
                    })
            except Exception as e:
                logger.debug(f"Skipping unreadable workflow state in {session_dir.name}: {e}")
                continue
        
        if not resumable_sessions:
            return None
        
        # Sort by last_updated descending and return the most recent
        resumable_sessions.sort(key=lambda x: x["last_updated"], reverse=True)
        
        most_recent = resumable_sessions[0]
        status = "interrupted" if most_recent.get("was_running") else "paused"
        logger.info(f"Found {status} session: {most_recent['session_id']} (last updated: {most_recent['last_updated']})")
        
        return most_recent

    async def list_all_sessions(self, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all research sessions.
        
        Returns:
            List of session metadata dictionaries
        """
        base_path = Path(base_dir or system_config.auto_sessions_base_dir)
        sessions = []
        
        if not base_path.exists():
            return sessions
        
        for session_dir in sorted(base_path.iterdir(), reverse=True):
            if session_dir.is_dir():
                metadata_path = session_dir / "session_metadata.json"
                if metadata_path.exists():
                    try:
                        async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.loads(await f.read())
                            metadata["path"] = str(session_dir)
                            
                            # Count items in subdirectories
                            brainstorms_dir = session_dir / "brainstorms"
                            papers_dir = session_dir / "papers"
                            
                            brainstorm_count = len(list(brainstorms_dir.glob("brainstorm_*.txt"))) if brainstorms_dir.exists() else 0
                            paper_count = len(list(papers_dir.glob("paper_*.txt"))) if papers_dir.exists() else 0
                            
                            metadata["brainstorm_count"] = brainstorm_count
                            metadata["paper_count"] = paper_count
                            
                            sessions.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to read session metadata: {session_dir}: {e}")
        
        return sessions


# Global singleton instance
session_manager = SessionManager()

