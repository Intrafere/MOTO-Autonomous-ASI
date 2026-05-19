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


NON_RESUMABLE_SESSION_STATUSES = {"cleared", "history_only", "archived", "complete"}


def _session_paper_has_section(content: str, section_name: str) -> bool:
    base_patterns = [
        rf"##\s*{section_name}",
        rf"#\s*{section_name}",
        rf"\*\*{section_name}\*\*",
        rf"^{section_name}\s*$",
        rf"^\\(?:section|chapter)\*?\{{{section_name}\}}\s*$",
    ]
    if section_name == "Introduction":
        base_patterns.append(rf"^I\.\s*{section_name}")
        base_patterns.append(rf"^\\(?:section|chapter)\*?\{{I\.?\s*{section_name}\}}\s*$")
    elif section_name == "Conclusion":
        base_patterns.append(rf"^[IVXLC]+\.\s*{section_name}")

    return any(re.search(pattern, content, re.IGNORECASE | re.MULTILINE) for pattern in base_patterns)


def _detect_session_paper_phase(paper_content: str) -> str:
    has_abstract = _session_paper_has_section(paper_content, "Abstract")
    has_intro = _session_paper_has_section(paper_content, "Introduction")
    has_conclusion = _session_paper_has_section(paper_content, "Conclusion")

    has_abstract_placeholder = "[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION" in paper_content
    has_intro_placeholder = "[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION" in paper_content
    has_conclusion_placeholder = "[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION" in paper_content
    has_body_content = bool(re.search(r"^[IVX]+\.\s+\w", paper_content or "", re.MULTILINE))

    if not has_conclusion or has_conclusion_placeholder:
        return "conclusion" if has_body_content else "body"
    if not has_intro or has_intro_placeholder:
        return "introduction"
    if not has_abstract or has_abstract_placeholder:
        return "abstract"
    return "abstract"


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
                    session_status = str(metadata.get("status", "")).lower()
                    if metadata.get("resume_disabled") or session_status in NON_RESUMABLE_SESSION_STATUSES:
                        logger.error(
                            "Refusing to resume non-resumable session: %s (status=%s)",
                            session_id,
                            session_status or "unknown",
                        )
                        self._session_path = None
                        self._user_prompt = None
                        self._session_id = None
                        return None
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
            workflow_state = None
            session_metadata = {}
            user_prompt = ""
            try:
                session_metadata_path = session_dir / "session_metadata.json"
                if session_metadata_path.exists():
                    async with aiofiles.open(session_metadata_path, 'r', encoding='utf-8') as f:
                        session_metadata = json.loads(await f.read())
                    user_prompt = session_metadata.get("user_prompt", "") or session_metadata.get("user_research_prompt", "")

                session_status = str(session_metadata.get("status", "")).lower()
                if session_metadata.get("resume_disabled") or session_status in NON_RESUMABLE_SESSION_STATUSES:
                    logger.debug(
                        "Skipping non-resumable session %s (status=%s)",
                        session_dir.name,
                        session_status or "unknown",
                    )
                    continue

                if workflow_state_path.exists():
                    async with aiofiles.open(workflow_state_path, 'r', encoding='utf-8') as f:
                        raw = await f.read()
                    if raw.strip().strip('\x00'):
                        workflow_state = json.loads(raw)
                # Check if this session is resumable.
                # Resumable means: has a tier AND (has a topic OR has completed papers).
                has_tier = bool(workflow_state and workflow_state.get("current_tier") is not None)
                has_topic = bool(workflow_state and workflow_state.get("current_topic_id") is not None)
                has_papers = bool(workflow_state and workflow_state.get("papers_completed_count", 0) > 0)

                # A stale idle workflow_state.json can coexist with valid session
                # stats/brainstorm files. Try the durable-file recovery before
                # deciding the session is not resumable.
                if not (has_tier and (has_topic or has_papers)):
                    recovered_state = await self._recover_workflow_state_from_session_files(session_dir)
                    if recovered_state is not None:
                        workflow_state = recovered_state
                        has_tier = workflow_state.get("current_tier") is not None
                        has_topic = workflow_state.get("current_topic_id") is not None
                        has_papers = workflow_state.get("papers_completed_count", 0) > 0
                if workflow_state is None:
                    continue
                
                if has_tier and (has_topic or has_papers):
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

    async def _recover_workflow_state_from_session_files(self, session_dir: Path) -> Optional[Dict[str, Any]]:
        """Build a conservative resume state from session stats/brainstorm files.

        This protects sessions where the workflow checkpoint was stale or absent
        but durable brainstorm metadata still shows work in progress.  It only
        resumes a current stats pointer, an in-progress brainstorm, or a completed
        brainstorm that has not produced a paper yet.
        """
        try:
            stats = {}
            stats_path = session_dir / "session_stats.json"
            if stats_path.exists():
                async with aiofiles.open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.loads(await f.read())

            topic_id = stats.get("current_brainstorm_id")
            paper_id = stats.get("current_paper_id")
            topic_metadata = None
            paper_metadata = None
            paper_title = None
            reference_paper_ids = []

            brainstorms_dir = session_dir / "brainstorms"
            papers_dir = session_dir / "papers"
            if paper_id and papers_dir.exists():
                paper_metadata_path = papers_dir / f"paper_{paper_id}_metadata.json"
                if paper_metadata_path.exists():
                    async with aiofiles.open(paper_metadata_path, 'r', encoding='utf-8') as f:
                        paper_metadata = json.loads(await f.read())
                    if paper_metadata.get("status") == "in_progress":
                        paper_title = paper_metadata.get("title")
                        reference_paper_ids = paper_metadata.get("referenced_papers") or []
                        if not topic_id:
                            source_ids = paper_metadata.get("source_brainstorm_ids") or []
                            topic_id = source_ids[0] if source_ids else None
                    else:
                        # `current_paper_id` is sticky in stats; a completed paper
                        # must not make a stale/idle session look like active paper writing.
                        paper_id = None
                else:
                    paper_id = None

            if not paper_id and papers_dir.exists():
                paper_candidates = []
                for paper_metadata_path in papers_dir.glob("paper_*_metadata.json"):
                    try:
                        async with aiofiles.open(paper_metadata_path, 'r', encoding='utf-8') as f:
                            data = json.loads(await f.read())
                        if data.get("status") == "in_progress":
                            paper_candidates.append(data)
                    except Exception:
                        continue
                if paper_candidates:
                    paper_candidates.sort(key=lambda item: item.get("created_at", ""), reverse=True)
                    paper_metadata = paper_candidates[0]
                    paper_id = paper_metadata.get("paper_id")
                    paper_title = paper_metadata.get("title")
                    reference_paper_ids = paper_metadata.get("referenced_papers") or []
                    if not topic_id:
                        source_ids = paper_metadata.get("source_brainstorm_ids") or []
                        topic_id = source_ids[0] if source_ids else None

            if topic_id and brainstorms_dir.exists():
                metadata_path = brainstorms_dir / f"brainstorm_{topic_id}_metadata.json"
                if metadata_path.exists():
                    async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                        topic_metadata = json.loads(await f.read())

            if topic_metadata is None and brainstorms_dir.exists():
                candidates = []
                for metadata_path in brainstorms_dir.glob("brainstorm_*_metadata.json"):
                    try:
                        async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                            data = json.loads(await f.read())
                        status = data.get("status")
                        papers_generated = data.get("papers_generated") or []
                        if status == "in_progress" or (status == "complete" and not papers_generated):
                            candidates.append(data)
                    except Exception:
                        continue
                if candidates:
                    candidates.sort(key=lambda item: item.get("last_activity", ""), reverse=True)
                    topic_metadata = candidates[0]
                    topic_id = topic_metadata.get("topic_id")

            if not topic_id and not paper_id:
                return None

            current_tier = "tier2_paper_writing" if paper_id else "tier1_aggregation"
            paper_phase = None
            if paper_id:
                paper_path = papers_dir / f"paper_{paper_id}.txt"
                if paper_path.exists():
                    async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                        paper_phase = _detect_session_paper_phase(await f.read())
                else:
                    paper_phase = "body"
            acceptance_count = int((topic_metadata or {}).get("submission_count") or 0)
            if (
                topic_metadata
                and topic_metadata.get("status") == "complete"
                and not paper_id
                and not (topic_metadata.get("papers_generated") or [])
            ):
                current_tier = "tier2_paper_writing"
                paper_phase = "brainstorm_proof_verification"
            elif topic_metadata and topic_metadata.get("status") == "complete" and not paper_id:
                return None

            return {
                "is_running": False,
                "current_tier": current_tier,
                "current_topic_id": topic_id,
                "current_paper_id": paper_id,
                "current_paper_title": paper_title,
                "paper_phase": paper_phase,
                "reference_paper_ids": reference_paper_ids,
                "acceptance_count": acceptance_count,
                "rejection_count": 0,
                "consecutive_rejections": 0,
                "exhaustion_signals": 0,
                "papers_completed_count": stats.get("total_papers_completed", 0),
                "last_redundancy_check_at": 0,
                "last_completion_review_at": 0,
                "last_tier3_check_at": 0,
                "brainstorm_paper_count": 0,
                "current_brainstorm_paper_ids": [],
                "proof_framing_active": False,
                "proof_framing_context": "",
                "proof_framing_reasoning": "",
                "tier3_active": False,
                "tier3_enabled": False,
                "tier3_format": None,
                "tier3_phase": None,
                "model_config": {},
                "last_updated": stats.get("last_updated") or (topic_metadata or {}).get("last_activity", ""),
            }
        except Exception as exc:
            logger.debug(f"Failed to recover workflow state from session files {session_dir.name}: {exc}")
            return None

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
                            metadata["path"] = session_dir.name
                            
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

