"""
Research Metadata - Central metadata tracking for autonomous research.
Manages brainstorm-paper associations and statistics.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import aiofiles

from backend.shared.config import system_config
from backend.shared.models import BrainstormMetadata, PaperMetadata
from backend.shared.path_safety import resolve_path_within_root

logger = logging.getLogger(__name__)


class ResearchMetadata:
    """
    Central metadata management for autonomous research.
    Tracks all brainstorms, papers, and their associations.
    Also persists workflow state for crash recovery/resume.
    
    Supports both:
    - Legacy mode: Uses system_config paths
    - Session mode: Uses session_manager paths
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._metadata_path = Path(system_config.auto_research_metadata_file)
        self._stats_path = Path(system_config.auto_research_stats_file)
        self._workflow_state_path = Path(system_config.auto_workflow_state_file)
        self._data: Dict[str, Any] = None
        self._stats: Dict[str, Any] = None
        self._workflow_state: Dict[str, Any] = None
        self._session_manager = None
    
    def set_session_manager(self, session_manager) -> None:
        """Set session manager for session-based path resolution."""
        self._session_manager = session_manager
        if session_manager and session_manager.is_session_active:
            session_path = resolve_path_within_root(
                session_manager.session_path.parent,
                session_manager.session_path.name,
            )
            self._metadata_path = session_path / "session_metadata.json"
            self._stats_path = session_path / "session_stats.json"
            self._workflow_state_path = session_path / "workflow_state.json"
            logger.info(f"Research metadata using session path: {session_path}")
    
    def _get_default_stats(self) -> Dict[str, Any]:
        """Default statistics structure."""
        return {
            "total_brainstorms_created": 0,
            "total_brainstorms_completed": 0,
            "total_papers_completed": 0,
            "total_papers_archived": 0,
            "total_submissions_accepted": 0,
            "total_submissions_rejected": 0,
            "topic_selection_rejections": 0,
            "completion_reviews_run": 0,
            "paper_redundancy_reviews_run": 0,
            "current_brainstorm_id": None,
            "current_paper_id": None,
            "acceptance_rate": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _ensure_initialized(self) -> None:
        """
        Ensure metadata, stats, and workflow state are loaded before use.
        This prevents NoneType errors when endpoints are hit before the
        autonomous coordinator has been initialized.
        """
        # Metadata
        if self._data is None:
            if self._metadata_path.exists():
                await self._load_metadata()
            if self._data is None:
                self._data = {
                    "user_research_prompt": "",
                    "base_user_research_prompt": "",
                    "proof_framing_active": False,
                    "proof_framing_context": "",
                    "proof_framing_reasoning": "",
                    "proof_runtime_config": {},
                    "brainstorms": [],
                    "papers": [],
                    "next_topic_id": 1,
                    "next_paper_id": 1,
                    "created_at": datetime.now().isoformat()
                }
                await self._save_metadata()
        
        # Workflow state
        if self._workflow_state is None:
            await self._load_workflow_state()
            if self._workflow_state is None:
                self._workflow_state = self._get_default_workflow_state()
        
        # Stats
        if self._stats is None:
            if self._stats_path.exists():
                await self._load_stats()
            if self._stats is None:
                self._stats = self._get_default_stats()
                await self._save_stats()
    
    async def initialize(self, user_research_prompt: str = "") -> None:
        """Initialize or load research metadata."""
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._metadata_path.exists():
            await self._load_metadata()
            # If prompt provided and differs from saved, optionally update
            if user_research_prompt and self._data.get("user_research_prompt") != user_research_prompt:
                logger.info("User research prompt updated")
                self._data["user_research_prompt"] = user_research_prompt
                if not self._data.get("base_user_research_prompt"):
                    self._data["base_user_research_prompt"] = user_research_prompt
                await self._save_metadata()
        else:
            self._data = {
                "user_research_prompt": user_research_prompt,
                "base_user_research_prompt": user_research_prompt,
                "proof_framing_active": False,
                "proof_framing_context": "",
                "proof_framing_reasoning": "",
                "proof_runtime_config": {},
                "brainstorms": [],
                "papers": [],
                "next_topic_id": 1,
                "next_paper_id": 1,
                "created_at": datetime.now().isoformat()
            }
            await self._save_metadata()
        
        # Load workflow state for resume capability
        await self._load_workflow_state()
        
        if self._stats_path.exists():
            await self._load_stats()
        else:
            self._stats = self._get_default_stats()
            await self._save_stats()
        
        logger.info("Research metadata initialized")
    
    async def _load_metadata(self) -> None:
        """Load metadata from file."""
        needs_save = False
        try:
            async with aiofiles.open(self._metadata_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                self._data = json.loads(content)
                
            # Ensure all required keys exist (schema migration)
            # This is OUTSIDE the async with block so we can safely save
            defaults = {
                "user_research_prompt": "",
                "base_user_research_prompt": "",
                "proof_framing_active": False,
                "proof_framing_context": "",
                "proof_framing_reasoning": "",
                "proof_runtime_config": {},
                "brainstorms": [],
                "papers": [],
                "next_topic_id": 1,
                "next_paper_id": 1,
                "created_at": datetime.now().isoformat()
            }
            for key, default_value in defaults.items():
                if key not in self._data:
                    self._data[key] = default_value
                    needs_save = True
                    logger.info(f"Added missing key '{key}' to research metadata")
                    
        except Exception as e:
            logger.error(f"Failed to load research metadata: {e}")
            self._data = {
                "user_research_prompt": "",
                "base_user_research_prompt": "",
                "proof_framing_active": False,
                "proof_framing_context": "",
                "proof_framing_reasoning": "",
                "proof_runtime_config": {},
                "brainstorms": [],
                "papers": [],
                "next_topic_id": 1,
                "next_paper_id": 1,
                "created_at": datetime.now().isoformat()
            }
            needs_save = True
        
        # Save if we added missing keys or created fresh defaults
        if needs_save:
            await self._save_metadata()
    
    async def _save_metadata(self) -> None:
        """Save metadata to file."""
        async with aiofiles.open(self._metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(self._data, indent=2))
    
    async def _load_stats(self) -> None:
        """Load statistics from file."""
        try:
            async with aiofiles.open(self._stats_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                self._stats = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load research stats: {e}")
            self._stats = self._get_default_stats()
    
    async def _save_stats(self) -> None:
        """Save statistics to file."""
        self._stats["last_updated"] = datetime.now().isoformat()
        async with aiofiles.open(self._stats_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(self._stats, indent=2))
    
    # ========================================================================
    # WORKFLOW STATE (For crash recovery / resume)
    # ========================================================================
    
    async def _load_workflow_state(self) -> None:
        """Load workflow state from file for resume capability."""
        if self._workflow_state_path.exists():
            try:
                async with aiofiles.open(self._workflow_state_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    self._workflow_state = json.loads(content)
                    logger.info(f"Workflow state loaded: tier={self._workflow_state.get('current_tier')}")
            except Exception as e:
                logger.error(f"Failed to load workflow state: {e}")
                self._workflow_state = self._get_default_workflow_state()
        else:
            self._workflow_state = self._get_default_workflow_state()
    
    def _get_default_workflow_state(self) -> Dict[str, Any]:
        """Get default workflow state structure."""
        return {
            "is_running": False,
            "current_tier": None,  # "tier1_aggregation", "tier2_paper_writing", or "tier3_final_answer"
            "current_topic_id": None,
            "current_paper_id": None,
            "current_paper_title": None,
            "paper_phase": None,  # "body", "conclusion", "introduction", "abstract"
            "base_user_research_prompt": "",
            "proof_framing_active": False,
            "proof_framing_context": "",
            "proof_framing_reasoning": "",
            "acceptance_count": 0,
            "rejection_count": 0,
            "consecutive_rejections": 0,
            "exhaustion_signals": 0,
            "papers_completed_count": 0,
            "last_redundancy_check_at": 0,
            "last_tier3_check_at": 0,
            # Tier 3 Final Answer state (for crash recovery)
            "tier3_active": False,  # Whether Tier 3 is currently running
            "tier3_format": None,   # "short_form" or "long_form"
            "tier3_phase": None,    # Current phase: "phase1_assessment", "phase2_format", etc.
            # Model configuration (for resume with same settings)
            "model_config": {
                "submitter_model": None,
                "validator_model": None,
                "high_context_model": None,
                "high_param_model": None,
                "submitter_context_window": 131072,
                "validator_context_window": 131072,
                "high_context_context_window": 131072,
                "high_param_context_window": 10000,
                "submitter_max_tokens": 25000,
                "validator_max_tokens": 15000,
                "high_context_max_tokens": 25000,
                "high_param_max_tokens": 15000
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def save_workflow_state(self, state: Dict[str, Any]) -> None:
        """Save workflow state for crash recovery / resume."""
        async with self._lock:
            self._workflow_state = state
            self._workflow_state["last_updated"] = datetime.now().isoformat()
            async with aiofiles.open(self._workflow_state_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self._workflow_state, indent=2))
    
    async def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        if self._workflow_state is None:
            await self._load_workflow_state()
        return self._workflow_state.copy()
    
    async def clear_workflow_state(self) -> None:
        """Clear workflow state (called on clean stop)."""
        async with self._lock:
            self._workflow_state = self._get_default_workflow_state()
            if self._workflow_state_path.exists():
                self._workflow_state_path.unlink()
            logger.info("Workflow state cleared")
    
    def has_interrupted_workflow(self) -> bool:
        """Check if there's an interrupted workflow that can be resumed.
        
        This returns True for BOTH:
        1. Crash recovery: is_running=True (process died unexpectedly)
        2. Pause/resume: is_running=False but current_tier/topic exists (user pressed stop)
        
        The user can resume their session by pressing Start again.
        Only clear_all_data() should prevent resume by clearing the workflow state.
        """
        if self._workflow_state is None:
            return False
        
        # Check for resumable state - either a crash recovery OR a paused session
        has_topic = self._workflow_state.get("current_topic_id") is not None
        has_tier = self._workflow_state.get("current_tier") is not None
        has_papers = self._workflow_state.get("papers_completed_count", 0) > 0
        tier3_active = self._workflow_state.get("tier3_active", False)
        
        # Can resume if there's a topic being worked on OR papers have been completed OR Tier 3 is active
        return has_tier and (has_topic or has_papers or tier3_active)
    
    # ========================================================================
    # ID GENERATION
    # ========================================================================
    
    async def generate_topic_id(self) -> str:
        """Generate a new unique topic ID."""
        async with self._lock:
            topic_id = f"topic_{self._data['next_topic_id']:03d}"
            self._data['next_topic_id'] += 1
            await self._save_metadata()
            return topic_id
    
    async def generate_paper_id(self) -> str:
        """Generate a new unique paper ID."""
        async with self._lock:
            paper_id = f"paper_{self._data['next_paper_id']:03d}"
            self._data['next_paper_id'] += 1
            await self._save_metadata()
            return paper_id
    
    # ========================================================================
    # USER PROMPT
    # ========================================================================
    
    async def get_user_prompt(self) -> str:
        """Get the user's research prompt."""
        await self._ensure_initialized()
        return self._data.get("user_research_prompt", "")

    async def get_base_user_prompt(self) -> str:
        """Get the original user research prompt before proof framing."""
        await self._ensure_initialized()
        return self._data.get("base_user_research_prompt") or self._data.get("user_research_prompt", "")
    
    async def set_user_prompt(self, prompt: str) -> None:
        """Set the user's research prompt."""
        async with self._lock:
            self._data["user_research_prompt"] = prompt
            await self._save_metadata()

    async def set_proof_framing_state(
        self,
        *,
        base_user_prompt: str,
        effective_user_prompt: str,
        active: bool,
        context: str,
        reasoning: str,
    ) -> None:
        """Persist the proof-framing decision in metadata."""
        async with self._lock:
            self._data["base_user_research_prompt"] = base_user_prompt
            self._data["user_research_prompt"] = effective_user_prompt
            self._data["proof_framing_active"] = active
            self._data["proof_framing_context"] = context
            self._data["proof_framing_reasoning"] = reasoning
            await self._save_metadata()

    async def get_proof_runtime_config(self) -> Dict[str, Any]:
        """Return the persisted proof runtime model configuration snapshot."""
        await self._ensure_initialized()
        value = self._data.get("proof_runtime_config")
        return value if isinstance(value, dict) else {}

    async def set_proof_runtime_config(self, config: Dict[str, Any]) -> None:
        """Persist the proof runtime model configuration snapshot."""
        async with self._lock:
            self._data["proof_runtime_config"] = config if isinstance(config, dict) else {}
            await self._save_metadata()
    
    # ========================================================================
    # BRAINSTORM REGISTRATION
    # ========================================================================
    
    async def register_brainstorm(self, metadata: BrainstormMetadata) -> None:
        """Register a new brainstorm in central metadata."""
        async with self._lock:
            # Check if already exists
            existing_ids = [b.get("topic_id") for b in self._data.get("brainstorms", [])]
            if metadata.topic_id in existing_ids:
                # Update existing
                for i, b in enumerate(self._data["brainstorms"]):
                    if b.get("topic_id") == metadata.topic_id:
                        self._data["brainstorms"][i] = self._brainstorm_to_dict(metadata)
                        break
            else:
                # Add new
                self._data.setdefault("brainstorms", []).append(
                    self._brainstorm_to_dict(metadata)
                )
            
            await self._save_metadata()
            
            # Update stats
            self._stats["total_brainstorms_created"] = len(self._data.get("brainstorms", []))
            self._stats["current_brainstorm_id"] = metadata.topic_id
            await self._save_stats()
    
    async def update_brainstorm(self, topic_id: str, **kwargs) -> None:
        """Update brainstorm metadata in central registry."""
        async with self._lock:
            for i, b in enumerate(self._data.get("brainstorms", [])):
                if b.get("topic_id") == topic_id:
                    for key, value in kwargs.items():
                        if isinstance(value, datetime):
                            value = value.isoformat()
                        self._data["brainstorms"][i][key] = value
                    break
            await self._save_metadata()
    
    async def mark_brainstorm_complete(self, topic_id: str) -> None:
        """Mark a brainstorm as complete."""
        await self.update_brainstorm(
            topic_id,
            status="complete",
            completed_at=datetime.now()
        )
        
        # Update stats
        completed_count = sum(
            1 for b in self._data.get("brainstorms", []) 
            if b.get("status") == "complete"
        )
        self._stats["total_brainstorms_completed"] = completed_count
        await self._save_stats()
    
    def _brainstorm_to_dict(self, metadata: BrainstormMetadata) -> Dict[str, Any]:
        """Convert BrainstormMetadata to dictionary for storage."""
        return {
            "topic_id": metadata.topic_id,
            "topic_prompt": metadata.topic_prompt,
            "status": metadata.status,
            "submission_count": metadata.submission_count,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
            "completed_at": metadata.completed_at.isoformat() if metadata.completed_at else None,
            "last_activity": metadata.last_activity.isoformat() if metadata.last_activity else None,
            "papers_generated": metadata.papers_generated
        }
    
    # ========================================================================
    # PAPER REGISTRATION
    # ========================================================================
    
    async def register_paper(self, metadata: PaperMetadata) -> None:
        """Register a new paper in central metadata."""
        async with self._lock:
            # Check if already exists
            existing_ids = [p.get("paper_id") for p in self._data.get("papers", [])]
            if metadata.paper_id in existing_ids:
                # Update existing
                for i, p in enumerate(self._data["papers"]):
                    if p.get("paper_id") == metadata.paper_id:
                        self._data["papers"][i] = self._paper_to_dict(metadata)
                        break
            else:
                # Add new
                self._data.setdefault("papers", []).append(
                    self._paper_to_dict(metadata)
                )
            
            # Also update brainstorm with paper reference
            for source_id in metadata.source_brainstorm_ids:
                for i, b in enumerate(self._data.get("brainstorms", [])):
                    if b.get("topic_id") == source_id:
                        if metadata.paper_id not in b.get("papers_generated", []):
                            self._data["brainstorms"][i].setdefault("papers_generated", []).append(
                                metadata.paper_id
                            )
                        break
            
            await self._save_metadata()
            
            # Update stats
            self._stats["total_papers_completed"] = sum(
                1 for p in self._data.get("papers", [])
                if p.get("status") == "complete"
            )
            self._stats["current_paper_id"] = metadata.paper_id
            await self._save_stats()
    
    async def archive_paper(self, paper_id: str) -> None:
        """Mark a paper as archived in central metadata."""
        async with self._lock:
            for i, p in enumerate(self._data.get("papers", [])):
                if p.get("paper_id") == paper_id:
                    self._data["papers"][i]["status"] = "archived"
                    break
            await self._save_metadata()
            
            # Update stats
            self._stats["total_papers_archived"] = sum(
                1 for p in self._data.get("papers", [])
                if p.get("status") == "archived"
            )
            await self._save_stats()
    
    def _paper_to_dict(self, metadata: PaperMetadata) -> Dict[str, Any]:
        """Convert PaperMetadata to dictionary for storage."""
        return {
            "paper_id": metadata.paper_id,
            "title": metadata.title,
            "abstract": metadata.abstract,
            "word_count": metadata.word_count,
            "source_brainstorm_ids": metadata.source_brainstorm_ids,
            "referenced_papers": metadata.referenced_papers,
            "status": metadata.status,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None
        }
    
    # ========================================================================
    # QUERIES
    # ========================================================================
    
    async def get_all_brainstorms_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all brainstorms for context."""
        await self._ensure_initialized()
        return self._data.get("brainstorms", [])
    
    async def get_all_papers_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all papers for context."""
        await self._ensure_initialized()
        return [
            p for p in self._data.get("papers", [])
            if p.get("status") == "complete"
        ]
    
    async def get_brainstorm_entry(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get brainstorm entry from central metadata."""
        await self._ensure_initialized()
        for b in self._data.get("brainstorms", []):
            if b.get("topic_id") == topic_id:
                return b
        return None
    
    async def get_paper_entry(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get paper entry from central metadata."""
        await self._ensure_initialized()
        for p in self._data.get("papers", []):
            if p.get("paper_id") == paper_id:
                return p
        return None
    
    async def get_papers_by_brainstorm(self, topic_id: str) -> List[Dict[str, Any]]:
        """Get all papers from a specific brainstorm."""
        await self._ensure_initialized()
        return [
            p for p in self._data.get("papers", [])
            if topic_id in p.get("source_brainstorm_ids", [])
            and p.get("status") == "complete"
        ]
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        await self._ensure_initialized()
        return self._stats.copy()
    
    async def increment_stat(self, stat_name: str, amount: int = 1) -> None:
        """Increment a statistic."""
        await self._ensure_initialized()
        async with self._lock:
            if self._stats and stat_name in self._stats:
                self._stats[stat_name] += amount
                
                # Update acceptance rate
                total = self._stats.get("total_submissions_accepted", 0) + self._stats.get("total_submissions_rejected", 0)
                if total > 0:
                    self._stats["acceptance_rate"] = self._stats.get("total_submissions_accepted", 0) / total
                
                await self._save_stats()
    
    async def set_stat(self, stat_name: str, value: Any) -> None:
        """Set a statistic value."""
        await self._ensure_initialized()
        async with self._lock:
            if self._stats:
                self._stats[stat_name] = value
                await self._save_stats()
    
    async def set_current_brainstorm(self, topic_id: Optional[str]) -> None:
        """Set current active brainstorm."""
        await self.set_stat("current_brainstorm_id", topic_id)
    
    async def set_current_paper(self, paper_id: Optional[str]) -> None:
        """Set current paper being written."""
        await self.set_stat("current_paper_id", paper_id)
    
    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================
    
    async def delete_brainstorm(self, topic_id: str) -> bool:
        """
        Remove brainstorm from central metadata.
        
        Args:
            topic_id: Brainstorm topic ID to remove
        
        Returns:
            True if removal successful, False otherwise
        """
        async with self._lock:
            try:
                # Remove from brainstorms list
                brainstorms = self._data.get("brainstorms", [])
                self._data["brainstorms"] = [
                    b for b in brainstorms if b.get("topic_id") != topic_id
                ]
                
                await self._save_metadata()
                
                # Update stats
                self._stats["total_brainstorms_created"] = len(self._data.get("brainstorms", []))
                completed_count = sum(
                    1 for b in self._data.get("brainstorms", []) 
                    if b.get("status") == "complete"
                )
                self._stats["total_brainstorms_completed"] = completed_count
                await self._save_stats()
                
                logger.info(f"Removed brainstorm {topic_id} from central metadata")
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove brainstorm {topic_id} from metadata: {e}")
                return False
    
    async def delete_paper(self, paper_id: str) -> bool:
        """
        Remove paper from central metadata.
        
        Args:
            paper_id: Paper ID to remove
        
        Returns:
            True if removal successful, False otherwise
        """
        async with self._lock:
            try:
                # Get paper info before removal
                paper_entry = None
                for p in self._data.get("papers", []):
                    if p.get("paper_id") == paper_id:
                        paper_entry = p
                        break
                
                # Remove from papers list
                papers = self._data.get("papers", [])
                self._data["papers"] = [
                    p for p in papers if p.get("paper_id") != paper_id
                ]
                
                # Remove paper reference from brainstorms
                if paper_entry:
                    for source_id in paper_entry.get("source_brainstorm_ids", []):
                        for i, b in enumerate(self._data.get("brainstorms", [])):
                            if b.get("topic_id") == source_id:
                                papers_generated = b.get("papers_generated", [])
                                if paper_id in papers_generated:
                                    papers_generated.remove(paper_id)
                                    self._data["brainstorms"][i]["papers_generated"] = papers_generated
                
                await self._save_metadata()
                
                # Update stats
                self._stats["total_papers_completed"] = sum(
                    1 for p in self._data.get("papers", [])
                    if p.get("status") == "complete"
                )
                self._stats["total_papers_archived"] = sum(
                    1 for p in self._data.get("papers", [])
                    if p.get("status") == "archived"
                )
                await self._save_stats()
                
                logger.info(f"Removed paper {paper_id} from central metadata")
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove paper {paper_id} from metadata: {e}")
                return False
    
    # ========================================================================
    # CLEAR / RESET
    # ========================================================================
    
    async def clear_all(self) -> None:
        """Clear all research metadata, statistics, and workflow state."""
        async with self._lock:
            self._data = {
                "user_research_prompt": "",
                "brainstorms": [],
                "papers": [],
                "next_topic_id": 1,
                "next_paper_id": 1,
                "created_at": datetime.now().isoformat()
            }
            self._stats = self._get_default_stats()
            self._workflow_state = self._get_default_workflow_state()
            await self._save_metadata()
            await self._save_stats()
            
            # Clear workflow state file
            if self._workflow_state_path.exists():
                self._workflow_state_path.unlink()
            
            logger.info("Research metadata cleared")


# Global instance
research_metadata = ResearchMetadata()

