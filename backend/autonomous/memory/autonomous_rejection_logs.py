"""
Autonomous Rejection Logs - Rejection tracking for autonomous research mode.
Handles topic selection rejections, completion feedback, and per-brainstorm submitter rejections.
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import aiofiles

from backend.shared.config import system_config
from backend.shared.path_safety import validate_single_path_component

logger = logging.getLogger(__name__)


class AutonomousRejectionLogs:
    """
    Manages rejection logs for autonomous research mode.
    Separate from brainstorm memory to handle cross-topic rejections.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._topic_rejections_path = Path(system_config.auto_research_topic_rejections_file)
        self._brainstorms_dir = Path(system_config.auto_brainstorms_dir)
    
    async def initialize(self) -> None:
        """Initialize rejection log files."""
        self._topic_rejections_path.parent.mkdir(parents=True, exist_ok=True)
        self._brainstorms_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Autonomous rejection logs initialized")

    def _safe_topic_id(self, topic_id: str) -> str:
        """Validate topic_id before using it in rejection-log filenames."""
        safe_topic_id = validate_single_path_component(topic_id, "topic ID")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", safe_topic_id):
            raise ValueError(f"Invalid topic ID: {topic_id}")
        return safe_topic_id
    
    # ========================================================================
    # TOPIC SELECTION REJECTIONS (Global - Last 5)
    # ========================================================================
    
    async def add_topic_selection_rejection(
        self,
        action: str,
        proposed_topic: str,
        rejection_reasoning: str
    ) -> None:
        """Add a topic selection rejection to the global log (max 5)."""
        async with self._lock:
            # Load existing rejections
            rejections = await self._load_topic_rejections()
            
            # Add new rejection
            rejections.append({
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "proposed_topic": proposed_topic[:500],
                "reasoning": rejection_reasoning[:750]
            })
            
            # Keep only last 5
            rejections = rejections[-5:]
            
            # Save
            await self._save_topic_rejections(rejections)
            logger.debug(f"Added topic selection rejection: {action}")
    
    async def get_topic_selection_rejections(self) -> List[Dict[str, Any]]:
        """Get topic selection rejections (last 5)."""
        return await self._load_topic_rejections()
    
    async def clear_topic_selection_rejections(self) -> None:
        """Clear all topic selection rejections."""
        async with self._lock:
            await self._save_topic_rejections([])
    
    async def _load_topic_rejections(self) -> List[Dict[str, Any]]:
        """Load topic selection rejections from file."""
        if not self._topic_rejections_path.exists():
            return []
        
        try:
            async with aiofiles.open(self._topic_rejections_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    return json.loads(content)
                return []
        except Exception as e:
            logger.error(f"Failed to load topic rejections: {e}")
            return []
    
    async def _save_topic_rejections(self, rejections: List[Dict[str, Any]]) -> None:
        """Save topic selection rejections to file."""
        async with aiofiles.open(self._topic_rejections_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(rejections, indent=2))
    
    # ========================================================================
    # FORMAT TOPIC REJECTIONS FOR CONTEXT
    # ========================================================================
    
    async def format_topic_rejections_for_context(self) -> str:
        """Format topic selection rejections for inclusion in prompt context."""
        rejections = await self.get_topic_selection_rejections()
        
        if not rejections:
            return ""
        
        lines = ["PREVIOUS TOPIC SELECTION REJECTIONS (Learn from these):", "=" * 60]
        
        for i, rej in enumerate(rejections, 1):
            lines.append(f"\nRejection #{i} ({rej.get('timestamp', 'Unknown time')})")
            lines.append(f"Action: {rej.get('action', 'Unknown')}")
            lines.append(f"Proposed: {rej.get('proposed_topic', 'N/A')}")
            lines.append(f"Rejection Reason: {rej.get('reasoning', 'N/A')}")
            lines.append("-" * 40)
        
        return "\n".join(lines)
    
    # ========================================================================
    # COMPLETION FEEDBACK (Per-Topic - Last 5)
    # ========================================================================
    
    def _get_completion_feedback_path(self, topic_id: str) -> Path:
        """Get path to completion feedback file for a topic."""
        return self._brainstorms_dir / f"completion_feedback_{self._safe_topic_id(topic_id)}.txt"
    
    async def add_completion_feedback(
        self,
        topic_id: str,
        decision: str,
        reasoning: str,
        suggested_additions: str = ""
    ) -> None:
        """Add completion review feedback for a topic (max 5)."""
        async with self._lock:
            feedback_path = self._get_completion_feedback_path(topic_id)
            
            # Load existing feedback
            feedbacks = []
            if feedback_path.exists():
                try:
                    async with aiofiles.open(feedback_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        if content.strip():
                            feedbacks = json.loads(content)
                except Exception:
                    feedbacks = []
            
            # Add new feedback
            feedbacks.append({
                "timestamp": datetime.now().isoformat(),
                "decision": decision,
                "reasoning": reasoning,
                "suggested_additions": suggested_additions
            })
            
            # Keep only last 5
            feedbacks = feedbacks[-5:]
            
            # Save
            async with aiofiles.open(feedback_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(feedbacks, indent=2))
            
            logger.debug(f"Added completion feedback for topic {topic_id}")
    
    async def get_completion_feedback(self, topic_id: str) -> List[Dict[str, Any]]:
        """Get completion feedback for a topic (last 5)."""
        feedback_path = self._get_completion_feedback_path(topic_id)
        
        if not feedback_path.exists():
            return []
        
        try:
            async with aiofiles.open(feedback_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    return json.loads(content)
                return []
        except Exception as e:
            logger.error(f"Failed to load completion feedback for {topic_id}: {e}")
            return []
    
    async def format_completion_feedback_for_context(self, topic_id: str) -> str:
        """Format completion feedback for inclusion in prompt context."""
        feedbacks = await self.get_completion_feedback(topic_id)
        
        if not feedbacks:
            return ""
        
        lines = ["PREVIOUS COMPLETION REVIEW FEEDBACK (Learn from these):", "=" * 60]
        
        for i, fb in enumerate(feedbacks, 1):
            lines.append(f"\nReview #{i} ({fb.get('timestamp', 'Unknown time')})")
            lines.append(f"Decision: {fb.get('decision', 'Unknown')}")
            lines.append(f"Reasoning: {fb.get('reasoning', 'N/A')}")
            if fb.get('suggested_additions'):
                lines.append(f"Suggested Additions: {fb.get('suggested_additions')}")
            lines.append("-" * 40)
        
        return "\n".join(lines)
    
    # ========================================================================
    # BRAINSTORM SUBMITTER REJECTIONS (Per-Topic, Per-Submitter - Last 5)
    # ========================================================================
    
    def _get_submitter_rejections_path(self, topic_id: str, submitter_id: int) -> Path:
        """Get path to submitter rejection log file."""
        return self._brainstorms_dir / (
            f"brainstorm_{self._safe_topic_id(topic_id)}_submitter_{submitter_id}_rejections.txt"
        )
    
    async def add_brainstorm_submitter_rejection(
        self,
        topic_id: str,
        submitter_id: int,
        rejection_summary: str,
        submission_preview: str
    ) -> None:
        """Add a rejection to submitter's local rejection log (max 5)."""
        async with self._lock:
            rejections_path = self._get_submitter_rejections_path(topic_id, submitter_id)
            
            # Load existing rejections
            rejections = []
            if rejections_path.exists():
                try:
                    async with aiofiles.open(rejections_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        if content.strip():
                            rejections = json.loads(content)
                except Exception:
                    rejections = []
            
            # Add new rejection
            rejections.append({
                "timestamp": datetime.now().isoformat(),
                "summary": rejection_summary[:750],
                "submission_preview": submission_preview[:750]
            })
            
            # Keep only last 5
            rejections = rejections[-5:]
            
            # Save
            async with aiofiles.open(rejections_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(rejections, indent=2))
            
            logger.debug(f"Added submitter {submitter_id} rejection for topic {topic_id}")
    
    async def get_brainstorm_submitter_rejections(
        self, 
        topic_id: str, 
        submitter_id: int
    ) -> List[Dict[str, Any]]:
        """Get rejection log for a submitter (last 5)."""
        rejections_path = self._get_submitter_rejections_path(topic_id, submitter_id)
        
        if not rejections_path.exists():
            return []
        
        try:
            async with aiofiles.open(rejections_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    return json.loads(content)
                return []
        except Exception as e:
            logger.error(f"Failed to read submitter rejections: {e}")
            return []
    
    async def clear_brainstorm_submitter_rejections(
        self, 
        topic_id: str, 
        submitter_id: int
    ) -> None:
        """Clear rejection log for a submitter."""
        rejections_path = self._get_submitter_rejections_path(topic_id, submitter_id)
        
        async with self._lock:
            async with aiofiles.open(rejections_path, 'w', encoding='utf-8') as f:
                await f.write("[]")
    
    async def format_submitter_rejections_for_context(
        self, 
        topic_id: str, 
        submitter_id: int
    ) -> str:
        """Format submitter rejections for inclusion in prompt context."""
        rejections = await self.get_brainstorm_submitter_rejections(topic_id, submitter_id)
        
        if not rejections:
            return ""
        
        lines = [f"YOUR LAST {len(rejections)} REJECTIONS (Learn from these):", "=" * 60]
        
        for i, rej in enumerate(rejections, 1):
            lines.append(f"\nRejection #{i} ({rej.get('timestamp', 'Unknown time')})")
            lines.append(f"Submission Preview: {rej.get('submission_preview', 'N/A')}")
            lines.append(f"Rejection Reason: {rej.get('summary', 'N/A')}")
            lines.append("-" * 40)
        
        return "\n".join(lines)
    
    # ========================================================================
    # CLEAR ALL FOR TOPIC
    # ========================================================================
    
    async def clear_all_for_topic(self, topic_id: str) -> None:
        """Clear all rejection logs for a topic."""
        async with self._lock:
            # Clear completion feedback
            feedback_path = self._get_completion_feedback_path(topic_id)
            if feedback_path.exists():
                feedback_path.unlink()
            
            # Clear submitter rejections for all 10 possible submitters
            for submitter_id in range(1, 11):
                rejections_path = self._get_submitter_rejections_path(topic_id, submitter_id)
                if rejections_path.exists():
                    rejections_path.unlink()
            
            logger.info(f"Cleared all rejection logs for topic {topic_id}")
    
    # ========================================================================
    # CLEAR ALL
    # ========================================================================
    
    async def clear_all(self) -> None:
        """Clear ALL autonomous rejection logs (topic selection + all topics)."""
        async with self._lock:
            # Clear global topic selection rejections
            if self._topic_rejections_path.exists():
                self._topic_rejections_path.unlink()
                logger.info(f"Cleared topic selection rejections: {self._topic_rejections_path}")
            
            # Clear all brainstorm-related rejection files
            if self._brainstorms_dir.exists():
                # Clear completion feedback files
                for path in self._brainstorms_dir.glob("completion_feedback_*.txt"):
                    try:
                        path.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete {path}: {e}")
                
                # Clear submitter rejection files
                for path in self._brainstorms_dir.glob("brainstorm_*_submitter_*_rejections.txt"):
                    try:
                        path.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete {path}: {e}")
            
            logger.info("Cleared all autonomous rejection logs")


# Global instance
autonomous_rejection_logs = AutonomousRejectionLogs()

