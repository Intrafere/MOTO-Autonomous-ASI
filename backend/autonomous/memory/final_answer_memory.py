"""
Final Answer Memory - State management for Tier 3 final answer generation.
Handles volume organization, chapter tracking, and Tier 3-specific rejection logs.

IMPORTANT: Tier 3 operates ONLY on Tier 2 papers, NOT on Tier 1 brainstorm databases.
This ensures the final answer is synthesized from validated, complete research papers.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import aiofiles

from backend.shared.config import system_config
from backend.shared.models import (
    FinalAnswerState,
    CertaintyAssessment,
    AnswerFormatSelection,
    VolumeOrganization,
    VolumeChapter,
    ModelUsageTracker,
    ModelUsageEntry
)

logger = logging.getLogger(__name__)


class FinalAnswerMemory:
    """
    Manages Tier 3 final answer state and persistence.
    
    Key responsibilities:
    - Store and manage FinalAnswerState
    - Track volume organization (long form)
    - Maintain Tier 3-specific rejection logs (independent from Tiers 1/2)
    - Handle crash recovery via state persistence
    
    Supports both:
    - Legacy mode: Uses system_config.data_dir / "auto_final_answer"
    - Session mode: Uses session_manager.get_final_answer_dir()
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._base_dir = Path(system_config.data_dir) / "auto_final_answer"
        self._state_path = self._base_dir / "final_answer_state.json"
        self._volume_path = self._base_dir / "volume_organization.json"
        self._rejections_path = self._base_dir / "tier3_rejections.txt"
        self._final_volume_path = self._base_dir / "final_volume.txt"
        
        # In-memory state
        self._state: Optional[FinalAnswerState] = None
        self._session_manager = None
    
    def set_session_manager(self, session_manager) -> None:
        """Set session manager for session-based path resolution."""
        self._session_manager = session_manager
        if session_manager and session_manager.is_session_active:
            self._base_dir = session_manager.get_final_answer_dir()
            self._state_path = self._base_dir / "final_answer_state.json"
            self._volume_path = self._base_dir / "volume_organization.json"
            self._rejections_path = self._base_dir / "tier3_rejections.txt"
            self._final_volume_path = self._base_dir / "final_volume.txt"
            logger.info(f"Final answer memory using session path: {self._base_dir}")
    
    async def initialize(self) -> None:
        """Initialize the final answer memory directories and load state."""
        # If session manager is active, use its path
        if self._session_manager and self._session_manager.is_session_active:
            self._base_dir = self._session_manager.get_final_answer_dir()
            self._state_path = self._base_dir / "final_answer_state.json"
            self._volume_path = self._base_dir / "volume_organization.json"
            self._rejections_path = self._base_dir / "tier3_rejections.txt"
            self._final_volume_path = self._base_dir / "final_volume.txt"
        
        self._base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing state if available
        if self._state_path.exists():
            await self._load_state()
        else:
            self._state = FinalAnswerState()
        
        logger.info(f"Final answer memory initialized at {self._base_dir}")
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    async def _load_state(self) -> None:
        """Load state from file."""
        try:
            async with aiofiles.open(self._state_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                
                # Handle nested objects
                if data.get("certainty_assessment"):
                    data["certainty_assessment"] = CertaintyAssessment(**data["certainty_assessment"])
                if data.get("volume_organization"):
                    vol_data = data["volume_organization"]
                    if vol_data.get("chapters"):
                        vol_data["chapters"] = [VolumeChapter(**ch) for ch in vol_data["chapters"]]
                    data["volume_organization"] = VolumeOrganization(**vol_data)
                
                # Handle model_usage (with nested ModelUsageEntry objects)
                if data.get("model_usage"):
                    usage_data = data["model_usage"]
                    if usage_data.get("models"):
                        # Convert each model entry dict to ModelUsageEntry
                        for model_id, entry_data in usage_data["models"].items():
                            usage_data["models"][model_id] = ModelUsageEntry(**entry_data)
                    data["model_usage"] = ModelUsageTracker(**usage_data)
                
                self._state = FinalAnswerState(**data)
                logger.info(f"Loaded Tier 3 state: status={self._state.status}")
        except Exception as e:
            logger.error(f"Failed to load final answer state: {e}")
            self._state = FinalAnswerState()
    
    async def _save_state(self) -> None:
        """Save state to file."""
        try:
            data = self._state.model_dump()
            
            # Handle datetime serialization
            def serialize_datetimes(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetimes(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetimes(item) for item in obj]
                return obj
            
            data = serialize_datetimes(data)
            
            async with aiofiles.open(self._state_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save final answer state: {e}")
    
    def get_state(self) -> FinalAnswerState:
        """Get current state."""
        if self._state is None:
            self._state = FinalAnswerState()
        return self._state
    
    async def update_state(self, **kwargs) -> None:
        """Update state with given values."""
        async with self._lock:
            if self._state is None:
                self._state = FinalAnswerState()
            
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            
            self._state.timestamp = datetime.now()
            await self._save_state()
    
    async def set_active(self, is_active: bool) -> None:
        """Set whether Tier 3 is active."""
        await self.update_state(is_active=is_active)
        if is_active:
            logger.info("Tier 3 Final Answer activated")
        else:
            logger.info("Tier 3 Final Answer deactivated")
    
    async def set_status(self, status: str) -> None:
        """Update the Tier 3 status."""
        await self.update_state(status=status)
        logger.info(f"Tier 3 status: {status}")
    
    # ========================================================================
    # CERTAINTY ASSESSMENT
    # ========================================================================
    
    async def save_certainty_assessment(self, assessment: CertaintyAssessment) -> None:
        """Save certainty assessment from Phase 1."""
        async with self._lock:
            self._state.certainty_assessment = assessment
            self._state.timestamp = datetime.now()
            await self._save_state()
            logger.info(f"Certainty assessment saved: {assessment.certainty_level}")
    
    def get_certainty_assessment(self) -> Optional[CertaintyAssessment]:
        """Get the certainty assessment."""
        if self._state:
            return self._state.certainty_assessment
        return None
    
    # ========================================================================
    # ANSWER FORMAT
    # ========================================================================
    
    async def set_answer_format(self, format_selection: AnswerFormatSelection) -> None:
        """Set the selected answer format from Phase 2."""
        async with self._lock:
            self._state.answer_format = format_selection.answer_format
            self._state.timestamp = datetime.now()
            await self._save_state()
            logger.info(f"Answer format selected: {format_selection.answer_format}")
    
    def get_answer_format(self) -> Optional[str]:
        """Get the answer format."""
        if self._state:
            return self._state.answer_format
        return None
    
    # ========================================================================
    # SHORT FORM TRACKING
    # ========================================================================
    
    async def set_short_form_paper_id(self, paper_id: str) -> None:
        """Set the paper ID for short form answer."""
        await self.update_state(short_form_paper_id=paper_id)
        logger.info(f"Short form paper ID: {paper_id}")
    
    async def set_short_form_references(self, paper_ids: List[str]) -> None:
        """Set reference papers for short form answer."""
        await self.update_state(short_form_reference_papers=paper_ids)
        logger.info(f"Short form reference papers: {paper_ids}")
    
    # ========================================================================
    # VOLUME ORGANIZATION (LONG FORM)
    # ========================================================================
    
    async def save_volume_organization(self, volume: VolumeOrganization) -> None:
        """Save volume organization."""
        async with self._lock:
            self._state.volume_organization = volume
            self._state.timestamp = datetime.now()
            await self._save_state()
            
            # Also save to dedicated volume file
            try:
                vol_data = volume.model_dump()
                for key, value in vol_data.items():
                    if isinstance(value, datetime):
                        vol_data[key] = value.isoformat()
                for ch in vol_data.get("chapters", []):
                    for k, v in ch.items():
                        if isinstance(v, datetime):
                            ch[k] = v.isoformat()
                
                async with aiofiles.open(self._volume_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(vol_data, indent=2))
            except Exception as e:
                logger.error(f"Failed to save volume organization file: {e}")
            
            logger.info(f"Volume organization saved: {volume.volume_title} ({len(volume.chapters)} chapters)")
    
    def get_volume_organization(self) -> Optional[VolumeOrganization]:
        """Get the volume organization."""
        if self._state:
            return self._state.volume_organization
        return None
    
    async def update_chapter_status(self, chapter_order: int, status: str, paper_id: str = None) -> None:
        """Update status of a specific chapter."""
        async with self._lock:
            if self._state and self._state.volume_organization:
                for chapter in self._state.volume_organization.chapters:
                    if chapter.order == chapter_order:
                        chapter.status = status
                        if paper_id:
                            chapter.paper_id = paper_id
                        break
                
                # Track completed chapters
                if status == "complete" and chapter_order not in self._state.completed_chapters:
                    self._state.completed_chapters.append(chapter_order)
                
                await self._save_state()
                logger.info(f"Chapter {chapter_order} status updated: {status}")
    
    async def set_current_writing_chapter(self, chapter_order: Optional[int]) -> None:
        """Set the chapter currently being written."""
        await self.update_state(current_writing_chapter=chapter_order)
        if chapter_order:
            logger.info(f"Now writing chapter {chapter_order}")
    
    def get_next_chapter_to_write(self) -> Optional[VolumeChapter]:
        """
        Get the next chapter that needs to be written.
        Writing order:
        1. Gap papers (body chapters) in order
        2. Conclusion paper
        3. Introduction paper
        """
        if not self._state or not self._state.volume_organization:
            return None
        
        chapters = self._state.volume_organization.chapters
        
        # First, write any pending gap papers in order
        gap_papers = sorted(
            [ch for ch in chapters if ch.chapter_type == "gap_paper" and ch.status == "pending"],
            key=lambda x: x.order
        )
        if gap_papers:
            return gap_papers[0]
        
        # Then conclusion
        conclusion = next(
            (ch for ch in chapters if ch.chapter_type == "conclusion" and ch.status == "pending"),
            None
        )
        if conclusion:
            return conclusion
        
        # Finally introduction
        introduction = next(
            (ch for ch in chapters if ch.chapter_type == "introduction" and ch.status == "pending"),
            None
        )
        if introduction:
            return introduction
        
        return None
    
    def is_volume_complete(self) -> bool:
        """Check if all chapters are complete."""
        if not self._state or not self._state.volume_organization:
            return False
        
        return all(
            ch.status == "complete" 
            for ch in self._state.volume_organization.chapters
        )
    
    # ========================================================================
    # REJECTION LOG (TIER 3 SPECIFIC - INDEPENDENT FROM TIERS 1/2)
    # ========================================================================
    
    async def add_rejection(
        self,
        phase: str,
        rejection_summary: str,
        submission_preview: str
    ) -> None:
        """
        Add a rejection to the Tier 3 rejection log.
        
        Args:
            phase: The phase that was rejected (assessment, format, volume, writing)
            rejection_summary: Validator's summary (max 750 chars)
            submission_preview: Preview of rejected submission (max 500 chars)
        """
        async with self._lock:
            # Update rejection counter
            if phase == "assessment":
                self._state.tier3_assessment_rejections += 1
            elif phase == "format":
                self._state.tier3_format_rejections += 1
            elif phase == "volume":
                self._state.tier3_volume_rejections += 1
            elif phase == "writing":
                self._state.tier3_writing_rejections += 1
            
            await self._save_state()
            
            # Append to rejection log file
            try:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "phase": phase,
                    "summary": rejection_summary[:750],
                    "submission_preview": submission_preview[:500]
                }
                
                # Load existing rejections
                rejections = []
                if self._rejections_path.exists():
                    async with aiofiles.open(self._rejections_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        if content.strip():
                            try:
                                rejections = json.loads(content)
                            except json.JSONDecodeError:
                                rejections = []
                
                # Add new and keep last 10
                rejections.append(entry)
                rejections = rejections[-10:]
                
                async with aiofiles.open(self._rejections_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(rejections, indent=2))
                
                logger.info(f"Tier 3 rejection logged: phase={phase}")
            except Exception as e:
                logger.error(f"Failed to write Tier 3 rejection log: {e}")
    
    async def get_rejections(self, phase: str = None) -> List[Dict[str, Any]]:
        """
        Get Tier 3 rejections, optionally filtered by phase.
        
        Args:
            phase: Optional filter for specific phase
        
        Returns:
            List of rejection entries
        """
        if not self._rejections_path.exists():
            return []
        
        try:
            async with aiofiles.open(self._rejections_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if not content.strip():
                    return []
                
                rejections = json.loads(content)
                
                if phase:
                    rejections = [r for r in rejections if r.get("phase") == phase]
                
                return rejections
        except Exception as e:
            logger.error(f"Failed to read Tier 3 rejection log: {e}")
            return []
    
    def format_rejection_context(self, phase: str) -> str:
        """
        Format rejections for a specific phase as context for the next attempt.
        Follows the same enhanced format as other tiers.
        """
        # This is a synchronous wrapper - call get_rejections from async context
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use sync wrapper in async context
                return ""
        except RuntimeError:
            pass
        return ""
    
    async def get_rejection_context_async(self, phase: str) -> str:
        """Format rejections for prompt context."""
        rejections = await self.get_rejections(phase)
        if not rejections:
            return ""
        
        # Keep last 5 for context
        recent = rejections[-5:]
        
        lines = [f"PREVIOUS TIER 3 {phase.upper()} REJECTIONS (learn from these):\n"]
        for i, r in enumerate(recent, 1):
            lines.append(f"\n--- Rejection {i} ---")
            lines.append(f"🚫 REJECTED BECAUSE: {r.get('summary', 'Unknown')}")
            lines.append(f"Submission preview: {r.get('submission_preview', '')[:300]}...")
        
        return "\n".join(lines)
    
    # ========================================================================
    # MODEL USAGE TRACKING (FOR AUTHOR ATTRIBUTION AND CREDITS)
    # ========================================================================
    
    async def initialize_model_tracking(self, user_prompt: str) -> None:
        """
        Initialize model usage tracking for a new Tier 3 session.
        Called when Tier 3 starts.
        
        Args:
            user_prompt: The user's original research prompt for attribution
        """
        async with self._lock:
            if self._state is None:
                self._state = FinalAnswerState()
            
            self._state.model_usage = ModelUsageTracker(
                models={},
                user_prompt=user_prompt,
                generation_date=datetime.now(),
                total_api_calls=0
            )
            await self._save_state()
            logger.info("Model usage tracking initialized for Tier 3")
    
    async def track_model_call(self, model_id: str) -> None:
        """
        Track an API call for a model.
        Same model used in multiple instances counts as ONE author,
        but all API calls are still tallied.
        
        Args:
            model_id: The model identifier (e.g., "deepseek-r1:70b")
        """
        async with self._lock:
            if self._state is None or self._state.model_usage is None:
                # Model tracking not initialized - likely a mid-generation volume
                # Gracefully skip tracking
                logger.debug(f"Model tracking not initialized, skipping call tracking for {model_id}")
                return
            
            # Track the call using the tracker's built-in method
            self._state.model_usage.track_call(model_id)
            await self._save_state()
            logger.debug(f"Tracked API call for model {model_id} (total: {self._state.model_usage.total_api_calls})")
    
    def get_author_attribution_text(self) -> str:
        """
        Generate the author attribution section text for the beginning of the volume.
        
        Format:
        ================================================================================
        AUTONOMOUS AI SOLUTION

        This was an autonomous AI solution generated with the MOTO harness.

        User's Research Prompt: [prompt text here]

        AI Model Authors: model_name_1, model_name_2, model_name_3

        Generated: 2025-12-31
        ================================================================================
        
        Returns:
            Formatted author attribution text, or empty string if tracking not available
        """
        if self._state is None or self._state.model_usage is None:
            return ""
        
        tracker = self._state.model_usage
        
        # Get unique authors (model IDs without call counts)
        authors = tracker.get_unique_authors()
        if not authors:
            return ""
        
        # Format the author list
        author_list = ", ".join(authors)
        
        # Format the date
        gen_date = tracker.generation_date.strftime("%Y-%m-%d")
        
        # Truncate prompt for attribution header to prevent embedding entire uploaded papers.
        # The full prompt is preserved in session_metadata.json for reference.
        MAX_PROMPT_LENGTH = 500
        display_prompt = tracker.user_prompt
        if len(display_prompt) > MAX_PROMPT_LENGTH:
            display_prompt = display_prompt[:MAX_PROMPT_LENGTH].rstrip() + "... [truncated]"
        
        # Build the attribution section
        lines = [
            "=" * 80,
            "AUTONOMOUS AI SOLUTION",
            "",
            "Disclaimer: This is an autonomous AI solution generated with the MOTO harness. This paper was not peer reviewed and was autonomously generated without user oversight or interaction beyond the original user prompt, therefore, this text may contain errors. These papers often contain ambitious content and/or extraordinary claims, all content should be viewed with extreme scrutiny.",
            "",
            f"User's Research Prompt: {display_prompt}",
            "",
            f"AI Model Authors: {author_list}",
            "",
            f"Generated: {gen_date}",
            "=" * 80,
            ""
        ]
        
        return "\n".join(lines)
    
    def get_model_credits_text(self) -> str:
        """
        Generate the model credits section text for the end of the volume.
        
        Format:
        ================================================================================
        MODEL CREDITS

        This autonomous solution attempt was generated with the Intrafere LLC AI Harness, 
        MOTO, and the following model(s):

        - deepseek-r1:70b (127 API calls)
        - qwen-2.5:32b (43 API calls)
        - llama-3.1:70b (18 API calls)

        Total API Calls: 188
        ================================================================================
        
        Returns:
            Formatted model credits text, or empty string if tracking not available
        """
        if self._state is None or self._state.model_usage is None:
            return ""
        
        tracker = self._state.model_usage
        
        # Get models sorted by usage (descending)
        models_by_usage = tracker.get_models_by_usage()
        if not models_by_usage:
            return ""
        
        # Build the credits section
        lines = [
            "",
            "=" * 80,
            "MODEL CREDITS",
            "",
            "This autonomous solution attempt was generated with the Intrafere LLC AI Harness,",
            "MOTO, and the following model(s):",
            ""
        ]
        
        # Add each model with its call count
        for entry in models_by_usage:
            lines.append(f"- {entry.model_id} ({entry.api_call_count} API calls)")
        
        lines.extend([
            "",
            f"Total API Calls: {tracker.total_api_calls}",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    # ========================================================================
    # CHAPTER PAPER FILES
    # ========================================================================
    
    def _get_chapter_paper_path(self, chapter_order: int) -> Path:
        """Get path for chapter paper file."""
        return self._base_dir / f"chapter_{chapter_order:02d}_paper.txt"
    
    def _get_chapter_outline_path(self, chapter_order: int) -> Path:
        """Get path for chapter outline file."""
        return self._base_dir / f"chapter_{chapter_order:02d}_outline.txt"
    
    def _get_source_papers_dir(self) -> Path:
        """Get path to source papers archive directory."""
        return self._base_dir / "source_papers"
    
    def _get_source_brainstorms_dir(self) -> Path:
        """Get path to source brainstorms archive directory."""
        return self._base_dir / "source_brainstorms"
    
    async def save_chapter_paper(
        self,
        chapter_order: int,
        content: str,
        outline: str = ""
    ) -> None:
        """Save a chapter paper (for gap/intro/conclusion papers)."""
        async with self._lock:
            # Save paper content
            paper_path = self._get_chapter_paper_path(chapter_order)
            async with aiofiles.open(paper_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            # Save outline if provided
            if outline:
                outline_path = self._get_chapter_outline_path(chapter_order)
                async with aiofiles.open(outline_path, 'w', encoding='utf-8') as f:
                    await f.write(outline)
            
            logger.info(f"Saved chapter {chapter_order} paper ({len(content)} chars)")
    
    async def get_chapter_paper(self, chapter_order: int) -> str:
        """Get chapter paper content."""
        paper_path = self._get_chapter_paper_path(chapter_order)
        if not paper_path.exists():
            return ""
        
        try:
            async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read chapter {chapter_order} paper: {e}")
            return ""
    
    # ========================================================================
    # RESEARCH LINEAGE ARCHIVAL (PAPERS AND BRAINSTORMS)
    # ========================================================================
    
    async def _archive_paper(self, paper_id: str) -> bool:
        """
        Copy a paper and all its files to the source_papers archive.
        
        Args:
            paper_id: Paper ID to archive
        
        Returns:
            True if successful, False otherwise
        """
        from backend.autonomous.memory.paper_library import paper_library
        
        try:
            source_papers_dir = self._get_source_papers_dir()
            source_papers_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy paper content
            content = await paper_library.get_paper_content(paper_id)
            if content:
                paper_path = source_papers_dir / f"paper_{paper_id}.txt"
                async with aiofiles.open(paper_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            
            # Copy abstract
            abstract = await paper_library.get_abstract(paper_id)
            if abstract:
                abstract_path = source_papers_dir / f"paper_{paper_id}_abstract.txt"
                async with aiofiles.open(abstract_path, 'w', encoding='utf-8') as f:
                    await f.write(abstract)
            
            # Copy outline
            outline = await paper_library.get_outline(paper_id)
            if outline:
                outline_path = source_papers_dir / f"paper_{paper_id}_outline.txt"
                async with aiofiles.open(outline_path, 'w', encoding='utf-8') as f:
                    await f.write(outline)
            
            # Copy metadata
            metadata = await paper_library.get_metadata(paper_id)
            if metadata:
                metadata_data = metadata.model_dump()
                for key, value in metadata_data.items():
                    if isinstance(value, datetime):
                        metadata_data[key] = value.isoformat()
                
                metadata_path = source_papers_dir / f"paper_{paper_id}_metadata.json"
                async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(metadata_data, indent=2))
            
            logger.info(f"Archived paper {paper_id} to final answer source_papers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive paper {paper_id}: {e}")
            return False
    
    async def _archive_brainstorm(self, topic_id: str) -> bool:
        """
        Copy a brainstorm and its metadata to the source_brainstorms archive.
        
        Args:
            topic_id: Brainstorm topic ID to archive
        
        Returns:
            True if successful, False otherwise
        """
        from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
        
        try:
            source_brainstorms_dir = self._get_source_brainstorms_dir()
            source_brainstorms_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy brainstorm database
            content = await brainstorm_memory.get_database_content(topic_id)
            if content:
                db_path = source_brainstorms_dir / f"brainstorm_{topic_id}.txt"
                async with aiofiles.open(db_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            
            # Copy metadata
            metadata = await brainstorm_memory.get_metadata(topic_id)
            if metadata:
                metadata_data = metadata.model_dump()
                for key, value in metadata_data.items():
                    if isinstance(value, datetime):
                        metadata_data[key] = value.isoformat()
                
                metadata_path = source_brainstorms_dir / f"brainstorm_{topic_id}_metadata.json"
                async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(metadata_data, indent=2))
            
            logger.info(f"Archived brainstorm {topic_id} to final answer source_brainstorms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive brainstorm {topic_id}: {e}")
            return False
    
    async def _archive_all_referenced_resources(
        self,
        paper_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Archive all papers and their source brainstorms for the final answer.
        
        Args:
            paper_ids: List of paper IDs to archive
        
        Returns:
            Dictionary with archive statistics
        """
        from backend.autonomous.memory.paper_library import paper_library
        
        archived_papers = []
        archived_brainstorms = []
        brainstorm_ids_seen = set()
        
        logger.info(f"Starting archival of {len(paper_ids)} papers and their source brainstorms")
        
        # Archive each paper
        for paper_id in paper_ids:
            success = await self._archive_paper(paper_id)
            if success:
                archived_papers.append(paper_id)
                
                # Get paper metadata to find source brainstorms
                metadata = await paper_library.get_metadata(paper_id)
                if metadata and metadata.source_brainstorm_ids:
                    for brainstorm_id in metadata.source_brainstorm_ids:
                        if brainstorm_id not in brainstorm_ids_seen:
                            brainstorm_ids_seen.add(brainstorm_id)
        
        # Archive each unique brainstorm
        for brainstorm_id in brainstorm_ids_seen:
            success = await self._archive_brainstorm(brainstorm_id)
            if success:
                archived_brainstorms.append(brainstorm_id)
        
        logger.info(
            f"Archival complete: {len(archived_papers)} papers, "
            f"{len(archived_brainstorms)} brainstorms"
        )
        
        return {
            "papers_archived": len(archived_papers),
            "brainstorms_archived": len(archived_brainstorms),
            "paper_ids": archived_papers,
            "brainstorm_ids": archived_brainstorms
        }
    
    async def get_archived_papers_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all archived papers with metadata.
        
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        source_papers_dir = self._get_source_papers_dir()
        
        if not source_papers_dir.exists():
            return papers
        
        for metadata_path in source_papers_dir.glob("paper_*_metadata.json"):
            try:
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    papers.append(data)
            except Exception as e:
                logger.error(f"Failed to read archived paper metadata: {e}")
        
        # Sort by title
        papers.sort(key=lambda x: x.get('title', ''))
        return papers
    
    async def get_archived_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full archived paper content.
        
        Args:
            paper_id: Paper ID
        
        Returns:
            Dictionary with paper content, abstract, outline, metadata
        """
        source_papers_dir = self._get_source_papers_dir()
        
        try:
            # Read content
            paper_path = source_papers_dir / f"paper_{paper_id}.txt"
            if not paper_path.exists():
                return None
            
            async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Read abstract
            abstract_path = source_papers_dir / f"paper_{paper_id}_abstract.txt"
            abstract = ""
            if abstract_path.exists():
                async with aiofiles.open(abstract_path, 'r', encoding='utf-8') as f:
                    abstract = await f.read()
            
            # Read outline
            outline_path = source_papers_dir / f"paper_{paper_id}_outline.txt"
            outline = ""
            if outline_path.exists():
                async with aiofiles.open(outline_path, 'r', encoding='utf-8') as f:
                    outline = await f.read()
            
            # Read metadata
            metadata_path = source_papers_dir / f"paper_{paper_id}_metadata.json"
            metadata = {}
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
            
            return {
                "paper_id": paper_id,
                "content": content,
                "abstract": abstract,
                "outline": outline,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Failed to read archived paper {paper_id}: {e}")
            return None
    
    async def get_archived_brainstorms_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all archived brainstorms with metadata.
        
        Returns:
            List of brainstorm metadata dictionaries
        """
        brainstorms = []
        source_brainstorms_dir = self._get_source_brainstorms_dir()
        
        if not source_brainstorms_dir.exists():
            return brainstorms
        
        for metadata_path in source_brainstorms_dir.glob("brainstorm_*_metadata.json"):
            try:
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    brainstorms.append(data)
            except Exception as e:
                logger.error(f"Failed to read archived brainstorm metadata: {e}")
        
        # Sort by topic_id
        brainstorms.sort(key=lambda x: x.get('topic_id', ''))
        return brainstorms
    
    async def get_archived_brainstorm(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full archived brainstorm content.
        
        Args:
            topic_id: Brainstorm topic ID
        
        Returns:
            Dictionary with brainstorm content and metadata
        """
        source_brainstorms_dir = self._get_source_brainstorms_dir()
        
        try:
            # Read database content
            db_path = source_brainstorms_dir / f"brainstorm_{topic_id}.txt"
            if not db_path.exists():
                return None
            
            async with aiofiles.open(db_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Read metadata
            metadata_path = source_brainstorms_dir / f"brainstorm_{topic_id}_metadata.json"
            metadata = {}
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
            
            return {
                "topic_id": topic_id,
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Failed to read archived brainstorm {topic_id}: {e}")
            return None

    
    # ========================================================================
    # FINAL VOLUME ASSEMBLY
    # ========================================================================
    
    async def assemble_final_volume(self) -> str:
        """
        Assemble the final volume from all chapters.
        Returns the complete volume text.
        
        Includes:
        - Author attribution section at the beginning (if model tracking available)
        - Volume content (title, TOC, chapters)
        - Model credits section at the end (if model tracking available)
        - **Archives all referenced papers and their source brainstorms**
        """
        if not self._state or not self._state.volume_organization:
            return ""
        
        volume = self._state.volume_organization
        
        # ====================================================================
        # ARCHIVE ALL REFERENCED PAPERS AND BRAINSTORMS
        # ====================================================================
        # Extract all paper IDs from existing_paper chapters
        paper_ids_to_archive = []
        for chapter in volume.chapters:
            if chapter.chapter_type == "existing_paper" and chapter.paper_id:
                paper_ids_to_archive.append(chapter.paper_id)
        
        if paper_ids_to_archive:
            logger.info(f"Archiving {len(paper_ids_to_archive)} papers and their source brainstorms")
            archive_stats = await self._archive_all_referenced_resources(paper_ids_to_archive)
            logger.info(f"Archive stats: {archive_stats}")
        
        parts = []
        
        # ====================================================================
        # AUTHOR ATTRIBUTION (at the very beginning)
        # ====================================================================
        author_attribution = self.get_author_attribution_text()
        if author_attribution:
            parts.append(author_attribution)
            parts.append("")
        
        # ====================================================================
        # VOLUME HEADER AND METADATA
        # ====================================================================
        parts.append("=" * 80)
        parts.append(f"FINAL ANSWER VOLUME")
        parts.append(f"Title: {volume.volume_title}")
        parts.append(f"Generated: {datetime.now().isoformat()}")
        parts.append("=" * 80)
        parts.append("")
        
        # Table of contents
        parts.append("TABLE OF CONTENTS")
        parts.append("-" * 40)
        for chapter in sorted(volume.chapters, key=lambda x: x.order):
            chapter_type_label = chapter.chapter_type.replace("_", " ").title()
            parts.append(f"  {chapter.order}. {chapter.title} [{chapter_type_label}]")
        parts.append("")
        parts.append("=" * 80)
        parts.append("")
        
        # ====================================================================
        # CHAPTERS
        # ====================================================================
        for chapter in sorted(volume.chapters, key=lambda x: x.order):
            parts.append(f"{'#' * 80}")
            parts.append(f"# CHAPTER {chapter.order}: {chapter.title}")
            parts.append(f"# Type: {chapter.chapter_type.replace('_', ' ').title()}")
            parts.append(f"{'#' * 80}")
            parts.append("")
            
            if chapter.chapter_type == "existing_paper" and chapter.paper_id:
                # Load existing paper from paper library
                from backend.autonomous.memory.paper_library import paper_library
                content = await paper_library.get_paper_content(chapter.paper_id)
                parts.append(content)
            else:
                # Load from chapter file
                content = await self.get_chapter_paper(chapter.order)
                parts.append(content)
            
            parts.append("")
            parts.append("")
        
        # ====================================================================
        # VOLUME FOOTER
        # ====================================================================
        parts.append("=" * 80)
        parts.append("END OF VOLUME")
        parts.append("=" * 80)
        
        # ====================================================================
        # MODEL CREDITS (at the very end)
        # ====================================================================
        model_credits = self.get_model_credits_text()
        if model_credits:
            parts.append(model_credits)
        
        final_text = "\n".join(parts)
        
        # Save to file
        try:
            async with aiofiles.open(self._final_volume_path, 'w', encoding='utf-8') as f:
                await f.write(final_text)
            logger.info(f"Final volume assembled: {len(final_text)} chars")
        except Exception as e:
            logger.error(f"Failed to save final volume: {e}")
        
        return final_text
    
    async def get_final_volume(self) -> str:
        """Get the assembled final volume."""
        if not self._final_volume_path.exists():
            return ""
        
        try:
            async with aiofiles.open(self._final_volume_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read final volume: {e}")
            return ""
    
    async def assemble_short_form_paper(self, paper_content: str, paper_title: str) -> str:
        """
        Assemble a short-form final answer paper with author attribution and credits.
        
        Args:
            paper_content: The raw paper content from the compiler
            paper_title: The paper title
        
        Returns:
            Complete paper text with author attribution at beginning and credits at end
            **Also archives all referenced papers and their source brainstorms**
        """
        # ====================================================================
        # ARCHIVE ALL REFERENCED PAPERS AND BRAINSTORMS
        # ====================================================================
        if self._state and self._state.short_form_reference_papers:
            paper_ids = self._state.short_form_reference_papers
            logger.info(f"Archiving {len(paper_ids)} reference papers and their source brainstorms")
            archive_stats = await self._archive_all_referenced_resources(paper_ids)
            logger.info(f"Archive stats: {archive_stats}")
        
        parts = []
        
        # ====================================================================
        # AUTHOR ATTRIBUTION (at the very beginning)
        # ====================================================================
        author_attribution = self.get_author_attribution_text()
        if author_attribution:
            parts.append(author_attribution)
            parts.append("")
        
        # ====================================================================
        # PAPER HEADER
        # ====================================================================
        parts.append("=" * 80)
        parts.append("FINAL ANSWER")
        parts.append(f"Title: {paper_title}")
        parts.append(f"Generated: {datetime.now().isoformat()}")
        parts.append("=" * 80)
        parts.append("")
        
        # ====================================================================
        # PAPER CONTENT
        # ====================================================================
        parts.append(paper_content)
        
        # ====================================================================
        # PAPER FOOTER
        # ====================================================================
        parts.append("")
        parts.append("=" * 80)
        parts.append("END OF PAPER")
        parts.append("=" * 80)
        
        # ====================================================================
        # MODEL CREDITS (at the very end)
        # ====================================================================
        model_credits = self.get_model_credits_text()
        if model_credits:
            parts.append(model_credits)
        
        final_text = "\n".join(parts)
        
        # Save to file
        try:
            short_form_path = self._base_dir / "final_short_form_paper.txt"
            async with aiofiles.open(short_form_path, 'w', encoding='utf-8') as f:
                await f.write(final_text)
            logger.info(f"Short-form final answer assembled: {len(final_text)} chars")
        except Exception as e:
            logger.error(f"Failed to save short-form final answer: {e}")
        
        return final_text
    
    async def get_short_form_paper(self) -> str:
        """Get the assembled short-form final answer paper."""
        short_form_path = self._base_dir / "final_short_form_paper.txt"
        if not short_form_path.exists():
            return ""
        
        try:
            async with aiofiles.open(short_form_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read short-form final answer: {e}")
            return ""
    
    # ========================================================================
    # CLEAR / RESET
    # ========================================================================
    
    async def clear(self) -> None:
        """Clear all Tier 3 state and files."""
        async with self._lock:
            self._state = FinalAnswerState()
            
            # Remove all files in the directory
            if self._base_dir.exists():
                import shutil
                try:
                    shutil.rmtree(self._base_dir)
                    self._base_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to clear final answer directory: {e}")
            
            logger.info("Tier 3 final answer memory cleared")
    
    async def reset_for_new_trigger(self) -> None:
        """
        Reset state for a new Tier 3 trigger while preserving statistics.
        Called when Tier 3 needs to re-evaluate after more papers are written.
        """
        async with self._lock:
            # Preserve statistics
            stats = {
                "tier3_assessment_rejections": self._state.tier3_assessment_rejections if self._state else 0,
                "tier3_format_rejections": self._state.tier3_format_rejections if self._state else 0,
                "tier3_volume_rejections": self._state.tier3_volume_rejections if self._state else 0,
                "tier3_writing_rejections": self._state.tier3_writing_rejections if self._state else 0,
            }
            
            # Reset state
            self._state = FinalAnswerState(**stats)
            await self._save_state()
            
            logger.info("Tier 3 state reset for new trigger")
    
    async def reset_title_selection(self) -> None:
        """Reset paper title selection, clearing selected title and restarting title selection phase."""
        async with self._lock:
            if self._state:
                self._state.paper_title = None
                self._state.tier3_phase = "title_selection"
                await self._save_state()
                logger.info("Tier 3 title selection reset")
    
    async def reset_current_chapter(self, chapter_index: int) -> None:
        """
        Reset a specific chapter in volume organization.
        Marks chapter as pending and clears any associated paper file.
        
        Args:
            chapter_index: Index of chapter to reset
        """
        async with self._lock:
            if self._state and self._state.volume_organization:
                chapters = self._state.volume_organization.chapters
                if 0 <= chapter_index < len(chapters):
                    chapters[chapter_index].status = "pending"
                    chapters[chapter_index].paper_content = None
                    await self._save_state()
                    
                    # Delete chapter paper file if exists
                    chapter_file = self._base_dir / f"chapter_{chapter_index}_paper.txt"
                    if chapter_file.exists():
                        chapter_file.unlink()
                        logger.info(f"Deleted chapter {chapter_index} paper file")
                    
                    # Delete chapter outline file if exists
                    outline_file = self._base_dir / f"chapter_{chapter_index}_outline.txt"
                    if outline_file.exists():
                        outline_file.unlink()
                        logger.info(f"Deleted chapter {chapter_index} outline file")
                    
                    logger.info(f"Tier 3 chapter {chapter_index} reset to pending")
    
    # ========================================================================
    # FINAL ANSWER LIBRARY - Browse all completed volumes/papers
    # ========================================================================
    
    async def list_all_final_answers(self) -> List[Dict[str, Any]]:
        """
        List all completed final answers from all sessions (legacy + session-based).
        
        Returns a list of dictionaries with metadata about each final answer:
        - answer_id: Unique identifier
        - format: "short_form" or "long_form"
        - title: Volume/paper title
        - user_prompt: Research question
        - certainty_level: Assessment result
        - word_count: Total words
        - chapter_count: Number of chapters (long form only)
        - completion_date: When it was completed
        - location: Path to the answer
        - session_id: Session identifier (or "legacy" for old format)
        """
        final_answers = []
        
        # ====================================================================
        # LEGACY LOCATION (auto_final_answer/)
        # ====================================================================
        legacy_dir = Path(system_config.data_dir) / "auto_final_answer"
        if legacy_dir.exists():
            legacy_state_path = legacy_dir / "final_answer_state.json"
            if legacy_state_path.exists():
                try:
                    async with aiofiles.open(legacy_state_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        state_data = json.loads(content)
                    
                    # Only include if status is "complete"
                    if state_data.get("status") == "complete":
                        answer_format = state_data.get("answer_format", "unknown")
                        
                        # Get content and calculate word count
                        if answer_format == "long_form":
                            volume_path = legacy_dir / "final_volume.txt"
                            if volume_path.exists():
                                async with aiofiles.open(volume_path, 'r', encoding='utf-8') as f:
                                    content = await f.read()
                                    word_count = len(content.split())
                            else:
                                word_count = 0
                            
                            volume_org = state_data.get("volume_organization", {})
                            title = volume_org.get("volume_title", "Untitled Volume")
                            chapter_count = len(volume_org.get("chapters", []))
                        else:
                            paper_path = legacy_dir / "final_short_form_paper.txt"
                            if paper_path.exists():
                                async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                                    content = await f.read()
                                    word_count = len(content.split())
                            else:
                                word_count = 0
                            
                            title = state_data.get("short_form_title", "Untitled Paper")
                            chapter_count = 0
                        
                        # Get user prompt from model_usage or certainty assessment
                        model_usage = state_data.get("model_usage", {})
                        user_prompt = model_usage.get("user_prompt", "")
                        if not user_prompt:
                            cert_assess = state_data.get("certainty_assessment", {})
                            user_prompt = cert_assess.get("user_prompt", "Unknown research question")
                        
                        certainty_level = state_data.get("certainty_assessment", {}).get("certainty_level", "unknown")
                        completion_date = state_data.get("timestamp", datetime.now().isoformat())
                        
                        final_answers.append({
                            "answer_id": "legacy",
                            "format": answer_format,
                            "title": title,
                            "user_prompt": user_prompt,
                            "certainty_level": certainty_level,
                            "word_count": word_count,
                            "chapter_count": chapter_count,
                            "completion_date": completion_date,
                            "location": str(legacy_dir),
                            "session_id": "legacy"
                        })
                except Exception as e:
                    logger.error(f"Failed to read legacy final answer: {e}")
        
        # ====================================================================
        # SESSION-BASED LOCATIONS (auto_sessions/)
        # ====================================================================
        sessions_dir = Path(system_config.data_dir) / "auto_sessions"
        if sessions_dir.exists():
            for session_folder in sessions_dir.iterdir():
                if not session_folder.is_dir():
                    continue
                
                final_answer_dir = session_folder / "final_answer"
                if not final_answer_dir.exists():
                    continue
                
                state_path = final_answer_dir / "final_answer_state.json"
                if not state_path.exists():
                    continue
                
                try:
                    async with aiofiles.open(state_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        state_data = json.loads(content)
                    
                    # Only include if status is "complete"
                    if state_data.get("status") == "complete":
                        answer_format = state_data.get("answer_format", "unknown")
                        
                        # Get content and calculate word count
                        if answer_format == "long_form":
                            volume_path = final_answer_dir / "final_volume.txt"
                            if volume_path.exists():
                                async with aiofiles.open(volume_path, 'r', encoding='utf-8') as f:
                                    content = await f.read()
                                    word_count = len(content.split())
                            else:
                                word_count = 0
                            
                            volume_org = state_data.get("volume_organization", {})
                            title = volume_org.get("volume_title", "Untitled Volume")
                            chapter_count = len(volume_org.get("chapters", []))
                        else:
                            paper_path = final_answer_dir / "final_short_form_paper.txt"
                            if paper_path.exists():
                                async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                                    content = await f.read()
                                    word_count = len(content.split())
                            else:
                                word_count = 0
                            
                            title = state_data.get("short_form_title", "Untitled Paper")
                            chapter_count = 0
                        
                        # Get user prompt
                        model_usage = state_data.get("model_usage", {})
                        user_prompt = model_usage.get("user_prompt", "")
                        if not user_prompt:
                            cert_assess = state_data.get("certainty_assessment", {})
                            user_prompt = cert_assess.get("user_prompt", "Unknown research question")
                        
                        certainty_level = state_data.get("certainty_assessment", {}).get("certainty_level", "unknown")
                        completion_date = state_data.get("timestamp", datetime.now().isoformat())
                        
                        final_answers.append({
                            "answer_id": session_folder.name,
                            "format": answer_format,
                            "title": title,
                            "user_prompt": user_prompt,
                            "certainty_level": certainty_level,
                            "word_count": word_count,
                            "chapter_count": chapter_count,
                            "completion_date": completion_date,
                            "location": str(final_answer_dir),
                            "session_id": session_folder.name
                        })
                except Exception as e:
                    logger.error(f"Failed to read final answer from {session_folder.name}: {e}")
        
        # Sort by completion date (newest first)
        final_answers.sort(key=lambda x: x["completion_date"], reverse=True)
        
        return final_answers
    
    async def get_final_answer_by_id(self, answer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full content of a specific final answer by its ID.
        
        Args:
            answer_id: Either "legacy" or a session folder name
        
        Returns:
            Dictionary with:
            - metadata: Same as list_all_final_answers entry
            - content: Full text of volume/paper
            - chapters: List of chapter details (long form only)
        """
        # Determine location
        if answer_id == "legacy":
            base_dir = Path(system_config.data_dir) / "auto_final_answer"
        else:
            base_dir = Path(system_config.data_dir) / "auto_sessions" / answer_id / "final_answer"
        
        if not base_dir.exists():
            return None
        
        state_path = base_dir / "final_answer_state.json"
        if not state_path.exists():
            return None
        
        try:
            # Load state
            async with aiofiles.open(state_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                state_data = json.loads(content)
            
            if state_data.get("status") != "complete":
                return None
            
            answer_format = state_data.get("answer_format", "unknown")
            
            # Get metadata
            if answer_format == "long_form":
                volume_org = state_data.get("volume_organization", {})
                title = volume_org.get("volume_title", "Untitled Volume")
                chapters = volume_org.get("chapters", [])
                
                # Get content
                volume_path = base_dir / "final_volume.txt"
                if volume_path.exists():
                    async with aiofiles.open(volume_path, 'r', encoding='utf-8') as f:
                        full_content = await f.read()
                else:
                    full_content = ""
            else:
                title = state_data.get("short_form_title", "Untitled Paper")
                chapters = []
                
                # Get content
                paper_path = base_dir / "final_short_form_paper.txt"
                if paper_path.exists():
                    async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                        full_content = await f.read()
                else:
                    full_content = ""
            
            model_usage = state_data.get("model_usage", {})
            user_prompt = model_usage.get("user_prompt", "")
            if not user_prompt:
                cert_assess = state_data.get("certainty_assessment", {})
                user_prompt = cert_assess.get("user_prompt", "Unknown research question")
            
            certainty_level = state_data.get("certainty_assessment", {}).get("certainty_level", "unknown")
            completion_date = state_data.get("timestamp", datetime.now().isoformat())
            
            return {
                "metadata": {
                    "answer_id": answer_id,
                    "format": answer_format,
                    "title": title,
                    "user_prompt": user_prompt,
                    "certainty_level": certainty_level,
                    "word_count": len(full_content.split()),
                    "chapter_count": len(chapters),
                    "completion_date": completion_date,
                    "location": str(base_dir),
                    "session_id": answer_id
                },
                "content": full_content,
                "chapters": chapters
            }
        except Exception as e:
            logger.error(f"Failed to get final answer {answer_id}: {e}")
            return None


# Global instance
final_answer_memory = FinalAnswerMemory()

