"""
Paper Library - Paper storage and archive management.
Handles file I/O for completed papers, abstracts, and source brainstorm caching.
"""
import asyncio
import json
import logging
import shutil
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import aiofiles

from backend.shared.config import system_config
from backend.shared.models import PaperMetadata

logger = logging.getLogger(__name__)


class PaperLibrary:
    """
    Manages completed papers in Tier 2.
    Handles paper storage, abstract extraction, and archiving.
    
    Supports both:
    - Legacy mode: Uses system_config.auto_papers_dir
    - Session mode: Uses session_manager.get_papers_dir()
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._base_dir = Path(system_config.auto_papers_dir)
        self._archive_dir = Path(system_config.auto_papers_archive_dir)
        self._session_manager = None
    
    def set_session_manager(self, session_manager) -> None:
        """Set session manager for session-based path resolution."""
        self._session_manager = session_manager
        if session_manager and session_manager.is_session_active:
            self._base_dir = session_manager.get_papers_dir()
            self._archive_dir = session_manager.get_papers_dir() / "archive"
            logger.info(f"Paper library using session path: {self._base_dir}")
    
    async def initialize(self) -> None:
        """Initialize the paper library directories."""
        # If session manager is active, use its path
        if self._session_manager and self._session_manager.is_session_active:
            self._base_dir = self._session_manager.get_papers_dir()
            self._archive_dir = self._base_dir / "archive"
        
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Paper library initialized at {self._base_dir}")
    
    def _get_paper_path(self, paper_id: str) -> Path:
        """Get path to paper file."""
        return self._base_dir / f"paper_{paper_id}.txt"
    
    def get_paper_path(self, paper_id: str) -> str:
        """
        Public method to get path to paper file.
        Uses session-aware path resolution.
        
        Returns:
            str: Absolute path to the paper file
        """
        return str(self._get_paper_path(paper_id))
    
    def _get_abstract_path(self, paper_id: str) -> Path:
        """Get path to abstract file."""
        return self._base_dir / f"paper_{paper_id}_abstract.txt"
    
    def _get_source_brainstorm_path(self, paper_id: str) -> Path:
        """Get path to cached source brainstorm file."""
        return self._base_dir / f"paper_{paper_id}_source_brainstorm.txt"
    
    def _get_outline_path(self, paper_id: str) -> Path:
        """Get path to paper outline file."""
        return self._base_dir / f"paper_{paper_id}_outline.txt"
    
    def _get_metadata_path(self, paper_id: str) -> Path:
        """Get path to paper metadata JSON file."""
        return self._base_dir / f"paper_{paper_id}_metadata.json"
    
    def _get_rejections_path(self, paper_id: str) -> Path:
        """Get path to paper compiler rejections file."""
        return self._base_dir / f"paper_{paper_id}_last_10_rejections.txt"
    
    # ========================================================================
    # CONTENT VALIDATION
    # ========================================================================
    
    async def _is_paper_complete(self, paper_id: str) -> bool:
        """
        Validate that a paper has all required sections (not just placeholders).
        
        Checks for:
        - Abstract section (actual content, not placeholder)
        - Introduction section (actual content, not placeholder)
        - Body content
        - Conclusion section (actual content, not placeholder)
        
        Returns:
            bool: True if paper has all required sections, False otherwise
        """
        paper_path = self._get_paper_path(paper_id)
        if not paper_path.exists():
            return False
        
        try:
            async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Check for placeholder markers (incomplete paper)
            placeholder_markers = [
                "[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION",
                "[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION",
                "[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION"
            ]
            
            for marker in placeholder_markers:
                if marker in content:
                    logger.debug(f"Paper {paper_id} incomplete: Contains placeholder {marker}")
                    return False
            
            # Check for abstract section
            abstract_patterns = [
                r"##\s*Abstract",
                r"#\s*Abstract",
                r"\*\*Abstract\*\*",
                r"^Abstract\s*$"  # Abstract on its own line
            ]
            
            has_abstract = False
            for pattern in abstract_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    has_abstract = True
                    break
            
            if not has_abstract:
                logger.debug(f"Paper {paper_id} incomplete: No abstract section found")
                return False
            
            # Check for introduction section
            intro_patterns = [
                r"##\s*Introduction",
                r"#\s*Introduction",
                r"\*\*Introduction\*\*",
                r"^I\.\s*Introduction",
                r"^Introduction\s*$"
            ]
            
            has_intro = False
            for pattern in intro_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    has_intro = True
                    break
            
            if not has_intro:
                logger.debug(f"Paper {paper_id} incomplete: No introduction section found")
                return False
            
            # Check for conclusion section
            conclusion_patterns = [
                r"##\s*Conclusion",
                r"#\s*Conclusion",
                r"\*\*Conclusion\*\*",
                r"^\w+\.\s*Conclusion",  # e.g., "V. Conclusion"
                r"^Conclusion\s*$"
            ]
            
            has_conclusion = False
            for pattern in conclusion_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    has_conclusion = True
                    break
            
            if not has_conclusion:
                logger.debug(f"Paper {paper_id} incomplete: No conclusion section found")
                return False
            
            # Check for body content (between intro and conclusion)
            # Simple check: paper must be > 1000 chars (excluding placeholders)
            if len(content) < 1000:
                logger.debug(f"Paper {paper_id} incomplete: Content too short ({len(content)} chars)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate paper {paper_id}: {e}")
            return False
    
    # ========================================================================
    # PAPER OPERATIONS
    # ========================================================================
    
    async def save_paper(
        self,
        paper_id: str,
        title: str,
        content: str,
        outline: str,
        abstract: str,
        source_brainstorm_ids: List[str],
        source_brainstorm_content: str,
        referenced_papers: List[str] = None,
        model_usage: Dict[str, int] = None,
        generation_date: datetime = None,
        status: str = "complete",
        wolfram_calls: int = None
    ) -> PaperMetadata:
        """
        Save a paper with all associated files.
        
        Args:
            paper_id: Unique paper identifier
            title: Paper title
            content: Full paper content
            outline: Paper outline
            abstract: Paper abstract
            source_brainstorm_ids: IDs of source brainstorms
            source_brainstorm_content: Full content of source brainstorm(s)
            referenced_papers: IDs of papers used as references
            model_usage: Dict mapping model_id -> API call count (per-paper tracking)
            generation_date: When the paper was generated
            status: Paper status ("complete" or "in_progress", default "complete")
        
        Returns:
            PaperMetadata for the saved paper
        """
        async with self._lock:
            # Count words in paper
            word_count = len(content.split())
            
            # Create metadata
            metadata = PaperMetadata(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                word_count=word_count,
                source_brainstorm_ids=source_brainstorm_ids,
                referenced_papers=referenced_papers or [],
                status=status,  # Use provided status (default "complete")
                created_at=datetime.now(),
                model_usage=model_usage,
                generation_date=generation_date or datetime.now(),
                wolfram_calls=wolfram_calls
            )
            
            # Save paper content
            paper_path = self._get_paper_path(paper_id)
            async with aiofiles.open(paper_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Paper saved: {paper_path}")
            
            # Save outline
            outline_path = self._get_outline_path(paper_id)
            async with aiofiles.open(outline_path, 'w', encoding='utf-8') as f:
                await f.write(outline)
            logger.info(f"Outline saved: {outline_path}")
            
            # Save abstract
            abstract_path = self._get_abstract_path(paper_id)
            async with aiofiles.open(abstract_path, 'w', encoding='utf-8') as f:
                await f.write(abstract)
            logger.info(f"Abstract saved: {abstract_path}")
            
            # Save source brainstorm cache
            source_path = self._get_source_brainstorm_path(paper_id)
            async with aiofiles.open(source_path, 'w', encoding='utf-8') as f:
                await f.write(f"# Source Brainstorm(s) for Paper: {paper_id}\n")
                await f.write(f"# Title: {title}\n")
                await f.write(f"# Source Topic IDs: {', '.join(source_brainstorm_ids)}\n")
                await f.write(f"# Cached: {datetime.now().isoformat()}\n")
                await f.write("=" * 80 + "\n\n")
                await f.write(source_brainstorm_content)
            
            # Save metadata
            await self._save_metadata(metadata)
            
            model_count = len(model_usage) if model_usage else 0
            logger.info(f"Saved paper {paper_id}: '{title}' ({word_count} words, {model_count} models tracked)")
            return metadata
    
    async def get_paper_content(self, paper_id: str) -> str:
        """Get full paper content."""
        paper_path = self._get_paper_path(paper_id)
        
        if not paper_path.exists():
            return ""
        
        try:
            async with aiofiles.open(paper_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read paper {paper_id}: {e}")
            return ""
    
    async def get_abstract(self, paper_id: str) -> str:
        """Get paper abstract."""
        abstract_path = self._get_abstract_path(paper_id)
        
        if not abstract_path.exists():
            return ""
        
        try:
            async with aiofiles.open(abstract_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read abstract for {paper_id}: {e}")
            return ""
    
    async def get_outline(self, paper_id: str) -> str:
        """Get paper outline."""
        outline_path = self._get_outline_path(paper_id)
        
        if not outline_path.exists():
            return ""
        
        try:
            async with aiofiles.open(outline_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read outline for {paper_id}: {e}")
            return ""
    
    async def get_source_brainstorm(self, paper_id: str) -> str:
        """Get cached source brainstorm content."""
        source_path = self._get_source_brainstorm_path(paper_id)
        
        if not source_path.exists():
            return ""
        
        try:
            async with aiofiles.open(source_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read source brainstorm for {paper_id}: {e}")
            return ""
    
    async def _save_metadata(self, metadata: PaperMetadata) -> None:
        """Save paper metadata to JSON file."""
        metadata_path = self._get_metadata_path(metadata.paper_id)
        
        try:
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata.dict(), indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.paper_id}: {e}")
    
    async def get_metadata(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get paper metadata."""
        metadata_path = self._get_metadata_path(paper_id)
        
        if not metadata_path.exists():
            return None
        
        try:
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                return PaperMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load metadata for {paper_id}: {e}")
            return None
    
    async def get_all_papers(self, include_archived: bool = False, include_in_progress: bool = False, validate_completeness: bool = True) -> List[PaperMetadata]:
        """
        Get metadata for all papers.
        
        Args:
            include_archived: If True, include archived papers
            include_in_progress: If True, include papers with status="in_progress" (default False)
            validate_completeness: If True, only return papers with all required sections (default True)
        
        Returns:
            List of PaperMetadata for papers matching criteria
        """
        papers = []
        
        if not self._base_dir.exists():
            return papers
        
        # Get active papers
        for path in self._base_dir.glob("paper_*_metadata.json"):
            try:
                async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    metadata = PaperMetadata(**data)
                    
                    # Filter by archive status
                    if metadata.status == "archived" and not include_archived:
                        continue
                    
                    # Filter by in_progress status
                    if metadata.status == "in_progress" and not include_in_progress:
                        logger.debug(f"Skipping in_progress paper {metadata.paper_id}")
                        continue
                    
                    # Validate completeness if requested
                    if validate_completeness:
                        is_complete = await self._is_paper_complete(metadata.paper_id)
                        if not is_complete:
                            logger.debug(f"Skipping incomplete paper {metadata.paper_id} (has placeholders or missing sections)")
                            continue
                    
                    papers.append(metadata)
            except Exception as e:
                logger.error(f"Failed to load paper metadata from {path}: {e}")
        
        # Sort by creation time (most recent first)
        papers.sort(key=lambda x: x.created_at, reverse=True)
        
        return papers
    
    async def get_papers_by_brainstorm(self, topic_id: str) -> List[PaperMetadata]:
        """Get all complete papers from a specific brainstorm."""
        all_papers = await self.get_all_papers(validate_completeness=True)
        return [p for p in all_papers if topic_id in p.source_brainstorm_ids]
    
    async def get_most_recent_incomplete_paper(self) -> Optional[PaperMetadata]:
        """
        Find the most recent paper that is incomplete (has placeholders or missing sections).
        
        Used for resume logic - when a paper was saved mid-construction and needs to be resumed.
        
        Returns:
            PaperMetadata for the most recent incomplete paper, or None if no incomplete papers exist
        """
        if not self._base_dir.exists():
            return None
        
        incomplete_papers = []
        
        for path in self._base_dir.glob("paper_*_metadata.json"):
            try:
                async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    metadata = PaperMetadata(**data)
                    
                    # Skip archived papers
                    if metadata.status == "archived":
                        continue
                    
                    # Check if paper is incomplete
                    is_complete = await self._is_paper_complete(metadata.paper_id)
                    if not is_complete:
                        incomplete_papers.append(metadata)
                        logger.debug(f"Found incomplete paper: {metadata.paper_id}")
            except Exception as e:
                logger.error(f"Failed to check paper completeness from {path}: {e}")
        
        if not incomplete_papers:
            return None
        
        # Sort by creation time (most recent first) and return the most recent
        incomplete_papers.sort(key=lambda x: x.created_at, reverse=True)
        return incomplete_papers[0]
    
    async def is_paper_complete(self, paper_id: str) -> bool:
        """
        Public method to check if a paper is complete (has all required sections, no placeholders).
        
        Args:
            paper_id: The paper ID to check
            
        Returns:
            True if paper is complete, False if incomplete or doesn't exist
        """
        return await self._is_paper_complete(paper_id)
    
    # ========================================================================
    # ARCHIVE OPERATIONS
    # ========================================================================
    
    async def archive_paper(self, paper_id: str) -> bool:
        """
        Archive a paper (move to archive directory).
        Used when paper is marked as redundant.
        """
        async with self._lock:
            try:
                # Get metadata
                metadata = await self.get_metadata(paper_id)
                if metadata is None:
                    logger.error(f"Cannot archive paper {paper_id}: metadata not found")
                    return False
                
                # Update status
                metadata.status = "archived"
                await self._save_metadata(metadata)
                
                # Move files to archive directory
                files_to_move = [
                    (self._get_paper_path(paper_id), self._archive_dir / f"paper_{paper_id}.txt"),
                    (self._get_abstract_path(paper_id), self._archive_dir / f"paper_{paper_id}_abstract.txt"),
                    (self._get_outline_path(paper_id), self._archive_dir / f"paper_{paper_id}_outline.txt"),
                    (self._get_source_brainstorm_path(paper_id), self._archive_dir / f"paper_{paper_id}_source_brainstorm.txt"),
                    (self._get_metadata_path(paper_id), self._archive_dir / f"paper_{paper_id}_metadata.json"),
                    (self._get_rejections_path(paper_id), self._archive_dir / f"paper_{paper_id}_last_10_rejections.txt")
                ]
                
                for source, dest in files_to_move:
                    if source.exists():
                        shutil.move(str(source), str(dest))
                
                logger.info(f"Paper {paper_id} archived successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to archive paper {paper_id}: {e}")
                return False
    
    async def get_papers_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all papers for topic selection context.
        Returns minimal metadata without full content.
        
        Returns:
            List of dicts with paper_id, title, abstract, outline, word_count, source_brainstorm_ids, created_at
        """
        return await self.get_all_papers_with_outlines()
    
    async def get_all_papers_with_outlines(self) -> List[Dict[str, Any]]:
        """
        Get all complete papers with their outlines included.
        Used for Tier 3 reference selection.
        
        Returns:
            List of dicts with paper_id, title, abstract, outline, word_count, source_brainstorm_ids
        """
        papers = await self.get_all_papers(validate_completeness=True)
        
        summaries = []
        for paper in papers:
            # Fetch outline for this paper
            outline = await self.get_outline(paper.paper_id)
            
            summaries.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "outline": outline,  # NEW: Include outline
                "word_count": paper.word_count,
                "source_brainstorm_ids": paper.source_brainstorm_ids,
                "created_at": paper.created_at.isoformat() if paper.created_at else None
            })
        
        return summaries
    
    async def count_papers(self) -> Dict[str, int]:
        """Count total, archived, in_progress, and active (complete) papers."""
        all_papers = await self.get_all_papers(include_archived=True, include_in_progress=True, validate_completeness=False)
        
        total = len(all_papers)
        archived = sum(1 for p in all_papers if p.status == "archived")
        in_progress = sum(1 for p in all_papers if p.status == "in_progress")
        active = total - archived - in_progress  # Only "complete" papers are active
        
        return {
            "total": total,
            "active": active,
            "in_progress": in_progress,
            "archived": archived
        }
    
    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================
    
    async def delete_paper(self, paper_id: str) -> bool:
        """
        Permanently delete a paper and all associated files.
        
        Args:
            paper_id: The paper ID to delete
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        async with self._lock:
            try:
                # Check if paper exists in active directory
                paper_path = self._get_paper_path(paper_id)
                abstract_path = self._get_abstract_path(paper_id)
                outline_path = self._get_outline_path(paper_id)
                source_path = self._get_source_brainstorm_path(paper_id)
                metadata_path = self._get_metadata_path(paper_id)
                rejections_path = self._get_rejections_path(paper_id)
                
                deleted_any = False
                
                # Delete from active directory
                for path in [paper_path, abstract_path, outline_path, source_path, metadata_path, rejections_path]:
                    if path.exists():
                        path.unlink()
                        deleted_any = True
                        logger.debug(f"Deleted: {path}")
                
                # Also check archive directory
                archive_files = [
                    self._archive_dir / f"paper_{paper_id}.txt",
                    self._archive_dir / f"paper_{paper_id}_abstract.txt",
                    self._archive_dir / f"paper_{paper_id}_outline.txt",
                    self._archive_dir / f"paper_{paper_id}_source_brainstorm.txt",
                    self._archive_dir / f"paper_{paper_id}_metadata.json",
                    self._archive_dir / f"paper_{paper_id}_last_10_rejections.txt"
                ]
                
                for path in archive_files:
                    if path.exists():
                        path.unlink()
                        deleted_any = True
                        logger.debug(f"Deleted from archive: {path}")
                
                if deleted_any:
                    logger.info(f"Paper {paper_id} deleted successfully")
                    return True
                else:
                    logger.warning(f"Paper {paper_id} not found in active or archive directories")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to delete paper {paper_id}: {e}")
                return False


# Global singleton instance
paper_library = PaperLibrary()
