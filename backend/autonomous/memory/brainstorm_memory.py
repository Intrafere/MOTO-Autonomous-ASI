"""
Brainstorm Memory - Per-brainstorm database management.
Handles file I/O for brainstorm databases and metadata.
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import aiofiles

from backend.shared.config import system_config
from backend.shared.models import BrainstormMetadata
from backend.shared.path_safety import validate_single_path_component

logger = logging.getLogger(__name__)


class BrainstormMemory:
    """
    Manages per-brainstorm databases and metadata.
    Each brainstorm has its own database file and metadata JSON.
    
    Supports both:
    - Legacy mode: Uses system_config.auto_brainstorms_dir
    - Session mode: Uses session_manager.get_brainstorms_dir()
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._base_dir = Path(system_config.auto_brainstorms_dir)
        self._session_manager = None
        
    def set_session_manager(self, session_manager) -> None:
        """Set session manager for session-based path resolution."""
        self._session_manager = session_manager
        if session_manager and session_manager.is_session_active:
            self._base_dir = session_manager.get_brainstorms_dir()
            logger.info(f"Brainstorm memory using session path: {self._base_dir}")
        
    async def initialize(self) -> None:
        """Initialize the brainstorm memory directory."""
        # If session manager is active, use its path
        if self._session_manager and self._session_manager.is_session_active:
            self._base_dir = self._session_manager.get_brainstorms_dir()
        
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Brainstorm memory initialized at {self._base_dir}")
    
    def _safe_topic_id(self, topic_id: str) -> str:
        """Validate topic_id as a single path component."""
        return validate_single_path_component(topic_id, "topic ID")

    def _get_database_path(self, topic_id: str) -> Path:
        """Get path to brainstorm database file."""
        return self._base_dir / f"brainstorm_{self._safe_topic_id(topic_id)}.txt"
    
    def get_database_path(self, topic_id: str) -> str:
        """
        Public method to get path to brainstorm database file.
        Uses session-aware path resolution.
        
        Returns:
            str: Absolute path to the brainstorm database file
        """
        return str(self._get_database_path(topic_id))
    
    def _get_metadata_path(self, topic_id: str) -> Path:
        """Get path to brainstorm metadata JSON file."""
        return self._base_dir / f"brainstorm_{self._safe_topic_id(topic_id)}_metadata.json"
    
    def _get_submitter_rejections_path(self, topic_id: str, submitter_id: int) -> Path:
        """Get path to submitter rejection log file."""
        return self._base_dir / f"brainstorm_{self._safe_topic_id(topic_id)}_submitter_{submitter_id}_rejections.txt"
    
    def _get_completion_feedback_path(self, topic_id: str) -> Path:
        """Get path to completion feedback file."""
        return self._base_dir / f"completion_feedback_{self._safe_topic_id(topic_id)}.txt"
    
    # ========================================================================
    # METADATA OPERATIONS
    # ========================================================================
    
    async def create_brainstorm(self, topic_id: str, topic_prompt: str) -> BrainstormMetadata:
        """Create a new brainstorm topic."""
        async with self._lock:
            metadata = BrainstormMetadata(
                topic_id=topic_id,
                topic_prompt=topic_prompt,
                status="in_progress",
                submission_count=0,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                papers_generated=[]
            )
            
            # Save metadata
            await self._save_metadata(metadata)
            
            # Create empty database file
            # NOTE: Do NOT write header comments here - they get interpreted as submission #1
            # by the fallback parsing in shared_training.py. The file starts empty and
            # submissions are added via shared_training_memory which handles formatting.
            db_path = self._get_database_path(topic_id)
            async with aiofiles.open(db_path, 'w', encoding='utf-8') as f:
                await f.write("")  # Empty file - shared_training_memory will manage content
            
            logger.info(f"Created new brainstorm: {topic_id}")
            return metadata
    
    async def get_metadata(self, topic_id: str) -> Optional[BrainstormMetadata]:
        """Get metadata for a brainstorm topic."""
        metadata_path = self._get_metadata_path(topic_id)
        
        if not metadata_path.exists():
            return None
        
        try:
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                return BrainstormMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load brainstorm metadata for {topic_id}: {e}")
            return None
    
    async def _save_metadata(self, metadata: BrainstormMetadata) -> None:
        """Save brainstorm metadata to file."""
        metadata_path = self._get_metadata_path(metadata.topic_id)
        
        # Convert to dict, handling datetime serialization
        data = metadata.model_dump()
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2))
    
    async def update_metadata(self, topic_id: str, **kwargs) -> Optional[BrainstormMetadata]:
        """Update specific fields in brainstorm metadata."""
        async with self._lock:
            metadata = await self.get_metadata(topic_id)
            if metadata is None:
                return None
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            metadata.last_activity = datetime.now()
            await self._save_metadata(metadata)
            return metadata
    
    async def mark_complete(self, topic_id: str) -> Optional[BrainstormMetadata]:
        """Mark a brainstorm as complete."""
        return await self.update_metadata(
            topic_id,
            status="complete",
            completed_at=datetime.now()
        )
    
    async def add_paper_reference(self, topic_id: str, paper_id: str) -> Optional[BrainstormMetadata]:
        """Add a paper reference to the brainstorm metadata."""
        metadata = await self.get_metadata(topic_id)
        if metadata is None:
            return None
        
        if paper_id not in metadata.papers_generated:
            metadata.papers_generated.append(paper_id)
            await self._save_metadata(metadata)
        
        return metadata

    async def remove_paper_reference(self, topic_id: str, paper_id: str) -> Optional[BrainstormMetadata]:
        """Remove a paper reference from the brainstorm metadata if it exists."""
        metadata = await self.get_metadata(topic_id)
        if metadata is None:
            return None

        if paper_id in metadata.papers_generated:
            metadata.papers_generated = [
                existing_paper_id
                for existing_paper_id in metadata.papers_generated
                if existing_paper_id != paper_id
            ]
            await self._save_metadata(metadata)

        return metadata
    
    async def get_all_brainstorms(self) -> List[BrainstormMetadata]:
        """Get metadata for all brainstorm topics."""
        brainstorms = []
        
        if not self._base_dir.exists():
            return brainstorms
        
        for path in self._base_dir.glob("brainstorm_*_metadata.json"):
            try:
                async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    brainstorms.append(BrainstormMetadata(**data))
            except Exception as e:
                logger.error(f"Failed to load brainstorm metadata from {path}: {e}")
        
        # Sort by last activity (most recent first)
        brainstorms.sort(key=lambda x: x.last_activity, reverse=True)
        return brainstorms
    
    # ========================================================================
    # DATABASE OPERATIONS
    # ========================================================================
    
    async def add_submission(self, topic_id: str, content: str, submission_number: int) -> bool:
        """Add an accepted submission to the brainstorm database."""
        async with self._lock:
            db_path = self._get_database_path(topic_id)
            
            if not db_path.exists():
                logger.error(f"Brainstorm database not found: {topic_id}")
                return False
            
            try:
                # Append submission to database
                async with aiofiles.open(db_path, 'a', encoding='utf-8') as f:
                    await f.write(f"\n{'=' * 80}\n")
                    await f.write(f"SUBMISSION #{submission_number} | Accepted: {datetime.now().isoformat()}\n")
                    await f.write(f"{'=' * 80}\n\n")
                    await f.write(content)
                    await f.write("\n")
                
                # Update metadata
                metadata = await self.get_metadata(topic_id)
                if metadata:
                    metadata.submission_count += 1
                    metadata.last_activity = datetime.now()
                    await self._save_metadata(metadata)
                
                logger.info(f"Added submission #{submission_number} to brainstorm {topic_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add submission to brainstorm {topic_id}: {e}")
                return False
    
    async def get_database_content(self, topic_id: str, *, strip_proofs: bool = False) -> str:
        """Get all content from a brainstorm database.

        Args:
            topic_id: The brainstorm topic ID.
            strip_proofs: When True, truncate content at the proof section header.
                Use this for compiler and RAG paths so that appended proof blocks
                (both novel and non-novel) do not pollute LLM context.  Novel
                proofs are available via proof_database.inject_into_prompt();
                non-novel proofs are browsable via
                proof_database.get_known_proofs_summary_for_browsing().
        """
        db_path = self._get_database_path(topic_id)

        if not db_path.exists():
            return ""

        try:
            async with aiofiles.open(db_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            if strip_proofs and content:
                marker = "=== PROOFS GENERATED FROM THIS BRAINSTORM"
                idx = content.find(marker)
                if idx > 0:
                    content = content[:idx].rstrip()
            return content
        except Exception as e:
            logger.error(f"Failed to read brainstorm database {topic_id}: {e}")
            return ""

    async def append_proofs_section(self, topic_id: str, proofs_data: Any) -> bool:
        """Append verified proofs to the bottom of a brainstorm database."""
        async with self._lock:
            db_path = self._get_database_path(topic_id)
            if not db_path.exists():
                logger.error(f"Brainstorm database not found for proof append: {topic_id}")
                return False

            proofs = proofs_data if isinstance(proofs_data, list) else [proofs_data]
            header = "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ==="

            try:
                async with aiofiles.open(db_path, "r", encoding="utf-8") as handle:
                    existing_content = await handle.read()

                after_header = existing_content.split(header, 1)[1] if header in existing_content else ""
                next_index = len(re.findall(r"(?m)^Proof \d+:", after_header)) + 1

                lines: List[str] = []
                if header not in existing_content:
                    lines.extend(["", "", header, ""])
                elif not existing_content.endswith("\n"):
                    lines.append("")

                for proof in proofs:
                    theorem_statement = str(getattr(proof, "theorem_statement", "") or proof.get("theorem_statement", "")).strip()
                    proof_id = str(getattr(proof, "proof_id", "") or proof.get("proof_id", "")).strip()
                    novel = bool(getattr(proof, "novel", False) if hasattr(proof, "novel") else proof.get("novel", False))
                    lean_code = str(getattr(proof, "lean_code", "") or proof.get("lean_code", "")).strip()
                    status = "Verified (Novel)" if novel else "Verified (Known)"

                    lines.extend(
                        [
                            f"Proof {next_index}: {theorem_statement}",
                            f"Status: {status}",
                            f"Proof ID: {proof_id or 'N/A'}",
                            "Lean 4 Code:",
                            lean_code or "[no Lean 4 code saved]",
                            "---",
                        ]
                    )
                    next_index += 1

                async with aiofiles.open(db_path, "a", encoding="utf-8") as handle:
                    await handle.write("\n".join(lines) + "\n")

                logger.info("Appended %s proof(s) to brainstorm %s", len(proofs), topic_id)
                return True
            except Exception as exc:
                logger.error(f"Failed to append proofs to brainstorm {topic_id}: {exc}")
                return False
    
    async def get_submissions_list(self, topic_id: str) -> List[Dict[str, Any]]:
        """Get list of submissions from a brainstorm database."""
        content = await self.get_database_content(topic_id)
        
        if not content:
            return []
        
        submissions = []
        parts = content.split("=" * 80)
        
        # Parse header/content pairs
        # Format: [header] SEPARATOR [content] SEPARATOR [header] SEPARATOR [content] ...
        # After split: part[0]=file header, part[1]=submission header, part[2]=content, part[3]=submission header, part[4]=content...
        import re
        
        for i, part in enumerate(parts):
            if "SUBMISSION #" in part:
                # This part has the submission header
                lines = part.strip().split("\n")
                header = lines[0] if lines else ""
                
                # Parse header for submission number and timestamp
                match = re.search(r'SUBMISSION #(\d+) \| Accepted: (.+)', header)
                if match:
                    sub_num = int(match.group(1))
                    timestamp = match.group(2)
                    
                    # Content is in the NEXT part after the header
                    content_text = ""
                    if i + 1 < len(parts):
                        content_text = parts[i + 1].strip()
                    
                    submissions.append({
                        "number": sub_num,
                        "timestamp": timestamp,
                        "content": content_text
                    })
        
        return submissions
    
    # ========================================================================
    # RETROACTIVE CORRECTION OPERATIONS (used during paper compilation)
    # ========================================================================
    
    async def edit_submission(self, topic_id: str, submission_number: int, new_content: str) -> bool:
        """
        Edit an existing submission's content in the brainstorm database.
        Preserves submission number and updates timestamp.
        """
        async with self._lock:
            db_path = self._get_database_path(topic_id)
            if not db_path.exists():
                logger.error(f"Brainstorm database not found for edit: {topic_id}")
                return False
            
            try:
                submissions = await self._parse_submissions_unlocked(db_path)
                found = False
                for sub in submissions:
                    if sub['number'] == submission_number:
                        sub['content'] = new_content
                        sub['timestamp'] = datetime.now().isoformat()
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Submission #{submission_number} not found in brainstorm {topic_id}")
                    return False
                
                await self._write_submissions_unlocked(db_path, submissions)
                logger.info(f"Retroactive edit: submission #{submission_number} in brainstorm {topic_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to edit submission #{submission_number} in {topic_id}: {e}")
                return False
    
    async def remove_submission(self, topic_id: str, submission_number: int) -> bool:
        """
        Remove a submission from the brainstorm database.
        Does not renumber remaining submissions.
        """
        async with self._lock:
            db_path = self._get_database_path(topic_id)
            if not db_path.exists():
                logger.error(f"Brainstorm database not found for removal: {topic_id}")
                return False
            
            try:
                submissions = await self._parse_submissions_unlocked(db_path)
                original_count = len(submissions)
                submissions = [s for s in submissions if s['number'] != submission_number]
                
                if len(submissions) == original_count:
                    logger.warning(f"Submission #{submission_number} not found in brainstorm {topic_id}")
                    return False
                
                await self._write_submissions_unlocked(db_path, submissions)
                
                metadata = await self.get_metadata(topic_id)
                if metadata:
                    metadata.submission_count = len(submissions)
                    metadata.last_activity = datetime.now()
                    await self._save_metadata(metadata)
                
                logger.info(f"Retroactive removal: submission #{submission_number} from brainstorm {topic_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to remove submission #{submission_number} from {topic_id}: {e}")
                return False
    
    async def add_submission_retroactive(self, topic_id: str, content: str) -> Optional[int]:
        """
        Add a new submission discovered during paper compilation.
        Returns the new submission number, or None on failure.
        """
        async with self._lock:
            db_path = self._get_database_path(topic_id)
            if not db_path.exists():
                logger.error(f"Brainstorm database not found for retroactive add: {topic_id}")
                return None
            
            try:
                submissions = await self._parse_submissions_unlocked(db_path)
                max_number = max((s['number'] for s in submissions), default=0)
                new_number = max_number + 1
                
                submissions.append({
                    'number': new_number,
                    'timestamp': datetime.now().isoformat(),
                    'content': content
                })
                
                await self._write_submissions_unlocked(db_path, submissions)
                
                metadata = await self.get_metadata(topic_id)
                if metadata:
                    metadata.submission_count = len(submissions)
                    metadata.last_activity = datetime.now()
                    await self._save_metadata(metadata)
                
                logger.info(f"Retroactive add: submission #{new_number} to brainstorm {topic_id}")
                return new_number
            except Exception as e:
                logger.error(f"Failed to retroactively add submission to {topic_id}: {e}")
                return None
    
    async def _parse_submissions_unlocked(self, db_path: Path) -> List[Dict[str, Any]]:
        """Parse submissions from a brainstorm database file. Caller must hold lock."""
        import re
        async with aiofiles.open(db_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        if not content.strip():
            return []
        
        submissions = []
        parts = content.split("=" * 80)
        
        for i, part in enumerate(parts):
            if "SUBMISSION #" in part:
                lines = part.strip().split("\n")
                header = lines[0] if lines else ""
                match = re.search(r'SUBMISSION #(\d+) \| Accepted: (.+)', header)
                if match:
                    sub_num = int(match.group(1))
                    timestamp = match.group(2).strip()
                    content_text = ""
                    if i + 1 < len(parts):
                        content_text = parts[i + 1].strip()
                    submissions.append({
                        'number': sub_num,
                        'timestamp': timestamp,
                        'content': content_text
                    })
        
        return submissions
    
    async def _write_submissions_unlocked(self, db_path: Path, submissions: List[Dict[str, Any]]) -> None:
        """Write submissions back to a brainstorm database file. Caller must hold lock."""
        formatted_sections = []
        separator = '=' * 80
        
        for sub in submissions:
            section = f"{separator}\nSUBMISSION #{sub['number']} | Accepted: {sub['timestamp']}\n{separator}\n\n{sub['content']}\n"
            formatted_sections.append(section)
        
        full_content = '\n\n'.join(formatted_sections)
        async with aiofiles.open(db_path, 'w', encoding='utf-8') as f:
            await f.write(full_content)
    
    # ========================================================================
    # REJECTION LOG OPERATIONS
    # ========================================================================
    
    async def add_submitter_rejection(
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
                "summary": rejection_summary[:750],  # Max 750 chars
                "submission_preview": submission_preview[:750]  # Max 750 chars
            })
            
            # Keep only last 5
            rejections = rejections[-5:]
            
            # Save
            async with aiofiles.open(rejections_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(rejections, indent=2))
    
    async def get_submitter_rejections(self, topic_id: str, submitter_id: int) -> List[Dict[str, Any]]:
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
    
    async def clear_submitter_rejections(self, topic_id: str, submitter_id: int) -> None:
        """Clear rejection log for a submitter."""
        rejections_path = self._get_submitter_rejections_path(topic_id, submitter_id)
        
        if rejections_path.exists():
            async with aiofiles.open(rejections_path, 'w', encoding='utf-8') as f:
                await f.write("[]")
    
    # ========================================================================
    # COMPLETION FEEDBACK OPERATIONS
    # ========================================================================
    
    # Note: Completion feedback methods moved to autonomous_rejection_logs.py
    # to avoid duplication and maintain single source of truth
    
    # ========================================================================
    # TOPIC COMBINATION
    # ========================================================================
    
    async def combine_topics(
        self, 
        new_topic_id: str, 
        new_topic_prompt: str,
        source_topic_ids: List[str]
    ) -> Optional[BrainstormMetadata]:
        """
        Combine multiple brainstorm topics into a new one.
        Merges all submissions from source topics.
        """
        async with self._lock:
            # Create new brainstorm
            metadata = BrainstormMetadata(
                topic_id=new_topic_id,
                topic_prompt=new_topic_prompt,
                status="in_progress",
                submission_count=0,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                papers_generated=[]
            )
            
            # Collect all papers from source topics
            for source_id in source_topic_ids:
                source_meta = await self.get_metadata(source_id)
                if source_meta:
                    metadata.papers_generated.extend(source_meta.papers_generated)
            
            # Remove duplicates
            metadata.papers_generated = list(set(metadata.papers_generated))
            
            # Create empty database file
            # NOTE: Do NOT write header comments here - they get interpreted as submission content
            # by the fallback parsing in shared_training.py
            db_path = self._get_database_path(new_topic_id)
            async with aiofiles.open(db_path, 'w', encoding='utf-8') as f:
                await f.write("")  # Empty file - submissions will be added below
            
            # Merge submissions from all source topics
            submission_counter = 0
            for source_id in source_topic_ids:
                submissions = await self.get_submissions_list(source_id)
                for sub in submissions:
                    submission_counter += 1
                    async with aiofiles.open(db_path, 'a', encoding='utf-8') as f:
                        await f.write(f"\n{'=' * 80}\n")
                        await f.write(f"SUBMISSION #{submission_counter} | Accepted: {datetime.now().isoformat()}\n")
                        await f.write(f"(Originally from {source_id})\n")
                        await f.write(f"{'=' * 80}\n\n")
                        await f.write(sub['content'])
                        await f.write("\n")
            
            metadata.submission_count = submission_counter
            await self._save_metadata(metadata)
            
            logger.info(f"Combined {len(source_topic_ids)} topics into {new_topic_id} with {submission_counter} submissions")
            return metadata
    
    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================
    
    async def delete_brainstorm(self, topic_id: str) -> bool:
        """
        Delete a brainstorm and all associated files.
        
        Args:
            topic_id: Brainstorm topic ID to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        async with self._lock:
            try:
                # Delete database file
                db_path = self._get_database_path(topic_id)
                if db_path.exists():
                    db_path.unlink()
                    logger.info(f"Deleted brainstorm database: {db_path}")
                
                # Delete metadata file
                metadata_path = self._get_metadata_path(topic_id)
                if metadata_path.exists():
                    metadata_path.unlink()
                    logger.info(f"Deleted brainstorm metadata: {metadata_path}")
                
                # Delete completion feedback file
                feedback_path = self._get_completion_feedback_path(topic_id)
                if feedback_path.exists():
                    feedback_path.unlink()
                    logger.info(f"Deleted completion feedback: {feedback_path}")
                
                # Delete all submitter rejection files
                # We don't know how many submitters were used, so scan for all
                for path in self._base_dir.glob(f"brainstorm_{topic_id}_submitter_*_rejections.txt"):
                    path.unlink()
                    logger.info(f"Deleted submitter rejections: {path}")
                
                logger.info(f"Successfully deleted brainstorm {topic_id} and all associated files")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete brainstorm {topic_id}: {e}")
                return False


# Global instance
brainstorm_memory = BrainstormMemory()

