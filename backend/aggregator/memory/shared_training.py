"""
Shared training database managed by validator.
Contains accepted submissions distributed to all submitters.
"""
import aiofiles
from pathlib import Path
from typing import List, Callable, Optional, Dict
import asyncio
import logging
import re
from datetime import datetime

from backend.shared.config import system_config, rag_config

logger = logging.getLogger(__name__)

PROOF_APPENDIX_HEADER = "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ==="
MANUAL_AGGREGATOR_PROMPT_FILE = "manual_aggregator_prompt.txt"


def get_manual_aggregator_prompt_path() -> Path:
    """Return the persisted manual Aggregator prompt path for this data root."""
    return Path(system_config.data_dir) / MANUAL_AGGREGATOR_PROMPT_FILE


async def save_manual_aggregator_prompt(prompt: str) -> None:
    """Persist the latest manual Aggregator prompt for stopped/restarted proof checks."""
    if not (prompt or "").strip():
        logger.warning("Refusing to overwrite manual Aggregator prompt with an empty value")
        return
    path = get_manual_aggregator_prompt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    async with aiofiles.open(temp_path, "w", encoding="utf-8") as handle:
        await handle.write(prompt or "")
    await asyncio.to_thread(temp_path.replace, path)


async def load_manual_aggregator_prompt() -> str:
    """Load the latest manual Aggregator prompt, if one has been persisted."""
    path = get_manual_aggregator_prompt_path()
    if not path.exists():
        return ""
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as handle:
            return await handle.read()
    except Exception as exc:
        logger.debug("Unable to load manual Aggregator prompt: %s", exc)
        return ""


async def clear_manual_aggregator_prompt() -> None:
    """Clear stale manual Aggregator prompt state after an explicit reset."""
    path = get_manual_aggregator_prompt_path()
    if path.exists():
        try:
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.debug("Unable to clear manual Aggregator prompt: %s", exc)


class SharedTrainingMemory:
    """
    Validator-distributed training database.
    Contains accepted submissions that are shared with all submitters.
    """
    
    def __init__(self):
        self.file_path = Path(system_config.shared_training_file)
        self.insights: List[Dict[str, str]] = []  # Now stores dicts with metadata
        self.max_insights = rag_config.max_shared_training_insights
        self.rechunk_callback: Optional[Callable] = None
        self._lock = asyncio.Lock()
        self.submission_count = 0
        self.last_ragged_submission_count = 0  # Track which submissions have been RAG'd
        self.proof_appendix = ""
    
    async def initialize(self) -> None:
        """Initialize shared training memory, creating file if needed."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_path.exists():
            # Load existing insights
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    content, proof_appendix = self._split_proof_appendix(content)
                    # Parse formatted submissions
                    parsed_insights = self._parse_formatted_file(content)
                    
                    # CRITICAL: Acquire lock before modifying shared state
                    # This prevents race conditions with concurrent access from
                    # aggregator cleanup reviews and compiler initialization
                    async with self._lock:
                        self.insights = parsed_insights
                        self.proof_appendix = proof_appendix
                        # Set submission_count to the highest submission number found
                        if self.insights:
                            max_number = max(
                                (insight.get('number', 0) for insight in self.insights),
                                default=0
                            )
                            self.submission_count = max_number
                        else:
                            self.submission_count = 0
            logger.info(f"Loaded {len(self.insights)} existing insights from shared training (submission count: {self.submission_count})")
        else:
            # Create empty file
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write("")
            logger.info("Created new shared training file")
    
    async def reload_insights_from_current_path(self) -> None:
        """
        Reload insights from the current file path.
        
        CRITICAL: Used when the file path has been changed (e.g., for brainstorm-specific databases)
        but the insights list needs to be refreshed from the new file to avoid data loss.
        
        Without this, old insights from the previous file would remain in memory and overwrite
        the new file's contents on the next save, causing data loss.
        """
        async with self._lock:
            # Clear current insights
            self.insights.clear()
            self.proof_appendix = ""
            self.submission_count = 0
            self.last_ragged_submission_count = 0  # Reset RAG tracking
            
            if self.file_path.exists():
                # Load insights from the new path
                async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content.strip():
                        content, proof_appendix = self._split_proof_appendix(content)
                        parsed_insights = self._parse_formatted_file(content)
                        self.insights = parsed_insights
                        self.proof_appendix = proof_appendix
                        
                        # Set submission_count to the highest submission number found
                        if self.insights:
                            max_number = max(
                                (insight.get('number', 0) for insight in self.insights),
                                default=0
                            )
                            self.submission_count = max_number
                            # Use entry count (not max number) so post-prune gaps
                            # don't cause the next acceptance to be skipped from RAG
                            self.last_ragged_submission_count = len(self.insights)
                        else:
                            self.submission_count = 0
                
            else:
                logger.info(f"Brainstorm database file doesn't exist yet: {self.file_path}")

    def _split_proof_appendix(self, content: str) -> tuple[str, str]:
        """Return accepted-submission content and a preserved proof appendix."""
        if PROOF_APPENDIX_HEADER not in content:
            return content, ""
        before, _, after = content.partition(PROOF_APPENDIX_HEADER)
        proof_appendix = f"{PROOF_APPENDIX_HEADER}{after}".strip()
        return before.rstrip(), proof_appendix

    @staticmethod
    def _proof_field(proof, field_name: str, default: str = "") -> str:
        if isinstance(proof, dict):
            return str(proof.get(field_name, default) or default)
        return str(getattr(proof, field_name, default) or default)
    
    def _parse_formatted_file(self, content: str) -> List[Dict[str, str]]:
        """Parse the formatted file to extract submissions and metadata."""
        insights = []
        
        # Pattern matches: separator + header + separator + content
        # The pattern captures the submission number, timestamp, and content
        pattern = r'={80}\s*SUBMISSION #(\d+)\s*\|\s*Accepted:\s*([^\n]+)\s*={80}\s*\n(.*?)(?=\n={80}\s*SUBMISSION|$)'
        
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            number = int(match.group(1))
            timestamp = match.group(2).strip()
            content_text = match.group(3).strip()
            
            if content_text:
                insights.append({
                    'content': content_text,
                    'timestamp': timestamp,
                    'number': number
                })
        
        # If no matches found using the pattern, try fallback parsing
        if not insights and content.strip():
            # Fallback: just extract any non-empty content blocks
            sections = content.split('=' * 80)
            fallback_number = 0
            for section in sections:
                section = section.strip()
                if section and 'SUBMISSION #' not in section:
                    # This is likely content without proper formatting
                    fallback_number += 1
                    insights.append({
                        'content': section,
                        'timestamp': datetime.now().isoformat(),
                        'number': fallback_number  # CRITICAL: Always set number to avoid #? display
                    })
        
        return insights
    
    async def add_accepted_submission(self, submission_content: str) -> None:
        """
        Add an accepted submission to shared training.
        Never truncates submission content.
        Triggers re-chunking callback if set.
        """
        async with self._lock:
            # Increment submission count
            self.submission_count += 1
            
            # Add submission with metadata (never truncate)
            self.insights.append({
                'content': submission_content,
                'timestamp': datetime.now().isoformat(),
                'number': self.submission_count
            })
            
            # NOTE: Never prune accepted submissions per design rules
            # "Accepted submission logs should never have its submission results truncated"
            # The max_insights limit exists only as a safety overflow check
            if len(self.insights) > self.max_insights:
                logger.critical(
                    f"WARNING: Shared training has {len(self.insights)} insights, "
                    f"exceeding safety limit of {self.max_insights}. "
                    f"Consider increasing max_shared_training_insights in config. "
                    f"NOT pruning per design rules - all accepted submissions preserved."
                )
            
            # Save to file
            await self._save()
            
            # Trigger re-chunking callback IMMEDIATELY after each acceptance
            if self.rechunk_callback:
                try:
                    logger.info(f"Triggering immediate re-chunking callback for acceptance #{self.submission_count}")
                    await self.rechunk_callback()
                except Exception as e:
                    logger.error(f"Re-chunking callback failed: {e}")
    
    async def get_all_content(self, *, strip_proofs: bool = False) -> str:
        """Get all shared training content as a single string (content only, no metadata)."""
        async with self._lock:
            content = '\n\n'.join([insight['content'] for insight in self.insights])
            if strip_proofs or not self.proof_appendix:
                return content
            return f"{content.rstrip()}\n\n{self.proof_appendix}\n" if content.strip() else self.proof_appendix
    
    async def get_all_content_formatted(self, *, strip_proofs: bool = False) -> str:
        """Get all shared training content with full formatting and metadata for export."""
        async with self._lock:
            formatted_sections = []
            
            for idx, insight in enumerate(self.insights, 1):
                # Create formatted section with clear separator
                separator = '=' * 80
                # CRITICAL: If number is missing, use position. This prevents #? display.
                number = insight.get('number') or idx
                timestamp = insight.get('timestamp', 'Unknown')
                content = insight['content']
                
                section = f"{separator}\nSUBMISSION #{number} | Accepted: {timestamp}\n{separator}\n\n{content}\n"
                formatted_sections.append(section)
            content = '\n\n'.join(formatted_sections)
            if strip_proofs or not self.proof_appendix:
                return content
            return f"{content.rstrip()}\n\n{self.proof_appendix}\n" if content.strip() else self.proof_appendix

    async def append_proofs_section(self, proofs_data) -> bool:
        """Append Lean-verified proof records without converting them into submissions."""
        async with self._lock:
            proofs = proofs_data if isinstance(proofs_data, list) else [proofs_data]
            existing_appendix = self.proof_appendix or PROOF_APPENDIX_HEADER
            existing_ids = set(re.findall(r"(?m)^Proof ID:\s*(.+?)\s*$", existing_appendix))
            after_header = existing_appendix.split(PROOF_APPENDIX_HEADER, 1)[1] if PROOF_APPENDIX_HEADER in existing_appendix else ""
            next_index = len(re.findall(r"(?m)^Proof \d+:", after_header)) + 1

            lines = [existing_appendix.rstrip()]
            appended = 0
            for proof in proofs:
                theorem_statement = self._proof_field(proof, "theorem_statement").strip()
                proof_id = self._proof_field(proof, "proof_id").strip()
                novel = bool(proof.get("novel", False) if isinstance(proof, dict) else getattr(proof, "novel", False))
                lean_code = self._proof_field(proof, "lean_code").strip()
                if proof_id and proof_id in existing_ids:
                    continue
                status = "Verified (Novel)" if novel else "Verified (Known)"
                lines.extend(
                    [
                        "",
                        f"Proof {next_index}: {theorem_statement}",
                        f"Status: {status}",
                        f"Proof ID: {proof_id or 'N/A'}",
                        "Lean 4 Code:",
                        lean_code or "[no Lean 4 code saved]",
                        "---",
                    ]
                )
                if proof_id:
                    existing_ids.add(proof_id)
                next_index += 1
                appended += 1

            if appended == 0:
                return True

            self.proof_appendix = "\n".join(lines).strip()
            await self._save()
            logger.info("Appended %s proof(s) to shared training database", appended)
            return True

    async def refresh_proof_appendix_from_file(self) -> None:
        """Refresh only the proof appendix from disk without touching live submissions."""
        try:
            if not self.file_path.exists():
                return
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        except Exception as exc:
            logger.debug("Unable to refresh proof appendix from shared training file: %s", exc)
            return

        _, proof_appendix = self._split_proof_appendix(content)
        async with self._lock:
            self.proof_appendix = proof_appendix
    
    async def get_insights_count(self) -> int:
        """Get number of insights in shared training."""
        async with self._lock:
            return len(self.insights)
    
    async def get_new_submissions_since_last_rag(self) -> List[Dict[str, str]]:
        """Get submissions that haven't been RAG'd yet."""
        async with self._lock:
            return self.insights[self.last_ragged_submission_count:]
    
    async def mark_submissions_ragged(self, up_to_count: int) -> None:
        """Mark submissions as RAG'd up to specified count."""
        async with self._lock:
            self.last_ragged_submission_count = up_to_count
            logger.debug(f"Marked {up_to_count} submissions as RAG'd")
    
    def set_rechunk_callback(self, callback: Callable) -> None:
        """Set callback to be called when training data updates."""
        self.rechunk_callback = callback
    
    async def _save(self) -> None:
        """Save insights to file with clear formatting and metadata."""
        formatted_sections = []
        
        for idx, insight in enumerate(self.insights, 1):
            # Create formatted section with clear separator
            separator = '=' * 80
            # CRITICAL: If number is missing, use position. This prevents #? display.
            number = insight.get('number') or idx
            timestamp = insight.get('timestamp', 'Unknown')
            content = insight['content']
            
            section = f"{separator}\nSUBMISSION #{number} | Accepted: {timestamp}\n{separator}\n\n{content}\n"
            formatted_sections.append(section)
        
        # Join all sections with double newline for clear separation
        full_content = '\n\n'.join(formatted_sections)
        if self.proof_appendix:
            full_content = f"{full_content.rstrip()}\n\n{self.proof_appendix}\n" if full_content.strip() else self.proof_appendix
        
        async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
            await f.write(full_content)
        logger.debug(f"Saved {len(self.insights)} insights to shared training")
    
    async def get_submission_content(self, submission_number: int) -> Optional[str]:
        """
        Get the content of a specific submission by number.
        
        Args:
            submission_number: The submission number to retrieve
            
        Returns:
            The submission content if found, None otherwise
        """
        async with self._lock:
            for insight in self.insights:
                if insight.get('number') == submission_number:
                    return insight['content']
            return None
    
    async def remove_submission(self, submission_number: int, trigger_rechunk: bool = True) -> bool:
        """
        Remove a submission from the shared training database.
        
        This is used during cleanup reviews to remove redundant or
        problematic submissions that were previously accepted.
        
        Args:
            submission_number: The submission number to remove
            trigger_rechunk: Whether to fire the incremental rechunk callback.
                Set False when the caller will do a full RAG rebuild instead.
            
        Returns:
            True if submission was found and removed, False otherwise
        """
        async with self._lock:
            # Find the submission by number
            original_count = len(self.insights)
            self.insights = [
                insight for insight in self.insights
                if insight.get('number') != submission_number
            ]
            
            if len(self.insights) < original_count:
                # Submission was removed
                logger.info(f"Removed submission #{submission_number} from shared training")
                
                # Save to file
                await self._save()
                
                # Trigger re-chunking callback to update RAG
                if trigger_rechunk and self.rechunk_callback:
                    try:
                        logger.info(f"Triggering re-chunking callback after removal of submission #{submission_number}")
                        await self.rechunk_callback()
                    except Exception as e:
                        logger.error(f"Re-chunking callback failed after removal: {e}")
                
                return True
            else:
                logger.warning(f"Submission #{submission_number} not found for removal")
                return False


# Global shared training memory instance
shared_training_memory = SharedTrainingMemory()


async def append_proof_to_manual_shared_training(proof_record) -> bool:
    """Append a proof to the manual Aggregator DB regardless of the singleton's current path."""
    manual_path = Path(system_config.shared_training_file)
    current_path = Path(shared_training_memory.file_path)
    try:
        paths_match = current_path.resolve() == manual_path.resolve()
    except Exception:
        paths_match = current_path == manual_path

    if paths_match:
        await shared_training_memory.refresh_proof_appendix_from_file()
        return await shared_training_memory.append_proofs_section(proof_record)

    scoped_memory = SharedTrainingMemory()
    scoped_memory.file_path = manual_path
    await scoped_memory.initialize()
    return await scoped_memory.append_proofs_section(proof_record)


async def clear_manual_shared_training_proof_appendix() -> None:
    """Remove manual Aggregator proof appendix without deleting accepted submissions."""
    manual_path = Path(system_config.shared_training_file)
    current_path = Path(shared_training_memory.file_path)
    try:
        paths_match = current_path.resolve() == manual_path.resolve()
    except Exception:
        paths_match = current_path == manual_path

    if paths_match:
        async with shared_training_memory._lock:
            shared_training_memory.proof_appendix = ""
            await shared_training_memory._save()
        return

    scoped_memory = SharedTrainingMemory()
    scoped_memory.file_path = manual_path
    await scoped_memory.initialize()
    async with scoped_memory._lock:
        scoped_memory.proof_appendix = ""
        await scoped_memory._save()

