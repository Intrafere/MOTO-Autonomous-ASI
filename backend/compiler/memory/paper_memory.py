"""
Paper memory manager for compiler.
Handles paper file I/O, real-time updates, and word count tracking.
"""
import aiofiles
import asyncio
from typing import Optional, Callable, List, Dict
from pathlib import Path
import logging

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

# Hardcoded paper anchor - serves as non-chronological stop token
PAPER_ANCHOR = "[HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]"

# Section placeholders - replaced when validated content is written
# These make it crystal clear to the AI where sections go and that they don't exist yet
ABSTRACT_PLACEHOLDER = "[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]"
INTRO_PLACEHOLDER = "[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]"
CONCLUSION_PLACEHOLDER = "[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]"

# Theorems appendix bracket markers - wrap the Lean-4-verified theorems appendix.
# System-managed; the body is free-form and may be empty (default placeholder line)
# or contain theorem entries appended by the rigor flow.
THEOREMS_APPENDIX_START = "[HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]"
THEOREMS_APPENDIX_END = "[HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]"
APPENDIX_EMPTY_PLACEHOLDER = "[Theorems appendix - verified Lean 4 theorems not placed inline will appear here]"


class PaperMemory:
    """
    Manages the current paper state.
    - File I/O to compiler_paper.txt
    - Real-time updates for GUI
    - Word count tracking
    - Thread-safe operations
    - Version tracking
    """
    
    def __init__(self):
        self.file_path = Path(system_config.compiler_paper_file)
        self.version = 0
        self.rechunk_callback: Optional[Callable] = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self.previous_versions = []  # Store previous body versions for UI display
    
    async def initialize(self) -> None:
        """Initialize paper memory."""
        async with self._lock:
            if self._initialized:
                return
            
            # Create file if doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                    await f.write("")
            
            self._initialized = True
            logger.info("Paper memory initialized")
    
    async def _get_paper_unlocked(self) -> str:
        """Get current paper content (internal, assumes lock already held)."""
        if not self.file_path.exists():
            return ""
        
        async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        return content
    
    async def get_paper(self) -> str:
        """Get current paper content."""
        async with self._lock:
            return await self._get_paper_unlocked()
    
    def _strip_duplicate_anchors(self, content: str) -> str:
        """
        Remove all anchor occurrences from content.
        
        Args:
            content: Content to clean
        
        Returns:
            Content with anchors removed
        """
        return content.replace(PAPER_ANCHOR, "").strip()
    
    def _ensure_anchor(self, content: str) -> str:
        """
        Ensure paper ends with anchor marker.
        
        Args:
            content: Paper content
        
        Returns:
            Content with anchor appended at end
        """
        if not content.strip():
            return ""
        
        # Add anchor if not already at end
        content_stripped = content.rstrip()
        if not content_stripped.endswith(PAPER_ANCHOR):
            return content_stripped + "\n\n" + PAPER_ANCHOR
        return content_stripped
    
    async def _update_paper_unlocked(self, new_paper: str) -> str:
        """
        Update paper file (internal, assumes lock already held).
        Does NOT trigger re-chunking callback - caller must handle that.
        
        Args:
            new_paper: New paper content
            
        Returns:
            The final content that was written (for rechunk callback)
        """
        # Strip any duplicate anchors from input
        cleaned = self._strip_duplicate_anchors(new_paper)
        
        # Ensure single anchor at end
        final_content = self._ensure_anchor(cleaned)
        
        # Write to file
        async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
            await f.write(final_content)
        
        # Increment version
        self.version += 1
        
        logger.info(f"Paper updated (version {self.version}, {self.get_word_count_sync(final_content)} words)")
        
        return final_content
    
    async def update_paper(self, new_paper: str) -> None:
        """
        Update paper and trigger re-chunking.
        Automatically strips duplicate anchors and ensures single anchor at end.
        
        Args:
            new_paper: New paper content
        """
        async with self._lock:
            final_content = await self._update_paper_unlocked(new_paper)
        
        # Trigger re-chunking callback OUTSIDE the lock to avoid deadlock
        if self.rechunk_callback:
            try:
                await self.rechunk_callback(final_content)
            except Exception as e:
                logger.error(f"Re-chunking callback failed: {e}")
    
    async def get_word_count(self) -> int:
        """Get current paper word count."""
        content = await self.get_paper()
        return self.get_word_count_sync(content)
    
    def get_word_count_sync(self, text: str) -> int:
        """Get word count from text (synchronous)."""
        if not text:
            return 0
        return len(text.split())
    
    async def is_empty(self) -> bool:
        """Check if paper is empty."""
        content = await self.get_paper()
        return len(content.strip()) == 0
    
    def set_rechunk_callback(self, callback: Callable) -> None:
        """Set callback to trigger re-chunking when paper updated."""
        self.rechunk_callback = callback
    
    def get_version(self) -> int:
        """Get current paper version."""
        return self.version
    
    async def initialize_with_placeholders(self, first_body_content: str) -> None:
        """
        Initialize paper with first body content and all section placeholders.
        Called when the first body section is accepted.
        
        Args:
            first_body_content: The first body section content (already validated)
        """
        # Build paper with placeholders framing the body content.
        # Appendix block sits between the conclusion placeholder and the paper anchor;
        # its body starts as the empty-placeholder line and is later replaced /
        # appended to by append_to_theorems_appendix.
        paper = (
            f"{ABSTRACT_PLACEHOLDER}\n\n"
            f"{INTRO_PLACEHOLDER}\n\n"
            f"{first_body_content}\n\n"
            f"{CONCLUSION_PLACEHOLDER}\n\n"
            f"{THEOREMS_APPENDIX_START}\n"
            f"{APPENDIX_EMPTY_PLACEHOLDER}\n"
            f"{THEOREMS_APPENDIX_END}\n\n"
            f"{PAPER_ANCHOR}"
        )
        
        async with self._lock:
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(paper)
            
            self.version += 1
            logger.info(f"Paper initialized with placeholders (version {self.version})")
        
        # Trigger re-chunking callback
        if self.rechunk_callback:
            try:
                await self.rechunk_callback(paper)
            except Exception as e:
                logger.error(f"Re-chunking callback failed: {e}")
    
    async def replace_placeholder(self, placeholder: str, content: str) -> bool:
        """
        Replace a placeholder with validated content.
        Each placeholder is replaced exactly once.
        
        Args:
            placeholder: The placeholder constant to replace
            content: The validated section content
        
        Returns:
            True if placeholder was found and replaced, False otherwise
        """
        async with self._lock:
            if not self.file_path.exists():
                logger.warning("Cannot replace placeholder: paper file does not exist")
                return False
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                paper = await f.read()
            
            if placeholder not in paper:
                logger.warning(f"Placeholder not found in paper: {placeholder[:50]}...")
                return False
            
            # Replace the placeholder with the content (only once)
            new_paper = paper.replace(placeholder, content, 1)
            
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(new_paper)
            
            self.version += 1
            logger.info(f"Placeholder replaced (version {self.version})")
        
        # Trigger re-chunking callback OUTSIDE the lock
        if self.rechunk_callback:
            try:
                await self.rechunk_callback(new_paper)
            except Exception as e:
                logger.error(f"Re-chunking callback failed: {e}")
        
        return True
    
    def has_placeholder(self, placeholder: str, paper_content: str) -> bool:
        """
        Check if a placeholder exists in the paper content.
        
        Args:
            placeholder: The placeholder constant to check
            paper_content: The current paper content
        
        Returns:
            True if placeholder is present, False otherwise
        """
        return placeholder in paper_content
    
    async def clear_body_section(self) -> None:
        """
        Clear body section but preserve placeholders.
        Removes all content between the first body content and CONCLUSION_PLACEHOLDER.
        Keeps Abstract, Introduction, and Conclusion placeholders intact.
        """
        final_content = None
        
        async with self._lock:
            # Use unlocked version since we already hold the lock
            current_paper = await self._get_paper_unlocked()
            
            if not current_paper:
                logger.warning("Cannot clear body - paper is empty")
                return
            
            # Find where body content starts (after Abstract and Introduction placeholders)
            lines = current_paper.split('\n')
            body_start_idx = None
            body_end_idx = None
            
            # Find first non-placeholder, non-anchor line (this is body content)
            for i, line in enumerate(lines):
                if (line.strip() and 
                    ABSTRACT_PLACEHOLDER not in line and
                    INTRO_PLACEHOLDER not in line and
                    CONCLUSION_PLACEHOLDER not in line and
                    THEOREMS_APPENDIX_START not in line and
                    THEOREMS_APPENDIX_END not in line and
                    PAPER_ANCHOR not in line):
                    body_start_idx = i
                    break
            
            # Find where body ends (at CONCLUSION_PLACEHOLDER, THEOREMS_APPENDIX_START, or PAPER_ANCHOR)
            for i, line in enumerate(lines):
                if (CONCLUSION_PLACEHOLDER in line
                        or THEOREMS_APPENDIX_START in line
                        or PAPER_ANCHOR in line):
                    body_end_idx = i
                    break
            
            if body_start_idx is None:
                logger.info("No body content found to clear")
                return
            
            # Reconstruct paper with body section removed
            new_lines = []
            
            # Keep placeholders before body
            new_lines.extend(lines[:body_start_idx])
            
            # Skip body content, add everything from conclusion placeholder onward
            if body_end_idx is not None:
                new_lines.extend(lines[body_end_idx:])
            
            new_paper = '\n'.join(new_lines)
            
            # Save using unlocked version since we already hold the lock
            final_content = await self._update_paper_unlocked(new_paper)
            
            logger.info(f"Cleared body section (lines {body_start_idx} to {body_end_idx})")
        
        # Trigger re-chunking callback OUTSIDE the lock to avoid deadlock
        if final_content and self.rechunk_callback:
            try:
                await self.rechunk_callback(final_content)
            except Exception as e:
                logger.error(f"Re-chunking callback failed after clearing body: {e}")
    
    async def store_previous_version(
        self,
        version: int,
        title: str,
        body: str,
        critique_feedback: str
    ) -> None:
        """
        Store previous body version for UI display.
        
        Args:
            version: Version number
            title: Paper title for this version
            body: Body section content
            critique_feedback: Critique feedback that triggered rewrite
        """
        async with self._lock:
            # Add to in-memory list
            version_data = {
                "version": version,
                "title": title,
                "body": body,
                "critique_feedback": critique_feedback
            }
            self.previous_versions.append(version_data)
            
            # Save to file
            version_file = Path(system_config.data_dir) / f"paper_version_{version}.txt"
            version_file.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(version_file, 'w', encoding='utf-8') as f:
                await f.write(f"VERSION {version}: {title}\n")
                await f.write(f"{'=' * 80}\n\n")
                await f.write(f"BODY SECTION:\n{body}\n\n")
                await f.write(f"{'=' * 80}\n\n")
                await f.write(f"CRITIQUE FEEDBACK THAT TRIGGERED REWRITE:\n{critique_feedback}\n")
            
            logger.info(f"Stored previous version {version} to {version_file}")
    
    async def get_previous_versions(self) -> list:
        """
        Get all previous versions for UI display.
        
        Returns:
            List of version dicts with version, title, body, critique_feedback
        """
        async with self._lock:
            return self.previous_versions.copy()
    
    def _extract_body_and_appendix(self, paper: str) -> tuple[str, str]:
        """
        Split existing paper into (body_lines_text, appendix_body_text).
        
        The "body" is all non-marker content that sits OUTSIDE the
        THEOREMS_APPENDIX_START / THEOREMS_APPENDIX_END brackets. The
        "appendix body" is everything BETWEEN those two markers (preserved
        verbatim so repair operations never lose verified-theorem entries).
        
        Marker lines themselves (placeholders + anchor + appendix brackets)
        are stripped from the body half; they get re-emitted by the repair
        logic.
        
        Args:
            paper: Raw paper content
        
        Returns:
            (body_text, appendix_body_text). Both stripped. Either may be "".
        """
        lines = paper.split('\n')
        body_lines: list[str] = []
        appendix_lines: list[str] = []
        in_appendix = False
        
        for line in lines:
            if THEOREMS_APPENDIX_START in line:
                in_appendix = True
                continue
            if THEOREMS_APPENDIX_END in line:
                in_appendix = False
                continue
            
            if in_appendix:
                appendix_lines.append(line)
                continue
            
            # Outside the appendix: skip structural markers
            if (ABSTRACT_PLACEHOLDER in line
                    or INTRO_PLACEHOLDER in line
                    or CONCLUSION_PLACEHOLDER in line
                    or PAPER_ANCHOR in line):
                continue
            
            body_lines.append(line)
        
        return '\n'.join(body_lines).strip(), '\n'.join(appendix_lines).strip()
    
    def _build_appendix_block(self, appendix_body: str) -> str:
        """
        Produce the fully-bracketed appendix block as it should appear in
        the paper file. When the caller has no entries yet, the empty
        placeholder line is used so the LLM-facing rendering stays readable.
        """
        body = (appendix_body or "").strip()
        if not body:
            body = APPENDIX_EMPTY_PLACEHOLDER
        return (
            f"{THEOREMS_APPENDIX_START}\n"
            f"{body}\n"
            f"{THEOREMS_APPENDIX_END}"
        )
    
    async def append_to_theorems_appendix(self, theorem_entry: str) -> bool:
        """
        Atomically append a verified-theorem entry to the Theorems Appendix.
        
        - Locates the THEOREMS_APPENDIX_END marker and inserts the new entry
          immediately above it.
        - If the appendix is still carrying the default APPENDIX_EMPTY_PLACEHOLDER
          line, that line is replaced wholesale by the new entry (first entry).
        - If neither marker exists yet, the paper is left untouched and False
          is returned (caller should trigger ensure_markers_intact first).
        - Triggers the re-chunking callback outside the lock so RAG sees the
          new theorem text.
        
        Args:
            theorem_entry: Preformatted entry (statement + Lean code + trailing
                separator). Caller controls formatting.
        
        Returns:
            True on success, False when the paper is missing or markers aren't
            present yet.
        """
        entry = (theorem_entry or "").strip()
        if not entry:
            logger.warning("append_to_theorems_appendix called with empty entry")
            return False
        
        final_content: Optional[str] = None
        
        async with self._lock:
            if not self.file_path.exists():
                logger.warning("Cannot append to appendix: paper file does not exist")
                return False
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                paper = await f.read()
            
            if THEOREMS_APPENDIX_START not in paper or THEOREMS_APPENDIX_END not in paper:
                logger.warning(
                    "Appendix markers missing - cannot append theorem entry. "
                    "Caller should call ensure_markers_intact() first."
                )
                return False
            
            body_text, appendix_body = self._extract_body_and_appendix(paper)
            
            # First real entry replaces the empty placeholder
            if not appendix_body or appendix_body.strip() == APPENDIX_EMPTY_PLACEHOLDER:
                new_appendix_body = entry
            else:
                new_appendix_body = f"{appendix_body}\n\n{entry}"
            
            # Reconstruct paper using the bracketed appendix block in the same
            # slot it currently occupies. Simplest correct approach: locate
            # the original bracket span and swap it out.
            start_idx = paper.find(THEOREMS_APPENDIX_START)
            end_idx = paper.find(THEOREMS_APPENDIX_END, start_idx)
            if start_idx < 0 or end_idx < 0:
                # Defensive: should not happen given the earlier guard.
                logger.error("Appendix markers disappeared between checks")
                return False
            end_idx += len(THEOREMS_APPENDIX_END)
            
            new_block = self._build_appendix_block(new_appendix_body)
            new_paper = paper[:start_idx] + new_block + paper[end_idx:]
            
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(new_paper)
            
            self.version += 1
            final_content = new_paper
            logger.info(
                f"Appended theorem to appendix (version {self.version}, "
                f"appendix now {len(new_appendix_body)} chars)"
            )
        
        # Trigger re-chunking callback OUTSIDE the lock to avoid deadlock
        if final_content and self.rechunk_callback:
            try:
                await self.rechunk_callback(final_content)
            except Exception as e:
                logger.error(f"Re-chunking callback failed after appendix append: {e}")
        
        return True
    
    async def ensure_placeholders_exist(self) -> bool:
        """
        Ensure placeholders exist in the paper when resuming from an existing file.
        
        Papers created before the placeholder system or from older code versions
        may not have placeholders. This causes "old_string not found" failures
        when the model tries to use placeholder text as old_string.
        
        This method:
        1. Checks if ABSTRACT_PLACEHOLDER, INTRO_PLACEHOLDER, CONCLUSION_PLACEHOLDER exist
        2. If any are missing, attempts to reconstruct the paper with placeholders
        3. Only adds MISSING placeholders - doesn't duplicate existing ones
        
        Returns:
            True if placeholders were added, False if they already existed
        """
        async with self._lock:
            if not self.file_path.exists():
                logger.warning("Cannot ensure placeholders: paper file does not exist")
                return False
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                paper = await f.read()
            
            if not paper.strip():
                logger.info("Paper is empty - no placeholders needed")
                return False
            
            # Check which placeholders are missing
            has_abstract_placeholder = ABSTRACT_PLACEHOLDER in paper
            has_intro_placeholder = INTRO_PLACEHOLDER in paper
            has_conclusion_placeholder = CONCLUSION_PLACEHOLDER in paper
            has_anchor = PAPER_ANCHOR in paper
            has_appendix_start = THEOREMS_APPENDIX_START in paper
            has_appendix_end = THEOREMS_APPENDIX_END in paper
            
            # Check for actual section content (not placeholders)
            # Use flexible patterns to detect if sections have been written
            # CRITICAL: Must distinguish between real content and fake placeholders inserted by model
            import re
            
            # Helper function to check if section has REAL content (not just a fake placeholder)
            def has_real_section_content(section_pattern: str, paper_text: str) -> bool:
                """Check if section exists with real content, not just fake placeholder text."""
                match = re.search(section_pattern, paper_text, re.IGNORECASE | re.MULTILINE)
                if not match:
                    return False
                
                # Get sample for keyword detection (300 chars)
                after_header_sample = paper_text[match.end():match.end() + 300].strip()
                
                # Get FULL content length to check if substantial
                full_content_after = paper_text[match.end():].strip()
                
                fake_placeholder_indicators = [
                    'will be replaced',
                    'to be written', 
                    'placeholder',
                    'this placeholder'
                ]
                has_placeholder_keywords = any(phrase in after_header_sample.lower() for phrase in fake_placeholder_indicators)
                
                # Decision logic:
                # - If FULL content >300 chars: REAL (substantial, keywords don't matter)
                # - If <300 chars WITH keywords: FAKE (short placeholder-style)
                # - If <300 chars NO keywords: REAL if >50 chars
                if len(full_content_after) > 300:
                    return True  # Substantial content is always real
                elif has_placeholder_keywords:
                    return False  # Short with keywords = fake
                else:
                    return len(after_header_sample) > 50  # No keywords, check substance
            
            has_abstract_content = has_real_section_content(
                r'^(?:Abstract\s*|\\(?:section|chapter)\*?\{Abstract\}|\\begin\{abstract\})\s*$',
                paper,
            )
            has_intro_content = has_real_section_content(
                r'^(?:(?:I\.?\s+)?Introduction|\\(?:section|chapter)\*?\{(?:I\.?\s*)?Introduction\})\s*$',
                paper,
            )
            has_conclusion_content = has_real_section_content(
                r'^(?:(?:[IVXLCDM]+\.?\s+|\d+\.?\s+)?Conclusion|\\(?:section|chapter)\*?\{Conclusion\})\s*$',
                paper,
            )
            
            # Real section content supersedes its placeholder. If both are
            # present, rebuild below to strip the stale placeholder.
            if (((has_abstract_placeholder or has_abstract_content) and not (has_abstract_placeholder and has_abstract_content)) and
                    ((has_intro_placeholder or has_intro_content) and not (has_intro_placeholder and has_intro_content)) and
                    ((has_conclusion_placeholder or has_conclusion_content) and not (has_conclusion_placeholder and has_conclusion_content)) and
                    has_appendix_start and has_appendix_end):
                logger.info("Placeholders check: all sections present and appendix intact")
                return False
            
            logger.info(
                f"Placeholder check - abstract: {has_abstract_placeholder}/{has_abstract_content}, "
                f"intro: {has_intro_placeholder}/{has_intro_content}, "
                f"conclusion: {has_conclusion_placeholder}/{has_conclusion_content}, "
                f"appendix: {has_appendix_start}/{has_appendix_end}"
            )
            
            # Need to add missing placeholders. Preserve any existing appendix
            # body content while we rebuild the skeleton.
            body_content, appendix_body = self._extract_body_and_appendix(paper)
            
            if not body_content:
                logger.warning("No body content found - cannot add placeholders to empty paper")
                return False
            
            # Reconstruct paper with placeholders for missing sections
            # Add placeholder for ANY section that hasn't been written yet (actual content doesn't exist)
            # This preserves placeholders during repair, even when repairing for other missing markers
            new_paper_parts = []
            
            # Abstract at top (if section not written yet - need placeholder)
            if not has_abstract_content:
                new_paper_parts.append(ABSTRACT_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Introduction after abstract (if section not written yet - need placeholder)
            if not has_intro_content:
                new_paper_parts.append(INTRO_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Body content
            new_paper_parts.append(body_content)
            new_paper_parts.append("")
            
            # Conclusion before anchor (if section not written yet - need placeholder)
            if not has_conclusion_content:
                new_paper_parts.append(CONCLUSION_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Theorems Appendix (preserve existing entries if any, otherwise
            # insert the empty placeholder line inside the bracket pair)
            new_paper_parts.append(self._build_appendix_block(appendix_body))
            new_paper_parts.append("")
            
            # Anchor at end
            new_paper_parts.append(PAPER_ANCHOR)
            
            new_paper = '\n'.join(new_paper_parts)
            
            # Write updated paper
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(new_paper)
            
            self.version += 1
            logger.info(f"Added missing placeholders to paper (version {self.version})")
            
            return True
    
    async def ensure_markers_intact(self) -> bool:
        """
        Lightweight check to ensure all required markers exist in the paper.
        
        Called BEFORE every old_string match operation to prevent failures
        caused by missing markers during normal operation.
        
        This is simpler than ensure_placeholders_exist() - it just checks
        for marker presence and adds them if missing, without complex
        resume logic.
        
        Returns:
            True if markers were repaired, False if they were already intact
        """
        async with self._lock:
            if not self.file_path.exists():
                return False
            
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                paper = await f.read()
            
            if not paper.strip():
                return False
            
            # Quick check: if all markers exist, no repair needed
            has_abstract_placeholder = ABSTRACT_PLACEHOLDER in paper
            has_intro_placeholder = INTRO_PLACEHOLDER in paper
            has_conclusion_placeholder = CONCLUSION_PLACEHOLDER in paper
            has_anchor = PAPER_ANCHOR in paper
            has_appendix_start = THEOREMS_APPENDIX_START in paper
            has_appendix_end = THEOREMS_APPENDIX_END in paper
            
            # Check for actual section content (not placeholders)
            # CRITICAL: Must distinguish between real content and fake placeholders inserted by model
            import re
            
            # Helper function to check if section has REAL content (not just a fake placeholder)
            def has_real_section_content(section_pattern: str, paper_text: str) -> bool:
                """Check if section exists with real content, not just fake placeholder text."""
                match = re.search(section_pattern, paper_text, re.IGNORECASE | re.MULTILINE)
                if not match:
                    return False
                
                # Get sample for keyword detection (300 chars)
                after_header_sample = paper_text[match.end():match.end() + 300].strip()
                
                # Get FULL content length to check if substantial
                full_content_after = paper_text[match.end():].strip()
                
                fake_placeholder_indicators = [
                    'will be replaced',
                    'to be written', 
                    'placeholder',
                    'this placeholder'
                ]
                has_placeholder_keywords = any(phrase in after_header_sample.lower() for phrase in fake_placeholder_indicators)
                
                # Decision logic:
                # - If FULL content >300 chars: REAL (substantial, keywords don't matter)
                # - If <300 chars WITH keywords: FAKE (short placeholder-style)
                # - If <300 chars NO keywords: REAL if >50 chars
                if len(full_content_after) > 300:
                    return True  # Substantial content is always real
                elif has_placeholder_keywords:
                    return False  # Short with keywords = fake
                else:
                    return len(after_header_sample) > 50  # No keywords, check substance
            
            has_abstract_content = has_real_section_content(
                r'^(?:Abstract\s*|\\(?:section|chapter)\*?\{Abstract\}|\\begin\{abstract\})\s*$',
                paper,
            )
            has_intro_content = has_real_section_content(
                r'^(?:(?:I\.?\s+)?Introduction|\\(?:section|chapter)\*?\{(?:I\.?\s*)?Introduction\})\s*$',
                paper,
            )
            has_conclusion_content = has_real_section_content(
                r'^(?:(?:[IVXLCDM]+\.?\s+|\d+\.?\s+)?Conclusion|\\(?:section|chapter)\*?\{Conclusion\})\s*$',
                paper,
            )
            
            # Real section content supersedes its placeholder. If both are
            # present, rebuild below to strip the stale placeholder.
            if ((has_abstract_placeholder or has_abstract_content) and not (has_abstract_placeholder and has_abstract_content)) and \
               ((has_intro_placeholder or has_intro_content) and not (has_intro_placeholder and has_intro_content)) and \
               ((has_conclusion_placeholder or has_conclusion_content) and not (has_conclusion_placeholder and has_conclusion_content)) and \
               has_anchor and has_appendix_start and has_appendix_end:
                return False
            
            # Need to repair - preserve body + appendix body across the rebuild
            body_content, appendix_body = self._extract_body_and_appendix(paper)
            
            if not body_content:
                # Empty paper - just ensure anchor exists
                if not has_anchor:
                    async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                        await f.write(PAPER_ANCHOR)
                    self.version += 1
                    return True
                return False
            
            # Reconstruct paper with missing markers
            new_paper_parts = []
            
            # Abstract at top (if section not written yet - need placeholder)
            if not has_abstract_content:
                new_paper_parts.append(ABSTRACT_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Introduction (if section not written yet - need placeholder)
            if not has_intro_content:
                new_paper_parts.append(INTRO_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Body content
            new_paper_parts.append(body_content)
            new_paper_parts.append("")
            
            # Conclusion (if section not written yet - need placeholder)
            if not has_conclusion_content:
                new_paper_parts.append(CONCLUSION_PLACEHOLDER)
                new_paper_parts.append("")
            
            # Theorems Appendix (preserve existing entries, default otherwise)
            new_paper_parts.append(self._build_appendix_block(appendix_body))
            new_paper_parts.append("")
            
            # Anchor at end
            new_paper_parts.append(PAPER_ANCHOR)
            
            new_paper = '\n'.join(new_paper_parts)
            
            # Write repaired paper
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                await f.write(new_paper)
            
            self.version += 1
            logger.info(f"Repaired missing markers in paper (version {self.version})")
            
            return True


# Global paper memory instance
paper_memory = PaperMemory()

