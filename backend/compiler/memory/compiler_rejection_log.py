"""
Compiler rejection/acceptance log manager.
Maintains last 10 rejections and acceptances (appended as text, not embedded).
"""
import aiofiles
import asyncio
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import logging

from backend.shared.config import system_config
from backend.shared.models import CompilerValidationResult

logger = logging.getLogger(__name__)


class CompilerRejectionLog:
    """
    Manages last 10 rejections and acceptances for compiler.
    - Appended as text (not embedded)
    - Helps submitter learn patterns
    - Thread-safe operations
    """
    
    def __init__(self):
        self.rejections_file = Path(system_config.compiler_rejections_file)
        self.acceptances_file = Path(system_config.compiler_acceptances_file)
        self.declines_file = Path(system_config.compiler_declines_file)
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # In-memory caches
        self.rejections: List[Dict] = []
        self.acceptances: List[Dict] = []
        self.declines: List[Dict] = []
        self.max_entries = 10
    
    async def initialize(self) -> None:
        """Initialize rejection/acceptance logs."""
        async with self._lock:
            if self._initialized:
                return
            
            # Create files if don't exist
            self.rejections_file.parent.mkdir(parents=True, exist_ok=True)
            self.acceptances_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.rejections_file.exists():
                async with aiofiles.open(self.rejections_file, 'w', encoding='utf-8') as f:
                    await f.write("")
            
            if not self.acceptances_file.exists():
                async with aiofiles.open(self.acceptances_file, 'w', encoding='utf-8') as f:
                    await f.write("")
            
            if not self.declines_file.exists():
                async with aiofiles.open(self.declines_file, 'w', encoding='utf-8') as f:
                    await f.write("")
            
            # Load existing logs
            await self._load_logs()
            
            self._initialized = True
            logger.info("Compiler rejection/acceptance logs initialized")
    
    async def _load_logs(self) -> None:
        """Load existing logs from files."""
        # Load rejections
        async with aiofiles.open(self.rejections_file, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        if content.strip():
            self.rejections = self._parse_log_entries(content)
        
        # Load acceptances
        async with aiofiles.open(self.acceptances_file, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        if content.strip():
            self.acceptances = self._parse_log_entries(content)
        
        # Load declines
        async with aiofiles.open(self.declines_file, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        if content.strip():
            self.declines = self._parse_log_entries(content)
    
    def _parse_log_entries(self, content: str) -> List[Dict]:
        """Parse log file content into entries."""
        entries = []
        blocks = content.strip().split('\n\n---\n\n')
        
        for block in blocks:
            if block.strip():
                entries.append({'text': block.strip()})
        
        # Keep only last 10
        return entries[-self.max_entries:]
    
    async def add_rejection(self, result: CompilerValidationResult, mode: str, submission_content: str = "") -> None:
        """
        Add a rejection to the log with enhanced formatting for better model feedback.
        
        Args:
            result: Validation result
            mode: Submission mode
            submission_content: Optional submission content for preview
        """
        async with self._lock:
            # Determine which criterion failed
            failed_criteria = []
            if not result.coherence_check:
                failed_criteria.append("Coherence")
            if not result.rigor_check:
                failed_criteria.append("Rigor")
            if not result.placement_check:
                # Distinguish between pre-validation (exact string match) and LLM validation (placement context)
                if result.validation_stage == "pre-validation":
                    failed_criteria.append("Exact String Match")
                else:
                    failed_criteria.append("Placement Context")
            
            failure_reason = " & ".join(failed_criteria) if failed_criteria else "Validation logic"
            
            # Create enhanced log entry with prominent formatting
            timestamp = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Build failure criteria section
            # Label placement check based on validation stage
            placement_label = "Exact String Match" if result.validation_stage == "pre-validation" else "Placement Context"
            criteria_status = f"""FAILURE CRITERION: {failure_reason}
- Coherence: {'✓' if result.coherence_check else '✗'}
- Rigor: {'✓' if result.rigor_check else '✗'}
- {placement_label}: {'✓' if result.placement_check else '✗'}"""
            
            # Add submission preview if provided
            preview_section = ""
            if submission_content:
                preview = submission_content[:300] + "..." if len(submission_content) > 300 else submission_content
                preview_section = f"\n\nSUBMISSION PREVIEW (first 300 chars):\n{preview}"
            
            # Build actionable guidance based on what failed
            what_to_fix = ""
            if not result.placement_check:
                # Use explicit validation_stage field instead of string matching
                if result.validation_stage == "pre-validation":
                    # Pre-validation failure - exact string matching issue
                    # The reasoning already contains detailed fix_suggestion, so keep this brief
                    what_to_fix = "\n\nWHAT TO FIX:\nThe old_string must exist EXACTLY (verbatim) and UNIQUELY in the document. Include more surrounding context (3-5 lines) to make the match unique. See 'VALIDATOR REASONING' above for specific details."
                else:
                    # LLM validation failure - placement context issue
                    what_to_fix = "\n\nWHAT TO FIX:\nThe content doesn't fit naturally at this location in the document. Consider:\n- Does this content logically follow from what's already written?\n- Is prerequisite material established first?\n- Does it align with the outline structure for this section?\n- Would it fit better elsewhere in the document?\nNote: The old_string was verified by pre-validation, so focus on content appropriateness."
            elif not result.coherence_check:
                what_to_fix = "\n\nWHAT TO FIX:\nEnsure the content flows naturally with existing document sections. Check for grammatical errors and maintain holistic coherence."
            elif not result.rigor_check:
                what_to_fix = "\n\nWHAT TO FIX:\nEnsure all claims meet the correctness standard appropriate to their domain and claim type. Mathematical claims require sound derivation, proof, or explicit assumptions; factual, logical, technical, empirical, and methodological claims require appropriate support and must avoid unfounded claims or logical fallacies."
            
            entry_text = f"""========================================
🚫 REJECTED BECAUSE: {failure_reason}
========================================
Mode: {mode.upper()}
Timestamp: {timestamp}
Submission ID: {result.submission_id}

{criteria_status}

VALIDATOR REASONING:
{result.reasoning}

SUMMARY:
{result.summary}{preview_section}{what_to_fix}"""
            
            # Add to in-memory cache
            self.rejections.append({'text': entry_text})
            
            # Keep only last 10
            if len(self.rejections) > self.max_entries:
                self.rejections = self.rejections[-self.max_entries:]
            
            # Write to file
            await self._write_rejections()
            
            logger.info(f"Added rejection to log (mode: {mode})")
    
    async def add_acceptance(self, submission_id: str, mode: str, content_preview: str) -> None:
        """
        Add an acceptance to the log.
        
        Args:
            submission_id: Submission ID
            mode: Submission mode
            content_preview: Preview of accepted content (first 500 chars)
        """
        async with self._lock:
            # Create log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            preview = content_preview[:500] + "..." if len(content_preview) > 500 else content_preview
            
            entry_text = f"""[ACCEPTANCE - {mode.upper()}] {timestamp}
Submission ID: {submission_id}
Content Preview: {preview}"""
            
            # Add to in-memory cache
            self.acceptances.append({'text': entry_text})
            
            # Keep only last 10
            if len(self.acceptances) > self.max_entries:
                self.acceptances = self.acceptances[-self.max_entries:]
            
            # Write to file
            await self._write_acceptances()
            
            logger.info(f"Added acceptance to log (mode: {mode})")
    
    async def add_decline(self, mode: str, reasoning: str) -> None:
        """
        Add a decline to the log (when submitter chooses not to submit).
        
        Args:
            mode: Submission mode
            reasoning: Why submission was declined
        """
        async with self._lock:
            # Create log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            entry_text = f"""[DECLINED - {mode.upper()}] {timestamp}
Reasoning: {reasoning}"""
            
            # Add to in-memory cache
            self.declines.append({'text': entry_text})
            
            # Keep only last 10
            if len(self.declines) > self.max_entries:
                self.declines = self.declines[-self.max_entries:]
            
            # Write to file
            await self._write_declines()
            
            logger.info(f"Added decline to log (mode: {mode})")
    
    async def _write_rejections(self) -> None:
        """Write rejections to file."""
        content = '\n\n---\n\n'.join([entry['text'] for entry in self.rejections])
        async with aiofiles.open(self.rejections_file, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    async def _write_acceptances(self) -> None:
        """Write acceptances to file."""
        content = '\n\n---\n\n'.join([entry['text'] for entry in self.acceptances])
        async with aiofiles.open(self.acceptances_file, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    async def _write_declines(self) -> None:
        """Write declines to file."""
        content = '\n\n---\n\n'.join([entry['text'] for entry in self.declines])
        async with aiofiles.open(self.declines_file, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    async def get_rejections_text(self) -> str:
        """Get rejections as text for context injection."""
        async with self._lock:
            if not self.rejections:
                return ""
            return '\n\n---\n\n'.join([entry['text'] for entry in self.rejections])
    
    async def get_acceptances_text(self) -> str:
        """Get acceptances as text for context injection."""
        async with self._lock:
            if not self.acceptances:
                return ""
            return '\n\n---\n\n'.join([entry['text'] for entry in self.acceptances])
    
    async def get_declines_text(self) -> str:
        """Get declines as text for context injection."""
        async with self._lock:
            if not self.declines:
                return ""
            return '\n\n---\n\n'.join([entry['text'] for entry in self.declines])


# Global rejection log instance
compiler_rejection_log = CompilerRejectionLog()

