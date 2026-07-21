"""
Autonomous API Logger - Logs all API calls during autonomous research mode.
Stores logs in a persistent file for viewing in the Autonomous Logs tab.
"""
import asyncio
import hashlib
import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text

logger = logging.getLogger(__name__)


def _payload_metadata(value: str, preview_chars: int) -> Dict[str, Any]:
    """Return safe log metadata for a prompt/response payload."""
    text = value or ""
    preview = redact_log_text(text, preview_chars)
    return {
        "preview": preview,
        "size": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest() if text else "",
    }


class AutonomousAPILogger:
    """
    Logger for autonomous research API call outputs.
    Stores logs in data/auto_api_log.txt with JSON entries.
    """
    
    MAX_LOG_ENTRIES = 1000  # Maximum entries to keep in log
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._prepared_root_identity = None
        logger.info("AutonomousAPILogger initialized")

    def _prepare_active_root(self) -> None:
        identity = system_config.runtime_root_identity()
        if self._prepared_root_identity == identity:
            return
        self._ensure_log_file()
        self._scrub_persisted_full_payloads()
        self._prepared_root_identity = identity
    
    def _ensure_log_file(self) -> None:
        """Ensure the log file and directory exist."""
        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not log_path.exists():
            log_path.write_text("")

    def _get_log_path(self) -> Path:
        """Return the instance-scoped autonomous API log path."""
        return Path(system_config.data_dir) / "auto_api_log.txt"

    def _scrub_persisted_full_payloads(self) -> None:
        """Remove legacy full prompt/response bodies from the on-disk JSONL log."""
        log_path = self._get_log_path()
        if not log_path.exists():
            return

        changed = False
        scrubbed_lines: List[str] = []

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    scrubbed_lines.append(line)
                    continue

                original_entry = dict(entry)
                prompt_full = str(entry.pop("prompt_full", "") or "")
                response_full = str(entry.pop("response_full", "") or "")
                prompt_source = prompt_full or str(entry.get("prompt_preview") or "")
                response_source = response_full or str(entry.get("response_preview") or "")

                if prompt_source:
                    prompt_meta = _payload_metadata(prompt_source, 1000)
                    entry["prompt_preview"] = prompt_meta["preview"]
                    entry["prompt_size"] = int(entry.get("prompt_size") or prompt_meta["size"])
                    entry.setdefault("prompt_sha256", prompt_meta["sha256"])
                if response_source:
                    response_meta = _payload_metadata(response_source, 2000)
                    entry["response_preview"] = response_meta["preview"]
                    entry["response_size"] = int(entry.get("response_size") or response_meta["size"])
                    entry.setdefault("response_sha256", response_meta["sha256"])

                entry["prompt_redacted"] = True
                entry["response_redacted"] = True
                entry["has_full_prompt"] = False
                entry["has_full_response"] = False
                if entry.get("error"):
                    entry["error"] = redact_log_text(entry["error"], 1000)

                if prompt_full or response_full or entry != original_entry:
                    changed = True
                scrubbed_lines.append(json.dumps(entry) + "\n")

            if changed:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.writelines(scrubbed_lines)
                logger.info("Scrubbed legacy full prompt/response payloads from autonomous API log")
        except Exception as e:
            logger.warning(f"Failed to scrub legacy autonomous API log payloads: {e}")
    
    async def log_api_call(
        self,
        task_id: str,
        role_id: str,
        model: str,
        provider: str,
        prompt: str,
        response_content: str,
        tokens_used: Optional[int] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        phase: str = "unknown",
        workflow: str = "autonomous",
    ) -> None:
        """
        Log an autonomous research API call.
        
        Args:
            task_id: Task ID for the call
            role_id: Role identifier (e.g., "topic_selector", "aggregator_submitter_1")
            model: Model used
            provider: "lm_studio" or "openrouter"
            prompt: Full prompt text
            response_content: Full response content
            tokens_used: Number of tokens used (if available)
            duration_ms: Duration of the call in milliseconds
            success: Whether the call succeeded
            error: Error message if call failed
            phase: Research phase ("topic_selection", "brainstorm", "paper_compilation", "tier3")
            workflow: Workflow namespace for this call ("autonomous" or "leanoj")
        """
        async with self._lock:
            try:
                self._prepare_active_root()
                prompt_meta = _payload_metadata(prompt, 1000)
                response_meta = _payload_metadata(response_content, 2000)
                store_full_payloads = bool(system_config.api_log_store_full_payloads)

                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_id,
                    "role_id": role_id,
                    "model": model,
                    "provider": provider,
                    "phase": phase,
                    "workflow": workflow,
                    "prompt_preview": prompt_meta["preview"],
                    "prompt_size": prompt_meta["size"],
                    "prompt_sha256": prompt_meta["sha256"],
                    "prompt_redacted": not store_full_payloads,
                    "has_full_prompt": store_full_payloads and bool(prompt),
                    "response_preview": response_meta["preview"],
                    "response_size": response_meta["size"],
                    "response_sha256": response_meta["sha256"],
                    "response_redacted": not store_full_payloads,
                    "has_full_response": store_full_payloads and bool(response_content),
                    "tokens_used": tokens_used,
                    "duration_ms": duration_ms,
                    "success": success,
                    "error": redact_log_text(error, 1000)
                }
                if store_full_payloads:
                    log_entry["prompt_full"] = prompt
                    log_entry["response_full"] = response_content
                
                # Append to log file
                with open(self._get_log_path(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                logger.debug(f"Logged autonomous API call: task={task_id}, model={model}, success={success}, phase={phase}")
                
                # Trim log if too large
                await self._trim_log_if_needed()
                
            except Exception as e:
                logger.error(f"Failed to log autonomous API call: {e}")
    
    async def _trim_log_if_needed(self) -> None:
        """Trim log file if it exceeds MAX_LOG_ENTRIES."""
        try:
            with open(self._get_log_path(), "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if len(lines) > self.MAX_LOG_ENTRIES:
                # Keep only the most recent entries
                lines = lines[-self.MAX_LOG_ENTRIES:]
                with open(self._get_log_path(), "w", encoding="utf-8") as f:
                    f.writelines(lines)
                logger.debug(f"Trimmed autonomous API log to {self.MAX_LOG_ENTRIES} entries")
                
        except Exception as e:
            logger.error(f"Failed to trim autonomous API log: {e}")
    
    async def get_logs(self, limit: int = 100, include_full: bool = True) -> List[Dict[str, Any]]:
        """
        Get recent autonomous API call logs.
        
        Args:
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entries (most recent first)
        """
        async with self._lock:
            try:
                self._prepare_active_root()
                log_path = self._get_log_path()
                if not os.path.exists(log_path):
                    return []
                
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = deque(f, maxlen=max(1, limit))
                
                logs = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            log_entry = json.loads(line)
                            if not include_full or not system_config.api_log_store_full_payloads:
                                prompt_full = str(log_entry.pop("prompt_full", "") or "")
                                response_full = str(log_entry.pop("response_full", "") or "")
                                log_entry["prompt_size"] = int(log_entry.get("prompt_size") or len(prompt_full))
                                log_entry["response_size"] = int(log_entry.get("response_size") or len(response_full))
                                log_entry["has_full_prompt"] = False
                                log_entry["has_full_response"] = False
                                if prompt_full and not log_entry.get("prompt_sha256"):
                                    log_entry["prompt_sha256"] = hashlib.sha256(prompt_full.encode("utf-8", errors="replace")).hexdigest()
                                if response_full and not log_entry.get("response_sha256"):
                                    log_entry["response_sha256"] = hashlib.sha256(response_full.encode("utf-8", errors="replace")).hexdigest()
                            logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue
                
                # Return most recent first, limited
                logs.reverse()
                return logs[:limit]
                
            except Exception as e:
                logger.error(f"Failed to get autonomous API logs: {e}")
                return []

    @staticmethod
    def _entry_workflow(entry: Dict[str, Any]) -> str:
        workflow = str(entry.get("workflow") or "").strip().lower()
        if workflow:
            return workflow

        role_id = str(entry.get("role_id") or "")
        task_id = str(entry.get("task_id") or "")
        if role_id.startswith("leanoj_") or task_id.startswith("leanoj_"):
            return "leanoj"
        return "autonomous"

    async def clear_logs(self, workflow: Optional[str] = None) -> None:
        """Clear autonomous API logs, optionally scoped to one workflow."""
        async with self._lock:
            try:
                self._prepare_active_root()
                if workflow:
                    log_path = self._get_log_path()
                    if not os.path.exists(log_path):
                        return

                    with open(log_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    retained_lines: List[str] = []
                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            entry = json.loads(stripped)
                        except json.JSONDecodeError:
                            retained_lines.append(line)
                            continue
                        if self._entry_workflow(entry) != workflow:
                            retained_lines.append(line)

                    with open(log_path, "w", encoding="utf-8") as f:
                        f.writelines(retained_lines)
                    logger.info("Autonomous API logs cleared for workflow %s", workflow)
                    return

                with open(self._get_log_path(), "w", encoding="utf-8") as f:
                    f.write("")
                logger.info("Autonomous API logs cleared")
            except Exception as e:
                logger.error(f"Failed to clear autonomous API logs: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about autonomous API calls.
        
        Returns:
            Dict with statistics (total calls, success rate, by phase, by model, etc.)
        """
        logs = await self.get_logs(limit=self.MAX_LOG_ENTRIES)
        
        if not logs:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "by_phase": {},
                "by_model": {},
                "by_provider": {}
            }
        
        successful = sum(1 for log in logs if log.get("success", True))
        failed = len(logs) - successful
        
        # Count by phase
        by_phase = {}
        for log in logs:
            phase = log.get("phase", "unknown")
            by_phase[phase] = by_phase.get(phase, 0) + 1
        
        # Count by model
        by_model = {}
        for log in logs:
            model = log.get("model", "unknown")
            by_model[model] = by_model.get(model, 0) + 1
        
        # Count by provider
        by_provider = {}
        for log in logs:
            provider = log.get("provider", "unknown")
            by_provider[provider] = by_provider.get(provider, 0) + 1
        
        return {
            "total_calls": len(logs),
            "successful_calls": successful,
            "failed_calls": failed,
            "success_rate": successful / len(logs) if logs else 0.0,
            "by_phase": by_phase,
            "by_model": by_model,
            "by_provider": by_provider
        }


# Global singleton instance
autonomous_api_logger = AutonomousAPILogger()

