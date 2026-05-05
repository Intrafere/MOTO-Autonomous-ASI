"""
Autonomous API Logger - Logs all API calls during autonomous research mode.
Stores logs in a persistent file for viewing in the Autonomous Logs tab.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.shared.config import system_config

logger = logging.getLogger(__name__)


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
        self._ensure_log_file()
        logger.info("AutonomousAPILogger initialized")
    
    def _ensure_log_file(self) -> None:
        """Ensure the log file and directory exist."""
        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not log_path.exists():
            log_path.write_text("")

    def _get_log_path(self) -> Path:
        """Return the instance-scoped autonomous API log path."""
        return Path(system_config.data_dir) / "auto_api_log.txt"
    
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
        phase: str = "unknown"
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
        """
        async with self._lock:
            try:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_id,
                    "role_id": role_id,
                    "model": model,
                    "provider": provider,
                    "phase": phase,
                    "prompt_preview": prompt[:1000] if prompt else "",
                    "prompt_full": prompt,
                    "response_preview": response_content[:2000] if response_content else "",
                    "response_full": response_content,
                    "tokens_used": tokens_used,
                    "duration_ms": duration_ms,
                    "success": success,
                    "error": error
                }
                
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
    
    async def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent autonomous API call logs.
        
        Args:
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entries (most recent first)
        """
        async with self._lock:
            try:
                log_path = self._get_log_path()
                if not os.path.exists(log_path):
                    return []
                
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                logs = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            log_entry = json.loads(line)
                            logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue
                
                # Return most recent first, limited
                logs.reverse()
                return logs[:limit]
                
            except Exception as e:
                logger.error(f"Failed to get autonomous API logs: {e}")
                return []
    
    async def clear_logs(self) -> None:
        """Clear all autonomous API logs."""
        async with self._lock:
            try:
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

