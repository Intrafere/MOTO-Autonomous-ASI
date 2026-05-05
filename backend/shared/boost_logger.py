"""
Boost Logger - Logs raw API outputs from boost (OpenRouter) API calls.
Stores logs in a persistent file so boost-routed calls can be merged into the
main API call log view.
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


class BoostLogger:
    """
    Logger for boost API call outputs.
    Stores logs in data/boost_api_log.txt with JSON entries.
    """
    
    MAX_LOG_ENTRIES = 500  # Maximum entries to keep in log
    
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
        logger.info("BoostLogger initialized")
    
    def _ensure_log_file(self) -> None:
        """Ensure the log file and directory exist."""
        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not log_path.exists():
            log_path.write_text("")

    def _get_log_path(self) -> Path:
        """Return the instance-scoped boost log path."""
        return Path(system_config.data_dir) / "boost_api_log.txt"
    
    async def log_boost_call(
        self,
        task_id: str,
        role_id: str,
        model: str,
        prompt_preview: str,
        response_content: str,
        tokens_used: Optional[int] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        boost_mode: str = "unknown"  # "next_count", "category", "task_id"
    ) -> None:
        """
        Log a boost API call.
        
        Args:
            task_id: Task ID for the call
            role_id: Role identifier (e.g., "aggregator_submitter_1")
            model: Model used (OpenRouter model ID)
            prompt_preview: First 500 chars of the prompt
            response_content: Full response content (will be truncated for preview)
            tokens_used: Number of tokens used (if available)
            duration_ms: Duration of the call in milliseconds
            success: Whether the call succeeded
            error: Error message if call failed
            boost_mode: Which boost mode triggered this ("next_count", "category", "task_id")
        """
        async with self._lock:
            try:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_id,
                    "role_id": role_id,
                    "model": model,
                    "boost_mode": boost_mode,
                    "prompt_preview": prompt_preview[:500] if prompt_preview else "",
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
                
                logger.debug(f"Logged boost call: task={task_id}, model={model}, success={success}")
                
                # Trim log if too large
                await self._trim_log_if_needed()
                
            except Exception as e:
                logger.error(f"Failed to log boost call: {e}")
    
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
                logger.debug(f"Trimmed boost log to {self.MAX_LOG_ENTRIES} entries")
                
        except Exception as e:
            logger.error(f"Failed to trim boost log: {e}")
    
    async def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent boost API call logs.
        
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
                logger.error(f"Failed to get boost logs: {e}")
                return []
    
    async def get_log_entry(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific log entry by index (0 = most recent).
        
        Args:
            index: Index of the log entry
            
        Returns:
            Log entry dict or None if not found
        """
        logs = await self.get_logs(limit=index + 1)
        if index < len(logs):
            return logs[index]
        return None
    
    async def clear_logs(self) -> None:
        """Clear all boost API logs."""
        async with self._lock:
            try:
                with open(self._get_log_path(), "w", encoding="utf-8") as f:
                    f.write("")
                logger.info("Boost logs cleared")
            except Exception as e:
                logger.error(f"Failed to clear boost logs: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about boost API calls.
        
        Returns:
            Dict with statistics (total calls, success rate, etc.)
        """
        logs = await self.get_logs(limit=self.MAX_LOG_ENTRIES)
        
        if not logs:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "by_mode": {},
                "by_model": {}
            }
        
        successful = sum(1 for log in logs if log.get("success", True))
        failed = len(logs) - successful
        
        # Count by boost mode
        by_mode = {}
        for log in logs:
            mode = log.get("boost_mode", "unknown")
            by_mode[mode] = by_mode.get(mode, 0) + 1
        
        # Count by model
        by_model = {}
        for log in logs:
            model = log.get("model", "unknown")
            by_model[model] = by_model.get(model, 0) + 1
        
        return {
            "total_calls": len(logs),
            "successful_calls": successful,
            "failed_calls": failed,
            "success_rate": successful / len(logs) if logs else 0.0,
            "by_mode": by_mode,
            "by_model": by_model
        }


# Global singleton instance
boost_logger = BoostLogger()

