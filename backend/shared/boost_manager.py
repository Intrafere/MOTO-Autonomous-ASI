"""
Boost Manager - Singleton for managing API boost configuration and task selection.
Tracks which workflow tasks should use the boost OpenRouter API key.

Supports three boost modes:
1. Boost Next X Calls - Counter-based, applies to next X API calls regardless of task ID
2. Category Boost - Role-based, boosts all calls matching a role prefix (e.g., all Submitter 1 calls)
3. Per-task Toggle - Task ID based (may have ID matching issues with workflow predictions)
"""
import asyncio
import logging
from typing import Optional, Set, Callable, Any, Dict, List

from backend.shared.models import BoostConfig

logger = logging.getLogger(__name__)


# Category prefixes for different roles
CATEGORY_PREFIXES = {
    # Aggregator
    "agg_sub1": "Submitter 1",
    "agg_sub2": "Submitter 2",
    "agg_sub3": "Submitter 3",
    "agg_sub4": "Submitter 4",
    "agg_sub5": "Submitter 5",
    "agg_sub6": "Submitter 6",
    "agg_sub7": "Submitter 7",
    "agg_sub8": "Submitter 8",
    "agg_sub9": "Submitter 9",
    "agg_sub10": "Submitter 10",
    "agg_val": "Aggregator Validator",
    # Compiler
    "comp_hc": "High-Context Submitter",
    "comp_hp": "High-Param Submitter",
    "comp_val": "Compiler Validator",
    # Autonomous
    "auto_te": "Topic Explorer",
    "auto_tev": "Topic Explorer Validator",
    "auto_ts": "Topic Selector",
    "auto_tv": "Topic Validator",
    "auto_cr": "Completion Reviewer",
    "auto_rs": "Reference Selector",
    "auto_pt": "Paper Title Selector",
    "auto_prc": "Paper Redundancy Checker",
}


class BoostManager:
    """
    Singleton manager for API boost configuration.
    Manages which tasks use the boost OpenRouter model.
    
    Supports three boost modes:
    - boost_next_count: Boost the next X API calls (counter-based)
    - boosted_categories: Boost all calls for specific role categories
    - boosted_task_ids: Boost specific task IDs (legacy, may have matching issues)
    """
    
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
        
        self.boost_config: Optional[BoostConfig] = None
        self.boosted_task_ids: Set[str] = set()
        self._broadcast_callback: Optional[Callable] = None
        
        # NEW: Counter-based boost mode
        self.boost_next_count: int = 0
        
        # NEW: Category-based boost mode (role prefixes like "agg_sub1", "comp_hc")
        self.boosted_categories: Set[str] = set()
        
        self._initialized = True
        
        logger.info("BoostManager initialized")
    
    def set_broadcast_callback(self, callback: Callable) -> None:
        """Set callback for broadcasting WebSocket events."""
        self._broadcast_callback = callback
    
    async def _broadcast(self, event: str, data: Dict[str, Any] = None) -> None:
        """Broadcast an event through WebSocket."""
        if self._broadcast_callback:
            await self._broadcast_callback(event, data or {})
    
    async def set_boost_config(self, config: BoostConfig) -> None:
        """
        Set boost configuration and enable boost mode.
        
        Args:
            config: Boost configuration with API key and model
        """
        async with self._lock:
            self.boost_config = config
            provider_info = f", provider={config.boost_provider}" if config.boost_provider else " (auto-routing)"
            logger.info(
                f"Boost enabled: model={config.boost_model_id}{provider_info}, "
                f"context={config.boost_context_window}, "
                f"max_tokens={config.boost_max_output_tokens}"
            )
            
            await self._broadcast("boost_enabled", {
                "model_id": config.boost_model_id,
                "provider": config.boost_provider,
                "context_window": config.boost_context_window,
                "max_output_tokens": config.boost_max_output_tokens
            })
    
    async def clear_boost(self) -> None:
        """Disable boost mode and clear configuration."""
        async with self._lock:
            if self.boost_config:
                logger.info("Boost disabled")
                self.boost_config = None
                self.boosted_task_ids.clear()
                self.boosted_categories.clear()
                self.boost_next_count = 0
                
                await self._broadcast("boost_disabled", {})
    
    async def toggle_task_boost(self, task_id: str) -> bool:
        """
        Toggle boost for a specific task.
        
        Args:
            task_id: Task ID to toggle
            
        Returns:
            True if task is now boosted, False if unboosted
        """
        async with self._lock:
            if task_id in self.boosted_task_ids:
                self.boosted_task_ids.remove(task_id)
                boosted = False
                logger.debug(f"Task {task_id} boost disabled")
            else:
                self.boosted_task_ids.add(task_id)
                boosted = True
                logger.debug(f"Task {task_id} boost enabled")
            
            await self._broadcast("task_boost_toggled", {
                "task_id": task_id,
                "boosted": boosted
            })
            
            return boosted
    
    def is_task_boosted(self, task_id: str) -> bool:
        """
        Check if a task should use the boost (legacy method for exact task ID match).
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if task is boosted and boost is enabled
        """
        return (
            self.boost_config is not None and 
            self.boost_config.enabled and 
            task_id in self.boosted_task_ids
        )
    
    async def set_boost_next_count(self, count: int) -> None:
        """
        Set the number of next API calls to boost.
        
        Args:
            count: Number of next API calls to boost (0 to disable)
        """
        async with self._lock:
            self.boost_next_count = max(0, count)
            logger.info(f"Boost next count set to {self.boost_next_count}")
            
            await self._broadcast("boost_next_count_updated", {
                "count": self.boost_next_count
            })
    
    async def toggle_category_boost(self, category: str) -> bool:
        """
        Toggle boost for an entire category (role prefix).
        
        Args:
            category: Category prefix (e.g., "agg_sub1", "comp_hc", "agg_val")
            
        Returns:
            True if category is now boosted, False if unboosted
        """
        async with self._lock:
            if category in self.boosted_categories:
                self.boosted_categories.remove(category)
                boosted = False
                logger.info(f"Category {category} boost disabled")
            else:
                self.boosted_categories.add(category)
                boosted = True
                logger.info(f"Category {category} boost enabled")
            
            await self._broadcast("category_boost_toggled", {
                "category": category,
                "boosted": boosted,
                "all_categories": list(self.boosted_categories)
            })
            
            return boosted
    
    def _extract_role_prefix(self, task_id: str) -> str:
        """
        Extract role prefix from task ID.
        
        Examples:
            "agg_sub1_001" -> "agg_sub1"
            "comp_hc_005" -> "comp_hc"
            "auto_ts_002" -> "auto_ts"
        """
        # Split on last underscore and take everything before it
        parts = task_id.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0]
        return task_id
    
    def should_use_boost(self, task_id: str) -> bool:
        """
        Unified check for whether a task should use boost.
        
        Checks in order:
        1. Is boost enabled at all?
        2. Is boost_next_count > 0? (will be decremented after use)
        3. Is the task's category in boosted_categories?
        4. Is the exact task_id in boosted_task_ids?
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if task should use boost
        """
        # Must have boost config enabled
        if not self.boost_config or not self.boost_config.enabled:
            return False
        
        # Check boost_next_count first (counter-based mode)
        if self.boost_next_count > 0:
            return True
        
        # Check category boost (role-based mode)
        role_prefix = self._extract_role_prefix(task_id)
        if role_prefix in self.boosted_categories:
            return True
        
        # Check exact task ID (legacy per-task mode)
        if task_id in self.boosted_task_ids:
            return True
        
        return False
    
    async def consume_boost_count(self) -> None:
        """
        Decrement the boost_next_count after a boost is used.
        Should be called after a successful boosted API call.
        """
        async with self._lock:
            if self.boost_next_count > 0:
                self.boost_next_count -= 1
                logger.debug(f"Boost count consumed, remaining: {self.boost_next_count}")
                
                await self._broadcast("boost_next_count_updated", {
                    "count": self.boost_next_count
                })
    
    def get_boost_status(self) -> Dict[str, Any]:
        """
        Get current boost status.
        
        Returns:
            Dict with boost configuration and active tasks
        """
        if not self.boost_config:
            return {
                "enabled": False,
                "model_id": None,
                "boosted_task_count": 0,
                "boost_next_count": 0,
                "boosted_categories": [],
                "boosted_tasks": []
            }
        
        return {
            "enabled": self.boost_config.enabled,
            "model_id": self.boost_config.boost_model_id,
            "provider": self.boost_config.boost_provider,
            "context_window": self.boost_config.boost_context_window,
            "max_output_tokens": self.boost_config.boost_max_output_tokens,
            "boosted_task_count": len(self.boosted_task_ids),
            "boosted_tasks": list(self.boosted_task_ids),
            # NEW: Include new boost modes
            "boost_next_count": self.boost_next_count,
            "boosted_categories": list(self.boosted_categories)
        }
    
    def get_available_categories(self, mode: str = "all") -> List[Dict[str, str]]:
        """
        Get list of available boost categories based on current workflow mode.
        
        Args:
            mode: "aggregator", "compiler", "autonomous", or "all"
            
        Returns:
            List of category dicts with id and label
        """
        categories = []
        
        if mode in ("aggregator", "all"):
            for i in range(1, 11):
                categories.append({
                    "id": f"agg_sub{i}",
                    "label": f"Sub {i}",
                    "group": "Aggregator"
                })
            categories.append({
                "id": "agg_val",
                "label": "Validator",
                "group": "Aggregator"
            })
        
        if mode in ("compiler", "all"):
            categories.extend([
                {"id": "comp_hc", "label": "High-Context", "group": "Compiler"},
                {"id": "comp_hp", "label": "High-Param", "group": "Compiler"},
                {"id": "comp_val", "label": "Validator", "group": "Compiler"},
            ])
        
        if mode in ("autonomous", "all"):
            categories.extend([
                {"id": "auto_te", "label": "Topic Explore", "group": "Autonomous"},
                {"id": "auto_tev", "label": "Topic Explore Val", "group": "Autonomous"},
                {"id": "auto_ts", "label": "Topic Sel", "group": "Autonomous"},
                {"id": "auto_tv", "label": "Topic Val", "group": "Autonomous"},
                {"id": "auto_cr", "label": "Completion", "group": "Autonomous"},
                {"id": "auto_rs", "label": "Ref Sel", "group": "Autonomous"},
                {"id": "auto_pt", "label": "Paper Title", "group": "Autonomous"},
                {"id": "auto_prc", "label": "Redundancy", "group": "Autonomous"},
            ])
        
        return categories
    
    def is_role_boosted(self, role_prefix: str) -> bool:
        """
        Check if ANY task for a given role prefix is boosted.
        
        This is a fallback check when exact task_id matching fails.
        For example, role_prefix="agg_sub1" matches "agg_sub1_001".
        
        Args:
            role_prefix: Role prefix (e.g., "agg_sub1", "comp_hc", "auto_ts")
            
        Returns:
            True if any task for this role is boosted
        """
        if not self.boost_config or not self.boost_config.enabled:
            return False
        
        for task_id in self.boosted_task_ids:
            if task_id.startswith(role_prefix):
                return True
        return False
    
    def get_boosted_roles(self) -> set:
        """
        Get set of role prefixes that have boosted tasks.
        
        Returns:
            Set of role prefixes (e.g., {"agg_sub1", "comp_val"})
        """
        roles = set()
        for task_id in self.boosted_task_ids:
            # Split on last underscore to get role prefix
            # e.g., "agg_sub1_001" -> "agg_sub1"
            parts = task_id.rsplit('_', 1)
            if len(parts) == 2:
                roles.add(parts[0])
        return roles
    
    def get_next_boosted_task_for_role(self, role_prefix: str) -> Optional[str]:
        """
        Get the next boosted task ID for a role prefix.
        
        Args:
            role_prefix: Role prefix (e.g., "agg_sub1", "comp_hc")
            
        Returns:
            Task ID if found, None otherwise
        """
        if not self.boost_config or not self.boost_config.enabled:
            return None
        
        # Find all matching tasks and return the one with lowest sequence number
        matching_tasks = [
            task_id for task_id in self.boosted_task_ids
            if task_id.startswith(role_prefix)
        ]
        
        if not matching_tasks:
            return None
        
        # Sort by sequence number (last part after underscore)
        try:
            matching_tasks.sort(key=lambda t: int(t.rsplit('_', 1)[1]))
            return matching_tasks[0]
        except (ValueError, IndexError):
            return matching_tasks[0] if matching_tasks else None


# Global singleton instance
boost_manager = BoostManager()

