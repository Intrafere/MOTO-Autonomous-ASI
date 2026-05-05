"""
Boost Manager - Singleton for managing API boost configuration and task selection.
Tracks which workflow tasks should use the boost OpenRouter API key.

Supports three boost modes:
1. Boost Next X Calls - Counter-based, applies to next X API calls regardless of task ID
2. Category Boost - Role-based, boosts all calls matching a role prefix (e.g., all Submitter 1 calls)
3. Always Prefer Boost - Every API call attempts boost first; falls back to primary on failure

Autonomous Research mode agents use the same role prefixes as their parent roles:
- Topic Selector, Completion Reviewer, Reference Selector, Paper Title Selector,
  Certainty Assessor, Format Selector, Volume Organizer → agg_sub1 (Submitter 1)
- Topic Validator, Redundancy Checker → agg_val (Agg Validator)
- Brainstorm aggregation submitters/validator → agg_sub1..10, agg_val (via Coordinator)
- Paper compilation → comp_hc, comp_hp, comp_val, comp_crit (via CompilerCoordinator)

State is persisted to backend/data/boost_state.json for crash recovery.
"""
import asyncio
import json
import logging
import os
from typing import Optional, Set, Callable, Any, Dict, List

from backend.shared.config import system_config
from backend.shared.models import BoostConfig

logger = logging.getLogger(__name__)

# Category prefixes for different roles — labels match Settings panel titles exactly.
# Autonomous Research agents share the same prefixes as their parent roles
# (see module docstring for full mapping).
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
    "agg_val": "Agg Validator",
    # Compiler
    "comp_hc": "High-Context Model",
    "comp_hp": "High-Param Model",
    "comp_val": "Compiler Validator",
    "comp_crit": "Critique Submitter",
}


class BoostManager:
    """
    Singleton manager for API boost configuration.
    Manages which tasks use the boost OpenRouter model.
    
    Supports three boost modes:
    - boost_next_count: Boost the next X API calls (counter-based)
    - boosted_categories: Boost all calls for specific role categories
    - boost_always_prefer: Try boost first for every API call, fall back on failure
    
    State is automatically persisted to disk for crash recovery.
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
        
        # Counter-based boost mode
        self.boost_next_count: int = 0
        
        # Category-based boost mode (role prefixes like "agg_sub1", "comp_hc")
        self.boosted_categories: Set[str] = set()
        
        # Always-prefer boost mode: try boost for every call, fall back on failure
        self.boost_always_prefer: bool = False
        
        self._initialized = True
        
        # Load persisted state on initialization
        self._load_state()
        
        logger.info("BoostManager initialized")

    @staticmethod
    def _get_state_file() -> str:
        """Return the instance-scoped boost state file."""
        return str(os.path.join(system_config.data_dir, "boost_state.json"))
    
    def _load_state(self) -> None:
        """Load persisted boost state from disk."""
        try:
            state_file = self._get_state_file()
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Restore boost config if it was enabled
                if state.get('enabled') and state.get('model_id'):
                    self.boost_config = BoostConfig(
                        enabled=True,
                        openrouter_api_key=state.get('api_key', ''),
                        boost_model_id=state.get('model_id'),
                        boost_provider=state.get('provider'),
                        boost_context_window=state.get('context_window', 131072),
                        boost_max_output_tokens=state.get('max_output_tokens', 25000)
                    )
                
                # Restore boost modes
                self.boost_next_count = state.get('boost_next_count', 0)
                self.boosted_categories = set(state.get('boosted_categories', []))
                self.boost_always_prefer = state.get('boost_always_prefer', False)
                self.boosted_task_ids = set(state.get('boosted_task_ids', []))
                
                logger.info(f"Loaded boost state: enabled={state.get('enabled')}, model={state.get('model_id')}, "
                           f"next_count={self.boost_next_count}, categories={len(self.boosted_categories)}, "
                           f"always_prefer={self.boost_always_prefer}")
        except Exception as e:
            logger.warning(f"Failed to load boost state: {e}")
    
    def _save_state(self) -> None:
        """Persist current boost state to disk."""
        try:
            # Ensure data directory exists
            state_file = self._get_state_file()
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            state = {
                'enabled': self.boost_config is not None and self.boost_config.enabled,
                'model_id': self.boost_config.boost_model_id if self.boost_config else None,
                'provider': self.boost_config.boost_provider if self.boost_config else None,
                'context_window': self.boost_config.boost_context_window if self.boost_config else 131072,
                'max_output_tokens': self.boost_config.boost_max_output_tokens if self.boost_config else 25000,
                'api_key': self.boost_config.openrouter_api_key if self.boost_config else '',
                'boost_next_count': self.boost_next_count,
                'boosted_categories': list(self.boosted_categories),
                'boost_always_prefer': self.boost_always_prefer,
                'boosted_task_ids': list(self.boosted_task_ids)
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            logger.debug("Boost state saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save boost state: {e}")
    
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
            
            # Persist state
            self._save_state()
            
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
                self.boost_always_prefer = False
                
                # Persist state
                self._save_state()
                
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
            
            # Persist state
            self._save_state()
            
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
            
            # Persist state
            self._save_state()
            
            await self._broadcast("boost_next_count_updated", {
                "count": self.boost_next_count
            })
    
    async def set_always_prefer(self, enabled: bool) -> None:
        """
        Enable or disable always-prefer-boost mode.
        
        When enabled, every API call attempts boost first and falls back to the
        primary model on any failure. Mutually exclusive with next_count and
        category modes (caller should clear those before enabling this).
        
        Args:
            enabled: True to enable, False to disable
        """
        async with self._lock:
            self.boost_always_prefer = enabled
            logger.info(f"Boost always-prefer {'enabled' if enabled else 'disabled'}")
            
            # Persist state
            self._save_state()
            
            await self._broadcast("boost_always_prefer_updated", {
                "enabled": enabled
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
            
            # Persist state
            self._save_state()
            
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
        
        # Check always-prefer mode (every call uses boost, fall back on failure)
        if self.boost_always_prefer:
            return True
        
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
                
                # Persist state
                self._save_state()
                
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
                "boost_always_prefer": False,
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
            "boost_next_count": self.boost_next_count,
            "boosted_categories": list(self.boosted_categories),
            "boost_always_prefer": self.boost_always_prefer
        }
    
    def get_available_categories(self, mode: str = "all") -> List[Dict[str, str]]:
        """
        Get list of all boost categories in the same order as Settings panels.
        All categories are always returned regardless of mode.
        
        Autonomous Research agents automatically inherit boosts from their parent roles:
        - Submitter 1 (agg_sub1) covers: Topic Selector, Completion Reviewer,
          Reference Selector, Paper Title Selector, Certainty Assessor, Format Selector,
          Volume Organizer
        - Agg Validator (agg_val) covers: Topic Validator, Redundancy Checker
        - Compiler roles cover paper compilation phases
        
        Args:
            mode: ignored — kept for API compatibility
            
        Returns:
            List of category dicts with id, label, and group
        """
        categories = []
        
        # Aggregator (matches AggregatorSettings order: Submitters 1-10, then Validator)
        for i in range(1, 11):
            categories.append({
                "id": f"agg_sub{i}",
                "label": f"Submitter {i}",
                "group": "Aggregator"
            })
        categories.append({
            "id": "agg_val",
            "label": "Agg Validator",
            "group": "Aggregator"
        })
        
        # Compiler (matches CompilerSettings order: Validator, High-Context, High-Param, Critique)
        categories.extend([
            {"id": "comp_val", "label": "Compiler Validator", "group": "Compiler"},
            {"id": "comp_hc", "label": "High-Context Model", "group": "Compiler"},
            {"id": "comp_hp", "label": "High-Param Model", "group": "Compiler"},
            {"id": "comp_crit", "label": "Critique Submitter", "group": "Compiler"},
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

