"""
Free Model Manager - Manages free model rotation and account-wide credit exhaustion detection.

When a free OpenRouter model fails, this manager provides:
1. Free Model Looping: rotate to next available free model (highest context first)
2. Auto-Selector Backup: fall back to openrouter/free as last resort
3. Account Exhaustion Detection: halt all free model requests on 402
"""
import logging
import time
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)

# How long to remember failed models before allowing retry (seconds)
FAILED_MODEL_EXPIRY = 300  # 5 minutes


class FreeModelManager:
    """Singleton managing free model rotation and account exhaustion state."""

    AUTO_SELECTOR_MODEL = "openrouter/free"
    AUTO_SELECTOR_CONTEXT = 131072

    def __init__(self):
        self.looping_enabled: bool = True
        self.auto_selector_enabled: bool = True

        self._cached_free_models: List[Dict[str, Any]] = []
        self._cache_timestamp: float = 0.0

        self._account_credits_exhausted: bool = False
        self._account_exhausted_timestamp: float = 0.0
        
        # Track models that failed with timestamps (model_id -> failure_time)
        self._failed_models: Dict[str, float] = {}

    def configure(self, looping: bool, auto_selector: bool) -> None:
        """Set free model settings from frontend."""
        self.looping_enabled = looping
        self.auto_selector_enabled = auto_selector
        logger.info(
            f"Free model settings: looping={looping}, auto_selector={auto_selector}"
        )

    def reset(self) -> None:
        """Reset all transient state. Call at workflow start."""
        self._failed_models.clear()
        logger.debug("FreeModelManager reset - cleared failed models")

    def update_cached_models(self, models: List[Dict[str, Any]]) -> None:
        """
        Cache free models sorted by context_length descending.
        Called when /api/openrouter/models is fetched.
        """
        free_models = []
        for m in models:
            pricing = m.get("pricing", {})
            try:
                is_free = (
                    float(pricing.get("prompt", "1")) == 0.0
                    and float(pricing.get("completion", "1")) == 0.0
                )
            except (ValueError, TypeError):
                continue
            if is_free:
                free_models.append(m)

        free_models.sort(
            key=lambda m: m.get("context_length", 0), reverse=True
        )
        self._cached_free_models = free_models
        self._cache_timestamp = time.time()
        logger.info(f"Cached {len(free_models)} free models for rotation")

    def _cleanup_expired_failures(self) -> None:
        """Remove models from failed list if their expiry has passed."""
        current_time = time.time()
        expired = [
            model_id for model_id, fail_time in self._failed_models.items()
            if current_time - fail_time > FAILED_MODEL_EXPIRY
        ]
        for model_id in expired:
            del self._failed_models[model_id]
            logger.debug(f"Model {model_id} failure expired, now available for retry")

    def get_alternative_free_model(
        self,
        current_model_id: str,
        skip_models: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Get the next available free model, sorted by highest context_length first,
        skipping the current model and any models in skip_models.

        Returns model ID string or None if all are skipped.
        """
        if not self._cached_free_models:
            logger.debug("No cached free models available for rotation")
            return None

        # Clean up expired failures before checking
        self._cleanup_expired_failures()

        skip = skip_models or set()
        skip = skip | set(self._failed_models.keys())  # Also skip recently failed models

        for m in self._cached_free_models:
            model_id = m.get("id", "")
            if not model_id or model_id == current_model_id or model_id in skip:
                continue
            return model_id

        return None
    
    def mark_model_failed(self, model_id: str) -> None:
        """Mark a model as failed with current timestamp."""
        self._failed_models[model_id] = time.time()
        logger.debug(f"Marked model as failed: {model_id} (expires in {FAILED_MODEL_EXPIRY}s)")
    
    def clear_failed_models(self) -> None:
        """Clear the failed models dict (call on successful request)."""
        if self._failed_models:
            logger.debug(f"Clearing {len(self._failed_models)} failed models")
            self._failed_models.clear()

    def mark_account_exhausted(self) -> None:
        """Mark that the OpenRouter account has no free credits (402 on free model)."""
        if not self._account_credits_exhausted:
            self._account_credits_exhausted = True
            self._account_exhausted_timestamp = time.time()
            logger.error(
                "OpenRouter account free credits exhausted (402). "
                "All free model requests will be blocked until credits are restored."
            )

    def is_account_exhausted(self) -> bool:
        """Check if account-wide free credits are exhausted."""
        return self._account_credits_exhausted

    def clear_account_exhaustion(self) -> None:
        """Clear the account exhaustion flag (e.g. after credits are added)."""
        if self._account_credits_exhausted:
            self._account_credits_exhausted = False
            logger.info("Account free credit exhaustion flag cleared")

    def get_status(self) -> Dict[str, Any]:
        """Get current free model manager status for API responses."""
        return {
            "looping_enabled": self.looping_enabled,
            "auto_selector_enabled": self.auto_selector_enabled,
            "cached_free_model_count": len(self._cached_free_models),
            "account_credits_exhausted": self._account_credits_exhausted,
            "failed_model_count": len(self._failed_models),
        }


# Global singleton
free_model_manager = FreeModelManager()
