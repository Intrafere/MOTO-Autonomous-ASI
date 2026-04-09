"""
Token Tracker - Tracks cumulative input/output token usage across the session,
with per-model breakdown and a research timer.
"""
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TokenTracker:
    """
    Singleton that accumulates prompt_tokens and completion_tokens
    from every successful LLM completion call, broken down by model.
    Also provides a simple elapsed-time research timer.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._total_input = 0
        self._total_output = 0
        self._by_model: Dict[str, Dict[str, int]] = {}
        self._start_time: Optional[float] = None
        self._stopped_elapsed: float = 0.0
        logger.info("TokenTracker initialized")

    def track(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for one successful API call."""
        self._total_input += prompt_tokens
        self._total_output += completion_tokens
        if model_id not in self._by_model:
            self._by_model[model_id] = {"input": 0, "output": 0}
        self._by_model[model_id]["input"] += prompt_tokens
        self._by_model[model_id]["output"] += completion_tokens

    def start_timer(self) -> None:
        """Start (or resume) the research timer."""
        if self._start_time is None:
            self._start_time = time.time()
            logger.info("TokenTracker timer started")

    def stop_timer(self) -> None:
        """Pause the timer, preserving elapsed time so it can be resumed."""
        if self._start_time is not None:
            self._stopped_elapsed += time.time() - self._start_time
            self._start_time = None
            logger.info(f"TokenTracker timer stopped (elapsed: {self._stopped_elapsed:.1f}s)")

    def get_elapsed_seconds(self) -> float:
        """Return total elapsed seconds (running + previously stopped segments)."""
        elapsed = self._stopped_elapsed
        if self._start_time is not None:
            elapsed += time.time() - self._start_time
        return elapsed

    def get_stats(self) -> Dict[str, Any]:
        """Return current cumulative stats for the frontend."""
        return {
            "total_input": self._total_input,
            "total_output": self._total_output,
            "by_model": dict(self._by_model),
            "elapsed_seconds": round(self.get_elapsed_seconds(), 1),
        }

    def reset(self) -> None:
        """Clear all counters and timer for a new session."""
        self._total_input = 0
        self._total_output = 0
        self._by_model.clear()
        self._start_time = None
        self._stopped_elapsed = 0.0
        logger.info("TokenTracker reset")


token_tracker = TokenTracker()
