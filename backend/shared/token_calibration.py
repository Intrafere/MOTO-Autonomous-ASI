"""Measured difference between MOTO's token estimate and a provider's own count.

MOTO sizes every prompt with one local tokenizer. Models that tokenize
differently can count more tokens for the same text, so a prompt MOTO believes
fits may be rejected. Every completed call already reports the provider's own
input-token count, so this module compares the two per model at no extra API
cost, and only after several agreeing measurements treats that model's context
window as proportionally smaller. It never treats a window as larger than the
user configured, and until a model is verified its budgets are unchanged.

Measurements are non-secret (provider id, model id, ratios, counters) and
persist under the active data root.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from backend.shared.atomic_json import atomic_write_json
from backend.shared.log_redaction import redact_log_text

logger = logging.getLogger(__name__)

TOKEN_CALIBRATION_FILENAME = "token_calibration.json"
_FILE_VERSION = 1

# A prompt must be at least this large, and at least this share of the role's
# context window, before it is compared. Small prompts are dominated by fixed
# per-request overhead (chat framing, provider instructions), which would
# overstate the ratio that matters near the limit.
MIN_SAMPLE_TOKENS = 8_000
MIN_WINDOW_FRACTION = 0.10
# Agreeing measurements needed before a model's ratio is trusted.
REQUIRED_SAMPLES = 10
# Rolling window of recent measurements kept per model.
MAX_SAMPLES = 50
# The ratio varies with content (Lean and math versus prose), so the high end
# of recent measurements is used rather than the average.
HIGH_QUANTILE = 0.9
# A verified ratio within this margin of 1.0 counts as matching MOTO's count.
MATCH_TOLERANCE = 0.02
# Ratios outside these bounds are not token-count differences and are ignored.
MIN_PLAUSIBLE_RATIO = 0.5
MAX_PLAUSIBLE_RATIO = 3.0
# Once the rolling window is full, compare one completed call in this many.
RESAMPLE_INTERVAL = 5

ModelKey = Tuple[str, str]

_LM_STUDIO_INSTANCE_SUFFIX = re.compile(r":\d+$")


def normalize_model_key(provider: Any, model: Any) -> Optional[ModelKey]:
    """Return the stable (provider, model) identity used for measurements."""
    provider_text = str(provider or "").strip()
    model_text = str(model or "").strip()
    if not provider_text or not model_text:
        return None
    if provider_text == "lm_studio":
        # Loaded sibling instances ("model:2") share one tokenizer.
        model_text = _LM_STUDIO_INSTANCE_SUFFIX.sub("", model_text)
    return provider_text, model_text


class _ModelMeasurement:
    __slots__ = ("ratios", "samples_total", "calls_seen", "updated_at", "verified_ratio")

    def __init__(self) -> None:
        self.ratios: Deque[float] = deque(maxlen=MAX_SAMPLES)
        self.samples_total = 0
        self.calls_seen = 0
        self.updated_at = 0.0
        self.verified_ratio: Optional[float] = None

    def refresh(self) -> None:
        if len(self.ratios) < REQUIRED_SAMPLES:
            self.verified_ratio = None
            return
        ordered = sorted(self.ratios)
        index = max(0, math.ceil(HIGH_QUANTILE * len(ordered)) - 1)
        self.verified_ratio = ordered[index]


class TokenCalibration:
    """Per-model token-count measurements and the window scale they justify."""

    def __init__(self) -> None:
        self._models: Dict[ModelKey, _ModelMeasurement] = {}
        self._role_windows: Dict[str, Tuple[ModelKey, int]] = {}
        self._window_models: Dict[int, Set[ModelKey]] = {}
        self._adjust_enabled = True
        self._loaded_path: Optional[Path] = None
        self._save_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------ storage

    @staticmethod
    def _path() -> Path:
        from backend.shared.config import system_config

        return Path(system_config.data_dir) / TOKEN_CALIBRATION_FILENAME

    def _ensure_loaded(self) -> None:
        """Load the active data root's measurements once (and again after a root change)."""
        try:
            path = self._path()
        except Exception:
            return
        if path == self._loaded_path:
            return
        self._loaded_path = path
        self._models = {}
        self._adjust_enabled = True
        try:
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "Ignoring unreadable token calibration file %s: %s",
                redact_log_text(path, 240),
                redact_log_text(exc, 240),
            )
            return
        if not isinstance(payload, dict) or payload.get("version") != _FILE_VERSION:
            return
        self._adjust_enabled = payload.get("adjust_enabled") is not False
        for entry in payload.get("models") or []:
            if not isinstance(entry, dict):
                continue
            key = normalize_model_key(entry.get("provider"), entry.get("model"))
            if key is None:
                continue
            measurement = _ModelMeasurement()
            for value in (entry.get("ratios") or [])[-MAX_SAMPLES:]:
                if isinstance(value, (int, float)) and MIN_PLAUSIBLE_RATIO <= value <= MAX_PLAUSIBLE_RATIO:
                    measurement.ratios.append(float(value))
            try:
                measurement.samples_total = max(len(measurement.ratios), int(entry.get("samples_total") or 0))
                measurement.updated_at = float(entry.get("updated_at") or 0.0)
            except (TypeError, ValueError):
                measurement.samples_total = len(measurement.ratios)
            measurement.refresh()
            self._models[key] = measurement

    def _payload(self) -> Dict[str, Any]:
        return {
            "version": _FILE_VERSION,
            "adjust_enabled": self._adjust_enabled,
            "models": [
                {
                    "provider": key[0],
                    "model": key[1],
                    "ratios": [round(value, 4) for value in measurement.ratios],
                    "samples_total": measurement.samples_total,
                    "updated_at": measurement.updated_at,
                }
                for key, measurement in sorted(self._models.items())
            ],
        }

    async def save(self) -> None:
        """Persist the current measurements without blocking the event loop."""
        if self._save_lock is None:
            self._save_lock = asyncio.Lock()
        async with self._save_lock:
            self._ensure_loaded()
            path = self._loaded_path
            if path is None:
                return
            payload = self._payload()
            try:
                await asyncio.to_thread(atomic_write_json, path, payload)
            except OSError as exc:
                logger.warning(
                    "Failed to persist token calibration file %s: %s",
                    redact_log_text(path, 240),
                    redact_log_text(exc, 240),
                )

    # ------------------------------------------------------------------- roles

    def register_role(self, role_id: str, provider: Any, model: Any, context_window: Any) -> None:
        """Record which model a configured role's context window belongs to."""
        key = normalize_model_key(provider, model)
        try:
            window = int(context_window or 0)
        except (TypeError, ValueError):
            window = 0
        if not role_id or key is None or window <= 0:
            return
        entry = (key, window)
        if self._role_windows.get(role_id) == entry:
            return
        self._role_windows[role_id] = entry
        index: Dict[int, Set[ModelKey]] = {}
        for role_key, role_window in self._role_windows.values():
            index.setdefault(role_window, set()).add(role_key)
        self._window_models = index

    def role_model_key(self, role_id: str) -> Optional[ModelKey]:
        entry = self._role_windows.get(role_id)
        return entry[0] if entry else None

    # ------------------------------------------------------------- measurement

    def wants_sample(self, key: ModelKey, prompt_characters: int, context_window: int = 0) -> bool:
        """Cheap pre-check before counting a prompt: is this call worth comparing?"""
        # Counted text is rarely shorter in characters than in tokens, so a prompt
        # below the token floor in characters cannot qualify.
        if prompt_characters < self.minimum_sample_tokens(context_window):
            return False
        self._ensure_loaded()
        measurement = self._models.get(key)
        if measurement is None or len(measurement.ratios) < MAX_SAMPLES:
            return True
        measurement.calls_seen += 1
        return measurement.calls_seen % RESAMPLE_INTERVAL == 0

    @staticmethod
    def minimum_sample_tokens(context_window: int = 0) -> int:
        try:
            window = int(context_window or 0)
        except (TypeError, ValueError):
            window = 0
        return max(MIN_SAMPLE_TOKENS, int(window * MIN_WINDOW_FRACTION))

    def record(self, key: ModelKey, moto_tokens: int, provider_tokens: int, context_window: int = 0) -> bool:
        """Record one comparison. Returns True when it was accepted."""
        if moto_tokens < self.minimum_sample_tokens(context_window) or provider_tokens <= 0:
            return False
        ratio = provider_tokens / moto_tokens
        if not MIN_PLAUSIBLE_RATIO <= ratio <= MAX_PLAUSIBLE_RATIO:
            return False
        self._ensure_loaded()
        measurement = self._models.setdefault(key, _ModelMeasurement())
        previous = measurement.verified_ratio
        measurement.ratios.append(ratio)
        measurement.samples_total += 1
        measurement.updated_at = time.time()
        measurement.refresh()
        if previous is None and measurement.verified_ratio is not None:
            logger.info(
                "Token count verified for %s / %s: provider counts %.3fx MOTO's estimate (%d measurements)",
                key[0],
                redact_log_text(key[1], 160),
                measurement.verified_ratio,
                len(measurement.ratios),
            )
        return True

    # -------------------------------------------------------------- adjustment

    @staticmethod
    def _scale_for_ratio(ratio: Optional[float]) -> float:
        if ratio is None or ratio <= 1.0 + MATCH_TOLERANCE:
            return 1.0
        return ratio

    def verified_ratio(self, key: ModelKey) -> Optional[float]:
        self._ensure_loaded()
        measurement = self._models.get(key)
        return measurement.verified_ratio if measurement else None

    def window_scale(self, context_window: Any) -> float:
        """Return the divisor (>= 1.0) for a configured context window.

        A window belongs to the configured roles that use that exact value. When
        models with different verified ratios share one value, the largest ratio
        applies, which can only make a budget smaller.
        """
        try:
            window = int(context_window or 0)
        except (TypeError, ValueError):
            return 1.0
        keys = self._window_models.get(window)
        if not keys:
            return 1.0
        self._ensure_loaded()
        if not self._adjust_enabled:
            return 1.0
        scale = 1.0
        for key in tuple(keys):
            measurement = self._models.get(key)
            if measurement is not None:
                scale = max(scale, self._scale_for_ratio(measurement.verified_ratio))
        return scale

    def adjusted_context_window(self, context_window: int) -> int:
        """The window MOTO budgets against after any verified adjustment."""
        scale = self.window_scale(context_window)
        if scale <= 1.0:
            return int(context_window)
        return max(1, int(int(context_window) / scale))

    # ----------------------------------------------------------------- control

    @property
    def adjust_enabled(self) -> bool:
        self._ensure_loaded()
        return self._adjust_enabled

    def set_adjust_enabled(self, enabled: bool) -> None:
        self._ensure_loaded()
        self._adjust_enabled = bool(enabled)

    def reset(self, provider: Any = None, model: Any = None) -> int:
        """Discard measurements for one model, or for every model. Returns the count removed."""
        self._ensure_loaded()
        if provider is None and model is None:
            removed = len(self._models)
            self._models = {}
            return removed
        key = normalize_model_key(provider, model)
        if key is None or key not in self._models:
            return 0
        del self._models[key]
        return 1

    def snapshot(self) -> Dict[str, Any]:
        """In-memory status for the settings UI."""
        self._ensure_loaded()
        windows_by_model: Dict[ModelKey, Set[int]] = {}
        for key, window in self._role_windows.values():
            windows_by_model.setdefault(key, set()).add(window)
        models: List[Dict[str, Any]] = []
        for key, measurement in sorted(self._models.items()):
            ratio = measurement.verified_ratio
            scale = self._scale_for_ratio(ratio)
            if ratio is None:
                state = "measuring"
            elif scale > 1.0:
                state = "counts_more"
            else:
                state = "matches"
            models.append(
                {
                    "provider": key[0],
                    "model": key[1],
                    "state": state,
                    "samples": len(measurement.ratios),
                    "samples_required": REQUIRED_SAMPLES,
                    "ratio": round(ratio, 4) if ratio is not None else None,
                    "applied": bool(self._adjust_enabled and scale > 1.0),
                    "context_windows": [
                        {
                            "configured": window,
                            "effective": self.adjusted_context_window(window),
                        }
                        for window in sorted(windows_by_model.get(key, ()))
                    ],
                    "updated_at": measurement.updated_at or None,
                }
            )
        return {
            "adjust_enabled": self._adjust_enabled,
            "samples_required": REQUIRED_SAMPLES,
            "minimum_sample_tokens": MIN_SAMPLE_TOKENS,
            "models": models,
        }


token_calibration = TokenCalibration()
