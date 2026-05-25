"""
Small helpers for removing obvious secrets from locally persisted log previews.
"""
from __future__ import annotations

import re
from typing import Any


_SECRET_PATTERNS = (
    re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r'("(?:api[_-]?key|appid|authorization|password|token|secret)"\s*:\s*)"[^"]*"', re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|appid|authorization|password|token|secret)\s*[=:]\s*)[^\s,&}\]]+", re.IGNORECASE),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
)


def redact_log_text(value: Any, max_chars: int | None = None) -> str:
    """Return text with common credential shapes redacted and optionally capped."""
    text = "" if value is None else str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1) if match.lastindex else ''}[redacted]",
            text,
        )

    # Prevent log forging by keeping caller-controlled values on one line.
    text = (
        text
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )

    if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text
