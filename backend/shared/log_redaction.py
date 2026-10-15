"""
Small helpers for removing obvious secrets from locally persisted log previews.

This is the lowest redaction layer (stdlib only). It removes credential shapes
and installation-linking/trade-only opaque identifiers everywhere; the
proof-trade data boundary in ``trade_data_boundary`` builds stricter exclusion
on top of it. Redaction is idempotent so already-redacted text is unchanged.
"""
from __future__ import annotations

import re
from typing import Any


_SECRET_PATTERNS = (
    re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r'("(?:api[_-]?key|appid|authorization|password|token|secret)"\s*:\s*)"[^"]*"', re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|appid|authorization|password|token|secret)\s*[=:]\s*)(?!\[redacted\])[^\s,&}\]]+", re.IGNORECASE),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
)

# Installation-linking and trade-only opaque identifiers never belong in logs
# or activity. Client proof/store IDs are local occurrence identity (proof-search
# IDs embed them), so only the stricter trade-data boundary removes those.
_TRADE_IDENTIFIER_PATTERN = re.compile(
    r"\b(moto_(?:installation|root|batch|selection)_)[0-9a-f]{32}\b",
    re.IGNORECASE,
)


def redact_sensitive_values(value: Any) -> str:
    """Redact credential shapes and trade identifiers, preserving line breaks."""
    text = "" if value is None else str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1) if match.lastindex else ''}[redacted]",
            text,
        )
    return _TRADE_IDENTIFIER_PATTERN.sub(
        lambda match: f"{match.group(1)}[redacted]",
        text,
    )


def bound_single_line(text: str, max_chars: int | None = None) -> str:
    """Keep caller-controlled text on one line and optionally cap its length."""
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


def redact_log_text(value: Any, max_chars: int | None = None) -> str:
    """Return text with common credential shapes redacted and optionally capped."""
    return bound_single_line(redact_sensitive_values(value), max_chars)
