"""Central proof-trade data exclusion boundary.

Every surface that could carry proof-trade data out of its owning module
(application/API logs, workflow diagnostics, support exports, updater
diagnostics, uncaught errors, and any future telemetry or crash reporting)
must route through this module:

* ``redact_trade_text`` / ``redact_trade_payload`` remove trade-sensitive
  values and prohibited fields while preserving payload shape;
* ``assert_trade_safe`` is the mandatory fail-closed gate for any future
  export, telemetry, or crash report and never echoes the offending value;
* ``trade_safe_error_detail`` builds client-visible errors from the exception
  type only;
* ``install_trade_data_log_boundary`` applies the boundary to every Python
  log record in the backend process.

The allowlisted proof-trade serializer (blocked on the frozen SyntheticLib
wire contract) consults ``TRADE_PROHIBITED_FIELDS`` but is the only place a
selected proof body may ever be emitted.
"""
from __future__ import annotations

import functools
import getpass
import logging
import os
import re
import socket
import threading
import traceback
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from backend.shared.log_redaction import bound_single_line, redact_sensitive_values


TRADE_DATA_BOUNDARY_VERSION = "moto-trade-data-boundary-v1"
_MAX_DEPTH = 12
_PROOF_BODY_CLASS = "proof_body"

TRADE_PROHIBITED_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType({
    "prompt": frozenset({
        "prompt", "prompts", "user_prompt", "system_prompt", "research_prompt",
        "source_prompt", "messages",
    }),
    "research_content": frozenset({
        "paper", "paper_content", "paper_text", "paper_title", "brainstorm",
        "brainstorm_content", "database", "submission", "submissions",
        "outline", "abstract",
    }),
    "source_identity": frozenset({
        "source_title", "source_id", "source_type", "session_id",
        "session_name", "run_id", "topic_id", "paper_id",
    }),
    "provider_details": frozenset({
        "provider", "provider_id", "model", "model_id", "model_name",
        "openrouter_provider", "reasoning", "reasoning_effort",
        "reasoning_summary", "thinking",
    }),
    "diagnostics": frozenset({
        "stderr", "stdout", "diagnostics", "error_output", "lean_output",
        "attempt_feedback", "feedback", "rejection_feedback",
        "validator_feedback", "traceback",
    }),
    "pruning": frozenset({
        "pruning", "prune_reason", "pruned_by", "live_context_status",
        "live_context_owner_run_id", "live_context_pruned_at",
        "live_context_pruned_by",
    }),
    "credentials": frozenset({
        "api_key", "apikey", "access_token", "refresh_token", "id_token",
        "token", "authorization", "password", "secret", "client_secret",
        "cookie", "session_cookie", "code_verifier",
    }),
    "host_identity": frozenset({
        "username", "user_name", "user", "hostname", "host", "machine",
        "path", "file_path", "data_root", "log_root", "home", "cwd",
    }),
    _PROOF_BODY_CLASS: frozenset({
        "lean_code", "lean_source", "checked_source", "source_code",
        "proof_source", "proof_text", "proof", "theorem", "theorem_statement",
        "statement", "formal_statement", "informal_statement",
    }),
})

_FIELD_CLASS_BY_NAME = {
    name: field_class
    for field_class, names in TRADE_PROHIBITED_FIELDS.items()
    for name in names
}

_OCCURRENCE_IDENTIFIER = re.compile(
    r"\b(moto_(?:proof|store)_)[0-9a-f]{32}\b",
    re.IGNORECASE,
)
_PATH_STOP = r"[^\s\"'<>|,;)\]}]*"
_ABSOLUTE_PATHS = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]" + _PATH_STOP),
    re.compile(r"\\\\[^\s\\/\"']+[\\/]" + _PATH_STOP),
    re.compile(
        r"(?<![\w.:/~-])/(?:home|Users|tmp|var|private|root|mnt|media|opt|srv|"
        r"etc|usr|Volumes|app)(?:/" + _PATH_STOP + r")?"
    ),
    re.compile(r"(?<![\w])~[\\/]" + _PATH_STOP),
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_GENERIC_LOCAL_NAMES = frozenset({
    "user", "users", "admin", "administrator", "root", "guest", "default",
    "owner", "runner", "localhost",
})

_TRADE_LOGGER_PREFIXES = (
    "backend.shared.syntheticlib",
    "backend.shared.trade_",
    "backend.shared.proof_trade_",
    "backend.shared.installation_identity",
    "backend.api.routes.syntheticlib4",
)
_BOUNDARY_FACTORY_MARKER = "_moto_trade_data_boundary"
_install_lock = threading.Lock()


class TradeDataBoundaryViolation(ValueError):
    """Raised when a payload would carry trade-sensitive data out of MOTO.

    The message lists reason codes only; it never contains the offending value.
    """

    def __init__(self, reason_codes: Iterable[str]):
        self.reason_codes = tuple(sorted(set(reason_codes)))
        super().__init__(
            "Trade-data boundary violation: " + ", ".join(self.reason_codes)
        )


def _normalize_field_name(name: Any) -> str:
    text = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", str(name).strip())
    return text.replace("-", "_").lower()


def classify_trade_field(name: Any) -> str | None:
    """Return the prohibited class for a payload field name, if any."""
    return _FIELD_CLASS_BY_NAME.get(_normalize_field_name(name))


@functools.lru_cache(maxsize=1)
def _local_identity_tokens() -> tuple[str, ...]:
    """Return the current OS username/hostname values worth redacting."""
    candidates: set[str] = set()
    for reader in (
        getpass.getuser,
        lambda: os.environ.get("USERNAME", ""),
        lambda: os.environ.get("USER", ""),
        socket.gethostname,
        lambda: os.environ.get("COMPUTERNAME", ""),
    ):
        try:
            value = str(reader() or "").strip()
        except Exception:
            continue
        if len(value) >= 3 and value.lower() not in _GENERIC_LOCAL_NAMES:
            candidates.add(value)
            short = value.split(".", 1)[0]
            if len(short) >= 3 and short.lower() not in _GENERIC_LOCAL_NAMES:
                candidates.add(short)
    return tuple(sorted(candidates, key=len, reverse=True))


def _redact_local_identity(text: str) -> str:
    for token in _local_identity_tokens():
        text = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])",
            "[local-identity-redacted]",
            text,
            flags=re.IGNORECASE,
        )
    return text


def redact_trade_values(value: Any) -> str:
    """Remove every trade-sensitive value class, preserving line breaks."""
    text = redact_sensitive_values(value)
    text = _OCCURRENCE_IDENTIFIER.sub(lambda match: f"{match.group(1)}[redacted]", text)
    for pattern in _ABSOLUTE_PATHS:
        text = pattern.sub("[path-redacted]", text)
    text = _EMAIL.sub("[email-redacted]", text)
    return _redact_local_identity(text)


def redact_trade_text(value: Any, max_chars: int | None = None) -> str:
    """Return single-line, optionally capped text with trade data removed."""
    return bound_single_line(redact_trade_values(value), max_chars)


def redact_trade_payload(value: Any, *, _depth: int = 0) -> Any:
    """Return a shape-preserving copy with prohibited fields and values removed."""
    if _depth >= _MAX_DEPTH:
        return "[bounded]"
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            field_class = classify_trade_field(key)
            if field_class == _PROOF_BODY_CLASS:
                redacted[str(key)] = {
                    "excluded": True,
                    "length": len(item) if isinstance(item, str) else None,
                }
            elif field_class is not None:
                redacted[str(key)] = "[trade-excluded]"
            else:
                redacted[str(key)] = redact_trade_payload(item, _depth=_depth + 1)
        return redacted
    if isinstance(value, list):
        return [redact_trade_payload(item, _depth=_depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_trade_payload(item, _depth=_depth + 1) for item in value)
    if isinstance(value, str):
        return redact_trade_values(value)
    return value


def trade_sensitive_value_kinds(value: Any) -> tuple[str, ...]:
    """Return the sensitive value classes present in text (no values echoed)."""
    text = "" if value is None else str(value)
    kinds: list[str] = []
    if redact_sensitive_values(text) != text:
        kinds.append("credential_or_installation_identifier")
    if _OCCURRENCE_IDENTIFIER.search(text):
        kinds.append("occurrence_identifier")
    if any(pattern.search(text) for pattern in _ABSOLUTE_PATHS):
        kinds.append("absolute_path")
    if _EMAIL.search(text):
        kinds.append("email")
    if _redact_local_identity(text) != text:
        kinds.append("local_identity")
    return tuple(kinds)


def _collect_violations(value: Any, reasons: set[str], depth: int) -> None:
    if depth >= _MAX_DEPTH:
        reasons.add("payload_too_deep")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            field_class = classify_trade_field(key)
            if field_class is not None:
                reasons.add(f"prohibited_field:{field_class}")
            for kind in trade_sensitive_value_kinds(key):
                reasons.add(f"sensitive_field_name:{kind}")
            _collect_violations(item, reasons, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_violations(item, reasons, depth + 1)
    elif isinstance(value, (str, bytes)):
        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        for kind in trade_sensitive_value_kinds(text):
            reasons.add(f"sensitive_value:{kind}")


def assert_trade_safe(payload: Any) -> None:
    """Fail closed if a payload carries any prohibited field or sensitive value."""
    reasons: set[str] = set()
    _collect_violations(payload, reasons, 0)
    if reasons:
        raise TradeDataBoundaryViolation(reasons)


def trade_safe_error_detail(prefix: str, exc: BaseException) -> str:
    """Build a client-visible error from the exception type, never its text."""
    return f"{prefix} ({type(exc).__name__})"


def is_trade_logger(name: str) -> bool:
    return any(str(name or "").startswith(prefix) for prefix in _TRADE_LOGGER_PREFIXES)


def apply_trade_data_log_boundary(record: logging.LogRecord) -> logging.LogRecord:
    """Redact one log record in place (credentials everywhere; trade data in trade loggers)."""
    trade_logger = is_trade_logger(record.name)
    redact = redact_trade_values if trade_logger else redact_sensitive_values
    try:
        message = record.getMessage()
    except Exception:
        return record
    redacted = redact(message)
    if redacted != message:
        record.msg = redacted
        record.args = ()
    exc_info = record.exc_info
    if exc_info and exc_info[0] is not None:
        if trade_logger:
            record.exc_text = (
                f"{exc_info[0].__name__} "
                "(traceback withheld by the proof-trade data boundary)"
            )
        else:
            try:
                formatted = "".join(traceback.format_exception(*exc_info)).rstrip("\n")
            except Exception:
                return record
            cleaned = redact_sensitive_values(formatted)
            if cleaned != formatted:
                record.exc_text = cleaned
    return record


def install_trade_data_log_boundary() -> None:
    """Install the boundary as the process log-record factory (idempotent)."""
    with _install_lock:
        current = logging.getLogRecordFactory()
        if getattr(current, _BOUNDARY_FACTORY_MARKER, False):
            return

        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            return apply_trade_data_log_boundary(current(*args, **kwargs))

        setattr(factory, _BOUNDARY_FACTORY_MARKER, True)
        logging.setLogRecordFactory(factory)
