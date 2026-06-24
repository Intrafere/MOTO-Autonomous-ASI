"""Shared user-facing context overflow messages."""

CONTEXT_OVERFLOW_STOP_REASON = "context_overflow"
CONTEXT_OVERFLOW_STOP_MESSAGE = (
    "Research stopped. Some required source content must be injected directly to preserve "
    "answer quality, and it reached the maximum context size for the selected model. "
    "Start a new session with a condensed prompt, or choose a model with a higher "
    "context limit."
)
CONTEXT_OVERFLOW_RESOLUTION = (
    "Start a new session with a condensed prompt, or choose a model with a higher context limit."
)
