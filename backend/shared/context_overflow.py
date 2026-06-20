"""Shared user-facing context overflow messages."""

CONTEXT_OVERFLOW_STOP_REASON = "context_overflow"
CONTEXT_OVERFLOW_STOP_MESSAGE = (
    "Research stopped. Mandatory direct injection content reached the maximum context size."
)
CONTEXT_OVERFLOW_RESOLUTION = (
    "Condense the prompt and restart, or select a model with a larger context window."
)
