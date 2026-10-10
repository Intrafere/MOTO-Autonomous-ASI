"""Process-local serialization for provider credential mutation transactions."""
import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar


_OPENROUTER_LOCK = asyncio.Lock()
_WOLFRAM_LOCK = asyncio.Lock()
_SAKANA_LOCK = asyncio.Lock()
_CODEX_LOCK = asyncio.Lock()
_XAI_GROK_LOCK = asyncio.Lock()

_T = TypeVar("_T")


def serialize_credential_mutation(
    lock: asyncio.Lock,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Hold one provider lock across validation, persistence, and publication."""
    def decorator(func: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @wraps(func)
        async def wrapped(*args: Any, **kwargs: Any) -> _T:
            async with lock:
                return await func(*args, **kwargs)

        return wrapped

    return decorator


openrouter_credential_mutation = serialize_credential_mutation(_OPENROUTER_LOCK)
wolfram_credential_mutation = serialize_credential_mutation(_WOLFRAM_LOCK)
sakana_credential_mutation = serialize_credential_mutation(_SAKANA_LOCK)
codex_credential_mutation = serialize_credential_mutation(_CODEX_LOCK)
xai_grok_credential_mutation = serialize_credential_mutation(_XAI_GROK_LOCK)
