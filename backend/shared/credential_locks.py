"""Process-local serialization for provider credential mutation transactions."""
import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, TypeVar

from backend.shared.subscription_providers.metadata import API_KEY_SUBSCRIPTION_PROVIDER_IDS


_OPENROUTER_LOCK = asyncio.Lock()
_WOLFRAM_LOCK = asyncio.Lock()
_CODEX_LOCK = asyncio.Lock()
_XAI_GROK_LOCK = asyncio.Lock()
_SUBSCRIPTION_API_KEY_LOCKS: Dict[str, asyncio.Lock] = {
    provider_id: asyncio.Lock() for provider_id in API_KEY_SUBSCRIPTION_PROVIDER_IDS
}

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


def subscription_api_key_lock(provider_id: str) -> asyncio.Lock:
    """Return the credential-mutation lock owned by one API-key subscription provider."""
    try:
        return _SUBSCRIPTION_API_KEY_LOCKS[provider_id]
    except KeyError as exc:
        raise KeyError(f"No credential lock for subscription provider: {provider_id}") from exc


openrouter_credential_mutation = serialize_credential_mutation(_OPENROUTER_LOCK)
wolfram_credential_mutation = serialize_credential_mutation(_WOLFRAM_LOCK)
codex_credential_mutation = serialize_credential_mutation(_CODEX_LOCK)
xai_grok_credential_mutation = serialize_credential_mutation(_XAI_GROK_LOCK)
