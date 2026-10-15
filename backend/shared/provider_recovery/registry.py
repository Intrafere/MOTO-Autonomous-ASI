"""Provider adapter protocol and registry."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

from backend.shared.provider_recovery.models import ProviderFailure


@runtime_checkable
class ProviderAdapter(Protocol):
    provider_id: str
    provider_label: str

    async def complete(self, **kwargs: Any) -> Dict[str, Any]:
        """Return a normalized OpenAI-compatible completion or raise ProviderFailure."""

    def classify_error(self, error: BaseException, **context: Any) -> ProviderFailure:
        """Translate a provider-specific error into a provider-neutral failure."""


AdapterFactory = Callable[[], ProviderAdapter]
AsyncAdapterFactory = Callable[[], Awaitable[ProviderAdapter]]


class ProviderRegistry:
    """Explicit provider registration; unknown providers fail closed."""

    def __init__(self) -> None:
        self._adapters: Dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter, *, replace: bool = False) -> None:
        provider_id = str(adapter.provider_id or "").strip()
        if not provider_id:
            raise ValueError("Provider adapter must define provider_id")
        if provider_id in self._adapters and not replace:
            raise ValueError(f"Provider adapter already registered: {provider_id}")
        self._adapters[provider_id] = adapter

    def get(self, provider_id: str) -> ProviderAdapter:
        provider_id = str(provider_id or "").strip()
        try:
            return self._adapters[provider_id]
        except KeyError as exc:
            raise KeyError(f"Provider adapter is not registered: {provider_id}") from exc

    def maybe_get(self, provider_id: str) -> Optional[ProviderAdapter]:
        return self._adapters.get(str(provider_id or "").strip())

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))
