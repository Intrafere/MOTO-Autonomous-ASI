"""Safe typed errors for model-provider routing failures."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from backend.shared.log_redaction import redact_log_text


@dataclass(frozen=True)
class ProviderRouteIdentity:
    """Non-secret identity for the route that produced a model error."""

    provider: str
    model: str
    role_id: str = ""
    task_id: str = ""
    host_provider: str = ""
    route_kind: str = "primary"
    configured_provider: str = ""
    configured_model: str = ""

    def with_context(
        self,
        *,
        role_id: Optional[str] = None,
        task_id: Optional[str] = None,
        route_kind: Optional[str] = None,
        configured_provider: Optional[str] = None,
        configured_model: Optional[str] = None,
    ) -> "ProviderRouteIdentity":
        return replace(
            self,
            role_id=self.role_id if role_id is None else role_id,
            task_id=self.task_id if task_id is None else task_id,
            route_kind=self.route_kind if route_kind is None else route_kind,
            configured_provider=(
                self.configured_provider
                if configured_provider is None
                else configured_provider
            ),
            configured_model=(
                self.configured_model
                if configured_model is None
                else configured_model
            ),
        )

    def safe_summary(self) -> str:
        parts = [
            f"provider={redact_log_text(self.provider or 'unknown', 80)}",
            f"model={redact_log_text(self.model or 'unknown', 160)}",
            f"route={redact_log_text(self.route_kind or 'primary', 40)}",
        ]
        if self.host_provider:
            parts.append(f"host={redact_log_text(self.host_provider, 80)}")
        if self.role_id:
            parts.append(f"role={redact_log_text(self.role_id, 120)}")
        if self.task_id:
            parts.append(f"task={redact_log_text(self.task_id, 120)}")
        return ", ".join(parts)


class ProviderRouteError(RuntimeError):
    """A provider failure carrying safe route identity and its original cause."""

    def __init__(
        self,
        message: str,
        *,
        route: ProviderRouteIdentity,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.route = route
        self.cause = cause
        self.safe_message = redact_log_text(message, 1000)
        super().__init__(f"{self.safe_message} ({route.safe_summary()})")

    def with_route_context(
        self,
        *,
        role_id: str,
        task_id: str,
        route_kind: Optional[str] = None,
        configured_provider: Optional[str] = None,
        configured_model: Optional[str] = None,
    ) -> "ProviderRouteError":
        return type(self)(
            self.safe_message,
            route=self.route.with_context(
                role_id=role_id,
                task_id=task_id,
                route_kind=route_kind,
                configured_provider=configured_provider,
                configured_model=configured_model,
            ),
            cause=self.cause,
        )


class ProviderContextLengthError(ValueError):
    """A provider rejected mandatory input because it exceeded context capacity."""

    def __init__(
        self,
        message: str,
        *,
        route: ProviderRouteIdentity,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.route = route
        self.cause = cause
        self.safe_message = redact_log_text(message, 1000)
        super().__init__(f"{self.safe_message} ({route.safe_summary()})")

    def with_route_context(
        self,
        *,
        role_id: str,
        task_id: str,
        route_kind: Optional[str] = None,
        configured_provider: Optional[str] = None,
        configured_model: Optional[str] = None,
    ) -> "ProviderContextLengthError":
        return type(self)(
            self.safe_message,
            route=self.route.with_context(
                role_id=role_id,
                task_id=task_id,
                route_kind=route_kind,
                configured_provider=configured_provider,
                configured_model=configured_model,
            ),
            cause=self.cause,
        )

