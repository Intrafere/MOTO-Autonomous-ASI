from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from tests.workflow_harness.real_adapters import EventCollector


@dataclass
class CallRecorder:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    result: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.result

    async def async_call(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    @property
    def count(self) -> int:
        return len(self.calls)


def inactive_workflow_flags() -> dict[str, object]:
    return {
        "coordinator": SimpleNamespace(is_running=False),
        "compiler_coordinator": SimpleNamespace(is_running=False),
        "autonomous_coordinator": SimpleNamespace(
            is_active=False,
            get_state=lambda: SimpleNamespace(is_running=False, current_tier="idle"),
        ),
        "leanoj_coordinator": SimpleNamespace(is_active=False),
    }


def patch_attributes(monkeypatch, target: object, attributes: dict[str, object]) -> None:
    for name, value in attributes.items():
        monkeypatch.setattr(target, name, value)


def assert_event_sequence(
    collector: EventCollector,
    *event_types: str,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> None:
    actual = [event_type for event_type, _payload in collector.events]
    cursor = 0
    for expected in event_types:
        try:
            cursor = actual.index(expected, cursor) + 1
        except ValueError as exc:
            raise AssertionError(
                f"Expected ordered event {expected!r} after index {cursor}; observed {actual!r}."
            ) from exc
    if predicate is not None and not any(predicate(payload) for _, payload in collector.events):
        raise AssertionError("No collected event payload satisfied the required predicate.")
