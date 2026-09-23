"""Explicitly scoped, isolated formalization fallback for Autonomous owners.

The caller owns registration and persistence. This helper never invokes novelty,
mutates a proof store, or catches exceptions from checkpoint/observer callbacks.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from typing import Any

from backend.shared.api_client_manager import RetryableProviderError
from backend.shared.model_error_utils import is_non_retryable_model_error
from backend.shared.provider_pause import is_provider_credit_pause_error
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRepairRequiredError, ProviderRouteError


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def secondary_provider_failure(exc: Exception) -> bool:
    """Never include arbitrary persistence, programming, or Lean errors."""
    if isinstance(exc, OSError):
        return False
    return isinstance(exc, (RetryableProviderError, FreeModelExhaustedError,
                            ProviderContextLengthError, ProviderRepairRequiredError,
                            ProviderRouteError)) or is_provider_credit_pause_error(exc) or is_non_retryable_model_error(exc)


class _BatchBarrier:
    """Fill a cohort without refilling slots until every member has drained.

    A partial cohort closes when it drains, so sparse fallback queues never
    wait for candidates that succeeded on an earlier model.
    """

    def __init__(self, size):
        self.size = size
        self.condition = asyncio.Condition()
        self.admitted = 0
        self.active = 0

    async def __aenter__(self):
        async with self.condition:
            await self.condition.wait_for(lambda: self.size <= 0 or self.admitted < self.size)
            self.admitted += 1
            self.active += 1

    async def __aexit__(self, *exc):
        async with self.condition:
            self.active -= 1
            if not self.active:
                self.admitted = 0
                self.condition.notify_all()


class ProofCompetition:
    """Reusable chain driver. ``execute(index, route, state, save)`` is Phase A only.

    Index zero is primary; secondary indices preserve configuration order.
    ``classify`` returns success/exhausted/blocked/unavailable. Only exhausted
    primary outcomes permit handoff. Each route has independent concurrency.
    State is JSON-serializable and must be persisted by the owning checkpoint.
    """

    def __init__(self, *, config, scope: str, identity: dict, batch_size: int,
                 state=None, checkpoint=None, observer=None):
        data = config.model_dump(mode="json") if hasattr(config, "model_dump") else dict(config or {})
        self.routes = list(data.get("secondaries") or []) if scope == "autonomous" and data.get("enabled") is True else []
        self.enabled = bool(self.routes)
        self.identity = deepcopy(identity)
        self.state = deepcopy(state or {})
        self.checkpoint = checkpoint
        self.observer = observer
        self.gates = [_BatchBarrier(batch_size) for _ in self.routes]
        self.unavailable = set()
        self.lock = asyncio.Lock()

    async def _save(self, key, record):
        async with self.lock:
            self.state[key] = deepcopy(record)
            if self.checkpoint:
                await self.checkpoint()
            if self.observer:
                # Stable event identity allows durable benchmark upserts on replay.
                metadata = {name: record[name] for name in (
                    "competitor_index", "route_revision", "theorem_id", "candidate_fingerprint", "status",
                    "error_type", "message", "attempt_started", "effective_routes") if name in record}
                metadata["attempts_consumed"] = len(record.get("attempts", []))
                metadata["usage"] = None
                await self.observer({"event_id": fingerprint([key, metadata]), "execution_key": key,
                                     **self.identity, **metadata})

    async def run_candidate(self, candidate, *, execute, classify, restore,
                            primary_finished=None):
        candidate_fingerprint = fingerprint([candidate.theorem_id, candidate.statement, candidate.formal_sketch])
        execution_scope = fingerprint({k: v for k, v in self.identity.items() if k != "primary_route"})
        # A successful Lean artifact is authoritative even after route repair.
        # Resume registration, never spend a new model's attempts on it.
        for saved in list(self.state.values()):
            if (saved.get("candidate_fingerprint") == candidate_fingerprint
                    and saved.get("execution_scope", execution_scope) == execution_scope
                    and saved.get("status") == "success"):
                if primary_finished:
                    primary_finished()
                outcome = restore(saved)
                outcome.competition_winner = {**self.identity, **{name: saved[name] for name in (
                    "candidate_fingerprint", "competitor_index", "route_revision")}}
                return outcome
        last = None
        for index, route in enumerate([None, *self.routes]):
            route_data = route.model_dump(mode="json") if hasattr(route, "model_dump") else route
            revision = fingerprint(route_data if index else self.identity.get("primary_route"))
            legacy_key = fingerprint([self.identity, candidate.theorem_id, candidate.statement,
                                      candidate.formal_sketch, index, revision])
            key = (fingerprint([execution_scope, candidate_fingerprint, index, revision])
                   if index else legacy_key)
            saved_record = self.state.get(key)
            if index and saved_record is None:
                # Older keys included the primary route. Match the persisted
                # private execution identity even when that route has changed.
                saved_record = self.state.get(legacy_key)
                if saved_record is None:
                    saved_record = next((saved for saved in self.state.values()
                        if saved.get("execution_scope") == execution_scope
                        and saved.get("candidate_fingerprint") == candidate_fingerprint
                        and saved.get("competitor_index") == index
                        and saved.get("route_revision") == revision), None)
            record = deepcopy(saved_record or {"competitor_index": index,
                "route_revision": revision, "theorem_id": candidate.theorem_id,
                "execution_scope": execution_scope,
                "candidate_fingerprint": fingerprint([candidate.theorem_id, candidate.statement, candidate.formal_sketch]),
                "status": "eligible", "attempts": []})

            async def save(update, _key=key, _record=record):
                _record.update(deepcopy(update))
                await self._save(_key, _record)

            async def invoke():
                if record["status"] in {"success", "exhausted"}:
                    return restore(record)
                if index and (revision in self.unavailable or record["status"] == "unavailable"):
                    await save({"status": "unavailable", "error_type": "route_unavailable"})
                    return None
                await save({"status": "eligible"})
                # Only execute's provider failures are contained. Callback failures
                # from save are explicitly marked and re-raised below.
                callback_error = None
                async def guarded_save(update):
                    nonlocal callback_error
                    try:
                        await save(update)
                    except BaseException as exc:
                        callback_error = exc
                        raise
                try:
                    outcome = await execute(index, route, deepcopy(record), guarded_save)
                except Exception as exc:
                    if callback_error is not None or not index or not secondary_provider_failure(exc):
                        raise
                    self.unavailable.add(revision)
                    await save({"status": "unavailable", "error_type": type(exc).__name__,
                                "message": "Secondary proof model unavailable; research continues."})
                    return None
                status = classify(outcome, index)
                if index and status == "unavailable":
                    self.unavailable.add(revision)
                await save({"status": status, "outcome": {
                    "candidate": outcome.candidate.model_dump(mode="json"),
                    "success": outcome.success, "theorem_name": outcome.theorem_name,
                    "lean_code": outcome.lean_code,
                    "attempts": [a.model_dump(mode="json") for a in outcome.attempts],
                    "context_overflow_payload": outcome.context_overflow_payload,
                }})
                return outcome

            if index:
                async with self.gates[index - 1]:
                    outcome = await invoke()
            else:
                outcome = await invoke()
                if primary_finished:
                    primary_finished()
            if outcome is not None:
                status = classify(outcome, index)
                if status in {"success", "blocked"}:
                    outcome.competition_winner = {**self.identity, **{name: record[name] for name in (
                        "candidate_fingerprint", "competitor_index", "route_revision")}}
                    return outcome
                if status == "exhausted" and not index:
                    # Primary recovery policy and reusable failed-target hints must
                    # never inherit a secondary's truncation/feedback history.
                    last = outcome
        return last
