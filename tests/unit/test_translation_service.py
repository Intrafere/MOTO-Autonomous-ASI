"""Auto Top-p Translate service behavior with a faked model route.

Every test fakes ``api_client_manager.generate_completion`` and the retry seams,
binds a private runtime root, and stops the service before returning. The fake
translator reads the task, the document views, and the earlier attempts from
the real production prompt, so the tests exercise the real prompt builders.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest

from backend.api.routes import websocket as websocket_routes
from backend.shared.activity import normalize_activity_payload
from backend.shared.api_client_manager import ProviderCooldownError, RetryableProviderError, api_client_manager
from backend.shared.background_call_scope import current_background_service
from backend.shared.config import bind_runtime_roots, system_config
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.openrouter_client import CreditExhaustionError, FreeModelExhaustedError
from backend.shared.provider_errors import ProviderRepairRequiredError, ProviderRouteError, ProviderRouteIdentity
from backend.shared.token_tracker import token_tracker
from backend.shared.translation import discovery, history
from backend.shared.translation import service as service_module
from backend.shared.translation.contracts import STATUS_ACCEPTED, STATUS_CHECK_FAILED, STATUS_KEPT, STATUS_REJECTED
from backend.shared.translation.discovery import (
    CurrentRunState,
    DiscoveryResult,
    DocumentSpec,
    proof_document_specs,
)
from backend.shared.translation.segmentation import PROFILE_DOCUMENT, build_document_plan, build_unit_views
from backend.shared.translation.settings import (
    AutoTranslateSettings,
    normalize_settings,
    save_auto_translate_settings,
)
from backend.shared.translation.store import SegmentTranslation, translation_store

pytestmark = pytest.mark.asyncio

SOURCE_TEXT = "First paragraph with math $x^2$ here.\n\nSecond paragraph about the results."
MASKED_FIRST = "First paragraph with math [[M1]] here."
THREE_PARAGRAPHS = "Alpha paragraph text.\n\nBeta paragraph text.\n\nGamma paragraph text."
TWO_LANGUAGES = [{"key": "es", "name": "Spanish"}, {"key": "fr", "name": "French"}]


# ----------------------------------------------------------------------
# Prompt readers
# ----------------------------------------------------------------------


def _all_text(record: Dict[str, Any]) -> str:
    return "\n".join(str(message.get("content") or "") for message in record["messages"])


def _block(record: Dict[str, Any], name: str) -> str:
    match = re.search(rf"<<<{name}\n(.*?)\n{name}>>>", _all_text(record), re.S)
    assert match, f"prompt has no {name} block"
    return match.group(1)


def _units(view: str) -> Dict[int, str]:
    units: Dict[int, str] = {}
    for item in view.split("\n\n"):
        match = re.match(r"<<U(\d+)>> (.*)", item, re.S)
        if match:
            units[int(match.group(1))] = match.group(2)
    return units


def _source_units(record: Dict[str, Any]) -> Dict[int, str]:
    return _units(_block(record, "SOURCE"))


def _translation_units(record: Dict[str, Any]) -> Dict[int, str]:
    return _units(_block(record, "TRANSLATION"))


def _task(record: Dict[str, Any]) -> Dict[str, Any]:
    """The translator task from the prompt's final reminder."""
    reminder = _all_text(record).rsplit("\n\n", 1)[-1]
    match = re.match(r"Translate units starting at U(\d+), ending no later than U(\d+),", reminder)
    if match:
        return {"kind": "first", "start": int(match.group(1)), "end": int(match.group(2))}
    match = re.match(r"Improve the part starting at U(\d+),", reminder)
    if match:
        end = re.search(r"ends no later than U(\d+)\.", _all_text(record))
        return {"kind": "retry", "start": int(match.group(1)), "end": int(end.group(1))}
    match = re.match(r"Improve units U(\d+) to U(\d+),", reminder)
    assert match, "translator prompt has no task reminder"
    return {"kind": "locked", "start": int(match.group(1)), "end": int(match.group(2))}


def _attempts_section(record: Dict[str, Any]) -> str:
    text = _all_text(record)
    if "EARLIER ATTEMPTS AT THIS PART (oldest first)\n" not in text:
        return ""
    return text.split("EARLIER ATTEMPTS AT THIS PART (oldest first)\n", 1)[1].split("\n\nSOURCE DOCUMENT", 1)[0]


def _accepted_numbers(record: Dict[str, Any]) -> List[int]:
    return [int(number) for number in re.findall(r"\[Attempt (\d+)\] Accepted", _attempts_section(record))]


def _attempt_numbers(record: Dict[str, Any]) -> List[int]:
    return [int(number) for number in re.findall(r"\[Attempt (\d+)\]", _attempts_section(record))]


def _judged(record: Dict[str, Any]) -> List[int]:
    match = re.search(r"one decision for each of attempts ([\d, ]+)\.$", _all_text(record))
    assert match, "validator prompt has no attempt list"
    return [int(number) for number in match.group(1).split(", ")]


def _target_language(record: Dict[str, Any]) -> str:
    match = re.search(r"TARGET LANGUAGE: (\w+)", _all_text(record))
    if match:
        return match.group(1)
    return "Spanish" if "into Spanish" in _all_text(record) else "French"


# ----------------------------------------------------------------------
# Model responses
# ----------------------------------------------------------------------


def _response(content: str, *, finish: str = "stop") -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def _translate(units: Dict[int, str], *, edits: Optional[List[Dict[str, Any]]] = None, notes: str = "") -> Dict[str, Any]:
    return _response(
        json.dumps(
            {
                "action": "translate",
                "units": [{"unit": number, "translation": text} for number, text in sorted(units.items())],
                "edits": edits or [],
                "translator_notes": notes,
            }
        )
    )


def _keep(numbers: List[int], notes: str = "") -> Dict[str, Any]:
    return _response(json.dumps({"action": "keep", "keep_attempts": numbers, "translator_notes": notes}))


def _translated_part(record: Dict[str, Any], *, suffix: str = "(es)", end: Optional[int] = None) -> Dict[int, str]:
    """Translate the task's units: the source plus a suffix, through ``end`` when given."""
    task = _task(record)
    source = _source_units(record)
    stop = task["end"] if end is None else min(task["end"], max(end, task["start"]))
    return {number: f"{source[number]} {suffix}" for number in range(task["start"], stop + 1)}


def _translate_ok(record: Dict[str, Any], notes: str = "") -> Dict[str, Any]:
    """Translate the whole allowed part, then keep the accepted attempt twice."""
    if _task(record)["kind"] == "locked":
        return _keep(_accepted_numbers(record)[-1:])
    suffix = {"Spanish": "(es)", "French": "(fr)"}.get(_target_language(record), "(xx)")
    return _translate(_translated_part(record, suffix=suffix), notes=notes)


def _one_unit_parts(record: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one unit per part, then keep it twice."""
    if _task(record)["kind"] == "locked":
        return _keep(_accepted_numbers(record)[-1:])
    task = _task(record)
    return _translate(_translated_part(record, end=task["start"]))


def _decisions(numbers: List[int], *, accept: bool = True, feedback: str = "Fix the terminology.") -> Dict[str, Any]:
    return _response(
        json.dumps(
            {
                "decisions": [
                    {
                        "attempt": number,
                        "decision": "accept" if accept else "reject",
                        "reasoning": "Faithful translation." if accept else "Terminology problem.",
                        "feedback": "" if accept else feedback,
                    }
                    for number in numbers
                ]
            }
        )
    )


def _accept_all(record: Dict[str, Any]) -> Dict[str, Any]:
    return _decisions(_judged(record))


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------


@pytest.fixture
def runtime_root(tmp_path):
    original_data = system_config.data_dir
    original_logs = system_config.logs_dir
    bind_runtime_roots(tmp_path / "data", tmp_path / "logs")
    yield tmp_path / "data"
    bind_runtime_roots(original_data, original_logs)


def _role(model_id: str = "model-a", provider: str = "openrouter", **overrides: Any) -> Dict[str, Any]:
    payload = {
        "provider": provider,
        "model_id": model_id,
        "context_window": 32000,
        "max_output_tokens": 4000,
    }
    payload.update(overrides)
    return payload


def _settings(
    *,
    translator: Optional[Dict[str, Any]] = None,
    validator: Optional[Dict[str, Any]] = None,
    languages: Optional[List[Dict[str, Any]]] = None,
    enabled: bool = True,
    paused: bool = False,
    batch: bool = False,
    modes: Optional[List[str]] = None,
    translate_all: bool = False,
) -> AutoTranslateSettings:
    payload: Dict[str, Any] = {
        "enabled": enabled,
        "paused": paused,
        "translate_all": translate_all,
        "batch_translation": batch,
        "languages": languages if languages is not None else [{"key": "es", "name": "Spanish"}],
        "translator": translator if translator is not None else _role("translator-a"),
        "validator": validator if validator is not None else _role("validator-model"),
    }
    if modes is not None:
        payload["modes"] = modes
    return AutoTranslateSettings.model_validate(payload)


def _doc(
    doc_key: str = "paper:test:p1",
    text: str = SOURCE_TEXT,
    *,
    kind: str = "paper",
    title: str = "Test paper",
    profile: str = "document",
) -> Dict[str, Any]:
    return {
        "doc_key": doc_key,
        "kind": kind,
        "title": title,
        "profile": profile,
        "parts": [{"name": "content", "text": text}],
    }


def _spec(doc: Dict[str, Any], source: str = "test_source") -> DocumentSpec:
    return replace(service_module.coerce_document_input(doc), source=source)


async def discover(service, *docs: Dict[str, Any]) -> List[str]:
    """Target ``docs`` as one current-run discovery scan does; returns their doc hashes."""
    specs = [_spec(doc) for doc in docs]
    await service._apply_discovery(DiscoveryResult(documents=tuple(specs)))
    return [service._doc_by_key[spec.doc_key] for spec in specs]


async def add_older_outputs(service, *docs: Dict[str, Any]) -> List[str]:
    """Target ``docs`` as one Translate all enumeration does (Translate all must be on)."""
    specs = [_spec(doc, "history_papers") for doc in docs]
    await service._apply_backlog(DiscoveryResult(documents=tuple(specs)))
    return [service._doc_by_key[spec.doc_key] for spec in specs]


def _keys_for_hash(store, doc_hash):
    """Doc keys registered on ``doc_hash``, sorted by key."""
    return sorted(
        (record for record in store.load_documents() if record.doc_hash == doc_hash),
        key=lambda record: record.doc_key,
    )


def _stored_documents() -> Dict[str, Any]:
    """Every record in the translation store, keyed by doc key."""
    return {record.doc_key: record for record in translation_store.load_documents()}


class FakeModel:
    """Records every completion call and answers by role."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.translator: Callable[[Dict[str, Any]], Any] = _translate_ok
        self.validator: Callable[[Dict[str, Any]], Any] = _accept_all
        self.probe: Callable[[Dict[str, Any]], Any] = lambda record: _response('{"languages": []}')
        self.track_tokens = False

    def role_calls(self, role_id: str) -> List[Dict[str, Any]]:
        return [call for call in self.calls if call["role_id"] == role_id]

    def translator_calls(self) -> List[Dict[str, Any]]:
        return self.role_calls("translation_submitter")

    def validator_calls(self) -> List[Dict[str, Any]]:
        return self.role_calls("translation_validator")

    async def __call__(
        self,
        task_id: str,
        role_id: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        record = {
            "task_id": task_id,
            "role_id": role_id,
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": dict(kwargs),
            "scope": current_background_service(),
        }
        self.calls.append(record)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for _ in range(3):
                await asyncio.sleep(0)
            if self.track_tokens:
                token_tracker.track(model, 100, 50)
            if role_id == "translation_validator":
                handler = self.validator
            elif role_id.startswith("translation_language_probe_"):
                handler = self.probe
            else:
                handler = self.translator
            result = handler(record)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            self.in_flight -= 1


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fake = FakeModel()
        self.events: List[tuple[str, Dict[str, Any]]] = []
        self.retry_waits: List[Dict[str, Any]] = []
        self.provider_waits: List[Dict[str, Any]] = []
        self.service = service_module.TranslationService()

        async def no_discovery(_modes) -> DiscoveryResult:
            return DiscoveryResult(documents=())

        async def no_older_outputs(_modes) -> DiscoveryResult:
            return DiscoveryResult(documents=())

        self.service._discover = no_discovery
        self.service._enumerate_backlog = no_older_outputs

        async def fake_broadcast(event_type: str, data: Dict[str, Any]) -> None:
            self.events.append((event_type, dict(data)))

        async def fake_emit(payload: Dict[str, Any], *, role_id: str = "", should_stop=None, error=None) -> bool:
            self.retry_waits.append({"payload": dict(payload), "role_id": role_id, "error": error})
            await asyncio.sleep(0)
            return False

        async def fake_wait(error, *, role_id: str = "", should_stop=None, activity_callback=None, workflow_mode: str = "") -> None:
            self.provider_waits.append({"error": error, "role_id": role_id, "workflow_mode": workflow_mode})
            await asyncio.sleep(0)

        monkeypatch.setattr(api_client_manager, "generate_completion", self.fake)
        monkeypatch.setattr(api_client_manager, "emit_and_wait_for_retry", fake_emit)
        monkeypatch.setattr(api_client_manager, "wait_for_retryable_provider_error", fake_wait)
        monkeypatch.setattr(websocket_routes, "broadcast_event", fake_broadcast)

    async def start(self, settings: AutoTranslateSettings) -> service_module.TranslationService:
        save_auto_translate_settings(normalize_settings(settings, generic_mode=False))
        await self.service.start()
        return self.service

    def event_names(self) -> List[str]:
        return [name for name, _payload in self.events]

    def events_named(self, name: str) -> List[Dict[str, Any]]:
        return [payload for event, payload in self.events if event == name]


@pytest.fixture
def harness(runtime_root, monkeypatch):
    return Harness(monkeypatch)


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


async def settle(iterations: int = 40) -> None:
    for _ in range(iterations):
        await asyncio.sleep(0)


async def lookup_status(service, doc_hash: str, language: str = "es") -> Dict[str, Any]:
    return (await service.lookup_documents(language, [doc_hash]))[0]


def _complete(service, doc_hash: str, language: str = "es") -> bool:
    doc = service._docs.get(doc_hash)
    return (
        doc is not None
        and service._final_counts.get((doc_hash, language), 0) >= doc.unit_count
        and (doc_hash, language) not in service._jobs
    )


async def register_and_complete(
    service, doc: Optional[Dict[str, Any]] = None, language: str = "es"
) -> Dict[str, Any]:
    row = (await service.register_documents(language, True, [doc or _doc()]))[0]
    await wait_until(lambda: _complete(service, row["doc_hash"], language))
    return await lookup_status(service, row["doc_hash"], language)


def _job_state(service, doc_hash: str, language: str = "es") -> Dict[str, Any]:
    return translation_store.get_job_state(doc_hash, language)


# ----------------------------------------------------------------------
# Settings and roles
# ----------------------------------------------------------------------


async def test_settings_apply_configures_exact_role_ids(harness):
    service = await harness.start(_settings(translator=_role("translator-a"), validator=_role("judge")))
    try:
        translator = api_client_manager.get_role_config("translation_submitter")
        assert translator.model_id == "translator-a"
        assert translator.supercharge_enabled is False
        validator = api_client_manager.get_role_config("translation_validator")
        assert validator.model_id == "judge"
        assert validator.supercharge_enabled is False

        await service.update_settings(_settings(translator=_role("translator-c"), validator=_role("judge-2")))
        assert api_client_manager.get_role_config("translation_submitter").model_id == "translator-c"
        assert api_client_manager.get_role_config("translation_validator").model_id == "judge-2"
        assert service.get_settings().translator.model_id == "translator-c"
    finally:
        await service.shutdown()


async def test_update_settings_rejects_invalid_language_without_persisting(harness):
    service = await harness.start(_settings())
    try:
        bad = _settings(languages=[{"key": "xx-Nope", "name": "Nope"}])
        with pytest.raises(ValueError):
            await service.update_settings(bad)
        assert [language.key for language in service.get_settings().languages] == ["es"]
    finally:
        await service.shutdown()


async def test_settings_update_keeps_enabled_and_paused_but_applies_batch(harness):
    service = await harness.start(_settings(enabled=True, paused=True))
    try:
        status = await service.update_settings(_settings(enabled=False, paused=False, batch=True))
        assert (status["enabled"], status["paused"]) == (True, True)
        settings = service.get_settings()
        assert (settings.enabled, settings.paused, settings.batch_translation) == (True, True, True)
        assert status["batch_translation"] is True
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# Registration, the part loop, and assembly
# ----------------------------------------------------------------------


async def test_register_documents_returns_pending_then_complete(harness):
    service = await harness.start(_settings(paused=True))
    try:
        rows = await service.register_documents("es", True, [_doc()])
        assert rows[0]["status"] == "pending"
        assert rows[0]["parts"] is None
        assert rows[0]["total_units"] == 2
        assert rows[0]["translated_units"] == 0
        assert rows[0]["queue_position"] == 1
        assert set(rows[0]) == {
            "doc_key", "doc_hash", "language", "status", "translated_units", "total_units",
            "queue_position", "parts", "requested", "automatic",
        }

        await service.control("resume")
        await wait_until(lambda: _complete(service, rows[0]["doc_hash"]))
        status = await lookup_status(service, rows[0]["doc_hash"])
        assert status["status"] == "complete"
        assert status["queue_position"] is None
        assert status["translated_units"] == status["total_units"] == 2
        # The document is assembled per occurrence; math is unmasked.
        assert status["parts"] == {
            "content": "First paragraph with math $x^2$ here. (es)\n\nSecond paragraph about the results. (es)"
        }
        doc = service._docs[rows[0]["doc_hash"]]
        stored = translation_store.get_translations([doc.views[0].source_hash], "es")
        assert stored[doc.views[0].source_hash] == f"{MASKED_FIRST} (es)"
        state = _job_state(service, rows[0]["doc_hash"])
        assert state["version"] == 1 and state["part"] is None
        assert set(state["finals"]) == {"0", "1"}

        finalized = harness.events_named("translation_part_finalized")
        assert len(finalized) == 1
        assert finalized[0]["reason"] == "kept_twice" and finalized[0]["final_attempt"] == 1
        assert (finalized[0]["part_start"], finalized[0]["part_end"]) == (1, 2)
        progress = harness.events_named("translation_progress")[-1]
        assert progress == {**progress, "language_key": "es", "doc_hashes": [rows[0]["doc_hash"]],
                            "translated_units": 2, "total_units": 2}
        snapshot = service.status_snapshot()
        assert snapshot["state"] == "idle" and snapshot["status_label"] == "ready"
        assert snapshot["languages"][0]["complete_documents"] == 1
        assert snapshot["usage"]["finalized_parts"] == 1
    finally:
        await service.shutdown()


async def test_empty_document_is_reported_empty_with_identity_parts(harness):
    service = await harness.start(_settings())
    try:
        rows = await service.register_documents("es", True, [_doc(text="$$x+y$$\n\n---\n")])
        assert rows[0]["status"] == "empty"
        assert rows[0]["parts"] == {"content": "$$x+y$$\n\n---\n"}
        await settle()
        assert harness.fake.calls == []
    finally:
        await service.shutdown()


async def test_status_snapshot_shape_matches_contract(harness):
    service = await harness.start(_settings(paused=True))
    try:
        await service.register_documents("es", True, [_doc()])
        snapshot = service.status_snapshot()
        assert set(snapshot) == {
            "success", "enabled", "paused", "batch_translation", "modes", "max_parallel_jobs", "state",
            "status_label", "message", "generation", "languages", "queue", "active", "repair",
            "translate_all", "usage", "recent_activity",
        }
        assert snapshot["modes"] == ["autonomous", "aggregator", "compiler"]
        assert snapshot["translate_all"] == {
            "requested": False,
            "state": "off",
            "documents": 0,
            "complete_documents": 0,
            "translated_units": 0,
            "pending_units": 0,
            "message": "",
        }
        assert set(snapshot["usage"]) == {
            "calls", "prompt_tokens", "completion_tokens", "accepted_attempts", "rejected_attempts",
            "finalized_parts", "check_failures",
        }
        assert set(snapshot["languages"][0]) == {
            "key", "name", "custom", "translated_units", "pending_units", "pending_documents",
            "complete_documents", "background_pending_units", "background_pending_documents",
        }
        assert snapshot["languages"][0]["pending_units"] == 2
        assert set(snapshot["queue"][0]) == {
            "doc_hash", "doc_key", "kind", "title", "language_key", "position", "viewing", "background",
            "translated_units", "total_units",
        }
        assert snapshot["queue"][0]["background"] is False
        assert snapshot["queue"][0]["position"] == 1
        assert snapshot["batch_translation"] is False and snapshot["max_parallel_jobs"] == 1
        summary = service.connectivity_summary()
        assert set(summary) == {"status", "enabled", "active", "message", "languages", "pending_units"}
        assert summary["languages"] == ["Spanish"]
        assert summary["pending_units"] == 2
    finally:
        await service.shutdown()


async def test_active_rows_show_the_part_attempt_and_counts(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()
    calls: List[Dict[str, Any]] = []

    async def translator(record):
        calls.append(record)
        if len(calls) == 2:
            await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: len(calls) == 2)
        [active] = service.status_snapshot()["active"]
        assert active == {
            "doc_hash": row["doc_hash"],
            "doc_title": "Test paper",
            "language_key": "es",
            "part_start": 1,
            "part_end": 2,
            "attempt": 2,
            "accepted_count": 1,
            "keep_streak": 0,
        }
        assert service.status_snapshot()["queue"][0]["position"] == 0
        assert (await lookup_status(service, row["doc_hash"]))["queue_position"] == 0
        gate.set()
        await wait_until(lambda: _complete(service, row["doc_hash"]))
    finally:
        gate.set()
        await service.shutdown()


# ----------------------------------------------------------------------
# Choosing the next part
# ----------------------------------------------------------------------


async def test_part_max_end_follows_the_run_and_the_output_budget():
    # budget = 0.8 * 540 - 300 = 132; each 10-token unit costs ceil(25) + 40 = 65.
    assert service_module.part_max_end([10, 10, 10, 10], {}, 0, 540) == 1
    # The part stops at the next final translation.
    assert service_module.part_max_end([10, 10, 10, 10], {"1": "x"}, 0, 100_000) == 0
    assert service_module.part_max_end([10, 10, 10, 10], {"0": "x"}, 1, 100_000) == 3
    # Only the output budget limits the source length: no fixed source cap.
    assert service_module.part_max_end([600, 600, 600, 600], {}, 0, 100_000) == 3
    # The part, shown with four translations of it, fits the input budget less its
    # reserve: each 600-token unit needs 600 + 4 * (1500 + 40) = 6760 input tokens.
    assert service_module.part_max_end([600, 600, 600, 600], {}, 0, 100_000, 4000 + 2 * 6760) == 1
    assert service_module.part_max_end([600, 600, 600, 600], {}, 0, 100_000, 4000 + 2 * 6760 - 1) == 0
    # One unit is always allowed the input budget.
    assert service_module.part_max_end([600, 10], {}, 0, 100_000, 1000) == 0
    # A unit whose estimated output does not fit cannot start a part.
    assert service_module.part_max_end([100], {}, 0, 300) is None


async def test_progress_percent_stays_below_100_until_everything_is_done():
    assert service_module.progress_percent(0, 4851) == 0
    assert service_module.progress_percent(1815, 4851) == 37
    assert service_module.progress_percent(4850, 4851) == 99
    assert service_module.progress_percent(4851, 4851) == 100
    assert service_module.progress_percent(0, 0) == 100


async def test_the_translator_chooses_the_part_end_and_the_next_part_starts_at_the_cursor(harness):
    service = await harness.start(_settings())
    text = "\n\n".join(f"Paragraph {name} about lattices." for name in ("one", "two", "three", "four"))
    harness.fake.translator = lambda record: (
        _keep(_accepted_numbers(record)[-1:])
        if _task(record)["kind"] == "locked"
        else _translate(_translated_part(record, end=_task(record)["start"] + 1))
    )
    try:
        status = await register_and_complete(service, _doc(text=text))
        assert status["status"] == "complete"
        first = [call for call in harness.fake.translator_calls() if _task(call)["kind"] == "first"]
        assert [(_task(call)["start"], _task(call)["end"]) for call in first] == [(1, 4), (3, 4)]
        finalized = harness.events_named("translation_part_finalized")
        assert [(event["part_start"], event["part_end"]) for event in finalized] == [(1, 2), (3, 4)]
    finally:
        await service.shutdown()


async def test_the_output_budget_limits_how_far_a_part_may_reach(harness):
    service = await harness.start(_settings(translator=_role("translator-a", max_output_tokens=540)))
    text = "\n\n".join(f"Paragraph {name} about lattices." for name in ("one", "two", "three", "four"))
    try:
        row = (await service.register_documents("es", True, [_doc(text=text)]))[0]
        await wait_until(lambda: bool(harness.fake.translator_calls()))
        doc = service._docs[row["doc_hash"]]
        expected = service_module.part_max_end(doc.token_counts, {}, 0, 540)
        assert expected is not None and expected < 3
        assert _task(harness.fake.translator_calls()[0]) == {"kind": "first", "start": 1, "end": expected + 1}
        await wait_until(lambda: _complete(service, row["doc_hash"]))
    finally:
        await service.shutdown()


async def test_a_unit_whose_output_cannot_fit_enters_repair_without_calls(harness):
    service = await harness.start(_settings(translator=_role("tiny-output", max_output_tokens=300)))
    try:
        await service.register_documents("es", True, [_doc()])
        await wait_until(lambda: service.status_snapshot()["state"] == "repair_required")
        repair = service.status_snapshot()["repair"]
        assert repair["reason"] == "max_output_tokens_too_small"
        assert repair["role_id"] == "translation_submitter"
        assert "too small to translate the next paragraph" in repair["message"]
        assert harness.fake.calls == []
    finally:
        await service.shutdown()


async def test_translator_sees_the_full_source_and_the_translation_so_far(harness):
    service = await harness.start(_settings())
    harness.fake.translator = _one_unit_parts
    try:
        await register_and_complete(service, _doc(text=f"{SOURCE_TEXT}\n\nThird paragraph closes."))
        second_part = [call for call in harness.fake.translator_calls() if _task(call)["start"] == 2][0]
        assert _source_units(second_part) == {
            1: MASKED_FIRST,
            2: "Second paragraph about the results.",
            3: "Third paragraph closes.",
        }
        assert _translation_units(second_part) == {
            1: f"{MASKED_FIRST} (es)",
            2: "(not translated yet)",
            3: "(not translated yet)",
        }
        # The validator sees the same views.
        validation = harness.fake.validator_calls()[1]
        assert _translation_units(validation)[1] == f"{MASKED_FIRST} (es)"
    finally:
        await service.shutdown()


async def test_views_are_windowed_around_the_part_when_the_document_does_not_fit(harness):
    text = "\n\n".join(
        f"Paragraph number {index} discusses the lattice structure and its symmetry group in detail."
        for index in range(1, 61)
    )
    service = await harness.start(
        _settings(translator=_role("small", context_window=3200, max_output_tokens=1000))
    )
    try:
        row = (await service.register_documents("es", True, [_doc(text=text)]))[0]
        await wait_until(lambda: bool(harness.fake.translator_calls()))
        first = harness.fake.translator_calls()[0]
        task = _task(first)
        assert task["start"] == 1
        prompt = _all_text(first)
        assert "units omitted" in prompt
        shown = _source_units(first)
        assert set(range(task["start"], task["end"] + 1)) <= set(shown)
        assert 60 not in shown
        assert service_module.count_tokens(prompt) <= 3200 - 1000
        await wait_until(lambda: len(harness.events_named("translation_part_finalized")) >= 1)
        assert service._final_counts[(row["doc_hash"], "es")] == task["end"]
    finally:
        await service.shutdown()


WINDOW_TEXT = "\n\n".join(
    f"Paragraph number {index} discusses the lattice structure and its symmetry group in detail."
    for index in range(1, 61)
)
LONG_PARAGRAPH = " ".join(["The lattice structure has a symmetry group of finite order."] * 40)


def _seed_final_translations(text: str, count: int) -> None:
    """Store final Spanish translations of the document's first ``count`` units in the cache."""
    plan = build_document_plan(PROFILE_DOCUMENT, [("content", text)])
    masked = {segment.source_hash: segment.masked_text for segment in plan.segments}
    for view in build_unit_views(plan)[:count]:
        translation_store.save_translation(
            SegmentTranslation(view.source_hash, "es", f"{masked[view.source_hash]} (es)", "",
                               "translation_submitter", "m", "v", 1, "2026-01-01T00:00:00+00:00")
        )


async def test_the_validator_window_always_shows_the_units_an_attempt_edits(harness):
    _seed_final_translations(WINDOW_TEXT, 54)
    service = await harness.start(
        _settings(validator=_role("small-judge", context_window=3000, max_output_tokens=500))
    )

    def translator(record):
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        edits = [{"unit": 1, "find": "number 1 discusses", "replace": "número 1 discusses"}]
        return _translate(_translated_part(record, end=task["start"]), edits=edits if task["start"] == 55 else None)

    harness.fake.translator = translator
    try:
        status = await register_and_complete(service, _doc(text=WINDOW_TEXT))
        first = harness.fake.validator_calls()[0]
        assert "units omitted" in _all_text(first)
        shown = _translation_units(first)
        assert {1, 55} <= set(shown) and 2 not in shown
        assert shown[1] == "Paragraph number 1 discusses the lattice structure and its symmetry group in detail. (es)"
        assert status["parts"]["content"].startswith("Paragraph número 1 discusses")
    finally:
        await service.shutdown()


async def test_edits_the_validator_cannot_be_shown_fail_the_automatic_checks(harness):
    text = f"{LONG_PARAGRAPH}\n\nBeta paragraph text.\n\nGamma paragraph text."
    _seed_final_translations(text, 1)
    service = await harness.start(
        _settings(validator=_role("small-judge", context_window=2000, max_output_tokens=500))
    )
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        units = _translated_part(record, end=task["start"])
        if task["start"] == 2 and task["kind"] == "first":
            return _translate(units, edits=[{"unit": 1, "find": "(es)", "replace": "(ES)"}])
        return _translate(units)

    harness.fake.translator = translator
    try:
        status = await register_and_complete(service, _doc(text=text))
        [check] = harness.events_named("translation_attempt_check_failed")
        assert check["round_index"] == 1 and check["part_start"] == 2
        assert check["problems"][0].startswith(
            "The validator's context window cannot show the part together with U1, which this attempt edits"
        )
        # The attempt never reached the validator; the retry without the edit did.
        assert not any("(ES)" in _judge_section(call) for call in harness.fake.validator_calls())
        assert "[Attempt 1] Failed the automatic checks" in _attempts_section(calls[1])
        assert status["parts"]["content"].startswith(f"{LONG_PARAGRAPH} (es)\n\nBeta")
    finally:
        await service.shutdown()


async def test_a_dispute_whose_edits_cannot_be_shown_fails_the_automatic_checks(harness):
    text = f"{LONG_PARAGRAPH}\n\nBeta paragraph text.\n\nGamma paragraph text."
    _seed_final_translations(text, 1)
    service = await harness.start(_settings())
    disputed = asyncio.Event()
    calls: List[Dict[str, Any]] = []

    async def translator(record):
        calls.append(record)
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        units = _translated_part(record, end=task["start"])
        section = _attempts_section(record)
        if task["start"] != 2 or "could not be judged again" in section:
            return _translate(units)
        if not section:
            return _translate(units, edits=[{"unit": 1, "find": "(es)", "replace": "(ES)"}])
        await disputed.wait()
        return _keep([1], notes="The capitals are the house style.")

    def validator(record):
        return _decisions(_judged(record), accept="(ES)" not in _judge_section(record), feedback="Keep (es).")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        row = (await service.register_documents("es", True, [_doc(text=text)]))[0]
        await wait_until(lambda: bool(harness.events_named("translation_attempt_rejected")))
        # The validator's context shrinks before the translator disputes the rejection.
        small = _role("small-judge", context_window=2000, max_output_tokens=500)
        await service.update_settings(_settings(validator=small))
        disputed.set()
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        [check] = harness.events_named("translation_attempt_check_failed")
        assert check["round_index"] == 2
        assert check["problems"][0].startswith(
            "The validator's context window cannot show the part together with U1, which the attempts you named edit"
        )
        assert len([call for call in harness.fake.validator_calls() if "(ES)" in _judge_section(call)]) == 1
        assert not any(event["dispute"] for event in harness.events_named("translation_attempt_kept"))
    finally:
        await service.shutdown()


async def test_a_windowed_prompt_keeps_as_much_feedback_as_fits(harness):
    service = await harness.start(
        _settings(translator=_role("small", context_window=3600, max_output_tokens=1000))
    )
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if _task(record)["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        return _translate(_translated_part(record, suffix=f"(v{len(calls)})"))

    def validator(record):
        number = _judged(record)[0]
        return _decisions([number], accept=number > 3, feedback=f"fix-{number}")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await service.register_documents("es", True, [_doc(text=WINDOW_TEXT)])
        await wait_until(lambda: len(calls) >= 4)
        fourth = _all_text(calls[3])
        assert "units omitted" in fourth
        # Not only the newest feedback entry: every rejection that fits the window is kept.
        assert all(f"fix-{number}" in fourth for number in (1, 2, 3))
        assert service_module.count_tokens(fourth) <= 3600 - 1000 - 500
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# Attempts: serial calls, feedback, keeps, disputes, edits
# ----------------------------------------------------------------------


async def test_attempts_run_strictly_in_series_at_temperature_zero(harness):
    service = await harness.start(_settings(languages=TWO_LANGUAGES))
    try:
        rows = await service.register_documents(None, True, [_doc("paper:a"), _doc("paper:b", THREE_PARAGRAPHS)])
        await wait_until(
            lambda: all(_complete(service, row["doc_hash"], language) for row in rows for language in ("es", "fr"))
        )
        assert harness.fake.max_in_flight == 1
        assert harness.fake.calls
        assert all(call["temperature"] == 0.0 for call in harness.fake.calls)
    finally:
        await service.shutdown()


async def test_rejection_feedback_and_earlier_attempts_reach_the_next_attempt(harness):
    service = await harness.start(_settings())
    translator_calls: List[Dict[str, Any]] = []
    validator_calls: List[Dict[str, Any]] = []

    def translator(record):
        translator_calls.append(record)
        return _translate(_translated_part(record, suffix=f"(v{len(translator_calls)})"))

    def validator(record):
        validator_calls.append(record)
        number = _judged(record)[0]
        if number in (1, 8, 9):
            return _decisions([number])
        return _decisions([number], accept=False, feedback=f"fix-{number}")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        status = await register_and_complete(service)
        assert status["status"] == "complete"
        assert len(translator_calls) == 9
        eighth = _all_text(translator_calls[7])
        # The accepted attempt is always shown; rejected ones are the newest five whole entries.
        assert "[Attempt 1] Accepted" in eighth
        assert "fix-2" not in eighth
        assert all(f"fix-{number}" in eighth for number in range(3, 8))
        assert _attempt_numbers(translator_calls[7]) == [1, 3, 4, 5, 6, 7]
        # The validator sees the newest three earlier feedback entries and the accepted attempt.
        eighth_validation = _all_text(validator_calls[7])
        assert "fix-4" not in eighth_validation
        assert all(f"fix-{number}" in eighth_validation for number in (5, 6, 7))
        assert "ACCEPTED TRANSLATION OF THIS PART\n[Attempt 1]" in eighth_validation
        # Three accepted attempts finalize the part with the latest one.
        finalized = harness.events_named("translation_part_finalized")[0]
        assert (finalized["reason"], finalized["final_attempt"]) == ("three_accepted", 9)
        assert status["parts"]["content"].endswith("results. (v9)")
        usage = service.status_snapshot()["usage"]
        assert (usage["accepted_attempts"], usage["rejected_attempts"]) == (3, 6)
        assert len(harness.events_named("translation_attempt_rejected")) == 6
    finally:
        await service.shutdown()


async def test_a_fresh_rejection_of_a_disputed_attempt_is_the_newest_feedback(harness):
    service = await harness.start(_settings())
    translator_calls: List[Dict[str, Any]] = []
    validator_calls: List[Dict[str, Any]] = []

    def translator(record):
        translator_calls.append(record)
        if _task(record)["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        if len(translator_calls) == 7:
            return _keep([1], notes="Attempt 1 uses the standard term.")
        return _translate(_translated_part(record, suffix=f"(v{len(translator_calls)})"))

    def validator(record):
        validator_calls.append(record)
        if len(validator_calls) <= 6:
            return _decisions(_judged(record), accept=False, feedback=f"fix-{_judged(record)[0]}")
        if len(validator_calls) == 7:
            return _decisions(_judged(record), accept=False, feedback="dispute-rejected")
        return _accept_all(record)

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service)
        eighth = translator_calls[7]
        # Judgment order: attempt 1's fresh rejection comes last, after the keep that disputed it,
        # and the newest five rejections are attempts 3-6 and 1.
        assert _attempt_numbers(eighth) == [3, 4, 5, 6, 7, 1]
        section = _attempts_section(eighth)
        assert "[Attempt 7] Kept earlier attempts 1\nTranslator notes: Attempt 1 uses the standard term." in section
        assert section.endswith("Feedback: dispute-rejected")
        assert "[Attempt 1] Rejected by the validator (judged again)" in section
        assert "fix-2" not in section
        # The validator's earlier feedback follows the same order: the newest three are 5, 6, and 1.
        earlier = _all_text(validator_calls[7]).split("EARLIER FEEDBACK ON THIS PART (newest last)\n", 1)[1]
        earlier = earlier.split("\n\nSOURCE DOCUMENT", 1)[0]
        assert re.findall(r"\[Attempt (\d+)\]", earlier) == ["5", "6", "1"]
        assert earlier.endswith("Feedback: dispute-rejected")
    finally:
        await service.shutdown()


async def test_three_accepted_attempts_finalize_with_the_latest(harness):
    service = await harness.start(_settings())
    calls: List[int] = []

    def translator(record):
        calls.append(1)
        return _translate(_translated_part(record, suffix=f"(take {len(calls)})"))

    harness.fake.translator = translator
    try:
        status = await register_and_complete(service)
        assert len(harness.fake.translator_calls()) == 3
        assert len(harness.fake.validator_calls()) == 3
        assert [_task(call)["kind"] for call in harness.fake.translator_calls()] == ["first", "locked", "locked"]
        # Follow-up attempts are judged against the accepted translation.
        assert "Improvement: these units already have an accepted translation" in _all_text(
            harness.fake.validator_calls()[1]
        )
        assert "Improvement:" not in _all_text(harness.fake.validator_calls()[0])
        assert status["parts"]["content"].endswith("results. (take 3)")
        finalized = harness.events_named("translation_part_finalized")[0]
        assert (finalized["reason"], finalized["final_attempt"], finalized["accepted_count"]) == (
            "three_accepted", 3, 3,
        )
    finally:
        await service.shutdown()


async def test_one_accepted_attempt_and_two_keeps_finalize_with_the_named_attempt(harness):
    service = await harness.start(_settings())
    try:
        status = await register_and_complete(service)
        assert [_task(call)["kind"] for call in harness.fake.translator_calls()] == ["first", "locked", "locked"]
        # Keeps are refusals: the validator judged only the first attempt.
        assert len(harness.fake.validator_calls()) == 1
        kept = harness.events_named("translation_attempt_kept")
        assert [(event["keep_attempts"], event["keep_streak"], event["dispute"]) for event in kept] == [
            ([1], 1, False),
            ([1], 2, False),
        ]
        finalized = harness.events_named("translation_part_finalized")[0]
        assert (finalized["reason"], finalized["final_attempt"]) == ("kept_twice", 1)
        assert status["parts"]["content"].endswith("results. (es)")
    finally:
        await service.shutdown()


async def test_a_translate_or_dispute_between_keeps_resets_the_streak(harness):
    service = await harness.start(_settings())
    script = ["translate", "translate", "keep1", "keep2", "keep1", "translate", "keep1", "keep1"]
    calls: List[int] = []

    def translator(record):
        step = script[len(calls)]
        calls.append(1)
        if step == "translate":
            return _translate(_translated_part(record, suffix=f"(take {len(calls)})"))
        return _keep([int(step[-1])], notes="Attempt 2 is correct." if step == "keep2" else "")

    def validator(record):
        number = _judged(record)[0]
        return _decisions(_judged(record), accept=number == 1, feedback=f"fix-{number}")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service)
        assert len(calls) == len(script)
        streaks = [event["keep_streak"] for event in harness.events_named("translation_attempt_kept")]
        # keep1 -> 1, dispute of 2 -> 0, keep1 -> 1, (translate resets), keep1 -> 1, keep1 -> 2.
        assert streaks == [1, 0, 1, 1, 2]
        assert [event["dispute"] for event in harness.events_named("translation_attempt_kept")] == [
            False, True, False, False, False,
        ]
        finalized = harness.events_named("translation_part_finalized")[0]
        assert (finalized["reason"], finalized["final_attempt"]) == ("kept_twice", 1)
    finally:
        await service.shutdown()


async def test_keeping_a_rejected_attempt_asks_the_validator_to_judge_it_again(harness):
    service = await harness.start(_settings())
    translator_calls: List[Dict[str, Any]] = []
    validator_calls: List[Dict[str, Any]] = []

    def translator(record):
        translator_calls.append(record)
        if len(translator_calls) == 1:
            return _translate(_translated_part(record))
        if len(translator_calls) == 2:
            return _keep([1], notes="The term is the standard rendering in Spanish.")
        return _keep(_accepted_numbers(record)[-1:])

    def validator(record):
        validator_calls.append(record)
        if len(validator_calls) == 1:
            return _decisions([1], accept=False, feedback="Use a different term for results.")
        return _accept_all(record)

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service)
        assert len(validator_calls) == 2
        rejudge = _all_text(validator_calls[1])
        assert _judged(validator_calls[1]) == [1]
        assert "Earlier feedback on this attempt: Use a different term for results." in rejudge
        assert "Translator notes on resubmission: The term is the standard rendering in Spanish." in rejudge
        # The re-judged attempt was accepted, locked the part, and was then kept twice.
        assert _task(translator_calls[2])["kind"] == "locked"
        accepted = harness.events_named("translation_attempt_accepted")
        assert [event["rejudged"] for event in accepted] == [True]
        assert accepted[0]["candidate_id"].endswith(":attempt1:rejudge2")
        finalized = harness.events_named("translation_part_finalized")[0]
        assert (finalized["reason"], finalized["final_attempt"]) == ("kept_twice", 1)
    finally:
        await service.shutdown()


async def test_a_dispute_naming_attempts_over_different_units_fails_the_automatic_checks(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return _translate(_translated_part(record))  # U1-U3
        if len(calls) == 2:
            return _translate(_translated_part(record, suffix="(b)", end=2))  # U1-U2
        if len(calls) == 3:
            return _keep([1, 2], notes="Both are correct.")
        if _task(record)["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        return _translate(_translated_part(record, suffix="(c)"))

    def validator(record):
        return _decisions(_judged(record), accept=_judged(record)[0] > 2, feedback="Too literal.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service, _doc(text=THREE_PARAGRAPHS))
        [check] = harness.events_named("translation_attempt_check_failed")
        assert check["round_index"] == 3
        assert any("cover different units" in problem for problem in check["problems"])
        assert [1, 2] not in [_judged(call) for call in harness.fake.validator_calls()]
        assert "[Attempt 3] Failed the automatic checks" in _all_text(calls[3])
    finally:
        await service.shutdown()


async def test_a_dispute_checks_each_named_attempt_after_the_edits_before_it(harness):
    service = await harness.start(_settings())
    script = ["a", "b", "keep12", "keep1"]
    edits = {"a": {"unit": 1, "find": "Alpha", "replace": "ALFA"}, "b": {"unit": 1, "find": "Alpha", "replace": "ALPHA"}}

    def translator(record):
        task = _task(record)
        units = _translated_part(record, end=task["start"])
        if task["start"] == 1 or task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:]) if task["kind"] == "locked" else _translate(units)
        step = script.pop(0)
        if step == "keep12":
            return _keep([1, 2], notes="Both are fine.")
        if step == "keep1":
            return _keep([1], notes="Attempt 1 is fine.")
        return _translate(_translated_part(record, suffix=f"({step})", end=2), edits=[edits[step]])

    def validator(record):
        # Both edited attempts are rejected at first; attempt 1 is accepted when disputed.
        judge = _judge_section(record)
        accept = '"Alpha" -> "AL' not in judge or "Translator notes on resubmission" in judge
        return _decisions(_judged(record), accept=accept, feedback="Too literal.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        status = await register_and_complete(service, _doc(text="Alpha paragraph text.\n\nBeta paragraph text."))
        # Attempt 2's different edit no longer matches once attempt 1's edit applies, so the pair
        # cannot be judged.
        [check] = harness.events_named("translation_attempt_check_failed")
        assert check["round_index"] == 3
        assert check["problems"][0].startswith(
            "Attempt 2 cannot be judged again as it is after the edits of attempt(s) 1: Edit 1 (U1)"
        )
        assert [1, 2] not in [_judged(call) for call in harness.fake.validator_calls()]
        accepted = [event for event in harness.events_named("translation_attempt_accepted") if event["part_start"] == 2]
        assert [(event["attempt_number"], event["edit_count"]) for event in accepted] == [(1, 1)]
        assert status["parts"]["content"] == "ALFA paragraph text. (es)\n\nBeta paragraph text. (a)"
    finally:
        await service.shutdown()


@pytest.mark.parametrize("replace", ["ALFA", "Alpha prime"])
async def test_disputing_rejected_attempts_that_share_an_edit_judges_them_and_applies_it_once(harness, replace):
    service = await harness.start(_settings())
    script = ["a", "b", "keep12"]
    edit = {"unit": 1, "find": "Alpha", "replace": replace}

    def translator(record):
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        if task["start"] == 1:
            return _translate(_translated_part(record, end=1))
        step = script.pop(0)
        if step == "keep12":
            return _keep([1, 2], notes="Both are correct.")
        return _translate(_translated_part(record, suffix=f"({step})", end=2), edits=[edit])

    def validator(record):
        # Both edited attempts are rejected at first and accepted when disputed together.
        judge = _judge_section(record)
        accept = '"Alpha" -> "' not in judge or "Translator notes on resubmission" in judge
        return _decisions(_judged(record), accept=accept, feedback="Too literal.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        status = await register_and_complete(service, _doc(text="Alpha paragraph text.\n\nBeta paragraph text."))
        assert harness.events_named("translation_attempt_check_failed") == []
        assert [1, 2] in [_judged(call) for call in harness.fake.validator_calls()]
        accepted = [event for event in harness.events_named("translation_attempt_accepted") if event["part_start"] == 2]
        assert [(event["attempt_number"], event["edit_count"]) for event in accepted] == [(1, 1), (2, 1)]
        assert status["parts"]["content"] == f"{replace} paragraph text. (es)\n\nBeta paragraph text. (b)"
    finally:
        await service.shutdown()


@pytest.mark.parametrize("replace", ["ALFA", "Alpha prime"])
async def test_a_follow_up_repeating_an_accepted_edit_applies_it_once(harness, replace):
    service = await harness.start(_settings())
    script = ["a", "b"]
    edit = {"unit": 1, "find": "Alpha", "replace": replace}

    def translator(record):
        task = _task(record)
        if task["start"] == 1:
            return _keep(_accepted_numbers(record)[-1:]) if task["kind"] == "locked" else _translate(
                _translated_part(record, end=1)
            )
        if script:
            step = script.pop(0)
            return _translate(_translated_part(record, suffix=f"({step})", end=2), edits=[edit])
        return _keep(_accepted_numbers(record)[-1:])

    harness.fake.validator = _accept_all
    harness.fake.translator = translator
    try:
        status = await register_and_complete(service, _doc(text="Alpha paragraph text.\n\nBeta paragraph text."))
        assert harness.events_named("translation_attempt_check_failed") == []
        accepted = [event for event in harness.events_named("translation_attempt_accepted") if event["part_start"] == 2]
        # The follow-up copied the accepted edit; it is already in the translation, so it is left out.
        assert [(event["attempt_number"], event["edit_count"]) for event in accepted] == [(1, 1), (2, 0)]
        assert status["parts"]["content"] == f"{replace} paragraph text. (es)\n\nBeta paragraph text. (b)"
    finally:
        await service.shutdown()


@pytest.mark.parametrize("replace", ["ALFA", "Alpha prime"])
async def test_disputing_an_attempt_that_shares_an_accepted_edit_judges_it_and_applies_the_edit_once(
    harness, replace
):
    service = await harness.start(_settings())
    script = ["a", "b", "keep1"]
    edit = {"unit": 1, "find": "Alpha", "replace": replace}

    def translator(record):
        task = _task(record)
        if task["start"] == 1:
            return _keep(_accepted_numbers(record)[-1:]) if task["kind"] == "locked" else _translate(
                _translated_part(record, end=1)
            )
        if script:
            step = script.pop(0)
            if step == "keep1":
                return _keep([1], notes="Attempt 1 is correct.")
            return _translate(_translated_part(record, suffix=f"({step})", end=2), edits=[edit])
        return _keep(_accepted_numbers(record)[-1:])

    def validator(record):
        # Attempt 1 is rejected at first and accepted when disputed; attempt 2 is accepted.
        judge = _judge_section(record)
        accept = "(a)" not in judge or "Translator notes on resubmission" in judge
        return _decisions(_judged(record), accept=accept, feedback="Too literal.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        status = await register_and_complete(service, _doc(text="Alpha paragraph text.\n\nBeta paragraph text."))
        assert harness.events_named("translation_attempt_check_failed") == []
        rejudged = [event for event in harness.events_named("translation_attempt_accepted") if event["rejudged"]]
        assert [(event["attempt_number"], event["edit_count"]) for event in rejudged] == [(1, 1)]
        assert status["parts"]["content"].split("\n\n")[0] == f"{replace} paragraph text. (es)"
    finally:
        await service.shutdown()


async def test_a_dispute_is_judged_when_its_edit_is_outside_the_translator_window(harness):
    _seed_final_translations(WINDOW_TEXT, 54)
    service = await harness.start(_settings())
    narrowed = asyncio.Event()
    calls: List[Dict[str, Any]] = []
    disputes: List[Dict[str, Any]] = []

    async def translator(record):
        calls.append(record)
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        if task["start"] != 55:
            return _translate(_translated_part(record, end=task["start"]))
        if not _attempts_section(record):
            edits = [{"unit": 1, "find": "number 1 discusses", "replace": "número 1 discusses"}]
            return _translate(_translated_part(record, end=55), edits=edits)
        await narrowed.wait()
        disputes.append(record)
        return _keep([1], notes="The edit to U1 is correct.")

    def validator(record):
        judge = _judge_section(record)
        accept = "número" not in judge or "Translator notes on resubmission" in judge
        return _decisions(_judged(record), accept=accept, feedback="Keep U1 as it is.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        row = (await service.register_documents("es", True, [_doc(text=WINDOW_TEXT)]))[0]
        await wait_until(lambda: bool(harness.events_named("translation_attempt_rejected")))
        # The translator's context shrinks, so its window no longer shows U1.
        await service.update_settings(_settings(translator=_role("small", context_window=3200, max_output_tokens=1000)))
        narrowed.set()
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        assert 1 not in _translation_units(disputes[-1])
        assert harness.events_named("translation_attempt_check_failed") == []
        rejudged = [event for event in harness.events_named("translation_attempt_accepted") if event["rejudged"]]
        assert [event["attempt_number"] for event in rejudged] == [1]
        status = await lookup_status(service, row["doc_hash"])
        assert status["parts"]["content"].startswith("Paragraph número 1 discusses")
    finally:
        await service.shutdown()


TWO_PARAGRAPHS = "Alpha paragraph text.\n\nBeta paragraph text."


def _part_two_translator(steps: List[Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """U1 translates and is kept; part U2 follows ``steps``, then keeps its newest accepted attempt.

    A step is ``(suffix, edits)`` for a translate response, or a list of attempt numbers to keep.
    """
    script = list(steps)

    def translator(record):
        task = _task(record)
        if task["start"] == 1:
            return _keep(_accepted_numbers(record)[-1:]) if task["kind"] == "locked" else _translate(
                _translated_part(record, end=1)
            )
        if not script:
            return _keep(_accepted_numbers(record)[-1:])
        step = script.pop(0)
        if isinstance(step, list):
            return _keep(step, notes="Attempt 1 is correct.")
        suffix, edits = step
        return _translate(_translated_part(record, suffix=f"({suffix})", end=2), edits=edits)

    return translator


async def test_resubmitting_an_accepted_attempt_whose_edits_were_partly_superseded_keeps_it(harness):
    service = await harness.start(_settings())
    first = [
        {"unit": 1, "find": "Alpha", "replace": "ALFA"},
        {"unit": 1, "find": "text.", "replace": "texto."},
    ]
    harness.fake.translator = _part_two_translator(
        [("a", first), ("b", [{"unit": 1, "find": "ALFA", "replace": "ALEPH"}]), ("a", first), [1]]
    )
    harness.fake.validator = _accept_all
    try:
        status = await register_and_complete(service, _doc(text=TWO_PARAGRAPHS))
        assert harness.events_named("translation_attempt_check_failed") == []
        kept = [event for event in harness.events_named("translation_attempt_kept") if event["part_start"] == 2]
        # The verbatim resubmission of attempt 1 is a keep of it, not a new attempt that fails its checks.
        assert [event["keep_attempts"] for event in kept] == [[1], [1]]
        finalized = harness.events_named("translation_part_finalized")[-1]
        assert (finalized["reason"], finalized["final_attempt"]) == ("kept_twice", 1)
        assert status["parts"]["content"] == "ALEPH paragraph texto. (es)\n\nBeta paragraph text. (a)"
    finally:
        await service.shutdown()


async def test_resubmitting_a_rejected_attempt_whose_edit_was_made_disputes_it_without_that_edit(harness):
    service = await harness.start(_settings())
    edit = {"unit": 1, "find": "Alpha", "replace": "ALFA"}
    harness.fake.translator = _part_two_translator([("a", [edit]), ("b", [edit]), ("a", [edit])])

    def validator(record):
        judge = _judge_section(record)
        accept = "(a)" not in judge or "Translator notes on resubmission" in judge
        return _decisions(_judged(record), accept=accept, feedback="Too literal.")

    harness.fake.validator = validator
    try:
        status = await register_and_complete(service, _doc(text=TWO_PARAGRAPHS))
        assert harness.events_named("translation_attempt_check_failed") == []
        kept = [event for event in harness.events_named("translation_attempt_kept") if event["part_start"] == 2]
        # The verbatim resubmission of rejected attempt 1 disputes it.
        assert (kept[0]["keep_attempts"], kept[0]["dispute"]) == ([1], True)
        [dispute] = [
            call for call in harness.fake.validator_calls()
            if "Translator notes on resubmission" in _judge_section(call)
        ]
        # Attempt 2 already made the edit, so the validator judges attempt 1 without it.
        assert _judged(dispute) == [1]
        assert '"Alpha" -> "ALFA"' not in _judge_section(dispute)
        assert status["parts"]["content"].split("\n\n")[0] == "ALFA paragraph text. (es)"
    finally:
        await service.shutdown()


@pytest.mark.parametrize(
    ("edits", "expected"),
    [
        # A later edit rewrote the replacement, which still occurs inside "Alpha".
        (
            [
                {"unit": 1, "find": "Alpha", "replace": "Al"},
                {"unit": 1, "find": "Al paragraph", "replace": "Alpha paragraph"},
            ],
            "Al paragraph text. (es)",
        ),
        # A later edit put deleted text back.
        (
            [
                {"unit": 1, "find": " (es)", "replace": ""},
                {"unit": 1, "find": "text.", "replace": "text. (es)"},
            ],
            "Alpha paragraph text.",
        ),
    ],
)
async def test_an_edit_made_again_after_a_later_edit_undid_it_is_applied(harness, edits, expected):
    service = await harness.start(_settings())
    harness.fake.translator = _part_two_translator([("a", [edits[0]]), ("b", [edits[1]]), ("c", [edits[0]])])
    harness.fake.validator = _accept_all
    try:
        status = await register_and_complete(service, _doc(text=TWO_PARAGRAPHS))
        accepted = [event for event in harness.events_named("translation_attempt_accepted") if event["part_start"] == 2]
        assert [(event["attempt_number"], event["edit_count"]) for event in accepted] == [(1, 1), (2, 1), (3, 1)]
        assert status["parts"]["content"].split("\n\n")[0] == expected
    finally:
        await service.shutdown()


async def test_an_edit_listed_twice_in_one_attempt_applies_once(harness):
    service = await harness.start(_settings())
    edit = {"unit": 1, "find": "Alpha", "replace": "Alpha prime"}
    harness.fake.translator = _part_two_translator([("a", [edit, edit])])
    harness.fake.validator = _accept_all
    try:
        status = await register_and_complete(service, _doc(text=TWO_PARAGRAPHS))
        assert status["parts"]["content"] == "Alpha prime paragraph text. (es)\n\nBeta paragraph text. (a)"
    finally:
        await service.shutdown()


async def test_identical_text_counts_as_keeping_the_earlier_attempt(harness):
    service = await harness.start(_settings())
    calls: List[int] = []

    def translator(record):
        calls.append(1)
        units = _translated_part(record)
        if len(calls) > 1:
            units = {number: f"  {text.replace(' ', '  ')}  " for number, text in units.items()}
        return _translate(units)

    harness.fake.translator = translator
    try:
        await register_and_complete(service)
        assert len(calls) == 3
        assert len(harness.fake.validator_calls()) == 1
        assert [event["keep_attempts"] for event in harness.events_named("translation_attempt_kept")] == [[1], [1]]
        finalized = harness.events_named("translation_part_finalized")[0]
        assert finalized["reason"] == "kept_twice"
    finally:
        await service.shutdown()


def _judge_section(record: Dict[str, Any]) -> str:
    return _all_text(record).rsplit("ATTEMPTS TO JUDGE\n", 1)[1]


async def test_dropping_a_rejected_edit_is_judged_as_a_new_attempt(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        units = _translated_part(record, end=task["start"])
        if task == {"kind": "first", "start": 2, "end": 3}:
            return _translate(units, edits=[{"unit": 1, "find": "Alpha", "replace": "BAD"}])
        # The retry fixes the rejection by dropping the edit, with the same unit text.
        return _translate(units)

    def validator(record):
        bad = "BAD" in _judge_section(record)
        return _decisions(_judged(record), accept=not bad, feedback="The edit to U1 is wrong; drop it.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        status = await register_and_complete(service, _doc(text=THREE_PARAGRAPHS))
        part_two = [call for call in harness.fake.validator_calls() if "<<U2>>" in _judge_section(call)]
        # The edit-free retry was judged as attempt 2 on its own, not as a dispute of attempt 1.
        assert [(_judged(call), "BAD" in _judge_section(call)) for call in part_two] == [([1], True), ([2], False)]
        assert not any(event["dispute"] for event in harness.events_named("translation_attempt_kept"))
        assert status["parts"]["content"].startswith("Alpha paragraph text. (es)\n\nBeta")
    finally:
        await service.shutdown()


async def test_repeating_an_accepted_attempt_without_a_rejected_edit_is_a_refusal(harness):
    service = await harness.start(_settings())
    script = ["edit", "plain", "plain"]

    def translator(record):
        task = _task(record)
        units = _translated_part(record, end=task["start"])
        if task["start"] == 1:
            return _keep(_accepted_numbers(record)[-1:]) if task["kind"] == "locked" else _translate(units)
        if task["kind"] == "first":
            return _translate(units)
        step = script.pop(0)
        edits = [{"unit": 1, "find": "Alpha", "replace": "BAD"}] if step == "edit" else None
        return _translate(units, edits=edits)

    def validator(record):
        bad = "BAD" in _judge_section(record)
        return _decisions(_judged(record), accept=not bad, feedback="The edit to U1 is wrong.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service, _doc(text="Alpha paragraph text.\n\nBeta paragraph text."))
        # 1 = X accepted, 2 = X with an edit rejected, 3 and 4 = X again: refusals keeping 1.
        kept = [event for event in harness.events_named("translation_attempt_kept") if event["part_start"] == 2]
        assert [(event["keep_attempts"], event["keep_streak"], event["dispute"]) for event in kept] == [
            ([1], 1, False),
            ([1], 2, False),
        ]
        finalized = [event for event in harness.events_named("translation_part_finalized") if event["part_start"] == 2]
        assert (finalized[0]["reason"], finalized[0]["final_attempt"]) == ("kept_twice", 1)
        assert len([call for call in harness.fake.validator_calls() if "<<U2>>" in _judge_section(call)]) == 2
    finally:
        await service.shutdown()


async def test_the_first_accepted_attempt_locks_the_range(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return _translate(_translated_part(record))  # U1-U3, rejected
        if len(calls) == 2:
            return _translate(_translated_part(record, suffix="(b)", end=2))  # U1-U2, accepted
        if len(calls) == 3:
            # Tries to grow the locked part again.
            source = _source_units(record)
            return _translate({number: f"{source[number]} (c)" for number in (1, 2, 3)})
        if _task(record)["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        return _translate(_translated_part(record, suffix=f"(take {len(calls)})"))

    def validator(record):
        return _decisions(_judged(record), accept=_judged(record) != [1], feedback="Too literal.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service, _doc(text=THREE_PARAGRAPHS))
        assert _task(calls[1]) == {"kind": "retry", "start": 1, "end": 3}
        assert _task(calls[2]) == {"kind": "locked", "start": 1, "end": 2}
        check = harness.events_named("translation_attempt_check_failed")
        assert len(check) == 1 and check[0]["round_index"] == 3
        assert any("U1 to U2" in problem for problem in check[0]["problems"])
        finalized = harness.events_named("translation_part_finalized")
        assert [(event["part_start"], event["part_end"]) for event in finalized] == [(1, 2), (3, 3)]
        # The check failure was never sent to the validator.
        assert [3] not in [_judged(call) for call in harness.fake.validator_calls()[:2]]
    finally:
        await service.shutdown()


async def test_accepted_edits_apply_at_once_and_rejected_ones_change_nothing(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        units = _translated_part(record, end=task["start"])
        if task["start"] == 2:
            edit = {"unit": 1, "find": "Alpha", "replace": "ALFA" if task["kind"] == "first" else "ALPHA"}
            return _translate(units, edits=[edit])
        return _translate(units)

    def validator(record):
        # The first attempt at U2 (with its edit) is rejected, the second accepted.
        rejected = "ALFA" in _all_text(record).split("ATTEMPTS TO JUDGE", 1)[1]
        return _decisions(_judged(record), accept=not rejected, feedback="The edit is wrong.")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        status = await register_and_complete(service, _doc(text=THREE_PARAGRAPHS))
        part_two = [call for call in calls if _task(call)["start"] == 2]
        # After the rejected edit, U1 is unchanged.
        assert _translation_units(part_two[1])[1] == "Alpha paragraph text. (es)"
        # After the accepted edit, the next prompt already shows the corrected U1.
        assert _translation_units(part_two[2])[1] == "ALPHA paragraph text. (es)"
        assert status["parts"]["content"].startswith("ALPHA paragraph text. (es)\n\nBeta")
        accepted = [event for event in harness.events_named("translation_attempt_accepted") if event["edit_count"]]
        assert len(accepted) == 1
        # The cross-document cache follows the corrected final.
        doc = service._docs[status["doc_hash"]]
        cached = translation_store.get_translations([doc.views[0].source_hash], "es")
        assert cached[doc.views[0].source_hash] == "ALPHA paragraph text. (es)"
    finally:
        await service.shutdown()


async def test_precheck_failure_is_automatic_feedback_not_a_validator_decision(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            # Drops the [[M1]] placeholder.
            return _translate({1: "Primer párrafo.", 2: "Segundo párrafo."})
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: len(calls) >= 2)
        second = _all_text(calls[1])
        assert "[Attempt 1] Failed the automatic checks" in second
        assert "Feedback: U1: Placeholder(s) [[M1]]" in second
        assert _task(calls[1])["kind"] == "retry"
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        assert [_judged(call) for call in harness.fake.validator_calls()] == [[2]]
        check = harness.events_named("translation_attempt_check_failed")
        assert len(check) == 1 and check[0]["problems"]
        assert check[0]["activity_classification"] == "operational"
        usage = service.status_snapshot()["usage"]
        assert (usage["check_failures"], usage["rejected_attempts"]) == (1, 0)
    finally:
        await service.shutdown()


async def test_state_is_persisted_after_every_attempt(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()
    calls: List[int] = []

    async def translator(record):
        calls.append(1)
        if len(calls) == 2:
            await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: len(calls) == 2)
        state = _job_state(service, row["doc_hash"])
        part = state["part"]
        assert (part["start"], part["end"], part["locked"]) == (0, 1, True)
        [record] = part["attempts"]
        assert (record["number"], record["status"], record["accepted_order"]) == (1, STATUS_ACCEPTED, 1)
        assert record["units"] == {"0": f"{MASKED_FIRST} (es)", "1": "Second paragraph about the results. (es)"}
        assert state["finals"] == {}
    finally:
        gate.set()
        await service.shutdown()


async def test_a_saved_part_that_meets_the_stop_rule_finalizes_without_calls(harness):
    plan = build_document_plan(PROFILE_DOCUMENT, [("content", SOURCE_TEXT)])
    record = {
        "number": 1, "action": "translate", "units": {"0": f"{MASKED_FIRST} (es)", "1": "Second. (es)"},
        "edits": [], "keep_attempts": [], "notes": "", "status": STATUS_ACCEPTED, "feedback": "",
        "reasoning": "ok", "accepted_order": 1,
    }
    keep = {**record, "number": 2, "action": "keep", "units": {}, "status": STATUS_KEPT,
            "keep_attempts": [1], "accepted_order": None, "reasoning": ""}
    state = {
        "version": 1, "doc_hash": plan.doc_hash, "language_key": "es", "finals": {},
        "part": {"start": 0, "end": 1, "locked": True, "attempts": [record, keep, {**keep, "number": 3}],
                 "keep_streak": 2, "truncations": {"translation_submitter": 0, "translation_validator": 0}},
    }
    service = await harness.start(_settings(paused=True))
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        translation_store.save_job_state(row["doc_hash"], "es", state)
        await service.control("resume")
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        assert harness.fake.calls == []
        status = await lookup_status(service, row["doc_hash"])
        assert status["parts"]["content"] == "First paragraph with math $x^2$ here. (es)\n\nSecond. (es)"
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# JSON contracts, truncation, context, and provider recovery
# ----------------------------------------------------------------------


async def test_validator_malformed_json_repairs_once_then_retries_the_same_attempt(harness):
    service = await harness.start(_settings())
    validator_calls: List[Dict[str, Any]] = []

    def validator(record):
        validator_calls.append(record)
        if len(validator_calls) <= 2:
            return _response("I think attempt one is good <|channel|> but no JSON here")
        return _accept_all(record)

    harness.fake.validator = validator
    try:
        await register_and_complete(service)
        assert len(validator_calls) == 3
        repair = validator_calls[1]
        assert repair["task_id"].endswith("_repair")
        assert repair["temperature"] == 0.0
        assert [message["role"] for message in repair["messages"]] == ["user", "assistant", "user"]
        assert "<|channel|>" not in repair["messages"][1]["content"]
        assert repair["messages"][2]["content"].startswith(
            "Your previous response did not match the required JSON format."
        )
        # The same attempt is judged again after the operational retry.
        judged = [_all_text(call).split("ATTEMPTS TO JUDGE", 1)[1] for call in (validator_calls[0], validator_calls[2])]
        assert judged[0] == judged[1]
        contract_waits = [
            wait for wait in harness.retry_waits if wait["payload"]["reason"] == "translation_validator_contract"
        ]
        assert len(contract_waits) == 1
        assert contract_waits[0]["payload"]["retry_after_seconds"] == 5
        assert contract_waits[0]["payload"]["workflow_mode"] == "translation"
        retries = harness.events_named("translation_validator_contract_retry")
        assert len(retries) == 1 and retries[0]["round_index"] == 1 and retries[0]["attempt"] == 1
        # The retry names the units of the attempt being judged, as its decision does.
        (accepted,) = harness.events_named("translation_attempt_accepted")
        assert (retries[0]["part_start"], retries[0]["part_end"]) == (1, 2)
        assert (accepted["part_start"], accepted["part_end"]) == (1, 2)
        # One translate attempt, then two keeps; nothing was rejected.
        assert [_task(call)["kind"] for call in harness.fake.translator_calls()] == ["first", "locked", "locked"]
        assert service.status_snapshot()["usage"]["rejected_attempts"] == 0
    finally:
        await service.shutdown()


async def test_validator_contract_retry_in_a_dispute_names_the_disputed_attempt_units(harness):
    """The disputed attempt covers more units than the part's latest attempt."""
    service = await harness.start(_settings())
    translator_calls: List[Dict[str, Any]] = []
    validator_calls: List[Dict[str, Any]] = []

    def translator(record):
        translator_calls.append(record)
        task = _task(record)
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        if len(translator_calls) == 1:
            return _translate(_translated_part(record))
        if len(translator_calls) == 2:
            return _translate(_translated_part(record, suffix="(b)", end=task["start"]))
        return _keep([1], notes="Attempt 1 already uses the standard terms.")

    def validator(record):
        validator_calls.append(record)
        if len(validator_calls) in (3, 4):
            return _response("no JSON from the validator at all, just words")
        return _decisions(_judged(record), accept=len(validator_calls) > 4)

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service, _doc(text=THREE_PARAGRAPHS))
        rejected = harness.events_named("translation_attempt_rejected")
        assert [(event["attempt_number"], event["part_end"]) for event in rejected] == [(1, 3), (2, 1)]
        assert "Translator notes on resubmission" in _all_text(validator_calls[4])
        (retry,) = harness.events_named("translation_validator_contract_retry")
        assert retry["round_index"] == 3
        assert (retry["part_start"], retry["part_end"]) == (1, 3)
        (accepted,) = harness.events_named("translation_attempt_accepted")
        assert accepted["rejudged"] is True and accepted["attempt_number"] == 1
        assert (accepted["part_start"], accepted["part_end"]) == (1, 3)
    finally:
        await service.shutdown()


async def test_translator_invalid_json_after_repair_becomes_automatic_feedback(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) <= 2:
            return _response("no json at all")
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await register_and_complete(service)
        assert calls[1]["task_id"].endswith("_repair")
        third = _all_text(calls[2])
        assert "Your response was not valid JSON in the required format" in third
        assert "[Attempt 1] Failed the automatic checks" in third
        assert service.status_snapshot()["usage"]["check_failures"] == 1
    finally:
        await service.shutdown()


async def test_two_consecutive_translator_truncations_enter_repair(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        return _response('{"action": "translate", "units": [{"unit": 1, "translation": "Primer', finish="length")

    harness.fake.translator = translator
    try:
        await service.register_documents("es", True, [_doc()])
        await wait_until(lambda: service.status_snapshot()["state"] == "repair_required")
        assert len(calls) == 2
        assert "Your response was cut off at the max output token limit" in _all_text(calls[1])
        repair = service.status_snapshot()["repair"]
        assert repair["reason"] == "output_truncated"
        assert "Raise Max Output Tokens for Translator Submitter" in repair["guidance"]
        assert harness.fake.validator_calls() == []
    finally:
        await service.shutdown()


async def test_two_consecutive_validator_truncations_enter_repair(harness):
    service = await harness.start(_settings())
    harness.fake.validator = lambda record: _response('{"decisions": [{"attempt": 1', finish="length")
    try:
        await service.register_documents("es", True, [_doc()])
        await wait_until(lambda: service.status_snapshot()["state"] == "repair_required")
        assert len(harness.fake.validator_calls()) == 2
        repair = service.status_snapshot()["repair"]
        assert (repair["reason"], repair["role_id"]) == ("output_truncated", "translation_validator")
        assert len(harness.events_named("translation_validator_contract_retry")) == 1
        assert harness.events_named("translation_attempt_rejected") == []
    finally:
        await service.shutdown()


@pytest.mark.parametrize("role_id", ["translation_submitter", "translation_validator"])
async def test_truncation_repair_starts_a_fresh_count_after_retry(harness, role_id):
    service = await harness.start(_settings())
    truncated = {"left": 3}
    second = asyncio.Event()
    calls: List[int] = []

    async def cut_off(record, answer):
        calls.append(1)
        if truncated["left"] == 0:
            return answer(record)
        truncated["left"] -= 1
        if len(calls) == 2:
            await second.wait()
        return _response('{"action": "translate", "units": [{"unit": 1, "translation": "Primer', finish="length")

    if role_id == "translation_submitter":
        harness.fake.translator = lambda record: cut_off(record, _translate_ok)
    else:
        harness.fake.validator = lambda record: cut_off(record, _accept_all)
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: len(calls) == 2)
        # The first truncation is saved for both roles.
        assert _job_state(service, row["doc_hash"])["part"]["truncations"][role_id] == 1
        second.set()
        await wait_until(lambda: service.status_snapshot()["state"] == "repair_required")
        # The second enters repair with the count reset and saved.
        assert _job_state(service, row["doc_hash"])["part"]["truncations"][role_id] == 0
        await service.control("retry")
        # One more truncation after Retry is a fresh first truncation, not a second repair.
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        assert truncated["left"] == 0
        assert len(harness.events_named("translation_repair_required")) == 1
    finally:
        await service.shutdown()


async def test_context_window_too_small_enters_repair_without_calls(harness):
    service = await harness.start(
        _settings(translator=_role("tiny", context_window=1700, max_output_tokens=1000))
    )
    try:
        await service.register_documents("es", True, [_doc()])
        await wait_until(lambda: service.status_snapshot()["state"] == "repair_required")
        repair = service.status_snapshot()["repair"]
        assert (repair["reason"], repair["role_id"]) == ("context_window_too_small", "translation_submitter")
        assert harness.fake.calls == []
    finally:
        await service.shutdown()


async def test_validator_context_too_small_enters_repair_before_judging(harness):
    service = await harness.start(
        _settings(validator=_role("tiny-judge", context_window=1700, max_output_tokens=1000))
    )
    try:
        await service.register_documents("es", True, [_doc()])
        await wait_until(lambda: service.status_snapshot()["state"] == "repair_required")
        assert service.status_snapshot()["repair"]["role_id"] == "translation_validator"
        assert len(harness.fake.translator_calls()) == 1
        assert harness.fake.validator_calls() == []
    finally:
        await service.shutdown()


async def test_provider_context_rejection_refits_the_prompt_with_a_window(harness):
    text = "\n\n".join(
        f"Paragraph number {index} discusses the lattice structure and its symmetry group in detail."
        for index in range(1, 61)
    )
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return RuntimeError("context_length_exceeded: prompt is too long")
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await service.register_documents("es", True, [_doc(text=text)])
        await wait_until(lambda: len(calls) >= 2)
        assert "units omitted" not in _all_text(calls[0])
        assert "units omitted" in _all_text(calls[1])
        assert service_module.count_tokens(_all_text(calls[1])) < service_module.count_tokens(_all_text(calls[0]))
        assert service.status_snapshot()["repair"] is None
    finally:
        await service.shutdown()


async def test_retryable_provider_error_waits_through_shared_seam_then_retries(harness, monkeypatch):
    # The real workflow backoff seam runs; only its emit-and-wait step is faked.
    monkeypatch.setattr(api_client_manager, "_retryable_provider_backoff_state", {})
    monkeypatch.setattr(
        api_client_manager,
        "wait_for_retryable_provider_error",
        type(api_client_manager).wait_for_retryable_provider_error.__get__(api_client_manager),
    )
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="translation_submitter",
                model="translator-a",
                reason="transient",
                message="upstream timeout",
            )
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await register_and_complete(service)
        [seam] = harness.retry_waits
        assert (seam["payload"]["workflow_mode"], seam["payload"]["retry_layer"]) == ("translation", "workflow")
        assert seam["role_id"] == "translation_submitter"
        assert len(calls) == 4
        [wait] = [
            item for item in service.status_snapshot()["recent_activity"]
            if item["event_type"] == "translation_provider_retry"
        ]
        # The recorded wait is the seam's wait: its route in fields and once in the line,
        # its duration, retry count, and the provider's error detail.
        assert (wait["workflow_mode"], wait["model"], wait["provider"]) == (
            "translation", "translator-a", "openrouter"
        )
        assert wait["route_identity"] == seam["payload"]["route_identity"]
        assert wait["retry_after_seconds"] == seam["payload"]["retry_after_seconds"] == 60
        assert wait["message"] == f"{wait['route_identity']} Transient; retrying in 60s (retry 1)."
        assert wait["error_summary"] == seam["payload"]["error_summary"]
        assert "upstream timeout" in wait["error_summary"]
    finally:
        await service.shutdown()


async def test_free_model_exhaustion_and_unexpected_errors_wait_through_shared_seam(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return FreeModelExhaustedError("All free model options exhausted")
        if len(calls) == 2:
            return RuntimeError("socket closed unexpectedly")
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await register_and_complete(service)
        reasons = [wait["payload"]["reason"] for wait in harness.retry_waits]
        assert reasons == ["free_models_exhausted", "translation_loop_retry"]
        free, loop = harness.retry_waits
        assert free["payload"]["retry_after_seconds"] == 120
        assert free["payload"]["workflow_mode"] == "translation"
        assert free["role_id"] == "translation_submitter"
        assert isinstance(free["error"], FreeModelExhaustedError)
        assert loop["payload"]["provider"] == "unknown"
        assert loop["payload"]["retry_after_seconds"] == 10
        assert loop["payload"]["workflow_mode"] == "translation"
        assert harness.events_named("translation_repair_required") == []
        assert len(calls) == 5
    finally:
        await service.shutdown()


async def test_an_unclassified_error_names_the_route_it_carries(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            route = ProviderRouteIdentity(provider="openrouter", model="translator-a", role_id="translation_submitter")
            return ProviderRouteError("unexpected upstream reply", route=route)
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await register_and_complete(service)
        [wait] = [
            item for item in service.status_snapshot()["recent_activity"]
            if item["event_type"] == "translation_provider_retry"
        ]
        assert (wait["reason"], wait["model"], wait["provider"]) == (
            "translation_loop_retry", "translator-a", "openrouter"
        )
        assert wait["message"] == (
            f"{wait['route_identity']} An unexpected error interrupted the attempt; retrying in 10s."
        )
        assert wait["error_summary"].startswith("ProviderRouteError: unexpected upstream reply")
    finally:
        await service.shutdown()


async def test_provider_repair_required_stops_dispatch_until_retry(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return ProviderRepairRequiredError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="translation_submitter",
                model="translator-a",
                reason="invalid_api_key",
                message="OpenRouter rejected the API key.",
                terminal_guidance="Add a valid OpenRouter key.",
            )
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: service.status_snapshot()["state"] == "repair_required")
        snapshot = service.status_snapshot()
        assert snapshot["status_label"] == "repair required"
        assert snapshot["repair"]["reason"] == "invalid_api_key"
        assert snapshot["repair"]["guidance"] == "Add a valid OpenRouter key."
        assert snapshot["repair"]["role_id"] == "translation_submitter"
        assert snapshot["repair"]["route_identity"]
        repair_events = harness.events_named("translation_repair_required")
        assert len(repair_events) == 1
        assert repair_events[0]["workflow_mode"] == "translation"
        assert repair_events[0]["role_id"] == "translation_submitter"
        assert repair_events[0]["model"] == "translator-a"
        assert repair_events[0]["provider"] == "openrouter"
        assert repair_events[0]["route_identity"]
        # The repair line names its route once, live and in the status copy.
        route = repair_events[0]["route_identity"]
        assert repair_events[0]["message"] == f"{route} OpenRouter rejected the API key."
        [recorded] = [item for item in snapshot["recent_activity"] if item["event_type"] == "translation_repair_required"]
        assert (recorded["message"], recorded["model"], recorded["provider"]) == (
            repair_events[0]["message"], "translator-a", "openrouter"
        )
        assert snapshot["repair"]["message"] == "OpenRouter rejected the API key."

        await settle()
        assert len(calls) == 1
        assert (await lookup_status(service, row["doc_hash"]))["status"] == "pending"

        await service.control("retry")
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        assert service.status_snapshot()["repair"] is None
    finally:
        await service.shutdown()


async def test_credit_exhaustion_waits_for_provider_reset(harness, monkeypatch):
    service = await harness.start(_settings())
    released = asyncio.Event()
    observed: List[int] = []

    async def fake_reset(generation, should_stop=None):
        observed.append(generation)
        await released.wait()

    monkeypatch.setattr(service_module, "wait_for_provider_reset", fake_reset)
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return CreditExhaustionError("OpenRouter credits exhausted")
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: service.status_snapshot()["state"] == "waiting_for_credits")
        assert service.status_snapshot()["status_label"] == "waiting"
        assert len(observed) == 1
        [wait] = [
            item for item in service.status_snapshot()["recent_activity"]
            if item["event_type"] == "translation_waiting_for_credits"
        ]
        assert wait["model"] == "translator-a"
        assert wait["message"] == f"{wait['route_identity']} Waiting for provider credits to be reset."
        released.set()
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        assert service.status_snapshot()["state"] == "idle"
        assert len(calls) == 4
        assert harness.events_named("translation_repair_required") == []
    finally:
        await service.shutdown()


async def test_a_usage_limit_cooldown_is_recorded_once_with_its_route_and_detail(harness):
    service = await harness.start(_settings())
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return ProviderCooldownError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="translation_submitter",
                model="translator-a",
                resets_at=None,
                resets_in_seconds=300,
                message="OpenRouter usage limit reached",
            )
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await register_and_complete(service)
        assert [type(wait["error"]) for wait in harness.provider_waits] == [ProviderCooldownError]
        [wait] = [
            item for item in service.status_snapshot()["recent_activity"]
            if item["event_type"] == "translation_provider_retry"
        ]
        assert (wait["reason"], wait["retry_after_seconds"], wait["model"], wait["provider"]) == (
            "usage_limit_reached", 300, "translator-a", "openrouter"
        )
        assert wait["message"] == f"{wait['route_identity']} Usage limit reached; retrying in 300s."
        assert "OpenRouter usage limit reached" in wait["error_summary"]
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# Queue, yielding, positions, and batch translation
# ----------------------------------------------------------------------


def _doc_of(record: Dict[str, Any]) -> str:
    return "A" if "Alpha" in _block(record, "SOURCE") else "B"


async def test_jobs_yield_after_each_part_so_a_higher_priority_job_runs_next(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()
    calls: List[Dict[str, Any]] = []

    async def translator(record):
        calls.append(record)
        if len(calls) == 1:
            await gate.wait()
        return _one_unit_parts(record)

    harness.fake.translator = translator
    try:
        [paper_hash] = await discover(service, _doc("paper:a", THREE_PARAGRAPHS))
        await wait_until(lambda: len(calls) == 1)
        viewed = (await service.register_documents(
            "es", True, [_doc("brainstorm:b", "A viewed brainstorm about groups.", kind="brainstorm")]
        ))[0]
        assert viewed["queue_position"] == 1
        # A viewed document never preempts current-run work; it runs at the next part boundary.
        assert not any(job.yield_requested for job in service._jobs.values())
        gate.set()
        await wait_until(lambda: _complete(service, paper_hash) and _complete(service, viewed["doc_hash"]))
        order = [_doc_of(call) for call in calls]
        # Part U1 of the paper finishes, then the viewed brainstorm runs before the paper's next part.
        assert order[:6] == ["A", "A", "A", "B", "B", "B"]
        assert order[6:] == ["A"] * (len(order) - 6)
    finally:
        gate.set()
        await service.shutdown()


async def test_a_job_yields_after_a_retry_wait(harness, monkeypatch):
    service = await harness.start(_settings())
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def gated_wait(error, **_kwargs):
        waiting.set()
        await release.wait()

    monkeypatch.setattr(api_client_manager, "wait_for_retryable_provider_error", gated_wait)
    calls: List[Dict[str, Any]] = []

    def translator(record):
        calls.append(record)
        if len(calls) == 1:
            return RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="translation_submitter",
                model="translator-a",
                reason="transient",
                message="upstream timeout",
            )
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        [paper_hash] = await discover(service, _doc("paper:a", THREE_PARAGRAPHS))
        await asyncio.wait_for(waiting.wait(), 5)
        viewed = (await service.register_documents(
            "es", True, [_doc("brainstorm:b", "A viewed brainstorm about groups.", kind="brainstorm")]
        ))[0]
        release.set()
        await wait_until(lambda: _complete(service, paper_hash) and _complete(service, viewed["doc_hash"]))
        # The interrupted call is retried in place; after that attempt the job yields to the viewed document.
        assert [_doc_of(call) for call in calls] == ["A", "A", "B", "B", "B", "A", "A"]
    finally:
        release.set()
        await service.shutdown()


async def test_priority_orders_viewing_then_proofs_papers_brainstorms_and_language_order(harness):
    service = await harness.start(_settings(paused=True, languages=TWO_LANGUAGES))
    try:
        brainstorm = _doc("brainstorm:current:b1", "Brainstorm idea about lattices.", kind="brainstorm")
        paper = _doc("paper:s:p1", "Paper body about lattices.", kind="paper")
        proof = {
            "doc_key": "proof:autonomous:r:p1",
            "kind": "proof",
            "title": "Lemma",
            "profile": "fields",
            "parts": [{"name": "theorem_statement", "text": "Every lattice is discrete."}],
        }
        viewed = _doc("brainstorm:current:b2", "A viewed brainstorm about groups.", kind="brainstorm")
        await discover(service, brainstorm, paper, proof)
        viewed_row = (await service.register_documents(None, True, [viewed]))[0]
        doc_hash = {key: service._doc_by_key[key] for key in service._doc_by_key}
        assert service._waiting_jobs() == [
            (doc_hash["brainstorm:current:b2"], "es"),
            (doc_hash["brainstorm:current:b2"], "fr"),
            (doc_hash["proof:autonomous:r:p1"], "es"),
            (doc_hash["proof:autonomous:r:p1"], "fr"),
            (doc_hash["paper:s:p1"], "es"),
            (doc_hash["paper:s:p1"], "fr"),
            (doc_hash["brainstorm:current:b1"], "es"),
            (doc_hash["brainstorm:current:b1"], "fr"),
        ]
        queue = service.status_snapshot()["queue"]
        assert [(item["doc_key"], item["language_key"], item["position"]) for item in queue[:3]] == [
            ("brainstorm:current:b2", "es", 1),
            ("brainstorm:current:b2", "fr", 2),
            ("proof:autonomous:r:p1", "es", 3),
        ]
        assert queue[0]["viewing"] is True
        assert viewed_row["doc_hash"] == doc_hash["brainstorm:current:b2"]
        positions = {
            row["doc_key"]: row["queue_position"]
            for row in await service.register_documents("fr", False, [brainstorm, paper, proof])
        }
        assert positions == {"brainstorm:current:b1": 8, "paper:s:p1": 6, "proof:autonomous:r:p1": 4}
    finally:
        await service.shutdown()


async def test_expired_viewing_boost_is_reevaluated_at_dispatch(harness):
    service = await harness.start(_settings(paused=True))
    try:
        brainstorm = _doc("brainstorm:current:b1", "A viewed brainstorm about groups.", kind="brainstorm")
        [viewed_hash, paper_hash] = await discover(
            service, brainstorm, _doc("paper:s:p1", "Paper body about lattices.", kind="paper")
        )
        await service.register_documents(None, True, [brainstorm])
        assert service._waiting_jobs()[0] == (viewed_hash, "es")
        # The brainstorm was viewed long ago: its boost must not outrank the paper.
        for entry in service._docs[viewed_hash].keys.values():
            entry.viewed_ts = time.time() - service_module.VIEWING_WINDOW_SECONDS - 60
        assert service._waiting_jobs() == [(paper_hash, "es"), (viewed_hash, "es")]
    finally:
        await service.shutdown()


async def test_a_viewing_lookup_keeps_a_document_in_the_viewing_tier(harness, monkeypatch):
    monkeypatch.setattr(service_module, "QUEUE_EVENT_MIN_INTERVAL_SECONDS", 0.0)
    service = await harness.start(_settings(paused=True))
    try:
        brainstorm = _doc("brainstorm:current:b1", "A viewed brainstorm about groups.", kind="brainstorm")
        paper_doc = _doc("paper:s:p1", "Paper body about lattices.", kind="paper")
        await discover(service, brainstorm, paper_doc)
        viewed = (await service.register_documents(None, True, [brainstorm]))[0]
        paper = {"doc_hash": service._doc_by_key["paper:s:p1"]}
        # The viewer kept the brainstorm open past the viewing window; the expiry timer fires.
        long_ago = time.time() - service_module.VIEWING_WINDOW_SECONDS - 60
        for entry in service._docs[viewed["doc_hash"]].keys.values():
            entry.viewed_ts = long_ago
            entry.record = replace(
                entry.record, last_viewed_at=datetime.fromtimestamp(long_ago, timezone.utc).isoformat()
            )
        service._viewing_window_expired()
        assert service._waiting_jobs() == [(paper["doc_hash"], "es"), (viewed["doc_hash"], "es")]
        # A plain lookup refreshes nothing.
        assert (await service.lookup_documents("es", [viewed["doc_hash"]]))[0]["queue_position"] == 2
        await settle()
        events = len(harness.events_named("translation_queue_changed"))
        viewed_at = datetime.now(timezone.utc).isoformat()
        monkeypatch.setattr(service_module, "_now_iso", lambda: viewed_at)
        (row,) = await service.lookup_documents("es", [viewed["doc_hash"]], viewing=True)
        assert row["queue_position"] == 1
        assert service._waiting_jobs() == [(viewed["doc_hash"], "es"), (paper["doc_hash"], "es")]
        assert service.status_snapshot()["queue"][0]["viewing"] is True
        await wait_until(lambda: len(harness.events_named("translation_queue_changed")) == events + 1)
        # The viewed time is stored as a viewing registration stores it, and its expiry is scheduled.
        stored = _stored_documents()["brainstorm:current:b1"]
        assert (stored.last_viewed_at, stored.last_seen_at) == (viewed_at, viewed_at)
        assert service._viewing_expiry_handle is not None
    finally:
        await service.shutdown()
    assert service._viewing_expiry_handle is None


async def test_frequent_viewing_lookups_save_the_viewed_time_once_per_interval(harness, monkeypatch):
    service = await harness.start(_settings(paused=True))
    keys = [f"proof:autonomous:r:p{index}" for index in range(3)]
    try:
        rows = await service.register_documents(
            None, True, [_doc(key, f"Proof {key} about groups.", kind="proof") for key in keys]
        )
        hashes = [row["doc_hash"] for row in rows]
        registered = {key: _stored_documents()[key].last_viewed_at for key in keys}
        saves: List[List[str]] = []
        mark_viewed = translation_store.mark_documents_viewed

        def recording(doc_keys, now):
            saves.append(list(doc_keys))
            mark_viewed(doc_keys, now)

        monkeypatch.setattr(translation_store, "mark_documents_viewed", recording)
        # The lookups a viewer sends on every queue change keep the viewing tier without a write.
        for _ in range(3):
            positions = [row["queue_position"] for row in await service.lookup_documents("es", hashes, viewing=True)]
            assert positions == [1, 2, 3]
        assert saves == []
        assert {key: _stored_documents()[key].last_viewed_at for key in keys} == registered
        assert all(service._viewing(service._docs[doc_hash]) for doc_hash in hashes)
        # Once the saved time is an interval old, one lookup saves every document in one write.
        later = (
            datetime.now(timezone.utc) + timedelta(seconds=service_module.VIEWED_SAVE_INTERVAL_SECONDS + 1)
        ).isoformat()
        monkeypatch.setattr(service_module, "_now_iso", lambda: later)
        await service.lookup_documents("es", hashes, viewing=True)
        await service.lookup_documents("es", hashes, viewing=True)
        assert saves == [keys]
        assert {key: _stored_documents()[key].last_viewed_at for key in keys} == dict.fromkeys(keys, later)
    finally:
        await service.shutdown()


async def test_queue_positions_follow_a_viewing_window_that_expires(harness, monkeypatch):
    monkeypatch.setattr(service_module, "QUEUE_EVENT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(service_module, "VIEWING_WINDOW_SECONDS", 0.3)
    monkeypatch.setattr(service_module, "VIEWING_EXPIRY_MARGIN_SECONDS", 0.05)
    service = await harness.start(_settings(paused=True))
    try:
        brainstorm = _doc("brainstorm:current:b1", "A viewed brainstorm about groups.", kind="brainstorm")
        await discover(service, _doc("paper:s:p1", "Paper body about lattices.", kind="paper"), brainstorm)
        paper = {"doc_hash": service._doc_by_key["paper:s:p1"]}
        viewed = (await service.register_documents(None, True, [brainstorm]))[0]
        assert viewed["queue_position"] == 1
        await settle()
        events = len(harness.events_named("translation_queue_changed"))
        # Without any request, the queue is re-evaluated once the viewing window ends.
        await wait_until(
            lambda: [item["doc_key"] for item in service.status_snapshot()["queue"]]
            == ["paper:s:p1", "brainstorm:current:b1"],
            timeout=3,
        )
        await wait_until(lambda: len(harness.events_named("translation_queue_changed")) == events + 1)
        rows = await service.lookup_documents("es", [paper["doc_hash"], viewed["doc_hash"]])
        assert [row["queue_position"] for row in rows] == [1, 2]
        assert service.status_snapshot()["queue"][1]["viewing"] is False
        # No viewing window remains, so no timer is pending.
        assert service._viewing_expiry_handle is None
    finally:
        await service.shutdown()


async def test_queue_changes_are_broadcast_at_most_once_per_second(harness, monkeypatch):
    monkeypatch.setattr(service_module, "QUEUE_EVENT_MIN_INTERVAL_SECONDS", 0.3)
    service = await harness.start(_settings(paused=True))
    try:
        await wait_until(lambda: len(harness.events_named("translation_queue_changed")) == 1)
        await asyncio.sleep(0.35)
        await service.register_documents("es", True, [_doc("paper:a")])
        await settle()
        assert len(harness.events_named("translation_queue_changed")) == 2
        first_at = time.monotonic()
        await service.register_documents("es", True, [_doc("paper:b", THREE_PARAGRAPHS)])
        await settle()
        assert len(harness.events_named("translation_queue_changed")) == 2
        await wait_until(lambda: len(harness.events_named("translation_queue_changed")) == 3)
        assert time.monotonic() - first_at >= 0.2
        events = harness.events_named("translation_queue_changed")
        assert [event["queue_version"] for event in events] == [1, 2, 3]
        assert len({event["activity_id"] for event in events}) == 3
        # An unchanged queue sends nothing more.
        await service.register_documents("es", True, [_doc("paper:a")])
        await asyncio.sleep(0.35)
        assert len(harness.events_named("translation_queue_changed")) == 3
    finally:
        await service.shutdown()


async def _blocked_jobs(harness, service, documents: int) -> int:
    gate = asyncio.Event()

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await service.register_documents(
            "es", True, [_doc(f"paper:{index}", f"Paper number {index} body text.") for index in range(documents)]
        )
        await wait_until(lambda: bool(service._jobs))
        await settle()
        # Every running job is blocked inside its model call.
        await wait_until(lambda: harness.fake.in_flight == len(service._jobs))
        return len(service._jobs)
    finally:
        gate.set()


async def test_without_batch_translation_one_job_runs_app_wide(harness):
    service = await harness.start(_settings())
    try:
        assert await _blocked_jobs(harness, service, 3) == 1
        assert service.status_snapshot()["max_parallel_jobs"] == 1
    finally:
        await service.shutdown()


async def test_batch_translation_runs_as_many_jobs_as_cloud_models_allow(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 2)
    service = await harness.start(_settings(batch=True))
    try:
        assert service.status_snapshot()["max_parallel_jobs"] == 2
        assert await _blocked_jobs(harness, service, 4) == 2
        await wait_until(lambda: not service._jobs and service.status_snapshot()["state"] == "idle")
        assert harness.fake.max_in_flight == 2
    finally:
        await service.shutdown()


async def test_batch_translation_with_lm_studio_uses_loaded_copies(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 5)
    loaded = ["local-model:1", "local-model:2", "local-model:3"]

    async def fake_loaded():
        return list(loaded)

    monkeypatch.setattr(lm_studio_client, "get_loaded_models", fake_loaded)
    local = _role("local-model", provider="lm_studio")
    service = await harness.start(_settings(translator=local, validator=local, batch=True))

    def limit() -> int:
        return service.status_snapshot()["max_parallel_jobs"]

    try:
        await wait_until(lambda: limit() == 3)
        # The smaller capacity of the two roles wins; the last count applies at once.
        await service.update_settings(_settings(translator=local, validator=_role("cloud"), batch=True))
        assert limit() == 3
        monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 2)
        await service.update_settings(_settings(translator=local, validator=_role("cloud-2"), batch=True))
        assert limit() == 2
        # One loaded copy means one job once the copies are counted again.
        loaded[:] = ["local-model"]
        await service.update_settings(_settings(translator=local, validator=local, batch=True))
        await wait_until(lambda: limit() == 1)
        # Batch translation off is always one job.
        loaded[:] = ["local-model:1", "local-model:2"]
        await service.update_settings(_settings(translator=local, validator=local, batch=False))
        assert limit() == 1
        await settle()
        assert limit() == 1
    finally:
        await service.shutdown()


class _HeldLookups:
    """A fake LM Studio loaded-model lookup that answers only when released."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self) -> List[str]:
        call: Dict[str, Any] = {"release": asyncio.Event(), "loaded": [], "cancelled": False}
        self.calls.append(call)
        try:
            await call["release"].wait()
        except asyncio.CancelledError:
            call["cancelled"] = True
            raise
        return list(call["loaded"])

    def answer(self, index: int, loaded: List[str]) -> None:
        self.calls[index]["loaded"] = loaded
        self.calls[index]["release"].set()


async def test_lm_studio_copies_are_counted_without_delaying_startup(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 5)
    lookups = _HeldLookups()
    monkeypatch.setattr(lm_studio_client, "get_loaded_models", lookups)
    local = _role("local-model", provider="lm_studio")
    gate = asyncio.Event()

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    service = await asyncio.wait_for(harness.start(_settings(translator=local, validator=local, batch=True)), 2)
    try:
        await wait_until(lambda: len(lookups.calls) == 1)
        # Until the copies are counted, one job runs.
        assert service.status_snapshot()["max_parallel_jobs"] == 1
        await service.register_documents(
            "es", True, [_doc(f"paper:{index}", f"Paper number {index} body text.") for index in range(4)]
        )
        await wait_until(lambda: harness.fake.in_flight == 1)
        await settle()
        assert len(service._jobs) == 1
        # The count raises the limit and wakes the worker.
        lookups.answer(0, ["local-model:1", "local-model:2", "local-model:3"])
        await wait_until(lambda: harness.fake.in_flight == 3)
        assert service.status_snapshot()["max_parallel_jobs"] == 3
        # A settings save does not wait for its count either.
        await asyncio.wait_for(
            service.update_settings(_settings(translator=local, validator=local, batch=True, languages=TWO_LANGUAGES)),
            2,
        )
        await wait_until(lambda: len(lookups.calls) == 2)
        assert service.status_snapshot()["max_parallel_jobs"] == 3
    finally:
        gate.set()
        await service.shutdown()
    # Shutdown cancels a count still in progress.
    assert lookups.calls[1]["cancelled"] is True
    assert service._background_tasks == set()


async def test_a_count_started_for_older_settings_is_discarded(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 2)
    lookups = _HeldLookups()
    monkeypatch.setattr(lm_studio_client, "get_loaded_models", lookups)
    local = _role("local-model", provider="lm_studio")
    service = await harness.start(_settings(translator=local, validator=local, batch=True, paused=True))
    try:
        await wait_until(lambda: len(lookups.calls) == 1)
        await service.update_settings(_settings(translator=local, validator=_role("cloud"), batch=True))
        await wait_until(lambda: len(lookups.calls) == 2)
        lookups.answer(0, ["local-model:1", "local-model:2", "local-model:3"])
        await settle()
        assert service.status_snapshot()["max_parallel_jobs"] == 1
        lookups.answer(1, ["local-model:1", "local-model:2", "local-model:3"])
        await wait_until(lambda: service.status_snapshot()["max_parallel_jobs"] == 2)
    finally:
        await service.shutdown()


def _calls_for_paper(harness, index: int) -> int:
    return sum(f"Paper number {index} " in _block(call, "SOURCE") for call in harness.fake.translator_calls())


async def test_turning_batch_translation_off_makes_the_excess_running_jobs_yield(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 3)
    service = await harness.start(_settings(batch=True))
    running_when_judged: List[int] = []

    def validator(record):
        running_when_judged.append(len(service._jobs))
        return _decisions(_judged(record), accept=False)

    # Every attempt is rejected, so no part ever finalizes on its own.
    harness.fake.validator = validator
    try:
        rows = await service.register_documents(
            "es", True, [_doc(f"paper:{index}", f"Paper number {index} body text.") for index in range(4)]
        )
        await wait_until(lambda: len(service._jobs) == 3 and all(_calls_for_paper(harness, i) >= 2 for i in range(3)))
        status = await service.update_settings(_settings(batch=False))
        assert status["max_parallel_jobs"] == 1
        await wait_until(lambda: len(service._jobs) == 1)
        judged = len(running_when_judged)
        await wait_until(lambda: len(running_when_judged) >= judged + 5)
        assert set(running_when_judged[judged:]) == {1}
        # The highest-priority job keeps running; the others wait in line with their attempts saved.
        assert list(service._jobs) == [(rows[0]["doc_hash"], "es")]
        for row in rows[1:3]:
            part = _job_state(service, row["doc_hash"])["part"]
            assert any(attempt["status"] == STATUS_REJECTED for attempt in part["attempts"])
        assert [(item["doc_key"], item["position"]) for item in service.status_snapshot()["queue"]] == [
            ("paper:0", 0),
            ("paper:1", 1),
            ("paper:2", 2),
            ("paper:3", 3),
        ]
    finally:
        await service.shutdown()


async def test_a_lowered_job_limit_applies_before_running_jobs_are_cancelled(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 3)
    service = await harness.start(_settings(batch=True))
    gate = asyncio.Event()

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    started: List[int] = []
    start_job = service._start_job

    def recording_start(doc_hash: str, language_key: str) -> None:
        start_job(doc_hash, language_key)
        started.append(sum(1 for job in service._jobs.values() if not job.cancelled))

    service._start_job = recording_start
    try:
        await service.register_documents(
            "es", True, [_doc(f"paper:{index}", f"Paper number {index} body text.") for index in range(4)]
        )
        await wait_until(lambda: harness.fake.in_flight == 3)
        before = len(started)
        # A new validator cancels every running job while Batch translation turns off.
        await service.update_settings(_settings(batch=False, validator=_role("judge-2")))
        await wait_until(lambda: harness.fake.in_flight == 1)
        await settle()
        assert started[before:] and max(started[before:]) == 1
        assert len(service._jobs) == 1
    finally:
        gate.set()
        await service.shutdown()


# ----------------------------------------------------------------------
# Changed documents, cross-document reuse, repeated units
# ----------------------------------------------------------------------


def _views(text: str):
    plan = build_document_plan(PROFILE_DOCUMENT, [("content", text)])
    return plan, tuple(build_unit_views(plan))


def _record(number: int, units: Dict[str, str], *, status: str = STATUS_ACCEPTED, order: Optional[int] = 1, edits=()):
    return {
        "number": number, "action": "translate", "units": units, "edits": list(edits), "keep_attempts": [],
        "notes": "", "status": status, "feedback": "", "reasoning": "",
        "accepted_order": order if status == STATUS_ACCEPTED else None,
    }


async def test_reconcile_job_state_keeps_matching_finals_and_unchanged_parts():
    _old_plan, old_views = _views("Alpha.\n\nBeta.\n\nGamma.\n\nDelta.")
    new_plan, new_views = _views("Alpha.\n\nInserted.\n\nBeta.\n\nGamma.\n\nDelta.")
    old_state = {
        "version": 1, "doc_hash": "old", "language_key": "es",
        "finals": {"0": "Alfa.", "1": "Beta (es)."},
        "part": {
            "start": 2, "end": 3, "locked": True,
            "attempts": [
                _record(1, {"2": "Gama.", "3": "Delta (es)."}, edits=[
                    {"occurrence": 0, "find": "Alfa", "replace": "ALFA"}
                ]),
                {**_record(2, {}, status=STATUS_KEPT), "action": "keep", "keep_attempts": [1]},
            ],
            "keep_streak": 1, "truncations": {"translation_submitter": 0, "translation_validator": 0},
        },
    }
    state = service_module.reconcile_job_state(
        old_state, old_views, new_views, doc_hash=new_plan.doc_hash, language_key="es"
    )
    assert state["doc_hash"] == new_plan.doc_hash and state["language_key"] == "es"
    assert state["finals"] == {"0": "Alfa.", "2": "Beta (es)."}
    part = state["part"]
    assert (part["start"], part["end"], part["keep_streak"]) == (3, 4, 1)
    assert part["attempts"][0]["units"] == {"3": "Gama.", "4": "Delta (es)."}
    assert part["attempts"][0]["edits"] == [{"occurrence": 0, "find": "Alfa", "replace": "ALFA"}]

    # A part over changed content is dropped; matching finals stay.
    _changed_plan, changed_views = _views("Alpha.\n\nBeta.\n\nGamma changed.\n\nDelta.")
    dropped = service_module.reconcile_job_state(
        old_state, old_views, changed_views, doc_hash="new", language_key="es"
    )
    assert dropped["part"] is None
    assert dropped["finals"] == {"0": "Alfa.", "1": "Beta (es)."}


async def test_reconcile_drops_only_the_attempts_over_changed_units():
    old_plan, old_views = _views("Alpha text.\n\nBeta text.\n\nGamma text.")
    _new_plan, new_views = _views("Alpha text.\n\nBeta text.\n\nGamma text changed.")

    def keep(number, named, seq):
        return {**_record(number, {}, status=STATUS_KEPT), "action": "keep", "keep_attempts": named, "seq": seq}

    rejected = {**_record(1, {"1": "Beta x.", "2": "Gamma x."}, status=STATUS_REJECTED), "seq": 4}
    accepted = {**_record(2, {"1": "Beta (es)."}), "seq": 2}
    old_state = {
        "version": 1, "doc_hash": old_plan.doc_hash, "language_key": "es", "finals": {"0": "Alfa."},
        "part": {
            "start": 1, "end": 1, "locked": True, "keep_streak": 1, "seq": 5,
            # Attempt 3 disputed attempt 1, which was rejected again; attempt 4 kept attempt 2.
            "attempts": [rejected, accepted, keep(3, [1], 3), keep(4, [2], 5)],
            "truncations": {"translation_submitter": 0, "translation_validator": 0},
        },
    }
    # Only the rejected attempt touched the changed unit: it and the keep naming it go, the part stays.
    state = service_module.reconcile_job_state(old_state, old_views, new_views, doc_hash="new", language_key="es")
    part = state["part"]
    assert [record["number"] for record in part["attempts"]] == [2, 4]
    assert (part["start"], part["end"], part["locked"], part["keep_streak"]) == (1, 1, True, 1)

    # A rejected attempt whose edit targets a changed unit is dropped too.
    edited = {**rejected, "units": {"1": "Beta x."}, "edits": [{"occurrence": 0, "find": "Alfa", "replace": "A"}]}
    _moved_plan, moved_views = _views("Alpha text changed.\n\nBeta text.\n\nGamma text.")
    moved = service_module.reconcile_job_state(
        {**old_state, "part": {**old_state["part"], "attempts": [edited, accepted, keep(4, [2], 5)]}},
        old_views, moved_views, doc_hash="new", language_key="es",
    )
    assert [record["number"] for record in moved["part"]["attempts"]] == [2, 4]

    # An accepted attempt over a changed unit still drops the whole part.
    locked_changed = {**old_state, "part": {**old_state["part"], "start": 2, "end": 2,
                                            "attempts": [{**accepted, "units": {"2": "Gama (es)."}}]}}
    assert service_module.reconcile_job_state(
        locked_changed, old_views, new_views, doc_hash="new", language_key="es"
    )["part"] is None


async def test_a_changed_document_keeps_matching_translations(harness):
    service = await harness.start(_settings())
    harness.fake.translator = _one_unit_parts
    try:
        first = await register_and_complete(service, _doc("paper:s:p1", THREE_PARAGRAPHS))
        old_hash = first["doc_hash"]
        harness.fake.calls.clear()
        changed = "Alpha paragraph text.\n\nBeta paragraph changed.\n\nGamma paragraph text."
        row = (await service.register_documents("es", True, [_doc("paper:s:p1", changed)]))[0]
        assert row["doc_hash"] != old_hash
        assert (row["status"], row["translated_units"], row["total_units"]) == ("pending", 2, 3)
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        tasks = [_task(call) for call in harness.fake.translator_calls()]
        assert {task["start"] for task in tasks} == {2}
        status = await lookup_status(service, row["doc_hash"])
        assert status["parts"]["content"] == (
            "Alpha paragraph text. (es)\n\nBeta paragraph changed. (es)\n\nGamma paragraph text. (es)"
        )
        # The old content's plan and job state are gone with its last key.
        assert old_hash not in service._docs
        assert translation_store.get_job_state(old_hash, "es") is None
    finally:
        await service.shutdown()


async def test_finalized_units_are_reused_across_documents(harness):
    service = await harness.start(_settings())
    try:
        await register_and_complete(service, _doc("paper:s:p1", THREE_PARAGRAPHS))
        harness.fake.calls.clear()
        other = (await service.register_documents(
            "es", True, [_doc("paper:s:p2", "Gamma paragraph text.\n\nDelta paragraph text.")]
        ))[0]
        await wait_until(lambda: _complete(service, other["doc_hash"]))
        first = harness.fake.translator_calls()[0]
        assert _task(first)["start"] == 2
        assert _translation_units(first)[1] == "Gamma paragraph text. (es)"
        status = await lookup_status(service, other["doc_hash"])
        assert status["parts"]["content"] == "Gamma paragraph text. (es)\n\nDelta paragraph text. (es)"
    finally:
        await service.shutdown()


async def test_repeated_units_finalize_together(harness):
    service = await harness.start(_settings())
    harness.fake.translator = _one_unit_parts
    try:
        status = await register_and_complete(
            service, _doc(text="Repeat me here.\n\nOther text here.\n\nRepeat me here.")
        )
        first_tasks = [_task(call) for call in harness.fake.translator_calls() if _task(call)["kind"] == "first"]
        assert [(task["start"], task["end"]) for task in first_tasks] == [(1, 3), (2, 2)]
        assert status["parts"]["content"] == "Repeat me here. (es)\n\nOther text here. (es)\n\nRepeat me here. (es)"
        assert len(harness.events_named("translation_part_finalized")) == 2
    finally:
        await service.shutdown()


async def test_normalize_job_state_drops_what_it_cannot_trust():
    new, changed = service_module.normalize_job_state(None, doc_hash="h", language_key="es", unit_count=3)
    assert (new, changed) == (service_module.new_job_state("h", "es"), False)
    stale, changed = service_module.normalize_job_state(
        {"version": 0, "doc_hash": "h", "language_key": "es"}, doc_hash="h", language_key="es", unit_count=3
    )
    assert stale["finals"] == {} and changed is True

    part = {"start": 1, "end": 1, "locked": False,
            "attempts": [{**_record(1, {"1": "x"}, status=STATUS_REJECTED), "seq": 3}],
            "keep_streak": 0, "seq": 3, "truncations": {"translation_submitter": 1, "translation_validator": 0}}
    good = {"version": 1, "doc_hash": "h", "language_key": "es", "finals": {"0": "a"}, "part": part}
    same, changed = service_module.normalize_job_state(good, doc_hash="h", language_key="es", unit_count=3)
    assert same == good and changed is False
    # Records stored without a judgment sequence take their attempt number; the part counter follows.
    legacy_part = {key: value for key, value in part.items() if key != "seq"}
    legacy_part["attempts"] = [_record(1, {"1": "x"}, status=STATUS_REJECTED),
                               _record(4, {"1": "y"}, status=STATUS_REJECTED)]
    legacy, changed = service_module.normalize_job_state(
        {**good, "part": legacy_part}, doc_hash="h", language_key="es", unit_count=3
    )
    assert [record["seq"] for record in legacy["part"]["attempts"]] == [1, 4]
    assert legacy["part"]["seq"] == 4 and changed is True
    # Out-of-range finals are dropped, and an unlocked part must start at the cursor.
    moved = {**good, "finals": {"0": "a", "1": "b", "9": "z"}, "part": {**part, "start": 2, "end": 2}}
    fixed, changed = service_module.normalize_job_state(moved, doc_hash="h", language_key="es", unit_count=3)
    assert fixed["finals"] == {"0": "a", "1": "b"} and fixed["part"] is not None and changed is True
    off_cursor = {**good, "finals": {}, "part": part}
    fixed, _ = service_module.normalize_job_state(off_cursor, doc_hash="h", language_key="es", unit_count=3)
    assert fixed["part"] is None
    # A part overlapping a final translation is dropped.
    overlap = {**good, "finals": {"0": "a", "1": "b"}, "part": {**part, "locked": True}}
    fixed, _ = service_module.normalize_job_state(overlap, doc_hash="h", language_key="es", unit_count=3)
    assert fixed["part"] is None


async def test_stored_attempts_keep_accepted_ones_and_the_newest_others():
    attempts = [_record(1, {"0": "a"}, order=1)]
    attempts += [_record(number, {"0": f"r{number}"}, status=STATUS_REJECTED) for number in range(2, 15)]
    part = {"attempts": attempts}
    service_module.prune_attempt_records(part)
    assert [record["number"] for record in part["attempts"]] == [1, *range(5, 15)]


def _kept(number: int, named: List[int], seq: int) -> Dict[str, Any]:
    return {**_record(number, {}, status=STATUS_KEPT), "action": "keep", "keep_attempts": named, "seq": seq}


def _rejected(number: int, seq: int) -> Dict[str, Any]:
    return {**_record(number, {"0": f"r{number}"}, status=STATUS_REJECTED), "seq": seq}


async def test_pruning_orders_by_judgment_and_never_lets_keeps_displace_rejections():
    # Attempts 1-11 are rejected; then twelve disputes each re-judge attempt 1, which is rejected again.
    attempts = [_rejected(number, number) for number in range(1, 12)]
    seq = 11
    for number in range(12, 24):
        attempts.append(_kept(number, [1], seq + 1))
        seq += 2
    attempts[0]["seq"] = seq
    part = {"attempts": attempts}
    service_module.prune_attempt_records(part)
    # The re-judged attempt is the newest rejection; only the keeps the stop rule
    # and the prompt need remain (the newest two, the last of which disputed attempt 1).
    assert [record["number"] for record in part["attempts"]] == [1, *range(3, 12), 22, 23]

    # A stored keep keeps every attempt it names, even an old rejection.
    attempts = [{**_rejected(1, 3)}, _kept(2, [1], 2), *(_rejected(number, number + 1) for number in range(3, 14))]
    part = {"attempts": attempts}
    service_module.prune_attempt_records(part)
    assert [record["number"] for record in part["attempts"]] == [1, 2, *range(4, 14)]


async def test_part_stop_decision_follows_the_rule():
    accepted = [_record(number, {"0": "a"}, order=order) for number, order in ((1, 1), (3, 2))]
    keep = {**_record(4, {}, status=STATUS_KEPT), "action": "keep", "keep_attempts": [1]}
    part = {"attempts": [*accepted, keep, {**keep, "number": 5}], "keep_streak": 2}
    reason, record = service_module.part_stop_decision(part)
    assert (reason, record["number"]) == ("kept_twice", 1)
    part = {"attempts": [*accepted, _record(6, {"0": "c"}, order=3)], "keep_streak": 0}
    reason, record = service_module.part_stop_decision(part)
    assert (reason, record["number"]) == ("three_accepted", 6)
    assert service_module.part_stop_decision({"attempts": accepted, "keep_streak": 1}) is None
    assert service_module.part_stop_decision({"attempts": [], "keep_streak": 5}) is None


async def test_kept_twice_finalizes_with_the_attempt_both_keeps_name():
    accepted = [_record(number, {"0": "a"}, order=order) for number, order in ((1, 1), (2, 2))]
    both = {"attempts": [*accepted, _kept(3, [1], 3), _kept(4, [1, 2], 4)], "keep_streak": 2}
    assert service_module.part_stop_decision(both)[1]["number"] == 1
    # Without a common attempt, the latest accepted attempt the last keep names wins.
    apart = {"attempts": [*accepted, _kept(3, [1], 3), _kept(4, [2], 4)], "keep_streak": 2}
    assert service_module.part_stop_decision(apart)[1]["number"] == 2
    # The last keep is the last one judged, not the highest attempt number.
    reordered = {"attempts": [*accepted, _kept(3, [1], 4), _kept(4, [2], 3)], "keep_streak": 2}
    assert service_module.part_stop_decision(reordered)[1]["number"] == 1


# ----------------------------------------------------------------------
# Call scope and activity identity
# ----------------------------------------------------------------------


async def test_calls_run_in_background_scope_with_supercharge_and_boost_disabled(harness):
    service = await harness.start(_settings())
    harness.fake.track_tokens = True
    before = token_tracker.get_stats()["total_input"]
    try:
        await register_and_complete(service)
        assert harness.fake.calls
        for call in harness.fake.calls:
            assert call["scope"] == "translation"
            assert call["kwargs"]["_moto_disable_supercharge"] is True
            assert call["kwargs"]["_moto_force_boost_mode"] == "__none__"
            assert call["max_tokens"] == 4000
        assert {call["role_id"] for call in harness.fake.calls} == {"translation_submitter", "translation_validator"}
        assert token_tracker.get_stats()["total_input"] == before
        assert all(re.fullmatch(r"translation_sub_\d{5}", call["task_id"]) for call in harness.fake.translator_calls())
        assert all(re.fullmatch(r"translation_val_\d{5}", call["task_id"]) for call in harness.fake.validator_calls())
        usage = service.status_snapshot()["usage"]
        assert usage["calls"] == 4
        assert usage["prompt_tokens"] == 44 and usage["completion_tokens"] == 28
    finally:
        await service.shutdown()


DROPPED_PLACEHOLDER = {1: "Primer párrafo.", 2: "Segundo párrafo."}


async def test_activity_ids_distinguish_attempts_parts_and_languages(harness):
    service = await harness.start(_settings(languages=TWO_LANGUAGES))
    translator_calls: Dict[str, int] = {}

    def translator(record):
        language = _target_language(record)
        translator_calls[language] = translator_calls.get(language, 0) + 1
        count = translator_calls[language]
        if count == 1:
            return _translate(DROPPED_PLACEHOLDER)
        if _task(record)["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        return _translate(_translated_part(record, suffix=f"(v{count})"))

    def validator(record):
        number = _judged(record)[0]
        return _decisions(_judged(record), accept=number >= 3, feedback=f"fix-{number}")

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        row = (await service.register_documents(None, True, [_doc()]))[0]
        await wait_until(lambda: _complete(service, row["doc_hash"], "es") and _complete(service, row["doc_hash"], "fr"))
        for name, expected in (
            ("translation_attempt_check_failed", 2),
            ("translation_attempt_rejected", 2),
            ("translation_attempt_accepted", 2),
            ("translation_attempt_kept", 4),
            ("translation_part_finalized", 2),
        ):
            events = harness.events_named(name)
            assert len(events) == expected, name
            ids = [normalize_activity_payload(name, event)["activity_id"] for event in events]
            assert len(set(ids)) == expected, name
            # The broadcast payload already carries the id the websocket would assign.
            assert ids == [event["activity_id"] for event in events]
        check = harness.events_named("translation_attempt_check_failed")[0]
        assert check["target_hash"] == f"{row['doc_hash']}:{check['language_key']}:1"
        assert check["candidate_id"] == f"{check['target_hash']}:attempt1"
        assert sorted(
            (event["language_key"], event["round_index"]) for event in harness.events_named("translation_attempt_rejected")
        ) == [("es", 2), ("fr", 2)]
    finally:
        await service.shutdown()


async def test_finalized_parts_of_one_document_have_distinct_identities(harness):
    service = await harness.start(_settings())
    harness.fake.translator = _one_unit_parts
    try:
        await register_and_complete(service, _doc(text=THREE_PARAGRAPHS))
        finalized = harness.events_named("translation_part_finalized")
        assert [event["part_start"] for event in finalized] == [1, 2, 3]
        assert len({event["target_hash"] for event in finalized}) == 3
        assert len({event["activity_id"] for event in finalized}) == 3
        accepted = harness.events_named("translation_attempt_accepted")
        assert len({event["activity_id"] for event in accepted}) == 3
    finally:
        await service.shutdown()


async def test_validator_contract_retries_in_one_attempt_have_distinct_ids(harness):
    service = await harness.start(_settings())
    calls: List[int] = []

    def validator(record):
        calls.append(1)
        if len(calls) <= 4:
            return _response("no JSON from the validator at all, just words")
        return _accept_all(record)

    harness.fake.validator = validator
    try:
        await register_and_complete(service)
        retries = harness.events_named("translation_validator_contract_retry")
        assert [event["attempt"] for event in retries] == [1, 2]
        assert [event["round_index"] for event in retries] == [1, 1]
        assert len({event["activity_id"] for event in retries}) == 2
    finally:
        await service.shutdown()


async def test_recent_activity_keeps_decisions_and_waits_with_live_event_ids(harness):
    service = await harness.start(_settings())
    translator_calls: List[int] = []
    validator_calls: List[int] = []

    def translator(record):
        translator_calls.append(1)
        if len(translator_calls) == 1:
            return RuntimeError("socket closed unexpectedly")
        if len(translator_calls) == 2:
            return _translate(DROPPED_PLACEHOLDER)
        if _task(record)["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        return _translate(_translated_part(record, suffix=f"(v{len(translator_calls)})"))

    def validator(record):
        validator_calls.append(1)
        return _decisions(_judged(record), accept=len(validator_calls) > 1)

    harness.fake.translator = translator
    harness.fake.validator = validator
    try:
        await register_and_complete(service)
        recent = service.status_snapshot()["recent_activity"]
        types = [item["event_type"] for item in recent]
        assert not {"translation_progress", "translation_status_changed", "translation_queue_changed"} & set(types)
        assert set(types) <= service_module.RECENT_ACTIVITY_EVENTS
        assert {
            "translation_provider_retry",
            "translation_attempt_check_failed",
            "translation_attempt_rejected",
            "translation_attempt_accepted",
            "translation_attempt_kept",
            "translation_part_finalized",
        } <= set(types)

        live = {
            payload["activity_id"]: payload
            for name, payload in harness.events
            if name in service_module.RECENT_ACTIVITY_EVENTS
        }
        broadcast_entries = [item for item in recent if item["event_type"] != "translation_provider_retry"]
        for item in broadcast_entries:
            assert item["activity_id"] in live
            assert item["activity_message"] == live[item["activity_id"]]["activity_message"]
        finalized = next(item for item in recent if item["event_type"] == "translation_part_finalized")
        assert (finalized["part_start"], finalized["part_end"], finalized["final_attempt"]) == (1, 2, 3)
        assert finalized["reason"] == "kept_twice"
        rejected = next(item for item in recent if item["event_type"] == "translation_attempt_rejected")
        assert rejected["feedback"] == "Fix the terminology."
        kept = next(item for item in recent if item["event_type"] == "translation_attempt_kept")
        assert kept["keep_attempts"] == [3]

        [retry] = [item for item in recent if item["event_type"] == "translation_provider_retry"]
        # A non-model wait names the workflow and role only, once, at the start of the line.
        assert retry["route_identity"] == "[Foreign Language Translation w/ Top-p · Translator Submitter]"
        assert retry["message"] == (
            "[Foreign Language Translation w/ Top-p · Translator Submitter] An unexpected error interrupted the attempt; "
            "retrying in 10s."
        )
        assert retry["activity_message"] == retry["message"]
        assert retry["retry_after_seconds"] == 10
        assert retry["error_summary"] == "RuntimeError: socket closed unexpectedly"
        assert retry["activity_id"].startswith("translation_provider_retry:")
    finally:
        await service.shutdown()


async def test_repeated_waits_keep_distinct_activity_ids(harness):
    service = await harness.start(_settings())
    try:
        route = service._non_model_route("translation_submitter")
        service._note_retry_wait("translation_submitter", "translation_loop_retry", 30, route=route)
        service._note_retry_wait("translation_submitter", "translation_loop_retry", 45, route=route)
        waits = [
            item for item in service.status_snapshot()["recent_activity"]
            if item["event_type"] == "translation_provider_retry"
        ]
        assert len(waits) == 2
        assert waits[0]["activity_id"] != waits[1]["activity_id"]
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# Availability, controls, fencing, shutdown
# ----------------------------------------------------------------------


async def test_disabled_service_returns_disabled_without_registering(harness):
    service = await harness.start(_settings(enabled=False))
    try:
        rows = await service.register_documents("es", True, [_doc()])
        assert rows[0]["status"] == "disabled"
        assert rows[0]["queue_position"] is None
        assert rows[0]["doc_hash"]
        assert service._docs == {}
        assert translation_store.load_documents() == []
        assert service.status_snapshot()["state"] == "disabled"
        assert (await service.lookup_documents("es", [rows[0]["doc_hash"]]))[0]["status"] == "disabled"
    finally:
        await service.shutdown()


async def test_unselected_language_is_unavailable(harness):
    service = await harness.start(_settings(paused=True))
    try:
        rows = await service.register_documents("fr", True, [_doc()])
        assert rows[0]["status"] == "unavailable"
        assert rows[0]["doc_hash"] in service._docs
        assert (await service.lookup_documents("fr", [rows[0]["doc_hash"]]))[0]["status"] == "unavailable"
        assert (await service.lookup_documents("es", ["missing-hash"]))[0]["status"] == "unknown"
    finally:
        await service.shutdown()


async def test_pause_cancels_in_flight_work_and_resume_continues(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()
    cancelled: List[bool] = []
    entered: List[bool] = []

    async def blocked(record):
        if len(harness.fake.translator_calls()) == 1:
            entered.append(True)
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        return _translate_ok(record)

    harness.fake.translator = blocked
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: bool(entered))
        status = await service.control("pause")
        assert status["state"] == "paused" and status["paused"] is True
        assert cancelled == [True]
        assert service._jobs == {}
        await settle()
        assert service.get_settings().paused is True
        assert (await lookup_status(service, row["doc_hash"]))["status"] == "pending"

        await service.control("resume")
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        assert (await lookup_status(service, row["doc_hash"]))["status"] == "complete"
    finally:
        gate.set()
        await service.shutdown()


async def test_settings_change_fences_stale_validation(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()
    validations: List[Dict[str, Any]] = []

    async def validator(record):
        validations.append(record)
        if len(validations) == 1:
            await gate.wait()
        return _accept_all(record)

    harness.fake.validator = validator
    try:
        row = (await service.register_documents("es", True, [_doc()]))[0]
        await wait_until(lambda: len(validations) == 1)
        await service.update_settings(_settings(validator=_role("judge-2")))
        gate.set()
        await wait_until(lambda: _complete(service, row["doc_hash"]))
        # The accepted attempt came from the new validator configuration.
        accepted = harness.events_named("translation_attempt_accepted")
        assert [event["validator_model"] for event in accepted] == ["judge-2"]
        assert len(validations) == 2
        assert (await lookup_status(service, row["doc_hash"]))["status"] == "complete"
    finally:
        gate.set()
        await service.shutdown()


async def test_disable_cancels_work_and_clear_cache_retranslates(harness):
    service = await harness.start(_settings())
    try:
        status = await register_and_complete(service)
        assert status["status"] == "complete"
        doc_hash = status["doc_hash"]
        status_generation = service.status_snapshot()["generation"]

        during: List[int] = []
        clear_and_restore = service._clear_and_restore_sync

        def recording_clear(records):
            during.append(service._generation)
            clear_and_restore(records)

        service._clear_and_restore_sync = recording_clear
        cleared = await service.clear_cache()
        assert cleared["success"] is True
        # Viewers that registered while the clear ran see a newer generation afterwards.
        assert cleared["generation"] > during[0] > status_generation
        await wait_until(lambda: service.status_snapshot()["usage"]["finalized_parts"] == 2 and not service._jobs)
        assert (await lookup_status(service, doc_hash))["status"] == "complete"
        assert [record.doc_key for record in _keys_for_hash(translation_store, doc_hash)] == ["paper:test:p1"]

        disabled = await service.control("disable")
        assert disabled["state"] == "disabled"
        assert service.get_settings().enabled is False
        assert (await service.register_documents("es", True, [_doc("paper:test:p2")]))[0]["status"] == "disabled"
        enabled = await service.control("enable")
        assert enabled["enabled"] is True
        with pytest.raises(ValueError):
            await service.control("explode")
    finally:
        await service.shutdown()


async def test_clear_cache_while_a_job_runs_never_saves_old_translations(harness):
    service = await harness.start(_settings())
    phase = {"suffix": "(old)"}
    gate = asyncio.Event()
    gate.set()
    tasks: List[Dict[str, Any]] = []

    async def translator(record):
        task = _task(record)
        tasks.append({**task, "suffix": phase["suffix"]})
        await gate.wait()
        if task["kind"] == "locked":
            return _keep(_accepted_numbers(record)[-1:])
        return _translate(_translated_part(record, suffix=phase["suffix"], end=task["start"]))

    harness.fake.translator = translator
    starts_while_clearing: List[bool] = []
    start_job = service._start_job

    def recording_start(doc_hash: str, language_key: str) -> None:
        starts_while_clearing.append(service._clearing)
        start_job(doc_hash, language_key)

    service._start_job = recording_start
    try:
        row = (await service.register_documents("es", True, [_doc("paper:a", THREE_PARAGRAPHS)]))[0]
        doc_hash = row["doc_hash"]
        # U1 is final with the old text; the clear runs while U2 is being translated.
        await wait_until(lambda: service._final_counts.get((doc_hash, "es"), 0) >= 1)
        gate.clear()
        await wait_until(lambda: any(task["start"] == 2 for task in tasks))
        phase["suffix"] = "(new)"
        await service.clear_cache()
        gate.set()
        await wait_until(lambda: _complete(service, doc_hash))
        assert not any(starts_while_clearing)
        finals = _job_state(service, doc_hash)["finals"]
        assert sorted(finals) == ["0", "1", "2"]
        assert all(text.endswith("(new)") for text in finals.values())
        assert any(task["start"] == 1 and task["suffix"] == "(new)" for task in tasks)
        content = (await lookup_status(service, doc_hash))["parts"]["content"]
        assert "(old)" not in content and content.count("(new)") == 3
    finally:
        gate.set()
        await service.shutdown()


async def test_shutdown_cancels_in_flight_calls_cleanly(harness):
    service = await harness.start(_settings())
    cancelled: List[bool] = []
    entered: List[bool] = []

    async def hang(record):
        entered.append(True)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    harness.fake.translator = hang
    await service.register_documents("es", True, [_doc()])
    await wait_until(lambda: bool(entered))
    await service.shutdown()
    assert cancelled == [True]
    assert service._jobs == {}
    assert service._background_tasks == set()
    assert service._observe_event not in websocket_routes._event_observers


async def test_observed_output_events_trigger_a_debounced_scan(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    scans: List[int] = []
    spec = DocumentSpec(
        doc_key="brainstorm:current:t1",
        kind="brainstorm",
        title="Topic",
        profile="document",
        parts=(("content", "An accepted brainstorm idea about symmetric groups."),),
    )

    async def discover(modes) -> DiscoveryResult:
        scans.append(set(modes))
        return DiscoveryResult(documents=(spec,))

    harness.service._discover = discover
    service = await harness.start(_settings())
    try:
        await wait_until(lambda: len(scans) >= 1)
        assert scans[0] == {"autonomous", "aggregator", "compiler"}
        await wait_until(lambda: service.status_snapshot()["usage"]["finalized_parts"] >= 1 and not service._jobs)
        before = len(scans)
        websocket_routes._notify_event_observers("paper_completed", {})
        await wait_until(lambda: len(scans) > before)
    finally:
        await service.shutdown()


async def test_discovery_keeps_documents_of_failed_sources_targeted(harness):
    service = await harness.start(_settings(paused=True))
    try:
        paper = DocumentSpec(
            "paper:s:p1", "paper", "P", "document", (("content", "Paper text about rings."),), "autonomous_papers"
        )
        proof = DocumentSpec(
            "proof:manual:r:x", "proof", "L", "fields", (("theorem_statement", "Rings exist."),), "aggregator_proofs"
        )
        await service._apply_discovery(DiscoveryResult(documents=(paper, proof)))
        assert service._auto_keys == {"paper:s:p1": "autonomous_papers", "proof:manual:r:x": "aggregator_proofs"}
        await service._apply_discovery(DiscoveryResult(documents=(), failed_sources=("aggregator_proofs",)))
        assert service._auto_keys == {"proof:manual:r:x": "aggregator_proofs"}
        languages = service.status_snapshot()["languages"][0]
        assert languages["pending_documents"] == 1
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# Language catalog and probe
# ----------------------------------------------------------------------


async def test_language_catalog_and_probe(harness):
    service = await harness.start(_settings(translator=_role("t-1"), validator=_role("judge")))

    def probe(record):
        if record["model"] == "t-1":
            return _response(json.dumps({"languages": ["Spanish", "French", "Klingon"]}))
        return _response(json.dumps({"languages": ["Spanish", "German", "Klingon"]}))

    harness.fake.probe = probe
    try:
        catalog = await service.language_catalog()
        assert catalog["probe"]["status"] == "never"
        assert {item["proficiency"] for item in catalog["languages"]} == {"unknown"}
        result = await service.probe_languages()
        assert result["probe"]["status"] == "complete"
        proficiency = {item["key"]: item["proficiency"] for item in result["languages"]}
        # Proficient only when both the translator and the validator list it.
        assert proficiency["es"] == "proficient"
        assert proficiency["fr"] == "uncertain" and proficiency["de"] == "uncertain"
        assert [extra["name"] for extra in result["extras"]] == ["Klingon"]
        probe_roles = {call["role_id"] for call in harness.fake.role_calls("translation_language_probe_1")}
        probe_roles |= {call["role_id"] for call in harness.fake.role_calls("translation_language_probe_2")}
        assert probe_roles == {"translation_language_probe_1", "translation_language_probe_2"}
        assert {row["role"]: row["ok"] for row in result["probe"]["models"]} == {"validator": True, "translator": True}

        await service.update_settings(_settings(translator=_role("t-9"), validator=_role("judge")))
        stale = await service.language_catalog()
        assert stale["probe"]["stale"] is True
        assert {item["proficiency"] for item in stale["languages"]} == {"unknown"}
    finally:
        await service.shutdown()


async def test_probe_fails_when_the_translator_model_cannot_answer(harness):
    service = await harness.start(_settings(translator=_role("t-1"), validator=_role("judge")))

    def probe(record):
        if record["model"] == "t-1":
            return RuntimeError("probe failed")
        return _response(json.dumps({"languages": ["Spanish"]}))

    harness.fake.probe = probe
    try:
        result = await service.probe_languages()
        assert result["probe"]["status"] == "failed"
        assert {item["proficiency"] for item in result["languages"]} == {"unknown"}
        models = {row["role"]: (row["ok"], bool(row["error"])) for row in result["probe"]["models"]}
        assert models == {"validator": (True, False), "translator": (False, True)}
    finally:
        await service.shutdown()


async def test_probe_result_is_tagged_with_the_probed_model_set(harness):
    service = await harness.start(_settings(translator=_role("t-1"), validator=_role("judge")))
    gate = asyncio.Event()

    async def probe(record):
        await gate.wait()
        return _response(json.dumps({"languages": ["Spanish"]}))

    harness.fake.probe = probe
    try:
        task = asyncio.create_task(service.probe_languages())
        await wait_until(lambda: len([c for c in harness.fake.calls if c["role_id"].startswith("translation_language_probe_")]) == 2)
        # The validator model changes while the probe is still asking the old one.
        await service.update_settings(_settings(translator=_role("t-1"), validator=_role("judge-2")))
        gate.set()
        result = await task
        assert result["probe"]["stale"] is True
        assert {item["proficiency"] for item in result["languages"]} == {"unknown"}
    finally:
        gate.set()
        await service.shutdown()


# ----------------------------------------------------------------------
# Document identity: several doc keys on one content hash
# ----------------------------------------------------------------------


async def test_viewer_key_on_auto_discovered_content_keeps_all_language_targeting(harness):
    service = await harness.start(_settings(paused=True, languages=TWO_LANGUAGES))
    try:
        auto = DocumentSpec("final_answer:s1", "final_answer", "Answer", "document", (("content", SOURCE_TEXT),))
        await service._apply_discovery(DiscoveryResult(documents=(auto,)))
        doc_hash = service._doc_by_key["final_answer:s1"]
        assert service._doc_targets(service._docs[doc_hash], "fr")

        # The final-answer viewer registers the same text under its own key, in Spanish only.
        rows = await service.register_documents(
            "es", True, [_doc("final:a1:p1", kind="final_answer", title="Answer")]
        )
        assert (rows[0]["doc_key"], rows[0]["doc_hash"]) == ("final:a1:p1", doc_hash)
        doc = service._docs[doc_hash]
        assert set(doc.keys) == {"final_answer:s1", "final:a1:p1"}
        assert service._doc_targets(doc, "fr") and service._doc_targets(doc, "es")
        assert service._viewing(doc)
        assert sorted(service._waiting_jobs()) == [(doc_hash, "es"), (doc_hash, "fr")]
        languages = {row["key"]: row for row in service.status_snapshot()["languages"]}
        assert languages["fr"]["pending_units"] == 2

        # Targeting holds when the auto key is not the key shown for the document.
        await service.register_documents("es", True, [_doc("other:viewer:x", kind="proof", title="Answer")])
        assert service._docs[doc_hash].primary.doc_key == "other:viewer:x"
        assert service._docs[doc_hash].priority_class == 0
        assert service._doc_targets(service._docs[doc_hash], "fr")
        await service._apply_discovery(DiscoveryResult(documents=(auto,)))
        assert service._doc_by_key["other:viewer:x"] == doc_hash
        await service.register_documents("es", True, [_doc("other:viewer:x", "Moved away.", kind="proof")])

        # A rescan keeps both keys instead of flipping the record between them.
        await service._apply_discovery(DiscoveryResult(documents=(auto,)))
        assert set(service._docs[doc_hash].keys) == {"final_answer:s1", "final:a1:p1"}
        assert [record.doc_key for record in _keys_for_hash(translation_store, doc_hash)] == [
            "final:a1:p1",
            "final_answer:s1",
        ]

        # Both keys survive a restart.
        await service.shutdown()
        await service.start()
        assert set(service._docs[doc_hash].keys) == {"final_answer:s1", "final:a1:p1"}
    finally:
        await service.shutdown()


async def test_moving_one_key_keeps_the_document_of_the_remaining_key(harness):
    service = await harness.start(_settings(paused=True))
    try:
        auto = DocumentSpec("paper:s:p1", "paper", "P", "document", (("content", SOURCE_TEXT),))
        await service._apply_discovery(DiscoveryResult(documents=(auto,)))
        old_hash = service._doc_by_key["paper:s:p1"]
        await service.register_documents("es", True, [_doc("paper:viewer:p1")])
        assert service._doc_by_key["paper:viewer:p1"] == old_hash

        # The viewer key moves to new content; the auto key keeps the original document.
        rows = await service.register_documents("es", True, [_doc("paper:viewer:p1", "Revised paper body text.")])
        assert rows[0]["doc_hash"] != old_hash
        assert set(service._docs[old_hash].keys) == {"paper:s:p1"}
        assert set(service._docs[rows[0]["doc_hash"]].keys) == {"paper:viewer:p1"}
        assert [record.doc_key for record in _keys_for_hash(translation_store, old_hash)] == ["paper:s:p1"]
    finally:
        await service.shutdown()


async def test_same_proof_in_two_part_orders_has_one_hash(harness):
    service = await harness.start(_settings(paused=True))
    try:
        record = SimpleNamespace(
            proof_id="p1",
            run_id="r1",
            theorem_name="Lemma",
            theorem_statement="Every lattice is discrete.",
            formal_sketch="Use compactness.",
            novelty_reasoning="No reference states it.",
            novelty_group_claim="",
        )
        [discovered] = proof_document_specs([record], "autonomous")
        await service._apply_discovery(DiscoveryResult(documents=(discovered,)))
        auto_hash = service._doc_by_key[discovered.doc_key]

        # A viewer lists the same fields in another order under the same key.
        viewer = {
            "doc_key": discovered.doc_key,
            "kind": "proof",
            "title": "Lemma",
            "profile": "fields",
            "parts": [
                {"name": "theorem_statement", "text": "Every lattice is discrete."},
                {"name": "novelty_reasoning", "text": "No reference states it."},
                {"name": "formal_sketch", "text": "Use compactness."},
            ],
        }
        rows = await service.register_documents("es", True, [viewer])
        assert rows[0]["doc_hash"] == auto_hash
        assert list(service._docs) == [auto_hash]
        assert service_module.expected_doc_hash(
            "fields", [("b", "x"), ("a", "y")]
        ) == service_module.expected_doc_hash("fields", [("a", "y"), ("b", "x")])
        # A spec built directly in field order plans to the same hash too.
        unsorted = DocumentSpec(
            "proof:manual:r:p1",
            "proof",
            "Lemma",
            "fields",
            (
                ("theorem_statement", "Every lattice is discrete."),
                ("formal_sketch", "Use compactness."),
                ("novelty_reasoning", "No reference states it."),
            ),
        )
        await service._apply_discovery(DiscoveryResult(documents=(discovered, unsorted)))
        assert service._doc_by_key["proof:manual:r:p1"] == auto_hash
    finally:
        await service.shutdown()


async def test_two_requested_keys_with_identical_content_get_their_own_rows(harness):
    service = await harness.start(_settings())
    try:
        documents = [_doc("proof:autonomous:run1:p1"), _doc("proof:autonomous:run1:p2")]
        rows = await service.register_documents("es", True, documents)
        assert [row["doc_key"] for row in rows] == ["proof:autonomous:run1:p1", "proof:autonomous:run1:p2"]
        assert rows[0]["doc_hash"] == rows[1]["doc_hash"]
        assert [row["status"] for row in rows] == ["pending", "pending"]

        await wait_until(lambda: _complete(service, rows[0]["doc_hash"]))
        rows = await service.register_documents("es", True, documents)
        assert [(row["doc_key"], row["status"]) for row in rows] == [
            ("proof:autonomous:run1:p1", "complete"),
            ("proof:autonomous:run1:p2", "complete"),
        ]
        assert service.status_snapshot()["usage"]["finalized_parts"] == 1
    finally:
        await service.shutdown()


async def test_stored_progress_survives_a_restart(harness):
    service = await harness.start(_settings(paused=True))
    try:
        row = (await service.register_documents("es", True, [_doc(text=THREE_PARAGRAPHS)]))[0]
        doc = service._docs[row["doc_hash"]]
        translation_store.save_job_state(
            row["doc_hash"], "es",
            {**service_module.new_job_state(row["doc_hash"], "es"), "finals": {"0": "Alfa (es)."}},
        )
        translation_store.save_translation(
            SegmentTranslation(doc.views[0].source_hash, "es", "Alfa (es).", "", "translation_submitter",
                               "m", "v", 1, "2026-01-01T00:00:00+00:00")
        )
        await service.shutdown()
        await service.start()
        status = await lookup_status(service, row["doc_hash"])
        assert (status["status"], status["translated_units"], status["total_units"]) == ("pending", 1, 3)
        assert service.status_snapshot()["languages"][0]["translated_units"] == 1
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# User translation requests
# ----------------------------------------------------------------------


def _viewed_long_ago(service, doc_hash: str) -> None:
    for entry in service._docs[doc_hash].keys.values():
        entry.viewed_ts = time.time() - service_module.VIEWING_WINDOW_SECONDS - 60


async def test_a_requested_document_keeps_translating_after_its_viewer_closes(harness):
    service = await harness.start(_settings(paused=True))
    try:
        viewed = (await service.register_documents("es", True, [_doc("paper:old:viewed", "A viewed ring.")]))[0]
        asked = (
            await service.register_documents("es", True, [_doc("paper:old:asked", "A requested field.")], request=True)
        )[0]
        assert (viewed["requested"], viewed["automatic"]) == (False, False)
        assert (asked["requested"], asked["automatic"]) == (True, False)
        assert translation_store.load_requests() == {"paper:old:asked": {"es"}}
        _viewed_long_ago(service, viewed["doc_hash"])
        _viewed_long_ago(service, asked["doc_hash"])
        # Only the request still targets its document once both viewing windows ended.
        assert service._waiting_jobs() == [(asked["doc_hash"], "es")]
        await service.control("resume")
        await wait_until(lambda: _complete(service, asked["doc_hash"]))
        assert not _complete(service, viewed["doc_hash"])
        assert (await lookup_status(service, asked["doc_hash"]))["requested"] is True
    finally:
        await service.shutdown()


async def test_a_request_without_a_language_covers_every_language_and_survives_a_restart(harness):
    languages = [{"key": "es", "name": "Spanish"}, {"key": "fr", "name": "French"}]
    service = await harness.start(_settings(paused=True, languages=languages))
    try:
        row = (await service.register_documents(None, False, [_doc("paper:old:asked")], request=True))[0]
        assert translation_store.load_requests() == {"paper:old:asked": {"es", "fr"}}
        await service.shutdown()
        await service.start()
        assert set(service._waiting_jobs()) == {(row["doc_hash"], "es"), (row["doc_hash"], "fr")}
    finally:
        await service.shutdown()


async def test_requests_outlive_a_cleared_cache_and_status_rows_flag_current_run_documents(harness):
    service = await harness.start(_settings(paused=True))
    try:
        [current_hash] = await discover(service, _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"))
        current = (await service.lookup_documents("es", [current_hash]))[0]
        assert (current["requested"], current["automatic"]) == (False, True)
        await service.register_documents("es", False, [_doc("paper:old:asked")], request=True)
        await service.clear_cache()
        assert translation_store.load_requests() == {"paper:old:asked": {"es"}}
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# Current runs, viewed items, and Translate all
# ----------------------------------------------------------------------


OLDER_TEXT = "Alpha older paragraph text.\n\nBeta older paragraph text."
CURRENT_TEXT = "A current-run brainstorm about groups."


def _translate_all_events(harness) -> List[Dict[str, Any]]:
    return harness.events_named("translation_translate_all")


async def translate_all_on(service) -> None:
    """Turn Translate all on without an enumeration, so a test adds the older outputs itself."""
    await service._save_translate_all(True)


async def test_a_document_opened_from_an_older_run_translates_only_while_viewed(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()
    calls: List[Dict[str, Any]] = []

    async def translator(record):
        calls.append(record)
        if len(calls) == 1:
            await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        row = (await service.register_documents("es", True, [_doc("paper:old:p1", THREE_PARAGRAPHS)]))[0]
        doc_hash = row["doc_hash"]
        await wait_until(lambda: len(calls) == 1)
        # The viewer closed the item more than the viewing window ago.
        for entry in service._docs[doc_hash].keys.values():
            entry.viewed_ts = time.time() - service_module.VIEWING_WINDOW_SECONDS - 60
        gate.set()
        # The job stops at its attempt boundary and is not queued again.
        await wait_until(lambda: not service._jobs)
        await settle()
        assert not _complete(service, doc_hash)
        assert service._waiting_jobs() == []
        status = await lookup_status(service, doc_hash)
        assert (status["status"], status["queue_position"]) == ("pending", None)
        assert service.status_snapshot()["languages"][0]["pending_units"] == 0
        # Reopening it continues from the saved part.
        reopened = (await service.lookup_documents("es", [doc_hash], viewing=True))[0]
        assert reopened["queue_position"] in (0, 1)
        await wait_until(lambda: _complete(service, doc_hash))
        assert len(harness.events_named("translation_part_finalized")) == 1
    finally:
        gate.set()
        await service.shutdown()


async def test_a_lookup_without_viewing_does_not_target_an_untargeted_document(harness):
    service = await harness.start(_settings(paused=True))
    try:
        row = (await service.register_documents("es", False, [_doc("paper:old:p1")]))[0]
        assert row["queue_position"] is None
        assert (await lookup_status(service, row["doc_hash"]))["queue_position"] is None
        assert service._waiting_jobs() == []
        assert service.status_snapshot()["queue"] == []
    finally:
        await service.shutdown()


async def test_a_background_job_yields_to_a_waiting_current_run_job(harness, monkeypatch):
    # The translator connection runs one job at a time, so Translate all has no lane of its own.
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 1)
    service = await harness.start(_settings())
    gate = asyncio.Event()
    calls: List[Dict[str, Any]] = []

    async def translator(record):
        calls.append(record)
        if len(calls) == 1:
            await gate.wait()
        return _one_unit_parts(record)

    harness.fake.translator = translator
    try:
        await translate_all_on(service)
        [older_hash] = await add_older_outputs(service, _doc("paper:old:p1", THREE_PARAGRAPHS))
        await wait_until(lambda: len(calls) == 1)
        [current_hash] = await discover(service, _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"))
        await wait_until(lambda: service._jobs[(older_hash, "es")].yield_requested)
        queue = service.status_snapshot()["queue"]
        assert [(row["doc_key"], row["position"], row["background"]) for row in queue] == [
            ("paper:old:p1", 0, True),
            ("brainstorm:current:b1", 1, False),
        ]
        gate.set()
        await wait_until(lambda: _complete(service, current_hash) and _complete(service, older_hash))
        order = [_doc_of(call) for call in calls]
        # The background attempt in flight finishes, then the current-run document runs first.
        assert order[:4] == ["A", "B", "B", "B"]
        assert order[4:] == ["A"] * (len(order) - 4)
    finally:
        gate.set()
        await service.shutdown()


async def test_background_documents_rank_below_every_current_run_document(harness):
    service = await harness.start(_settings(paused=True))
    try:
        await translate_all_on(service)
        [older_hash] = await add_older_outputs(
            service, _doc("proof:library:s:p1", "An older proof statement.", kind="proof")
        )
        [current_hash] = await discover(service, _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"))
        # A proof ranks above a brainstorm by kind, but the current run ranks above any background document.
        assert service._waiting_jobs() == [(current_hash, "es"), (older_hash, "es")]
        # Viewing the background document moves it to the viewing tier.
        await service.register_documents(
            "es", True, [_doc("proof:library:s:p1", "An older proof statement.", kind="proof")]
        )
        assert service._waiting_jobs() == [(older_hash, "es"), (current_hash, "es")]
        [viewed_row] = [row for row in service.status_snapshot()["queue"] if row["doc_hash"] == older_hash]
        assert (viewed_row["viewing"], viewed_row["background"]) == (True, False)
    finally:
        await service.shutdown()


async def test_translate_all_translates_older_outputs_then_turns_itself_off(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    enumerations: List[tuple] = []
    older = _spec(_doc("paper:old:p1", OLDER_TEXT), "history_papers")

    async def enumerate_older(modes) -> DiscoveryResult:
        enumerations.append(set(modes))
        return DiscoveryResult(documents=(older,))

    harness.service._enumerate_backlog = enumerate_older
    service = await harness.start(_settings())
    try:
        status = await service.control("translate_all")
        assert status["translate_all"]["requested"] is True
        assert status["translate_all"]["state"] == "preparing"
        assert service_module.load_auto_translate_settings().translate_all is True
        await wait_until(lambda: not service.get_settings().translate_all)
        assert enumerations == [{"autonomous", "aggregator", "compiler"}]
        doc_hash = service._doc_by_key["paper:old:p1"]
        assert _complete(service, doc_hash)
        assert _stored_documents()["paper:old:p1"].origin == "backlog"
        events = _translate_all_events(harness)
        assert [event["state"] for event in events] == ["started", "finished"]
        assert events[-1]["message"] == "Translate all finished: 1 item in 1 language."
        await wait_until(lambda: service.status_snapshot()["translate_all"]["state"] == "off")
        snapshot = service.status_snapshot()
        assert snapshot["translate_all"]["requested"] is False
        assert service._backlog_keys == {}
        assert service_module.load_auto_translate_settings().translate_all is False
        recent = [item for item in snapshot["recent_activity"] if item["event_type"] == "translation_translate_all"]
        assert [item["message"] for item in recent] == [
            "Translate all finished: 1 item in 1 language.",
            "Translate all started.",
        ]
    finally:
        await service.shutdown()


async def test_translate_all_with_nothing_older_finishes_at_once(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    service = await harness.start(_settings())
    try:
        await service.control("translate_all")
        await wait_until(lambda: not service.get_settings().translate_all)
        assert _translate_all_events(harness)[-1]["message"] == "Nothing else to translate for the checked modes."
    finally:
        await service.shutdown()


async def test_translate_all_finishes_after_a_failed_source_and_names_it(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    enumerations: List[set] = []
    gate = asyncio.Event()
    older = _spec(_doc("final:old", OLDER_TEXT, kind="final_answer"), "history_final_answers")

    async def partly_failing(modes) -> DiscoveryResult:
        enumerations.append(set(modes))
        return DiscoveryResult(documents=(older,), failed_sources=("history_papers", "leanoj_library"))

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    harness.service._enumerate_backlog = partly_failing
    service = await harness.start(_settings())
    try:
        await service.control("translate_all")
        await wait_until(lambda: bool(service._jobs))
        # The enumeration has finished, so the state follows the remaining work.
        assert service.status_snapshot()["translate_all"]["state"] == "translating"
        gate.set()
        await wait_until(lambda: not service.get_settings().translate_all)
        assert _complete(service, service._doc_by_key["final:old"])
        assert _translate_all_events(harness)[-1]["message"] == (
            "Translate all finished: 1 item in 1 language. These older items could not be listed: "
            "Stage 2 Papers History, Proof Solver Completed Proof Works. Press Translate all again to retry them."
        )
        # A failed source is never listed again on its own; pressing Translate all again retries it.
        await settle()
        assert len(enumerations) == 1
        await service.control("translate_all")
        await wait_until(lambda: len(enumerations) == 2)
    finally:
        gate.set()
        await service.shutdown()


async def test_translate_all_finishes_when_every_source_failed(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    attempts: List[int] = []

    async def failing(_modes) -> DiscoveryResult:
        attempts.append(1)
        return DiscoveryResult(documents=(), failed_sources=("history_papers",))

    harness.service._enumerate_backlog = failing
    service = await harness.start(_settings())
    try:
        await service.control("translate_all")
        await wait_until(lambda: not service.get_settings().translate_all)
        await settle()
        assert attempts == [1]
        assert service.status_snapshot()["translate_all"]["state"] == "off"
        assert _translate_all_events(harness)[-1]["message"] == (
            "Nothing else to translate for the checked modes. These older items could not be listed: "
            "Stage 2 Papers History. Press Translate all again to retry them."
        )
    finally:
        await service.shutdown()


async def test_an_enumeration_that_raises_counts_every_checked_source_as_failed(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)

    async def broken(_modes) -> DiscoveryResult:
        raise RuntimeError("listing failed")

    harness.service._enumerate_backlog = broken
    service = await harness.start(_settings(modes=["compiler"]))
    try:
        await service.control("translate_all")
        await wait_until(lambda: not service.get_settings().translate_all)
        assert _translate_all_events(harness)[-1]["message"] == (
            "Nothing else to translate for the checked modes. These older items could not be listed: "
            "Manual Mathematical Proofs, Completed Proof Works, Single Paper Writer Live Paper. "
            "Press Translate all again to retry them."
        )
    finally:
        await service.shutdown()


async def test_the_status_keeps_a_finish_message_naming_every_mode_source_whole(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)

    async def broken(_modes) -> DiscoveryResult:
        raise RuntimeError("listing failed")

    harness.service._enumerate_backlog = broken
    service = await harness.start(_settings(modes=["autonomous", "aggregator", "compiler", "leanoj"]))
    try:
        await service.control("translate_all")
        await wait_until(lambda: not service.get_settings().translate_all)
        message = _translate_all_events(harness)[-1]["message"]
        assert len(message) > service_module._ACTIVITY_TEXT_CHARS
        assert message.endswith("Press Translate all again to retry them.")
        # After a reload only the status copy is left, so it keeps every source and the retry instruction.
        [finished] = [
            item
            for item in service.status_snapshot()["recent_activity"]
            if item["event_type"] == "translation_translate_all" and item["state"] == "finished"
        ]
        assert finished["message"] == message
    finally:
        await service.shutdown()


async def test_translate_all_status_counts_background_work(harness):
    service = await harness.start(_settings(paused=True))
    try:
        await translate_all_on(service)
        [older_hash] = await add_older_outputs(service, _doc("paper:old:p1", THREE_PARAGRAPHS))
        await discover(service, _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"))
        snapshot = service.status_snapshot()
        assert snapshot["translate_all"] == {
            "requested": True,
            "state": "waiting",
            "documents": 1,
            "complete_documents": 0,
            "translated_units": 0,
            "pending_units": 3,
            "message": "Translation is paused.",
        }
        [language] = snapshot["languages"]
        assert (language["pending_units"], language["pending_documents"]) == (1, 1)
        assert (language["background_pending_units"], language["background_pending_documents"]) == (3, 1)
        assert [(row["doc_hash"], row["background"]) for row in snapshot["queue"]][-1] == (older_hash, True)
        assert service.connectivity_summary()["pending_units"] == 4

        await service.control("resume")
        await wait_until(lambda: not service.get_settings().translate_all, timeout=10)
        assert service.status_snapshot()["translate_all"]["state"] == "off"
    finally:
        await service.shutdown()


async def test_status_messages_split_current_and_background_units(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await translate_all_on(service)
        await add_older_outputs(service, _doc("paper:old:p1", THREE_PARAGRAPHS))
        await wait_until(lambda: bool(service._jobs))
        snapshot = service.status_snapshot()
        assert snapshot["message"] == "Translate all: 1 older item(s) left in the background."
        assert snapshot["translate_all"]["state"] == "translating"
        await discover(service, _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"))
        snapshot = service.status_snapshot()
        assert snapshot["message"] == "Translating: 0% of current runs done; Translate all: 1 older item(s) left."
        assert snapshot["translate_all"]["message"].startswith("Translate all")
    finally:
        gate.set()
        await service.shutdown()


async def test_translate_all_is_rejected_while_translation_is_off(harness):
    service = await harness.start(_settings(enabled=False))
    try:
        with pytest.raises(ValueError, match="before using Translate all"):
            await service.control("translate_all")
        assert service.get_settings().translate_all is False
    finally:
        await service.shutdown()


async def test_cancel_stops_background_jobs_and_keeps_finished_parts(harness):
    service = await harness.start(_settings())
    gate = asyncio.Event()
    calls: List[Dict[str, Any]] = []

    async def translator(record):
        calls.append(record)
        if len(calls) == 4:
            await gate.wait()
        return _one_unit_parts(record)

    harness.fake.translator = translator
    try:
        await translate_all_on(service)
        [older_hash] = await add_older_outputs(service, _doc("paper:old:p1", THREE_PARAGRAPHS))
        await wait_until(lambda: len(calls) == 4)
        status = await service.control("cancel_translate_all")
        assert status["translate_all"]["requested"] is False
        assert service.get_settings().translate_all is False
        assert service_module.load_auto_translate_settings().translate_all is False
        assert service._backlog_keys == {}
        await wait_until(lambda: not service._jobs)
        gate.set()
        await settle()
        assert len(calls) == 4
        # The first part (unit 1) was finalized before the cancel and stays.
        assert service._final_counts[(older_hash, "es")] == 1
        assert _translate_all_events(harness)[-1]["message"] == "Translate all cancelled."
        assert service._waiting_jobs() == []
    finally:
        gate.set()
        await service.shutdown()


async def test_disable_turns_translate_all_off_and_pause_keeps_it(harness):
    service = await harness.start(_settings())
    try:
        await service.control("translate_all")
        await service.control("pause")
        assert service.get_settings().translate_all is True
        assert service.status_snapshot()["translate_all"]["state"] in ("preparing", "waiting")
        await service.control("disable")
        assert service.get_settings().translate_all is False
        assert service_module.load_auto_translate_settings().translate_all is False
    finally:
        await service.shutdown()


async def test_settings_saves_keep_translate_all_and_omitted_modes(harness):
    service = await harness.start(_settings(paused=True))
    try:
        await service.control("translate_all")
        await service.update_settings(_settings(modes=["compiler"]))
        assert service.get_settings().modes == ["compiler"]
        assert service.status_snapshot()["modes"] == ["compiler"]
        # A save without modes (a client that does not know the field) keeps them.
        await service.update_settings(_settings(batch=True))
        settings = service.get_settings()
        assert (settings.modes, settings.batch_translation, settings.translate_all) == (["compiler"], True, True)
        stored = service_module.load_auto_translate_settings()
        assert (stored.modes, stored.translate_all) == (["compiler"], True)
        # The modes request translate_all=False never turns it off.
        await service.update_settings(_settings(modes=["autonomous", "leanoj"], translate_all=False))
        assert service.get_settings().translate_all is True
        assert service.get_settings().modes == ["autonomous", "leanoj"]
    finally:
        await service.shutdown()


async def test_unchecking_a_mode_drops_its_current_run_documents(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    scans: List[set] = []
    aggregator = _spec(_doc("aggregator:results", "An accepted idea.", kind="aggregator_results"), "aggregator_results")
    paper = _spec(_doc("paper:s:p1", "Current paper text."), "autonomous_papers")

    async def discover_modes(modes) -> DiscoveryResult:
        scans.append(set(modes))
        documents = [spec for spec, mode in ((paper, "autonomous"), (aggregator, "aggregator")) if mode in modes]
        return DiscoveryResult(documents=tuple(documents))

    harness.service._discover = discover_modes
    service = await harness.start(_settings(paused=True))
    try:
        await wait_until(lambda: set(service._auto_keys) == {"paper:s:p1", "aggregator:results"})
        await service.update_settings(_settings(paused=True, modes=["autonomous"]))
        await wait_until(lambda: service._auto_keys == {"paper:s:p1": "autonomous_papers"})
        assert scans[-1] == {"autonomous"}
        aggregator_doc = service._docs[service._doc_by_key["aggregator:results"]]
        assert not service._doc_targets(aggregator_doc, "es")
        # The cached document and its finished translations stay.
        assert "aggregator:results" in _stored_documents()
    finally:
        await service.shutdown()


async def test_a_mode_change_enumerates_older_outputs_again(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    gate = asyncio.Event()
    enumerations: List[set] = []

    async def enumerate_older(modes) -> DiscoveryResult:
        enumerations.append(set(modes))
        await gate.wait()
        older = _spec(_doc(f"paper:old:{'-'.join(sorted(modes))}", OLDER_TEXT), "history_papers")
        return DiscoveryResult(documents=(older,))

    harness.service._enumerate_backlog = enumerate_older
    service = await harness.start(_settings(paused=True))
    try:
        await service.control("translate_all")
        await wait_until(lambda: len(enumerations) == 1)
        # Modes change while the enumeration runs: its result is stale and a new one starts.
        await service.update_settings(_settings(paused=True, modes=["leanoj"]))
        assert service._backlog_busy() and not service._backlog_enumerated
        gate.set()
        await wait_until(lambda: len(enumerations) == 2 and not service._backlog_busy())
        assert enumerations[-1] == {"leanoj"}
        assert set(service._backlog_keys) == {"paper:old:leanoj"}
        assert "paper:old:aggregator-autonomous-compiler" not in service._doc_by_key
    finally:
        gate.set()
        await service.shutdown()


async def test_clearing_the_cache_turns_translate_all_off(harness):
    service = await harness.start(_settings(paused=True))
    try:
        await translate_all_on(service)
        await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
        status = await service.clear_cache()
        assert status["translate_all"]["requested"] is False
        assert service._backlog_keys == {}
        assert service_module.load_auto_translate_settings().translate_all is False
        assert _translate_all_events(harness)[-1]["message"] == (
            "Translate all was turned off because stored translations were cleared."
        )
    finally:
        await service.shutdown()


async def test_after_a_restart_nothing_older_is_targeted_before_a_fresh_enumeration(harness, monkeypatch):
    service = await harness.start(_settings(paused=True))
    await translate_all_on(service)
    [older_hash] = await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
    await service.shutdown()

    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    gate = asyncio.Event()
    release = asyncio.Event()

    async def gated(_modes) -> DiscoveryResult:
        await gate.wait()
        return DiscoveryResult(documents=(_spec(_doc("paper:old:p1", OLDER_TEXT), "history_papers"),))

    async def translator(record):
        await release.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    service._enumerate_backlog = gated
    await service.start()
    try:
        await service.control("resume")
        assert service.get_settings().translate_all is True
        assert _stored_documents()["paper:old:p1"].origin == "backlog"
        doc = service._docs[older_hash]
        assert not service._doc_targets(doc, "es")
        assert service._waiting_jobs() == []
        assert service._jobs == {}
        assert service.status_snapshot()["translate_all"]["state"] == "preparing"
        gate.set()
        # The resumed worker starts the older output's job at once; the held translator keeps it running.
        await wait_until(lambda: service._backlog_enumerated and bool(harness.fake.translator_calls()))
        assert service._doc_tier(service._docs[older_hash], "es") == service_module.TIER_BACKGROUND
        assert list(service._jobs) == [(older_hash, "es")]
        assert service._waiting_jobs() == []
        [call] = harness.fake.translator_calls()
        assert "Alpha older paragraph text." in _source_units(call).values()
    finally:
        release.set()
        gate.set()
        await service.shutdown()


async def test_a_key_keeps_its_strongest_origin(harness):
    service = await harness.start(_settings(paused=True))
    try:
        await translate_all_on(service)
        await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
        await service.register_documents("es", True, [_doc("paper:old:p1", OLDER_TEXT)])
        assert _stored_documents()["paper:old:p1"].origin == "backlog"
        # A viewer key on the same content does not become the key shown before the backlog key.
        await service.register_documents("es", True, [_doc("paper:viewer:p1", OLDER_TEXT)])
        doc = service._docs[service._doc_by_key["paper:old:p1"]]
        assert doc.primary.doc_key == "paper:old:p1"
        await discover(service, _doc("paper:viewer:p1", OLDER_TEXT))
        assert _stored_documents()["paper:viewer:p1"].origin == "auto"
        assert service._docs[service._doc_by_key["paper:old:p1"]].primary.doc_key == "paper:viewer:p1"
    finally:
        await service.shutdown()


async def test_lifecycle_events_and_route_notes_request_an_enumeration(harness):
    service = await harness.start(_settings(paused=True))
    try:
        await service.control("translate_all")
        service._backlog_event.clear()
        service._scan_event.clear()
        service._observe_event("paper_completed", {})
        assert service._scan_event.is_set() and not service._backlog_event.is_set()
        for event in ("compiler_started", "leanoj_cleared", "system_reset", "auto_research_started"):
            service._backlog_event.clear()
            service._observe_event(event, {})
            assert service._backlog_event.is_set(), event
        service._backlog_event.clear()
        service._scan_event.clear()
        service.note_outputs_changed("aggregator_cleared")
        assert service._scan_event.is_set() and service._backlog_event.is_set()
        # Without Translate all only the current-run scan is requested.
        await service.control("cancel_translate_all")
        service._backlog_event.clear()
        service.note_outputs_changed("compiler_cleared")
        assert not service._backlog_event.is_set()
    finally:
        await service.shutdown()


# ----------------------------------------------------------------------
# Translate all's lane, stop rescans, and outputs moving to history
# ----------------------------------------------------------------------


SECOND_CURRENT_TEXT = "Another current-run brainstorm about rings."


def _gate_by_source(older_gate: asyncio.Event, current_gate: asyncio.Event):
    """A translator whose older documents wait on one gate and current-run documents on another."""

    async def translator(record):
        await (older_gate if "older" in _block(record, "SOURCE") else current_gate).wait()
        return _translate_ok(record)

    return translator


def _jobs_for(service, doc_hashes) -> int:
    return sum(1 for doc_hash, _language in service._jobs if doc_hash in doc_hashes)


async def test_translate_all_runs_one_background_job_beside_current_run_work(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 3)
    service = await harness.start(_settings())
    gate = asyncio.Event()
    running: List[tuple] = []
    current_hashes: set = set()

    async def translator(record):
        running.append((len(service._jobs), _jobs_for(service, current_hashes)))
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await translate_all_on(service)
        current_hashes.update(
            await discover(
                service,
                _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"),
                _doc("brainstorm:current:b2", SECOND_CURRENT_TEXT, kind="brainstorm"),
            )
        )
        older = await add_older_outputs(
            service, _doc("paper:old:p1", OLDER_TEXT), _doc("paper:old:p2", "Gamma older paper body.")
        )
        # The job limit stays one, and Translate all's lane runs one background job beside it.
        await wait_until(lambda: harness.fake.in_flight == 2)
        await settle()
        assert service.status_snapshot()["max_parallel_jobs"] == 1
        assert _jobs_for(service, current_hashes) == 1 and _jobs_for(service, older) == 1
        assert not any(job.yield_requested for job in service._jobs.values())
        assert service.status_snapshot()["translate_all"]["state"] == "translating"
        gate.set()
        await wait_until(lambda: all(_complete(service, doc_hash) for doc_hash in (*current_hashes, *older)))
        # Never more than one current-run job, or more than one job beyond the limit.
        assert max(total for total, _current in running) == 2
        assert max(current for _total, current in running) == 1
    finally:
        gate.set()
        await service.shutdown()


async def test_translate_all_waits_behind_current_runs_on_a_one_job_connection(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 1)
    service = await harness.start(_settings())
    gate = asyncio.Event()

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    try:
        await translate_all_on(service)
        [current_hash] = await discover(service, _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"))
        await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
        await wait_until(lambda: harness.fake.in_flight == 1)
        await settle()
        assert list(service._jobs) == [(current_hash, "es")]
        translate_all = service.status_snapshot()["translate_all"]
        assert (translate_all["state"], translate_all["message"]) == (
            "waiting",
            "Translate all waits for current-run and viewed translations: "
            "the translator connection runs one job at a time.",
        )
    finally:
        gate.set()
        await service.shutdown()


async def test_with_batch_translation_translate_all_keeps_one_of_the_connections_jobs(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 2)
    service = await harness.start(_settings(batch=True))
    older_gate = asyncio.Event()
    current_gate = asyncio.Event()
    harness.fake.translator = _gate_by_source(older_gate, current_gate)
    try:
        await translate_all_on(service)
        current_hashes = await discover(
            service,
            _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"),
            _doc("brainstorm:current:b2", SECOND_CURRENT_TEXT, kind="brainstorm"),
        )
        [older_hash] = await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
        # Two current-run documents wait, and the lane still runs the background job beside one of them.
        await wait_until(lambda: harness.fake.in_flight == 2)
        await settle()
        assert service.status_snapshot()["max_parallel_jobs"] == 2
        assert _jobs_for(service, current_hashes) == 1 and service._jobs[(older_hash, "es")].lane
        assert not any(job.yield_requested for job in service._jobs.values())
        assert service.status_snapshot()["translate_all"]["state"] == "translating"
        older_gate.set()
        current_gate.set()
        await wait_until(lambda: all(_complete(service, doc_hash) for doc_hash in (*current_hashes, older_hash)))
        assert harness.fake.max_in_flight == 2
    finally:
        older_gate.set()
        current_gate.set()
        await service.shutdown()


async def test_a_dropped_job_limit_yields_current_run_work_beyond_what_the_lane_leaves(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 3)
    service = await harness.start(_settings(batch=True))
    older_gate = asyncio.Event()
    current_gate = asyncio.Event()
    harness.fake.translator = _gate_by_source(older_gate, current_gate)
    try:
        await translate_all_on(service)
        current_hashes = await discover(
            service,
            _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"),
            _doc("brainstorm:current:b2", SECOND_CURRENT_TEXT, kind="brainstorm"),
        )
        [older_hash] = await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
        await wait_until(lambda: harness.fake.in_flight == 3)
        lane_job = service._jobs[(older_hash, "es")]
        assert lane_job.lane
        # The connection now runs two jobs: the lane keeps one, so one current-run job yields.
        monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 2)
        await service.update_settings(_settings(batch=True))
        await settle()
        assert service.status_snapshot()["max_parallel_jobs"] == 2
        current_jobs = [service._jobs[(doc_hash, "es")] for doc_hash in current_hashes]
        assert sum(job.yield_requested for job in current_jobs) == 1
        assert not lane_job.yield_requested
        older_gate.set()
        current_gate.set()
        await wait_until(lambda: all(_complete(service, doc_hash) for doc_hash in (*current_hashes, older_hash)))
    finally:
        older_gate.set()
        current_gate.set()
        await service.shutdown()


async def test_turning_translate_all_on_never_asks_running_current_run_work_to_yield(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 2)
    service = await harness.start(_settings(batch=True))
    older_gate = asyncio.Event()
    current_gate = asyncio.Event()
    harness.fake.translator = _gate_by_source(older_gate, current_gate)
    try:
        current_hashes = await discover(
            service,
            _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"),
            _doc("brainstorm:current:b2", SECOND_CURRENT_TEXT, kind="brainstorm"),
        )
        await wait_until(lambda: harness.fake.in_flight == 2)
        await translate_all_on(service)
        [older_hash] = await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
        await settle()
        # The lane waits for a free job on the connection instead of preempting current-run work.
        assert set(service._jobs) == {(doc_hash, "es") for doc_hash in current_hashes}
        assert not any(job.yield_requested for job in service._jobs.values())
        current_gate.set()
        older_gate.set()
        await wait_until(lambda: all(_complete(service, doc_hash) for doc_hash in (*current_hashes, older_hash)))
        assert harness.fake.max_in_flight == 2
    finally:
        older_gate.set()
        current_gate.set()
        await service.shutdown()


async def test_background_work_in_a_spare_normal_slot_yields_but_the_lane_job_runs_on(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 3)
    service = await harness.start(_settings(batch=True))
    older_gate = asyncio.Event()
    current_gate = asyncio.Event()
    harness.fake.translator = _gate_by_source(older_gate, current_gate)
    current_doc = _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm")
    try:
        await translate_all_on(service)
        [current_hash] = await discover(service, current_doc)
        older = await add_older_outputs(
            service, _doc("paper:old:p1", OLDER_TEXT), _doc("paper:old:p2", "Gamma older paper body.")
        )
        # The lane runs one background job, and the normal slot current-run work leaves free runs the other.
        await wait_until(lambda: harness.fake.in_flight == 3)
        older_jobs = [service._jobs[(doc_hash, "es")] for doc_hash in older]
        [lane_job] = [job for job in older_jobs if job.lane]
        [spare_job] = [job for job in older_jobs if not job.lane]
        [_, second_hash] = await discover(
            service, current_doc, _doc("brainstorm:current:b2", SECOND_CURRENT_TEXT, kind="brainstorm")
        )
        # A waiting current-run job takes the spare slot back at the background job's attempt boundary.
        await wait_until(lambda: spare_job.yield_requested)
        assert not lane_job.yield_requested
        assert not service._jobs[(current_hash, "es")].yield_requested
        older_gate.set()
        current_gate.set()
        await wait_until(lambda: all(_complete(service, doc_hash) for doc_hash in (current_hash, second_hash, *older)))
        assert harness.fake.max_in_flight == 3
    finally:
        older_gate.set()
        current_gate.set()
        await service.shutdown()


async def test_a_lane_job_that_is_viewed_keeps_the_lane_beside_current_run_work(harness, monkeypatch):
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 3)
    service = await harness.start(_settings())
    older_gate = asyncio.Event()
    current_gate = asyncio.Event()
    harness.fake.translator = _gate_by_source(older_gate, current_gate)
    older_docs = [_doc("paper:old:p1", OLDER_TEXT), _doc("paper:old:p2", "Gamma older paper body.")]
    try:
        await translate_all_on(service)
        [current_hash] = await discover(service, _doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"))
        older = await add_older_outputs(service, *older_docs)
        await wait_until(lambda: harness.fake.in_flight == 2)
        [lane_doc] = [doc for doc, doc_hash in zip(older_docs, older) if (doc_hash, "es") in service._jobs]
        # The user opens the older paper the lane is translating: it keeps the lane, so the
        # current-run job keeps its slot and no other background job starts.
        await service.register_documents("es", True, [lane_doc])
        await settle()
        assert len(service._jobs) == 2
        assert not service._jobs[(current_hash, "es")].yield_requested
        older_gate.set()
        current_gate.set()
        await wait_until(lambda: all(_complete(service, doc_hash) for doc_hash in (current_hash, *older)))
        assert harness.fake.max_in_flight == 2
    finally:
        older_gate.set()
        current_gate.set()
        await service.shutdown()


async def test_translate_all_counts_lm_studio_copies_for_its_lane(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 5)
    loaded = ["local-model"]
    lookups: List[int] = []

    async def fake_loaded():
        lookups.append(1)
        return list(loaded)

    monkeypatch.setattr(lm_studio_client, "get_loaded_models", fake_loaded)
    current = _spec(_doc("brainstorm:current:b1", CURRENT_TEXT, kind="brainstorm"), "autonomous_brainstorms")
    older = _spec(_doc("paper:old:p1", OLDER_TEXT), "history_papers")

    async def discover_current(_modes) -> DiscoveryResult:
        return DiscoveryResult(documents=(current,))

    async def enumerate_older(_modes) -> DiscoveryResult:
        return DiscoveryResult(documents=(older,))

    harness.service._discover = discover_current
    harness.service._enumerate_backlog = enumerate_older
    gate = asyncio.Event()

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    local = _role("local-model", provider="lm_studio")
    service = await harness.start(_settings(translator=local, validator=local))
    try:
        await wait_until(lambda: harness.fake.in_flight == 1)
        # Without batch translation the copies are counted once Translate all starts.
        assert lookups == []
        await service.control("translate_all")
        await wait_until(lambda: bool(lookups) and "paper:old:p1" in service._backlog_keys)
        await settle()
        # One loaded copy: the background job waits.
        assert len(service._jobs) == 1 and harness.fake.in_flight == 1
        loaded[:] = ["local-model:1", "local-model:2"]
        await service.update_settings(_settings(translator=local, validator=local))
        await wait_until(lambda: harness.fake.in_flight == 2)
        assert service.status_snapshot()["max_parallel_jobs"] == 1
    finally:
        gate.set()
        await service.shutdown()


async def test_translate_all_says_why_it_waits_and_reports_each_state_it_defines(harness):
    service = await harness.start(_settings(paused=True))
    gate = asyncio.Event()

    async def translator(record):
        await gate.wait()
        return _translate_ok(record)

    harness.fake.translator = translator
    seen: set = set()

    def translate_all() -> tuple:
        block = service.status_snapshot()["translate_all"]
        seen.add(block["state"])
        return block["state"], block["message"]

    try:
        assert translate_all() == ("off", "")
        await translate_all_on(service)
        assert translate_all() == ("waiting", "Translation is paused.")
        await service.control("resume")
        assert translate_all() == ("preparing", "Finding older items to translate.")
        await service.update_settings(_settings(languages=[]))
        assert translate_all() == (
            "waiting",
            "Translate all waits until translation is set up: Select at least one translation language.",
        )
        await service.update_settings(_settings())
        service._credit_waiters = 1
        assert translate_all() == ("waiting", "Translate all waits for provider credits to be reset.")
        service._credit_waiters = 0
        await service._enter_repair(
            reason="test_repair", message="Fix the translator.", guidance="", role_id="translation_submitter"
        )
        assert translate_all() == ("waiting", "Translate all waits until the translation repair is resolved.")
        await service.control("retry")
        await add_older_outputs(service, _doc("paper:old:p1", OLDER_TEXT))
        await wait_until(lambda: bool(service._jobs))
        assert translate_all() == ("translating", "Translate all: 1 older item(s) left in the background.")
        assert seen == set(service_module.TRANSLATE_ALL_STATES)
    finally:
        gate.set()
        await service.shutdown()


@pytest.mark.parametrize(
    ("event", "coordinator"),
    [
        ("compiler_stopped", "backend/compiler/core/compiler_coordinator.py"),
        ("compiler_error", "backend/compiler/core/compiler_coordinator.py"),
        ("auto_research_stopped", "backend/autonomous/core/autonomous_coordinator.py"),
        ("system_stopped", "backend/aggregator/core/coordinator.py"),
        ("leanoj_stopped", "backend/leanoj/core/leanoj_coordinator.py"),
    ],
)
async def test_a_run_stopping_requests_a_rescan_and_an_enumeration(harness, event, coordinator):
    # The coordinator broadcasts this event when its run stops.
    source = (Path(__file__).resolve().parents[2] / coordinator).read_text(encoding="utf-8")
    assert f'"{event}"' in source
    service = await harness.start(_settings(paused=True))
    try:
        await service.control("translate_all")
        service._scan_event.clear()
        service._backlog_event.clear()
        service._observe_event(event, {})
        assert service._scan_event.is_set() and service._backlog_event.is_set()
    finally:
        await service.shutdown()


async def test_the_live_paper_translates_once_the_compiler_stops(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    writing = {"value": True}

    async def fake_state() -> CurrentRunState:
        return CurrentRunState(
            autonomous_session="",
            aggregator_live=False,
            compiler_live=True,
            compiler_writing=writing["value"],
            leanoj_session="",
        )

    async def live_paper() -> str:
        return "1. Introduction\n\nThe paper the compiler wrote."

    async def no_proofs(_state) -> List[DocumentSpec]:
        return []

    monkeypatch.setattr(discovery, "current_run_state", fake_state)
    monkeypatch.setattr(discovery, "read_compiler_live_paper", live_paper)
    monkeypatch.setattr(discovery, "_scan_compiler_proofs", no_proofs)
    scans: List[DiscoveryResult] = []

    async def scan(modes) -> DiscoveryResult:
        result = await discovery.discover_current_outputs(modes)
        scans.append(result)
        return result

    harness.service._discover = scan
    service = await harness.start(_settings(paused=True, modes=["compiler"]))
    try:
        await wait_until(lambda: len(scans) == 1)
        await settle()
        # Discovery leaves the Live Paper out while the compiler writes it.
        assert "compiler:live-paper" not in service._auto_keys
        writing["value"] = False
        service._observe_event("compiler_stopped", {})
        await wait_until(lambda: "compiler:live-paper" in service._doc_by_key)
        doc = service._docs[service._doc_by_key["compiler:live-paper"]]
        assert service._doc_tier(doc, "es") == service_module.TIER_CURRENT
    finally:
        await service.shutdown()


async def test_an_output_moving_from_the_current_run_to_history_stays_targeted(harness, monkeypatch):
    monkeypatch.setattr(service_module, "SCAN_DEBOUNCE_SECONDS", 0.0)
    session = {"current": "s1"}
    scan_gate = asyncio.Event()
    scan_gate.set()
    paper_text = "Alpha paper of the first session."

    async def discover_current(_modes) -> DiscoveryResult:
        await scan_gate.wait()
        live = (_spec(_doc("paper:s1:p1", paper_text), "autonomous_papers"),)
        return DiscoveryResult(documents=live if session["current"] == "s1" else ())

    async def fake_state() -> CurrentRunState:
        return CurrentRunState(
            autonomous_session=session["current"],
            aggregator_live=False,
            compiler_live=False,
            compiler_writing=False,
            leanoj_session="",
        )

    async def history_papers(scope) -> List[DocumentSpec]:
        # An older session keeps Translate all busy; s1 is listed once it is no longer current.
        papers = [("s0", "p0", "Beta paper of an older session.")]
        if scope.state.autonomous_session != "s1":
            papers.append(("s1", "p1", paper_text))
        return [history._paper_spec(sid, pid, {"content": body, "title": pid}) for sid, pid, body in papers]

    async def nothing(_scope) -> List[DocumentSpec]:
        return []

    monkeypatch.setattr(history, "current_run_state", fake_state)
    for _modes, _source, name in history._HISTORY_SOURCES:
        monkeypatch.setattr(history, name, history_papers if name == "_enumerate_history_papers" else nothing)
    harness.service._discover = discover_current
    harness.service._enumerate_backlog = history.enumerate_displayed_outputs
    service = await harness.start(_settings(paused=True, modes=["autonomous"]))
    try:
        await wait_until(lambda: "paper:s1:p1" in service._auto_keys)
        await service.control("translate_all")
        await wait_until(lambda: "paper:s0:p0" in service._backlog_keys and not service._backlog_busy())
        # A new session starts: the enumeration runs before the scan has dropped s1's paper.
        session["current"] = "s2"
        scan_gate.clear()
        service._observe_event("auto_research_started", {})
        await wait_until(lambda: "paper:s1:p1" in service._backlog_keys and not service._backlog_busy())
        scan_gate.set()
        await wait_until(lambda: "paper:s1:p1" not in service._auto_keys)
        doc = service._docs[service._doc_by_key["paper:s1:p1"]]
        assert service._doc_tier(doc, "es") == service_module.TIER_BACKGROUND
        assert service.status_snapshot()["translate_all"]["documents"] == 2
    finally:
        scan_gate.set()
        await service.shutdown()

async def test_note_outputs_changed_never_raises_before_start():
    service = service_module.TranslationService()
    service.note_outputs_changed("autonomous_cleared")
