"""Auto Top-p Translate REST routes: typed contract, validation, and safe errors.

Each test mounts the translation router on a bare FastAPI app (shared
middleware authentication is covered elsewhere), swaps in a fresh
TranslationService that is never started, binds a private runtime root, and
blocks model calls.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, get_args

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.routes import translation as translation_routes
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import bind_runtime_roots, system_config
from backend.shared.runtime_settings import RuntimeSettingsError
from backend.shared.translation import service as service_module
from backend.shared.translation.discovery import DOCUMENT_KINDS
from backend.shared.translation.languages import PINNED_LANGUAGE_KEYS, custom_language_key
from backend.shared.translation.settings import GENERIC_MODE_PROVIDER_ERROR

TEXT = "First paragraph of the paper.\n\nSecond paragraph with more words."
EXPECTED_PATHS = {
    ("/api/translation/status", "get"),
    ("/api/translation/settings", "get"),
    ("/api/translation/settings", "put"),
    ("/api/translation/control", "post"),
    ("/api/translation/languages", "get"),
    ("/api/translation/languages/probe", "post"),
    ("/api/translation/documents", "post"),
    ("/api/translation/documents/lookup", "post"),
    ("/api/translation/documents/progress", "post"),
    ("/api/translation/cache", "delete"),
}
STATUS_KEYS = {
    "success",
    "enabled",
    "paused",
    "batch_translation",
    "modes",
    "max_parallel_jobs",
    "state",
    "status_label",
    "message",
    "generation",
    "languages",
    "queue",
    "active",
    "repair",
    "translate_all",
    "usage",
    "recent_activity",
}
SETTINGS_KEYS = {
    "enabled", "paused", "translate_all", "batch_translation", "modes", "languages", "translator", "validator",
}
TRANSLATE_ALL_KEYS = {
    "requested", "state", "documents", "complete_documents", "translated_units", "pending_units", "message",
}
USAGE_KEYS = {
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "accepted_attempts",
    "rejected_attempts",
    "finalized_parts",
    "check_failures",
}
LANGUAGE_PROGRESS_KEYS = {
    "key", "name", "custom", "translated_units", "pending_units", "pending_documents", "complete_documents",
    "background_pending_units", "background_pending_documents",
}
QUEUE_ITEM_KEYS = {
    "doc_hash", "doc_key", "kind", "title", "language_key", "position", "viewing", "background",
    "translated_units", "total_units",
}
ACTIVE_ITEM_KEYS = {
    "doc_hash", "doc_title", "language_key", "part_start", "part_end", "attempt", "accepted_count", "keep_streak",
}
DOCUMENT_STATUS_KEYS = {
    "doc_key", "doc_hash", "language", "status", "translated_units", "total_units", "queue_position", "parts",
    "requested", "automatic",
}


@pytest.fixture
def runtime_root(tmp_path):
    original_data = system_config.data_dir
    original_logs = system_config.logs_dir
    bind_runtime_roots(tmp_path / "data", tmp_path / "logs")
    yield tmp_path / "data"
    bind_runtime_roots(original_data, original_logs)


@pytest_asyncio.fixture
async def service(runtime_root, monkeypatch):
    fresh = service_module.TranslationService()

    async def no_model_calls(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("translation routes must not call models in these tests")

    monkeypatch.setattr(api_client_manager, "generate_completion", no_model_calls)
    monkeypatch.setattr(translation_routes, "translation_service", fresh)
    yield fresh
    await fresh.shutdown()


@pytest_asyncio.fixture
async def client(service):
    app = FastAPI()
    app.include_router(translation_routes.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http


def _role(model_id: str = "model-a", provider: str = "openrouter", **overrides: Any) -> Dict[str, Any]:
    payload = {"provider": provider, "model_id": model_id, "context_window": 32000, "max_output_tokens": 4000}
    payload.update(overrides)
    return payload


def _settings(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "enabled": True,
        "paused": False,
        "languages": [{"key": "es", "name": "Spanish"}],
        "translator": _role("translator-a"),
        "validator": _role("validator-model"),
    }
    payload.update(overrides)
    return payload


def _document(doc_key: str = "live_paper:test", text: str = TEXT, **overrides: Any) -> Dict[str, Any]:
    payload = {
        "doc_key": doc_key,
        "kind": "live_paper",
        "title": "Live paper",
        "profile": "document",
        "parts": [{"name": "content", "text": text}],
    }
    payload.update(overrides)
    return payload


async def _enable(client: AsyncClient, **overrides: Any) -> Dict[str, Any]:
    """Save settings, then enable through the control route unless ``enabled=False``.

    A settings save never changes enabled/paused, so enabling always goes
    through ``POST /control``.
    """
    response = await client.put("/api/translation/settings", json=_settings(**overrides))
    assert response.status_code == 200, response.text
    payload = response.json()
    if overrides.get("enabled", True):
        enabled = await client.post("/api/translation/control", json={"action": "enable"})
        assert enabled.status_code == 200, enabled.text
        payload["status"] = enabled.json()["status"]
    return payload


# ----------------------------------------------------------------------
# Contract and OpenAPI
# ----------------------------------------------------------------------


def test_route_enums_match_the_service_contract() -> None:
    assert set(get_args(translation_routes.DocumentKind)) == set(DOCUMENT_KINDS)
    assert set(get_args(translation_routes.ControlAction)) == set(service_module.CONTROL_ACTIONS)
    assert set(get_args(translation_routes.TranslationState)) == set(service_module.STATUS_LABELS)
    assert set(get_args(translation_routes.TranslationStatusLabel)) == set(service_module.STATUS_LABELS.values())
    assert set(get_args(translation_routes.TranslateAllState)) == set(service_module.TRANSLATE_ALL_STATES)
    assert {"translate_all", "cancel_translate_all"} <= set(service_module.CONTROL_ACTIONS)


def test_openapi_exposes_typed_translation_routes() -> None:
    app = FastAPI()
    app.include_router(translation_routes.router)
    openapi = app.openapi()
    paths = openapi["paths"]
    assert {(path, method) for path, methods in paths.items() for method in methods} == EXPECTED_PATHS

    def response_ref(path: str, method: str) -> str:
        return paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]

    def request_ref(path: str, method: str) -> str:
        return paths[path][method]["requestBody"]["content"]["application/json"]["schema"]["$ref"]

    schemas = openapi["components"]["schemas"]
    assert response_ref("/api/translation/status", "get").endswith("/TranslationStatusResponse")
    assert set(schemas["TranslationStatusResponse"]["properties"]) == STATUS_KEYS
    assert response_ref("/api/translation/languages", "get").endswith("/TranslationLanguageCatalogResponse")
    assert response_ref("/api/translation/languages/probe", "post").endswith("/TranslationLanguageCatalogResponse")
    assert response_ref("/api/translation/documents", "post").endswith("/TranslationDocumentsResponse")
    assert response_ref("/api/translation/documents/lookup", "post").endswith("/TranslationDocumentsResponse")
    assert response_ref("/api/translation/documents/progress", "post").endswith(
        "/TranslationDocumentProgressResponse"
    )
    progress_request = schemas[request_ref("/api/translation/documents/progress", "post").rsplit("/", 1)[-1]]
    assert set(progress_request["properties"]) == {"documents"}
    assert progress_request["additionalProperties"] is False
    assert progress_request["properties"]["documents"]["maxItems"] == 200
    assert set(schemas["TranslationDocumentProgress"]["properties"]) == {"doc_key", "doc_hash", "languages"}
    assert set(schemas["TranslationDocumentLanguageProgress"]["properties"]["status"]["enum"]) == {
        "complete", "pending", "unknown", "disabled",
    }
    assert response_ref("/api/translation/cache", "delete").endswith("/TranslationStatusEnvelope")
    assert response_ref("/api/translation/control", "post").endswith("/TranslationStatusEnvelope")
    assert "AutoTranslateSettings" in request_ref("/api/translation/settings", "put")
    settings_schema = schemas["AutoTranslateSettings"]
    assert set(settings_schema["properties"]) == SETTINGS_KEYS
    assert settings_schema["additionalProperties"] is False
    assert settings_schema["properties"]["batch_translation"]["default"] is False
    assert settings_schema["properties"]["translate_all"]["default"] is False
    modes = settings_schema["properties"]["modes"]
    assert set(modes["items"]["enum"]) == {"autonomous", "aggregator", "compiler", "leanoj"}
    assert "modes" not in settings_schema.get("required", [])
    for role in ("translator", "validator"):
        role_refs = {item.get("$ref", "") for item in settings_schema["properties"][role]["anyOf"]}
        assert any(ref.endswith("/TranslationRoleConfig") for ref in role_refs)
    status_schema = schemas["TranslationStatusResponse"]
    assert status_schema["properties"]["queue"]["maxItems"] == service_module.STATUS_QUEUE_LIMIT
    assert status_schema["properties"]["max_parallel_jobs"]["minimum"] == 1
    assert set(status_schema["properties"]["modes"]["items"]["enum"]) == {"autonomous", "aggregator", "compiler", "leanoj"}
    assert set(schemas["TranslationLanguageProgress"]["properties"]) == LANGUAGE_PROGRESS_KEYS
    assert set(schemas["TranslationQueueItem"]["properties"]) == QUEUE_ITEM_KEYS
    assert set(schemas["TranslationTranslateAllStatus"]["properties"]) == TRANSLATE_ALL_KEYS
    assert set(schemas["TranslationTranslateAllStatus"]["properties"]["state"]["enum"]) == set(
        service_module.TRANSLATE_ALL_STATES
    )
    assert status_schema["properties"]["translate_all"]["$ref"].endswith("/TranslationTranslateAllStatus")
    assert schemas["TranslationQueueItem"]["properties"]["position"]["minimum"] == 0
    assert set(schemas["TranslationActiveItem"]["properties"]) == ACTIVE_ITEM_KEYS
    assert set(schemas["TranslationUsage"]["properties"]) == USAGE_KEYS
    assert set(schemas["TranslationProbeModel"]["properties"]["role"]["enum"]) == {"translator", "validator"}

    documents_request = schemas[request_ref("/api/translation/documents", "post").rsplit("/", 1)[-1]]
    assert documents_request["additionalProperties"] is False
    assert documents_request["properties"]["documents"]["minItems"] == 1
    assert documents_request["properties"]["documents"]["maxItems"] == 200
    document_input = schemas["TranslationDocumentInput"]
    assert document_input["properties"]["doc_key"]["maxLength"] == 300
    assert document_input["properties"]["title"]["maxLength"] == 300
    assert set(document_input["properties"]["kind"]["enum"]) == set(DOCUMENT_KINDS)
    assert schemas["TranslationDocumentPart"]["properties"]["name"]["maxLength"] == 64
    lookup_request = schemas[request_ref("/api/translation/documents/lookup", "post").rsplit("/", 1)[-1]]
    assert set(lookup_request["properties"]) == {"language", "doc_hashes", "viewing"}
    assert lookup_request["additionalProperties"] is False
    assert lookup_request["properties"]["doc_hashes"]["maxItems"] == 500
    assert lookup_request["properties"]["viewing"]["type"] == "boolean"
    assert lookup_request["properties"]["viewing"]["default"] is False
    assert lookup_request["required"] == ["language", "doc_hashes"]
    control_request = schemas[request_ref("/api/translation/control", "post").rsplit("/", 1)[-1]]
    assert set(control_request["properties"]["action"]["enum"]) == set(service_module.CONTROL_ACTIONS)
    document_status = schemas["TranslationDocumentStatus"]
    assert set(document_status["properties"]) == DOCUMENT_STATUS_KEYS
    assert set(document_status["properties"]["status"]["enum"]) == {
        "complete", "pending", "empty", "unavailable", "disabled", "unknown",
    }
    queue_position = {item.get("type") for item in document_status["properties"]["queue_position"]["anyOf"]}
    assert queue_position == {"integer", "null"}
    cache_parameters = {parameter["name"]: parameter for parameter in paths["/api/translation/cache"]["delete"]["parameters"]}
    assert cache_parameters["confirm"]["in"] == "query"


def test_browser_smoke_fixture_mirrors_the_status_contract_once() -> None:
    fixture = (Path(__file__).resolve().parents[1] / "workflow_browser_smoke" / "fixtures.js").read_text(
        encoding="utf-8"
    )
    block = re.search(r"const DISABLED_TRANSLATION_STATUS = \{\n(.*?)\n\};", fixture, re.S)
    assert block, "fixtures.js has no DISABLED_TRANSLATION_STATUS"
    keys = re.findall(r"^  (\w+):", block.group(1), re.M)
    assert len(keys) == len(set(keys)), keys
    assert set(keys) == STATUS_KEYS


def test_full_app_registers_translation_routes() -> None:
    from backend.api.main import app as full_app

    registered = {
        (path, method)
        for path, methods in full_app.openapi()["paths"].items()
        for method in methods
        if path.startswith("/api/translation")
    }
    assert registered == EXPECTED_PATHS


# ----------------------------------------------------------------------
# Status, settings, control
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_is_in_memory_and_not_cacheable(client, service, monkeypatch) -> None:
    def no_disk(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("status must not touch the translation store")

    for name in (
        "load_documents",
        "get_translations",
        "translated_hashes_by_language",
        "get_job_state",
        "load_job_keys",
    ):
        monkeypatch.setattr(service_module.translation_store, name, no_disk)
    response = await client.get("/api/translation/status")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert set(payload) == STATUS_KEYS
    assert payload["success"] is True
    assert payload["state"] == "disabled"
    assert payload["status_label"] == "disabled"
    assert payload["repair"] is None
    assert payload["batch_translation"] is False
    assert payload["modes"] == ["autonomous", "aggregator", "compiler"]
    assert payload["max_parallel_jobs"] == 1
    assert payload["queue"] == [] and payload["active"] == []
    assert set(payload["usage"]) == USAGE_KEYS


@pytest.mark.asyncio
async def test_settings_round_trip_normalizes_and_persists(client, service, runtime_root) -> None:
    initial = await client.get("/api/translation/settings")
    assert initial.status_code == 200
    assert initial.json() == {"success": True, "settings": service_module.AutoTranslateSettings().model_dump(mode="json")}

    payload = await _enable(
        client,
        enabled=False,
        languages=[
            {"key": "es", "name": "anything"},
            {"key": "custom-pending", "name": "Klingon", "custom": True},
            {"key": "es", "name": "Spanish"},
        ],
    )
    languages = payload["settings"]["languages"]
    assert [language["key"] for language in languages] == ["es", custom_language_key("Klingon")]
    assert languages[0]["name"] == "Spanish"
    assert languages[1]["custom"] is True
    assert payload["status"]["state"] == "disabled"
    assert set(payload["status"]) == STATUS_KEYS
    assert set(payload["settings"]) == SETTINGS_KEYS
    assert payload["settings"]["modes"] == ["autonomous", "aggregator", "compiler"]
    assert payload["settings"]["translate_all"] is False
    assert payload["settings"]["translator"]["model_id"] == "translator-a"
    assert payload["settings"]["validator"]["model_id"] == "validator-model"
    assert payload["status"]["languages"] == [
        {
            "key": "es",
            "name": "Spanish",
            "custom": False,
            "translated_units": 0,
            "pending_units": 0,
            "pending_documents": 0,
            "complete_documents": 0,
            "background_pending_units": 0,
            "background_pending_documents": 0,
        },
        {
            "key": custom_language_key("Klingon"),
            "name": "Klingon",
            "custom": True,
            "translated_units": 0,
            "pending_units": 0,
            "pending_documents": 0,
            "complete_documents": 0,
            "background_pending_units": 0,
            "background_pending_documents": 0,
        },
    ]
    stored = json.loads((runtime_root / "runtime_settings.json").read_text(encoding="utf-8"))
    assert [language["key"] for language in stored["auto_translate"]["languages"]] == ["es", custom_language_key("Klingon")]
    assert stored["auto_translate"]["translator"]["model_id"] == "translator-a"
    assert "submitters" not in stored["auto_translate"]
    assert (await client.get("/api/translation/settings")).json()["settings"] == payload["settings"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"languages": [{"key": "xx-unknown", "name": "Nowhere"}]}, "Unknown language key."),
        ({"languages": [{"key": "custom-pending", "name": "1234", "custom": True}]}, "letter"),
    ],
)
async def test_settings_normalization_errors_are_422_with_safe_messages(client, service, runtime_root, overrides, expected) -> None:
    response = await client.put("/api/translation/settings", json=_settings(**overrides))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str) and expected in detail
    assert not (runtime_root / "runtime_settings.json").exists()


@pytest.mark.asyncio
async def test_settings_schema_errors_never_echo_submitted_values(client, service) -> None:
    secret_like = "sk-" + "Ab3" * 16
    body = _settings(unexpected_field=secret_like)
    body["translator"] = _role("translator-a", context_window=100, max_output_tokens=500)
    response = await client.put("/api/translation/settings", json=body)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    assert all(set(item) == {"loc", "msg", "type"} for item in detail)
    assert secret_like not in response.text
    assert any("unexpected_field" in item["loc"] for item in detail)
    assert any("smaller than the context window" in item["msg"] for item in detail)


@pytest.mark.asyncio
async def test_v1_submitters_list_is_rejected_without_echo(client, service, runtime_root) -> None:
    body = _settings()
    body["submitters"] = [_role("translator-secret-model-name")]
    response = await client.put("/api/translation/settings", json=body)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("submitters" in item["loc"] for item in detail)
    assert "translator-secret-model-name" not in response.text
    assert not (runtime_root / "runtime_settings.json").exists()


@pytest.mark.asyncio
async def test_settings_save_applies_batch_translation_and_reports_parallel_jobs(client, service, monkeypatch) -> None:
    monkeypatch.setattr(system_config, "max_model_concurrency_per_model", 3)
    single = await _enable(client)
    assert single["settings"]["batch_translation"] is False
    assert (single["status"]["batch_translation"], single["status"]["max_parallel_jobs"]) == (False, 1)

    batched = await client.put("/api/translation/settings", json=_settings(batch_translation=True, enabled=False))
    assert batched.status_code == 200, batched.text
    payload = batched.json()
    # batch_translation applies on save while enabled keeps the control-set value.
    assert (payload["settings"]["batch_translation"], payload["settings"]["enabled"]) == (True, True)
    assert (payload["status"]["batch_translation"], payload["status"]["max_parallel_jobs"]) == (True, 3)
    status = (await client.get("/api/translation/status")).json()
    assert (status["batch_translation"], status["max_parallel_jobs"]) == (True, 3)

    off = await client.put("/api/translation/settings", json=_settings(batch_translation=False))
    assert (off.json()["status"]["batch_translation"], off.json()["status"]["max_parallel_jobs"]) == (False, 1)


@pytest.mark.asyncio
async def test_generic_mode_accepts_only_openrouter_roles(client, service, monkeypatch) -> None:
    monkeypatch.setattr(system_config, "generic_mode", True)
    rejected = await client.put(
        "/api/translation/settings",
        json=_settings(translator=_role("local-model", provider="lm_studio")),
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"] == GENERIC_MODE_PROVIDER_ERROR

    accepted = await _enable(client, validator=_role("validator-model", lm_studio_fallback_id="local-fallback"))
    assert accepted["settings"]["validator"]["lm_studio_fallback_id"] is None


@pytest.mark.asyncio
async def test_settings_save_failure_is_a_generic_500(client, service, monkeypatch) -> None:
    def failing_save(_settings: Any) -> None:
        raise RuntimeSettingsError("could not write C:/private/path/runtime_settings.json")

    monkeypatch.setattr(service_module, "save_auto_translate_settings", failing_save)
    response = await client.put("/api/translation/settings", json=_settings())
    assert response.status_code == 500
    assert response.json() == {"detail": "Foreign Language Translation w/ Top-p could not save its settings."}
    assert "private" not in response.text


@pytest.mark.asyncio
async def test_settings_save_keeps_the_current_enabled_and_paused_state(client, service) -> None:
    # A save carrying a stale enabled/paused value never overrides control actions.
    saved = await client.put("/api/translation/settings", json=_settings(enabled=True, paused=True))
    assert saved.status_code == 200
    assert saved.json()["settings"]["enabled"] is False
    assert saved.json()["settings"]["paused"] is False
    assert saved.json()["status"]["state"] == "disabled"

    assert (await client.post("/api/translation/control", json={"action": "enable"})).status_code == 200
    assert (await client.post("/api/translation/control", json={"action": "pause"})).status_code == 200
    stale = await client.put(
        "/api/translation/settings",
        json=_settings(enabled=False, paused=False, languages=[{"key": "fr", "name": "French"}]),
    )
    assert stale.status_code == 200
    settings = stale.json()["settings"]
    assert (settings["enabled"], settings["paused"]) == (True, True)
    assert [language["key"] for language in settings["languages"]] == ["fr"]
    assert stale.json()["status"]["state"] == "paused"


@pytest.mark.asyncio
async def test_control_actions_and_validation(client, service) -> None:
    await _enable(client, enabled=False)
    enabled = await client.post("/api/translation/control", json={"action": "enable"})
    assert enabled.status_code == 200
    assert enabled.json()["status"]["enabled"] is True
    paused = await client.post("/api/translation/control", json={"action": "pause"})
    assert paused.json()["status"]["state"] == "paused"
    retried = await client.post("/api/translation/control", json={"action": "retry"})
    assert retried.status_code == 200
    started = await client.post("/api/translation/control", json={"action": "translate_all"})
    assert started.status_code == 200, started.text
    assert started.json()["status"]["translate_all"]["requested"] is True
    assert set(started.json()["status"]["translate_all"]) == TRANSLATE_ALL_KEYS
    cancelled = await client.post("/api/translation/control", json={"action": "cancel_translate_all"})
    assert cancelled.json()["status"]["translate_all"] == {
        "requested": False,
        "state": "off",
        "documents": 0,
        "complete_documents": 0,
        "translated_units": 0,
        "pending_units": 0,
        "message": "",
    }
    for body in ({"action": "explode"}, {}, {"action": "enable", "extra": True}):
        assert (await client.post("/api/translation/control", json=body)).status_code == 422


@pytest.mark.asyncio
async def test_translate_all_is_422_while_translation_is_off(client, service, runtime_root) -> None:
    await _enable(client, enabled=False)
    response = await client.post("/api/translation/control", json={"action": "translate_all"})
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Turn on Foreign Language Translation w/ Top-p before using Translate all."
    )
    stored = json.loads((runtime_root / "runtime_settings.json").read_text(encoding="utf-8"))
    assert stored["auto_translate"]["translate_all"] is False


@pytest.mark.asyncio
async def test_settings_save_keeps_omitted_modes_and_translate_all(client, service) -> None:
    await _enable(client, modes=["compiler", "leanoj", "compiler"])
    assert service.get_settings().modes == ["compiler", "leanoj"]
    assert (await client.post("/api/translation/control", json={"action": "translate_all"})).status_code == 200
    # A client that does not know the newer fields sends neither.
    saved = await client.put("/api/translation/settings", json=_settings(batch_translation=True))
    assert saved.status_code == 200, saved.text
    settings = saved.json()["settings"]
    assert (settings["modes"], settings["translate_all"], settings["batch_translation"]) == (
        ["compiler", "leanoj"], True, True,
    )
    assert saved.json()["status"]["modes"] == ["compiler", "leanoj"]
    # A sent translate_all is ignored; a sent modes list applies.
    saved = await client.put("/api/translation/settings", json=_settings(modes=[], translate_all=False))
    assert (saved.json()["settings"]["modes"], saved.json()["settings"]["translate_all"]) == ([], True)
    assert (await client.get("/api/translation/status")).json()["modes"] == []
    rejected = await client.put("/api/translation/settings", json=_settings(modes=["manual"]))
    assert rejected.status_code == 422


# ----------------------------------------------------------------------
# Languages
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_language_catalog_and_probe_without_roles(client, service) -> None:
    catalog = await client.get("/api/translation/languages")
    assert catalog.status_code == 200
    payload = catalog.json()
    assert payload["pinned_keys"] == list(PINNED_LANGUAGE_KEYS)
    assert payload["probe"] == {"status": "never", "probed_at": None, "models": [], "stale": False}
    assert {language["proficiency"] for language in payload["languages"]} == {"unknown"}

    probe = await client.post("/api/translation/languages/probe")
    assert probe.status_code == 422
    assert "Configure translator models" in probe.json()["detail"]


@pytest.mark.asyncio
async def test_language_probe_reports_the_translator_and_validator_roles(client, service, monkeypatch) -> None:
    await _enable(client, enabled=False)
    asked = []

    async def fake_probe(role_id: str, role: Any) -> list:
        asked.append((role_id, role.model_id))
        return ["Spanish", "French"] if role.model_id == "translator-a" else ["Spanish"]

    monkeypatch.setattr(service, "_probe_model", fake_probe)
    probe = await client.post("/api/translation/languages/probe")
    assert probe.status_code == 200, probe.text
    payload = probe.json()
    assert sorted(model_id for _role_id, model_id in asked) == ["translator-a", "validator-model"]
    assert all(role_id.startswith(service_module.PROBE_ROLE_PREFIX) for role_id, _model_id in asked)
    assert {(model["role"], model["model_id"], model["ok"]) for model in payload["probe"]["models"]} == {
        ("translator", "translator-a", True),
        ("validator", "validator-model", True),
    }
    assert payload["probe"]["status"] == "complete"
    proficiency = {language["key"]: language["proficiency"] for language in payload["languages"]}
    # Proficient only when both the translator and the validator list the language.
    assert proficiency["es"] == "proficient"
    assert proficiency["fr"] == "uncertain"


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_documents_report_disabled_without_registering(client, service) -> None:
    response = await client.post("/api/translation/documents", json={"language": "es", "documents": [_document()]})
    assert response.status_code == 200
    (row,) = response.json()["documents"]
    assert set(row) == DOCUMENT_STATUS_KEYS
    assert row["status"] == "disabled"
    assert row["doc_key"] == "live_paper:test"
    assert len(row["doc_hash"]) == 64
    assert (row["translated_units"], row["total_units"], row["queue_position"]) == (0, 0, None)
    assert row["parts"] is None
    assert not service._docs


@pytest.mark.asyncio
async def test_documents_register_pending_and_lookup(client, service) -> None:
    await _enable(client)
    registered = await client.post(
        "/api/translation/documents",
        json={"language": "es", "viewing": True, "documents": [_document()]},
    )
    assert registered.status_code == 200, registered.text
    (row,) = registered.json()["documents"]
    assert set(row) == DOCUMENT_STATUS_KEYS
    assert row["status"] == "pending"
    assert row["language"] == "es"
    # Two paragraphs are two units; nothing is translated yet.
    assert (row["translated_units"], row["total_units"]) == (0, 2)
    # The service is never started, so the job waits first in line.
    assert row["queue_position"] == 1
    assert row["parts"] is None
    assert (row["requested"], row["automatic"]) == (False, False)

    requested = await client.post(
        "/api/translation/documents",
        json={"language": "es", "viewing": False, "request": True, "documents": [_document()]},
    )
    assert requested.status_code == 200, requested.text
    assert requested.json()["documents"][0]["requested"] is True

    unselected = await client.post("/api/translation/documents", json={"language": "fr", "documents": [_document()]})
    (unavailable,) = unselected.json()["documents"]
    assert (unavailable["status"], unavailable["queue_position"]) == ("unavailable", None)

    lookup = await client.post(
        "/api/translation/documents/lookup",
        json={"language": "es", "doc_hashes": [row["doc_hash"], "0" * 64]},
    )
    assert lookup.status_code == 200
    rows = {item["doc_hash"]: item for item in lookup.json()["documents"]}
    assert {doc_hash: item["status"] for doc_hash, item in rows.items()} == {
        row["doc_hash"]: "pending",
        "0" * 64: "unknown",
    }
    assert rows[row["doc_hash"]]["queue_position"] == 1
    assert rows["0" * 64]["queue_position"] is None

    status = (await client.get("/api/translation/status")).json()
    (queued,) = status["queue"]
    assert queued == {
        "doc_hash": row["doc_hash"],
        "doc_key": "live_paper:test",
        "kind": "live_paper",
        "title": "Live paper",
        "language_key": "es",
        "position": 1,
        "viewing": True,
        "background": False,
        "translated_units": 0,
        "total_units": 2,
    }
    assert status["active"] == []
    (language,) = status["languages"]
    assert (language["pending_units"], language["pending_documents"]) == (2, 1)


@pytest.mark.asyncio
async def test_document_progress_reports_every_language_without_side_effects(client, service) -> None:
    body = {"documents": [_document()]}
    languages = [{"key": "es", "name": "Spanish"}, {"key": "fr", "name": "French"}]
    await _enable(client, enabled=False, languages=languages)
    off = await client.post("/api/translation/documents/progress", json=body)
    assert off.status_code == 200, off.text
    (row,) = off.json()["documents"]
    assert (row["doc_key"], len(row["doc_hash"])) == ("live_paper:test", 64)
    assert [(item["language"], item["status"]) for item in row["languages"]] == [("es", "disabled"), ("fr", "disabled")]

    assert (await client.post("/api/translation/control", json={"action": "enable"})).status_code == 200
    unseen = (await client.post("/api/translation/documents/progress", json=body)).json()["documents"][0]
    assert [(item["language"], item["status"]) for item in unseen["languages"]] == [("es", "unknown"), ("fr", "unknown")]
    # Reading progress never registers, views, or requests the document.
    assert not service._docs
    assert service.status_snapshot()["queue"] == []

    registered = await client.post(
        "/api/translation/documents", json={"language": "es", "viewing": True, "documents": [_document()]}
    )
    doc_hash = registered.json()["documents"][0]["doc_hash"]
    entry = service._docs[doc_hash].keys["live_paper:test"]
    viewed_ts, requested = entry.viewed_ts, set(entry.requested_languages)
    service._final_counts[(doc_hash, "fr")] = 2

    progress = (await client.post("/api/translation/documents/progress", json=body)).json()["documents"][0]
    assert progress["doc_hash"] == doc_hash
    assert progress["languages"] == [
        {"language": "es", "status": "pending", "translated_units": 0, "total_units": 2},
        {"language": "fr", "status": "complete", "translated_units": 2, "total_units": 2},
    ]
    assert (entry.viewed_ts, entry.requested_languages, entry.requested_all) == (viewed_ts, requested, False)


@pytest.mark.asyncio
async def test_viewing_lookup_refreshes_the_viewing_priority(client, service) -> None:
    await _enable(client)
    registered = await client.post(
        "/api/translation/documents",
        json={"language": "es", "viewing": False, "documents": [_document()]},
    )
    (row,) = registered.json()["documents"]
    body = {"language": "es", "doc_hashes": [row["doc_hash"]]}

    async def queue_viewing() -> list:
        return [item["viewing"] for item in (await client.get("/api/translation/status")).json()["queue"]]

    # A document no current run covers translates only while it is viewed; a lookup
    # without "viewing" does not queue it.
    assert (await client.post("/api/translation/documents/lookup", json=body)).status_code == 200
    assert await queue_viewing() == []
    viewed = await client.post("/api/translation/documents/lookup", json={**body, "viewing": True})
    assert viewed.status_code == 200, viewed.text
    (item,) = viewed.json()["documents"]
    assert (item["status"], item["queue_position"]) == ("pending", 1)
    assert await queue_viewing() == [True]
    stored = {record.doc_key: record for record in service_module.translation_store.load_documents()}
    assert stored["live_paper:test"].last_viewed_at


@pytest.mark.asyncio
async def test_queue_positions_follow_dispatch_order(client, service) -> None:
    await _enable(client)
    paper = _document("paper:first", "Only paragraph of the first paper.", kind="paper", title="First")
    proof = _document("proof:second", "Only paragraph of the proof.", kind="proof", title="Proof")
    response = await client.post(
        "/api/translation/documents",
        json={"language": "es", "viewing": True, "documents": [paper, proof]},
    )
    assert response.status_code == 200, response.text
    positions = {item["doc_key"]: item["queue_position"] for item in response.json()["documents"]}
    # Proofs come before papers when both are being viewed.
    assert positions == {"proof:second": 1, "paper:first": 2}
    status = (await client.get("/api/translation/status")).json()
    assert [(item["doc_key"], item["position"]) for item in status["queue"]] == [
        ("proof:second", 1),
        ("paper:first", 2),
    ]


@pytest.mark.asyncio
async def test_empty_document_returns_its_parts(client, service) -> None:
    await _enable(client)
    response = await client.post(
        "/api/translation/documents",
        json={"language": "es", "documents": [_document(parts=[{"name": "content", "text": "   \n"}])]},
    )
    (row,) = response.json()["documents"]
    assert row["status"] == "empty"
    assert (row["translated_units"], row["total_units"], row["queue_position"]) == (0, 0, None)
    assert row["parts"] == {"content": "   \n"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"language": "es", "documents": []},
        {"language": "es", "documents": [_document(f"other:{index}") for index in range(201)]},
        {"language": "es", "documents": [_document("x" * 301)]},
        {"language": "es", "documents": [_document("bad\nkey")]},
        {"language": "es", "documents": [_document(title="t" * 301)]},
        {"language": "es", "documents": [_document(kind="novel")]},
        {"language": "es", "documents": [_document(profile="lean")]},
        {"language": "es", "documents": [_document(parts=[{"name": "n" * 65, "text": "x"}])]},
        {"language": "es", "documents": [_document(parts=[{"name": "", "text": "x"}])]},
        {"language": "es", "documents": [_document(parts=[{"name": "a", "text": "x"}, {"name": "a", "text": "y"}])]},
        {"language": "es", "documents": [_document(extra="field")]},
        {"language": "e", "documents": [_document()]},
        {"language": "es", "documents": [_document()], "unexpected": 1},
    ],
)
async def test_document_request_limits_are_422(client, service, body) -> None:
    response = await client.post("/api/translation/documents", json=body)
    assert response.status_code == 422
    assert all(set(item) == {"loc", "msg", "type"} for item in response.json()["detail"])


@pytest.mark.asyncio
async def test_document_total_text_limit_is_422_without_echo(client, service) -> None:
    half = "a" * 2_000_001
    body = {
        "language": "es",
        "documents": [
            _document("live_paper:one", half),
            _document("live_paper:two", half),
        ],
    }
    response = await client.post("/api/translation/documents", json=body)
    assert response.status_code == 422
    assert len(response.content) < 2000
    assert "4,000,000 characters" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"language": "es", "doc_hashes": []},
        {"language": "es", "doc_hashes": ["0" * 64] * 501},
        {"language": "es", "doc_hashes": ["NOT-A-HASH"]},
        {"language": None, "doc_hashes": ["0" * 64]},
        {"doc_hashes": ["0" * 64]},
        {"language": "es", "doc_hashes": ["0" * 64], "viewing": "maybe"},
        {"language": "es", "doc_hashes": ["0" * 64], "unexpected": 1},
    ],
)
async def test_lookup_request_limits_are_422(client, service, body) -> None:
    assert (await client.post("/api/translation/documents/lookup", json=body)).status_code == 422


@pytest.mark.asyncio
async def test_unexpected_failures_return_generic_500_without_exception_text(client, service, monkeypatch) -> None:
    async def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("leaked detail C:/private sk-" + "Ab3" * 16)

    monkeypatch.setattr(service, "lookup_documents", explode)
    response = await client.post("/api/translation/documents/lookup", json={"language": "es", "doc_hashes": ["0" * 64]})
    assert response.status_code == 500
    assert response.json() == {"detail": "Foreign Language Translation w/ Top-p could not look up those documents."}


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_clear_requires_confirmation(client, service) -> None:
    await _enable(client)
    refused = await client.delete("/api/translation/cache")
    assert refused.status_code == 400
    refused_false = await client.delete("/api/translation/cache?confirm=false")
    assert refused_false.status_code == 400
    cleared = await client.delete("/api/translation/cache?confirm=true")
    assert cleared.status_code == 200
    payload = cleared.json()
    assert payload["success"] is True
    assert set(payload["status"]) == STATUS_KEYS
