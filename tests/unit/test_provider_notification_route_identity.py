"""Provider notification storage keeps route identity for reconnect recovery."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.api.routes.cloud_access import ProviderNotificationRecord, ProviderNotificationsResponse
from backend.shared.config import system_config
from backend.shared.provider_notification_store import (
    PROVIDER_NOTIFICATIONS_FILENAME,
    list_provider_notifications,
    record_provider_notification,
)
from backend.shared.provider_route_identity import ROUTE_IDENTITY_MAX_CHARS, route_identity_fields


ROUTE_IDENTITY_KEYS = (
    "model_role_id",
    "role_label",
    "route_kind",
    "configured_host_provider",
    "effective_host_provider",
    "host_routing",
    "provider_request_id",
    "route_identity",
)
SYNTHETIC_MODEL = "vendor/" + "test-free-model-" * 11 + ":free"


@pytest.fixture
def temp_data_dir(tmp_path):
    old_data_dir = system_config.data_dir
    system_config.data_dir = str(tmp_path)
    try:
        yield tmp_path
    finally:
        system_config.data_dir = old_data_dir


def _long_identity_fields() -> dict:
    return route_identity_fields(
        workflow_mode="manual_proof_check",
        role_id="autonomous_proof_candidate_list_validator_compiler_manual_paper",
        route_kind="free_rotation",
        model=SYNTHETIC_MODEL,
        provider="openrouter",
        configured_host_provider="TestHost" + "x" * 72,
        upstream_host_provider="TestUpstream" + "y" * 68,
        provider_request_id="gen-test-123",
    )


def _retry_payload(**overrides) -> dict:
    payload = {
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "role_id": "aggregator_single_model",
        "model_role_id": "aggregator_submitter_2",
        "model": "vendor/test-free:free",
        "role_label": "Submitter 2",
        "route_kind": "primary",
        "configured_host_provider": "",
        "effective_host_provider": "TestHost",
        "host_routing": "auto",
        "provider_request_id": "gen-test-1",
        "route_identity": (
            "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter"
            " · host auto → TestHost]"
        ),
        "task_id": "agg_sub2_001",
        "reason": "provider_retry_cooldown",
        "recoverable": True,
        "message": "OpenRouter API call will retry after a 2-second cooldown (retry 1).",
        "notification_kind": "provider_retry",
        "workflow_mode": "autonomous",
        "event_id": "provider_retry:test-correlation:1",
    }
    payload.update(overrides)
    return payload


def test_record_keeps_every_route_identity_key(temp_data_dir):
    payload = _retry_payload()

    stored = record_provider_notification("provider_retry_cooldown", payload)

    for key in ROUTE_IDENTITY_KEYS:
        assert stored[key] == payload[key], key
    assert stored["task_id"] == "agg_sub2_001"
    assert stored["workflow_mode"] == "autonomous"
    [reloaded] = list_provider_notifications()
    for key in ROUTE_IDENTITY_KEYS:
        assert reloaded[key] == payload[key], key


def test_route_identity_is_stored_byte_for_byte(temp_data_dir):
    fields = _long_identity_fields()
    identity = fields["route_identity"]
    assert 450 <= len(identity) <= ROUTE_IDENTITY_MAX_CHARS
    payload = _retry_payload(
        **{key: value for key, value in fields.items() if key != "workflow_mode"},
        workflow_mode="manual_proof_check",
        message=f"{identity} OpenRouter returned no completion choices.",
    )

    stored = record_provider_notification("provider_retry_cooldown", payload)

    assert stored["route_identity"] == identity
    assert stored["role_label"] == fields["role_label"]
    assert stored["configured_host_provider"] == fields["configured_host_provider"]
    assert stored["effective_host_provider"] == fields["effective_host_provider"]
    [reloaded] = list_provider_notifications()
    assert reloaded["route_identity"] == identity
    assert reloaded["route_identity"].encode("utf-8") == identity.encode("utf-8")
    record = ProviderNotificationRecord.model_validate(reloaded)
    assert record.route_identity == identity
    assert record.message.startswith(identity)


def test_route_identity_keys_follow_model_before_persistence_keys(temp_data_dir):
    stored = record_provider_notification("provider_retry_cooldown", _retry_payload())

    keys = list(stored)
    model_index = keys.index("model")
    assert tuple(keys[model_index + 1: model_index + 1 + len(ROUTE_IDENTITY_KEYS)]) == (
        ROUTE_IDENTITY_KEYS
    )
    persistence_index = min(keys.index(key) for key in keys if key.startswith("persistence_"))
    assert all(keys.index(key) < persistence_index for key in ROUTE_IDENTITY_KEYS)

    dumped_keys = list(ProviderNotificationRecord.model_validate(stored).model_dump())
    dumped_model_index = dumped_keys.index("model")
    assert tuple(
        dumped_keys[dumped_model_index + 1: dumped_model_index + 1 + len(ROUTE_IDENTITY_KEYS)]
    ) == ROUTE_IDENTITY_KEYS


def test_auto_host_is_distinct_from_unknown_host(temp_data_dir):
    auto = record_provider_notification(
        "provider_retry_cooldown",
        _retry_payload(event_id="provider_retry:auto:1"),
    )
    unknown_payload = _retry_payload(event_id="provider_retry:unknown:1")
    for key in ("configured_host_provider", "effective_host_provider", "host_routing"):
        unknown_payload.pop(key)
    unknown = record_provider_notification("provider_retry_cooldown", unknown_payload)

    assert auto["configured_host_provider"] == ""
    assert "configured_host_provider" not in unknown
    assert "host_routing" not in unknown
    reloaded = {item["notification_key"]: item for item in list_provider_notifications()}
    assert reloaded["provider_retry:auto:1"]["configured_host_provider"] == ""
    assert "configured_host_provider" not in reloaded["provider_retry:unknown:1"]
    assert ProviderNotificationRecord.model_validate(
        reloaded["provider_retry:unknown:1"]
    ).configured_host_provider is None


def test_invalid_workflow_mode_is_still_dropped(temp_data_dir):
    stored = record_provider_notification(
        "provider_retry_cooldown",
        _retry_payload(workflow_mode="construction"),
    )

    assert stored["workflow_mode"] == ""
    assert stored["route_identity"].startswith("[Autonomous")


def test_legacy_records_without_route_identity_still_load(temp_data_dir):
    legacy = {
        "id": "openrouter:aggregator_submitter_2:provider_error:vendor/test-free:free",
        "notification_key": "openrouter:aggregator_submitter_2:provider_error:vendor/test-free_free",
        "event_type": "provider_retry_cooldown",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "role_id": "aggregator_submitter_2",
        "model": "vendor/test-free:free",
        "reason": "provider_error",
        "recoverable": True,
        "message": "OpenRouter API call will retry after a 2-second cooldown (retry 1).",
        "notification_kind": "provider_retry",
        "workflow_mode": "autonomous",
    }
    Path(temp_data_dir, PROVIDER_NOTIFICATIONS_FILENAME).write_text(
        json.dumps({"notifications": [legacy]}),
        encoding="utf-8",
    )

    notifications = list_provider_notifications()

    assert notifications == [legacy]
    response = ProviderNotificationsResponse.model_validate(
        {"success": True, "notifications": notifications}
    )
    [record] = response.notifications
    for key in ROUTE_IDENTITY_KEYS:
        assert getattr(record, key) is None, key
    assert record.message == legacy["message"]
