from fastapi import FastAPI
from pydantic import ValidationError
import pytest

from backend.api.routes.features import FeaturesResponse, router


EXPECTED_FEATURE_FIELDS = {
    "version",
    "build_commit",
    "update_channel",
    "api_contract_version",
    "generic_mode",
    "lm_studio_enabled",
    "pdf_download_available",
    "openai_codex_oauth_available",
    "xai_grok_oauth_available",
    "sakana_fugu_available",
    "syntheticlib_product_state",
}


def test_features_openapi_uses_strict_typed_response() -> None:
    app = FastAPI()
    app.include_router(router)
    openapi = app.openapi()

    response_schema = openapi["paths"]["/api/features"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    assert response_schema["$ref"] == "#/components/schemas/FeaturesResponse"

    model_schema = openapi["components"]["schemas"]["FeaturesResponse"]
    assert set(model_schema["properties"]) == EXPECTED_FEATURE_FIELDS
    assert set(model_schema["required"]) == EXPECTED_FEATURE_FIELDS
    assert model_schema["additionalProperties"] is False


def test_features_response_rejects_unknown_fields() -> None:
    payload = {
        "version": "1.0.0",
        "build_commit": "abc123",
        "update_channel": "main",
        "api_contract_version": "build6-v94",
        "generic_mode": False,
        "lm_studio_enabled": True,
        "pdf_download_available": True,
        "openai_codex_oauth_available": True,
        "xai_grok_oauth_available": True,
        "sakana_fugu_available": True,
        "syntheticlib_product_state": "coming_soon",
        "unexpected": "not part of the stable contract",
    }

    with pytest.raises(ValidationError):
        FeaturesResponse.model_validate(payload)


def test_features_openapi_exposes_syntheticlib_lifecycle_enum() -> None:
    app = FastAPI()
    app.include_router(router)
    model_schema = app.openapi()["components"]["schemas"]["FeaturesResponse"]
    lifecycle_schema = model_schema["properties"]["syntheticlib_product_state"]
    assert lifecycle_schema["enum"] == ["coming_soon", "developer_preview", "launched"]


PROVIDER_ROUTE_IDENTITY_FIELDS = (
    "model_role_id",
    "role_label",
    "route_kind",
    "configured_host_provider",
    "effective_host_provider",
    "host_routing",
    "provider_request_id",
    "route_identity",
)


def _optional_string_property(schema: dict) -> bool:
    variants = schema.get("anyOf") or [schema]
    return {"type": "string"} in variants and schema.get("default") is None


def test_provider_notification_record_documents_route_identity_fields() -> None:
    from backend.api.routes import cloud_access

    app = FastAPI()
    app.include_router(cloud_access.router)
    openapi = app.openapi()

    for path in (
        "/api/cloud-access/provider-notifications",
        "/api/cloud-access/provider-recovery/events",
    ):
        schema = openapi["paths"][path]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert schema["$ref"] == "#/components/schemas/ProviderNotificationsResponse"
    properties = openapi["components"]["schemas"]["ProviderNotificationRecord"]["properties"]
    names = list(properties)
    model_index = names.index("model")
    assert tuple(names[model_index + 1: model_index + 1 + len(PROVIDER_ROUTE_IDENTITY_FIELDS)]) == (
        PROVIDER_ROUTE_IDENTITY_FIELDS
    )
    for field in PROVIDER_ROUTE_IDENTITY_FIELDS:
        assert _optional_string_property(properties[field]), field


def test_autonomous_terminal_event_documents_route_identity_fields() -> None:
    from backend.api.routes import autonomous

    app = FastAPI()
    app.include_router(autonomous.router)
    openapi = app.openapi()

    status_schema = openapi["components"]["schemas"]["AutonomousResearchStatusResponse"]
    terminal_ref = status_schema["properties"]["terminal_event"]["anyOf"][0]["$ref"]
    assert terminal_ref == "#/components/schemas/AutonomousTerminalEvent"
    properties = openapi["components"]["schemas"]["AutonomousTerminalEvent"]["properties"]
    for field in (
        "configured_host_provider",
        "effective_host_provider",
        "host_routing",
        "role_label",
        "route_identity",
        "route_kind",
    ):
        assert _optional_string_property(properties[field]), field


def test_syntheticlib_trade_evidence_route_has_typed_ineligible_response() -> None:
    from backend.api.routes import syntheticlib4

    app = FastAPI()
    app.include_router(syntheticlib4.router)
    openapi = app.openapi()

    response_schema = openapi["paths"]["/api/syntheticlib4/developer/trade-evidence"][
        "get"
    ]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"] == (
        "#/components/schemas/TradeEvidenceDiagnosticsResponse"
    )
    schemas = openapi["components"]["schemas"]
    assert schemas["TradeEvidenceDiagnosticsResponse"]["properties"][
        "eligible_to_offer"
    ]["const"] is False
    assert schemas["TradeEvidenceDiagnosticResponse"]["properties"][
        "eligible_to_offer"
    ]["const"] is False


def test_syntheticlib_trade_ledger_controls_have_typed_no_upload_responses() -> None:
    from backend.api.routes import syntheticlib4

    app = FastAPI()
    app.include_router(syntheticlib4.router)
    openapi = app.openapi()
    paths = openapi["paths"]
    schemas = openapi["components"]["schemas"]

    expected = {
        ("/api/syntheticlib4/developer/trade-ledger/entries", "get"): "TradeLedgerEntriesResponse",
        ("/api/syntheticlib4/developer/trade-ledger/snapshots", "post"): "TradeLedgerSnapshotMutationResponse",
        ("/api/syntheticlib4/developer/trade-ledger/snapshots/reconcile", "post"): "TradeLedgerSnapshotMutationResponse",
        ("/api/syntheticlib4/developer/trade-ledger/snapshots/cancel", "post"): "TradeLedgerSnapshotMutationResponse",
        ("/api/syntheticlib4/developer/trade-ledger/batches/prepare", "post"): "TradeLedgerPrepareBatchResponse",
        ("/api/syntheticlib4/developer/trade-ledger/batches/cancel", "post"): "TradeLedgerBatchMutationResponse",
        ("/api/syntheticlib4/developer/trade-ledger/archive-reset", "post"): "TradeLedgerArchiveResetResponse",
        ("/api/syntheticlib4/developer/trade-ledger/archive/delete", "post"): "TradeLedgerArchiveDeleteResponse",
    }
    for (path, method), model_name in expected.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema["$ref"] == f"#/components/schemas/{model_name}"
        properties = schemas[model_name]["properties"]
        if model_name in {"TradeLedgerArchiveResetResponse", "TradeLedgerArchiveDeleteResponse"}:
            assert properties["server_side_deletion_claimed"]["const"] is False
        else:
            assert properties["upload_enabled"]["const"] is False


def test_legacy_port_migration_and_evidence_class_are_typed() -> None:
    from backend.api.routes import syntheticlib4

    app = FastAPI()
    app.include_router(syntheticlib4.router)
    openapi = app.openapi()
    schemas = openapi["components"]["schemas"]
    route = openapi["paths"]["/api/syntheticlib4/developer/legacy-port/migrate"]["post"]
    schema = route["responses"]["200"]["content"]["application/json"]["schema"]

    assert schema["$ref"] == "#/components/schemas/LegacyPortMigrateResponse"
    properties = schemas["LegacyPortMigrateResponse"]["properties"]
    assert properties["upload_enabled"]["const"] is False
    assert properties["server_side_claim"]["const"] is False
    evidence_class = schemas["TradeEvidenceDiagnosticResponse"]["properties"]["evidence_class"]
    assert set(evidence_class["enum"]) == {"full", "legacy_port", "none"}
    assert "/api/proofs/{proof_id}/reverify-evidence" not in openapi["paths"]
