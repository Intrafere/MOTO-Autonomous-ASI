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
        "api_contract_version": "build6-v91",
        "generic_mode": False,
        "lm_studio_enabled": True,
        "pdf_download_available": True,
        "openai_codex_oauth_available": True,
        "xai_grok_oauth_available": True,
        "sakana_fugu_available": True,
        "unexpected": "not part of the stable contract",
    }

    with pytest.raises(ValidationError):
        FeaturesResponse.model_validate(payload)
