from backend.shared.context_overflow import context_overflow_model_payload
from backend.shared.models import ModelConfig


def test_context_overflow_model_payload_exposes_configured_identity() -> None:
    payload = context_overflow_model_payload(
        ModelConfig(provider="openrouter", model_id="example/model")
    )

    assert payload == {
        "configured_model": "example/model",
        "configured_provider": "openrouter",
    }


def test_context_overflow_model_payload_allows_missing_config() -> None:
    assert context_overflow_model_payload(None) == {}
