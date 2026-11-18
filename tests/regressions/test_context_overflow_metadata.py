from backend.aggregator.core.context_allocator import ContextAllocationError
from backend.shared.context_overflow import (
    context_overflow_model_payload,
    context_overflow_token_payload,
)
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteIdentity
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


def test_overflow_tokens_prefer_structured_error_counts() -> None:
    error = ContextAllocationError(
        "mandatory context does not fit",
        required_tokens=9000,
        available_tokens=7000,
        context_window=8192,
        output_reserve=1192,
    )

    assert context_overflow_token_payload(error) == {
        "required_tokens": 9000,
        "available_tokens": 7000,
        "context_window": 8192,
        "output_reserve": 1192,
    }


def test_overflow_tokens_read_wrapped_provider_rejection() -> None:
    provider_error = ProviderContextLengthError(
        "LM Studio rejected the prompt.",
        route=ProviderRouteIdentity(provider="lm_studio", model="local/model"),
        required_tokens=50000,
        context_window=32768,
    )
    wrapper = ContextAllocationError.from_provider_error(provider_error, "overflow")

    payload = context_overflow_token_payload(wrapper)

    assert payload["required_tokens"] == 50000
    assert payload["context_window"] == 32768
    routed = provider_error.with_route_context(role_id="r", task_id="t")
    assert (routed.required_tokens, routed.context_window) == (50000, 32768)


def test_overflow_tokens_fall_back_to_message_text_and_role_config() -> None:
    config = ModelConfig(
        provider="lm_studio",
        model_id="local/model",
        context_window=32768,
        max_output_tokens=6000,
    )

    assert context_overflow_token_payload(
        ValueError("Prompt too large: 40,123 tokens > 26768 max"),
        config=config,
    ) == {
        "required_tokens": 40123,
        "available_tokens": 26768,
        "context_window": 32768,
    }
    assert context_overflow_token_payload(
        ValueError("Prompt (50000 tokens) exceeds model's context window (32768 tokens).")
    ) == {"required_tokens": 50000, "context_window": 32768}
    assert context_overflow_token_payload(ValueError("no counts here"), config=config) == {
        "context_window": 32768,
        "output_reserve": 6000,
    }
    assert context_overflow_token_payload(ValueError("no counts here")) == {}
