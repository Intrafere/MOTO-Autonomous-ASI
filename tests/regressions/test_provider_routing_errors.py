import unittest
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import backend.shared.api_client_manager as api_manager_module
from backend.shared.api_client_manager import (
    APIClientManager,
    ProviderCooldownError,
    RetryableProviderError,
    _is_retryable_codex_completion_error,
    _typed_provider_context_error,
)
from backend.aggregator.agents.submitter import SubmitterAgent
from backend.shared.boost_manager import boost_manager
from backend.shared.config import system_config
from backend.shared.lm_studio_client import LMStudioClient
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_provider_context_length_error,
    is_transient_model_call_error,
)
from backend.shared.models import BoostConfig, ModelConfig
from backend.shared.openai_codex_client import (
    OpenAICodexAuthError,
    OpenAICodexRequestError,
)
from backend.shared.openrouter_client import (
    OpenRouterClient,
    _reported_upstream,
    _safe_error_json,
)
from backend.shared.provider_errors import (
    ProviderContextLengthError,
    ProviderRepairRequiredError,
    ProviderRouteError,
    ProviderRouteIdentity,
)
from backend.shared.subscription_providers.errors import SubscriptionEntitlementError
from backend.shared.subscription_providers.registry import get_api_key_subscription_client


@pytest.fixture(autouse=True)
def _isolated_provider_notification_root(tmp_path, monkeypatch):
    """Keep direct-call retry notifications out of the real backend/data store."""
    monkeypatch.setattr(system_config, "data_dir", str(tmp_path))


class ProviderErrorUtilityTests(unittest.TestCase):
    def test_ambiguous_codex_completion_rejection_is_retryable(self):
        error = OpenAICodexRequestError(
            "OpenAI Codex completion failed: opaque backend rejection"
        )

        self.assertTrue(_is_retryable_codex_completion_error(error))

    def test_hard_codex_failures_are_not_retryable(self):
        errors = (
            OpenAICodexAuthError("OpenAI Codex OAuth is not configured."),
            OpenAICodexRequestError(
                "OpenAI Codex completion failed: unsupported model",
                status_code=400,
                error_code="unsupported_model",
                failure_kind="http_rejected",
            ),
            OpenAICodexRequestError(
                "OpenAI Codex completion failed: context_length_exceeded",
                status_code=400,
                error_code="context_length_exceeded",
                failure_kind="http_rejected",
            ),
            OpenAICodexRequestError(
                "OpenAI Codex completion failed: previously unseen deterministic rejection",
                status_code=422,
                error_code="new_invalid_parameter",
                failure_kind="http_rejected",
            ),
        )

        for error in errors:
            with self.subTest(error=str(error)):
                self.assertFalse(_is_retryable_codex_completion_error(error))

    def test_structured_codex_transport_and_stream_failures_are_retryable(self):
        errors = (
            OpenAICodexRequestError(
                "OpenAI Codex connection failed after retries",
                failure_kind="transport_exhausted",
            ),
            OpenAICodexRequestError(
                "OpenAI Codex streamed response contained no completion output.",
                failure_kind="empty_stream",
            ),
            OpenAICodexRequestError(
                "OpenAI Codex transient completion failed",
                status_code=503,
                error_code="server_error",
                failure_kind="transient_http_exhausted",
            ),
        )

        for error in errors:
            with self.subTest(error=str(error)):
                self.assertTrue(_is_retryable_codex_completion_error(error))

    def test_context_error_is_typed_and_never_exposes_secret(self):
        error = ProviderContextLengthError(
            "context_length_exceeded Bearer secret-value",
            route=ProviderRouteIdentity(
                provider="openrouter",
                model="vendor/model",
                host_provider="host",
            ),
        )

        self.assertTrue(is_provider_context_length_error(error))
        self.assertTrue(is_non_retryable_model_error(error))
        self.assertNotIn("secret-value", str(error))
        self.assertIn("vendor/model", str(error))

    def test_wrapped_transient_error_uses_original_cause(self):
        cause = ValueError("OpenRouter connection failed after retries")
        error = ProviderRouteError(
            "Provider request failed",
            route=ProviderRouteIdentity(provider="openrouter", model="vendor/model"),
            cause=cause,
        )

        self.assertTrue(is_transient_model_call_error(error))
        self.assertFalse(is_non_retryable_model_error(error))

    def test_provider_repair_error_preserves_only_redacted_safe_detail(self):
        normalized = APIClientManager._as_provider_repair_error(
            provider="lm_studio",
            provider_label="LM Studio",
            role_id="proof_identifier",
            model="local-model",
            error=ProviderRouteError(
                "LM Studio rejected the request: Bearer secret-value",
                route=ProviderRouteIdentity(provider="lm_studio", model="local-model"),
            ),
        )

        self.assertIn("LM Studio rejected the request", normalized.error_detail)
        self.assertNotIn("secret-value", normalized.error_detail)

    def test_timeout_subclass_is_transient(self):
        request = httpx.Request("POST", "https://provider.invalid")
        cause = httpx.ReadTimeout("slow", request=request)
        error = ProviderRouteError(
            "Provider timed out",
            route=ProviderRouteIdentity(provider="openrouter", model="vendor/model"),
            cause=cause,
        )

        self.assertTrue(is_transient_model_call_error(error))

    def test_oauth_and_sakana_context_wrapping_is_typed_and_redacted(self):
        for provider, model in (
            ("sakana_fugu", "fugu-model"),
            ("openai_codex_oauth", "codex-model"),
            ("xai_grok_oauth", "grok-model"),
        ):
            with self.subTest(provider=provider):
                error = _typed_provider_context_error(
                    ValueError("context_length_exceeded Bearer secret-value"),
                    provider=provider,
                    model=model,
                )
                self.assertEqual(error.route.provider, provider)
                self.assertEqual(error.route.model, model)
                self.assertNotIn("secret-value", str(error))


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
UPSTREAM_ERROR_BODY = {
    "id": "gen-test-0001",
    "error": {
        "code": 429,
        "message": "TestHost is temporarily rate-limited upstream",
        "metadata": {"provider_name": "TestHost"},
    },
}


def _openrouter_response(status_code, **kwargs):
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", OPENROUTER_URL),
        **kwargs,
    )


def _success_response():
    return _openrouter_response(
        200,
        json={"choices": [{"message": {"content": "recovered"}}]},
    )


class OpenRouterUpstreamReportTests(unittest.TestCase):
    def test_safe_error_json_ignores_html_gateway_body(self):
        response = _openrouter_response(
            502,
            text="<html><body>502 Bad Gateway</body></html>",
            headers={"content-type": "text/html"},
        )

        self.assertEqual(_safe_error_json(response), {})

    def test_safe_error_json_ignores_json_list_body(self):
        response = _openrouter_response(500, json=[{"error": {"message": "x"}}])

        self.assertEqual(_safe_error_json(response), {})

    def test_safe_error_json_returns_object_body_and_tolerates_missing_json(self):
        response = _openrouter_response(503, json=UPSTREAM_ERROR_BODY)

        self.assertEqual(_safe_error_json(response), UPSTREAM_ERROR_BODY)
        self.assertEqual(_safe_error_json(object()), {})

    def test_reported_upstream_prefers_error_metadata_provider_name(self):
        reported = _reported_upstream({**UPSTREAM_ERROR_BODY, "provider": "OtherHost"})

        self.assertEqual(
            reported,
            {
                "upstream_host": "TestHost",
                "upstream_error_code": "429",
                "upstream_error_message": "TestHost is temporarily rate-limited upstream",
                "request_id": "gen-test-0001",
            },
        )

    def test_reported_upstream_falls_back_to_top_level_provider(self):
        reported = _reported_upstream({"id": "gen-test-0002", "provider": "TestHost"})

        self.assertEqual(reported["upstream_host"], "TestHost")
        self.assertEqual(reported["request_id"], "gen-test-0002")
        self.assertEqual(reported["upstream_error_code"], "")
        self.assertEqual(reported["upstream_error_message"], "")

    def test_reported_upstream_is_total_for_missing_or_malformed_shapes(self):
        empty = {
            "upstream_host": "",
            "upstream_error_code": "",
            "upstream_error_message": "",
            "request_id": "",
        }
        for data in (
            None,
            [],
            "text body",
            {},
            {"error": "plain string", "id": 7, "provider": {"name": "TestHost"}},
            {"error": {"metadata": ["TestHost"], "code": True, "message": None}},
        ):
            with self.subTest(data=data):
                self.assertEqual(_reported_upstream(data), empty)

    def test_reported_upstream_redacts_and_bounds_upstream_message(self):
        reported = _reported_upstream({
            "error": {
                "code": "upstream_error",
                "message": "Bearer secret-value " + ("x" * 2000),
            },
        })

        self.assertNotIn("secret-value", reported["upstream_error_message"])
        self.assertLess(len(reported["upstream_error_message"]), 300)
        self.assertEqual(reported["upstream_error_code"], "upstream_error")


class ProviderClientTypedErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_openrouter_context_rejection_is_typed(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(
            400,
            request=request,
            json={"error": {"message": "context_length_exceeded"}},
        )
        client.client.post = AsyncMock(return_value=response)
        client.MAX_RETRIES = 1
        try:
            with self.assertRaises(ProviderContextLengthError) as raised:
                await client.generate_completion(
                    model="vendor/model",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    provider="safe-host",
                )
            self.assertEqual(raised.exception.route.provider, "openrouter")
            self.assertEqual(raised.exception.route.host_provider, "safe-host")
        finally:
            await client.close()

    async def test_openrouter_timeout_is_typed(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        client.client.post = AsyncMock(side_effect=httpx.ReadTimeout("slow", request=request))
        client.MAX_RETRIES = 1
        try:
            with self.assertRaises(ProviderRouteError) as raised:
                await client.generate_completion(
                    model="vendor/model",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
            self.assertIsInstance(raised.exception.cause, httpx.TimeoutException)
        finally:
            await client.close()

    async def test_openrouter_missing_choices_retries_inside_provider_adapter(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        responses = [
            httpx.Response(
                200,
                request=request,
                json={"error": {"message": "upstream provider timeout"}},
            ),
            httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": "recovered"}}]},
            ),
        ]
        client.client.post = AsyncMock(side_effect=responses)
        client.MAX_RETRIES = 3
        retry_callback = AsyncMock()
        try:
            response = await client.generate_completion(
                model="nvidia/model:free",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                retry_callback=retry_callback,
            )
            self.assertEqual(response["choices"][0]["message"]["content"], "recovered")
            self.assertEqual(client.client.post.await_count, 2)
            retry_callback.assert_awaited_once()
            self.assertEqual(
                retry_callback.await_args.args[0]["reason"],
                "invalid_response",
            )
        finally:
            await client.close()

    async def test_openrouter_missing_choices_exhaustion_is_typed_transient(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(
            200,
            request=request,
            json={"error": {"message": "upstream provider timeout"}},
        )
        client.client.post = AsyncMock(return_value=response)
        client.MAX_RETRIES = 3
        retry_callback = AsyncMock()
        try:
            with self.assertRaises(ProviderRouteError) as raised:
                await client.generate_completion(
                    model="nvidia/model:free",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    retry_callback=retry_callback,
                )
            self.assertEqual(client.client.post.await_count, 3)
            self.assertEqual(retry_callback.await_count, 2)
            self.assertTrue(is_transient_model_call_error(raised.exception))
            self.assertFalse(is_non_retryable_model_error(raised.exception))
            self.assertNotIn("upstream provider timeout", str(raised.exception))
        finally:
            await client.close()

    async def test_openrouter_malformed_choices_type_uses_invalid_response_retry(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        responses = [
            httpx.Response(
                200,
                request=request,
                json={"choices": {"message": {"content": "wrong shape"}}},
            ),
            httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": "recovered"}}]},
            ),
        ]
        client.client.post = AsyncMock(side_effect=responses)
        retry_callback = AsyncMock()
        try:
            response = await client.generate_completion(
                model="vendor/model",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                retry_callback=retry_callback,
            )
            self.assertEqual(response["choices"][0]["message"]["content"], "recovered")
            self.assertEqual(client.client.post.await_count, 2)
            retry_callback.assert_awaited_once()
            self.assertEqual(
                retry_callback.await_args.args[0]["reason"],
                "invalid_response",
            )
        finally:
            await client.close()

    async def test_openrouter_empty_choices_retry_payload_reports_auto_upstream_route(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(side_effect=[
            _openrouter_response(200, json=UPSTREAM_ERROR_BODY),
            _success_response(),
        ])
        retry_callback = AsyncMock()
        try:
            await client.generate_completion(
                model="vendor/test-free:free",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                retry_callback=retry_callback,
            )
            payload = retry_callback.await_args.args[0]
            self.assertEqual(payload["provider"], "openrouter")
            self.assertEqual(payload["model"], "vendor/test-free:free")
            self.assertEqual(payload["configured_host_provider"], "")
            self.assertEqual(payload["upstream_host_provider"], "TestHost")
            self.assertEqual(payload["auto_fallback_from_host"], "")
            self.assertEqual(payload["provider_request_id"], "gen-test-0001")
            self.assertEqual(
                payload["error_summary"],
                "OpenRouter returned a successful response without completion choices "
                "(upstream 429: TestHost is temporarily rate-limited upstream).",
            )
        finally:
            await client.close()

    async def test_openrouter_empty_choices_retry_payload_reports_pinned_host(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(side_effect=[
            _openrouter_response(200, json={"id": "gen-test-0002"}),
            _success_response(),
        ])
        retry_callback = AsyncMock()
        try:
            await client.generate_completion(
                model="vendor/test-free:free",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                provider="TestHost",
                retry_callback=retry_callback,
            )
            payload = retry_callback.await_args.args[0]
            self.assertEqual(payload["configured_host_provider"], "TestHost")
            self.assertEqual(payload["upstream_host_provider"], "")
            self.assertEqual(payload["auto_fallback_from_host"], "")
            self.assertEqual(payload["provider_request_id"], "gen-test-0002")
            self.assertEqual(
                payload["error_summary"],
                "OpenRouter returned a successful response without completion choices.",
            )
        finally:
            await client.close()

    async def test_openrouter_empty_choices_exhaustion_route_carries_upstream_host(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(
            return_value=_openrouter_response(200, json=UPSTREAM_ERROR_BODY)
        )
        client.MAX_RETRIES = 2
        try:
            with self.assertRaises(ProviderRouteError) as raised:
                await client.generate_completion(
                    model="vendor/test-free:free",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    retry_callback=AsyncMock(),
                )
            route = raised.exception.route
            self.assertEqual(route.host_provider, "")
            self.assertEqual(route.upstream_host_provider, "TestHost")
            self.assertEqual(route.provider_request_id, "gen-test-0001")
            self.assertEqual(route.auto_fallback_from_host, "")
            self.assertEqual(
                str(raised.exception),
                "OpenRouter returned no completion choices after 2 attempts. "
                "(provider=openrouter, model=vendor/test-free:free, route=primary)",
            )
            self.assertTrue(is_transient_model_call_error(raised.exception))
        finally:
            await client.close()

    async def test_openrouter_auto_fallback_marks_subsequent_retry_payloads(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(side_effect=[
            _openrouter_response(
                404,
                json={"error": {"code": 404, "message": "No endpoints found for vendor/test-free:free."}},
            ),
            _openrouter_response(200, json={"id": "gen-test-0003", "provider": "TestHost"}),
            _success_response(),
        ])
        retry_callback = AsyncMock()
        try:
            response = await client.generate_completion(
                model="vendor/test-free:free",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                provider="PinnedHost",
                allow_provider_auto_fallback=True,
                retry_callback=retry_callback,
            )
            self.assertEqual(
                response["_moto_openrouter_provider_auto_fallback"],
                {"from_provider": "PinnedHost"},
            )
            retry_callback.assert_awaited_once()
            payload = retry_callback.await_args.args[0]
            self.assertEqual(payload["auto_fallback_from_host"], "PinnedHost")
            self.assertEqual(payload["configured_host_provider"], "PinnedHost")
            self.assertEqual(payload["upstream_host_provider"], "TestHost")
            self.assertEqual(payload["provider_request_id"], "gen-test-0003")
        finally:
            await client.close()

    async def test_openrouter_auto_fallback_exhaustion_route_keeps_fallback_origin(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(side_effect=[
            _openrouter_response(
                404,
                json={"error": {"code": 404, "message": "No endpoints found for vendor/test-free:free."}},
            ),
            _openrouter_response(200, json=UPSTREAM_ERROR_BODY),
        ])
        client.MAX_RETRIES = 2
        try:
            with self.assertRaises(ProviderRouteError) as raised:
                await client.generate_completion(
                    model="vendor/test-free:free",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    provider="PinnedHost",
                    allow_provider_auto_fallback=True,
                    retry_callback=AsyncMock(),
                )
            route = raised.exception.route
            self.assertEqual(route.host_provider, "")
            self.assertEqual(route.auto_fallback_from_host, "PinnedHost")
            self.assertEqual(route.upstream_host_provider, "TestHost")
            self.assertNotIn("PinnedHost", str(raised.exception))
            self.assertNotIn("TestHost", str(raised.exception))
        finally:
            await client.close()

    async def test_openrouter_non_json_retry_reports_no_upstream_host(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(side_effect=[
            _openrouter_response(
                200,
                text="<html>gateway hiccup</html>",
                headers={"content-type": "text/html"},
            ),
            _success_response(),
        ])
        retry_callback = AsyncMock()
        try:
            await client.generate_completion(
                model="vendor/test-model",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                retry_callback=retry_callback,
            )
            payload = retry_callback.await_args.args[0]
            self.assertEqual(payload["reason"], "invalid_response")
            self.assertEqual(payload["configured_host_provider"], "")
            self.assertEqual(payload["upstream_host_provider"], "")
            self.assertEqual(payload["provider_request_id"], "")
        finally:
            await client.close()

    async def test_openrouter_http_error_retry_and_route_report_upstream_host(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(return_value=_openrouter_response(
            503,
            json={
                "id": "gen-test-0004",
                "error": {
                    "code": 503,
                    "message": "upstream unavailable",
                    "metadata": {"provider_name": "TestHost"},
                },
            },
        ))
        client.MAX_RETRIES = 2
        retry_callback = AsyncMock()
        try:
            with self.assertRaises(ProviderRouteError) as raised:
                await client.generate_completion(
                    model="vendor/test-model",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    provider="TestHost",
                    retry_callback=retry_callback,
                )
            payload = retry_callback.await_args.args[0]
            self.assertEqual(payload["reason"], "http_error")
            self.assertEqual(payload["configured_host_provider"], "TestHost")
            self.assertEqual(payload["upstream_host_provider"], "TestHost")
            self.assertEqual(payload["provider_request_id"], "gen-test-0004")
            route = raised.exception.route
            self.assertEqual(route.host_provider, "TestHost")
            self.assertEqual(route.upstream_host_provider, "TestHost")
            self.assertEqual(route.provider_request_id, "gen-test-0004")
        finally:
            await client.close()

    async def test_openrouter_paid_rate_limit_retry_reports_upstream_host(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(side_effect=[
            _openrouter_response(429, json=UPSTREAM_ERROR_BODY),
            _success_response(),
        ])
        retry_callback = AsyncMock()
        try:
            await client.generate_completion(
                model="vendor/test-paid",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                retry_callback=retry_callback,
            )
            payload = retry_callback.await_args.args[0]
            self.assertEqual(payload["reason"], "rate_limited")
            self.assertEqual(payload["upstream_host_provider"], "TestHost")
            self.assertEqual(payload["provider_request_id"], "gen-test-0001")
        finally:
            await client.close()

    async def test_openrouter_direct_retry_wait_receives_same_route_payload(self):
        client = OpenRouterClient("test-key")
        client.client.post = AsyncMock(side_effect=[
            _openrouter_response(200, json=UPSTREAM_ERROR_BODY),
            _success_response(),
        ])
        try:
            with patch(
                "backend.shared.openrouter_client.assured_retry_wait",
                new=AsyncMock(return_value=True),
            ) as retry_wait:
                await client.generate_completion(
                    model="vendor/test-free:free",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
            retry_wait.assert_awaited_once()
            payload = retry_wait.await_args.args[0]
            self.assertEqual(retry_wait.await_args.kwargs["role_id"], "openrouter_direct")
            self.assertEqual(payload["configured_host_provider"], "")
            self.assertEqual(payload["upstream_host_provider"], "TestHost")
            self.assertEqual(payload["provider_request_id"], "gen-test-0001")
            self.assertIn("upstream 429:", payload["error_summary"])
        finally:
            await client.close()

    async def test_openrouter_opaque_upstream_404_is_transient_after_internal_retries(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(
            404,
            request=request,
            json={
                "error": {
                    "message": "Provider returned error",
                    "code": 404,
                    "metadata": {"raw": "", "provider_name": "Nvidia"},
                }
            },
        )
        client.client.post = AsyncMock(return_value=response)
        client.MAX_RETRIES = 3
        retry_callback = AsyncMock()
        try:
            with patch(
                "backend.shared.openrouter_client.asyncio.sleep",
                new=AsyncMock(),
            ):
                with self.assertRaises(ProviderRouteError) as raised:
                    await client.generate_completion(
                        model="nvidia/model:free",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                        retry_callback=retry_callback,
                    )
            self.assertEqual(client.client.post.await_count, 3)
            self.assertEqual(retry_callback.await_count, 2)
            self.assertEqual(
                retry_callback.await_args_list[0].args[0]["reason"],
                "upstream_provider_error",
            )
            self.assertIn(
                "opaque HTTP 404",
                retry_callback.await_args_list[0].args[0]["error_summary"],
            )
            self.assertIn(
                "Nvidia",
                retry_callback.await_args_list[0].args[0]["error_summary"],
            )
            self.assertEqual(
                retry_callback.await_args_list[0].args[0]["upstream_host_provider"],
                "Nvidia",
            )
            self.assertEqual(raised.exception.route.upstream_host_provider, "Nvidia")
            self.assertTrue(is_transient_model_call_error(raised.exception))
            self.assertFalse(is_non_retryable_model_error(raised.exception))
        finally:
            await client.close()

    async def test_openrouter_retry_callback_exposes_redacted_http_error(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(
            503,
            request=request,
            json={
                "error": {
                    "message": "upstream unavailable",
                    "authorization": "Bearer secret-value",
                    "metadata": {
                        "messages": [
                            {
                                "role": "user",
                                "content": "private prompt ] still private",
                            }
                        ]
                    },
                }
            },
        )
        client.client.post = AsyncMock(return_value=response)
        client.MAX_RETRIES = 2
        retry_callback = AsyncMock()
        try:
            with patch(
                "backend.shared.openrouter_client.asyncio.sleep",
                new=AsyncMock(),
            ):
                with self.assertRaises(ProviderRouteError):
                    await client.generate_completion(
                        model="vendor/model",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                        retry_callback=retry_callback,
                    )
            retry_payload = retry_callback.await_args_list[0].args[0]
            self.assertEqual(retry_payload["reason"], "http_error")
            self.assertIn("upstream unavailable", retry_payload["error_summary"])
            self.assertNotIn("secret-value", retry_payload["error_summary"])
            self.assertNotIn("private prompt", retry_payload["error_summary"])
            self.assertIn("[redacted]", retry_payload["error_summary"])
        finally:
            await client.close()

    async def test_openrouter_provider_wrapper_with_specific_detail_is_not_marked_transient(self):
        client = OpenRouterClient("test-key")
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(
            404,
            request=request,
            json={
                "error": {
                    "message": "Provider returned error",
                    "code": 404,
                    "metadata": {
                        "raw": "invalid model deployment",
                        "provider_name": "Nvidia",
                    },
                }
            },
        )
        client.client.post = AsyncMock(return_value=response)
        client.MAX_RETRIES = 1
        try:
            with self.assertRaises(ProviderRouteError) as raised:
                await client.generate_completion(
                    model="nvidia/model:free",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
            self.assertFalse(is_transient_model_call_error(raised.exception))
        finally:
            await client.close()

    async def test_lm_studio_connection_failure_is_typed(self):
        client = LMStudioClient(base_url="http://127.0.0.1:1")
        client.client.post = AsyncMock(side_effect=httpx.ConnectError("offline"))
        try:
            with patch("backend.shared.lm_studio_client.asyncio.sleep", new=AsyncMock()):
                with self.assertRaises(ProviderRouteError) as raised:
                    await client.generate_completion(
                        model="local-model",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                        skip_semaphore=True,
                    )
            self.assertEqual(raised.exception.route.provider, "lm_studio")
            self.assertTrue(is_transient_model_call_error(raised.exception))
        finally:
            await client.client.aclose()

    async def test_lm_studio_timeout_is_typed(self):
        client = LMStudioClient(base_url="http://127.0.0.1:1")
        request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
        client.client.post = AsyncMock(side_effect=httpx.ReadTimeout("slow", request=request))
        try:
            with patch("backend.shared.lm_studio_client.asyncio.sleep", new=AsyncMock()):
                with self.assertRaises(ProviderRouteError) as raised:
                    await client.generate_completion(
                        model="local-model",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                        skip_semaphore=True,
                    )
            self.assertIsInstance(raised.exception.cause, httpx.TimeoutException)
        finally:
            await client.client.aclose()

    async def test_lm_studio_n_keep_context_rejection_is_typed_without_retry(self):
        client = LMStudioClient(base_url="http://127.0.0.1:1")
        request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
        response = httpx.Response(
            400,
            request=request,
            json={
                "error": (
                    "The number of tokens to keep from the initial prompt is greater "
                    "than the context length (n_keep: 57936>= n_ctx: 5632). "
                    "Bearer secret-value"
                )
            },
        )
        client.client.post = AsyncMock(return_value=response)
        retry_callback = AsyncMock()
        try:
            with self.assertRaises(ProviderContextLengthError) as raised:
                await client.generate_completion(
                    model="local-model",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    skip_semaphore=True,
                    retry_callback=retry_callback,
                )
            self.assertEqual(client.client.post.await_count, 1)
            retry_callback.assert_not_awaited()
            self.assertEqual(raised.exception.route.provider, "lm_studio")
            self.assertIn("5632", str(raised.exception))
            self.assertNotIn("secret-value", str(raised.exception))
        finally:
            await client.client.aclose()


class APIClientManagerRouteContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_ambiguous_codex_request_error_routes_to_existing_retry_path(self):
        manager = APIClientManager()
        manager.configure_role(
            "proof_identifier",
            ModelConfig(
                provider="openai_codex_oauth",
                model_id="gpt-5.5",
                context_window=4096,
                max_output_tokens=512,
            ),
        )
        codex_error = OpenAICodexRequestError(
            "OpenAI Codex completion failed: opaque backend rejection"
        )

        with (
            patch(
                "backend.shared.api_client_manager.openai_codex_client.generate_completion",
                new=AsyncMock(side_effect=codex_error),
            ),
            patch.object(
                manager,
                "_broadcast_unrecoverable_codex_error",
                new=AsyncMock(),
            ) as unrecoverable_notification,
        ):
            with self.assertRaises(RetryableProviderError) as raised:
                await manager._generate_completion_once(
                    task_id="proof_id_001",
                    role_id="proof_identifier",
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "hello"}],
                )

        self.assertEqual(raised.exception.provider, "openai_codex_oauth")
        unrecoverable_notification.assert_not_awaited()

    async def test_codex_auth_error_remains_repair_required(self):
        manager = APIClientManager()
        manager.configure_role(
            "proof_identifier",
            ModelConfig(
                provider="openai_codex_oauth",
                model_id="gpt-5.5",
                context_window=4096,
                max_output_tokens=512,
            ),
        )
        auth_error = OpenAICodexAuthError(
            "OpenAI Codex OAuth is not configured."
        )

        with (
            patch(
                "backend.shared.api_client_manager.openai_codex_client.generate_completion",
                new=AsyncMock(side_effect=auth_error),
            ),
            patch.object(
                manager,
                "_broadcast_unrecoverable_codex_error",
                new=AsyncMock(),
            ) as unrecoverable_notification,
        ):
            with self.assertRaises(ProviderRepairRequiredError):
                await manager._generate_completion_once(
                    task_id="proof_id_001",
                    role_id="proof_identifier",
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "hello"}],
                )

        unrecoverable_notification.assert_awaited_once()

    async def test_unconfigured_public_role_fails_before_routing_in_both_modes(self):
        previous_generic_mode = system_config.generic_mode
        previous_next_count = boost_manager.boost_next_count
        try:
            boost_manager.boost_next_count = 1
            for generic_mode in (False, True):
                with self.subTest(generic_mode=generic_mode):
                    system_config.generic_mode = generic_mode
                    manager = APIClientManager()
                    with (
                        patch.object(
                            manager,
                            "_maybe_add_assistant_memory_context",
                            new=AsyncMock(),
                        ) as assistant_context,
                        patch.object(
                            manager,
                            "_generate_completion_once",
                            new=AsyncMock(),
                        ) as routed_completion,
                        patch.object(
                            manager,
                            "_generate_supercharged_completion",
                            new=AsyncMock(),
                        ) as supercharged_completion,
                    ):
                        with self.assertRaises(ProviderRepairRequiredError) as raised:
                            await manager.generate_completion(
                                task_id="proof_novelty_001",
                                role_id="autonomous_proof_novelty_missing",
                                model="cloud-validator",
                                messages=[{"role": "user", "content": "hello"}],
                                max_tokens=10,
                            )

                    self.assertEqual(raised.exception.provider, "unconfigured")
                    self.assertEqual(raised.exception.reason, "role_not_configured")
                    self.assertEqual(
                        raised.exception.role_id,
                        "autonomous_proof_novelty_missing",
                    )
                    assistant_context.assert_not_awaited()
                    routed_completion.assert_not_awaited()
                    supercharged_completion.assert_not_awaited()
                    self.assertEqual(boost_manager.boost_next_count, 1)
        finally:
            system_config.generic_mode = previous_generic_mode
            boost_manager.boost_next_count = previous_next_count

    async def test_configured_lm_studio_primary_still_routes_from_public_boundary(self):
        manager = APIClientManager()
        manager.configure_role(
            "local_validator",
            ModelConfig(
                provider="lm_studio",
                model_id="local-model",
                context_window=4096,
                max_output_tokens=512,
            ),
        )
        expected = {"choices": [{"message": {"content": "{}"}}]}

        with patch.object(
            manager,
            "_generate_completion_once",
            new=AsyncMock(return_value=expected),
        ) as routed_completion:
            result = await manager.generate_completion(
                task_id="agg_val_001",
                role_id="local_validator",
                model="local-model",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
            )

        self.assertEqual(result, expected)
        routed_completion.assert_awaited_once()

    async def test_aggregator_submitter_waits_on_provider_cooldown_in_outer_loop(self):
        submitter = SubmitterAgent(
            submitter_id=3,
            model_name="fugu",
            user_prompt="test",
            user_files_content={},
        )
        cooldown = ProviderCooldownError(
            provider="sakana_fugu",
            provider_label="Sakana Fugu",
            role_id=submitter.role_id,
            model="fugu",
            resets_at=123,
            resets_in_seconds=3600,
        )
        submitter._generate_submission = AsyncMock(side_effect=cooldown)
        submitter.is_running = True

        async def stop_after_wait(*args, **kwargs):
            submitter.is_running = False

        with patch.object(
            api_manager_module.api_client_manager,
            "wait_for_retryable_provider_error",
            new=AsyncMock(side_effect=stop_after_wait),
        ) as wait_for_retry:
            await submitter._run_loop()

        wait_for_retry.assert_awaited_once()
        self.assertIs(wait_for_retry.await_args.args[0], cooldown)
        self.assertEqual(
            wait_for_retry.await_args.kwargs["role_id"],
            "aggregator_submitter_3",
        )

    async def test_configured_lm_studio_fallback_state_still_routes(self):
        manager = APIClientManager()
        manager.configure_role(
            "cloud_validator",
            ModelConfig(
                provider="openrouter",
                model_id="vendor/model",
                openrouter_model_id="vendor/model",
                lm_studio_fallback_id="local-model",
                context_window=4096,
                max_output_tokens=512,
            ),
        )
        manager._role_fallback_state["cloud_validator"] = "lm_studio"
        expected = {"choices": [{"message": {"content": "{}"}}]}

        with patch(
            "backend.shared.api_client_manager.lm_studio_client.generate_completion",
            new=AsyncMock(return_value=expected),
        ) as local_completion:
            result = await manager.generate_completion(
                task_id="agg_val_001",
                role_id="cloud_validator",
                model="vendor/model",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "{}")
        self.assertEqual(local_completion.await_args.kwargs["model"], "local-model")

    async def test_sakana_entitlement_failure_requires_repair_even_with_fallback(self):
        manager = APIClientManager()
        manager.configure_role(
            "validator",
            ModelConfig(
                provider="sakana_fugu",
                model_id="fugu",
                lm_studio_fallback_id="local-model",
                context_window=4096,
                max_output_tokens=512,
            ),
        )
        entitlement_error = SubscriptionEntitlementError(
            "subscription is not entitled",
            provider="sakana_fugu",
            provider_label="Sakana Fugu",
            error_code="not_entitled",
            status_code=403,
        )

        with (
            patch.object(
                get_api_key_subscription_client("sakana_fugu"),
                "generate_completion",
                new=AsyncMock(side_effect=entitlement_error),
            ),
            patch.object(
                manager,
                "_broadcast_unrecoverable_subscription_error",
                new=AsyncMock(),
            ),
        ):
            with self.assertRaises(ProviderRepairRequiredError) as raised:
                await manager._generate_completion_once(
                    task_id="agg_val_001",
                    role_id="validator",
                    model="fugu",
                    messages=[{"role": "user", "content": "hello"}],
                )

        self.assertEqual(raised.exception.provider, "sakana_fugu")
        self.assertEqual(raised.exception.reason, "entitlement_required")
        self.assertEqual(manager.get_fallback_state("validator"), "sakana_fugu")

    async def test_manager_normalizes_lm_studio_fallback_transport_failure(self):
        manager = APIClientManager()
        manager.configure_role(
            "writer",
            ModelConfig(
                provider="openrouter",
                model_id="vendor/model",
                openrouter_model_id="vendor/model",
                lm_studio_fallback_id="local-model",
                context_window=4096,
                max_output_tokens=512,
            ),
        )
        manager._role_fallback_state["writer"] = "lm_studio"

        base_error = ProviderRouteError(
            "LM Studio offline",
            route=ProviderRouteIdentity(provider="lm_studio", model="local-model"),
            cause=httpx.ConnectError("offline"),
        )
        with patch(
            "backend.shared.api_client_manager.lm_studio_client.generate_completion",
            new=AsyncMock(side_effect=base_error),
        ):
            with self.assertRaises(RetryableProviderError) as raised:
                await manager._generate_completion_once(
                    task_id="comp_writer_001",
                    role_id="writer",
                    model="local-model",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=100,
                )

        self.assertEqual(raised.exception.provider, "lm_studio")
        self.assertEqual(raised.exception.role_id, "writer")
        self.assertEqual(raised.exception.model, "local-model")
        self.assertEqual(raised.exception.reason, "transient_provider_error")

    async def test_reset_provider_fallbacks_restores_configured_cloud_roles(self):
        manager = APIClientManager()
        for provider in ("sakana_fugu", "openai_codex_oauth", "xai_grok_oauth"):
            role_id = f"{provider}_validator"
            manager.configure_role(
                role_id,
                ModelConfig(
                    provider=provider,
                    model_id=f"{provider}-model",
                    lm_studio_fallback_id="local-model",
                    context_window=4096,
                    max_output_tokens=512,
                ),
            )
            manager._role_fallback_state[role_id] = "lm_studio"

            reset = await manager.reset_provider_fallbacks(provider)

            self.assertEqual(reset, {role_id: provider})
            self.assertEqual(manager.get_fallback_state(role_id), provider)

    def test_provider_failure_classifier_covers_typed_and_lm_transport_errors(self):
        repair = api_manager_module.ProviderRepairRequiredError(
            provider="sakana_fugu",
            provider_label="Sakana Fugu",
            role_id="validator",
            model="fugu-model",
            reason="authentication",
            message="repair required",
        )
        route = ProviderRouteError(
            "LM Studio offline",
            route=ProviderRouteIdentity(provider="lm_studio", model="local-model"),
            cause=httpx.ConnectError("offline"),
        )

        self.assertTrue(APIClientManager.is_provider_failure(repair))
        self.assertTrue(APIClientManager.is_provider_failure(route))

    async def test_free_rotation_context_error_preserves_effective_route(self):
        manager = APIClientManager()
        manager._openrouter_client = AsyncMock()
        context_error = ProviderContextLengthError(
            "Bearer secret-value context_length_exceeded",
            route=ProviderRouteIdentity(provider="openrouter", model="alternate/free"),
        )
        manager._openrouter_client.generate_completion = AsyncMock(side_effect=context_error)

        with (
            patch.object(api_manager_module.free_model_manager, "looping_enabled", True),
            patch.object(api_manager_module.free_model_manager, "auto_selector_enabled", False),
            patch.object(
                api_manager_module.free_model_manager,
                "get_alternative_free_model",
                side_effect=["alternate/free", None],
            ),
        ):
            with self.assertRaises(ProviderContextLengthError) as raised:
                await manager._try_free_model_rotation(
                    task_id="agg_sub1_001",
                    role_id="submitter",
                    original_model="original/free",
                    configured_model="original/free",
                    configured_provider="openrouter",
                    messages=[{"role": "user", "content": "hello"}],
                    temperature=0.0,
                    max_tokens=10,
                    response_format=None,
                )

        self.assertEqual(raised.exception.route.route_kind, "free_rotation")
        self.assertEqual(raised.exception.route.model, "alternate/free")
        self.assertNotIn("secret-value", str(raised.exception))

    async def test_auto_selector_context_error_has_route_kind(self):
        manager = APIClientManager()
        manager._openrouter_client = AsyncMock()
        manager._openrouter_client.generate_completion = AsyncMock(
            side_effect=ProviderContextLengthError(
                "context limit",
                route=ProviderRouteIdentity(provider="openrouter", model="openrouter/free"),
            )
        )

        with (
            patch.object(api_manager_module.free_model_manager, "looping_enabled", False),
            patch.object(api_manager_module.free_model_manager, "auto_selector_enabled", True),
        ):
            with self.assertRaises(ProviderContextLengthError) as raised:
                await manager._try_free_model_rotation(
                    task_id="agg_sub1_001",
                    role_id="submitter",
                    original_model="original/free",
                    configured_model="original/free",
                    configured_provider="openrouter",
                    messages=[{"role": "user", "content": "hello"}],
                    temperature=0.0,
                    max_tokens=10,
                    response_format=None,
                )

        self.assertEqual(raised.exception.route.route_kind, "auto_selector")
        self.assertEqual(raised.exception.route.model, "openrouter/free")

    async def test_strict_boost_preserves_typed_context_error(self):
        manager = APIClientManager()
        previous_config = boost_manager.boost_config
        boost_manager.boost_config = BoostConfig(
            enabled=True,
            openrouter_api_key="test-key",
            boost_model_id="boost/model",
            boost_context_window=4096,
            boost_max_output_tokens=512,
        )
        boost_error = ProviderContextLengthError(
            "context limit",
            route=ProviderRouteIdentity(provider="openrouter", model="boost/model"),
        )
        fake_client = AsyncMock()
        fake_client.generate_completion = AsyncMock(side_effect=boost_error)
        try:
            with (
                patch.object(api_manager_module, "OpenRouterClient", return_value=fake_client),
                patch.object(
                    api_manager_module.boost_logger,
                    "log_boost_call",
                    new=AsyncMock(),
                ),
            ):
                with self.assertRaises(ProviderContextLengthError) as raised:
                    await manager._generate_completion_once(
                        task_id="comp_writer_001",
                        role_id="writer",
                        model="primary-model",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                        _moto_force_boost_mode="supercharge",
                        _moto_strict_boost=True,
                    )
            self.assertEqual(raised.exception.route.route_kind, "boost")
            self.assertEqual(raised.exception.route.model, "boost/model")
        finally:
            boost_manager.boost_config = previous_config


if __name__ == "__main__":
    unittest.main()
