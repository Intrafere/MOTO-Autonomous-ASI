import unittest
from unittest.mock import AsyncMock, patch

import httpx

import backend.shared.api_client_manager as api_manager_module
from backend.shared.api_client_manager import (
    APIClientManager,
    RetryableProviderError,
    _is_retryable_codex_completion_error,
    _typed_provider_context_error,
)
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
from backend.shared.openrouter_client import OpenRouterClient
from backend.shared.provider_errors import (
    ProviderContextLengthError,
    ProviderRepairRequiredError,
    ProviderRouteError,
    ProviderRouteIdentity,
)
from backend.shared.sakana_fugu_client import SakanaFuguEntitlementError


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
        entitlement_error = SakanaFuguEntitlementError(
            "subscription is not entitled",
            error_code="not_entitled",
            status_code=403,
        )

        with (
            patch(
                "backend.shared.api_client_manager.sakana_fugu_client.generate_completion",
                new=AsyncMock(side_effect=entitlement_error),
            ),
            patch.object(
                manager,
                "_broadcast_unrecoverable_sakana_fugu_error",
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
