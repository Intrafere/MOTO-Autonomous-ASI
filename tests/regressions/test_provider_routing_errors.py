import unittest
from unittest.mock import AsyncMock, patch

import httpx

import backend.shared.api_client_manager as api_manager_module
from backend.shared.api_client_manager import APIClientManager, _typed_provider_context_error
from backend.shared.boost_manager import boost_manager
from backend.shared.lm_studio_client import LMStudioClient
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_provider_context_length_error,
    is_transient_model_call_error,
)
from backend.shared.models import BoostConfig, ModelConfig
from backend.shared.openrouter_client import OpenRouterClient
from backend.shared.provider_errors import (
    ProviderContextLengthError,
    ProviderRouteError,
    ProviderRouteIdentity,
)


class ProviderErrorUtilityTests(unittest.TestCase):
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
    async def test_manager_adds_role_task_and_fallback_route_identity(self):
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
            with self.assertRaises(ProviderRouteError) as raised:
                await manager._generate_completion_once(
                    task_id="comp_writer_001",
                    role_id="writer",
                    model="local-model",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=100,
                )

        route = raised.exception.route
        self.assertEqual(route.role_id, "writer")
        self.assertEqual(route.task_id, "comp_writer_001")
        self.assertEqual(route.route_kind, "fallback")

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
