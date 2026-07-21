import tempfile
import time
from unittest import IsolatedAsyncioTestCase, mock

from backend.shared.api_client_manager import (
    APIClientManager,
    OAuthProviderCooldownError,
    RetryableProviderError,
)
from backend.shared.config import system_config
from backend.shared.model_error_utils import is_retryable_model_output_error, is_transient_model_call_error
from backend.shared.models import ModelConfig
from backend.shared.openai_codex_client import OAuthUsageLimitError, OpenAICodexClient, OpenAICodexRequestError
from backend.shared.provider_notification_store import record_provider_notification
from backend.shared.proof_search.assistant_coordinator import (
    _assistant_oauth_provider_is_cooling_down,
)


class OAuthUsageLimitParsingTests(IsolatedAsyncioTestCase):
    def test_usage_limit_error_from_payload_extracts_reset_metadata(self) -> None:
        now = int(time.time())
        payload = {
            "type": "usage_limit_reached",
            "message": "The usage limit has been reached.",
            "plan_type": "plus",
            "resets_at": now + 3600,
            "resets_in_seconds": 3600,
        }
        error = OpenAICodexClient._usage_limit_error_from_payload(payload)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertEqual(error.provider, "openai_codex_oauth")
        self.assertEqual(error.plan_type, "plus")
        self.assertEqual(error.resets_at, now + 3600)
        self.assertEqual(error.resets_in_seconds, 3600)

    def test_usage_limit_error_from_nested_response_payload(self) -> None:
        payload = {
            "response": {
                "error": {
                    "type": "usage_limit_reached",
                    "message": "The usage limit has been reached.",
                    "resets_in_seconds": 120,
                }
            }
        }
        error = OpenAICodexClient._usage_limit_error_from_payload(payload)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertEqual(error.resets_in_seconds, 120)


class APIClientManagerOAuthCooldownTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.manager = APIClientManager()

    def test_mark_and_read_provider_cooldown(self) -> None:
        error = OAuthUsageLimitError(
            provider="openai_codex_oauth",
            provider_label="OpenAI Codex",
            message="The usage limit has been reached.",
            plan_type="plus",
            resets_in_seconds=90,
        )
        payload = self.manager._mark_oauth_provider_cooldown(
            error,
            role_id="agg_sub1",
            model="gpt-5.5",
        )
        self.assertTrue(self.manager.is_provider_cooling_down("openai_codex_oauth"))
        active = self.manager.get_provider_cooldown("openai_codex_oauth")
        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active["role_id"], "agg_sub1")
        self.assertEqual(active["model"], "gpt-5.5")
        self.assertEqual(active["reason"], "usage_limit_reached")
        self.assertGreaterEqual(active["resets_in_seconds"], 1)
        self.assertEqual(payload["provider_label"], "OpenAI Codex")

    def test_expired_provider_cooldown_is_cleared(self) -> None:
        self.manager._oauth_provider_cooldowns["openai_codex_oauth"] = {
            "provider": "openai_codex_oauth",
            "cooldown_until": int(time.time()) - 5,
            "resets_at": int(time.time()) - 5,
        }
        self.assertFalse(self.manager.is_provider_cooling_down("openai_codex_oauth"))
        self.assertIsNone(self.manager.get_provider_cooldown("openai_codex_oauth"))

    async def test_cooldown_lm_fallback_restores_to_codex_after_reset(self) -> None:
        self.manager.configure_role(
            "agg_sub1",
            ModelConfig(
                provider="openai_codex_oauth",
                model_id="gpt-5.5",
                lm_studio_fallback_id="local-fallback",
                context_window=128000,
                max_output_tokens=8192,
            ),
        )
        self.manager._role_fallback_state["agg_sub1"] = "lm_studio"
        self.manager._oauth_cooldown_fallback_roles.add("agg_sub1")
        self.manager._oauth_provider_cooldowns["openai_codex_oauth"] = {
            "provider": "openai_codex_oauth",
            "cooldown_until": int(time.time()) - 5,
            "resets_at": int(time.time()) - 5,
        }

        with mock.patch(
            "backend.shared.api_client_manager.openai_codex_client.generate_completion",
            new=mock.AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]}),
        ):
            await self.manager._generate_completion_once(
                task_id="agg_sub1_001",
                role_id="agg_sub1",
                model="gpt-5.5",
                messages=[{"role": "user", "content": "hello"}],
            )
        self.assertEqual(self.manager._role_fallback_state["agg_sub1"], "openai_codex_oauth")
        self.assertNotIn("agg_sub1", self.manager._oauth_cooldown_fallback_roles)

    async def test_codex_output_truncation_bypasses_fallback_and_unrecoverable_notification(self) -> None:
        role_id = "autonomous_proof_formalization_brainstorm"
        self.manager.configure_role(
            role_id,
            ModelConfig(
                provider="openai_codex_oauth",
                model_id="gpt-5.5",
                lm_studio_fallback_id="local-fallback",
                context_window=128000,
                max_output_tokens=8192,
            ),
        )
        truncation_error = OpenAICodexRequestError(
            "OpenAI Codex completion failed: "
            '{"type":"response.incomplete","response":{"status":"incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"}}}'
        )

        with (
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch(
                "backend.shared.api_client_manager.openai_codex_client.generate_completion",
                new=mock.AsyncMock(side_effect=truncation_error),
            ),
            mock.patch.object(
                self.manager,
                "_broadcast_unrecoverable_codex_error",
                new=mock.AsyncMock(),
            ) as unrecoverable_broadcast,
        ):
            with self.assertRaises(OpenAICodexRequestError) as ctx:
                await self.manager.generate_completion(
                    task_id="proof_form_000",
                    role_id=role_id,
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "prove"}],
                )

        self.assertTrue(is_retryable_model_output_error(ctx.exception))
        self.assertEqual(self.manager._role_fallback_state[role_id], "openai_codex_oauth")
        unrecoverable_broadcast.assert_not_awaited()

    async def test_wait_for_oauth_provider_cooldown_sleeps_until_expired(self) -> None:
        self.manager._oauth_provider_cooldowns["openai_codex_oauth"] = {
            "provider": "openai_codex_oauth",
            "cooldown_until": int(time.time()) + 300,
            "resets_at": int(time.time()) + 300,
            "resets_in_seconds": 1,
        }
        error = OAuthProviderCooldownError(
            provider="openai_codex_oauth",
            provider_label="OpenAI Codex",
            role_id="agg_sub1",
            model="gpt-5.5",
            resets_at=int(time.time()) + 300,
            resets_in_seconds=1,
        )
        async def expire_after_sleep(_seconds: int, _should_stop=None) -> None:
            self.manager._oauth_provider_cooldowns["openai_codex_oauth"]["cooldown_until"] = int(time.time()) - 1

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(side_effect=expire_after_sleep),
        ) as sleep_mock:
            await self.manager.wait_for_oauth_provider_cooldown(error, role_id="agg_sub1")
        sleep_mock.assert_awaited()

    async def test_retryable_provider_error_uses_shared_exponential_backoff(self) -> None:
        error = RetryableProviderError(
            provider="openai_codex_oauth",
            provider_label="OpenAI Codex",
            role_id="autonomous_proof_formalization_brainstorm",
            model="gpt-5.5",
            reason="transient_provider_error",
            message="OpenAI Codex transient transport failure",
        )

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(),
        ) as sleep_mock:
            await self.manager.wait_for_retryable_provider_error(error)
            await self.manager.wait_for_retryable_provider_error(error)

        sleep_mock.assert_has_awaits([
            mock.call(60, None),
            mock.call(120, None),
        ])

    async def test_retryable_provider_backoff_clears_after_success_with_display_role_override(self) -> None:
        error = RetryableProviderError(
            provider="openrouter",
            provider_label="OpenRouter",
            role_id="agg_val",
            model="openrouter/model",
            reason="transient_provider_error",
            message="OpenRouter transient provider failure",
        )

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(),
        ) as sleep_mock:
            await self.manager.wait_for_retryable_provider_error(
                error,
                role_id="aggregator_validator",
            )
            self.manager._clear_retryable_provider_backoff(
                "openrouter",
                "agg_val",
                "openrouter/model",
            )
            await self.manager.wait_for_retryable_provider_error(
                error,
                role_id="aggregator_validator",
            )

        sleep_mock.assert_has_awaits([
            mock.call(60, None),
            mock.call(60, None),
        ])

    async def test_retryable_provider_backoff_stop_returns_without_cancellation(self) -> None:
        error = RetryableProviderError(
            provider="openrouter",
            provider_label="OpenRouter",
            role_id="agg_val",
            model="openrouter/model",
            reason="transient_provider_error",
            message="OpenRouter transient provider failure",
        )

        await self.manager.wait_for_retryable_provider_error(
            error,
            should_stop=lambda: True,
        )

        self.assertEqual(
            self.manager._retryable_provider_backoff_state[
                "openrouter|agg_val|openrouter/model|transient_provider_error"
            ],
            1,
        )

    def test_transient_classifier_does_not_retry_sakana_auth_failure(self) -> None:
        error = RuntimeError("Sakana Fugu request failed: HTTP 401: invalid api key")

        self.assertFalse(is_transient_model_call_error(error))

    def test_transient_classifier_retries_sakana_server_failure(self) -> None:
        error = RuntimeError("Sakana Fugu request failed: HTTP 503: service unavailable")

        self.assertTrue(is_transient_model_call_error(error))

    def test_provider_notification_preserves_cooldown_metadata(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                stored = record_provider_notification(
                    "oauth_provider_usage_limited",
                    {
                        "provider": "openai_codex_oauth",
                        "provider_label": "OpenAI Codex",
                        "role_id": "agg_sub1",
                        "model": "gpt-5.5",
                        "reason": "usage_limit_reached",
                        "recoverable": True,
                        "message": "Codex cooling down.",
                        "resets_at": 12345,
                        "cooldown_until": 12345,
                        "resets_in_seconds": 60,
                        "fallback_model": "local-fallback",
                        "plan_type": "plus",
                    },
                )
            finally:
                system_config.data_dir = old_data_dir

        self.assertEqual(stored["resets_at"], 12345)
        self.assertEqual(stored["cooldown_until"], 12345)
        self.assertEqual(stored["resets_in_seconds"], 60)
        self.assertEqual(stored["fallback_model"], "local-fallback")
        self.assertEqual(stored["plan_type"], "plus")
        self.assertIn("12345", stored["notification_key"])


class AssistantOAuthCooldownReuseTests(IsolatedAsyncioTestCase):
    def test_assistant_oauth_cooling_down_reflects_manager_state(self) -> None:
        from backend.shared.api_client_manager import api_client_manager

        api_client_manager.configure_role(
            "aggregator_assistant",
            ModelConfig(
                provider="openai_codex_oauth",
                model_id="gpt-5.5",
                context_window=128000,
                max_output_tokens=8192,
            ),
        )
        api_client_manager._oauth_provider_cooldowns["openai_codex_oauth"] = {
            "provider": "openai_codex_oauth",
            "cooldown_until": int(time.time()) + 120,
            "resets_at": int(time.time()) + 120,
            "resets_in_seconds": 120,
        }
        try:
            self.assertTrue(_assistant_oauth_provider_is_cooling_down("aggregator_assistant"))
        finally:
            api_client_manager._oauth_provider_cooldowns.pop("openai_codex_oauth", None)
