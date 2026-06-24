import tempfile
import time
from unittest import IsolatedAsyncioTestCase, mock

from backend.shared.api_client_manager import APIClientManager, OAuthProviderCooldownError
from backend.shared.config import system_config
from backend.shared.models import ModelConfig
from backend.shared.openai_codex_client import OAuthUsageLimitError, OpenAICodexClient
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
        async def expire_after_sleep(_seconds: int) -> None:
            self.manager._oauth_provider_cooldowns["openai_codex_oauth"]["cooldown_until"] = int(time.time()) - 1

        with mock.patch(
            "backend.shared.api_client_manager.asyncio.sleep",
            new=mock.AsyncMock(side_effect=expire_after_sleep),
        ) as sleep_mock:
            await self.manager.wait_for_oauth_provider_cooldown(error, role_id="agg_sub1")
        sleep_mock.assert_awaited()

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
