import tempfile
import time
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

import httpx

from backend.shared.api_client_manager import (
    APIClientManager,
    OAuthProviderCooldownError,
    ProviderCooldownError,
    RetryableProviderError,
    _active_notification_workflow_mode,
    _retry_workflow_mode,
)
from backend.shared.config import system_config
from backend.shared.model_error_utils import is_retryable_model_output_error, is_transient_model_call_error
from backend.shared.models import ModelConfig
from backend.shared.openai_codex_client import (
    OAuthUsageLimitError,
    OpenAICodexClient,
    OpenAICodexRequestError,
    OpenAICodexUntimedUsageLimitError,
)
from backend.shared.openrouter_client import RateLimitError
from backend.shared.provider_errors import ProviderRouteError, ProviderRouteIdentity
from backend.shared.provider_notification_store import record_provider_notification
from backend.shared.sakana_fugu_client import SakanaFuguUsageLimitError
from backend.shared.proof_search.assistant_coordinator import (
    _assistant_oauth_provider_is_cooling_down,
    _assistant_provider_is_cooling_down,
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

    def test_usage_limit_without_reset_is_not_treated_as_timed_cooldown(self) -> None:
        payload = {
            "type": "usage_limit_reached",
            "message": "The usage limit has been reached.",
        }
        self.assertIsInstance(
            OpenAICodexClient._usage_limit_error_from_payload(payload),
            OpenAICodexUntimedUsageLimitError,
        )
        self.assertIsInstance(
            OpenAICodexClient._usage_limit_error_from_text(
                '{"type":"usage_limit_reached","message":"The usage limit has been reached."}'
            ),
            OpenAICodexUntimedUsageLimitError,
        )


class APIClientManagerOAuthCooldownTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Retry activity is persisted by the provider notification store; keep
        # it out of the real backend/data root.
        self._data_dir = tempfile.TemporaryDirectory()
        self._old_data_dir = system_config.data_dir
        system_config.data_dir = self._data_dir.name
        # Session History Memory is not under test here; background Assistant
        # retrieval would otherwise outlive the temporary data root.
        memory_patch = mock.patch.object(system_config, "agent_conversation_memory_enabled", False)
        memory_patch.start()
        self.addCleanup(memory_patch.stop)
        self.manager = APIClientManager()

    def tearDown(self) -> None:
        system_config.data_dir = self._old_data_dir
        self._data_dir.cleanup()

    def test_manual_compiler_proof_only_owner_maps_to_manual_proof_check(self) -> None:
        with mock.patch(
            "backend.shared.api_client_manager.workflow_start_guard",
            new=SimpleNamespace(active_owner="manual_compiler_proof_only"),
        ):
            self.assertEqual(_active_notification_workflow_mode(), "manual_proof_check")
            self.assertEqual(_retry_workflow_mode("compiler_writer"), "manual_proof_check")

    def test_manual_proof_retry_roles_use_manual_proof_activity_scope(self) -> None:
        role_ids = (
            "autonomous_proof_identification_manual_aggregator_run_123",
            "autonomous_proof_formalization_manual_compiler_run_123",
            "autonomous_proof_prune_proposer_manual_brainstorm_run_123",
            "manual_proof_assistant",
        )

        with mock.patch(
            "backend.shared.api_client_manager._active_notification_workflow_mode",
            return_value="",
        ):
            self.assertEqual(
                [_retry_workflow_mode(role_id) for role_id in role_ids],
                ["manual_proof_check"] * len(role_ids),
            )

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

    async def test_sakana_usage_limit_uses_timed_fallback_and_restores(self) -> None:
        role_id = "agg_sub1"
        self.manager.configure_role(
            role_id,
            ModelConfig(
                provider="sakana_fugu",
                model_id="fugu",
                lm_studio_fallback_id="local-fallback",
                context_window=128000,
                max_output_tokens=8192,
            ),
        )
        usage_error = SakanaFuguUsageLimitError(
            "usage limit reached",
            resets_in_seconds=120,
            plan_type="pro",
        )

        with (
            mock.patch(
                "backend.shared.api_client_manager.sakana_fugu_client.generate_completion",
                new=mock.AsyncMock(side_effect=usage_error),
            ),
            mock.patch(
                "backend.shared.api_client_manager.lm_studio_client.generate_completion",
                new=mock.AsyncMock(return_value={"choices": [{"message": {"content": "fallback"}}]}),
            ),
        ):
            await self.manager._generate_completion_once(
                task_id="agg_sub1_001",
                role_id=role_id,
                model="fugu",
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(self.manager._role_fallback_state[role_id], "lm_studio")
        self.assertIn(role_id, self.manager._oauth_cooldown_fallback_roles)
        self.assertTrue(self.manager.is_provider_cooling_down("sakana_fugu"))

        self.manager._oauth_provider_cooldowns["sakana_fugu"]["cooldown_until"] = int(time.time()) - 1
        with (
            mock.patch(
                "backend.shared.api_client_manager.sakana_fugu_client.generate_completion",
                new=mock.AsyncMock(return_value={"choices": [{"message": {"content": "restored"}}]}),
            ),
            mock.patch.object(
                self.manager,
                "_broadcast",
                new=mock.AsyncMock(),
            ) as broadcast,
        ):
            await self.manager._generate_completion_once(
                task_id="agg_sub1_002",
                role_id=role_id,
                model="fugu",
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(self.manager._role_fallback_state[role_id], "sakana_fugu")
        self.assertNotIn(role_id, self.manager._oauth_cooldown_fallback_roles)
        self.assertTrue(any(
            call.args and call.args[0] == "provider_usage_limit_resumed"
            for call in broadcast.await_args_list
        ))

    async def test_sakana_preflight_without_fallback_raises_stoppable_cooldown(self) -> None:
        role_id = "agg_sub1"
        self.manager.configure_role(
            role_id,
            ModelConfig(
                provider="sakana_fugu",
                model_id="fugu",
                context_window=128000,
                max_output_tokens=8192,
            ),
        )
        self.manager._oauth_provider_cooldowns["sakana_fugu"] = {
            "provider": "sakana_fugu",
            "provider_label": "Sakana Fugu",
            "cooldown_until": int(time.time()) + 120,
            "resets_at": int(time.time()) + 120,
            "resets_in_seconds": 120,
        }

        with self.assertRaises(ProviderCooldownError) as raised:
            await self.manager._generate_completion_once(
                task_id="agg_sub1_001",
                role_id=role_id,
                model="fugu",
                messages=[{"role": "user", "content": "hello"}],
            )
        self.assertEqual(raised.exception.provider, "sakana_fugu")
        self.assertIsInstance(raised.exception, OAuthProviderCooldownError)

    async def test_cooldown_wait_does_not_claim_resume_before_provider_success(self) -> None:
        reset_at = int(time.time()) - 1
        self.manager._oauth_provider_cooldowns["sakana_fugu"] = {
            "provider": "sakana_fugu",
            "provider_label": "Sakana Fugu",
            "cooldown_until": reset_at,
            "resets_at": reset_at,
            "workflow_mode": "aggregator",
        }
        error = ProviderCooldownError(
            provider="sakana_fugu",
            provider_label="Sakana Fugu",
            role_id="agg_sub1",
            model="fugu",
            resets_at=reset_at,
            resets_in_seconds=1,
        )
        with (
            mock.patch.object(
                self.manager,
                "is_provider_cooling_down",
                side_effect=[True, False],
            ),
            mock.patch.object(
                self.manager,
                "get_provider_cooldown",
                return_value={
                    "provider": "sakana_fugu",
                    "provider_label": "Sakana Fugu",
                    "cooldown_until": reset_at,
                    "resets_at": reset_at,
                    "resets_in_seconds": 1,
                    "workflow_mode": "aggregator",
                },
            ),
            mock.patch.object(
                self.manager,
                "_sleep_with_optional_stop",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(self.manager, "_broadcast", new=mock.AsyncMock()) as broadcast,
        ):
            await self.manager.wait_for_provider_cooldown(error, role_id="agg_sub1")
        self.assertFalse(any(
            call.args and call.args[0] == "provider_usage_limit_resumed"
            for call in broadcast.await_args_list
        ))
        self.assertIn(("sakana_fugu", "agg_sub1", "fugu"), self.manager._provider_resume_pending)

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

    async def test_provider_cooldown_wait_returns_immediately_after_stop(self) -> None:
        reset_at = int(time.time()) + 3600
        self.manager._oauth_provider_cooldowns["openai_codex_oauth"] = {
            "provider": "openai_codex_oauth",
            "cooldown_until": reset_at,
            "resets_at": reset_at,
        }
        error = OAuthProviderCooldownError(
            provider="openai_codex_oauth",
            provider_label="OpenAI Codex",
            role_id="agg_sub1",
            model="gpt-5.5",
            resets_at=reset_at,
            resets_in_seconds=3600,
        )

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(),
        ) as sleep_mock:
            await self.manager.wait_for_oauth_provider_cooldown(
                error,
                role_id="agg_sub1",
                should_stop=lambda: True,
            )

        sleep_mock.assert_not_awaited()
        self.assertNotIn(
            ("openai_codex_oauth", "agg_sub1", "gpt-5.5"),
            self.manager._provider_resume_pending,
        )

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

    async def test_retryable_provider_error_reports_wait_and_resume_activity(self) -> None:
        error = RetryableProviderError(
            provider="openai_codex_oauth",
            provider_label="OpenAI Codex",
            role_id="autonomous_proof_formalization_brainstorm",
            model="gpt-5.5",
            reason="transient_provider_error",
            message="OpenAI Codex transient transport failure",
        )
        activity_callback = mock.AsyncMock()
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(),
        ):
            await self.manager.wait_for_retryable_provider_error(
                error,
                activity_callback=activity_callback,
            )

        self.assertEqual(
            [call.args[0] for call in activity_callback.await_args_list],
            ["waiting", "resuming"],
        )
        waiting_payload = activity_callback.await_args_list[0].args[1]
        self.assertEqual(waiting_payload["retry_attempt"], 1)
        self.assertEqual(waiting_payload["retry_after_seconds"], 60)
        self.assertEqual(waiting_payload["provider_label"], "OpenAI Codex")
        self.assertIn("OpenAI Codex API error", waiting_payload["message"])
        self.assertIn("60-second cooldown", waiting_payload["message"])
        retry_events = [
            call for call in broadcast.await_args_list
            if call.args and call.args[0] == "provider_retry_cooldown"
        ]
        self.assertEqual(len(retry_events), 1)
        retry_payload = retry_events[0].args[1]
        self.assertEqual(retry_payload["event_type"], "provider_retry_cooldown")
        self.assertTrue(
            retry_payload["event_id"].startswith("provider_retry_cooldown:")
        )
        self.assertEqual(retry_payload["retry_after_seconds"], 60.0)
        self.assertEqual(retry_payload["retry_attempt"], 1)
        self.assertEqual(retry_payload["reason"], "transient_provider_error")
        self.assertEqual(
            retry_payload["error_summary"],
            "OpenAI Codex transient transport failure",
        )
        self.assertIn("OpenAI Codex API error", retry_payload["message"])
        self.assertIn("60-second cooldown", retry_payload["message"])
        self.assertEqual(retry_payload["message"], waiting_payload["message"])
        self.assertEqual(
            waiting_payload["route_identity"],
            "[Autonomous · Proof Formalization (brainstorm) · gpt-5.5 via OpenAI Codex]",
        )
        self.assertTrue(retry_payload["message"].startswith(waiting_payload["route_identity"]))
        self.assertEqual(retry_payload["host_routing"], "direct")
        self.assertNotIn("[Autonomous", retry_payload["error_summary"])

    async def test_retry_attempts_share_correlation_but_have_unique_event_ids(self) -> None:
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)
        callback = self.manager._provider_retry_callback("agg_val")

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(return_value=True),
        ):
            await callback({
            "provider": "openrouter",
            "provider_label": "OpenRouter",
            "role_id": "agg_val",
            "model": "vendor/model",
            "retry_attempt": 1,
            "retry_after_seconds": 2,
            "reason": "http_error",
            })
            await callback({
            "provider": "openrouter",
            "provider_label": "OpenRouter",
            "role_id": "agg_val",
            "model": "vendor/model",
            "retry_attempt": 2,
            "retry_after_seconds": 4,
            "reason": "http_error",
            })

        first = broadcast.await_args_list[0].args[1]
        second = broadcast.await_args_list[1].args[1]
        self.assertEqual(first["correlation_id"], second["correlation_id"])
        self.assertNotEqual(first["event_id"], second["event_id"])
        self.assertEqual(first["contract_version"], 1)
        self.assertEqual(second["retry_attempt"], 2)

    async def test_separate_retry_episodes_receive_distinct_correlations(self) -> None:
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)
        payload = {
            "provider": "openrouter",
            "provider_label": "OpenRouter",
            "role_id": "agg_val",
            "model": "vendor/model",
            "retry_attempt": 1,
            "retry_after_seconds": 2,
            "reason": "http_error",
        }

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(return_value=True),
        ):
            await self.manager.emit_and_wait_for_retry(payload)
            await self.manager.emit_and_wait_for_retry(payload)

        first = broadcast.await_args_list[0].args[1]
        second = broadcast.await_args_list[1].args[1]
        self.assertNotEqual(first["correlation_id"], second["correlation_id"])
        self.assertNotEqual(first["event_id"], second["event_id"])

    async def test_retry_activity_redacts_error_summary_at_broadcast_boundary(self) -> None:
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)

        await self.manager._broadcast_provider_retry_cooldown({
            "provider": "openrouter",
            "provider_label": "OpenRouter",
            "role_id": "agg_val",
            "model": "vendor/model",
            "retry_attempt": 1,
            "retry_after_seconds": 60,
            "reason": "http_error",
            "error_summary": "HTTP 503 Bearer secret-value upstream unavailable",
        })

        retry_payload = broadcast.await_args.args[1]
        self.assertIn("HTTP 503", retry_payload["error_summary"])
        self.assertIn("[redacted]", retry_payload["error_summary"])
        self.assertNotIn("secret-value", str(retry_payload))
        self.assertIn("OpenRouter server-side error", retry_payload["message"])

    def _configure_openrouter_role(self, role_id: str, *, host: str | None = None) -> None:
        self.manager.configure_role(
            role_id,
            ModelConfig(
                provider="openrouter",
                model_id="vendor/test-free:free",
                openrouter_model_id="vendor/test-free:free",
                openrouter_provider=host,
                context_window=32000,
                max_output_tokens=4000,
            ),
        )

    async def test_adapter_retry_broadcast_carries_route_identity_and_prefixed_message(self) -> None:
        self._configure_openrouter_role("aggregator_submitter_2")
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)
        callback = self.manager._provider_retry_callback(
            "aggregator_submitter_2",
            task_id="agg_sub2_001",
            configured_model="vendor/test-free:free",
            configured_provider="openrouter",
            configured_host_provider="",
        )

        with (
            mock.patch(
                "backend.shared.api_client_manager._active_notification_workflow_mode",
                return_value="autonomous",
            ),
            mock.patch.object(
                self.manager,
                "_sleep_with_optional_stop",
                new=mock.AsyncMock(return_value=True),
            ),
        ):
            await callback({
                "provider": "openrouter",
                "provider_label": "OpenRouter",
                "model": "vendor/test-free:free",
                "retry_attempt": 1,
                "max_attempts": 3,
                "retry_after_seconds": 2,
                "reason": "invalid_response",
                "error_summary": "OpenRouter returned a successful response without completion choices.",
                "upstream_host_provider": "TestHost",
                "provider_request_id": "gen-test-1",
            })

        retry_payload = broadcast.await_args.args[1]
        identity = (
            "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · "
            "host auto → TestHost]"
        )
        self.assertEqual(retry_payload["workflow_mode"], "autonomous")
        self.assertEqual(retry_payload["role_id"], "aggregator_submitter_2")
        self.assertEqual(retry_payload["role_label"], "Submitter 2")
        self.assertEqual(retry_payload["model"], "vendor/test-free:free")
        self.assertEqual(retry_payload["provider"], "openrouter")
        self.assertEqual(retry_payload["route_kind"], "primary")
        self.assertEqual(retry_payload["task_id"], "agg_sub2_001")
        self.assertEqual(retry_payload["configured_host_provider"], "")
        self.assertEqual(retry_payload["effective_host_provider"], "TestHost")
        self.assertEqual(retry_payload["host_routing"], "auto")
        self.assertEqual(retry_payload["provider_request_id"], "gen-test-1")
        self.assertEqual(retry_payload["route_identity"], identity)
        self.assertEqual(
            retry_payload["message"],
            f"{identity} OpenRouter server-side error: OpenRouter returned a successful "
            "response without completion choices. OpenRouter API call will retry after "
            "a 2-second cooldown (retry 1).",
        )
        self.assertEqual(
            retry_payload["error_summary"],
            "OpenRouter returned a successful response without completion choices.",
        )
        self.assertNotIn("TestHost", retry_payload["event_id"])
        self.assertNotIn("TestHost", retry_payload["correlation_id"])

    async def test_retry_broadcast_backfills_pinned_host_from_role_config(self) -> None:
        self._configure_openrouter_role("aggregator_validator", host="TestHost")
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)

        with mock.patch(
            "backend.shared.api_client_manager._active_notification_workflow_mode",
            return_value="aggregator",
        ):
            await self.manager._broadcast_provider_retry_cooldown({
                "provider": "openrouter",
                "provider_label": "OpenRouter",
                "role_id": "aggregator_validator",
                "retry_attempt": 1,
                "retry_after_seconds": 4,
                "reason": "http_error",
            })

        retry_payload = broadcast.await_args.args[1]
        self.assertEqual(retry_payload["model"], "vendor/test-free:free")
        self.assertEqual(retry_payload["configured_model"], "vendor/test-free:free")
        self.assertEqual(retry_payload["configured_host_provider"], "TestHost")
        self.assertEqual(retry_payload["host_routing"], "pinned")
        self.assertTrue(retry_payload["message"].startswith(
            "[Aggregator · Validator · vendor/test-free:free via OpenRouter · "
            "host TestHost (pinned)] OpenRouter API call will retry"
        ))

    async def test_workflow_loop_retry_names_mode_and_role_only(self) -> None:
        self._configure_openrouter_role("aggregator_submitter_1")
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)

        with (
            mock.patch(
                "backend.shared.api_client_manager._active_notification_workflow_mode",
                return_value="aggregator",
            ),
            mock.patch.object(
                self.manager,
                "_sleep_with_optional_stop",
                new=mock.AsyncMock(return_value=True),
            ),
        ):
            await self.manager.emit_and_wait_for_retry(
                {
                    "provider": "unknown",
                    "provider_label": "Aggregator submitter 1",
                    "role_id": "aggregator_submitter_1",
                    "retry_attempt": 1,
                    "retry_after_seconds": 5,
                    "reason": "submitter_loop_retry",
                    "error_summary": "loop failure",
                },
                role_id="aggregator_submitter_1",
            )

        retry_payload = broadcast.await_args.args[1]
        self.assertEqual(retry_payload["provider"], "unknown")
        self.assertEqual(retry_payload["model"], "")
        self.assertNotIn("host_routing", retry_payload)
        self.assertEqual(retry_payload["route_identity"], "[Aggregator · Submitter 1]")
        self.assertTrue(retry_payload["message"].startswith("[Aggregator · Submitter 1] "))

    async def test_emit_with_error_fills_identity_without_overriding_producer_keys(self) -> None:
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)
        error = ProviderRouteError(
            "OpenRouter connection failed",
            route=ProviderRouteIdentity(
                provider="openrouter",
                model="vendor/test-free:free",
                role_id="compiler_writer",
                host_provider="TestHost",
            ),
            cause=httpx.ConnectError("connection reset"),
        )

        with mock.patch.object(
            self.manager,
            "_sleep_with_optional_stop",
            new=mock.AsyncMock(return_value=True),
        ):
            await self.manager.emit_and_wait_for_retry(
                {
                    "provider": "configured_role",
                    "provider_label": "Compiler writer",
                    "role_id": "compiler_critique",
                    "workflow_mode": "compiler",
                    "retry_attempt": 1,
                    "retry_after_seconds": 5,
                    "reason": "compiler_retry",
                },
                role_id="compiler_critique",
                error=error,
            )

        retry_payload = broadcast.await_args.args[1]
        self.assertEqual(retry_payload["workflow_mode"], "compiler")
        self.assertEqual(retry_payload["role_id"], "compiler_critique")
        self.assertEqual(retry_payload["model_role_id"], "compiler_writer")
        self.assertEqual(retry_payload["role_label"], "Writing Submitter")
        self.assertEqual(retry_payload["provider"], "openrouter")
        self.assertEqual(retry_payload["provider_label"], "OpenRouter")
        self.assertEqual(retry_payload["model"], "vendor/test-free:free")
        self.assertEqual(retry_payload["host_routing"], "pinned")
        self.assertTrue(retry_payload["message"].startswith(
            "[Compiler · Writing Submitter · vendor/test-free:free via OpenRouter · "
            "host TestHost (pinned)] OpenRouter API call will retry"
        ))

    async def test_display_role_override_keeps_backoff_key_and_names_model_role(self) -> None:
        self._configure_openrouter_role("aggregator_submitter_2")
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)
        activity_callback = mock.AsyncMock()
        error = RetryableProviderError(
            provider="openrouter",
            provider_label="OpenRouter",
            role_id="aggregator_submitter_2",
            model="vendor/test-free:free",
            reason="transient_provider_error",
            message="OpenRouter transient provider failure",
            route_kind="primary",
            configured_host_provider="",
            upstream_host_provider="TestHost",
            provider_request_id="gen-test-2",
        )

        with (
            mock.patch(
                "backend.shared.api_client_manager._active_notification_workflow_mode",
                return_value="aggregator",
            ),
            mock.patch.object(
                self.manager,
                "_sleep_with_optional_stop",
                new=mock.AsyncMock(return_value=True),
            ),
        ):
            await self.manager.wait_for_retryable_provider_error(
                error,
                role_id="aggregator_single_model",
                activity_callback=activity_callback,
            )

        waiting_payload = activity_callback.await_args_list[0].args[1]
        retry_payload = broadcast.await_args.args[1]
        identity = (
            "[Aggregator · Submitter 2 · vendor/test-free:free via OpenRouter · "
            "host auto → TestHost]"
        )
        self.assertEqual(waiting_payload["route_identity"], identity)
        self.assertEqual(waiting_payload["role_id"], "aggregator_single_model")
        self.assertEqual(waiting_payload["model_role_id"], "aggregator_submitter_2")
        self.assertEqual(retry_payload["message"], waiting_payload["message"])
        self.assertTrue(retry_payload["message"].startswith(f"{identity} OpenRouter API error"))
        self.assertEqual(retry_payload["provider_request_id"], "gen-test-2")
        self.assertIn(
            "openrouter|aggregator_submitter_2|vendor/test-free:free|transient_provider_error",
            self.manager._retryable_provider_backoff_state,
        )
        self.assertEqual(
            str(error),
            "OpenRouter transient provider failure",
        )

    async def test_explicit_manual_proof_mode_wins_over_active_owner(self) -> None:
        role_id = "autonomous_proof_formalization_manual_brainstorm"
        self._configure_openrouter_role(role_id)
        broadcast = mock.AsyncMock()
        self.manager.set_broadcast_callback(broadcast)
        activity_callback = mock.AsyncMock()
        error = RetryableProviderError(
            provider="openrouter",
            provider_label="OpenRouter",
            role_id=role_id,
            model="vendor/test-free:free",
            reason="transient_provider_error",
            message="OpenRouter transient provider failure",
            configured_host_provider="",
            upstream_host_provider="TestHost",
        )

        with (
            mock.patch(
                "backend.shared.api_client_manager._active_notification_workflow_mode",
                return_value="aggregator",
            ),
            mock.patch.object(
                self.manager,
                "_sleep_with_optional_stop",
                new=mock.AsyncMock(return_value=True),
            ),
        ):
            await self.manager.wait_for_retryable_provider_error(
                error,
                role_id=role_id,
                activity_callback=activity_callback,
                workflow_mode="manual_proof_check",
            )

        waiting_payload = activity_callback.await_args_list[0].args[1]
        retry_payload = broadcast.await_args.args[1]
        self.assertEqual(waiting_payload["workflow_mode"], "manual_proof_check")
        self.assertEqual(retry_payload["workflow_mode"], "manual_proof_check")
        self.assertTrue(
            retry_payload["route_identity"].startswith(
                "[Manual proof check · Proof Formalization (manual brainstorm) · "
            )
        )
        self.assertTrue(retry_payload["message"].startswith(retry_payload["route_identity"]))

    async def test_free_model_rotation_runs_and_types_transient_alternative_failure(self) -> None:
        self._configure_openrouter_role("aggregator_submitter_2")
        self.manager._autonomous_logger_callback = None
        no_choices = ProviderRouteError(
            "OpenRouter returned no completion choices after 3 attempts.",
            route=ProviderRouteIdentity(provider="openrouter", model="vendor/test-alt:free"),
            cause=httpx.ReadError("OpenRouter upstream response omitted completion choices"),
        )
        openrouter_client = SimpleNamespace(generate_completion=mock.AsyncMock(side_effect=[
            RateLimitError(
                "OpenRouter free model hit rate limit.",
                model="vendor/test-free:free",
                retry_after=time.time() + 3600,
            ),
            no_choices,
        ]))
        free_models = mock.Mock(
            looping_enabled=True,
            auto_selector_enabled=False,
            AUTO_SELECTOR_MODEL="openrouter/free",
        )
        free_models.is_account_exhausted.return_value = False
        free_models.get_alternative_free_model.side_effect = ["vendor/test-alt:free", None]
        self.manager._openrouter_client = openrouter_client

        with (
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch("backend.shared.api_client_manager.free_model_manager", new=free_models),
        ):
            with self.assertRaises(RetryableProviderError) as raised:
                await self.manager.generate_completion(
                    task_id="agg_sub2_001",
                    role_id="aggregator_submitter_2",
                    model="vendor/test-free:free",
                    messages=[{"role": "user", "content": "hello"}],
                )

        self.assertEqual(openrouter_client.generate_completion.await_count, 2)
        error = raised.exception
        self.assertEqual(error.route_kind, "free_rotation")
        self.assertEqual(error.model, "vendor/test-alt:free")
        self.assertEqual(error.role_id, "aggregator_submitter_2")
        self.assertEqual(error.task_id, "agg_sub2_001")
        self.assertEqual(error.configured_host_provider, "")
        self.assertIsInstance(error.__cause__, ProviderRouteError)
        fields = self.manager.route_identity_for_role(
            "aggregator_submitter_2",
            workflow_mode="aggregator",
            error=error,
        )
        self.assertEqual(fields["role_label"], "Submitter 2 (free rotation)")
        self.assertEqual(fields["host_routing"], "auto")

    def test_route_identity_for_role_backfills_settings_but_not_pseudo_roles(self) -> None:
        self._configure_openrouter_role("aggregator_submitter_3", host="TestHost")

        configured = self.manager.route_identity_for_role(
            "aggregator_submitter_3",
            workflow_mode="aggregator",
        )
        pseudo = self.manager.route_identity_for_role("compiler", workflow_mode="compiler")
        non_model = self.manager.route_identity_for_role(
            "aggregator_submitter_3",
            workflow_mode="aggregator",
            provider="unknown",
        )

        self.assertEqual(configured["model"], "vendor/test-free:free")
        self.assertEqual(configured["provider"], "openrouter")
        self.assertEqual(configured["host_routing"], "pinned")
        self.assertEqual(
            configured["route_identity"],
            "[Aggregator · Submitter 3 · vendor/test-free:free via OpenRouter · "
            "host TestHost (pinned)]",
        )
        self.assertEqual(pseudo["route_identity"], "[Compiler · Compiler]")
        self.assertNotIn("model", pseudo)
        self.assertEqual(non_model["route_identity"], "[Aggregator · Submitter 3]")
        self.assertNotIn("model", non_model)

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

    def test_provider_notification_preserves_canonical_contract(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                stored = record_provider_notification(
                    "oauth_provider_usage_limited",
                    {
                        "event_type": "provider_cooldown_started",
                        "event_id": "provider_cooldown_started:abc",
                        "correlation_id": "abc",
                        "contract_version": 1,
                        "provider": "openai_codex_oauth",
                        "role_id": "agg_sub1",
                        "reason": "usage_limit_reached",
                        "retry_attempt": 2,
                        "max_attempts": 5,
                        "retry_after_seconds": 60,
                    },
                )
            finally:
                system_config.data_dir = old_data_dir

        self.assertEqual(stored["event_type"], "provider_cooldown_started")
        self.assertEqual(stored["legacy_event_type"], "oauth_provider_usage_limited")
        self.assertEqual(stored["contract_version"], 1)
        self.assertEqual(stored["retry_attempt"], 2)
        self.assertEqual(stored["max_attempts"], 5)
        self.assertEqual(stored["retry_after_seconds"], 60)


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

    def test_assistant_sakana_cooling_down_reflects_manager_state(self) -> None:
        from backend.shared.api_client_manager import api_client_manager

        api_client_manager.configure_role(
            "aggregator_assistant",
            ModelConfig(
                provider="sakana_fugu",
                model_id="fugu",
                context_window=128000,
                max_output_tokens=8192,
            ),
        )
        api_client_manager._oauth_provider_cooldowns["sakana_fugu"] = {
            "provider": "sakana_fugu",
            "cooldown_until": int(time.time()) + 120,
            "resets_at": int(time.time()) + 120,
        }
        try:
            self.assertTrue(_assistant_provider_is_cooling_down("aggregator_assistant"))
            self.assertTrue(_assistant_oauth_provider_is_cooling_down("aggregator_assistant"))
        finally:
            api_client_manager._oauth_provider_cooldowns.pop("sakana_fugu", None)
