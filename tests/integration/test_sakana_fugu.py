import tempfile
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException

from backend.api.routes import cloud_access as cloud_access_route
from backend.api.routes import features as features_route
from backend.shared.api_client_manager import APIClientManager
from backend.shared import secret_store
from backend.shared.config import system_config
from backend.shared.model_error_utils import is_retryable_model_output_error
from backend.shared.models import ModelConfig
from backend.shared.sakana_fugu_client import SakanaFuguAuthError, SakanaFuguClient, SakanaFuguRequestError


class SakanaFuguClientTests(IsolatedAsyncioTestCase):
    def test_normalize_reasoning_effort_filters_invalid_values(self) -> None:
        self.assertEqual(SakanaFuguClient._normalize_reasoning_effort("auto"), "xhigh")
        self.assertEqual(SakanaFuguClient._normalize_reasoning_effort("xhigh"), "xhigh")
        self.assertEqual(SakanaFuguClient._normalize_reasoning_effort("high"), "high")
        self.assertIsNone(SakanaFuguClient._normalize_reasoning_effort("medium"))
        self.assertIsNone(SakanaFuguClient._normalize_reasoning_effort("none"))

    def test_normalize_responses_to_chat_shape(self) -> None:
        payload = {
            "id": "resp_123",
            "model": "fugu",
            "output_text": '{"submission":"ok"}',
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        normalized = SakanaFuguClient._normalize_responses_to_chat(payload, "fugu")
        self.assertEqual(normalized["object"], "chat.completion")
        self.assertEqual(normalized["choices"][0]["message"]["content"], '{"submission":"ok"}')
        self.assertEqual(normalized["usage"]["prompt_tokens"], 10)
        self.assertEqual(normalized["usage"]["completion_tokens"], 5)
        self.assertEqual(normalized["_moto_sakana_wire_api"], "responses")

    def test_messages_need_chat_completions_for_tools(self) -> None:
        self.assertTrue(
            SakanaFuguClient._messages_need_chat_completions(
                [{"role": "user", "content": "hi"}],
                [{"type": "function", "function": {"name": "tool"}}],
            )
        )
        self.assertTrue(
            SakanaFuguClient._messages_need_chat_completions(
                [{"role": "tool", "content": "result"}],
                None,
            )
        )
        self.assertFalse(
            SakanaFuguClient._messages_need_chat_completions(
                [{"role": "user", "content": "hi"}],
                None,
            )
        )

    async def test_generate_completion_uses_chat_completions_for_tool_calls(self) -> None:
        client = SakanaFuguClient()
        client._api_key = "test-key"
        with mock.patch.object(
            client,
            "_generate_via_chat_completions",
            mock.AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]}),
        ) as chat_mock:
            await client.generate_completion(
                model="fugu",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "tool"}}],
            )
        chat_mock.assert_awaited_once()


class SakanaFuguSecretStoreTests(IsolatedAsyncioTestCase):
    async def test_secret_store_round_trip(self) -> None:
        with (
            mock.patch.object(secret_store, "_set_secret") as set_secret,
            mock.patch.object(secret_store, "_get_secret", return_value="sk-sakana-test"),
            mock.patch.object(secret_store, "_delete_secret") as delete_secret,
        ):
            secret_store.store_sakana_fugu_api_key("sk-sakana-test")
            self.assertEqual(secret_store.load_sakana_fugu_api_key(), "sk-sakana-test")
            secret_store.clear_sakana_fugu_api_key()
            set_secret.assert_called_once()
            delete_secret.assert_called_once()


class SakanaFuguRouteTests(IsolatedAsyncioTestCase):
    async def test_features_exposes_sakana_capability(self) -> None:
        with mock.patch.object(features_route.system_config, "generic_mode", False):
            payload = await features_route.get_features()
        self.assertIn("sakana_fugu_available", payload)
        self.assertTrue(payload["sakana_fugu_available"])

        with mock.patch.object(features_route.system_config, "generic_mode", True):
            hosted_payload = await features_route.get_features()
        self.assertFalse(hosted_payload["sakana_fugu_available"])

    async def test_set_api_key_rejected_in_generic_mode(self) -> None:
        with mock.patch.object(cloud_access_route.system_config, "generic_mode", True):
            with self.assertRaises(HTTPException) as ctx:
                await cloud_access_route.set_sakana_fugu_api_key(
                    cloud_access_route.SakanaFuguApiKeyRequest(api_key="sk-test")
                )
        self.assertEqual(ctx.exception.status_code, 501)

    async def test_set_api_key_validates_before_persisting(self) -> None:
        with (
            mock.patch.object(cloud_access_route.system_config, "generic_mode", False),
            mock.patch.object(
                cloud_access_route.sakana_fugu_client,
                "list_models",
                mock.AsyncMock(side_effect=SakanaFuguAuthError("bad key")),
            ) as list_models,
            mock.patch.object(cloud_access_route.sakana_fugu_client, "set_api_key") as set_api_key,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await cloud_access_route.set_sakana_fugu_api_key(
                    cloud_access_route.SakanaFuguApiKeyRequest(api_key=" bad-key ")
                )

        self.assertEqual(ctx.exception.status_code, 400)
        list_models.assert_awaited_once_with(api_key="bad-key")
        set_api_key.assert_not_called()

    async def test_cloud_access_status_includes_sakana_provider(self) -> None:
        async def sakana_status():
            return {"configured": True, "provider": "sakana_fugu"}

        with (
            mock.patch.object(cloud_access_route.system_config, "generic_mode", False),
            mock.patch.object(cloud_access_route.sakana_fugu_client, "status", sakana_status),
            mock.patch.object(
                cloud_access_route.openai_codex_client,
                "status",
                mock.AsyncMock(return_value={"configured": False}),
            ),
            mock.patch.object(
                cloud_access_route.xai_grok_client,
                "status",
                mock.AsyncMock(return_value={"configured": False}),
            ),
        ):
            payload = await cloud_access_route.get_cloud_access_status()

        self.assertTrue(payload["providers"]["sakana_fugu"]["configured"])
        self.assertTrue(payload["providers"]["sakana_fugu"]["available"])


class SakanaFuguAPIClientManagerTests(IsolatedAsyncioTestCase):
    async def test_generate_completion_routes_sakana_role_and_tracks_usage(self) -> None:
        manager = APIClientManager()
        manager.configure_role(
            "agg_sub1",
            ModelConfig(
                provider="sakana_fugu",
                model_id="fugu",
                context_window=1000000,
                max_output_tokens=100000,
            ),
        )
        tracked_models = []
        logger_calls = []

        async def track_model(model_id: str) -> None:
            tracked_models.append(model_id)

        async def log_call(**kwargs) -> None:
            logger_calls.append(kwargs)

        manager._model_tracking_callback = track_model
        manager.set_autonomous_logger_callback(log_call)

        def fake_response() -> dict:
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "fugu",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }

        with (
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch.object(
                cloud_access_route.sakana_fugu_client,
                "generate_completion",
                mock.AsyncMock(side_effect=lambda **_kwargs: fake_response()),
            ) as generate_completion,
        ):
            result = await manager.generate_completion(
                task_id="agg_sub1_001",
                role_id="agg_sub1",
                model="fugu",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=200,
            )

        generate_completion.assert_awaited_once()
        metadata = manager.extract_call_metadata(result)
        self.assertEqual(metadata["effective_provider"], "sakana_fugu")
        self.assertEqual(metadata["effective_model"], "fugu")
        self.assertEqual(tracked_models, ["fugu"])
        self.assertEqual(logger_calls[0]["provider"], "sakana_fugu")
        self.assertEqual(logger_calls[0]["tokens_used"], 5)

    async def test_output_truncation_bypasses_fallback_and_unrecoverable_notification(self) -> None:
        manager = APIClientManager()
        role_id = "autonomous_proof_formalization_brainstorm"
        manager.configure_role(
            role_id,
            ModelConfig(
                provider="sakana_fugu",
                model_id="fugu",
                lm_studio_fallback_id="local-fallback",
                context_window=1000000,
                max_output_tokens=100000,
            ),
        )
        truncation_error = SakanaFuguRequestError(
            "Sakana Fugu completion failed: "
            '{"type":"response.incomplete","response":{"status":"incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"}}}'
        )

        with (
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch.object(
                cloud_access_route.sakana_fugu_client,
                "generate_completion",
                mock.AsyncMock(side_effect=truncation_error),
            ),
            mock.patch.object(
                manager,
                "_broadcast_unrecoverable_sakana_fugu_error",
                new=mock.AsyncMock(),
            ) as unrecoverable_broadcast,
        ):
            with self.assertRaises(SakanaFuguRequestError) as ctx:
                await manager.generate_completion(
                    task_id="proof_form_000",
                    role_id=role_id,
                    model="fugu",
                    messages=[{"role": "user", "content": "prove"}],
                )

        self.assertTrue(is_retryable_model_output_error(ctx.exception))
        self.assertEqual(manager._role_fallback_state[role_id], "sakana_fugu")
        unrecoverable_broadcast.assert_not_awaited()
