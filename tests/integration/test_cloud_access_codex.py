import asyncio
import base64
import json
import tempfile
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException
import httpx
from keyring.errors import PasswordDeleteError

from backend.api.routes import cloud_access as cloud_access_route
from backend.api.routes import features as features_route
from backend.shared import secret_store
from backend.shared.api_client_manager import oauth_live_activity_error_message
from backend.shared.config import system_config
from backend.shared.openai_codex_client import (
    OpenAICodexAuthError,
    OpenAICodexClient,
    OpenAICodexRequestError,
)
from backend.shared.provider_notification_store import list_provider_notifications, record_provider_notification
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.xai_grok_client import (
    XAIGrokClient,
    XAIGrokRequestError,
    XAIGrokSpendingLimitError,
    XAIGrokUsageLimitError,
)


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"header.{encoded}.sig"


class ProviderNotificationStoreTests(IsolatedAsyncioTestCase):
    def test_provider_notification_reports_degraded_persistence(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                with mock.patch(
                    "backend.shared.provider_notification_store._write_payload",
                    return_value=False,
                ):
                    stored = record_provider_notification(
                        "provider_retry_cooldown",
                        {
                            "provider": "openrouter",
                            "role_id": "agg_val",
                            "reason": "http_error",
                        },
                    )

                self.assertTrue(stored["persistence_degraded"])
                self.assertEqual(stored["persistence_status"], "degraded")
                self.assertIn("durable recovery storage", stored["persistence_detail"])
            finally:
                system_config.data_dir = old_data_dir

    async def test_model_error_notification_round_trips_explicit_run_scoped_fields(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                stored = record_provider_notification(
                    "auto_research_stopped",
                    {
                        "notification_key": "proof-truncation:session-a",
                        "notification_kind": "model_error",
                        "provider": "proof_model",
                        "role_id": "autonomous_proof_formalization",
                        "reason": "proof_output_truncation_recovery_exhausted",
                        "run_id": "session-a",
                        "title": "Proof model output repeatedly truncated",
                        "terminal_guidance": "Increase output allowance and restart.",
                    },
                )
                route_payload = await cloud_access_route.get_provider_notifications()

                self.assertEqual(stored["notification_key"], "proof-truncation:session-a")
                self.assertEqual(stored["notification_kind"], "model_error")
                self.assertEqual(stored["run_id"], "session-a")
                self.assertEqual(route_payload["notifications"][0]["title"], stored["title"])
                self.assertEqual(
                    route_payload["notifications"][0]["terminal_guidance"],
                    stored["terminal_guidance"],
                )
            finally:
                system_config.data_dir = old_data_dir

    async def test_provider_notifications_persist_for_route_hydration(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                stored = record_provider_notification(
                    "openai_codex_oauth_error",
                    {
                        "role_id": "autonomous_proof_formalization_brainstorm",
                        "model": "gpt-5.5",
                        "provider": "openai_codex_oauth",
                        "provider_label": "OpenAI Codex",
                        "reason": "unrecoverable_codex_error",
                        "message": "Check your OpenAI Codex OAuth connection, sign in again, and retry.",
                        "error_summary": "server_error for Bearer secret-value",
                        "oauth_error_message": "server_error for Bearer secret-value",
                    },
                )

                self.assertEqual(stored["event_type"], "openai_codex_oauth_error")
                self.assertEqual(stored["provider"], "openai_codex_oauth")
                self.assertIn("[redacted]", stored["error_summary"])
                self.assertIn("[redacted]", stored["oauth_error_message"])

                listed = list_provider_notifications()
                self.assertEqual(len(listed), 1)
                self.assertEqual(listed[0]["id"], stored["id"])

                route_payload = await cloud_access_route.get_provider_notifications()
                self.assertTrue(route_payload["success"])
                self.assertEqual(route_payload["notifications"][0]["id"], stored["id"])
                self.assertEqual(route_payload["notifications"][0]["notification_key"], stored["notification_key"])
            finally:
                system_config.data_dir = old_data_dir

    async def test_provider_notifications_dedupe_by_stable_logical_key(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                first = record_provider_notification(
                    "openai_codex_oauth_error",
                    {
                        "role_id": "agg_sub2",
                        "model": "gpt-5.5",
                        "provider": "openai_codex_oauth",
                        "reason": "unrecoverable_codex_error",
                        "message": "first message",
                    },
                )
                second = record_provider_notification(
                    "openai_codex_oauth_error",
                    {
                        "role_id": "agg_sub2",
                        "model": "gpt-5.5",
                        "provider": "openai_codex_oauth",
                        "reason": "unrecoverable_codex_error",
                        "message": "second message",
                    },
                )

                self.assertEqual(first["notification_key"], second["notification_key"])
                listed = list_provider_notifications()
                self.assertEqual(len(listed), 1)
                self.assertEqual(listed[0]["notification_key"], first["notification_key"])
                self.assertEqual(listed[0]["message"], "second message")
            finally:
                system_config.data_dir = old_data_dir

    async def test_grok_provider_notifications_persist_for_route_hydration(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                stored = record_provider_notification(
                    "oauth_provider_error",
                    {
                        "role_id": "autonomous_proof_formalization_brainstorm",
                        "model": "grok-4.3",
                        "provider": "xai_grok_oauth",
                        "provider_label": "xAI Grok",
                        "reason": "unrecoverable_xai_grok_error",
                        "message": "Check your xAI Grok OAuth connection, sign in again, and retry.",
                        "error_summary": "subscription error for Bearer secret-value",
                        "oauth_error_message": "subscription error for Bearer secret-value",
                    },
                )

                self.assertEqual(stored["event_type"], "oauth_provider_error")
                self.assertEqual(stored["provider"], "xai_grok_oauth")
                self.assertEqual(stored["provider_label"], "xAI Grok")
                self.assertIn("[redacted]", stored["error_summary"])
                self.assertIn("[redacted]", stored["oauth_error_message"])

                route_payload = await cloud_access_route.get_provider_notifications()
                self.assertTrue(route_payload["success"])
                self.assertEqual(route_payload["notifications"][0]["id"], stored["id"])
                self.assertEqual(route_payload["notifications"][0]["provider"], "xai_grok_oauth")
            finally:
                system_config.data_dir = old_data_dir

    async def test_provider_notification_oauth_error_message_is_capped(self) -> None:
        old_data_dir = system_config.data_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config.data_dir = temp_dir
            try:
                stored = record_provider_notification(
                    "openai_codex_oauth_error",
                    {
                        "provider": "openai_codex_oauth",
                        "role_id": "autonomous_proof_identification_manual_brainstorm",
                        "reason": "unrecoverable_codex_error",
                        "oauth_error_message": "x" * 2200,
                    },
                )

                self.assertLessEqual(len(stored["oauth_error_message"]), 1800)
                self.assertTrue(stored["oauth_error_message"].endswith("..."))
            finally:
                system_config.data_dir = old_data_dir


class OAuthLiveActivityErrorTests(IsolatedAsyncioTestCase):
    async def test_embedded_responses_input_is_redacted_from_provider_error(self) -> None:
        message = sanitize_provider_error_text(
            'request failed: {"input":[{"role":"user","content":"private-user-text"}],'
            '"error":{"message":"gateway failed"}}',
            max_chars=1800,
        )

        self.assertIn('"input":[redacted]', message)
        self.assertNotIn("private-user-text", message)

    async def test_codex_context_length_message_is_extracted_for_live_activity(self) -> None:
        message = oauth_live_activity_error_message(
            RuntimeError(
                'OpenAI Codex completion failed: {"code": "context_length_exceeded", '
                '"message": "Your input exceeds the context window of this model. Please adjust your input and try again."}'
            )
        )

        self.assertEqual(
            message,
            "context_length_exceeded: Your input exceeds the context window of this model. Please adjust your input and try again.",
        )
        self.assertLessEqual(len(message), 1800)

    async def test_grok_plain_error_is_capped_for_live_activity(self) -> None:
        message = oauth_live_activity_error_message(
            RuntimeError("xAI Grok completion failed: " + ("subscription quota exceeded " * 120))
        )

        self.assertLessEqual(len(message), 1800)
        self.assertTrue(message.endswith("..."))


class OpenAICodexClientTests(IsolatedAsyncioTestCase):
    def test_authorization_url_uses_pkce_and_codex_client_id(self) -> None:
        verifier, challenge = OpenAICodexClient.generate_pkce_pair()
        url = OpenAICodexClient.build_authorization_url(
            code_challenge=challenge,
            state="state-1",
            redirect_uri="http://localhost:1455/auth/callback",
        )

        self.assertGreater(len(verifier), 40)
        self.assertIn("client_id=app_EMoamEEZ73f0CkXaXp7hrann", url)
        self.assertIn("code_challenge=", url)
        self.assertIn("code_challenge_method=S256", url)
        self.assertIn("api.connectors.read", url)
        self.assertIn("api.connectors.invoke", url)
        self.assertIn("codex_cli_simplified_flow=true", url)
        self.assertIn("originator=moto-autonomous-asi", url)
        self.assertIn("state=state-1", url)

    def test_token_payload_extracts_safe_status_fields(self) -> None:
        token = _jwt({
            "exp": 1234567890,
            "email": "user@example.com",
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        })

        normalized = OpenAICodexClient._normalize_token_payload({
            "access_token": token,
            "refresh_token": "refresh",
        })

        self.assertEqual(normalized["expires_at"], 1234567890)
        self.assertEqual(normalized["email"], "user@example.com")
        self.assertEqual(normalized["account_id"], "acct_123")
        self.assertNotIn("access_token", OpenAICodexClient.safe_status(normalized))

    def test_extract_output_prefers_aggregate_output_text_without_duplication(self) -> None:
        content, tool_calls = OpenAICodexClient._extract_output({
            "output_text": '{"ok": true}',
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"ok": true}'}],
                }
            ],
        })

        self.assertEqual(content, '{"ok": true}')
        self.assertEqual(tool_calls, [])

    async def test_generate_completion_returns_chat_completion_shape(self) -> None:
        client = OpenAICodexClient()

        class FakeHttp:
            async def post(self, url, json=None, headers=None):
                self.url = url
                self.payload = json
                self.headers = headers

                class Response:
                    status_code = 200
                    text = json.dumps({
                        "id": "resp_1",
                        "output_text": "hello",
                        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    })

                    def json(self):
                        return {
                            "id": "resp_1",
                            "output_text": "hello",
                            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                        }

                return Response()

        fake_http = FakeHttp()
        client.client = fake_http

        with mock.patch.object(client, "get_valid_tokens", return_value={
            "access_token": "access",
            "refresh_token": "refresh",
            "account_id": "acct_123",
        }):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                max_tokens=100,
                response_format={"type": "json_object"},
            )

        self.assertTrue(fake_http.url.endswith("/responses"))
        self.assertEqual(fake_http.headers["ChatGPT-Account-ID"], "acct_123")
        self.assertEqual(fake_http.headers["Accept"], "text/event-stream")
        self.assertEqual(fake_http.payload["instructions"], "system")
        self.assertTrue(fake_http.payload["stream"])
        self.assertEqual(fake_http.payload["text"]["format"]["type"], "json_object")
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")
        self.assertEqual(response["usage"]["total_tokens"], 5)

    def test_decode_streamed_codex_response(self) -> None:
        raw_stream = "\n\n".join([
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hel"}',
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"lo"}',
            (
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"id":"resp_1",'
                '"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}'
            ),
            "data: [DONE]",
        ])

        data = OpenAICodexClient._decode_response_body(raw_stream)

        self.assertEqual(data["id"], "resp_1")
        self.assertEqual(data["output_text"], "hello")
        self.assertEqual(data["usage"]["total_tokens"], 5)

    def test_decode_stream_requires_completed_terminal_event(self) -> None:
        raw_stream = "\n\n".join([
            'data: {"type":"response.output_text.delta","delta":"partial"}',
            'data: {"type":"response.in_progress","response":{"id":"resp_1","status":"in_progress"}}',
            "data: [DONE]",
        ])

        with self.assertRaisesRegex(OpenAICodexRequestError, "without response.completed") as raised:
            OpenAICodexClient._decode_response_body(raw_stream)

        self.assertEqual(raised.exception.failure_kind, "terminal_missing")
        self.assertIn("Remote outcome is unknown", str(raised.exception))

    def test_decode_stream_rejects_incomplete_with_structured_reason(self) -> None:
        raw_stream = (
            'data: {"type":"response.incomplete","response":{"id":"resp_1",'
            '"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}\n\n'
        )

        with self.assertRaises(OpenAICodexRequestError) as raised:
            OpenAICodexClient._decode_response_body(raw_stream)

        self.assertEqual(raised.exception.failure_kind, "output_truncated")
        self.assertEqual(raised.exception.error_code, "max_output_tokens")
        self.assertIn("max_output_tokens", str(raised.exception))

    def test_decode_stream_classifies_top_level_error_event(self) -> None:
        raw_stream = (
            'data: {"type":"error","code":"invalid_request_error",'
            '"message":"Unsupported request shape"}\n\n'
        )

        with self.assertRaises(OpenAICodexRequestError) as raised:
            OpenAICodexClient._decode_response_body(raw_stream)

        self.assertEqual(raised.exception.failure_kind, "stream_rejected")
        self.assertEqual(raised.exception.error_code, "invalid_request_error")
        self.assertIn("Unsupported request shape", str(raised.exception))

    def test_decode_plain_json_rejects_nonterminal_status(self) -> None:
        with self.assertRaisesRegex(OpenAICodexRequestError, "status=in_progress") as raised:
            OpenAICodexClient._decode_response_body(
                json.dumps({"id": "resp_1", "status": "in_progress", "output_text": "partial"})
            )
        self.assertEqual(raised.exception.failure_kind, "terminal_missing")

    async def test_stream_transport_failure_after_completed_event_returns_success(self) -> None:
        client = OpenAICodexClient()
        completed = (
            'data: {"type":"response.completed","response":{"id":"resp_1",'
            '"status":"completed","output_text":"finished"}}'
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            class BrokenAfterComplete(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield completed.encode("utf-8") + b"\n\n"
                    raise httpx.RemoteProtocolError("incomplete chunked read")

            return httpx.Response(200, stream=BrokenAfterComplete(), request=request)

        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
            )
        await client.client.aclose()

        self.assertEqual(response["choices"][0]["message"]["content"], "finished")

    async def test_stream_transport_failure_before_terminal_retries_with_safe_activity(self) -> None:
        client = OpenAICodexClient()
        calls = 0
        activities = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                class BrokenBeforeComplete(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        yield b'data: {"type":"response.output_text.delta","delta":"secret partial"}\n\n'
                        raise httpx.RemoteProtocolError(
                            "peer closed connection; Authorization: Bearer hidden-token"
                        )

                return httpx.Response(200, stream=BrokenBeforeComplete(), request=request)
            stream = (
                'data: {"type":"response.completed","response":{"id":"resp_2",'
                '"status":"completed","output_text":"recovered"}}\n\n'
            )
            return httpx.Response(200, content=stream, request=request)

        async def retry_callback(payload):
            activities.append(payload)

        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)
        with (
            mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}),
            mock.patch("backend.shared.openai_codex_client.asyncio.sleep", return_value=None),
        ):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
                retry_callback=retry_callback,
            )
        await client.client.aclose()

        self.assertEqual(calls, 2)
        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(activities[0]["reason"], "connection_error")
        self.assertIn("remote outcome is unknown", activities[0]["message"])
        self.assertIn("[redacted]", activities[0]["message"])
        self.assertNotIn("hidden-token", activities[0]["message"])
        self.assertNotIn("secret partial", activities[0]["message"])

    async def test_stream_cancellation_does_not_retry(self) -> None:
        client = OpenAICodexClient()
        calls = 0
        activities = []
        entered = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1

            class BlockingStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    entered.set()
                    await asyncio.Event().wait()
                    yield b""

            return httpx.Response(200, stream=BlockingStream(), request=request)

        async def retry_callback(payload):
            activities.append(payload)

        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            task = asyncio.create_task(client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
                retry_callback=retry_callback,
            ))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        await client.client.aclose()

        self.assertEqual(calls, 1)
        self.assertEqual(activities, [])

    async def test_list_models_normalizes_codex_catalog_limits(self) -> None:
        client = OpenAICodexClient()

        class FakeHttp:
            async def get(self, url, headers=None):
                class Response:
                    status_code = 200

                    def json(self):
                        return {
                            "models": [
                                {
                                    "slug": "gpt-5.5",
                                    "title": "GPT-5.5",
                                    "context_window": 272000,
                                    "max_context_window": 272000,
                                    "effective_context_window_percent": 95,
                                    "max_output_tokens": 128000,
                                }
                            ]
                        }

                return Response()

        client.client = FakeHttp()
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            models = await client.list_models()

        self.assertEqual(models[0]["id"], "gpt-5.5")
        self.assertEqual(models[0]["context_length"], 400000)
        self.assertEqual(models[0]["input_context_window"], 272000)
        self.assertEqual(models[0]["effective_input_context_window"], 258400)
        self.assertEqual(models[0]["max_output_tokens"], 128000)

    async def test_list_models_includes_codex_spark_high_catalog_fallback(self) -> None:
        client = OpenAICodexClient()

        class FakeHttp:
            async def get(self, url, headers=None):
                class Response:
                    status_code = 200

                    def json(self):
                        return {"models": []}

                return Response()

        client.client = FakeHttp()
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            models = await client.list_models()

        self.assertEqual(models[0]["id"], "gpt-5.3-codex-spark-high")
        self.assertEqual(models[0]["canonical_model"], "gpt-5.3-codex-spark")
        self.assertEqual(models[0]["reasoning_effort"], "high")
        self.assertEqual(models[0]["context_length"], 128000)
        self.assertEqual(models[0]["max_output_tokens"], 32768)

    async def test_generate_completion_maps_codex_spark_high_alias(self) -> None:
        client = OpenAICodexClient()

        class FakeHttp:
            async def post(self, url, json=None, headers=None):
                import json as json_module

                self.payload = json
                response_text = json_module.dumps({
                    "id": "resp_1",
                    "output_text": "hello",
                    "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                })

                class Response:
                    status_code = 200
                    text = response_text

                    def json(self):
                        return {
                            "id": "resp_1",
                            "output_text": "hello",
                            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                        }

                return Response()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            response = await client.generate_completion(
                model="gpt-5.3-codex-spark-high",
                messages=[{"role": "user", "content": "user"}],
            )

        self.assertEqual(fake_http.payload["model"], "gpt-5.3-codex-spark")
        self.assertEqual(fake_http.payload["reasoning"]["effort"], "high")
        self.assertEqual(response["model"], "gpt-5.3-codex-spark-high")

    def test_resolve_gpt_56_named_variants(self) -> None:
        self.assertEqual(
            OpenAICodexClient._resolve_model_request("gpt-5.6-sol", None),
            ("gpt-5.6-sol", None),
        )
        self.assertEqual(
            OpenAICodexClient._resolve_model_request("gpt-5.6-luna", "xhigh"),
            ("gpt-5.6-luna", "xhigh"),
        )
        self.assertEqual(
            OpenAICodexClient._resolve_model_request("gpt-5.6-terra", "low"),
            ("gpt-5.6-terra", "low"),
        )

    async def test_generate_completion_preserves_provider_listed_model_and_reasoning(self) -> None:
        class FakeHttp:
            async def post(self, url, json=None, headers=None):
                import json as json_module

                self.payload = json
                response_text = json_module.dumps({
                    "id": "resp_1",
                    "output_text": "hello",
                    "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                })

                class Response:
                    status_code = 200
                    text = response_text

                    def json(self):
                        return {
                            "id": "resp_1",
                            "output_text": "hello",
                            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                        }

                return Response()

        for selected_model, selected_effort in (
            ("gpt-5.6-terra", "low"),
            ("gpt-5.6-luna", "xhigh"),
            ("catalog-none", "none"),
            ("catalog-minimal", "minimal"),
            ("catalog-max", "max"),
            ("catalog-high", "auto"),
            ("catalog-max", "auto"),
            ("gpt-5.5", "auto"),
            ("gpt-5.3-codex-spark-high", "max"),
        ):
            with self.subTest(model=selected_model, reasoning_effort=selected_effort):
                client = OpenAICodexClient()
                fake_http = FakeHttp()
                client.client = fake_http
                client._model_reasoning_cache = {
                    "catalog-high": {"supported_reasoning_levels": ["low", "medium", "high"]},
                    "catalog-max": {"supported_reasoning_levels": ["low", "xhigh", "max"]},
                }
                with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
                    response = await client.generate_completion(
                        model=selected_model,
                        messages=[{"role": "user", "content": "user"}],
                        reasoning_effort=selected_effort,
                    )

                expected_model = selected_model
                expected_effort = selected_effort
                if selected_effort == "auto":
                    expected_effort = {"catalog-high": "high", "catalog-max": "max", "gpt-5.5": "xhigh"}[selected_model]
                if selected_model == "gpt-5.3-codex-spark-high":
                    expected_model, expected_effort = "gpt-5.3-codex-spark", "high"
                self.assertEqual(fake_http.payload["model"], expected_model)
                self.assertEqual(fake_http.payload["reasoning"]["effort"], expected_effort)
                self.assertEqual(response["model"], selected_model)

    def test_codex_models_response_preserves_reasoning_and_extra_metadata(self) -> None:
        from backend.api.routes.cloud_access import CodexModelsResponse

        response = CodexModelsResponse.model_validate({"success": True, "models": [{
            "id": "provider-verbatim", "name": "Provider variant", "context_length": 123456,
            "supported_reasoning_levels": ["minimal", "high", "max"],
            "default_reasoning_level": "high",
        }]})
        row = response.model_dump(exclude_none=True)["models"][0]
        self.assertEqual(row["supported_reasoning_levels"], ["minimal", "high", "max"])
        self.assertEqual(row["default_reasoning_level"], "high")
        self.assertEqual(row["context_length"], 123456)

    def test_reasoning_catalog_normalization_and_unknown_safety(self) -> None:
        normalized = OpenAICodexClient._normalize_model_metadata({
            "slug": "provider-verbatim-variant",
            "supported_reasoning_levels": [
                {"effort": "high", "description": "deep"}, "minimal", "none", "max",
                "high", {"effort": "invented"}, None,
            ],
            "default_reasoning_level": "minimal",
        })
        self.assertEqual(normalized["supported_reasoning_levels"], ["high", "minimal", "none", "max"])
        self.assertEqual(normalized["default_reasoning_level"], "minimal")
        self.assertEqual(OpenAICodexClient._reasoning_config("auto", normalized), {"effort": "max"})
        self.assertIsNone(OpenAICodexClient._reasoning_config("auto", {"supported_reasoning_levels": []}))
        malformed = OpenAICodexClient._normalize_reasoning_metadata({"supported_reasoning_levels": [None, {"effort": "invented"}]})
        self.assertNotIn("supported_reasoning_levels", malformed)
        for metadata in ({}, malformed, {"default_reasoning_level": "medium"}):
            with self.assertRaises(OpenAICodexRequestError):
                OpenAICodexClient._reasoning_config("auto", metadata)
        for effort in ("minimal", "none", "max"):
            self.assertEqual(OpenAICodexClient._reasoning_config(effort), {"effort": effort})
        unknown = OpenAICodexClient._normalize_model_metadata({"slug": "gpt-5.5-unlisted-variant"})
        self.assertNotIn("supported_reasoning_levels", unknown)
        explicit_empty = OpenAICodexClient._normalize_model_metadata({"slug": "gpt-5.5", "supported_reasoning_levels": []})
        self.assertEqual(explicit_empty["supported_reasoning_levels"], [])

    def test_known_model_malformed_catalog_does_not_invent_reasoning_levels(self) -> None:
        for raw_levels in ([None, {"effort": "invented"}], "high", None):
            with self.subTest(raw_levels=raw_levels):
                normalized = OpenAICodexClient._normalize_model_metadata({
                    "slug": "gpt-5.5", "supported_reasoning_levels": raw_levels,
                })
                self.assertNotIn("supported_reasoning_levels", normalized)
                with self.assertRaises(OpenAICodexRequestError):
                    OpenAICodexClient._reasoning_config("auto", normalized)
                self.assertEqual(
                    OpenAICodexClient._reasoning_config("low", normalized),
                    {"effort": "low"},
                )
        missing = OpenAICodexClient._normalize_model_metadata({"slug": "gpt-5.5"})
        self.assertEqual(OpenAICodexClient._reasoning_config("auto", missing), {"effort": "xhigh"})

    async def test_reasoning_cold_catalog_load_and_cache(self) -> None:
        import httpx

        client = OpenAICodexClient()
        await client.client.aclose()
        response = httpx.Response(200, json={"models": [{
            "slug": "provider-cold-variant", "supported_reasoning_levels": [{"effort": "medium"}, {"effort": "high"}],
            "default_reasoning_level": "medium",
        }]})
        client.client = mock.AsyncMock()
        client.client.get.return_value = response
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            metadata = await client._model_reasoning_metadata("provider-cold-variant")
            self.assertEqual(OpenAICodexClient._reasoning_config("auto", metadata), {"effort": "high"})
            self.assertEqual(await client._model_reasoning_metadata("provider-cold-variant"), metadata)
            client.client.get.assert_awaited_once()
            unknown = await client._model_reasoning_metadata("unknown-variant")
            self.assertEqual(unknown, {})
            with self.assertRaises(OpenAICodexRequestError):
                OpenAICodexClient._reasoning_config("auto", unknown)

    async def test_list_models_retries_with_newer_stored_token_after_revocation(self) -> None:
        client = OpenAICodexClient()
        old_tokens = {"access_token": "old-access", "refresh_token": "refresh"}
        new_tokens = {"access_token": "new-access", "refresh_token": "refresh"}

        class FakeResponse:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload
                self.text = json.dumps(payload)

            def json(self):
                return self._payload

        class FakeHttp:
            def __init__(self):
                self.auth_headers = []

            async def get(self, url, headers=None):
                self.auth_headers.append(headers.get("Authorization"))
                if len(self.auth_headers) == 1:
                    return FakeResponse(
                        401,
                        {
                            "error": {
                                "message": "Encountered invalidated oauth token for user, failing request",
                                "code": "token_revoked",
                            },
                            "status": 401,
                        },
                    )
                return FakeResponse(
                    200,
                    {"models": [{"slug": "gpt-5.5", "title": "GPT-5.5"}]},
                )

        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(client, "get_valid_tokens", return_value=old_tokens),
            mock.patch.object(client, "load_tokens", return_value=new_tokens),
            mock.patch.object(client, "refresh_tokens", wraps=client.refresh_tokens) as refresh_mock,
        ):
            models = await client.list_models()

        self.assertEqual(fake_http.auth_headers, ["Bearer old-access", "Bearer new-access"])
        self.assertEqual(refresh_mock.await_count, 0)
        self.assertEqual(models[0]["id"], "gpt-5.5")

    async def test_generate_completion_omits_unsupported_codex_request_knobs(self) -> None:
        client = OpenAICodexClient()

        class FakeHttp:
            async def post(self, url, json=None, headers=None):
                self.payload = json

                class Response:
                    status_code = 200
                    text = json.dumps({
                        "id": "resp_1",
                        "output_text": "hello",
                        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    })

                    def json(self):
                        return {
                            "id": "resp_1",
                            "output_text": "hello",
                            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                        }

                return Response()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
                max_tokens=250000,
                temperature=0.7,
            )

        self.assertNotIn("max_output_tokens", fake_http.payload)
        self.assertNotIn("temperature", fake_http.payload)
        self.assertTrue(fake_http.payload["stream"])
        self.assertEqual(
            fake_http.payload["instructions"],
            OpenAICodexClient.DEFAULT_INSTRUCTIONS,
        )

    async def test_generate_completion_retries_transient_gateway_response(self) -> None:
        client = OpenAICodexClient()

        class FakeResponse:
            def __init__(self, status_code, text):
                self.status_code = status_code
                self.text = text

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, url, json=None, headers=None):
                self.calls += 1
                if self.calls == 1:
                    return FakeResponse(
                        503,
                        "upstream connect error or disconnect/reset before headers. "
                        "retried and the latest reset reason: connection timeout",
                    )
                return FakeResponse(
                    200,
                    json_module.dumps({
                        "id": "resp_retry",
                        "output_text": "recovered",
                        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    }),
                )

        json_module = json
        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}),
            mock.patch("backend.shared.openai_codex_client.asyncio.sleep", return_value=None),
        ):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
            )

        self.assertEqual(fake_http.calls, 2)
        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")

    async def test_generate_completion_stops_after_four_exponential_codex_retries(self) -> None:
        client = OpenAICodexClient()

        class FakeResponse:
            status_code = 503
            text = "server_error: upstream provider timeout; you can retry your request"

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, url, json=None, headers=None):
                self.calls += 1
                return FakeResponse()

        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}),
            mock.patch(
                "backend.shared.openai_codex_client.assured_retry_wait",
                new=mock.AsyncMock(return_value=True),
            ) as wait_mock,
        ):
            with self.assertRaisesRegex(OpenAICodexRequestError, "after 4 retries"):
                await client.generate_completion(
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 5)
        self.assertEqual(
            [call.args[0]["retry_after_seconds"] for call in wait_mock.await_args_list],
            [2.0, 4.0, 8.0, 16.0],
        )

    async def test_generate_completion_retries_transient_codex_stream_failure(self) -> None:
        client = OpenAICodexClient()

        class FakeResponse:
            def __init__(self, text):
                self.status_code = 200
                self.text = text

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, url, json=None, headers=None):
                self.calls += 1
                if self.calls == 1:
                    return FakeResponse(
                        'data: {"type":"response.failed","error":{"code":"server_error",'
                        '"message":"Upstream provider timeout. You can retry your request."}}\n\n'
                    )
                return FakeResponse(
                    json_module.dumps({
                        "id": "resp_retry",
                        "output_text": "recovered",
                        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    })
                )

        json_module = json
        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}),
            mock.patch(
                "backend.shared.openai_codex_client.assured_retry_wait",
                new=mock.AsyncMock(return_value=True),
            ) as wait_mock,
        ):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
            )

        self.assertEqual(fake_http.calls, 2)
        wait_mock.assert_awaited_once()
        self.assertEqual(
            wait_mock.await_args.args[0]["retry_after_seconds"],
            2.0,
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")

    async def test_mixed_stream_and_transport_failures_share_one_request_budget(self) -> None:
        client = OpenAICodexClient()
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                body = (
                    'data: {"type":"response.failed","error":{"code":"server_error",'
                    '"message":"Upstream provider timeout. You can retry your request."}}\n\n'
                )
                return httpx.Response(200, content=body, request=request)

            class BrokenStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    raise httpx.RemoteProtocolError("incomplete chunked read")
                    yield b""

            return httpx.Response(200, stream=BrokenStream(), request=request)

        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)
        with (
            mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}),
            mock.patch("backend.shared.openai_codex_client.asyncio.sleep", return_value=None),
        ):
            with self.assertRaises(OpenAICodexRequestError):
                await client.generate_completion(
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "user"}],
                )
        await client.client.aclose()

        self.assertEqual(calls, 5)

    async def test_generate_completion_retries_with_newer_stored_token_after_revocation(self) -> None:
        client = OpenAICodexClient()
        old_tokens = {"access_token": "old-access", "refresh_token": "refresh"}
        new_tokens = {"access_token": "new-access", "refresh_token": "refresh"}

        class FakeResponse:
            def __init__(self, status_code, text):
                self.status_code = status_code
                self.text = text

        class FakeHttp:
            def __init__(self):
                self.auth_headers = []

            async def post(self, url, json=None, headers=None):
                self.auth_headers.append(headers.get("Authorization"))
                if len(self.auth_headers) == 1:
                    return FakeResponse(
                        401,
                        json_module.dumps({
                            "error": {
                                "message": "Encountered invalidated oauth token for user, failing request",
                                "code": "token_revoked",
                            },
                            "status": 401,
                        }),
                    )
                return FakeResponse(
                    200,
                    json_module.dumps({
                        "id": "resp_retry",
                        "output_text": "recovered",
                        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    }),
                )

        json_module = json
        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(client, "get_valid_tokens", return_value=old_tokens),
            mock.patch.object(client, "load_tokens", return_value=new_tokens),
            mock.patch.object(client, "refresh_tokens", wraps=client.refresh_tokens) as refresh_mock,
        ):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
            )

        self.assertEqual(fake_http.auth_headers, ["Bearer old-access", "Bearer new-access"])
        self.assertEqual(refresh_mock.await_count, 0)
        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")

    async def test_generate_completion_refreshes_current_token_after_revocation(self) -> None:
        client = OpenAICodexClient()
        old_tokens = {"access_token": "old-access", "refresh_token": "refresh"}
        refreshed_tokens = {"access_token": "fresh-access", "refresh_token": "refresh"}

        class FakeResponse:
            def __init__(self, status_code, text):
                self.status_code = status_code
                self.text = text

        class FakeHttp:
            def __init__(self):
                self.auth_headers = []

            async def post(self, url, json=None, headers=None):
                self.auth_headers.append(headers.get("Authorization"))
                if len(self.auth_headers) == 1:
                    return FakeResponse(
                        401,
                        json_module.dumps({
                            "error": {
                                "message": "Encountered invalidated oauth token for user, failing request",
                                "code": "token_revoked",
                            },
                            "status": 401,
                        }),
                    )
                return FakeResponse(
                    200,
                    json_module.dumps({
                        "id": "resp_retry",
                        "output_text": "recovered",
                        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    }),
                )

        json_module = json
        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(client, "get_valid_tokens", return_value=old_tokens),
            mock.patch.object(client, "load_tokens", return_value=old_tokens),
            mock.patch.object(client, "refresh_tokens", return_value=refreshed_tokens) as refresh_mock,
        ):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
            )

        self.assertEqual(fake_http.auth_headers, ["Bearer old-access", "Bearer fresh-access"])
        self.assertEqual(refresh_mock.await_count, 1)
        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")


    # ---- Responses websocket transport (websocket first, SSE fallback) ----

    def _enable_fake_websocket(self, client, connects):
        """Enable the websocket transport on this instance with a scripted connector."""
        self.assertFalse(
            OpenAICodexClient.websocket_transport_enabled,
            "the test overlay must keep the real websocket transport disabled",
        )
        client.websocket_transport_enabled = True
        attempts = []

        def open_websocket(url, headers):
            attempts.append({"url": url, "headers": dict(headers)})
            if not connects:
                raise AssertionError("no scripted websocket connection left")
            return connects.pop(0)

        client._open_websocket = open_websocket
        return attempts

    @staticmethod
    def _sse_client_that_must_not_run():
        async def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("SSE transport must not be used when the websocket turn completes")

        return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)

    @staticmethod
    def _sse_client_completing_with(text: str):
        async def handler(request: httpx.Request) -> httpx.Response:
            body = (
                'data: {"type":"response.completed","response":{"id":"resp_sse",'
                f'"status":"completed","output_text":"{text}"}}}}\n\n'
            )
            return httpx.Response(200, content=body, request=request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)

    async def test_websocket_transport_completes_turn_without_sse(self) -> None:
        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_that_must_not_run()
        websocket = _FakeCodexWebsocket([
            _ws_event({"type": "codex.rate_limits", "rate_limits": {}}),
            _ws_event({"type": "response.created", "response": {"id": "resp_ws", "status": "in_progress"}}),
            _ws_event({"type": "response.output_text.delta", "delta": "websocket "}),
            _ws_event({"type": "response.output_text.delta", "delta": "done"}),
            _ws_event({
                "type": "response.completed",
                "response": {
                    "id": "resp_ws",
                    "status": "completed",
                    "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                },
            }),
        ])
        attempts = self._enable_fake_websocket(client, [_FakeCodexWebsocketConnect(websocket)])

        with mock.patch.object(
            client,
            "get_valid_tokens",
            return_value={"access_token": "access", "account_id": "acct_123"},
        ):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                ],
                reasoning_effort="low",
            )
        await client.client.aclose()

        self.assertEqual(response["choices"][0]["message"]["content"], "websocket done")
        self.assertEqual(response["usage"]["total_tokens"], 6)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["url"], "wss://chatgpt.com/backend-api/codex/responses")
        headers = attempts[0]["headers"]
        self.assertEqual(headers["OpenAI-Beta"], "responses_websockets=2026-02-06")
        self.assertEqual(headers["originator"], "moto-autonomous-asi")
        self.assertEqual(headers["Authorization"], "Bearer access")
        self.assertEqual(headers["ChatGPT-Account-ID"], "acct_123")
        self.assertNotIn("Accept", headers)
        self.assertNotIn("Content-Type", headers)
        self.assertEqual(len(websocket.sent), 1)
        sent = json.loads(websocket.sent[0])
        self.assertEqual(sent["type"], "response.create")
        self.assertEqual(sent["model"], "gpt-5.5")
        self.assertIs(sent["stream"], True)
        self.assertIs(sent["store"], False)
        self.assertEqual(sent["instructions"], "sys")
        self.assertEqual(sent["input"], [{"role": "user", "content": "user"}])
        self.assertEqual(sent["reasoning"], {"effort": "low", "summary": "auto"})

    async def test_websocket_unsupported_handshake_falls_back_to_sse_and_stays_off(self) -> None:
        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response as WsResponse

        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_completing_with("sse fallback")
        rejected = InvalidStatus(WsResponse(426, "Upgrade Required", Headers(), b"upgrade required"))
        attempts = self._enable_fake_websocket(
            client,
            [
                _FakeCodexWebsocketConnect(handshake_error=rejected),
                _FakeCodexWebsocketConnect(handshake_error=rejected),
            ],
        )

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            first = await client.generate_completion(
                model="gpt-5.5", messages=[{"role": "user", "content": "user"}],
            )
            second = await client.generate_completion(
                model="gpt-5.5", messages=[{"role": "user", "content": "user"}],
            )
        await client.client.aclose()

        self.assertEqual(first["choices"][0]["message"]["content"], "sse fallback")
        self.assertEqual(second["choices"][0]["message"]["content"], "sse fallback")
        self.assertEqual(len(attempts), 1, "an unsupported transport is not retried for later requests")
        self.assertIn("HTTP 426", client._websocket_disabled_reason)

    async def test_websocket_connect_failure_falls_back_without_disabling_transport(self) -> None:
        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_completing_with("sse fallback")
        websocket = _FakeCodexWebsocket([_ws_completed("resp_ws", "websocket done")])
        attempts = self._enable_fake_websocket(
            client,
            [
                _FakeCodexWebsocketConnect(handshake_error=OSError("connect refused")),
                _FakeCodexWebsocketConnect(websocket),
            ],
        )

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            first = await client.generate_completion(
                model="gpt-5.5", messages=[{"role": "user", "content": "user"}],
            )
            second = await client.generate_completion(
                model="gpt-5.5", messages=[{"role": "user", "content": "user"}],
            )
        await client.client.aclose()

        self.assertEqual(first["choices"][0]["message"]["content"], "sse fallback")
        self.assertEqual(second["choices"][0]["message"]["content"], "websocket done")
        self.assertEqual(len(attempts), 2)
        self.assertIsNone(client._websocket_disabled_reason)

    async def test_websocket_close_before_terminal_retries_with_safe_activity(self) -> None:
        from websockets.exceptions import ConnectionClosedError
        from websockets.frames import Close

        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_that_must_not_run()
        broken = _FakeCodexWebsocket([
            _ws_event({"type": "response.output_text.delta", "delta": "secret partial"}),
            ConnectionClosedError(
                Close(1011, "keepalive ping timeout; Authorization: Bearer hidden-token"),
                None,
            ),
        ])
        recovered = _FakeCodexWebsocket([_ws_completed("resp_2", "recovered")])
        attempts = self._enable_fake_websocket(
            client,
            [_FakeCodexWebsocketConnect(broken), _FakeCodexWebsocketConnect(recovered)],
        )
        activities = []

        async def retry_callback(payload):
            activities.append(payload)

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
                retry_callback=retry_callback,
            )
        await client.client.aclose()

        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(activities), 1)
        activity = activities[0]
        self.assertEqual(activity["reason"], "connection_error")
        self.assertEqual(activity["retry_attempt"], 1)
        self.assertEqual(activity["provider"], "openai_codex_oauth")
        self.assertIn("closed the Codex websocket stream", activity["error_summary"])
        self.assertIn("close_code=1011", activity["error_summary"])
        self.assertIn("remote outcome is unknown", activity["error_summary"])
        self.assertIn("closed this connection on its side", activity["error_summary"])
        self.assertIn("How to fix: lower this role's reasoning effort in Settings", activity["error_summary"])
        self.assertIn("How to fix", activity["message"])
        self.assertNotIn("hidden-token", activity["error_summary"])
        self.assertNotIn("secret partial", activity["error_summary"])

    async def test_websocket_close_after_terminal_uses_confirmed_response(self) -> None:
        from websockets.exceptions import ConnectionClosedOK
        from websockets.frames import Close

        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_that_must_not_run()
        websocket = _FakeCodexWebsocket([_ws_completed("resp_1", "finished")])
        attempts = self._enable_fake_websocket(
            client,
            [_FakeCodexWebsocketConnect(websocket, close_error=ConnectionClosedOK(Close(1000, ""), None))],
        )
        activities = []

        async def retry_callback(payload):
            activities.append(payload)

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
                retry_callback=retry_callback,
            )
        await client.client.aclose()

        self.assertEqual(response["choices"][0]["message"]["content"], "finished")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(activities, [])

    async def test_websocket_connection_limit_error_event_reconnects(self) -> None:
        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_that_must_not_run()
        limited = _FakeCodexWebsocket([
            _ws_event({
                "type": "error",
                "status": 409,
                "error": {
                    "code": "websocket_connection_limit_reached",
                    "message": "Responses websocket connection limit reached (60 minutes).",
                },
            }),
        ])
        recovered = _FakeCodexWebsocket([_ws_completed("resp_2", "recovered")])
        attempts = self._enable_fake_websocket(
            client,
            [_FakeCodexWebsocketConnect(limited), _FakeCodexWebsocketConnect(recovered)],
        )
        activities = []

        async def retry_callback(payload):
            activities.append(payload)

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
                retry_callback=retry_callback,
            )
        await client.client.aclose()

        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0]["reason"], "connection_error")
        self.assertIn("websocket_connection_limit_reached", activities[0]["error_summary"])

    async def test_websocket_provider_error_event_raises_without_sse_fallback(self) -> None:
        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_that_must_not_run()
        rejected = _FakeCodexWebsocket([
            _ws_event({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "unsupported_parameter",
                    "message": "Unsupported parameter.",
                },
            }),
        ])
        attempts = self._enable_fake_websocket(client, [_FakeCodexWebsocketConnect(rejected)])

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            with self.assertRaises(OpenAICodexRequestError) as raised:
                await client.generate_completion(
                    model="gpt-5.5", messages=[{"role": "user", "content": "user"}],
                )
        await client.client.aclose()

        self.assertEqual(raised.exception.failure_kind, "stream_rejected")
        self.assertEqual(raised.exception.error_code, "unsupported_parameter")
        self.assertEqual(len(attempts), 1)

    async def test_websocket_usage_limit_error_event_raises_typed_cooldown(self) -> None:
        from backend.shared.openai_codex_client import OAuthUsageLimitError

        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_that_must_not_run()
        limited = _FakeCodexWebsocket([
            _ws_event({
                "type": "error",
                "error": {
                    "type": "usage_limit_reached",
                    "message": "Usage window exhausted.",
                    "resets_in_seconds": 120,
                    "plan_type": "plus",
                },
            }),
        ])
        attempts = self._enable_fake_websocket(client, [_FakeCodexWebsocketConnect(limited)])

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            with self.assertRaises(OAuthUsageLimitError) as raised:
                await client.generate_completion(
                    model="gpt-5.5", messages=[{"role": "user", "content": "user"}],
                )
        await client.client.aclose()

        self.assertEqual(raised.exception.resets_in_seconds, 120)
        self.assertEqual(len(attempts), 1)

    async def test_websocket_transport_disabled_uses_sse_only(self) -> None:
        client = OpenAICodexClient()
        await client.client.aclose()
        client.client = self._sse_client_completing_with("sse only")
        attempts = self._enable_fake_websocket(client, [])
        client.websocket_transport_enabled = False

        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            response = await client.generate_completion(
                model="gpt-5.5", messages=[{"role": "user", "content": "user"}],
            )
        await client.client.aclose()

        self.assertEqual(response["choices"][0]["message"]["content"], "sse only")
        self.assertEqual(attempts, [])


def _ws_event(event: dict) -> str:
    return json.dumps(event)


def _ws_completed(response_id: str, output_text: str) -> str:
    return _ws_event({
        "type": "response.completed",
        "response": {"id": response_id, "status": "completed", "output_text": output_text},
    })


class _FakeCodexWebsocket:
    """Scripted websocket connection: strings are server frames, exceptions are raised in order."""

    def __init__(self, script):
        self.sent = []
        self._script = list(script)

    async def send(self, text):
        self.sent.append(text)

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for item in self._script:
            if isinstance(item, BaseException):
                raise item
            yield item


class _FakeCodexWebsocketConnect:
    """Async context manager standing in for websockets.asyncio.client.connect()."""

    def __init__(self, websocket=None, *, handshake_error=None, close_error=None):
        self.websocket = websocket
        self.handshake_error = handshake_error
        self.close_error = close_error

    async def __aenter__(self):
        if self.handshake_error is not None:
            raise self.handshake_error
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is None and self.close_error is not None:
            raise self.close_error
        return False


class XAIGrokClientTests(IsolatedAsyncioTestCase):
    def test_reasoning_effort_normalization_matches_xai_controls(self) -> None:
        self.assertEqual(XAIGrokClient._reasoning_effort("auto"), "high")
        self.assertEqual(XAIGrokClient._reasoning_effort("high"), "high")
        self.assertEqual(XAIGrokClient._reasoning_effort("medium"), "medium")
        self.assertEqual(XAIGrokClient._reasoning_effort("low"), "low")
        self.assertIsNone(XAIGrokClient._reasoning_effort("none"))

    async def test_spending_limit_bypasses_transient_retries(self) -> None:
        client = XAIGrokClient()

        class FakeResponse:
            status_code = 429
            headers = {}
            text = json.dumps({
                "code": "personal-team-blocked:spending-limit",
                "error": "You have run out of credits or need a Grok subscription.",
            })

            def json(self):
                return json.loads(self.text)

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(
            client,
            "get_valid_tokens",
            return_value={"access_token": "access"},
        ):
            with self.assertRaises(XAIGrokSpendingLimitError):
                await client.generate_completion(
                    model="grok-4.3",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 1)

    async def test_spending_limit_with_retry_after_header_stays_untimed(self) -> None:
        client = XAIGrokClient()

        class FakeResponse:
            status_code = 429
            headers = {"retry-after": "120", "x-ratelimit-reset": "120"}
            text = json.dumps({
                "code": "personal-team-blocked:spending-limit",
                "error": "You have run out of credits or need a Grok subscription.",
            })

            def json(self):
                return json.loads(self.text)

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(
            client,
            "get_valid_tokens",
            return_value={"access_token": "access"},
        ):
            with self.assertRaises(XAIGrokSpendingLimitError):
                await client.generate_completion(
                    model="grok-4.3",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 1)

    async def test_spending_limit_with_body_reset_metadata_becomes_timed_usage_limit(self) -> None:
        client = XAIGrokClient()

        class FakeResponse:
            status_code = 429
            headers = {"retry-after": "5"}
            text = json.dumps({
                "code": "usage_limit_reached",
                "error": "Usage window exhausted.",
                "resets_in_seconds": 120,
                "plan_type": "grok",
            })

            def json(self):
                return json.loads(self.text)

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(
            client,
            "get_valid_tokens",
            return_value={"access_token": "access"},
        ):
            with self.assertRaises(XAIGrokUsageLimitError) as caught:
                await client.generate_completion(
                    model="grok-4.3",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 1)
        self.assertEqual(caught.exception.resets_in_seconds, 120)
        self.assertEqual(caught.exception.plan_type, "grok")
        self.assertIsNotNone(caught.exception.resets_at)

    async def test_rate_limit_retry_after_becomes_timed_usage_limit(self) -> None:
        client = XAIGrokClient()

        class FakeResponse:
            status_code = 429
            headers = {"retry-after": "90"}
            text = json.dumps({
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Usage window exhausted",
                }
            })

            def json(self):
                return json.loads(self.text)

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(
            client,
            "get_valid_tokens",
            return_value={"access_token": "access"},
        ):
            with self.assertRaises(XAIGrokUsageLimitError) as caught:
                await client.generate_completion(
                    model="grok-4.3",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 1)
        self.assertEqual(caught.exception.resets_in_seconds, 90)
        self.assertIsNotNone(caught.exception.resets_at)

    async def test_server_error_retry_after_stays_transient(self) -> None:
        client = XAIGrokClient()

        class FakeResponse:
            status_code = 503
            headers = {"retry-after": "30"}
            text = json.dumps({
                "error": {
                    "code": "server_error",
                    "message": "Temporary unavailable",
                }
            })

            def json(self):
                return json.loads(self.text)

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse()

        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(
                client,
                "get_valid_tokens",
                return_value={"access_token": "access"},
            ),
            mock.patch.object(client, "_retry_delay", return_value=0),
        ):
            with self.assertRaises(XAIGrokRequestError):
                await client.generate_completion(
                    model="grok-4.3",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertGreater(fake_http.calls, 1)

    def test_authorization_url_uses_pkce_and_loopback_callback(self) -> None:
        verifier, challenge = XAIGrokClient.generate_pkce_pair()
        url = XAIGrokClient.build_authorization_url(
            code_challenge=challenge,
            state="state-1",
            nonce="nonce-1",
            redirect_uri="http://127.0.0.1:56121/callback",
        )

        self.assertGreater(len(verifier), 40)
        self.assertTrue(url.startswith("https://auth.x.ai/oauth2/authorize?"))
        self.assertIn("client_id=", url)
        self.assertIn("code_challenge=", url)
        self.assertIn("code_challenge_method=S256", url)
        self.assertIn("grok-cli%3Aaccess", url)
        self.assertIn("api%3Aaccess", url)
        self.assertIn("plan=generic", url)
        self.assertIn("referrer=moto-autonomous-asi", url)
        self.assertIn("state=state-1", url)
        self.assertIn("nonce=nonce-1", url)

    async def test_exchange_code_persists_normalized_tokens(self) -> None:
        client = XAIGrokClient()

        class FakeHttp:
            async def post(self, url, data=None, headers=None):
                self.url = url
                self.data = data
                self.headers = headers

                class Response:
                    status_code = 200

                    def json(self):
                        return {
                            "access_token": "access",
                            "refresh_token": "refresh",
                            "expires_in": 3600,
                        }

                return Response()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch("backend.shared.xai_grok_client.store_xai_grok_oauth_tokens") as store_tokens:
            status = await client.exchange_code(
                code="code",
                code_verifier="verifier",
                code_challenge="challenge",
            )

        self.assertTrue(status["configured"])
        self.assertEqual(fake_http.url, "https://auth.x.ai/oauth2/token")
        self.assertEqual(fake_http.data["code_challenge"], "challenge")
        self.assertEqual(fake_http.data["code_challenge_method"], "S256")
        stored = store_tokens.call_args.args[0]
        self.assertEqual(stored["provider"], "xai_grok_oauth")

    async def test_list_models_normalizes_xai_catalog(self) -> None:
        client = XAIGrokClient()

        class FakeHttp:
            async def get(self, url, headers=None):
                class Response:
                    status_code = 200
                    text = "{}"

                    def json(self):
                        return {
                            "data": [
                                {
                                    "id": "grok-4.3",
                                    "name": "Grok 4.3",
                                    "context_length": 1000000,
                                    "max_output_tokens": 131072,
                                },
                                {
                                    "id": "grok-4.2",
                                    "name": "Grok 4.2",
                                },
                                {
                                    "id": "grok-4.20-multi-agent-0309",
                                    "name": "Grok 4.20 Multi Agent",
                                },
                                {
                                    "id": "grok-4-fast",
                                    "name": "Grok 4 Fast",
                                }
                            ]
                        }

                return Response()

        client.client = FakeHttp()
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            models = await client.list_models()

        self.assertEqual(models[0]["id"], "grok-4.3")
        self.assertEqual(models[0]["context_length"], 1000000)
        self.assertEqual(models[0]["max_output_tokens"], 131072)
        self.assertEqual(models[0]["pricing"]["prompt"], "subscription")
        self.assertEqual(models[1]["id"], "grok-4.2")
        self.assertEqual(models[1]["context_length"], 1000000)
        self.assertEqual(models[1]["max_output_tokens"], 131072)
        self.assertEqual(models[2]["id"], "grok-4-fast")
        self.assertNotIn("context_length", models[2])
        self.assertNotIn("max_output_tokens", models[2])
        self.assertNotIn("grok-4.20-multi-agent-0309", [model["id"] for model in models])

    async def test_generate_completion_rejects_xai_multi_agent_model_before_request(self) -> None:
        client = XAIGrokClient()

        class FakeHttp:
            async def post(self, url, json=None, headers=None):
                raise AssertionError("multi-agent models should not be sent to chat completions")

        client.client = FakeHttp()
        with self.assertRaisesRegex(XAIGrokRequestError, "not supported by the OAuth chat-completions route"):
            await client.generate_completion(
                model="grok-4.20-multi-agent-0309",
                messages=[{"role": "user", "content": "user"}],
            )

    async def test_generate_completion_uses_chat_completions_shape(self) -> None:
        client = XAIGrokClient()

        class FakeHttp:
            async def post(self, url, json=None, headers=None):
                self.url = url
                self.payload = json
                self.headers = headers

                class Response:
                    status_code = 200
                    text = "{}"

                    def json(self):
                        return {
                            "id": "chatcmpl_1",
                            "object": "chat.completion",
                            "model": "grok-4.3",
                            "choices": [
                                {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                            ],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                        }

                return Response()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            response = await client.generate_completion(
                model="grok-4.3",
                messages=[{"role": "user", "content": "user"}],
                max_tokens=100,
                response_format={"type": "json_object"},
                reasoning_effort="xhigh",
            )

        self.assertTrue(fake_http.url.endswith("/chat/completions"))
        self.assertEqual(fake_http.payload["max_tokens"], 100)
        self.assertEqual(fake_http.payload["response_format"]["type"], "json_object")
        self.assertEqual(fake_http.payload["reasoning_effort"], "high")
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")

    async def test_generate_completion_omits_disabled_reasoning_effort(self) -> None:
        client = XAIGrokClient()

        class FakeHttp:
            async def post(self, url, json=None, headers=None):
                self.payload = json

                class Response:
                    status_code = 200
                    text = "{}"

                    def json(self):
                        return {
                            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                        }

                return Response()

        fake_http = FakeHttp()
        client.client = fake_http
        with mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}):
            await client.generate_completion(
                model="grok-4.3",
                messages=[{"role": "user", "content": "user"}],
                reasoning_effort="none",
            )

        self.assertNotIn("reasoning_effort", fake_http.payload)

    async def test_generate_completion_stops_after_four_exponential_xai_retries(self) -> None:
        client = XAIGrokClient()

        class FakeResponse:
            status_code = 503
            text = "server_error: upstream provider timeout; you can retry your request"

        class FakeHttp:
            def __init__(self):
                self.calls = 0

            async def post(self, url, json=None, headers=None):
                self.calls += 1
                return FakeResponse()

        fake_http = FakeHttp()
        client.client = fake_http
        with (
            mock.patch.object(client, "get_valid_tokens", return_value={"access_token": "access"}),
            mock.patch(
                "backend.shared.xai_grok_client.assured_retry_wait",
                new=mock.AsyncMock(return_value=True),
            ) as wait_mock,
        ):
            with self.assertRaisesRegex(XAIGrokRequestError, "after 4 retries"):
                await client.generate_completion(
                    model="grok-4.3",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 5)
        self.assertEqual(
            [call.args[0]["retry_after_seconds"] for call in wait_mock.await_args_list],
            [2.0, 4.0, 8.0, 16.0],
        )


class FeaturesContractTests(IsolatedAsyncioTestCase):
    async def test_features_exposes_desktop_oauth_capabilities(self) -> None:
        with mock.patch.object(features_route.system_config, "generic_mode", False):
            payload = await features_route.get_features()
        self.assertIn("openai_codex_oauth_available", payload)
        self.assertTrue(payload["openai_codex_oauth_available"])
        self.assertIn("xai_grok_oauth_available", payload)
        self.assertTrue(payload["xai_grok_oauth_available"])
        self.assertIn("sakana_fugu_available", payload)
        self.assertTrue(payload["sakana_fugu_available"])
        self.assertIn("kimi_code_available", payload)
        self.assertTrue(payload["kimi_code_available"])


class CloudAccessStatusTests(IsolatedAsyncioTestCase):
    async def test_status_survives_one_oauth_provider_status_failure(self) -> None:
        async def fail_xai_status():
            raise RuntimeError("corrupt token payload")

        async def codex_status():
            return {"configured": True, "email": "user@example.com"}

        with (
            mock.patch.object(cloud_access_route.system_config, "generic_mode", False),
            mock.patch.object(cloud_access_route.openai_codex_client, "status", codex_status),
            mock.patch.object(cloud_access_route.xai_grok_client, "status", fail_xai_status),
        ):
            payload = await cloud_access_route.get_cloud_access_status()

        self.assertTrue(payload["providers"]["openai_codex_oauth"]["configured"])
        self.assertFalse(payload["providers"]["xai_grok_oauth"]["configured"])
        self.assertIn("status_error", payload["providers"]["xai_grok_oauth"])


class CodexCallbackServerStateTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cloud_access_route._CODEX_CALLBACK_SERVER_STATE.server = None
        cloud_access_route._PENDING_CODEX_OAUTH.clear()

    async def asyncTearDown(self) -> None:
        server = cloud_access_route._CODEX_CALLBACK_SERVER_STATE.server
        if server is not None:
            server.close()
            await server.wait_closed()
        cloud_access_route._CODEX_CALLBACK_SERVER_STATE.server = None
        cloud_access_route._PENDING_CODEX_OAUTH.clear()

    async def test_callback_server_start_is_serialized(self) -> None:
        class FakeServer:
            def __init__(self) -> None:
                self.closed = False

            def is_serving(self) -> bool:
                return not self.closed

            def close(self) -> None:
                self.closed = True

            async def wait_closed(self) -> None:
                return None

        calls = 0

        async def fake_start_server(*args, **kwargs):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return FakeServer()

        with mock.patch.object(cloud_access_route.asyncio, "start_server", side_effect=fake_start_server):
            results = await asyncio.gather(
                cloud_access_route._ensure_codex_callback_server(),
                cloud_access_route._ensure_codex_callback_server(),
            )

        self.assertEqual(results, [True, True])
        self.assertEqual(calls, 1)

    async def test_codex_oauth_rejects_custom_redirect_uri_without_pending_state(self) -> None:
        with mock.patch.object(cloud_access_route.system_config, "generic_mode", False):
            with self.assertRaises(HTTPException) as ctx:
                await cloud_access_route.start_openai_codex_oauth(
                    cloud_access_route.CodexOAuthStartRequest(redirect_uri="https://example.com/callback")
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(cloud_access_route._PENDING_CODEX_OAUTH, {})

    async def test_manual_terminal_exchange_failure_consumes_pending_state(self) -> None:
        class FakeServer:
            def __init__(self) -> None:
                self.closed = False
                self.waited = False

            def close(self) -> None:
                self.closed = True

            async def wait_closed(self) -> None:
                self.waited = True

        fake_server = FakeServer()
        cloud_access_route._CODEX_CALLBACK_SERVER_STATE.server = fake_server
        cloud_access_route._PENDING_CODEX_OAUTH["state-1"] = {
            "code_verifier": "verifier",
            "redirect_uri": OpenAICodexClient.DEFAULT_REDIRECT_URI,
            "expires_at": 9999999999,
        }

        async def fail_exchange(**_kwargs):
            raise OpenAICodexAuthError("exchange failed")

        with mock.patch.object(cloud_access_route.system_config, "generic_mode", False):
            with mock.patch.object(cloud_access_route.openai_codex_client, "exchange_code", fail_exchange):
                with self.assertRaises(HTTPException):
                    await cloud_access_route.exchange_openai_codex_oauth(
                        cloud_access_route.CodexOAuthExchangeRequest(code="code", state="state-1")
                    )

        self.assertTrue(fake_server.closed)
        self.assertTrue(fake_server.waited)
        self.assertIsNone(cloud_access_route._CODEX_CALLBACK_SERVER_STATE.server)
        self.assertNotIn("state-1", cloud_access_route._PENDING_CODEX_OAUTH)

    async def test_manual_unexpected_exchange_failure_retains_pending_state(self) -> None:
        cloud_access_route._PENDING_CODEX_OAUTH["state-retry"] = {
            "code_verifier": "verifier",
            "redirect_uri": OpenAICodexClient.DEFAULT_REDIRECT_URI,
            "expires_at": 9999999999,
        }

        async def fail_exchange(**_kwargs):
            raise RuntimeError("temporary persistence failure")

        with mock.patch.object(cloud_access_route.system_config, "generic_mode", False):
            with mock.patch.object(cloud_access_route.openai_codex_client, "exchange_code", fail_exchange):
                with self.assertRaises(RuntimeError):
                    await cloud_access_route.exchange_openai_codex_oauth(
                        cloud_access_route.CodexOAuthExchangeRequest(
                            code="code",
                            state="state-retry",
                        )
                    )

        self.assertIn("state-retry", cloud_access_route._PENDING_CODEX_OAUTH)


class XAIGrokCallbackServerStateTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cloud_access_route._XAI_GROK_CALLBACK_SERVER_STATE.server = None
        cloud_access_route._PENDING_XAI_GROK_OAUTH.clear()

    async def asyncTearDown(self) -> None:
        server = cloud_access_route._XAI_GROK_CALLBACK_SERVER_STATE.server
        if server is not None:
            server.close()
            await server.wait_closed()
        cloud_access_route._XAI_GROK_CALLBACK_SERVER_STATE.server = None
        cloud_access_route._PENDING_XAI_GROK_OAUTH.clear()

    def test_callback_binding_uses_configured_redirect_uri(self) -> None:
        with mock.patch.object(
            cloud_access_route.xai_grok_client,
            "DEFAULT_REDIRECT_URI",
            "http://localhost:61234/custom-callback",
        ):
            host, port, path = cloud_access_route._xai_grok_callback_binding()

        self.assertEqual(host, "localhost")
        self.assertEqual(port, 61234)
        self.assertEqual(path, "/custom-callback")

    async def test_callback_server_binds_to_configured_redirect_uri(self) -> None:
        class FakeServer:
            def __init__(self) -> None:
                self.closed = False

            def is_serving(self) -> bool:
                return not self.closed

            def close(self) -> None:
                self.closed = True

            async def wait_closed(self) -> None:
                return None

        calls = []

        async def fake_start_server(*args, **kwargs):
            calls.append(kwargs)
            return FakeServer()

        with (
            mock.patch.object(
                cloud_access_route.xai_grok_client,
                "DEFAULT_REDIRECT_URI",
                "http://localhost:61234/custom-callback",
            ),
            mock.patch.object(cloud_access_route.asyncio, "start_server", side_effect=fake_start_server),
        ):
            result = await cloud_access_route._ensure_xai_grok_callback_server()

        self.assertTrue(result)
        self.assertEqual(calls[0]["host"], "localhost")
        self.assertEqual(calls[0]["port"], 61234)

    def test_callback_binding_rejects_non_loopback_redirect_uri(self) -> None:
        with mock.patch.object(
            cloud_access_route.xai_grok_client,
            "DEFAULT_REDIRECT_URI",
            "https://example.com/callback",
        ):
            with self.assertRaises(ValueError):
                cloud_access_route._xai_grok_callback_binding()


class CodexSecretStoreTests(IsolatedAsyncioTestCase):
    def test_codex_tokens_are_chunked_for_windows_keyring_limits(self) -> None:
        stored = {}

        def fake_get_password(service_name, secret_name):
            return stored.get((service_name, secret_name))

        def fake_set_password(service_name, secret_name, secret_value):
            self.assertLessEqual(len(secret_value), 1000)
            stored[(service_name, secret_name)] = secret_value

        def fake_delete_password(service_name, secret_name):
            key = (service_name, secret_name)
            if key not in stored:
                raise PasswordDeleteError("missing")
            del stored[key]

        tokens = {
            "access_token": "a" * 3200,
            "refresh_token": "r" * 3200,
            "id_token": "i" * 3200,
            "expires_at": 123,
        }

        with mock.patch.object(secret_store.keyring, "get_password", fake_get_password):
            with mock.patch.object(secret_store.keyring, "set_password", fake_set_password):
                with mock.patch.object(secret_store.keyring, "delete_password", fake_delete_password):
                    secret_store.store_openai_codex_oauth_tokens(tokens)
                    loaded = secret_store.load_openai_codex_oauth_tokens()

        self.assertEqual(loaded, tokens)
        chunk_entries = [
            key for (_service, key) in stored
            if key.startswith("openai_codex_oauth_chunk_")
        ]
        self.assertGreater(len(chunk_entries), 1)

    def test_scalar_marker_failure_removes_uncommitted_candidate(self) -> None:
        stored = {}

        def fake_get_password(service_name, secret_name):
            return stored.get((service_name, secret_name))

        def fake_set_password(service_name, secret_name, secret_value):
            if secret_name == "openrouter_api_key_active":
                raise secret_store.KeyringError("marker failed")
            stored[(service_name, secret_name)] = secret_value

        def fake_delete_password(service_name, secret_name):
            key = (service_name, secret_name)
            if key not in stored:
                raise PasswordDeleteError("missing")
            del stored[key]

        with mock.patch.object(secret_store.keyring, "get_password", fake_get_password):
            with mock.patch.object(secret_store.keyring, "set_password", fake_set_password):
                with mock.patch.object(secret_store.keyring, "delete_password", fake_delete_password):
                    with self.assertRaises(secret_store.SecretStoreError):
                        secret_store.store_openrouter_api_key("candidate-secret")

        self.assertNotIn(
            (secret_store.get_active_service_name(), "openrouter_api_key_a"),
            stored,
        )

    def test_chunked_marker_failure_removes_uncommitted_candidate(self) -> None:
        stored = {}

        def fake_get_password(service_name, secret_name):
            return stored.get((service_name, secret_name))

        def fake_set_password(service_name, secret_name, secret_value):
            if secret_name == "openai_codex_oauth_chunk_count_active":
                raise secret_store.KeyringError("marker failed")
            stored[(service_name, secret_name)] = secret_value

        def fake_delete_password(service_name, secret_name):
            key = (service_name, secret_name)
            if key not in stored:
                raise PasswordDeleteError("missing")
            del stored[key]

        with mock.patch.object(secret_store.keyring, "get_password", fake_get_password):
            with mock.patch.object(secret_store.keyring, "set_password", fake_set_password):
                with mock.patch.object(secret_store.keyring, "delete_password", fake_delete_password):
                    with self.assertRaises(secret_store.SecretStoreError):
                        secret_store.store_openai_codex_oauth_tokens({
                            "access_token": "a" * 2400,
                            "refresh_token": "r" * 2400,
                        })

        service_name = secret_store.get_active_service_name()
        self.assertFalse(
            any(
                service == service_name
                and secret_name.startswith("openai_codex_oauth_chunk_a")
                for service, secret_name in stored
            )
        )
