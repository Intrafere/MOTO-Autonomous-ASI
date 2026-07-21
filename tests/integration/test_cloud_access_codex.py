import asyncio
import base64
import json
import tempfile
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException
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
from backend.shared.xai_grok_client import XAIGrokClient, XAIGrokRequestError


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"header.{encoded}.sig"


class ProviderNotificationStoreTests(IsolatedAsyncioTestCase):
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
            OpenAICodexClient._resolve_model_request("gpt-5.6-luna", None),
            ("gpt-5.6-sol", "high"),
        )
        self.assertEqual(
            OpenAICodexClient._resolve_model_request("gpt-5.6-terra", None),
            ("gpt-5.6-sol", "medium"),
        )

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
            mock.patch("backend.shared.openai_codex_client.asyncio.sleep", return_value=None) as sleep_mock,
        ):
            with self.assertRaisesRegex(OpenAICodexRequestError, "after 4 retries"):
                await client.generate_completion(
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 5)
        sleep_mock.assert_has_awaits([mock.call(2.0), mock.call(4.0), mock.call(8.0), mock.call(16.0)])

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
            mock.patch("backend.shared.openai_codex_client.asyncio.sleep", return_value=None) as sleep_mock,
        ):
            response = await client.generate_completion(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "user"}],
            )

        self.assertEqual(fake_http.calls, 2)
        sleep_mock.assert_awaited_once_with(2.0)
        self.assertEqual(response["choices"][0]["message"]["content"], "recovered")

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


class XAIGrokClientTests(IsolatedAsyncioTestCase):
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
            mock.patch("backend.shared.xai_grok_client.asyncio.sleep", return_value=None) as sleep_mock,
        ):
            with self.assertRaisesRegex(XAIGrokRequestError, "after 4 retries"):
                await client.generate_completion(
                    model="grok-4.3",
                    messages=[{"role": "user", "content": "user"}],
                )

        self.assertEqual(fake_http.calls, 5)
        sleep_mock.assert_has_awaits([mock.call(2.0), mock.call(4.0), mock.call(8.0), mock.call(16.0)])


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

    async def test_manual_exchange_failure_releases_callback_server_when_pending_consumed(self) -> None:
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
