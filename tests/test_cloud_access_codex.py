import asyncio
import base64
import json
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException
from keyring.errors import PasswordDeleteError

from backend.api.routes import cloud_access as cloud_access_route
from backend.api.routes import features as features_route
from backend.shared import secret_store
from backend.shared.openai_codex_client import OpenAICodexAuthError, OpenAICodexClient


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"header.{encoded}.sig"


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
        self.assertIn("originator=moto", url)
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

    async def test_list_models_fallback_does_not_invent_160k_limits(self) -> None:
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

        by_id = {model["id"]: model for model in models}
        self.assertEqual(by_id["gpt-5.5"]["context_length"], 400000)
        self.assertEqual(by_id["gpt-5.5"]["max_output_tokens"], 128000)
        self.assertNotIn("context_length", by_id["gpt-5.4"])

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


class FeaturesContractTests(IsolatedAsyncioTestCase):
    async def test_features_exposes_codex_oauth_capability(self) -> None:
        with mock.patch.object(features_route.system_config, "generic_mode", False):
            payload = await features_route.get_features()
        self.assertIn("openai_codex_oauth_available", payload)
        self.assertTrue(payload["openai_codex_oauth_available"])


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
