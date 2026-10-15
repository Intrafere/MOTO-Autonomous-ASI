"""Focused production tests: provider/API-error activity names its full route.

These run the real ``APIClientManager`` and real ``OpenRouterClient`` with a
faked HTTP transport, a faked stop-aware sleep, a recording broadcaster and a
temporary data root. Every OpenRouter response shape, model and host name here
is synthetic. The assertions prove that retry activity names the workflow mode,
role, model, provider and (for OpenRouter) the configured and upstream host,
while exception text, error summaries and correlation identities stay free of
that display identity.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from unittest import IsolatedAsyncioTestCase, mock

import httpx

import backend.shared.api_client_manager as api_manager_module
from backend.shared.api_client_manager import (
    APIClientManager,
    ProviderCooldownError,
    RetryableProviderError,
)
from backend.shared.activity import normalize_activity_payload
from backend.shared.config import system_config
from backend.shared.models import BoostConfig, ModelConfig
from backend.shared.openai_codex_client import OAuthUsageLimitError
from backend.shared.openrouter_client import FreeModelExhaustedError, OpenRouterClient
from backend.shared.provider_notification_store import list_provider_notifications
from backend.shared.provider_route_identity import ROUTE_IDENTITY_FIELD_NAMES
from backend.shared.workflow_start_guard import WorkflowStartGuard

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ROLE = "aggregator_submitter_2"
TASK = "agg_sub2_001"
MODEL = "vendor/test-free:free"
ALT_MODEL = "vendor/test-alt:free"
ALT_MODEL_2 = "vendor/test-alt-2:free"
AUTO_SELECTOR_MODEL = "openrouter/free"
BOOST_MODEL = "vendor/test-boost"
FALLBACK_MODEL = "local-test-fallback"

UPSTREAM_ERROR_BODY = {
    "id": "gen-test",
    "error": {
        "code": 429,
        "message": "upstream overloaded",
        "metadata": {"provider_name": "TestHost"},
    },
}
UPSTREAM_ERROR_WITHOUT_METADATA = {
    "id": "gen-test",
    "error": {"code": 429, "message": "upstream overloaded"},
}
TOP_LEVEL_PROVIDER_BODY = {"id": "gen-test", "provider": "TestHost"}
FREE_MODEL_RATE_LIMIT_BODY = {
    "error": {"code": 429, "message": "Rate limit exceeded: free-models-per-min."},
}
NO_ENDPOINTS_BODY = {
    "error": {"code": 404, "message": "No endpoints found for vendor/test-free:free."},
}

ADAPTER_NO_CHOICES_SUMMARY = (
    "OpenRouter returned a successful response without completion choices "
    "(upstream 429: upstream overloaded)."
)
FINAL_NO_CHOICES_SUMMARY = "OpenRouter returned no completion choices after 3 attempts."
SUBMITTER_2_IDENTITY = (
    "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · host auto → TestHost]"
)


def _response(status_code: int, **kwargs: Any) -> Callable[[], httpx.Response]:
    def build() -> httpx.Response:
        return httpx.Response(
            status_code,
            request=httpx.Request("POST", OPENROUTER_URL),
            **kwargs,
        )

    return build


def _success(content: str = "recovered") -> Callable[[], httpx.Response]:
    return _response(200, json={"choices": [{"message": {"content": content}}]})


def _connect_error() -> httpx.Response:
    raise httpx.ConnectError("synthetic offline", request=httpx.Request("POST", OPENROUTER_URL))


class ScriptedOpenRouterTransport:
    """Fake OpenRouter HTTP transport scripted per requested model.

    Each model has an ordered list of response factories; the last one repeats.
    """

    def __init__(self) -> None:
        self._scripts: Dict[str, List[Callable[[], httpx.Response]]] = {}
        self.requests: List[Dict[str, Any]] = []

    def script(self, model: str, *responses: Callable[[], httpx.Response]) -> None:
        self._scripts[model] = list(responses)

    async def post(self, url: str, *, json: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
        # The adapter mutates its request payload on auto-fallback; keep each send.
        self.requests.append(copy.deepcopy(json))
        queue = self._scripts[json["model"]]
        factory = queue.pop(0) if len(queue) > 1 else queue[0]
        return factory()

    @property
    def models(self) -> List[str]:
        return [request["model"] for request in self.requests]


def _fake_completion_adapter(retry_payload: Dict[str, Any], content: str = "ok") -> SimpleNamespace:
    """A provider adapter that reports one retry wait, then succeeds."""

    async def generate_completion(*, retry_callback=None, **_kwargs: Any) -> Dict[str, Any]:
        await retry_callback(dict(retry_payload))
        return {"choices": [{"message": {"content": content}}]}

    return SimpleNamespace(
        generate_completion=generate_completion,
        extract_routing_metadata=lambda _result: {},
    )


class ProviderErrorRouteIdentityTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Retry activity is persisted by the provider notification store; keep
        # it in a temporary data root, never the real backend/data.
        self._data_dir = tempfile.TemporaryDirectory()
        self._stack = contextlib.ExitStack()
        enter = self._stack.enter_context
        enter(mock.patch.object(system_config, "data_dir", self._data_dir.name))
        enter(mock.patch.object(system_config, "generic_mode", False))
        enter(mock.patch.object(system_config, "agent_conversation_memory_enabled", False))
        enter(mock.patch.dict(OpenRouterClient._rate_limited_models, clear=True))
        enter(mock.patch.dict(OpenRouterClient._model_semaphores, clear=True))
        self.free_models = mock.Mock(
            looping_enabled=False,
            auto_selector_enabled=False,
            AUTO_SELECTOR_MODEL=AUTO_SELECTOR_MODEL,
        )
        self.free_models.is_account_exhausted.return_value = False
        enter(mock.patch.object(api_manager_module, "free_model_manager", self.free_models))
        self._use_workflow_owner("autonomous")

        self.manager = APIClientManager()
        self.manager._autonomous_logger_callback = None
        self.broadcasts: List[tuple[str, Dict[str, Any]]] = []
        self.hung_alert_seen = asyncio.Event()
        self.manager.set_broadcast_callback(self._record_broadcast)
        self.sleep = mock.AsyncMock(return_value=True)
        enter(mock.patch.object(self.manager, "_sleep_with_optional_stop", self.sleep))

        self.transport = ScriptedOpenRouterTransport()
        self._clients: List[OpenRouterClient] = []
        self.manager._openrouter_client = self._openrouter_client("test-key")

    async def asyncTearDown(self) -> None:
        for client in self._clients:
            await client.close()
        self._stack.close()
        self._data_dir.cleanup()

    def _use_workflow_owner(self, owner: str) -> None:
        guard = WorkflowStartGuard()
        with mock.patch("backend.shared.workflow_start_guard.sleep_inhibitor.acquire"):
            guard.commit(owner)
        self._stack.enter_context(
            mock.patch.object(api_manager_module, "workflow_start_guard", guard)
        )

    def _openrouter_client(self, api_key: str) -> OpenRouterClient:
        client = OpenRouterClient(api_key)
        client.client.post = self.transport.post
        self._clients.append(client)
        return client

    async def _record_broadcast(self, event: str, data: Dict[str, Any]) -> None:
        self.broadcasts.append((event, data))
        if event == "hung_connection_alert":
            self.hung_alert_seen.set()

    def _events(self, event_name: str) -> List[Dict[str, Any]]:
        return [data for event, data in self.broadcasts if event == event_name]

    def _retry_events(self) -> List[Dict[str, Any]]:
        return self._events("provider_retry_cooldown")

    def _configure_role(
        self,
        role_id: str = ROLE,
        *,
        provider: str = "openrouter",
        model: str = MODEL,
        host: Optional[str] = None,
        fallback: Optional[str] = None,
    ) -> None:
        self.manager.configure_role(
            role_id,
            ModelConfig(
                provider=provider,
                model_id=model,
                openrouter_model_id=model if provider == "openrouter" else None,
                openrouter_provider=host,
                lm_studio_fallback_id=fallback,
                context_window=32000,
                max_output_tokens=4000,
            ),
        )

    async def _generate(self, role_id: str = ROLE, task_id: str = TASK, **kwargs: Any) -> Dict[str, Any]:
        return await self.manager.generate_completion(
            task_id=task_id,
            role_id=role_id,
            model=MODEL,
            messages=[{"role": "user", "content": "hello"}],
            **kwargs,
        )

    async def _no_choices_retryable_error(self, role_id: str = ROLE) -> RetryableProviderError:
        with self.assertRaises(RetryableProviderError) as raised:
            await self._generate(role_id=role_id)
        return raised.exception

    def _assert_route_identity(self, payload: Dict[str, Any], **expected: Any) -> None:
        for key, value in expected.items():
            self.assertEqual(payload.get(key), value, key)
        identity = payload["route_identity"]
        self.assertTrue(payload["message"].startswith(f"{identity} "), payload["message"])
        self.assertEqual(payload["message"].count(identity), 1)
        self.assertNotIn(identity, payload.get("error_summary", ""))
        self.assertFalse(payload.get("error_summary", "").startswith("["))
        for display_value in (identity, payload.get("role_label"), payload.get("effective_host_provider")):
            if display_value:
                self.assertNotIn(display_value, payload["event_id"])
                self.assertNotIn(display_value, payload["correlation_id"])

    async def test_openrouter_no_choices_retries_carry_mode_role_model_and_upstream_host(self) -> None:
        self._configure_role()
        self.transport.script(MODEL, _response(200, json=UPSTREAM_ERROR_BODY))

        error = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(error, role_id=ROLE)

        retries = self._retry_events()
        self.assertEqual(len(retries), 3)
        self.assertEqual(self.transport.models, [MODEL] * 3)
        self.assertEqual(self.transport.requests[0]["provider"], {"ignore": ["Venice"]})
        for payload in retries:
            self._assert_route_identity(
                payload,
                workflow_mode="autonomous",
                role_id=ROLE,
                role_label="Submitter 2",
                model=MODEL,
                provider="openrouter",
                provider_label="OpenRouter",
                route_kind="primary",
                task_id=TASK,
                configured_host_provider="",
                host_routing="auto",
                effective_host_provider="TestHost",
                provider_request_id="gen-test",
                route_identity=SUBMITTER_2_IDENTITY,
            )
            self.assertNotIn("TestHost", payload["notification_key"])

        adapter_retries, backoff = retries[:2], retries[2]
        for attempt, payload in enumerate(adapter_retries, start=1):
            self.assertEqual(payload["retry_attempt"], attempt)
            self.assertEqual(payload["retry_layer"], "adapter")
            self.assertEqual(payload["error_summary"], ADAPTER_NO_CHOICES_SUMMARY)
            self.assertEqual(
                payload["message"],
                f"{SUBMITTER_2_IDENTITY} OpenRouter server-side error: OpenRouter returned a "
                "successful response without completion choices (upstream 429: upstream "
                f"overloaded). OpenRouter API call will retry after a {2 * attempt}-second "
                f"cooldown (retry {attempt}).",
            )
        self.assertEqual(adapter_retries[0]["correlation_id"], adapter_retries[1]["correlation_id"])
        self.assertNotEqual(adapter_retries[0]["event_id"], adapter_retries[1]["event_id"])
        self.assertNotEqual(backoff["correlation_id"], adapter_retries[0]["correlation_id"])
        self.assertEqual(backoff["retry_after_seconds"], 60)
        self.assertEqual(backoff["error_summary"], FINAL_NO_CHOICES_SUMMARY)
        self.assertEqual(
            backoff["message"],
            f"{SUBMITTER_2_IDENTITY} OpenRouter API error: OpenRouter returned no completion "
            "choices after 3 attempts. OpenRouter API call will retry after a 60-second "
            "cooldown (retry 1).",
        )
        self.assertEqual([call.args[0] for call in self.sleep.await_args_list], [2.0, 4.0, 60])

        # The typed error, its text and the backoff key carry no display identity.
        self.assertEqual(error.provider, "openrouter")
        self.assertEqual(error.model, MODEL)
        self.assertEqual(error.route_kind, "primary")
        self.assertEqual(error.configured_host_provider, "")
        self.assertEqual(error.upstream_host_provider, "TestHost")
        self.assertEqual(error.provider_request_id, "gen-test")
        self.assertEqual(
            str(error),
            "OpenRouter transient provider failure for role 'aggregator_submitter_2' after "
            f"internal retries: {FINAL_NO_CHOICES_SUMMARY}",
        )
        self.assertEqual(error.error_summary, FINAL_NO_CHOICES_SUMMARY)
        self.assertEqual(
            list(self.manager._retryable_provider_backoff_state),
            ["openrouter|aggregator_submitter_2|vendor/test-free:free|transient_provider_error"],
        )

    async def test_upstream_without_metadata_names_request_id(self) -> None:
        self._configure_role()
        self.transport.script(MODEL, _response(200, json=UPSTREAM_ERROR_WITHOUT_METADATA))

        error = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(error, role_id=ROLE)

        identity = (
            "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · "
            "host auto (upstream not reported; request gen-test)]"
        )
        retries = self._retry_events()
        self.assertEqual(len(retries), 3)
        for payload in retries:
            self._assert_route_identity(
                payload,
                host_routing="auto",
                configured_host_provider="",
                provider_request_id="gen-test",
                route_identity=identity,
            )
            self.assertNotIn("effective_host_provider", payload)
        self.assertEqual(retries[0]["error_summary"], ADAPTER_NO_CHOICES_SUMMARY)

    async def test_top_level_provider_key_names_upstream_host(self) -> None:
        self._configure_role()
        self.transport.script(MODEL, _response(200, json=TOP_LEVEL_PROVIDER_BODY))

        error = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(error, role_id=ROLE)

        retries = self._retry_events()
        self.assertEqual(len(retries), 3)
        for payload in retries:
            self._assert_route_identity(
                payload,
                host_routing="auto",
                effective_host_provider="TestHost",
                provider_request_id="gen-test",
                route_identity=SUBMITTER_2_IDENTITY,
            )
        self.assertEqual(
            retries[0]["error_summary"],
            "OpenRouter returned a successful response without completion choices.",
        )

    async def test_pinned_host_is_reported_as_pinned(self) -> None:
        self._configure_role(host="TestHost")
        self.transport.script(MODEL, _response(200, json=UPSTREAM_ERROR_BODY))

        error = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(error, role_id=ROLE)

        identity = (
            "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · "
            "host TestHost (pinned)]"
        )
        self.assertEqual(
            self.transport.requests[0]["provider"],
            {"order": ["TestHost"], "allow_fallbacks": False},
        )
        retries = self._retry_events()
        self.assertEqual(len(retries), 3)
        for payload in retries:
            self._assert_route_identity(
                payload,
                host_routing="pinned",
                configured_host_provider="TestHost",
                effective_host_provider="TestHost",
                route_identity=identity,
            )

    async def test_pinned_host_without_endpoints_reports_auto_fallback(self) -> None:
        self._configure_role("autonomous_assistant", host="PinnedHost")
        self.transport.script(
            MODEL,
            _response(404, json=NO_ENDPOINTS_BODY),
            _response(200, json=UPSTREAM_ERROR_BODY),
        )

        error = await self._no_choices_retryable_error(role_id="autonomous_assistant")
        await self.manager.wait_for_retryable_provider_error(error, role_id="autonomous_assistant")

        identity = (
            "[Autonomous · Assistant (autonomous) · vendor/test-free:free via OpenRouter · "
            "host auto (pinned PinnedHost unavailable) → TestHost]"
        )
        self.assertEqual(
            [request["provider"] for request in self.transport.requests],
            [
                {"order": ["PinnedHost"], "allow_fallbacks": False},
                {"ignore": ["Venice"]},
                {"ignore": ["Venice"]},
            ],
        )
        retries = self._retry_events()
        self.assertEqual(len(retries), 2)
        for payload in retries:
            self._assert_route_identity(
                payload,
                role_id="autonomous_assistant",
                role_label="Assistant (autonomous)",
                host_routing="auto_fallback",
                configured_host_provider="PinnedHost",
                effective_host_provider="TestHost",
                route_identity=identity,
            )
        self.assertEqual(error.auto_fallback_from_host, "PinnedHost")
        self.assertNotIn("PinnedHost", str(error))

    async def test_boost_route_retries_name_boost_model_and_route_kind(self) -> None:
        self._configure_role()
        self.transport.script(BOOST_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        self.transport.script(MODEL, _success("primary answer"))
        boost = SimpleNamespace(
            boost_config=BoostConfig(
                enabled=True,
                openrouter_api_key="test-boost-key",
                boost_model_id=BOOST_MODEL,
                boost_context_window=32000,
                boost_max_output_tokens=4000,
            ),
            consume_boost_count=mock.AsyncMock(),
        )

        with (
            mock.patch.object(api_manager_module, "boost_manager", boost),
            mock.patch.object(api_manager_module, "boost_logger", mock.AsyncMock()),
            mock.patch.object(api_manager_module, "OpenRouterClient", side_effect=self._openrouter_client),
        ):
            result = await self._generate(_moto_force_boost_mode="always_prefer")

        self.assertEqual(result["choices"][0]["message"]["content"], "primary answer")
        self.assertEqual(self.transport.models, [BOOST_MODEL] * 3 + [MODEL])
        identity = (
            "[Autonomous · Submitter 2 (boost) · vendor/test-boost via OpenRouter · "
            "host auto → TestHost]"
        )
        retries = self._retry_events()
        self.assertEqual(len(retries), 2)
        for payload in retries:
            self._assert_route_identity(
                payload,
                role_id=ROLE,
                role_label="Submitter 2 (boost)",
                model=BOOST_MODEL,
                route_kind="boost",
                configured_model=MODEL,
                configured_provider="openrouter",
                host_routing="auto",
                effective_host_provider="TestHost",
                route_identity=identity,
            )

    async def test_free_rotation_no_choices_retries_name_rotation_route(self) -> None:
        self._configure_role()
        self.free_models.looping_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, None]
        self.transport.script(
            MODEL,
            _response(200, json=UPSTREAM_ERROR_BODY),
            _response(429, json=FREE_MODEL_RATE_LIMIT_BODY),
        )
        self.transport.script(ALT_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))

        error = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(error, role_id=ROLE)

        self.assertEqual(self.transport.models, [MODEL, MODEL, ALT_MODEL, ALT_MODEL, ALT_MODEL])
        self.free_models.mark_model_failed.assert_any_call(MODEL)
        # No route is left and no LM Studio fallback exists, so the alternative's
        # transient failure is raised for workflow retry and the model is skipped.
        self.free_models.mark_model_failed.assert_any_call(ALT_MODEL)
        self.assertEqual(error.route_kind, "free_rotation")
        self.assertEqual(error.model, ALT_MODEL)
        self.assertEqual(error.configured_host_provider, "")

        retries = self._retry_events()
        self.assertEqual(len(retries), 4)
        primary, rotation = retries[0], retries[1:]
        self._assert_route_identity(
            primary,
            route_kind="primary",
            role_label="Submitter 2",
            model=MODEL,
            route_identity=SUBMITTER_2_IDENTITY,
        )
        rotation_identity = (
            "[Autonomous · Submitter 2 (free rotation) · vendor/test-alt:free via OpenRouter · "
            "host auto → TestHost]"
        )
        for payload in rotation:
            self._assert_route_identity(
                payload,
                workflow_mode="autonomous",
                role_id=ROLE,
                route_kind="free_rotation",
                role_label="Submitter 2 (free rotation)",
                model=ALT_MODEL,
                configured_model=MODEL,
                host_routing="auto",
                configured_host_provider="",
                effective_host_provider="TestHost",
                route_identity=rotation_identity,
            )
        self.assertEqual(rotation[0]["correlation_id"], rotation[1]["correlation_id"])
        self.assertNotEqual(rotation[0]["correlation_id"], primary["correlation_id"])
        self.assertNotEqual(rotation[2]["correlation_id"], rotation[0]["correlation_id"])

        rate_limit = self._events("openrouter_rate_limit")
        self.assertEqual(len(rate_limit), 1)
        self.assertEqual(rate_limit[0]["workflow_mode"], "autonomous")
        self.assertEqual(rate_limit[0]["role_label"], "Submitter 2")
        self.assertEqual(rate_limit[0]["host_routing"], "auto")
        self.assertEqual(
            rate_limit[0]["route_identity"],
            "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · "
            "host auto (upstream not reported)]",
        )
        rotated = self._events("free_model_rotated")
        self.assertEqual(len(rotated), 1)
        self.assertEqual(rotated[0]["route_kind"], "free_rotation")
        self.assertEqual(rotated[0]["role_label"], "Submitter 2 (free rotation)")
        self.assertEqual(rotated[0]["model"], ALT_MODEL)

    async def test_auto_selector_no_choices_retries_name_auto_selector_route(self) -> None:
        self._configure_role()
        self.free_models.auto_selector_enabled = True
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(AUTO_SELECTOR_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))

        error = await self._no_choices_retryable_error()

        self.assertEqual(error.route_kind, "auto_selector")
        self.assertEqual(error.model, AUTO_SELECTOR_MODEL)
        retries = self._retry_events()
        self.assertEqual(len(retries), 2)
        for payload in retries:
            self._assert_route_identity(
                payload,
                route_kind="auto_selector",
                role_label="Submitter 2 (auto-selector)",
                model=AUTO_SELECTOR_MODEL,
                configured_model=MODEL,
                host_routing="auto",
                route_identity=(
                    "[Autonomous · Submitter 2 (auto-selector) · openrouter/free via "
                    "OpenRouter · host auto → TestHost]"
                ),
            )
        selector = self._events("free_model_auto_selector_used")
        self.assertEqual(len(selector), 1)
        self.assertEqual(selector[0]["role_label"], "Submitter 2 (auto-selector)")

    async def test_exhausted_free_rotation_falls_back_to_lm_studio(self) -> None:
        self._configure_role(fallback=FALLBACK_MODEL)
        self.free_models.looping_enabled = True
        self.free_models.auto_selector_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, None]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(ALT_MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(AUTO_SELECTOR_MODEL, _response(402, json={"error": {"code": 402}}))
        lm_studio = _fake_completion_adapter({
            "provider": "lm_studio",
            "provider_label": "LM Studio",
            "model": FALLBACK_MODEL,
            "retry_attempt": 1,
            "max_attempts": 3,
            "retry_after_seconds": 1.0,
            "reason": "transient_http_error",
            "error_summary": "HTTP 400 transient",
        }, content="fallback answer")

        with mock.patch.object(api_manager_module, "lm_studio_client", lm_studio):
            result = await self._generate()

        self.assertEqual(result["choices"][0]["message"]["content"], "fallback answer")
        self.assertEqual(self.transport.models, [MODEL, ALT_MODEL, AUTO_SELECTOR_MODEL])
        self.assertEqual(len(self._events("free_model_auto_selector_used")), 1)
        retries = self._retry_events()
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["route_kind"], "fallback")
        self.assertEqual(retries[0]["role_label"], "Submitter 2 (LM Studio fallback)")

    async def test_exhausted_free_rotation_without_fallback_raises_free_model_exhausted(self) -> None:
        self._configure_role()
        self.free_models.looping_enabled = True
        self.free_models.auto_selector_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, None]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(ALT_MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(AUTO_SELECTOR_MODEL, _response(402, json={"error": {"code": 402}}))

        # Rotation must run rather than failing on an unexpected keyword argument.
        with self.assertRaises(FreeModelExhaustedError) as raised:
            await self._generate()

        self.assertEqual(self.transport.models, [MODEL, ALT_MODEL, AUTO_SELECTOR_MODEL])
        self.assertEqual(len(self._events("free_model_rotated")), 1)
        self.assertEqual(len(self._events("free_model_auto_selector_used")), 1)
        # The exhaustion carries its route as structured fields only, so workflow
        # lines can name the real role and model; the text is unchanged.
        error = raised.exception
        self.assertEqual(
            (error.role_id, error.model, error.provider, error.route_kind),
            (ROLE, MODEL, "openrouter", "free_rotation"),
        )
        self.assertEqual(error.configured_host_provider, "")
        self.assertEqual(
            str(error),
            f"All free model options exhausted for role '{ROLE}'. No LM Studio fallback configured.",
        )

    async def test_transient_rotation_failure_moves_on_to_next_alternative(self) -> None:
        self._configure_role()
        self.free_models.looping_enabled = True
        self.free_models.auto_selector_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, ALT_MODEL_2, None]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(ALT_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        self.transport.script(ALT_MODEL_2, _success("second alternative answer"))

        result = await self._generate()

        self.assertEqual(result["choices"][0]["message"]["content"], "second alternative answer")
        self.assertEqual(
            self.transport.models,
            [MODEL, ALT_MODEL, ALT_MODEL, ALT_MODEL, ALT_MODEL_2],
        )
        self.free_models.mark_model_failed.assert_any_call(ALT_MODEL)
        self.assertEqual(
            [payload["model"] for payload in self._events("free_model_rotated")],
            [ALT_MODEL, ALT_MODEL_2],
        )
        self.assertEqual(self._events("free_model_auto_selector_used"), [])

    async def test_transient_rotation_failure_moves_on_to_auto_selector(self) -> None:
        self._configure_role()
        self.free_models.looping_enabled = True
        self.free_models.auto_selector_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, None]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(ALT_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        self.transport.script(AUTO_SELECTOR_MODEL, _success("auto-selector answer"))

        result = await self._generate()

        self.assertEqual(result["choices"][0]["message"]["content"], "auto-selector answer")
        self.assertEqual(self.transport.models[-1], AUTO_SELECTOR_MODEL)
        self.free_models.mark_model_failed.assert_any_call(ALT_MODEL)
        self.assertEqual(len(self._events("free_model_auto_selector_used")), 1)

    async def test_transient_rotation_and_auto_selector_failures_use_lm_studio_fallback(self) -> None:
        self._configure_role(fallback=FALLBACK_MODEL)
        self.free_models.looping_enabled = True
        self.free_models.auto_selector_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, None]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(ALT_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        self.transport.script(AUTO_SELECTOR_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        lm_studio = _fake_completion_adapter({
            "provider": "lm_studio",
            "provider_label": "LM Studio",
            "model": FALLBACK_MODEL,
            "retry_attempt": 1,
            "max_attempts": 3,
            "retry_after_seconds": 1.0,
            "reason": "transient_http_error",
            "error_summary": "HTTP 400 transient",
        }, content="fallback answer")

        with mock.patch.object(api_manager_module, "lm_studio_client", lm_studio):
            result = await self._generate()

        self.assertEqual(result["choices"][0]["message"]["content"], "fallback answer")
        self.assertEqual(
            self.transport.models,
            [MODEL] + [ALT_MODEL] * 3 + [AUTO_SELECTOR_MODEL] * 3,
        )
        self.free_models.mark_model_failed.assert_any_call(ALT_MODEL)
        self.assertEqual(self._retry_events()[-1]["route_kind"], "fallback")

    async def test_transient_auto_selector_failure_uses_lm_studio_fallback(self) -> None:
        self._configure_role(fallback=FALLBACK_MODEL)
        self.free_models.auto_selector_enabled = True
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(AUTO_SELECTOR_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        lm_studio = _fake_completion_adapter({
            "provider": "lm_studio",
            "provider_label": "LM Studio",
            "model": FALLBACK_MODEL,
            "retry_attempt": 1,
            "max_attempts": 3,
            "retry_after_seconds": 1.0,
            "reason": "transient_http_error",
            "error_summary": "HTTP 400 transient",
        }, content="fallback answer")

        with mock.patch.object(api_manager_module, "lm_studio_client", lm_studio):
            result = await self._generate()

        self.assertEqual(result["choices"][0]["message"]["content"], "fallback answer")
        self.assertEqual(self.transport.models, [MODEL] + [AUTO_SELECTOR_MODEL] * 3)
        fallback = self._retry_events()[-1]
        self.assertEqual(fallback["route_kind"], "fallback")
        self.assertEqual(fallback["model"], FALLBACK_MODEL)

    async def test_transient_rotation_failure_without_fallback_is_retried_by_workflow(self) -> None:
        self._configure_role()
        self.free_models.looping_enabled = True
        self.free_models.auto_selector_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, None]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(ALT_MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        self.transport.script(AUTO_SELECTOR_MODEL, _response(402, json={"error": {"code": 402}}))

        error = await self._no_choices_retryable_error()

        self.assertEqual(error.route_kind, "free_rotation")
        self.assertEqual(error.model, ALT_MODEL)
        self.assertEqual(self.transport.models[-1], AUTO_SELECTOR_MODEL)
        self.assertIsNotNone(error.__cause__)

    async def test_unreachable_openrouter_ends_free_model_looping(self) -> None:
        self._configure_role()
        self.free_models.looping_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [ALT_MODEL, ALT_MODEL_2, None]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(ALT_MODEL, _connect_error)
        self.transport.script(ALT_MODEL_2, _success("unexpected"))

        error = await self._no_choices_retryable_error()

        self.assertEqual(error.route_kind, "free_rotation")
        self.assertEqual(error.model, ALT_MODEL)
        self.assertNotIn(ALT_MODEL_2, self.transport.models)

    async def test_rotated_success_clears_alternative_backoff(self) -> None:
        self._configure_role()
        self.free_models.looping_enabled = True
        self.free_models.get_alternative_free_model.side_effect = [
            ALT_MODEL, None,
            ALT_MODEL,
            ALT_MODEL, None,
        ]
        self.transport.script(MODEL, _response(429, json=FREE_MODEL_RATE_LIMIT_BODY))
        self.transport.script(
            ALT_MODEL,
            _response(200, json=UPSTREAM_ERROR_BODY),
            _response(200, json=UPSTREAM_ERROR_BODY),
            _response(200, json=UPSTREAM_ERROR_BODY),
            _success("rotated answer"),
            _response(200, json=UPSTREAM_ERROR_BODY),
        )
        backoff_key = f"openrouter|{ROLE}|{ALT_MODEL}|transient_provider_error"

        first = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(first, role_id=ROLE)
        self.assertEqual(self.manager._retryable_provider_backoff_state.get(backoff_key), 1)
        self.assertEqual(self._retry_events()[-1]["retry_after_seconds"], 60)

        result = await self._generate()
        self.assertEqual(result["choices"][0]["message"]["content"], "rotated answer")
        self.assertNotIn(backoff_key, self.manager._retryable_provider_backoff_state)

        again = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(again, role_id=ROLE)
        self.assertEqual(self.manager._retryable_provider_backoff_state.get(backoff_key), 1)
        self.assertEqual(self._retry_events()[-1]["retry_after_seconds"], 60)

    async def test_manual_proof_roles_name_their_own_workflow_under_any_owner(self) -> None:
        manual_role = "autonomous_proof_formalization_manual_brainstorm_manual_r_0123456789"
        for owner, owner_mode in (("autonomous", "autonomous"), ("manual_aggregator", "aggregator")):
            with self.subTest(owner=owner):
                self._use_workflow_owner(owner)
                manual = self.manager.route_identity_for_role(
                    manual_role, provider="openrouter", model=MODEL,
                )
                pseudo = self.manager.route_identity_for_role("manual_proof_paper")
                own_run = self.manager.route_identity_for_role(
                    "autonomous_proof_formalization_brainstorm", provider="openrouter", model=MODEL,
                )
                saved_paper = self.manager.route_identity_for_role(
                    "autonomous_proof_formalization_compiler_manual_paper",
                    provider="openrouter",
                    model=MODEL,
                )
                child = self.manager.route_identity_for_role(ROLE)

                self.assertEqual(manual["workflow_mode"], "manual_proof_check")
                self.assertTrue(manual["route_identity"].startswith("[Manual proof check · "))
                self.assertEqual(pseudo["workflow_mode"], "manual_proof_check")
                self.assertEqual(own_run["workflow_mode"], owner_mode)
                # ProofVerificationStage labels the saved-paper check compiler.
                self.assertEqual(saved_paper["workflow_mode"], "compiler")
                self.assertEqual(child["workflow_mode"], owner_mode)
                # An explicit valid mode still wins over the role signature.
                explicit = self.manager.route_identity_for_role(
                    manual_role, workflow_mode="leanoj", provider="openrouter", model=MODEL,
                )
                self.assertEqual(explicit["workflow_mode"], "leanoj")

    async def test_manual_proof_direct_provider_events_name_manual_workflow(self) -> None:
        manual_role = "autonomous_proof_identification_manual_paper_manual_r_0123456789"
        self._use_workflow_owner("autonomous")

        await self.manager.emit_and_wait_for_retry(
            {
                "provider": "openrouter",
                "provider_label": "OpenRouter",
                "model": MODEL,
                "retry_attempt": 1,
                "retry_after_seconds": 1.0,
                "reason": "transient_http_error",
                "error_summary": "HTTP 503 service unavailable",
            },
            role_id=manual_role,
        )
        await self.manager._broadcast_unrecoverable_codex_error(
            role_id=manual_role,
            model="gpt-test-codex",
            error=RuntimeError("synthetic codex failure"),
        )
        await self.manager._broadcast_unrecoverable_xai_grok_error(
            role_id=manual_role,
            model="grok-test",
            error=RuntimeError("synthetic grok failure"),
        )

        for event in ("provider_retry_cooldown", "openai_codex_oauth_error", "oauth_provider_error"):
            with self.subTest(event=event):
                (payload,) = self._events(event)
                self.assertEqual(payload["workflow_mode"], "manual_proof_check")
                self.assertTrue(payload["route_identity"].startswith("[Manual proof check · "))
        # Stable notification and in-memory dedupe keys keep their composition.
        self.assertEqual(
            self._events("openai_codex_oauth_error")[0]["notification_key"],
            f"openai_codex_oauth:{manual_role}:unrecoverable_codex_error:gpt-test-codex",
        )
        self.assertIn(
            f"xai_grok_oauth:autonomous:{manual_role}:unrecoverable_xai_grok_error:grok-test",
            self.manager._oauth_error_notified,
        )

    async def test_reused_provider_cooldown_gets_its_own_event_identity_and_workflow(self) -> None:
        provider = "openai_codex_oauth"
        first_role, second_role = "aggregator_submitter_1", "leanoj_final_solver"
        error = OAuthUsageLimitError(
            provider=provider,
            provider_label="OpenAI Codex",
            message="The usage limit has been reached.",
            plan_type="plus",
            resets_in_seconds=90,
        )
        self._use_workflow_owner("manual_aggregator")
        first = self.manager._mark_provider_cooldown(
            error, role_id=first_role, model="gpt-test-codex", task_id="agg_sub1_001",
        )
        await self.manager._broadcast_provider_usage_limit(first)

        self._use_workflow_owner("leanoj")
        cooldown = self.manager.get_provider_cooldown(provider)
        second = self.manager._cooldown_payload_for_role(
            cooldown, role_id=second_role, model="gpt-test-codex-mini", task_id="leanoj_final_001",
        )
        await self.manager._broadcast_provider_usage_limit(second)
        # The role that first hit the limit keeps its original event identity.
        repeat_first = self.manager._cooldown_payload_for_role(
            self.manager.get_provider_cooldown(provider),
            role_id=first_role,
            model="gpt-test-codex",
            task_id="agg_sub1_002",
        )

        self.assertEqual(first["workflow_mode"], "aggregator")
        self.assertEqual(second["workflow_mode"], "leanoj")
        self.assertTrue(second["route_identity"].startswith("[LeanOJ · "))
        self.assertEqual(second["task_id"], "leanoj_final_001")
        self.assertNotEqual(second["event_id"], first["event_id"])
        self.assertNotEqual(second["correlation_id"], first["correlation_id"])
        self.assertEqual(repeat_first["event_id"], first["event_id"])
        for display_value in (second["route_identity"], second["role_label"]):
            self.assertNotIn(display_value, second["event_id"])
            self.assertNotIn(display_value, second["correlation_id"])

        stored = [
            item
            for item in await asyncio.to_thread(list_provider_notifications)
            if item["event_type"] == "provider_cooldown_started"
        ]
        self.assertEqual(
            sorted(item["role_id"] for item in stored),
            sorted([first_role, second_role]),
        )
        self.assertEqual(len({item["notification_key"] for item in stored}), 2)
        broadcasts = self._events("oauth_provider_usage_limited")
        self.assertEqual(
            len({
                normalize_activity_payload("oauth_provider_usage_limited", payload)["activity_id"]
                for payload in broadcasts
            }),
            2,
        )

    async def test_cooldown_resume_names_the_waiting_roles_workflow(self) -> None:
        provider = "openai_codex_oauth"
        waiting_role = "leanoj_final_solver"
        self._configure_role(waiting_role, provider=provider, model="gpt-test-codex")
        self._use_workflow_owner("manual_aggregator")
        self.manager._mark_provider_cooldown(
            OAuthUsageLimitError(
                provider=provider,
                provider_label="OpenAI Codex",
                message="The usage limit has been reached.",
                resets_in_seconds=90,
            ),
            role_id="aggregator_submitter_1",
            model="gpt-test-codex",
        )
        self._use_workflow_owner("leanoj")
        reset_at = int(self.manager.get_provider_cooldown(provider)["cooldown_until"])

        async def expire_cooldown(_seconds: float, _should_stop=None) -> bool:
            self.manager._oauth_provider_cooldowns[provider]["cooldown_until"] = int(time.time()) - 1
            return True

        self.sleep.side_effect = expire_cooldown
        await self.manager.wait_for_provider_cooldown(
            ProviderCooldownError(
                provider=provider,
                provider_label="OpenAI Codex",
                role_id=waiting_role,
                model="gpt-test-codex",
                resets_at=reset_at,
                resets_in_seconds=30,
            ),
            role_id=waiting_role,
        )
        await self.manager._confirm_provider_usage_limit_resumed(
            provider=provider,
            provider_label="OpenAI Codex",
            role_id=waiting_role,
            model="gpt-test-codex",
        )

        (resumed,) = self._events("provider_usage_limit_resumed")
        self.assertEqual(resumed["workflow_mode"], "leanoj")
        self.assertTrue(resumed["route_identity"].startswith("[LeanOJ · "))

    async def test_lm_studio_fallback_retry_names_fallback_route(self) -> None:
        self._configure_role(fallback=FALLBACK_MODEL)
        self.transport.script(MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        lm_studio = _fake_completion_adapter({
            "provider": "lm_studio",
            "provider_label": "LM Studio",
            "model": FALLBACK_MODEL,
            "retry_attempt": 1,
            "max_attempts": 3,
            "retry_after_seconds": 1.0,
            "reason": "transient_http_error",
            "error_summary": "HTTP 400 transient",
        }, content="fallback answer")

        with mock.patch.object(api_manager_module, "lm_studio_client", lm_studio):
            result = await self._generate()

        self.assertEqual(result["choices"][0]["message"]["content"], "fallback answer")
        retries = self._retry_events()
        self.assertEqual([payload["route_kind"] for payload in retries], ["primary", "primary", "fallback"])
        fallback = retries[2]
        self._assert_route_identity(
            fallback,
            workflow_mode="autonomous",
            role_id=ROLE,
            role_label="Submitter 2 (LM Studio fallback)",
            model=FALLBACK_MODEL,
            provider="lm_studio",
            provider_label="LM Studio",
            configured_provider="openrouter",
            configured_model=MODEL,
            host_routing="direct",
            route_identity=(
                "[Autonomous · Submitter 2 (LM Studio fallback) · local-test-fallback via LM Studio]"
            ),
        )
        self.assertNotIn("configured_host_provider", fallback)
        self.assertNotIn("effective_host_provider", fallback)
        self.assertNotEqual(fallback["correlation_id"], retries[0]["correlation_id"])

    async def test_subscription_provider_retries_have_no_host_segment(self) -> None:
        self._use_workflow_owner("manual_compiler")
        cases = (
            ("openai_codex_oauth", "OpenAI Codex", "openai_codex_client", "gpt-test-codex"),
            ("xai_grok_oauth", "xAI Grok", "xai_grok_client", "grok-test"),
            ("sakana_fugu", "Sakana Fugu", "sakana_fugu_client", "fugu-test"),
        )
        for provider, label, client_name, model in cases:
            with self.subTest(provider=provider):
                self.broadcasts.clear()
                self._configure_role("compiler_writer", provider=provider, model=model)
                adapter = _fake_completion_adapter({
                    "provider": provider,
                    "provider_label": label,
                    "model": model,
                    "retry_attempt": 1,
                    "max_attempts": 3,
                    "retry_after_seconds": 2.0,
                    "reason": "transient_http_error",
                    "error_summary": "HTTP 503 service unavailable",
                })

                with mock.patch.object(api_manager_module, client_name, adapter):
                    await self._generate(role_id="compiler_writer", task_id="comp_writer_001")

                retries = self._retry_events()
                self.assertEqual(len(retries), 1)
                payload = retries[0]
                identity = f"[Compiler · Writing Submitter · {model} via {label}]"
                self._assert_route_identity(
                    payload,
                    workflow_mode="compiler",
                    role_id="compiler_writer",
                    role_label="Writing Submitter",
                    model=model,
                    provider=provider,
                    provider_label=label,
                    route_kind="primary",
                    host_routing="direct",
                    route_identity=identity,
                )
                self.assertNotIn("host", payload["route_identity"])
                self.assertNotIn("configured_host_provider", payload)
                self.assertEqual(payload["error_summary"], "HTTP 503 service unavailable")
        self.assertEqual(self.transport.requests, [])

    async def test_single_model_display_role_uses_model_role_label(self) -> None:
        self._use_workflow_owner("manual_aggregator")
        self._configure_role()
        self.transport.script(MODEL, _response(200, json=UPSTREAM_ERROR_BODY))

        error = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(
            error,
            role_id="aggregator_single_model",
        )

        backoff = self._retry_events()[-1]
        self._assert_route_identity(
            backoff,
            workflow_mode="aggregator",
            role_id="aggregator_single_model",
            model_role_id=ROLE,
            role_label="Submitter 2",
            route_identity=(
                "[Aggregator · Submitter 2 · vendor/test-free:free via OpenRouter · "
                "host auto → TestHost]"
            ),
        )
        self.assertIn(
            "openrouter|aggregator_submitter_2|vendor/test-free:free|transient_provider_error",
            self.manager._retryable_provider_backoff_state,
        )

    async def test_provider_cooldown_wait_names_route_without_host(self) -> None:
        self._use_workflow_owner("manual_compiler")
        self._configure_role("compiler_writer", provider="xai_grok_oauth", model="grok-test")
        reset_at = int(time.time()) + 300
        self.manager._oauth_provider_cooldowns["xai_grok_oauth"] = {
            "provider": "xai_grok_oauth",
            "cooldown_until": reset_at,
            "resets_at": reset_at,
        }
        error = ProviderCooldownError(
            provider="xai_grok_oauth",
            provider_label="xAI Grok",
            role_id="compiler_writer",
            model="grok-test",
            resets_at=reset_at,
            resets_in_seconds=30,
            route_kind="primary",
            task_id="comp_writer_001",
        )

        async def expire_cooldown(_seconds: float, _should_stop=None) -> bool:
            self.manager._oauth_provider_cooldowns["xai_grok_oauth"]["cooldown_until"] = (
                int(time.time()) - 1
            )
            return True

        self.sleep.side_effect = expire_cooldown
        await self.manager.wait_for_provider_cooldown(error, role_id="compiler_writer")

        retries = self._retry_events()
        self.assertEqual(len(retries), 1)
        self._assert_route_identity(
            retries[0],
            workflow_mode="compiler",
            role_id="compiler_writer",
            role_label="Writing Submitter",
            model="grok-test",
            provider="xai_grok_oauth",
            provider_label="xAI Grok",
            reason="usage_limit_cooldown",
            task_id="comp_writer_001",
            host_routing="direct",
            route_identity="[Compiler · Writing Submitter · grok-test via xAI Grok]",
        )
        self.assertEqual(str(error), "xAI Grok usage limit reached; resets in 30 seconds")

    async def test_activity_callback_payload_matches_broadcast_identity(self) -> None:
        proof_role = "autonomous_proof_formalization_brainstorm"
        self._configure_role(proof_role, host="TestHost")
        self.transport.script(MODEL, _response(200, json=UPSTREAM_ERROR_BODY))
        activity_callback = mock.AsyncMock()

        error = await self._no_choices_retryable_error(role_id=proof_role)
        await self.manager.wait_for_retryable_provider_error(
            error,
            role_id=proof_role,
            activity_callback=activity_callback,
        )

        self.assertEqual(
            [call.args[0] for call in activity_callback.await_args_list],
            ["waiting", "resuming"],
        )
        broadcast = self._retry_events()[-1]
        self._assert_route_identity(
            broadcast,
            role_label="Proof Formalization (brainstorm)",
            host_routing="pinned",
            route_identity=(
                "[Autonomous · Proof Formalization (brainstorm) · vendor/test-free:free via "
                "OpenRouter · host TestHost (pinned)]"
            ),
        )
        for _phase, activity in (call.args for call in activity_callback.await_args_list):
            for key in ROUTE_IDENTITY_FIELD_NAMES:
                self.assertEqual(activity.get(key), broadcast.get(key), key)
            self.assertEqual(activity["message"], broadcast["message"])

    async def test_notification_store_round_trip_keeps_route_identity(self) -> None:
        self._configure_role()
        self.transport.script(MODEL, _response(200, json=UPSTREAM_ERROR_BODY))

        error = await self._no_choices_retryable_error()
        await self.manager.wait_for_retryable_provider_error(error, role_id=ROLE)

        broadcasts = {payload["event_id"]: payload for payload in self._retry_events()}
        stored = [
            item
            for item in await asyncio.to_thread(list_provider_notifications)
            if item["event_type"] == "provider_retry_cooldown"
        ]
        self.assertEqual(len(stored), 3)
        for record in stored:
            broadcast = broadcasts[record["event_id"]]
            for key in ROUTE_IDENTITY_FIELD_NAMES:
                self.assertEqual(record.get(key), broadcast.get(key), key)
            self.assertEqual(record["message"], broadcast["message"])
            self.assertEqual(record["route_identity"], SUBMITTER_2_IDENTITY)
            self.assertTrue(record["message"].startswith(SUBMITTER_2_IDENTITY))
        self.assertTrue(all(payload["persistence_status"] == "durable" for payload in broadcasts.values()))

    async def test_hung_connection_alert_carries_route_identity(self) -> None:
        self._configure_role(host="TestHost")

        async def slow_call() -> str:
            await self.hung_alert_seen.wait()
            return "done"

        result = await self.manager._with_hung_connection_watchdog(
            slow_call(),
            role_id=ROLE,
            model=MODEL,
            provider="TestHost",
            timeout_seconds=0,
            provider_id="openrouter",
            route_kind="primary",
            task_id=TASK,
            configured_host_provider="TestHost",
        )

        self.assertEqual(result, "done")
        alerts = self._events("hung_connection_alert")
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert["provider"], "TestHost")
        self.assertEqual(alert["provider_label"], "OpenRouter")
        self.assertEqual(alert["workflow_mode"], "autonomous")
        self.assertEqual(alert["role_label"], "Submitter 2")
        self.assertEqual(alert["model"], MODEL)
        self.assertEqual(alert["host_routing"], "pinned")
        self.assertEqual(alert["configured_host_provider"], "TestHost")
        self.assertEqual(
            alert["route_identity"],
            "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · "
            "host TestHost (pinned)]",
        )
