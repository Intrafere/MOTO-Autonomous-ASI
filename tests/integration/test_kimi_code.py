"""Kimi Code subscription provider: request shaping, region detection, and recovery typing."""
import json
from pathlib import Path
import tempfile
import time
from typing import Callable, List
from unittest import IsolatedAsyncioTestCase, mock

import httpx

from backend.shared.config import system_config
from backend.shared.model_error_utils import is_provider_context_length_error
from backend.shared.subscription_providers.api_key_client import SubscriptionApiKeyClient
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionEntitlementError,
    SubscriptionRequestError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.kimi_code import KIMI_CODE_SETTINGS, KimiCodeProfile

_CLIENT_MODULE = "backend.shared.subscription_providers.api_key_client"
INTL = "https://api.kimi.ai/coding/v1"
CN = "https://api.kimi.com/coding/v1"
KEY = "sk-kimi-test-0123456789abcdef"


def sse(*chunks: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def completion_chunks(text: str = "OK", *, finish: str | None = "stop", reasoning: str = "", tool_calls=None) -> List[dict]:
    delta = {"role": "assistant"}
    if reasoning:
        delta["reasoning_content"] = reasoning
    chunks = [{"id": "c1", "model": "k3", "choices": [{"index": 0, "delta": delta}]}]
    if tool_calls:
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": tool_calls}}]})
    if text:
        chunks.append({"choices": [{"index": 0, "delta": {"content": text}}]})
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    chunks.append(final)
    return chunks


class KimiClientHarness(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._data_dir = tempfile.TemporaryDirectory()
        self._patches = [
            mock.patch.object(system_config, "data_dir", self._data_dir.name),
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch(f"{_CLIENT_MODULE}.load_subscription_api_key", return_value=None),
        ]
        for patcher in self._patches:
            patcher.start()
        self.profile = KimiCodeProfile(KIMI_CODE_SETTINGS)
        self.client = SubscriptionApiKeyClient(self.profile)
        self.client._api_key = KEY
        self.client._region_id = "international"
        self.requests: List[httpx.Request] = []

    def tearDown(self) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()
        self._data_dir.cleanup()

    def use(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        def recording(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        self.client.client = httpx.AsyncClient(transport=httpx.MockTransport(recording))

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


class KimiRequestShapingTests(KimiClientHarness):
    def test_reasoning_effort_maps_to_kimi_values(self) -> None:
        normalize = self.profile.normalize_reasoning_effort
        self.assertIsNone(normalize("auto", model="k3"))
        self.assertEqual(normalize("max", model="k3"), "max")
        self.assertEqual(normalize("xhigh", model="k3"), "max")
        self.assertEqual(normalize("medium", model="k3"), "high")
        self.assertEqual(normalize("minimal", model="k3"), "low")
        self.assertEqual(normalize("none", model="k3"), "none")
        self.assertIsNone(normalize("max", model="kimi-for-coding-highspeed"))

    async def test_streamed_completion_is_normalized_and_identifies_moto(self) -> None:
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks("Hello", reasoning="think"))))
        result = await self.client.generate_completion(
            model="k3",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="max",
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["total_tokens"], 7)
        self.assertEqual(result["_moto_wire_api"], "chat.completions.stream")
        request = self.requests[0]
        self.assertEqual(str(request.url), f"{INTL}/chat/completions")
        self.assertTrue(request.headers["User-Agent"].startswith("MOTO-Autonomous-ASI/"))
        self.assertNotIn("Kimi", request.headers["User-Agent"])
        body = self.body()
        self.assertTrue(body["stream"])
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertNotIn("thinking", body)

    async def test_effort_none_turns_thinking_off(self) -> None:
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks())))
        await self.client.generate_completion(
            model="k3", messages=[{"role": "user", "content": "hi"}], reasoning_effort="none",
        )
        body = self.body()
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", body)

    async def test_tool_call_reasoning_is_sent_back_and_missing_reasoning_disables_thinking(self) -> None:
        tool_calls = [{"index": 0, "id": "call_1", "type": "function",
                       "function": {"name": "lookup", "arguments": "{}"}}]
        responses = [
            httpx.Response(200, content=sse(*completion_chunks("", finish="tool_calls", reasoning="plan", tool_calls=tool_calls))),
            httpx.Response(200, content=sse(*completion_chunks("42"))),
            httpx.Response(200, content=sse(*completion_chunks("43"))),
        ]
        self.use(lambda request: responses.pop(0))
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        first = await self.client.generate_completion(
            model="k3", messages=[{"role": "user", "content": "go"}], tools=tools, reasoning_effort="high",
        )
        message = first["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["id"], "call_1")
        # The caller rebuilds the assistant turn without reasoning_content, as the Wolfram loop does.
        follow_up = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": message["tool_calls"]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
        await self.client.generate_completion(model="k3", messages=follow_up, tools=tools, reasoning_effort="high")
        sent = self.body()
        self.assertEqual(sent["messages"][1]["reasoning_content"], "plan")
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertNotIn("reasoning_content", follow_up[1])

        unknown = [dict(turn) for turn in follow_up]
        unknown[1] = {**unknown[1], "tool_calls": [{**message["tool_calls"][0], "id": "call_unknown"}]}
        await self.client.generate_completion(model="k3", messages=unknown, tools=tools, reasoning_effort="high")
        sent = self.body()
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", sent)

    async def test_stream_without_finish_reason_is_retried_then_reported_as_transient(self) -> None:
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks(finish=None))))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            with self.assertRaises(SubscriptionRequestError) as ctx:
                await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(self.requests), KIMI_CODE_SETTINGS.max_retries)
        self.assertEqual(wait.await_count, KIMI_CODE_SETTINGS.max_retries - 1)
        self.assertEqual(wait.await_args.args[0]["reason"], "stream_incomplete")
        self.assertIn("Kimi Code connection failed after retries", str(ctx.exception))


class KimiErrorClassificationTests(KimiClientHarness):
    def _error(self, status: int, message: str) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message}})

    async def test_five_hour_limit_reads_reset_from_usage_and_is_not_retried(self) -> None:
        reset_at = int(time.time()) + 3600

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/usages"):
                return httpx.Response(200, json={"usages": {
                    "limit_5h": {"used_ratio": 1.0, "reset_time": reset_at},
                    "limit_month_total": {"used_ratio": 0.2, "reset_time": reset_at + 86400},
                }})
            return self._error(403, "You've reached your 5-hour usage limit")

        self.use(handler)
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock()) as wait:
            with self.assertRaises(SubscriptionUsageLimitError) as ctx:
                await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        wait.assert_not_awaited()
        self.assertEqual(ctx.exception.resets_at, reset_at)
        self.assertEqual(ctx.exception.provider, "kimi_code")
        self.assertEqual(ctx.exception.plan_type, "five_hour")
        completion_requests = [r for r in self.requests if r.url.path.endswith("/chat/completions")]
        self.assertEqual(len(completion_requests), 1)

    async def test_usage_limit_without_credible_reset_requires_repair(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/usages"):
                return httpx.Response(500, text="usage backend down")
            return self._error(403, "You've reached your monthly usage limit for this billing cycle")

        self.use(handler)
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)):
            with self.assertRaises(SubscriptionEntitlementError) as ctx:
                await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertNotIsInstance(ctx.exception, SubscriptionUsageLimitError)
        self.assertIn("Kimi Code console", ctx.exception.terminal_guidance)

    async def test_concurrency_limit_is_retried(self) -> None:
        responses = [
            self._error(403, "You've reached your concurrent request limit"),
            httpx.Response(200, content=sse(*completion_chunks())),
        ]
        self.use(lambda request: responses.pop(0))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            result = await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")
        wait.assert_awaited_once()

    async def test_token_limit_is_a_context_limit(self) -> None:
        self.use(lambda request: self._error(400, "Your request exceeded model token limit: 262144"))
        with self.assertRaises(SubscriptionContextLimitError) as ctx:
            await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertTrue(is_provider_context_length_error(ctx.exception))

    async def test_plan_context_cap_names_the_fix(self) -> None:
        self.use(lambda request: self._error(401, "Your current plan supports only kimi-k3 up to 256K context"))
        with self.assertRaises(SubscriptionEntitlementError) as ctx:
            await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertIn("k3-256k", ctx.exception.terminal_guidance)

    async def test_echoed_key_never_reaches_error_text(self) -> None:
        self.use(lambda request: httpx.Response(400, json={"error": {"message": f"high risk input {KEY}"}}))
        with self.assertRaises(SubscriptionRequestError) as ctx:
            await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertNotIn(KEY, str(ctx.exception))


class KimiRegionDetectionTests(KimiClientHarness):
    async def test_detects_china_region_when_international_rejects_the_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).startswith(INTL):
                return httpx.Response(401, json={"error": {"message": "Invalid Authentication"}})
            return httpx.Response(200, json={"data": [{"id": "k3"}, {"id": "kimi-for-coding"}]})

        self.use(handler)
        validation = await self.client.validate_candidate(f"  {KEY}  ")
        self.assertEqual(validation.region_id, "china")
        self.assertEqual([model["id"] for model in validation.models], ["k3", "kimi-for-coding"])
        self.assertEqual(validation.models[0]["context_length"], 1_048_576)
        self.assertNotIn("max_output_tokens", validation.models[0])
        self.assertEqual([str(r.url) for r in self.requests], [f"{INTL}/models", f"{CN}/models"])

    async def test_rejected_in_both_regions_explains_platform_keys(self) -> None:
        self.use(lambda request: httpx.Response(401, json={"error": {"message": "Invalid Authentication"}}))
        with self.assertRaises(SubscriptionAuthError) as ctx:
            await self.client.validate_candidate(KEY)
        self.assertIn("Kimi Platform", str(ctx.exception))

    async def test_missing_model_list_validates_through_usage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(404, text="method not found")
            return httpx.Response(200, json={"usages": {}})

        self.use(handler)
        validation = await self.client.validate_candidate(KEY)
        self.assertEqual(validation.region_id, "international")
        self.assertEqual({model["id"] for model in validation.models},
                         {"k3", "k3-256k", "kimi-for-coding", "kimi-for-coding-highspeed"})

    async def test_saved_region_is_persisted_as_non_secret_runtime_setting(self) -> None:
        from backend.shared.runtime_settings import get_subscription_endpoint_region

        with (
            mock.patch(f"{_CLIENT_MODULE}.store_subscription_api_key") as store,
            mock.patch(f"{_CLIENT_MODULE}.load_subscription_api_key", return_value=KEY),
        ):
            self.client.set_api_key(KEY, region_id="china")
        store.assert_called_once_with("kimi_code_api_key", KEY)
        self.assertEqual(get_subscription_endpoint_region("kimi_code"), "china")
        self.assertEqual(self.client.base_url, CN)
        settings_text = (Path(self._data_dir.name) / "runtime_settings.json").read_text(encoding="utf-8")
        self.assertNotIn(KEY, settings_text)
        with mock.patch(f"{_CLIENT_MODULE}.clear_subscription_api_key"):
            await self.client.clear_api_key()
        self.assertEqual(get_subscription_endpoint_region("kimi_code"), "")


class KimiUsageParsingTests(IsolatedAsyncioTestCase):
    def test_parses_pools_with_iso_and_millisecond_reset_times(self) -> None:
        profile = KimiCodeProfile(KIMI_CODE_SETTINGS)
        usage = profile.parse_usage({"usages": {
            "limit_5h": {"used_ratio": 42, "reset_time": "2026-10-15T12:00:00Z"},
            "limit_month_code": {"used": 30, "limit": 60, "reset_time": 1793000000000},
            "unrelated": {"value": 1},
        }})
        pools = {pool["id"]: pool for pool in usage["pools"]}
        self.assertEqual(set(pools), {"limit_5h", "limit_month_code"})
        self.assertAlmostEqual(pools["limit_5h"]["used_ratio"], 0.42)
        self.assertEqual(pools["limit_5h"]["window"], "five_hour")
        self.assertEqual(pools["limit_month_code"]["reset_at"], 1793000000)
        self.assertAlmostEqual(pools["limit_month_code"]["used_ratio"], 0.5)
