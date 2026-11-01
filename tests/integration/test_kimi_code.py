"""Kimi Code subscription provider: request shaping, region detection, and recovery typing."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
from typing import Callable, List
from unittest import IsolatedAsyncioTestCase, mock

import httpx

from backend.shared import trade_data_boundary
from backend.shared.api_client_manager import APIClientManager
from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text
from backend.shared.model_error_utils import is_provider_context_length_error
from backend.shared.models import ModelConfig
from backend.shared.subscription_providers.api_key_client import SubscriptionApiKeyClient
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionEntitlementError,
    SubscriptionParameterRejectedError,
    SubscriptionRequestError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.kimi_code import KIMI_CODE_SETTINGS, KimiCodeProfile
from backend.shared.subscription_providers.registry import get_api_key_subscription_client

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

    async def test_temperature_is_never_sent_because_kimi_fixes_it(self) -> None:
        # Kimi answers any other value with 400 "invalid temperature: only 1 is allowed".
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks())))
        for temperature, effort in ((0.0, "max"), (0.4, "high"), (0.9, "none"), (1.0, None)):
            await self.client.generate_completion(
                model="k3",
                messages=[{"role": "user", "content": "hi"}],
                temperature=temperature,
                reasoning_effort=effort,
            )
            self.assertNotIn("temperature", self.body())

    async def test_request_matches_kimis_own_client(self) -> None:
        # Kimi's own client sends max_completion_tokens (max_tokens is deprecated) and
        # asks for streamed usage; fixed sampling parameters are never sent.
        usage_chunk = {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}}
        chunks = completion_chunks("Hi")
        chunks[-1] = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": None}
        chunks.append(usage_chunk)
        self.use(lambda request: httpx.Response(200, content=sse(*chunks)))
        result = await self.client.generate_completion(
            model="k3", messages=[{"role": "user", "content": "hi"}], max_tokens=32_000,
        )
        body = self.body()
        self.assertEqual(body["max_completion_tokens"], 32_000)
        self.assertEqual(body["stream_options"], {"include_usage": True})
        for fixed in ("max_tokens", "temperature", "top_p", "n", "presence_penalty", "frequency_penalty"):
            self.assertNotIn(fixed, body)
        self.assertEqual(result["usage"]["total_tokens"], 13)

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

    async def test_tool_call_turn_drops_empty_content(self) -> None:
        # Kimi rejects empty text content; the Wolfram loop rebuilds the turn with content "".
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks("done"))))
        call = {"id": "call_9", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        self.profile.observe_chat_message({"role": "assistant", "tool_calls": [call], "reasoning_content": "plan"})
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_9", "content": "42"},
        ]
        await self.client.generate_completion(model="k3", messages=messages)
        turn = self.body()["messages"][1]
        self.assertNotIn("content", turn)
        self.assertEqual(turn["reasoning_content"], "plan")
        self.assertEqual(messages[1]["content"], "")

    async def test_highspeed_tool_follow_up_never_disables_thinking(self) -> None:
        # HighSpeed always thinks and Kimi rejects thinking: disabled for it, so an
        # unknown tool turn gets the empty reasoning Kimi's own client sends instead.
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks("done"))))
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_new", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_new", "content": "42"},
        ]
        await self.client.generate_completion(
            model="kimi-for-coding-highspeed", messages=messages, reasoning_effort="none",
        )
        body = self.body()
        self.assertNotIn("thinking", body)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["messages"][1]["reasoning_content"], "")

    async def test_stream_without_finish_reason_is_retried_then_reported_as_transient(self) -> None:
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks(finish=None))))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            with self.assertRaises(SubscriptionRequestError) as ctx:
                await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(self.requests), KIMI_CODE_SETTINGS.max_retries)
        self.assertEqual(wait.await_count, KIMI_CODE_SETTINGS.max_retries - 1)
        self.assertEqual(wait.await_args.args[0]["reason"], "stream_incomplete")
        self.assertIn("Kimi Code connection failed after retries", str(ctx.exception))


_CACHE_KEY_SHAPE = re.compile(r"moto-[0-9a-f]{32}")


class KimiPromptCacheKeyTests(KimiClientHarness):
    """Kimi Code asks for a stable per-session ``prompt_cache_key``; it must be a non-secret digest."""

    async def send(self, scope, *, model: str = "k3", messages=None, **extra) -> dict:
        await self.client.generate_completion(
            model=model,
            messages=messages or [{"role": "user", "content": "hi"}],
            cache_scope=scope,
            **extra,
        )
        return self.body()

    def serve(self) -> None:
        self.use(lambda request: httpx.Response(200, content=sse(*completion_chunks())))

    async def test_key_is_sent_for_a_scope_and_omitted_without_one(self) -> None:
        self.serve()
        body = await self.send("aggregator_submitter_1\nrun-1")
        self.assertRegex(body["prompt_cache_key"], _CACHE_KEY_SHAPE)
        self.assertTrue(body["stream"])
        body = await self.send(None)
        self.assertNotIn("prompt_cache_key", body)
        body = await self.send("")
        self.assertNotIn("prompt_cache_key", body)

    async def test_key_is_stable_across_calls_requests_and_restarts(self) -> None:
        self.serve()
        scope = "aggregator_submitter_1\nrun-1"
        first = (await self.send(scope))["prompt_cache_key"]
        variants = (
            {"messages": [{"role": "user", "content": "a different, longer prompt"}]},
            {"max_tokens": 4096},
            {"reasoning_effort": "max"},
            {"reasoning_effort": "none"},
            {"model": "kimi-for-coding"},
            {"temperature": 0.7},
        )
        for variant in variants:
            with self.subTest(variant=sorted(variant)):
                self.assertEqual((await self.send(scope, **variant))["prompt_cache_key"], first)
        # A restarted backend resuming the same run must reuse the key.
        restarted = SubscriptionApiKeyClient(KimiCodeProfile(KIMI_CODE_SETTINGS))
        restarted._api_key = KEY
        restarted._region_id = "international"
        restarted.client = self.client.client
        await restarted.generate_completion(
            model="k3", messages=[{"role": "user", "content": "hi"}], cache_scope=scope,
        )
        self.assertEqual(self.body()["prompt_cache_key"], first)

    async def test_key_is_identical_on_every_retry_attempt_of_one_call(self) -> None:
        responses = iter([httpx.Response(503, json={"error": {"message": "busy"}})])
        self.use(lambda request: next(responses, None) or httpx.Response(200, content=sse(*completion_chunks())))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)):
            await self.send("aggregator_validator\nrun-1")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.body(0)["prompt_cache_key"], self.body(1)["prompt_cache_key"])

    async def test_key_separates_roles_and_runs(self) -> None:
        self.serve()
        keys = {
            (await self.send(scope))["prompt_cache_key"]
            for scope in (
                "aggregator_submitter_1\nrun-1",
                "aggregator_submitter_2\nrun-1",
                "aggregator_submitter_1\nrun-2",
                "aggregator_submitter_1\n",
                "aggregator_submitter_1",
            )
        }
        self.assertEqual(len(keys), 5)

    async def test_key_carries_no_secret_prompt_path_or_identifier(self) -> None:
        self.serve()
        prompt_like_session = "prove_the_goldbach_conjecture_for_alice_2026-10-15_21-02"
        private_values = (
            prompt_like_session,
            "aggregator_submitter_1",
            r"C:\Users\alice\Desktop\MOTO\backend\data",
            "/home/alice/moto/data",
            "alice@example.com",
            "moto_installation_" + "a1" * 16,
            "moto_root_" + "b2" * 16,
            "moto_proof_" + "c3" * 16,
        )
        scope = "\n".join((*private_values, KEY))
        body = await self.send(scope, messages=[{"role": "user", "content": "unrelated question"}])
        key = body["prompt_cache_key"]
        self.assertRegex(key, _CACHE_KEY_SHAPE)

        for private in (*private_values, KEY):
            self.assertNotIn(private, key)
        request = self.requests[0]
        wire_text = request.content.decode("utf-8") + "\n".join(
            f"{name}: {value}" for name, value in request.headers.items() if name.lower() != "authorization"
        )
        for private in (*private_values, KEY):
            self.assertNotIn(private, wire_text)

        # Leaves nothing for MOTO's own redaction layers to remove from logs or exports.
        with mock.patch.object(trade_data_boundary, "_local_identity_tokens", return_value=("alice",)):
            self.assertEqual(redact_log_text(key), key)
            self.assertEqual(trade_data_boundary.trade_sensitive_value_kinds(key), ())
            self.assertEqual(trade_data_boundary.redact_trade_text(key), key)

    def test_digest_is_domain_separated_and_deterministic(self) -> None:
        derive = KimiCodeProfile.cache_key_for_scope
        self.assertEqual(derive("role\nrun"), derive("role\nrun"))
        self.assertNotEqual(derive("role\nrun"), derive("role\nrun2"))
        undomained = hashlib.sha256(b"role\nrun").hexdigest()[:32]
        self.assertNotEqual(derive("role\nrun"), f"moto-{undomained}")


class KimiCacheScopeRoutingTests(IsolatedAsyncioTestCase):
    """The router hands Kimi an in-process role+run scope and leaves other providers untouched."""

    def setUp(self) -> None:
        self._data_dir = tempfile.TemporaryDirectory()
        self._patches = [
            mock.patch.object(system_config, "data_dir", self._data_dir.name),
            mock.patch.object(system_config, "generic_mode", False),
            mock.patch.object(system_config, "agent_conversation_memory_enabled", False),
        ]
        for patcher in self._patches:
            patcher.start()
        self.manager = APIClientManager()

    def tearDown(self) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()
        self._data_dir.cleanup()

    def configure(self, role_id: str, provider: str, model_id: str) -> None:
        self.manager.configure_role(
            role_id,
            ModelConfig(provider=provider, model_id=model_id, context_window=100_000, max_output_tokens=4_000),
        )

    async def call(self, role_id: str, model: str, task_id: str = "task_001", **kwargs):
        client = get_api_key_subscription_client(self.manager._role_model_configs[role_id].provider)
        success = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        with mock.patch.object(client, "generate_completion", mock.AsyncMock(return_value=success)) as generate:
            await self.manager.generate_completion(
                task_id=task_id,
                role_id=role_id,
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                **kwargs,
            )
        return generate

    async def test_kimi_roles_get_a_stable_scope_independent_of_the_task_id(self) -> None:
        self.configure("aggregator_submitter_1", "kimi_code", "k3")
        self.configure("aggregator_submitter_2", "kimi_code", "k3")
        with mock.patch.object(
            APIClientManager, "_assistant_run_id_for_workflow", new=mock.AsyncMock(return_value="run-1"),
        ):
            first = (await self.call("aggregator_submitter_1", "k3", "agg_sub1_001")).await_args.kwargs
            second = (await self.call("aggregator_submitter_1", "k3", "agg_sub1_002")).await_args.kwargs
            other_role = (await self.call("aggregator_submitter_2", "k3", "agg_sub2_001")).await_args.kwargs
        self.assertEqual(first["cache_scope"], second["cache_scope"])
        self.assertNotEqual(first["cache_scope"], other_role["cache_scope"])
        self.assertIn("aggregator_submitter_1", first["cache_scope"])
        self.assertIn("run-1", first["cache_scope"])

    async def test_a_new_run_changes_the_scope(self) -> None:
        self.configure("aggregator_submitter_1", "kimi_code", "k3")
        scopes = []
        for run_id in ("run-1", "run-1", "run-2"):
            with mock.patch.object(
                APIClientManager, "_assistant_run_id_for_workflow", new=mock.AsyncMock(return_value=run_id),
            ):
                scopes.append((await self.call("aggregator_submitter_1", "k3")).await_args.kwargs["cache_scope"])
        self.assertEqual(scopes[0], scopes[1])
        self.assertNotEqual(scopes[1], scopes[2])

    async def test_workflow_mode_hint_selects_the_owning_run_and_never_reaches_the_provider(self) -> None:
        self.configure("aggregator_submitter_1", "kimi_code", "k3")
        resolver = mock.AsyncMock(return_value="autonomous-session")
        with mock.patch.object(APIClientManager, "_assistant_run_id_for_workflow", new=resolver):
            hinted = await self.call(
                "aggregator_submitter_1", "k3", _moto_assistant_workflow_mode="autonomous",
            )
            await self.call("aggregator_submitter_1", "k3")
        self.assertEqual(resolver.await_args_list[0].args, ("autonomous",))
        self.assertEqual(resolver.await_args_list[1].args, ("aggregator",))
        self.assertNotIn("_moto_assistant_workflow_mode", hinted.await_args.kwargs)

    async def test_shared_child_roles_take_the_active_workflow_run(self) -> None:
        # Autonomous reuses the child Aggregator/Compiler role IDs, and manual
        # Compiler runs its critique roles; each must key on the run that owns it.
        cases = (
            ("autonomous", "compiler_writer", "autonomous"),
            ("autonomous", "aggregator_validator", "autonomous"),
            ("manual_compiler", "critique_validator", "compiler"),
            ("autonomous", "manual_proof_validator", "manual_proof_check"),
        )
        for owner, role_id, expected_mode in cases:
            with self.subTest(owner=owner, role_id=role_id):
                self.configure(role_id, "kimi_code", "k3")
                resolver = mock.AsyncMock(return_value="run-1")
                with mock.patch(
                    "backend.shared.api_client_manager.workflow_start_guard",
                    new=SimpleNamespace(active_owner=owner),
                ), mock.patch.object(APIClientManager, "_assistant_run_id_for_workflow", new=resolver):
                    await self.call(role_id, "k3")
                self.assertEqual(resolver.await_args.args, (expected_mode,))

    async def test_autonomous_child_roles_leave_the_manual_proof_store_alone(self) -> None:
        self.configure("compiler_writer", "kimi_code", "k3")
        create_run = mock.AsyncMock(return_value="manual-run")
        with mock.patch(
            "backend.shared.api_client_manager.workflow_start_guard",
            new=SimpleNamespace(active_owner="autonomous"),
        ), mock.patch(
            "backend.autonomous.memory.proof_database.manual_proof_database.get_or_create_active_run_id",
            new=create_run,
        ):
            await self.call("compiler_writer", "k3")
        create_run.assert_not_awaited()

    async def test_supercharge_attempts_share_the_role_scope(self) -> None:
        self.manager.configure_role(
            "aggregator_submitter_1",
            ModelConfig(
                provider="kimi_code", model_id="k3", context_window=100_000, max_output_tokens=4_000,
                supercharge_enabled=True,
            ),
        )
        with mock.patch.object(
            APIClientManager, "_assistant_run_id_for_workflow", new=mock.AsyncMock(return_value="run-1"),
        ):
            generate = await self.call("aggregator_submitter_1", "k3")
        scopes = {call.kwargs["cache_scope"] for call in generate.await_args_list}
        self.assertEqual(generate.await_count, 5)
        self.assertEqual(len(scopes), 1)

    async def test_lookup_failure_degrades_to_a_role_only_scope(self) -> None:
        self.configure("aggregator_submitter_1", "kimi_code", "k3")
        with mock.patch(
            "backend.autonomous.memory.proof_database.manual_proof_database.get_or_create_active_run_id",
            new=mock.AsyncMock(side_effect=RuntimeError("store unavailable")),
        ):
            kwargs = (await self.call("aggregator_submitter_1", "k3")).await_args.kwargs
        self.assertEqual(kwargs["cache_scope"], "aggregator_submitter_1\n")

    async def test_sakana_requests_are_unchanged(self) -> None:
        self.configure("aggregator_submitter_1", "sakana_fugu", "fugu")
        resolver = mock.AsyncMock(return_value="run-1")
        with mock.patch.object(APIClientManager, "_assistant_run_id_for_workflow", new=resolver):
            generate = await self.call("aggregator_submitter_1", "fugu")
        self.assertNotIn("cache_scope", generate.await_args.kwargs)
        resolver.assert_not_awaited()


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

    def _invalid_request(self, message: str, *, status: int = 400) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error"}})

    async def test_rejected_request_setting_is_not_reported_as_a_key_or_plan_problem(self) -> None:
        self.use(lambda request: self._invalid_request("invalid temperature: only 1 is allowed for this model"))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            with self.assertRaises(SubscriptionParameterRejectedError) as ctx:
                await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        wait.assert_not_awaited()
        self.assertEqual(len(self.requests), 1)
        error = ctx.exception
        self.assertNotIsInstance(error, (SubscriptionEntitlementError, SubscriptionAuthError))
        self.assertEqual((error.provider, error.status_code), ("kimi_code", 400))
        self.assertIn("invalid temperature", str(error))
        self.assertIn("MOTO", error.terminal_guidance)
        self.assertIn("not a problem with your Kimi Code API key or plan", error.terminal_guidance)
        self.assertIn("another provider", error.terminal_guidance)

    async def test_limit_and_entitlement_rejections_keep_their_own_types(self) -> None:
        cases = (
            ("Your request exceeded model token limit: 262144", SubscriptionContextLimitError),
            ("Your API key does not have access to model k3", SubscriptionEntitlementError),
            ("model id does not exist", SubscriptionEntitlementError),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                self.use(lambda request, message=message: self._invalid_request(message))
                with self.assertRaises(expected) as ctx:
                    await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
                self.assertNotIsInstance(ctx.exception, SubscriptionParameterRejectedError)

    async def test_bad_requests_that_are_not_typed_setting_rejections_keep_the_generic_error(self) -> None:
        cases = (
            httpx.Response(400, json={"error": {"message": "invalid temperature: only 1 is allowed"}}),
            self._invalid_request("The request was rejected because it was considered high risk"),
            self._invalid_request("invalid temperature: only 1 is allowed", status=422),
        )
        for response in cases:
            with self.subTest(body=response.text):
                self.use(lambda request, response=response: response)
                with self.assertRaises(SubscriptionRequestError) as ctx:
                    await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
                self.assertNotIsInstance(ctx.exception, SubscriptionParameterRejectedError)
                self.assertEqual(ctx.exception.terminal_guidance, "")

    async def test_echoed_key_never_reaches_a_rejected_setting_error(self) -> None:
        self.use(lambda request: self._invalid_request(f"invalid reasoning_effort {KEY}"))
        with self.assertRaises(SubscriptionParameterRejectedError) as ctx:
            await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertNotIn(KEY, ctx.exception.terminal_guidance)

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


class _ByteParts(httpx.AsyncByteStream):
    """Yield byte parts, record whether the body was read to its end, optionally drop after."""

    def __init__(self, *parts: bytes, drop: BaseException | None = None) -> None:
        self.parts = parts
        self.drop = drop
        self.read_to_end = False

    async def __aiter__(self):
        for part in self.parts:
            yield part
        self.read_to_end = True
        if self.drop is not None:
            raise self.drop


def raw_sse(*chunks: dict) -> bytes:
    """SSE body whose JSON keeps non-ASCII characters raw, as Kimi sends them."""
    events = [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks]
    return "".join(events + ["data: [DONE]\n\n"]).encode("utf-8")


class KimiStreamReassemblyTests(KimiClientHarness):
    """The streamed reader is the shared provider-neutral one; these pin the defects it fixed."""

    async def _complete(self) -> dict:
        return await self.client.generate_completion(model="k3", messages=[{"role": "user", "content": "hi"}])

    async def test_unicode_line_separators_inside_a_delta_are_kept(self) -> None:
        # JSON leaves U+2028/U+2029/U+0085 unescaped; splitting lines at them cut the chunk in half.
        text = "first second third\u0085end"
        self.use(lambda request: httpx.Response(200, content=raw_sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"reasoning_content": "plan more"}}]},
            {"choices": [{"index": 0, "delta": {"content": text}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )))
        message = (await self._complete())["choices"][0]["message"]
        self.assertEqual(message["content"], text)
        self.assertEqual(message["reasoning_content"], "plan more")
        self.assertEqual(len(self.requests), 1)

    async def test_parallel_tool_calls_sharing_an_index_stay_separate(self) -> None:
        # Merging by index alone concatenated both calls into one with invalid arguments.
        fragments = [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "lookup", "arguments": '{"q": '}},
            {"index": 0, "function": {"arguments": '"a"}'}},
            {"index": 0, "id": "call_b", "type": "function", "function": {"name": "lookup", "arguments": '{"q": '}},
            {"index": 0, "function": {"arguments": '"b"}'}},
        ]
        chunks = completion_chunks("", finish="tool_calls", reasoning="plan")
        chunks[1:1] = [{"choices": [{"index": 0, "delta": {"tool_calls": [fragment]}}]} for fragment in fragments]
        self.use(lambda request: httpx.Response(200, content=sse(*chunks)))
        message = (await self._complete())["choices"][0]["message"]
        calls = message["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_a", "call_b"])
        self.assertEqual([json.loads(call["function"]["arguments"]) for call in calls], [{"q": "a"}, {"q": "b"}])
        self.assertEqual({call["function"]["name"] for call in calls}, {"lookup"})
        self.assertEqual(message["reasoning_content"], "plan")

    async def test_null_tool_call_indexes_are_matched_by_id(self) -> None:
        fragments = [
            {"index": None, "id": "call_x", "type": "function", "function": {"name": "f", "arguments": '{"n": '}},
            {"index": None, "id": "call_x", "function": {"arguments": "1}"}},
            {"index": None, "id": "call_y", "type": "function", "function": {"name": "f", "arguments": '{"n": 2}'}},
        ]
        self.use(lambda request: httpx.Response(200, content=sse(
            *completion_chunks("", finish="tool_calls", tool_calls=fragments),
        )))
        calls = (await self._complete())["choices"][0]["message"]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_x", "call_y"])
        self.assertEqual([json.loads(call["function"]["arguments"]) for call in calls], [{"n": 1}, {"n": 2}])

    async def test_unreadable_chunk_is_retried_instead_of_accepted(self) -> None:
        # A skipped chunk could have carried part of the answer, so the stream is not complete.
        chunks = completion_chunks("Hello")
        unreadable = b'data: {"choices": [\n\n'
        responses = [
            httpx.Response(200, content=sse(*chunks[:1], done=False) + unreadable + sse(*chunks[1:])),
            httpx.Response(200, content=sse(*completion_chunks("Hello again"))),
        ]
        self.use(lambda request: responses.pop(0))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            result = await self._complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello again")
        self.assertEqual(len(self.requests), 2)
        retry = wait.await_args.args[0]
        self.assertEqual(retry["reason"], "stream_incomplete")
        self.assertIn("unreadable data", retry["error_summary"])

    async def test_finished_answer_survives_a_later_transport_drop(self) -> None:
        body = _ByteParts(sse(*completion_chunks("Kept"), done=False), drop=httpx.ReadError("connection reset"))
        self.use(lambda request: httpx.Response(200, stream=body))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            result = await self._complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "Kept")
        self.assertEqual(len(self.requests), 1)
        wait.assert_not_awaited()

    async def test_unfinished_answer_is_retried_after_a_transport_drop(self) -> None:
        responses = [
            httpx.Response(200, stream=_ByteParts(
                sse(*completion_chunks("Part", finish=None), done=False), drop=httpx.ReadError("connection reset"),
            )),
            httpx.Response(200, content=sse(*completion_chunks("Whole"))),
        ]
        self.use(lambda request: responses.pop(0))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            result = await self._complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "Whole")
        self.assertEqual(wait.await_args.args[0]["reason"], "connection_error")

    async def test_body_is_read_past_done_so_the_connection_can_be_reused(self) -> None:
        late = b'data: {"choices": [{"index": 0, "delta": {"content": " late"}}]}\n\n'
        body = _ByteParts(sse(*completion_chunks("Done")), late)
        self.use(lambda request: httpx.Response(200, stream=body))
        result = await self._complete()
        self.assertEqual(result["choices"][0]["message"]["content"], "Done")
        self.assertTrue(body.read_to_end)

    async def test_error_chunk_raises_without_retrying_or_leaking_the_key(self) -> None:
        chunks = completion_chunks("partial", finish=None)
        chunks.append({"error": {"message": f"upstream failed for {KEY}"}})
        self.use(lambda request: httpx.Response(200, content=sse(*chunks)))
        with mock.patch(f"{_CLIENT_MODULE}.assured_retry_wait", new=mock.AsyncMock(return_value=True)) as wait:
            with self.assertRaises(SubscriptionRequestError) as ctx:
                await self._complete()
        self.assertIn("Kimi Code completion failed: upstream failed for", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertEqual(len(self.requests), 1)
        wait.assert_not_awaited()

    async def test_usage_on_the_final_choice_and_generic_reasoning_key_are_kept(self) -> None:
        self.use(lambda request: httpx.Response(200, content=sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"reasoning": "think"}}]},
            {"choices": [{"index": 0, "delta": {"content": "OK"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop",
                          "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}]},
        )))
        result = await self._complete()
        message = result["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "think")
        self.assertNotIn("reasoning", message)
        self.assertEqual(result["usage"]["total_tokens"], 4)
